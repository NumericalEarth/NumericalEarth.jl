#####
##### GHSL built-up raster ingest: download the R2023A Mollweide tiles intersecting a
##### BoundingBox, mosaic + warp them to EPSG:4326 (reading inside each `.zip` archive
##### with GDAL's `/vsizip/`), and write the regional lat/lon NetCDF the read path expects.
#####

# Cache the tile archive keyed by tile, so it is reused across regions.
function ghsl_download_tile(dataset, row, column, cache_dir)
    mkpath(cache_dir)
    url = ghsl_tile_url(dataset, row, column)
    zip_path = joinpath(cache_dir, basename(url))
    # Download to a staging file and rename on success, so an interrupted transfer
    # never leaves a truncated archive that the `isfile` guard would keep forever.
    if !isfile(zip_path)
        staging = zip_path * ".part"
        try
            Downloads.download(url, staging)
            mv(staging, zip_path; force = true)
        finally
            rm(staging; force = true)
        end
    end
    inner_tif = ghsl_tile_tif_name(dataset, row, column)
    return string("/vsizip/", zip_path, "/", inner_tif)
end

function ghsl_sources(dataset, region, cache_dir)
    # JRC omits all-ocean tiles from the grid, so a coastal window can reference a tile
    # that 404s; skip those and mosaic the land tiles that do exist.
    sources = String[]
    for (row, column) in ghsl_tiles_in_bbox(region)
        source = try
            ghsl_download_tile(dataset, row, column, cache_dir)
        catch err
            (err isa Downloads.RequestError && err.response.status == 404) || rethrow()
            @warn "GHSL tile R$(row)_C$(column) is not published (likely all-ocean); skipping."
            nothing
        end
        isnothing(source) || push!(sources, source)
    end
    isempty(sources) &&
        error("No GHSL tiles are published for the requested region; it may be entirely ocean.")

    return [ArchGDAL.read(source) for source in sources]
end

function ghsl_reprojected_raster(dataset, raster, cache_dir)
    west, east   = raster.region.longitude
    south, north = raster.region.latitude
    datasets = ghsl_sources(dataset, raster.region, cache_dir)
    try
        ArchGDAL.gdalwarp(datasets,
            # `-ts` pins the cell count: with a pixel size, GDAL rounds the extent instead
            # and the raster comes back short of the grid.
            ["-s_srs", "ESRI:54009",
             "-t_srs", "EPSG:4326",
             "-te",    string(west), string(south), string(east), string(north),
             "-ts",    string(raster.Nx), string(raster.Ny),
             "-r",     "bilinear",
             "-dstnodata", "nan",  # honor the source no-data: keep it out of the bilinear blend and write NaN for gaps
             "-ot",    "Float32"]) do warped
            data = Float64.(ArchGDAL.read(warped, 1))
            return reverse(data, dims = 2)  # GDAL writes y north→south
        end
    finally
        for d in datasets
            ArchGDAL.destroy(d)
        end
    end
end

function ghsl_raster_fields(dataset::GHSBuiltS, raster, cache_dir, arch)
    surface = ghsl_reprojected_raster(dataset, raster, cache_dir)
    return (; built_up_fraction = built_surface_to_fraction.(surface, native_resolution(dataset)^2))
end

ghsl_raster_fields(dataset::GHSBuiltH, raster, cache_dir, arch) =
    (; building_height = mask_building_height.(ghsl_reprojected_raster(dataset, raster, cache_dir)))

# Mean of a linear segment clipped to [0, width].
@inline function clipped_linear_mean(start_value, end_value, width)
    lower, upper = minmax(start_value, end_value)
    lower_clipped, upper_clipped = clamp(lower, 0, width), clamp(upper, 0, width)
    integral = (lower_clipped + upper_clipped) * (upper_clipped - lower_clipped) / 2 +
               width * max(0, upper - max(width, lower))
    return ifelse(lower == upper, lower_clipped, integral / (upper - lower))
end

@kernel function _conservative_ghsl_moments!(built_fraction, building_volume, surface, height,
                                            meridians, projected_latitudes, areas, west, north, pixel_size, refinement, radius)
    i, j = @index(Global, NTuple)
    @inbounds begin
        south_edge, north_edge = projected_latitudes[j], projected_latitudes[j + 1]
        first_row = max(1, floor(Int, (north - north_edge) / pixel_size) + 1)
        last_row = min(size(surface, 2), ceil(Int, (north - south_edge) / pixel_size))
        built_area = built_volume = convert(eltype(built_fraction), 0)
        for row in first_row:last_row
            south_overlap = max(south_edge, north - row * pixel_size)
            north_overlap = min(north_edge, north - (row - 1) * pixel_size)
            # Mollweide meridians: x = (2√2 R / π) λ √(1 - y² / (2R²)).
            south_scale = sqrt(max(0, 1 - south_overlap^2 / (2radius^2)))
            north_scale = sqrt(max(0, 1 - north_overlap^2 / (2radius^2)))
            west_south, west_north = meridians[i] * south_scale, meridians[i] * north_scale
            east_south, east_north = meridians[i + 1] * south_scale, meridians[i + 1] * north_scale
            first_column = max(1, floor(Int, (min(west_south, west_north) - west) / pixel_size) + 1)
            last_column = min(size(surface, 1), ceil(Int, (max(east_south, east_north) - west) / pixel_size))
            for column in first_column:last_column
                pixel_west = west + (column - 1) * pixel_size
                # Shared meridians are linear within each row overlap, partitioning its area.
                east_width = clipped_linear_mean(east_south - pixel_west, east_north - pixel_west, pixel_size)
                west_width = clipped_linear_mean(west_south - pixel_west, west_north - pixel_west, pixel_size)
                area = (north_overlap - south_overlap) * (east_width - west_width)
                fraction = built_surface_to_fraction(surface[column, row], pixel_size^2)
                building_height = mask_building_height(height[cld(column, refinement), cld(row, refinement)])
                valid = isfinite(fraction) & isfinite(building_height)
                built_area += ifelse(valid, area * fraction, 0)
                built_volume += ifelse(valid, area * fraction * building_height, 0)
            end
        end
        built_fraction[i, j] = built_area / areas[j]
        building_volume[i, j] = built_volume / areas[j]
    end
end

function ghsl_raster_fields(dataset::GHSBuilt, raster, cache_dir, arch)
    west, east, south, north = ghsl_pixel_window(raster.region, native_resolution(dataset.height))
    pixel_size = native_resolution(dataset.surface)
    refinement = native_resolution(dataset.height) ÷ pixel_size

    # Aligned native footprints, with net height constant within each 100 m pixel.
    surface, height = map((dataset.surface, dataset.height)) do product
        native_pixel_size = native_resolution(product)
        sources = ghsl_sources(product, raster.region, cache_dir)
        try
            ArchGDAL.gdalbuildvrt(sources) do mosaic
                ArchGDAL.gdalwarp([mosaic],
                    ["-of", "MEM", "-s_srs", "ESRI:54009", "-t_srs", "ESRI:54009",
                     "-te", string(west), string(south), string(east), string(north),
                     "-tr", string(native_pixel_size), string(native_pixel_size), "-r", "near", "-ovr", "NONE",
                     "-dstnodata", "nan", "-ot", "Float32"]) do windowed
                    on_architecture(arch, ArchGDAL.read(windowed, 1))
                end
            end
        finally
            foreach(ArchGDAL.destroy, sources)
        end
    end

    longitude_faces = range(raster.region.longitude...; length = raster.Nx + 1)
    latitude_faces = range(raster.region.latitude...; length = raster.Ny + 1)
    meridians = on_architecture(arch, (2sqrt(2) * MOLLWEIDE_RADIUS / π) .* deg2rad.(longitude_faces))
    projected_latitudes = on_architecture(arch, last.(longitude_latitude_to_mollweide.(0, latitude_faces)))
    areas = on_architecture(arch, 2MOLLWEIDE_RADIUS^2 * deg2rad(step(longitude_faces)) .*
                                 cosd.(raster.latitude) .* sind.(diff(latitude_faces) ./ 2))
    built_up_fraction = zeros(arch, Float64, raster.Nx, raster.Ny)
    building_volume = similar(built_up_fraction)
    launch!(arch, nothing, size(built_up_fraction), _conservative_ghsl_moments!,
            built_up_fraction, building_volume, surface, height,
            meridians, projected_latitudes, areas, west, north, pixel_size, refinement, MOLLWEIDE_RADIUS)
    return (; built_up_fraction, building_volume)
end

function NumericalEarth.DataWrangling.GHSL.ghsl_tiles_to_netcdf(metadatum::GHSLMetadatum, nc_path, arch = CPU())
    raster = ghsl_regional_raster(metadatum)
    fields = ghsl_raster_fields(metadatum.dataset, raster, joinpath(dirname(nc_path), "tiles"), arch)
    variable_names = available_variables(metadatum.dataset)

    NCDataset(nc_path, "c") do ds
        defDim(ds, "lon", raster.Nx)
        defDim(ds, "lat", raster.Ny)
        lon_var = defVar(ds, "lon", Float64, ("lon",);
                         attrib = ["units" => "degrees_east", "long_name" => "longitude"])
        lat_var = defVar(ds, "lat", Float64, ("lat",);
                         attrib = ["units" => "degrees_north", "long_name" => "latitude"])
        lon_var[:] = raster.longitude
        lat_var[:] = raster.latitude
        for (name, data) in pairs(fields)
            var = defVar(ds, variable_names[name], Float64, ("lon", "lat"))
            var[:, :] = on_architecture(CPU(), data)
        end
    end

    return nothing
end
