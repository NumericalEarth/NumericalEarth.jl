#####
##### GHSL built-up raster ingest: download the R2023A Mollweide tiles intersecting a
##### BoundingBox and aggregate their native pixels onto longitude/latitude cells.
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

function ghsl_mosaic(f, dataset, region, cache_dir)
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
    isempty(sources) && return nothing

    datasets = [ArchGDAL.read(source) for source in sources]
    try
        return ArchGDAL.gdalbuildvrt(f, datasets)
    finally
        foreach(ArchGDAL.destroy, datasets)
    end
end

# Shift a contiguous window by one multiple of 360° so its west edge lies in [-180°, 180°).
mollweide_window_longitudes(longitudes) = longitudes .+ (longitude_in_same_window(first(longitudes), 0) - first(longitudes))

# Shifts that bring each 360° period of a window starting in [-180°, 180°) onto the map.
antimeridian_shifts(east) = 0:-360:-360 * ceil(Int, max(0, east - 180) / 360)

function ghsl_reprojected_raster(dataset, raster, cache_dir)
    west, east   = mollweide_window_longitudes(raster.region.longitude)
    south, north = raster.region.latitude
    reprojected = fill(NaN, raster.Nx, raster.Ny)
    for shift in antimeridian_shifts(east)
        region = BoundingBox(longitude = (west, east) .+ shift, latitude = (south, north))
        period = ghsl_mosaic(dataset, region, cache_dir) do mosaic
            ArchGDAL.gdalwarp([mosaic],
                # `-ts` pins the cell count: with a pixel size, GDAL rounds the extent instead
                # and the raster comes back short of the grid.
                ["-s_srs", "ESRI:54009",
                 "-t_srs", "EPSG:4326",
                 "-te",    string(west + shift), string(south), string(east + shift), string(north),
                 "-ts",    string(raster.Nx), string(raster.Ny),
                 "-r",     "bilinear",
                 "-dstnodata", "nan",  # honor the source no-data: keep it out of the bilinear blend and write NaN for gaps
                 "-ot",    "Float32"]) do warped
                data = Float64.(ArchGDAL.read(warped, 1))
                return reverse(data, dims = 2)  # GDAL writes y north→south
            end
        end
        isnothing(period) || (reprojected .= ifelse.(isnan.(reprojected), period, reprojected))
    end
    return reprojected
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

@kernel function _conservative_ghsl_moments!(built_fraction, measured_built_fraction, building_volume, surface, height,
                                            meridians, projected_latitudes, grid, west, north, pixel_size, refinement, height_offset, radius)
    i, j = @index(Global, NTuple)
    @inbounds begin
        south_edge, north_edge = projected_latitudes[j], projected_latitudes[j + 1]
        first_row = max(1, floor(Int, (north - north_edge) / pixel_size) + 1)
        last_row = min(size(surface, 2), ceil(Int, (north - south_edge) / pixel_size))
        built_area = measured_built_area = built_volume = convert(eltype(built_fraction), 0)
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
                building_height = mask_building_height(height[cld(column + height_offset[1], refinement),
                                                              cld(row + height_offset[2], refinement)])
                built = ifelse(isfinite(fraction), area * fraction, 0)
                # A built pixel without a positive height counts as built but carries no height.
                measured = ifelse(building_height > 0, built, 0)
                built_area += built
                measured_built_area += measured
                built_volume += ifelse(measured > 0, measured * building_height, 0)
            end
        end
        area = Azᶜᶜᶜ(i, j, 1, grid)
        built_fraction[i, j] += built_area / area
        measured_built_fraction[i, j] += measured_built_area / area
        building_volume[i, j] += built_volume / area
    end
end

function ghsl_raster_fields(dataset::GHSBuilt, raster, cache_dir, arch; maximum_raster_cells = 100_000_000)
    longitude_faces = mollweide_window_longitudes(Float64.(raster.longitude_faces))
    latitude_faces = Float64.(raster.latitude_faces)
    pixel_size = native_resolution(dataset.surface)
    refinement = native_resolution(dataset.height) ÷ pixel_size

    grid = LatitudeLongitudeGrid(arch, Float64; size = (raster.Nx, raster.Ny),
                                 longitude = longitude_faces, latitude = latitude_faces,
                                 radius = MOLLWEIDE_RADIUS, topology = (Bounded, Bounded, Flat))
    projected_latitudes = on_architecture(arch, last.(longitude_latitude_to_mollweide.(0, latitude_faces)))
    built_up_fraction = zeros(arch, Float64, raster.Nx, raster.Ny)
    measured_built_fraction = zeros(arch, Float64, raster.Nx, raster.Ny)
    building_volume = zeros(arch, Float64, raster.Nx, raster.Ny)

    for shift in antimeridian_shifts(last(longitude_faces))
        # Clamped faces collapse the parts of cells off the map onto its edge meridian.
        map_faces = clamp.(longitude_faces .+ shift, -180, 180)
        region = BoundingBox(longitude = (first(map_faces), last(map_faces)),
                             latitude = (first(latitude_faces), last(latitude_faces)))
        west, east, south, north = ghsl_pixel_window(region, native_resolution(dataset.height))
        meridians = on_architecture(arch, first.(longitude_latitude_to_mollweide.(map_faces, 0)))
        columns = round(Int, (east - west) / pixel_size)
        rows = round(Int, (north - south) / pixel_size)
        window_columns = min(columns, isqrt(maximum_raster_cells))
        window_rows = min(rows, maximum_raster_cells ÷ window_columns)

        ghsl_mosaic(dataset.surface, region, cache_dir) do surface_mosaic
            ghsl_mosaic(dataset.height, region, cache_dir) do height_mosaic
                for first_row in 1:window_rows:rows, first_column in 1:window_columns:columns
                    last_column = min(columns, first_column + window_columns - 1)
                    last_row = min(rows, first_row + window_rows - 1)
                    # Each product window follows its native pixel faces.
                    surface, height = map((surface_mosaic, height_mosaic), (1, refinement)) do mosaic, factor
                        step = pixel_size * factor
                        window_west = west + fld(first_column - 1, factor) * step
                        window_east = west + cld(last_column, factor) * step
                        window_north = north - fld(first_row - 1, factor) * step
                        window_south = north - cld(last_row, factor) * step
                        ArchGDAL.gdalwarp([mosaic],
                            ["-of", "MEM", "-s_srs", "ESRI:54009", "-t_srs", "ESRI:54009",
                             "-te", string(window_west), string(window_south), string(window_east), string(window_north),
                             "-tr", string(step), string(step), "-r", "near", "-ovr", "NONE",
                             "-dstnodata", "nan", "-ot", "Float32"]) do windowed
                            on_architecture(arch, ArchGDAL.read(windowed, 1))
                        end
                    end
                    surface_west = west + (first_column - 1) * pixel_size
                    surface_north = north - (first_row - 1) * pixel_size
                    height_offset = (mod(first_column - 1, refinement), mod(first_row - 1, refinement))
                    launch!(arch, nothing, size(built_up_fraction), _conservative_ghsl_moments!,
                            built_up_fraction, measured_built_fraction, building_volume, surface, height, meridians,
                            projected_latitudes, grid, surface_west, surface_north, pixel_size, refinement, height_offset,
                            MOLLWEIDE_RADIUS)
                end
            end
        end
    end
    return (; built_up_fraction, measured_built_fraction, measured_building_volume = building_volume)
end

function NumericalEarth.DataWrangling.GHSL.ghsl_tiles_to_netcdf(metadatum::GHSLMetadatum, nc_path)
    raster = ghsl_regional_raster(metadatum)
    fields = ghsl_raster_fields(metadatum.dataset, raster, joinpath(dirname(nc_path), "tiles"), CPU())
    if metadatum.dataset isa GHSBuilt
        building_volume = ifelse.(fields.built_up_fraction .> 0,
                                 fields.built_up_fraction .* fields.measured_building_volume ./ fields.measured_built_fraction, 0)
        fields = (; fields..., building_volume)
    end
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
            var[:, :] = data
        end
    end

    return nothing
end

function NumericalEarth.Lands.building_morphometry(grid::LatitudeLongitudeGrid, dataset::GHSBuilt; maximum_raster_cells = 100_000_000)
    arch = architecture(grid)
    longitude_faces = on_architecture(CPU(), λnodes(grid, Face()))
    latitude_faces  = on_architecture(CPU(), φnodes(grid, Face()))
    λᵖ = Field{Center, Center, Nothing}(grid)
    h  = Field{Center, Center, Nothing}(grid)
    measured_plan_area_index = Field{Center, Center, Nothing}(grid)

    raster = (; Nx = size(grid, 1), Ny = size(grid, 2), longitude_faces, latitude_faces)
    moments = ghsl_raster_fields(dataset, raster, joinpath(download_GHSL_cache, "tiles"), arch; maximum_raster_cells)
    interior(λᵖ, :, :, 1) .= moments.built_up_fraction
    interior(measured_plan_area_index, :, :, 1) .= moments.measured_built_fraction
    interior(h, :, :, 1) .= ifelse.(moments.built_up_fraction .> 0,
                                  moments.measured_building_volume ./ moments.measured_built_fraction, 0)
    fill_halo_regions!((λᵖ, h, measured_plan_area_index))

    return (; plan_area_index = λᵖ, mean_building_height = h, measured_plan_area_index)
end
