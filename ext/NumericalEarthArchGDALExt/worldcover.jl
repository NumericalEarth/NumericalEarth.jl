#####
##### ESA WorldCover: anonymous COG tiles → regional NetCDF
#####
##### The `Map` band is a UInt8 land-cover class code (no-data = 0). Class codes
##### must never be averaged, so the mosaic of 3° tiles is read at its native
##### 10 m resolution with nearest resampling (which only clips/aligns — it never
##### invents an intermediate code), one bounded chunk at a time; each chunk is
##### counted onto the coarse lat/lon lattice with `aggregate_landcover` and
##### written to the NetCDF before the next is read. This keeps the categorical
##### field on its native EPSG:4326 grid — no reprojection.
#####

# S3 key for one 3°×3° tile named by its SW corner (e.g. "N51E006").
function worldcover_tile_url(dataset, tile)
    year = version_year(dataset)
    version = version_string(dataset)
    key = string(version, "/", year, "/map/",
                 "ESA_WorldCover_10m_", year, "_", version, "_", tile, "_Map.tif")
    return string("/vsis3/esa-worldcover/", key)
end

# SW-corner tile label for the 3° grid cell containing (longitude, latitude).
function worldcover_tile_label(longitude, latitude)
    tile_latitude  = 3 * fld(latitude, 3)
    tile_longitude = 3 * fld(longitude, 3)
    ns = tile_latitude  ≥ 0 ? "N" : "S"
    ew = tile_longitude ≥ 0 ? "E" : "W"
    return string(ns, lpad(abs(Int(tile_latitude)),  2, '0'),
                  ew, lpad(abs(Int(tile_longitude)), 3, '0'))
end

# Labels of every 3° tile overlapping the native-pixel window.
function worldcover_tiles(i₁, i₂, j₁, j₂)
    tile_span = 3 * ESA_WORLDCOVER_PIXELS_PER_DEGREE
    return [worldcover_tile_label(λ, φ) for φ in 3 * fld(j₁, tile_span) : 3 : 3 * fld(j₂ - 1, tile_span)
                                        for λ in 3 * fld(i₁, tile_span) : 3 : 3 * fld(i₂ - 1, tile_span)]
end

function NumericalEarth.DataWrangling.WorldCover.worldcover_cog_to_netcdf(metadatum::ESAWorldCoverMetadatum, nc_path)
    configure_vsicurl!()

    dataset = metadatum.dataset
    region  = metadatum.region
    factor  = dataset.aggregation_factor

    window = worldcover_window(region.longitude, region.latitude, factor)
    tile_urls = [worldcover_tile_url(dataset, tile) for tile in worldcover_tiles(window...)]

    # Read the anonymous, unsigned public bucket; `environment` restores any prior
    # AWS/GDAL config afterwards, so a signed `/vsis3` read elsewhere in the same
    # session is not left with signing disabled.
    ArchGDAL.environment(globalconfig = ["AWS_NO_SIGN_REQUEST" => "YES",
                                         "AWS_REGION" => "eu-central-1"]) do
        # ESA WorldCover only publishes tiles that contain land; a 3° cell that is
        # entirely ocean has no tile and reads as no-data.
        sources = ArchGDAL.IDataset[]
        for url in tile_urls
            try
                push!(sources, ArchGDAL.read(url))
            catch tile_error
                @warn "Skipping unavailable ESA WorldCover tile" url exception = tile_error
            end
        end
        isempty(sources) && error("No ESA WorldCover tiles are published for the region " *
                                  "longitude $(region.longitude), latitude $(region.latitude)")

        try
            aggregate_worldcover_tiles(sources, window, factor, nc_path)
        finally
            foreach(ArchGDAL.destroy, sources)
        end
    end

    return nothing
end

# Count the mosaic of `sources` over the native-pixel `window` onto the `factor`-pixel
# lattice and write every product to `nc_path`, holding at most `tile_bytes` of pixels at a time.
function aggregate_worldcover_tiles(sources, window, factor, nc_path; tile_bytes = default_tile_bytes)
    i₁, i₂, j₁, j₂ = window
    native_step = ESA_WORLDCOVER_NATIVE_STEP
    nx = (i₂ - i₁) ÷ factor
    ny = (j₂ - j₁) ÷ factor
    Δ = factor * native_step
    longitude = range(i₁ * native_step + Δ / 2, step = Δ, length = nx)
    latitude  = range(j₁ * native_step + Δ / 2, step = Δ, length = ny)

    chunks = ceil(Int, sqrt((i₂ - i₁) * (j₂ - j₁) / tile_bytes))
    chunks_x = min(chunks, nx)
    chunks_y = min(chunks, ny)

    staging = tempname(dirname(nc_path))
    NCDataset(staging, "c") do ds
        defDim(ds, "lon", nx)
        defDim(ds, "lat", ny)

        longitude_variable = defVar(ds, "lon", Float64, ("lon",);
                                    attrib = ["units" => "degrees_east", "long_name" => "longitude"])
        latitude_variable  = defVar(ds, "lat", Float64, ("lat",);
                                    attrib = ["units" => "degrees_north", "long_name" => "latitude"])
        longitude_variable[:] = collect(longitude)
        latitude_variable[:]  = collect(latitude)

        class_variable = defVar(ds, "landcover_class", Float32, ("lon", "lat");
                                attrib = ["long_name" => "majority land-cover class code",
                                          "missing_value" => 0])
        vegetation_variable = defVar(ds, "vegetation_fraction", Float32, ("lon", "lat");
                                     attrib = ["long_name" => "vegetated area fraction",
                                               "units" => "1"])
        fraction_variables = map(keys(ESA_WORLDCOVER_CLASS_NAMES)) do name
            defVar(ds, string(class_fraction_variable_name(name)), Float32, ("lon", "lat");
                   attrib = ["long_name" => string(name, " area fraction"), "units" => "1"])
        end

        ArchGDAL.gdalbuildvrt(sources) do mosaic
            for chunk_j in 1:chunks_y, chunk_i in 1:chunks_x
                cells_i = tile_indices(nx, chunks_x, chunk_i)
                cells_j = tile_indices(ny, chunks_y, chunk_j)
                west  = (i₁ + factor * (first(cells_i) - 1)) * native_step
                east  = (i₁ + factor * last(cells_i)) * native_step
                south = (j₁ + factor * (first(cells_j) - 1)) * native_step
                north = (j₁ + factor * last(cells_j)) * native_step

                pixels = ArchGDAL.gdalwarp([mosaic],
                    ["-te", string(west), string(south), string(east), string(north),
                     "-tr", string(native_step), string(native_step),
                     "-r",  "near",
                     "-ot", "Byte"]) do windowed
                    reverse!(ArchGDAL.read(windowed, 1), dims = 2)  # GDAL rows run north to south
                end

                aggregated = aggregate_landcover(pixels, factor)
                class_variable[cells_i, cells_j]      = Float32.(aggregated.landcover_class)
                vegetation_variable[cells_i, cells_j] = Float32.(aggregated.vegetation_fraction)
                for (variable, fraction) in zip(fraction_variables, aggregated.class_fractions)
                    variable[cells_i, cells_j] = Float32.(fraction)
                end
            end
        end
    end
    mv(staging, nc_path; force = true)

    return nothing
end
