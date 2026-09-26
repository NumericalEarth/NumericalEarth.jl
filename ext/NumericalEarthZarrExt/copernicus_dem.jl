#####
##### Copernicus DEM Zarr → regional NetCDF
#####

function copernicus_dem_url(dataset)
    token = get(ENV, "DESTINE_ACCESS_TOKEN", nothing)
    isnothing(token) && error(
        "Set the DESTINE_ACCESS_TOKEN environment variable to read Copernicus DEM. " *
        "Register at https://platform.destine.eu/ and create a token at " *
        "https://earthdatahub.destine.eu/account-settings#my-personal-access-tokens.")

    return string("https://edh:", token, "@", CopernicusDEM.zarr_host_path(dataset))
end

# The Earth Data Hub Copernicus DEM stores name their coordinates "lon"/"lat" and
# store `dsm` as (lon, lat); both coordinates are ascending. Ascending vs descending
# is detected and handled regardless, so only the coordinate names and dimension
# order are assumed here.
#
# The gateway intermittently 403s valid requests under concurrent CI load; retry
# with exponential backoff rather than failing on the first hiccup.
function CopernicusDEM.zarr_to_netcdf(metadatum::CopernicusDEM.CopernicusDEMMetadatum, nc_path; max_retries = 5)
    url = copernicus_dem_url(metadatum.dataset)

    for attempt in 1:max_retries
        try
            store = zopen(url; consolidated = true)

            grid = native_grid(metadatum)
            λ₁, λ₂ = x_domain(grid)
            φ₁, φ₂ = y_domain(grid)
            Nx, Ny, _ = size(grid)

            Δλ = (λ₂ - λ₁) / Nx
            Δφ = (φ₂ - φ₁) / Ny
            first_longitude = λ₁ + Δλ / 2
            first_latitude  = φ₁ + Δφ / 2

            longitude = store["lon"][:]
            latitude  = store["lat"][:]

            longitude_range, longitude_ascending = ascending_window(longitude, first_longitude, Nx)
            latitude_range,  latitude_ascending  = ascending_window(latitude,  first_latitude,  Ny)

            variable_name = CopernicusDEM.dataset_zarr_variable_name
            elevation = Float32.(store[variable_name][longitude_range, latitude_range])  # (lon, lat)

            # The native LatitudeLongitudeGrid is ascending in both lon and lat.
            longitude_ascending || (elevation = reverse(elevation, dims = 1))
            latitude_ascending  || (elevation = reverse(elevation, dims = 2))

            window_longitude = longitude_ascending ? longitude[longitude_range] : reverse(longitude[longitude_range])
            window_latitude  = latitude_ascending  ? latitude[latitude_range]   : reverse(latitude[latitude_range])

            NCDataset(nc_path, "c") do dataset
                defDim(dataset, "lon", length(window_longitude))
                defDim(dataset, "lat", length(window_latitude))

                longitude_variable = defVar(dataset, "lon", Float64, ("lon",);
                                            attrib = ["units" => "degrees_east", "long_name" => "longitude"])
                latitude_variable  = defVar(dataset, "lat", Float64, ("lat",);
                                            attrib = ["units" => "degrees_north", "long_name" => "latitude"])
                elevation_variable = defVar(dataset, "z", Float32, ("lon", "lat");
                                            attrib = ["units" => "m", "long_name" => "surface elevation"])

                longitude_variable[:] = window_longitude
                latitude_variable[:]  = window_latitude
                elevation_variable[:, :] = elevation
            end

            return nothing
        catch e
            attempt < max_retries || rethrow(e)
            @warn "Copernicus DEM Zarr read attempt $attempt/$max_retries failed; retrying..." exception=(e, catch_backtrace())
            sleep(min(60, 5.0 * 2^(attempt - 1)))
        end
    end
end

#####
##### Copernicus DEM Zarr → target grid, streamed in tiles
#####

Bathymetry.download_for_regridding(::CopernicusDEM.CopernicusDEMMetadatum) = nothing

# The gateway intermittently 403s valid requests under load; retry with exponential backoff.
function read_tile(elevation, tile_i, tile_j; max_retries = 5)
    attempt = 1
    while true
        try
            return elevation[tile_i, tile_j]
        catch e
            attempt < max_retries || rethrow(e)
            @warn "Copernicus DEM tile read attempt $attempt/$max_retries failed; retrying..." exception=(e, catch_backtrace())
            sleep(min(60, 5.0 * 2^(attempt - 1)))
            attempt += 1
        end
    end
end

# The window is streamed one tile at a time; each source cell accumulates, weighted by
# cos(latitude), into the target cell holding its center, so at most `tile_size²` source
# cells are resident. The averaging already coarsens to the target scale, so
# `interpolation_passes` is not used. The default `tile_size` is a multiple of both
# stores' chunk edges (2400 for GLO-90, 3600 for GLO-30).
function Bathymetry.regrid_bottom_height(target_grid, metadatum::CopernicusDEM.CopernicusDEMMetadatum;
                                         height_above_water, interpolation_passes, tile_size = 7200)
    store = zopen(copernicus_dem_url(metadatum.dataset); consolidated = true)
    elevation = store[CopernicusDEM.dataset_zarr_variable_name]

    grid = native_grid(metadatum)
    λ₁, λ₂ = x_domain(grid)
    φ₁, φ₂ = y_domain(grid)
    Nλ, Nφ, _ = size(grid)
    Δλ = (λ₂ - λ₁) / Nλ
    Δφ = (φ₂ - φ₁) / Nφ

    longitude = store["lon"][:]
    latitude  = store["lat"][:]
    window_i, _ = ascending_window(longitude, λ₁ + Δλ / 2, Nλ)
    window_j, _ = ascending_window(latitude,  φ₁ + Δφ / 2, Nφ)

    Nx, Ny, _ = size(target_grid)
    longitude_faces = Array(λnodes(target_grid, Face(), Center(), Center()))
    latitude_faces  = Array(φnodes(target_grid, Center(), Face(), Center()))

    elevation_sum = zeros(Nx, Ny)
    weight_sum = zeros(Nx, Ny)

    for tile_j in tile_ranges(window_j, tile_size), tile_i in tile_ranges(window_i, tile_size)
        block = read_tile(elevation, tile_i, tile_j)
        isnothing(height_above_water) ||
            (block = map(z -> z > 0 ? oftype(z, height_above_water) : z, block))

        tile_columns = target_cells(longitude_faces, view(longitude, tile_i), Nx)
        tile_rows    = target_cells(latitude_faces,  view(latitude,  tile_j), Ny)
        tile_latitude = view(latitude, tile_j)

        for (jj, j) in pairs(tile_rows)
            j == 0 && continue
            w = cosd(tile_latitude[jj])
            for (ii, i) in pairs(tile_columns)
                i == 0 && continue
                @inbounds begin
                    elevation_sum[i, j] += w * block[ii, jj]
                    weight_sum[i, j] += w
                end
            end
        end
    end

    target_z = Field{Center, Center, Nothing}(target_grid)
    set!(target_z, elevation_sum ./ weight_sum)

    return target_z
end
