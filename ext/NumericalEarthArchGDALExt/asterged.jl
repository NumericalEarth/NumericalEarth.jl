#####
##### Advanced Spaceborne Thermal Emission and Reflection Radiometer
##### Global Emissivity Database version 3 (ASTER-GEDv3):
##### https://lpdaac.usgs.gov/products/ag100v003/
#####
##### ASTER GED emissivity ingest: resolve the 1°×1° HDF5 tiles intersecting the region
##### via NASA CMR, download them with Earthdata credentials, and write a regional NetCDF
##### of the broadband emissivity and uncertainty. Requires GDAL_jll with the HDF5 driver.
#####

# Open an HDF5 subdataset via GDAL's `HDF5:"file"://path` syntax and return the full
# raster array. Multi-band datasets come back as `(Nx, Ny, nbands)`.
function read_asterged_subdataset(h5_path, layer)
    name = string("HDF5:\"", h5_path, "\":", layer)
    return ArchGDAL.read(name) do dataset
        ArchGDAL.read(dataset)
    end
end

function NumericalEarth.DataWrangling.ASTERGED.asterged_tiles_to_netcdf(metadatum, nc_path::AbstractString)
    dataset = metadatum.dataset
    bbox = metadatum.region
    (bbox isa BoundingBox && !isnothing(bbox.longitude) && !isnothing(bbox.latitude)) ||
        error("asterged_tiles_to_netcdf requires a BoundingBox region.")

    # Write on the same native grid the Field will use, so tile cells land on file
    # cells by construction.
    grid = native_grid(metadatum, CPU())
    longitude = collect(λnodes(grid, Center()))
    latitude  = collect(φnodes(grid, Center()))
    Nx, Ny = length(longitude), length(latitude)
    Δλ = (longitude[end] - longitude[1]) / max(Nx - 1, 1)
    Δφ = (latitude[end]  - latitude[1])  / max(Ny - 1, 1)

    short_name = asterged_short_name(dataset)
    version = asterged_version(dataset)
    coefficients = OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS

    # Pad the query a cell so every file cell is covered. CMR needs [-180, 180]
    # longitudes, and reads a folded west > east as crossing the antimeridian.
    to_pm180(λ) = rem(λ, 360, RoundNearest)
    query = BoundingBox(longitude = (to_pm180(longitude[1] - Δλ), to_pm180(longitude[end] + Δλ)),
                        latitude  = (latitude[1]  - Δφ, latitude[end]  + Δφ))
    granule_urls = cmr_granules(short_name, version, query)
    isempty(granule_urls) &&
        error("CMR returned no $(short_name).$(version) tiles for region $(bbox).")

    emissivity  = fill(NaN32, Nx, Ny)
    uncertainty = fill(NaN32, Nx, Ny)
    # 0 = land, 1 = water, −9999 = fill (the GEE coding on the AG100/AG1KM tiles, not
    # the LP DAAC 1/2 coding).
    land_water_map = fill(NaN32, Nx, Ny)

    tile_cache = joinpath(dirname(nc_path), string(short_name, "_tiles"))
    mkpath(tile_cache)

    for url in granule_urls
        h5 = earthdata_download_cached(url, tile_cache)

        tile_longitude = Float64.(read_asterged_subdataset(h5, "//Geolocation/Longitude")[:, 1, 1])
        tile_latitude  = Float64.(read_asterged_subdataset(h5, "//Geolocation/Latitude")[1, :, 1])

        # GDAL returns (Nx, Ny, 5) band-last; permute to the band-first (5, Nx, Ny)
        # the broadband collapse expects.
        mean_bands = permutedims(asterged_decode_emissivity.(read_asterged_subdataset(h5, "//Emissivity/Mean")), (3, 1, 2))
        sdev_bands = permutedims(asterged_decode_uncertainty.(read_asterged_subdataset(h5, "//Emissivity/SDev")), (3, 1, 2))

        lwmap_tile = Float32.(read_asterged_subdataset(h5, "//Land_Water_Map/LWmap")[:, :, 1])

        place_tile!(emissivity,     broadband_map(mean_bands, coefficients), tile_longitude, tile_latitude, longitude, latitude)
        place_tile!(uncertainty,    broadband_map(sdev_bands, coefficients), tile_longitude, tile_latitude, longitude, latitude)
        place_tile!(land_water_map, lwmap_tile,                              tile_longitude, tile_latitude, longitude, latitude)
    end

    all(isnan, emissivity) &&
        error("No ASTER GED tile cells fell within region $(bbox).")

    NCDataset(nc_path, "c") do ds
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)
        defVar(ds, "lon", Float64, ("lon",); attrib = ["units" => "degrees_east", "long_name" => "longitude"])[:] = longitude
        defVar(ds, "lat", Float64, ("lat",); attrib = ["units" => "degrees_north", "long_name" => "latitude"])[:] = latitude
        defVar(ds, "emissivity", Float32, ("lon", "lat"))[:, :] = emissivity
        defVar(ds, "emissivity_uncertainty", Float32, ("lon", "lat"))[:, :] = uncertainty
        defVar(ds, "land_water_map", Float32, ("lon", "lat"))[:, :] = land_water_map
    end

    return nothing
end
