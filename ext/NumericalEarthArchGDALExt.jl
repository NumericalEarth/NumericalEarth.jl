module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using Downloads: Downloads
using NCDatasets: NCDataset, defDim, defVar
using NumericalEarth: NumericalEarth

using Oceananigans: Center, CPU
using Oceananigans.Grids: λnodes, φnodes

using NumericalEarth.DataWrangling: BoundingBox, netrc_downloader, native_grid
using NumericalEarth.DataWrangling.ASTERGED: asterged_short_name, asterged_version,
                                             asterged_cmr_granules_url,
                                             asterged_decode_emissivity, asterged_decode_uncertainty,
                                             broadband_map, place_tile!,
                                             OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS

function NumericalEarth.DataWrangling.IBCAO.reproject_ibcao_to_netcdf(tiff_path, nc_path)
    ArchGDAL.read(tiff_path) do src
        # Warp from EPSG:3996 (Polar Stereographic) to EPSG:4326 (WGS84)
        # at 0.01° resolution, clipping to 64–90°N
        ArchGDAL.gdalwarp([src],
            ["-t_srs", "EPSG:4326",
             "-te",    "-180", "64", "180", "90",  # xmin ymin xmax ymax
             "-tr",    "0.01", "0.01",             # target resolution (degrees)
             "-r",     "bilinear",                 # resampling method
             "-ot",    "Float32"]) do warped

            # ArchGDAL returns data as (Nx, Ny) with y from north to south (GDAL convention)
            data = Float32.(ArchGDAL.read(warped, 1))
            data = reverse(data, dims=2)

            Nx, Ny = size(data)  # expected: (36000, 2600)

            NCDataset(nc_path, "c") do ds
                defDim(ds, "lon", Nx)
                defDim(ds, "lat", Ny)

                lon_var = defVar(ds, "lon", Float64, ("lon",);
                                attrib = ["units" => "degrees_east",
                                          "long_name" => "longitude"])
                lat_var = defVar(ds, "lat", Float64, ("lat",);
                                attrib = ["units" => "degrees_north",
                                          "long_name" => "latitude"])
                z_var   = defVar(ds, "z",   Float32, ("lon", "lat");
                                attrib = ["long_name" => "elevation",
                                          "units"     => "m"])

                lon_var[:] = range(-180 + 0.005, 180 - 0.005; length=Nx)
                lat_var[:] = range(64 + 0.005, 90 - 0.005; length=Ny)
                z_var[:, :] = data
            end
        end
    end

    return nothing
end

#####
##### ASTER GED emissivity ingest (HDF5 tiles on a plain lat/lon grid)
#####
##### Resolves the 1°×1° HDF5 tiles intersecting the region via NASA CMR, downloads
##### them with Earthdata credentials (cached per tile, so overlapping regions and
##### both variables reuse them), reads the `/Emissivity/Mean`, `/Emissivity/SDev`
##### and `/Geolocation/*` subdatasets through GDAL's HDF5 driver, decodes and
##### collapses the five TIR bands to one broadband float per cell, and writes a
##### regional NetCDF of the broadband emissivity + uncertainty on the analytic
##### native grid (NaN over gaps/water, for the downstream inpainting).
#####
##### Requires GDAL_jll built with the HDF5 driver.
#####

# Earthdata-authenticated download of a single tile, retried on transient failures.
# Credentials come from the EARTHDATA_USERNAME / EARTHDATA_PASSWORD env variables.
function earthdata_download(url, path; attempts = 3)
    username = get(ENV, "EARTHDATA_USERNAME", nothing)
    password = get(ENV, "EARTHDATA_PASSWORD", nothing)

    if isnothing(username)
        error("NASA Earthdata credentials not found: EARTHDATA_USERNAME is not set. " *
              "Register free at https://urs.earthdata.nasa.gov.")
    elseif isnothing(password)
        error("NASA Earthdata credentials not found: EARTHDATA_PASSWORD is not set. " *
              "Register free at https://urs.earthdata.nasa.gov.")
    end

    mktempdir() do tmp
        downloader = netrc_downloader(username, password, "urs.earthdata.nasa.gov", tmp)
        for attempt in 1:attempts
            try
                Downloads.download(url, path; downloader)
                break
            catch error
                rm(path, force = true)
                attempt == attempts && rethrow()
                @warn "ASTER GED tile download failed (attempt $attempt of $attempts); retrying..." url error
                sleep(2attempt)
            end
        end
    end
    return path
end

# Persistent per-tile cache: overlapping regions and both variables reuse tiles
# instead of re-downloading. `basename(url)` is the tile granule name.
function earthdata_download_cached(url, cache_dir; attempts = 3)
    path = joinpath(cache_dir, basename(url))
    isfile(path) && return path
    return earthdata_download(url, path; attempts)
end

# Query CMR for the ASTER GED tile `.h5` download URLs intersecting `bbox`,
# de-duplicated by granule (CMR lists both a protected and a public endpoint per
# granule; keep the protected `data#` endpoint).
function NumericalEarth.DataWrangling.ASTERGED.earthdata_cmr_granules(short_name, version, bbox::BoundingBox)
    url = asterged_cmr_granules_url(short_name, version, bbox)
    urls = String[]
    mktempdir() do tmp
        json_path = joinpath(tmp, "cmr_granules.json")
        Downloads.download(url, json_path)
        text = read(json_path, String)
        for match in eachmatch(r"https://[^\"]+\.h5", text)
            push!(urls, match.match)
        end
    end
    by_granule = Dict{String, String}()
    for u in urls
        key = basename(u)
        if !haskey(by_granule, key) || occursin("protected", u)
            by_granule[key] = u
        end
    end
    return collect(values(by_granule))
end

# Open an HDF5 subdataset via GDAL's `HDF5:"file"://path` syntax and return the
# full raster array. Multi-band datasets come back as `(Nx, Ny, nbands)`.
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

    # Target axes = the same native grid the Field will use, so tile cells land on
    # file cells by construction (no empirical axis reconstruction, no half-pixel
    # registration guesswork).
    grid = native_grid(metadatum, CPU())
    longitude = collect(λnodes(grid, Center()))
    latitude  = collect(φnodes(grid, Center()))
    Nx, Ny = length(longitude), length(latitude)
    Δλ = (longitude[end] - longitude[1]) / max(Nx - 1, 1)
    Δφ = (latitude[end]  - latitude[1])  / max(Ny - 1, 1)

    short_name = asterged_short_name(dataset)
    version = asterged_version(dataset)
    coefficients = OGAWA_SCHMUGGE_2004_BROADBAND_COEFFICIENTS

    # Query CMR over the native extent (padded a cell) so every file cell is covered.
    query = BoundingBox(longitude = (longitude[1] - Δλ, longitude[end] + Δλ),
                        latitude  = (latitude[1]  - Δφ, latitude[end]  + Δφ))
    granule_urls = NumericalEarth.DataWrangling.ASTERGED.earthdata_cmr_granules(short_name, version, query)
    isempty(granule_urls) &&
        error("CMR returned no $(short_name).$(version) tiles for region $(bbox).")

    emissivity  = fill(NaN32, Nx, Ny)
    uncertainty = fill(NaN32, Nx, Ny)

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

        place_tile!(emissivity,  broadband_map(mean_bands, coefficients), tile_longitude, tile_latitude, longitude, latitude)
        place_tile!(uncertainty, broadband_map(sdev_bands, coefficients), tile_longitude, tile_latitude, longitude, latitude)
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
    end

    return nothing
end

end # module NumericalEarthArchGDALExt
