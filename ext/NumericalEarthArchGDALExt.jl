module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using Downloads: Downloads
using NCDatasets: NCDataset, defDim, defVar
using NumericalEarth: NumericalEarth

using NumericalEarth.DataWrangling: BoundingBox, netrc_downloader
using NumericalEarth.DataWrangling.ASTERGED: asterged_short_name,
                                             asterged_version, asterged_cmr_granules_url

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
#####
##### Resolves the 1°×1° HDF5 tiles intersecting a BoundingBox via NASA CMR,
##### downloads them with Earthdata credentials, reads the `/Emissivity/Mean`,
##### `/Emissivity/SDev` and `/Land_Water_Map/LWmap` subdatasets (+ the
##### `/Geolocation/*` arrays) through GDAL's HDF5 driver, clips + mosaics them to
##### the region, and writes a regional NetCDF of *raw digital numbers* (no scale
##### applied — the ASTERGED module decodes/blends/masks on read). ASTER GED is
##### already on a plain geographic (WGS84 lat/lon) grid, so no reprojection is
##### needed; the coordinates come from the `/Geolocation/*` arrays (the HDF5
##### subdatasets carry an identity geotransform).
#####
##### Requires GDAL_jll built with the HDF5 driver.
#####

# Earthdata-authenticated download of a single tile, retried on transient
# failures (Earthdata Cloud drops connections occasionally). Credentials come
# from the EARTHDATA_USERNAME / EARTHDATA_PASSWORD environment variables.
function earthdata_download(url, path; attempts = 3)
    haskey(ENV, "EARTHDATA_USERNAME") && haskey(ENV, "EARTHDATA_PASSWORD") ||
        error("NASA Earthdata credentials not found. Set EARTHDATA_USERNAME and " *
              "EARTHDATA_PASSWORD (register free at https://urs.earthdata.nasa.gov).")
    username = ENV["EARTHDATA_USERNAME"]
    password = ENV["EARTHDATA_PASSWORD"]
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

# Index in the sorted vector `sorted` whose value is closest to `x`.
function nearest_index(sorted, x)
    i = searchsortedfirst(sorted, x)
    i <= 1 && return 1
    i > length(sorted) && return length(sorted)
    return (x - sorted[i-1]) <= (sorted[i] - x) ? i - 1 : i
end

function NumericalEarth.DataWrangling.ASTERGED.asterged_tiles_to_netcdf(metadatum, nc_path::AbstractString)
    bbox = metadatum.region
    (bbox isa BoundingBox && !isnothing(bbox.longitude) && !isnothing(bbox.latitude)) ||
        error("asterged_tiles_to_netcdf requires a BoundingBox region.")

    short_name = asterged_short_name(metadatum.dataset)
    version = asterged_version(metadatum.dataset)

    granule_urls = NumericalEarth.DataWrangling.ASTERGED.earthdata_cmr_granules(short_name, version, bbox)
    isempty(granule_urls) &&
        error("CMR returned no $(short_name).$(version) tiles for region $(bbox).")

    west, east = bbox.longitude
    south, north = bbox.latitude
    margin = 0.02  # pad the clip a few native cells so the model grid stays covered

    mktempdir() do tmp
        # Download the intersecting tiles.
        hdf_paths = String[]
        for (n, url) in enumerate(granule_urls)
            hdf_path = joinpath(tmp, string(short_name, "_tile_", n, ".h5"))
            earthdata_download(url, hdf_path)
            push!(hdf_paths, hdf_path)
        end

        # Read + clip each tile to the (padded) bbox using its geolocation arrays.
        # Per tile the mean/sdev arrays are (Nx, Ny, 5) with longitude increasing
        # along dim 1 and latitude *decreasing* along dim 2 (north→south).
        clipped = NamedTuple[]
        for h5 in hdf_paths
            longitude = Float64.(read_asterged_subdataset(h5, "//Geolocation/Longitude")[:, 1, 1])
            latitude  = Float64.(read_asterged_subdataset(h5, "//Geolocation/Latitude")[1, :, 1])

            ii = findall(λ -> (west - margin) <= λ <= (east + margin), longitude)
            jj = findall(φ -> (south - margin) <= φ <= (north + margin), latitude)
            (isempty(ii) || isempty(jj)) && continue

            mean_dn = read_asterged_subdataset(h5, "//Emissivity/Mean")[ii, jj, :]
            sdev_dn = read_asterged_subdataset(h5, "//Emissivity/SDev")[ii, jj, :]
            lwmap   = read_asterged_subdataset(h5, "//Land_Water_Map/LWmap")[ii, jj, 1]

            push!(clipped, (longitude = longitude[ii], latitude = latitude[jj],
                            mean_dn = mean_dn, sdev_dn = sdev_dn, lwmap = lwmap))
        end
        isempty(clipped) &&
            error("No ASTER GED tile cells fell within region $(bbox).")

        # Build the mosaic coordinate axes (ascending; south→north for latitude)
        # by taking the sorted union of the clipped tiles' cell centers. Rounding
        # collapses the Float32 boundary cell shared by adjacent tiles.
        all_longitude = sort(unique(round.(vcat((t.longitude for t in clipped)...), digits = 6)))
        all_latitude  = sort(unique(round.(vcat((t.latitude  for t in clipped)...), digits = 6)))
        Nx = length(all_longitude)
        Ny = length(all_latitude)

        # Scatter each tile's clipped cells onto the mosaic grid. Unfilled cells
        # get the emissivity fill (-9999 → NaN on decode) and land (0) for LWmap.
        mean_out = fill(Int16(-9999), 5, Nx, Ny)
        sdev_out = fill(Int16(-9999), 5, Nx, Ny)
        lwmap_out = zeros(Int16, Nx, Ny)
        for t in clipped
            for (jl, φ) in enumerate(t.latitude)
                j = nearest_index(all_latitude, φ)
                for (il, λ) in enumerate(t.longitude)
                    i = nearest_index(all_longitude, λ)
                    @views mean_out[:, i, j] .= Int16.(t.mean_dn[il, jl, :])
                    @views sdev_out[:, i, j] .= Int16.(t.sdev_dn[il, jl, :])
                    lwmap_out[i, j] = Int16(t.lwmap[il, jl])
                end
            end
        end

        NCDataset(nc_path, "c") do ds
            defDim(ds, "band", 5)
            defDim(ds, "lon", Nx)
            defDim(ds, "lat", Ny)
            lon_var = defVar(ds, "lon", Float64, ("lon",);
                             attrib = ["units" => "degrees_east", "long_name" => "longitude"])
            lat_var = defVar(ds, "lat", Float64, ("lat",);
                             attrib = ["units" => "degrees_north", "long_name" => "latitude"])
            lon_var[:] = all_longitude
            lat_var[:] = all_latitude

            # Raw digital numbers, no CF scale/offset — the ASTERGED module decodes.
            mean_var = defVar(ds, "emissivity_mean", Int16, ("band", "lon", "lat"))
            mean_var[:, :, :] = mean_out
            sdev_var = defVar(ds, "emissivity_sdev", Int16, ("band", "lon", "lat"))
            sdev_var[:, :, :] = sdev_out
            lwmap_var = defVar(ds, "land_water_map", Int16, ("lon", "lat"))
            lwmap_var[:, :] = lwmap_out
        end
    end

    return nothing
end

end # module NumericalEarthArchGDALExt
