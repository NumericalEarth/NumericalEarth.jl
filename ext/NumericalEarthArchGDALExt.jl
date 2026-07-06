module NumericalEarthArchGDALExt

using ArchGDAL: ArchGDAL
using NCDatasets: NCDataset, defDim, defVar
using NumericalEarth: NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, metadata_path, netrc_downloader

using Downloads: Downloads
using Printf: @sprintf
using Dates: Dates, DateTime, Day, Hour, Second, year, dayofyear, hour

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
##### Land surface temperature ingest (gated behind ArchGDAL)
#####
##### Both products are fetched, decoded (via the pure `goes_lst`/`ecostress_lst`
##### core), and reprojected/clipped to a clean regional lat/lon NetCDF
##### (`lon`, `lat`, `LST` in Kelvin with `NaN` cloud/no-retrieval gaps) at
##### download time, mirroring the MODISLand ingest. The generic `Field` /
##### `set_region_data!` machinery then brackets that raster onto the native grid.
#####

const LST = NumericalEarth.DataWrangling.LandSurfaceTemperature

# Write a regional lat/lon NetCDF of decoded Kelvin LST. `decoded` is already
# flipped to south→north; `gt` is the (north→south) GDAL geotransform of the warp.
function write_lst_netcdf(nc_path, decoded, gt)
    Nx, Ny = size(decoded)
    Δφ = gt[6]  # negative
    longitude = collect(range(gt[1] + gt[2] / 2; step = gt[2], length = Nx))
    latitude  = collect(range(gt[4] + Δφ / 2; step = Δφ, length = Ny))
    reverse!(latitude)  # match the reversed (south→north) data

    NCDataset(nc_path, "c") do ds
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)
        lon_var = defVar(ds, "lon", Float64, ("lon",);
                         attrib = ["units" => "degrees_east", "long_name" => "longitude"])
        lat_var = defVar(ds, "lat", Float64, ("lat",);
                         attrib = ["units" => "degrees_north", "long_name" => "latitude"])
        lst_var = defVar(ds, "LST", Float32, ("lon", "lat");
                         attrib = ["units" => "K", "long_name" => "land surface temperature"])
        lon_var[:] = longitude
        lat_var[:] = latitude
        lst_var[:, :] = decoded
    end
    return nothing
end

#####
##### GOES-R ABI-L2-LSTC — anonymous AWS S3 + geostationary → lat/lon warp
#####

# GOES ABI granule keys carry the start-of-scan tag `_sYYYYDDDHHMMSSt` (day-of-year,
# no `T` separator, trailing tenths). Parse it so we can pick the object closest to
# the requested hour. (The exported `granule_timestamp` handles the `sYYYYDDDThhmmss`
# form; GOES S3 keys omit the `T`.)
function goes_key_timestamp(key)
    m = match(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", key)
    m === nothing && return nothing
    yr  = parse(Int, m.captures[1])
    doy = parse(Int, m.captures[2])
    hr  = parse(Int, m.captures[3])
    mn  = parse(Int, m.captures[4])
    sc  = parse(Int, m.captures[5])
    return DateTime(yr) + Day(doy - 1) + Hour(hr) + Second(60mn + sc)
end

# List the ABI-L2-LSTC objects in the (bucket, YYYY/DDD/HH) prefix anonymously and
# return the (bucket, key) whose start-of-scan time is nearest the requested date.
function goes_object_key(satellite, date)
    bucket = "noaa-" * String(satellite)
    prefix = @sprintf("ABI-L2-LSTC/%04d/%03d/%02d/", year(date), dayofyear(date), hour(date))
    list_url = "https://$bucket.s3.amazonaws.com/?list-type=2&prefix=$prefix"

    keys = String[]
    mktempdir() do tmp
        xml = joinpath(tmp, "list.xml")
        Downloads.download(list_url, xml)
        text = read(xml, String)
        for m in eachmatch(r"<Key>([^<]+\.nc)</Key>", text)
            push!(keys, m.captures[1])
        end
    end
    isempty(keys) &&
        error("No GOES-R ABI-L2-LSTC objects found at s3://$bucket/$prefix (anonymous listing).")

    # Pick the object whose scan time is closest to the requested date.
    best_key = keys[1]
    best_gap = typemax(Int)
    for key in keys
        t = goes_key_timestamp(key)
        t === nothing && continue
        gap = abs(Dates.value(Second(t - date)))
        gap < best_gap && (best_gap = gap; best_key = key)
    end
    return bucket, best_key
end

function LST.goes_granule_to_netcdf(metadatum::LST.GOESMetadatum, nc_path)
    region = metadatum.region
    (region isa BoundingBox && !isnothing(region.longitude) && !isnothing(region.latitude)) ||
        error("goes_granule_to_netcdf requires a BoundingBox region.")

    date = metadatum.dates
    bucket, key = goes_object_key(metadatum.dataset.satellite, date)
    λ = region.longitude
    φ = region.latitude
    Δ = LST.GOES_RESOLUTION

    mktempdir() do tmp
        granule = joinpath(tmp, basename(key))
        Downloads.download("https://$bucket.s3.amazonaws.com/$key", granule)

        # Warp the geostationary `LST`/`DQF` subdatasets to EPSG:4326, clipped to bbox.
        warp(field) = ArchGDAL.read("NETCDF:\"$granule\":$field") do src
            ArchGDAL.gdalwarp([src],
                ["-t_srs", "EPSG:4326",
                 "-te", string(λ[1]), string(φ[1]), string(λ[2]), string(φ[2]),
                 "-tr", string(Δ), string(Δ),
                 "-r", "near"]) do w
                (ArchGDAL.read(w, 1), ArchGDAL.getgeotransform(w))
            end
        end

        dn,  gt = warp("LST")
        dqf, _  = warp("DQF")

        # Pure decode (scale/offset + fill/valid-range + DQF masking), then flip
        # GDAL's north→south rows to south→north to match the NetCDF lat axis.
        K = Float32.(reverse(LST.goes_lst.(round.(Int, dn), round.(Int, dqf)), dims = 2))
        write_lst_netcdf(nc_path, K, gt)
    end
    return nothing
end

#####
##### ECOSTRESS ECO_L2G_LSTE — Earthdata CMR discovery + HDF5 (GDAL) read
#####

# ECOSTRESS L2G granule ids embed a calendar acquisition tag `YYYYMMDDThhmmss`.
function ecostress_id_timestamp(name)
    m = match(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})", name)
    m === nothing && return nothing
    return DateTime(parse.(Int, (m.captures[1], m.captures[2], m.captures[3],
                                 m.captures[4], m.captures[5], m.captures[6]))...)
end

# Build the CMR granule-search URL for ECO_L2G_LSTE intersecting `bbox` (W,S,E,N).
function ecostress_cmr_url(version, bbox, start_date, end_date; page_size = 200)
    west, east = bbox.longitude
    south, north = bbox.latitude
    temporal = string(cmr_time(start_date), ",", cmr_time(end_date))
    return string("https://cmr.earthdata.nasa.gov/search/granules.json",
                  "?short_name=ECO_L2G_LSTE",
                  "&version=", version,
                  "&bounding_box=", west, ",", south, ",", east, ",", north,
                  "&temporal=", temporal,
                  "&page_size=", page_size)
end

cmr_time(date::DateTime) = string(Dates.format(date, "yyyy-mm-ddTHH:MM:SS"), "Z")

# Query CMR and return the (timestamp, .h5 download URL) pairs, sorted by time.
function ecostress_cmr_granules(version, bbox, start_date, end_date)
    url = ecostress_cmr_url(version, bbox, start_date, end_date)
    granules = Tuple{DateTime, String}[]
    mktempdir() do tmp
        json = joinpath(tmp, "cmr.json")
        Downloads.download(url, json)
        text = read(json, String)
        # Protected .h5 granule download URLs (exclude the *_mvs.h5 quicklook aux).
        for m in eachmatch(r"https://[^\"]+/(ECOv002_L2G_LSTE_[^\"/]+)\.h5", text)
            endswith(m.match, "_mvs.h5") && continue
            t = ecostress_id_timestamp(m.captures[1])
            t === nothing && continue
            push!(granules, (t, m.match))
        end
    end
    unique!(granules)
    sort!(granules; by = first)
    return granules
end

# Real (irregular) overpass discovery for `all_dates` / diagnostics.
function LST.ecostress_cmr_overpasses(region::BoundingBox, start_date, end_date; version = "002")
    (!isnothing(region.longitude) && !isnothing(region.latitude)) ||
        error("ecostress_cmr_overpasses requires a bounded BoundingBox region.")
    granules = ecostress_cmr_granules(version, region, start_date, end_date)
    return first.(granules)
end

# Earthdata-authenticated download. Credentials come from EARTHDATA_USERNAME /
# EARTHDATA_PASSWORD (the variables the Python `earthaccess` library also honours).
function earthdata_download(url, path)
    haskey(ENV, "EARTHDATA_USERNAME") && haskey(ENV, "EARTHDATA_PASSWORD") ||
        error("NASA Earthdata credentials not found. Set EARTHDATA_USERNAME and " *
              "EARTHDATA_PASSWORD (register free at https://urs.earthdata.nasa.gov).")
    username = ENV["EARTHDATA_USERNAME"]
    password = ENV["EARTHDATA_PASSWORD"]
    mktempdir() do tmp
        downloader = netrc_downloader(username, password, "urs.earthdata.nasa.gov", tmp)
        Downloads.download(url, path; downloader)
    end
    return path
end

# Locate the `HDF5:"…"://…/Data_Fields/<field>` subdataset without hardcoding the
# per-resolution grid name (`ECO_L2G_LSTE_70m`).
function ecostress_subdataset(h5_path, field)
    ArchGDAL.read(h5_path) do ds
        for entry in ArchGDAL.metadata(ds; domain = "SUBDATASETS")
            occursin("_NAME=", entry) || continue
            name = split(entry, "_NAME="; limit = 2)[2]
            endswith(name, "/" * field) && return name
        end
        error("Field $(field) not found among the ECO_L2G_LSTE HDF5 subdatasets of $(h5_path).")
    end
end

function LST.ecostress_granule_to_netcdf(metadatum::LST.ECOSTRESSMetadatum, nc_path)
    region = metadatum.region
    (region isa BoundingBox && !isnothing(region.longitude) && !isnothing(region.latitude)) ||
        error("ecostress_granule_to_netcdf requires a BoundingBox region.")

    version = metadatum.dataset.version
    date = metadatum.dates
    # Search a ±1-day window around the requested overpass and pick the nearest.
    granules = ecostress_cmr_granules(version, region, date - Day(1), date + Day(1))
    isempty(granules) &&
        error("CMR returned no ECO_L2G_LSTE.$(version) granules for region $(region) near $(date).")

    # Pick the overpass whose acquisition time is closest to the requested date.
    gaps = [abs(Dates.value(Second(t - date))) for (t, _) in granules]
    _, url = granules[argmin(gaps)]

    λ = region.longitude
    φ = region.latitude
    Δ = LST.ECOSTRESS_RESOLUTION

    mktempdir() do tmp
        h5 = joinpath(tmp, "ecostress.h5")
        earthdata_download(url, h5)

        # Clip the (already EPSG:4326) LST + cloud subdatasets to the bbox.
        clip(field, resampler) = ArchGDAL.read(ecostress_subdataset(h5, field)) do src
            ArchGDAL.gdalwarp([src],
                ["-t_srs", "EPSG:4326",
                 "-te", string(λ[1]), string(φ[1]), string(λ[2]), string(φ[2]),
                 "-tr", string(Δ), string(Δ),
                 "-r", resampler]) do w
                (ArchGDAL.read(w, 1), ArchGDAL.getgeotransform(w))
            end
        end

        # Nearest-neighbour so the 0 fill (no-retrieval / off-swath) is never
        # blended into a bogus intermediate temperature; source ≈ target resolution.
        lst,   gt = clip("LST", "near")
        cloud, _  = clip("cloud", "near")

        # ECO_L2G stores 0 as fill; treat non-positive as no-retrieval (NaN), then
        # apply the pure cloud-mask decode. Flip GDAL rows north→south → south→north.
        raw = ifelse.(Float32.(lst) .> 0, Float32.(lst), NaN32)
        K   = Float32.(reverse(LST.ecostress_lst.(raw, round.(Int, cloud)), dims = 2))
        write_lst_netcdf(nc_path, K, gt)
    end
    return nothing
end

end # module NumericalEarthArchGDALExt
