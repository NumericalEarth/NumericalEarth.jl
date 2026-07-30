#####
##### GloFAS river-discharge download (Copernicus Emergency Management Service)
#####
##### GloFAS lives on the Early Warning Data Store (EWDS), a separate Copernicus
##### endpoint from the ERA5 CDS. Configure `~/.cdsapirc` with the EWDS API url
##### (https://ewds.climate.copernicus.eu/api) and key, and accept the
##### `cems-glofas-historical` licence, before downloading.
#####

glofas_product(::GloFASDataset) = "cems-glofas-historical"

const GLOFAS_EWDS_URL = "https://ewds.climate.copernicus.eu/api"

restore_env!(name, ::Nothing) = (delete!(ENV, name); nothing)
restore_env!(name, value) = (ENV[name] = value; nothing)

# GloFAS lives on EWDS, a different Copernicus endpoint than the ERA5 CDS. We
# point CDSAPI at the EWDS url by temporarily setting `CDSAPI_URL` (which CDSAPI
# reads above `~/.cdsapirc`), so a `~/.cdsapirc` pointed at the ERA5 CDS keeps
# working — the ECMWF token is shared across data stores, so only the url
# differs. The `GLOFAS_CDSAPI_URL` / `GLOFAS_CDSAPI_KEY` environment variables
# override the defaults (an empty key falls back to the key CDSAPI already
# resolves from the environment or `~/.cdsapirc`).
function glofas_retrieve(product, request, path)
    url = get(ENV, "GLOFAS_CDSAPI_URL", GLOFAS_EWDS_URL)
    key = get(ENV, "GLOFAS_CDSAPI_KEY", "")

    saved_url = get(ENV, "CDSAPI_URL", nothing)
    saved_key = get(ENV, "CDSAPI_KEY", nothing)

    ENV["CDSAPI_URL"] = url
    isempty(key) || (ENV["CDSAPI_KEY"] = key)

    try
        return retrieve_with_retries(product, request, path)
    finally
        restore_env!("CDSAPI_URL", saved_url)
        isempty(key) || restore_env!("CDSAPI_KEY", saved_key)
    end
end

const GLOFAS_COORD_VARS = Set(["longitude", "latitude",
                               "time", "valid_time", "step", "surface"])

"""
    build_glofas_request(dataset, datetimes, region) -> Dict{String, Any}

Construct the EWDS request for a batch of GloFAS dates that share a `(year, month)`.
GloFAS uses the `hyear`/`hmonth`/`hday` date keys (interpreted as a Cartesian product).
A `BoundingBox` `region` is sent as an `area` key so the EWDS subsets server-side.
"""
function build_glofas_request(dataset, datetimes, region)
    dts = datetimes isa AbstractVector ? datetimes : [datetimes]

    years  = unique(string.(Dates.year.(dts)))
    months = unique(lpad.(string.(Dates.month.(dts)), 2, '0'))
    days   = unique(lpad.(string.(Dates.day.(dts)), 2, '0'))

    request = Dict{String, Any}(
        "system_version"     => [dataset.system_version],
        "hydrological_model" => ["lisflood"],
        "product_type"       => ["consolidated"],
        "variable"           => ["river_discharge_in_the_last_24_hours"],
        "hyear"              => years,
        "hmonth"             => months,
        "hday"               => days,
        "data_format"        => "netcdf",
        "download_format"    => "unarchived",
    )

    area = glofas_request_area(region)
    isnothing(area) || (request["area"] = area)

    return request
end

glofas_request_area(region) = nothing

# Pad the box by a few native (0.05°) cells so the file fully covers the
# center-bracketed native grid the data is interpolated onto (cf. ERA5).
function glofas_request_area(bbox::BBOX)
    (isnothing(bbox.longitude) || isnothing(bbox.latitude)) && return nothing
    pad = 0.2
    north = min(bbox.latitude[2]  + pad,  90)
    south = max(bbox.latitude[1]  - pad, -90)
    west  = bbox.longitude[1] - pad
    east  = bbox.longitude[2] + pad
    return [north, west, south, east]
end

"""
    download(meta::GloFASMetadatum; skip_existing=true)

Download GloFAS river discharge for a single date via the EWDS CDS API.
"""
function Downloads.download(meta::GloFASMetadatum; skip_existing=true)
    output_path = metadata_path(meta)
    skip_existing && isfile(output_path) && return output_path

    mkpath(dirname(output_path))
    request = build_glofas_request(meta.dataset, meta.dates, meta.region)
    @root glofas_retrieve(glofas_product(meta.dataset), request, output_path)

    return output_path
end

"""
    download(metadata::GloFASMetadata; skip_existing=true, cleanup=true)

Download GloFAS river discharge for multiple dates, batching by calendar month
and splitting the multi-timestep NetCDF into one file per day.
"""
function Downloads.download(metadata::GloFASMetadata; skip_existing=true, cleanup=true)
    dates = metadata.dates isa AbstractVector ? metadata.dates : [metadata.dates]
    monthly = group_by_calendar_month(dates)

    paths = String[]
    for key in sort(collect(keys(monthly)))
        batch = sort(unique(monthly[key]))
        append!(paths, download_glofas_month(metadata.name, metadata.dataset, batch;
                                             region = metadata.region,
                                             dir = metadata.dir,
                                             skip_existing, cleanup))
    end

    return paths
end

function download_glofas_month(name, dataset, dates; region, dir, skip_existing, cleanup)
    meta_filename = NumericalEarth.DataWrangling.metadata_filename

    dt_path_pairs = [(dt, joinpath(dir, meta_filename(dataset, name, dt, region))) for dt in dates]
    pending = skip_existing ? filter(dt_path -> !isfile(dt_path[2]), dt_path_pairs) : dt_path_pairs
    isempty(pending) && return map(dt_path -> dt_path[2], dt_path_pairs)

    mkpath(dir)
    sorted_dts = sort(unique([dt for (dt, _) in pending]))
    request = build_glofas_request(dataset, sorted_dts, region)

    dt0 = first(sorted_dts)
    tmp_path = joinpath(dir, "_tmp_glofas_$(Dates.year(dt0))$(lpad(Dates.month(dt0), 2, '0')).nc")
    nc_varname = GloFAS_netcdf_variable_names[name]
    nc_triples = [(nc_varname, dt, path) for (dt, path) in pending]

    time_dimnames = Set(["time", "valid_time"])
    @root begin
        glofas_retrieve(glofas_product(dataset), request, tmp_path)
        foreach_nc(tmp_path, dir) do nc_path
            split_era5_nc_multistep(nc_path, nc_triples, GLOFAS_COORD_VARS, time_dimnames)
        end
        cleanup && rm(tmp_path; force=true)
    end

    return map(dt_path -> dt_path[2], dt_path_pairs)
end
