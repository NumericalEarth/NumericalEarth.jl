#####
##### ERA5 downloads from the Copernicus Climate Data Store
#####
##### Every entry point takes `client` (a `CDSAPIClient()` or `CopernicusClimateDataStoreClient()`,
##### defaulting to `default_cds_client()`) and `retrieve`, the `(product, request, path)` function
##### that client fetches with. The requests, batching, and splitting below do not depend on the client.
#####

#####
##### Dispatch helpers — encapsulate single-level vs pressure-level differences
#####

cds_product(::ERA5Dataset)               = "reanalysis-era5-single-levels"
cds_product(::ERA5PressureLevelsDataset) = "reanalysis-era5-pressure-levels"

cds_varnames(::ERA5Dataset)               = ERA5_dataset_variable_names
cds_varnames(::ERA5PressureLevelsDataset) = ERA5PL_dataset_variable_names

extra_request_keys!(request, ::ERA5Dataset) = nothing
function extra_request_keys!(request, ds::ERA5PressureLevelsDataset)
    p_hPa = [round(Int, p * 1e-2) for p in ds.pressure_levels]
    request["pressure_level"] = [string(p) for p in p_hPa]
end

#####
##### CDS request construction — pure, network-free
#####

"""
    build_era5_request(name_or_names, dataset, datetimes; region) -> Dict{String, Any}

Construct the CDS API request dictionary for one batch of ERA5 data.

`name_or_names` is a `Symbol` or `Vector{Symbol}` of internal variable names.
`datetimes` is a single `DateTime` or a vector of `DateTime`s; all entries must share
the same `(year, month)` (CDS interprets `year`/`month`/`day`/`time` as a Cartesian
product, so mixing months would request the cross product of invalid dates). One
`day` and one `time` string are emitted per unique day and per unique hour found in
`datetimes`. `region` is `nothing`, a `BoundingBox`, or a `Column`.

The returned dictionary always uses zero-padded month/day/hour strings, sets the `area`
key only when `region` produces one, and adds dataset-specific extras (e.g.
`pressure_level` for pressure-level datasets).
"""
function build_era5_request(name_or_names, dataset, datetimes; region)
    names = name_or_names isa Symbol ? [name_or_names] : name_or_names
    cds_vars = unique([cds_varnames(dataset)[n] for n in names])

    dts = datetimes isa AbstractVector ? datetimes : [datetimes]

    years  = unique(string.(Dates.year.(dts)))
    months = unique(lpad.(string.(Dates.month.(dts)), 2, '0'))
    days   = unique(lpad.(string.(Dates.day.(dts)), 2, '0'))
    hours  = unique([lpad(string(Dates.hour(dt)), 2, '0') * ":00" for dt in dts])

    request = Dict{String, Any}(
        "product_type"    => ["reanalysis"],
        "variable"        => cds_vars,
        "year"            => years,
        "month"           => months,
        "day"             => days,
        "time"            => hours,
        "data_format"     => "netcdf",
        "download_format" => "unarchived",
    )

    extra_request_keys!(request, dataset)

    area = era5_request_area(region, dataset, first(names))
    isnothing(area) || (request["area"] = area)

    return request
end

#####
##### Single-date download
#####

"""
$(TYPEDSIGNATURES)

Download ERA5 data for a single date/time from the Copernicus Climate Data Store.

# Keyword Arguments
- `skip_existing`: Skip download if the file already exists (default: `true`).
- `client`: the CDS client, `CDSAPIClient()` or `CopernicusClimateDataStoreClient()`
  (default: [`default_cds_client()`](@ref NumericalEarth.DataWrangling.default_cds_client)).
- `retrieve`: the function `(product, request, path)` that fetches the request (default: `client`'s).

# Environment Setup
Before downloading, you must:
1. Create an account at https://cds.climate.copernicus.eu/
2. Accept the Terms of Use for the ERA5 dataset on the dataset page
3. Set up your API credentials in `~/.cdsapirc`

See https://cds.climate.copernicus.eu/how-to-api for details.
"""
function Downloads.download(meta::ERA5Metadatum; skip_existing = true,
                            client = default_cds_client(), retrieve = cds_retrieve(client))
    output_path = metadata_path(meta)

    skip_existing && isfile(output_path) && return output_path

    mkpath(dirname(output_path))

    request = build_era5_request(meta.name, meta.dataset, meta.dates; region=meta.region)

    @root retrieve_with_retries(cds_product(meta.dataset), request, output_path; retrieve)

    return output_path
end

#####
##### Multi-date download — batches by calendar month, capped by CDS cost
#####

function Downloads.download(metadata::ERA5Metadata; skip_existing = true, cleanup = true,
                            client = default_cds_client(), retrieve = cds_retrieve(client))
    dates = metadata.dates isa AbstractVector ? metadata.dates : [metadata.dates]
    batches = batch_datetimes_for_cds(dates, metadata.dataset, 1)

    paths = String[]
    for batch in batches
        path = download_era5_month(metadata.name, metadata.dataset, batch;
                                   region = metadata.region,
                                   dir = metadata.dir,
                                   skip_existing, cleanup, retrieve)
        append!(paths, path)
    end

    return paths
end

"""
    plan_era5_month(name, dataset, dates; region, dir, skip_existing) -> NamedTuple

Pure planner for a single-variable ERA5 download whose `dates` all share the
same `(year, month)`. Computes the per-datetime output paths, filters to the
subset that needs downloading, and (when there is work to do) builds the CDS
request, the temporary download path, and the NetCDF splitting triples. No
I/O beyond `isfile` checks; no network.

Returned NamedTuple fields:
- `dt_path_pairs`: every `(datetime, path)` pair the caller should report.
- `pending`: subset of `dt_path_pairs` that still need a download.
- `request`, `tmp_path`, `nc_triples`: `nothing` when `pending` is empty; otherwise the
  CDS request dict, the temporary multi-step NetCDF path, and the per-datetime split
  triples consumed by `split_era5_nc_by_datetime`.
"""
function plan_era5_month(name, dataset, dates; region, dir, skip_existing)
    dt_path_pairs = [(dt, joinpath(dir, metadata_filename(dataset, name, dt, region)))
                     for dt in dates]

    pending = if skip_existing
        filter(dt_path -> !isfile(dt_path[2]), dt_path_pairs)
    else
        dt_path_pairs
    end

    if isempty(pending)
        return (; dt_path_pairs, pending,
                  request=nothing, tmp_path=nothing, nc_triples=nothing)
    end

    sorted_dts = sort(unique([dt for (dt, _) in pending]))

    request = build_era5_request(name, dataset, sorted_dts; region)

    dt0   = first(sorted_dts)
    year  = string(Dates.year(dt0))
    month = lpad(string(Dates.month(dt0)), 2, '0')
    day   = lpad(string(Dates.day(dt0)),   2, '0')

    tmp_path   = joinpath(dir, "_tmp_$(year)$(month)$(day).nc")
    nc_varname = nc_varnames(dataset)[name]
    nc_triples = [(nc_varname, dt, filepath) for (dt, filepath) in pending]

    return (; dt_path_pairs, pending, request, tmp_path, nc_triples)
end

function download_era5_month(name, dataset, dates;
                             region, dir, skip_existing, cleanup,
                             retrieve = cds_retrieve(default_cds_client()))

    plan = plan_era5_month(name, dataset, dates; region, dir, skip_existing)
    isempty(plan.pending) && return map(dt_path -> dt_path[2], plan.dt_path_pairs)

    mkpath(dir)

    @root begin
        retrieve_with_retries(cds_product(dataset), plan.request, plan.tmp_path; retrieve)
        foreach_nc(plan.tmp_path, dir) do nc_path
            split_era5_nc_by_datetime(nc_path, plan.nc_triples, coord_vars(dataset), ERA5_TIME_DIMNAMES)
        end
        cleanup && rm(plan.tmp_path; force=true)
    end

    return map(dt_path -> dt_path[2], plan.dt_path_pairs)
end

#####
##### Multi-variable ERA5 download
#####

"""
    Downloads.download(names::Vector{Symbol}, metadata::ERA5Metadata; kwargs...)

Download multiple ERA5 variables for every date in `metadata`, bundling variables
and datetimes into month-batched CDS requests.
"""
function Downloads.download(names::Vector{Symbol}, metadata::ERA5Metadata; kwargs...)
    dates = metadata.dates isa AbstractVector ? metadata.dates : [metadata.dates]
    return Downloads.download(names, metadata.dataset, dates;
                              region = metadata.region,
                              dir = metadata.dir,
                              kwargs...)
end

"""
    Downloads.download(mset::MetadataSet{<:ERA5Dataset}; kwargs...)

Route a `MetadataSet` of ERA5 variables through the multi-variable batched CDS
path, instead of falling back to per-variable requests via the default
`Downloads.download(::MetadataSet)`: each calendar-month batch of variables ×
datetimes is bundled into one CDS API request (capped by the CDS cost limit).
"""
function Downloads.download(mset::MetadataSet{<:ERA5Dataset}; kwargs...)
    names = collect(getfield(mset, :names))
    dates = getfield(mset, :dates)
    dates = dates isa AbstractVector ? dates : [dates]

    return Downloads.download(names, getfield(mset, :dataset), dates;
                              region = getfield(mset, :region),
                              dir = getfield(mset, :dir),
                              kwargs...)
end

"""
$(TYPEDSIGNATURES)

Download multiple ERA5 pressure-level variables for a single date in one CDS API request.
The multi-variable NetCDF is split into individual per-variable files.
"""
function Downloads.download(names::Vector{Symbol}, meta::ERA5PressureMetadatum; skip_existing = true,
                            client = default_cds_client(), retrieve = cds_retrieve(client))
    name_path_pairs = []
    for name in names
        metadatum = Metadatum(name;
                              dataset = meta.dataset,
                              region = meta.region,
                              date = meta.dates,
                              dir = meta.dir)
        path = metadata_path(metadatum)
        push!(name_path_pairs, (name, path))
    end

    pending = if skip_existing
        filter(name_path -> !isfile(name_path[2]), name_path_pairs)
    else
        name_path_pairs
    end

    isempty(pending) && return map(name_path -> name_path[2], name_path_pairs)

    pending_names = [varname for (varname, _) in pending]
    request = build_era5_request(pending_names, meta.dataset, meta.dates; region=meta.region)

    date  = meta.dates
    year  = string(Dates.year(date))
    month = lpad(string(Dates.month(date)), 2, '0')
    day   = lpad(string(Dates.day(date)),   2, '0')
    hour  = lpad(string(Dates.hour(date)),  2, '0') * ":00"

    mkpath(meta.dir)
    tmp_path = joinpath(meta.dir, "_tmp_multi_$(year)$(month)$(day)T$(hour[1:2]).nc")

    nc_name_path_pairs = [(nc_varnames(meta.dataset)[varname], filepath) for (varname, filepath) in pending]

    @root begin
        retrieve_with_retries(cds_product(meta.dataset), request, tmp_path; retrieve)
        foreach_nc(tmp_path, meta.dir) do nc_path
            split_era5_nc(nc_path, nc_name_path_pairs, coord_vars(meta.dataset))
        end
        rm(tmp_path; force=true)
    end

    return map(name_path -> name_path[2], name_path_pairs)
end

"""
$(TYPEDSIGNATURES)

Download one or more ERA5 variables at a single datetime.
"""
function Downloads.download(names::Vector{Symbol}, dataset::ERA5Dataset, datetime;
                            region = nothing,
                            dir = default_download_directory(dataset),
                            kw...)
    meta = Metadatum(first(names); dataset, date=datetime, region, dir)
    return Downloads.download(names, meta; kw...)
end

function Downloads.download(name::Symbol, dataset::ERA5Dataset, datetime;
                            region = nothing,
                            dir = default_download_directory(dataset),
                            kw...)
    return Downloads.download([name], dataset, datetime; region, dir, kw...)
end

"""
$(TYPEDSIGNATURES)

Download one or more ERA5 variables for multiple datetimes, batching by calendar month.
"""
function Downloads.download(names::Vector{Symbol},
                            dataset::ERA5Dataset,
                            datetimes::AbstractVector;
                            region = nothing,
                            dir = default_download_directory(dataset),
                            skip_existing = true,
                            cleanup = true,
                            client = default_cds_client(),
                            retrieve = cds_retrieve(client))

    batches = batch_datetimes_for_cds(datetimes, dataset, length(names))

    paths = String[]
    for batch in batches
        path = download_era5_multivar_month(names, dataset, batch;
                                            region, dir, skip_existing, cleanup, retrieve)
        append!(paths, path)
    end

    return paths
end

function Downloads.download(name::Symbol,
                            dataset::ERA5Dataset,
                            datetimes::AbstractVector;
                            region = nothing,
                            dir = default_download_directory(dataset),
                            kw...)
    return Downloads.download([name], dataset, datetimes; region, dir, kw...)
end

"""
    plan_era5_multivar_month(names, dataset, dates; region, dir, skip_existing) -> NamedTuple

Pure planner for a multi-variable ERA5 download whose `dates` all share the
same `(year, month)`. Same shape as [`plan_era5_month`](@ref), but indexed by
`(name, datetime, path)` triples so each split file is identified by both the
variable name and the timestep.

Returned NamedTuple fields:
- `name_dt_paths`: every `(name, datetime, path)` triple the caller should report.
- `pending`: subset that still needs a download.
- `request`, `tmp_path`, `nc_triples`: `nothing` when `pending` is empty; otherwise the
  CDS request dict, the temporary multi-step NetCDF path, and the per-(name, time) split
  triples consumed by `split_era5_nc_by_datetime`.
"""
function plan_era5_multivar_month(names, dataset, dates; region, dir, skip_existing)
    name_dt_paths = [(varname, dt, joinpath(dir, metadata_filename(dataset, varname, dt, region)))
                     for varname in names for dt in dates]

    pending = if skip_existing
        filter(name_dt_path -> !isfile(name_dt_path[3]), name_dt_paths)
    else
        name_dt_paths
    end

    if isempty(pending)
        return (; name_dt_paths, pending,
                  request=nothing, tmp_path=nothing, nc_triples=nothing)
    end

    pending_names = unique(map(name_dt_path -> name_dt_path[1], pending))
    sorted_dts    = sort(unique(map(name_dt_path -> name_dt_path[2], pending)))

    request = build_era5_request(pending_names, dataset, sorted_dts; region)

    dt0   = first(sorted_dts)
    year  = string(Dates.year(dt0))
    month = lpad(string(Dates.month(dt0)), 2, '0')
    day   = lpad(string(Dates.day(dt0)),   2, '0')

    tmp_path   = joinpath(dir, "_tmp_multi_$(year)$(month)$(day).nc")
    nc_triples = [(nc_varnames(dataset)[varname], dt, filepath)
                  for (varname, dt, filepath) in pending]

    return (; name_dt_paths, pending, request, tmp_path, nc_triples)
end

function download_era5_multivar_month(names, dataset, dates;
                                      region, dir, skip_existing, cleanup,
                                      retrieve = cds_retrieve(default_cds_client()))

    plan = plan_era5_multivar_month(names, dataset, dates; region, dir, skip_existing)
    isempty(plan.pending) && return map(name_dt_path -> name_dt_path[3], plan.name_dt_paths)

    mkpath(dir)

    @root begin
        retrieve_with_retries(cds_product(dataset), plan.request, plan.tmp_path; retrieve)
        foreach_nc(plan.tmp_path, dir) do nc_path
            split_era5_nc_by_datetime(nc_path, plan.nc_triples, coord_vars(dataset), ERA5_TIME_DIMNAMES)
        end
        cleanup && rm(plan.tmp_path; force=true)
    end

    return map(name_dt_path -> name_dt_path[3], plan.name_dt_paths)
end

#####
##### Yearly and monthly products: one file per year or month rather than per hourly step,
##### fetched through CopernicusClimateDataStore's `yearly` / `monthly` entry points.
#####

const NativeGranularityERA5Dataset = Union{ERA5YearlySingleLevel,
                                           ERA5MonthlySingleLevel,
                                           ERA5MonthlyPressureLevels}

"""
$(TYPEDSIGNATURES)

Download the yearly or monthly ERA5 file containing `meta` with `client`, which must be
`CopernicusClimateDataStoreClient()` (the default when CopernicusClimateDataStore is loaded).
"""
function Downloads.download(meta::Metadatum{<:NativeGranularityERA5Dataset}; client = default_cds_client(), kwargs...)
    return download_native_granularity_era5(client, meta; kwargs...)
end

function Downloads.download(metadata::Metadata{<:NativeGranularityERA5Dataset}; kwargs...)
    return [Downloads.download(metadatum; kwargs...) for metadatum in metadata]
end

function Downloads.download(mset::MetadataSet{<:NativeGranularityERA5Dataset}; kwargs...)
    return reduce(vcat, [Downloads.download(metadata; kwargs...) for metadata in mset]; init=String[])
end

download_native_granularity_era5(client, meta; kwargs...) =
    throw(ArgumentError("Downloading $(nameof(typeof(meta.dataset))) is implemented only for " *
                        "`CopernicusClimateDataStoreClient()`; load CopernicusClimateDataStore."))

#####
##### Area/bounding box utilities
#####

build_era5_area(::Nothing) = nothing

# Columns and unbounded regions: the area is a pure function of the region.
era5_request_area(region, dataset, name) = build_era5_area(region)

# Bounding box: the native grid is built by center-bracketing `restrict`, which
# can reach one cell past a boundary-aligned edge. Fetch two native cells of
# margin (in the bbox's own longitude convention) so the downloaded file always
# covers the grid the data is interpolated onto — otherwise downscaling leaves
# NaNs at the domain edges. Over-fetching is harmless: `restrict` selects the
# exact cells from the larger file.
function era5_request_area(bbox::BoundingBox, dataset, name)
    (isnothing(bbox.longitude) || isnothing(bbox.latitude)) && return nothing
    Nx, Ny, _ = size(dataset, name)
    Δλ = 360 / Nx
    Δφ = 180 / Ny
    lon = bbox.longitude
    lat = bbox.latitude
    padded = BoundingBox(longitude = (lon[1] - 2Δλ, lon[2] + 2Δλ),
                         latitude  = (max(lat[1] - 2Δφ, -90), min(lat[2] + 2Δφ, 90)))
    return build_era5_area(padded)
end

function build_era5_area(bbox::BoundingBox)
    lon = bbox.longitude
    lat = bbox.latitude

    if isnothing(lon) || isnothing(lat)
        return nothing
    end

    west  = lon[1]
    east  = lon[2]
    south = lat[1]
    north = lat[2]

    return [north, west, south, east]
end

# Column with Nearest interpolation: tight box; CDS returns the nearest cell.
function build_era5_area(col::Column{<:Any, <:Any, <:Any, <:Nearest})
    lon, lat = col.longitude, col.latitude
    ε = 1e-3
    return [lat + ε, lon - ε, lat - ε, lon + ε]  # [N, W, S, E]
end

# Column with Linear interpolation: pad by slightly more than ERA5's native
# 0.25° spacing so the file contains the 2x2 stencil bilinear interp needs.
function build_era5_area(col::Column{<:Any, <:Any, <:Any, <:DataWrangling.Linear})
    lon, lat = col.longitude, col.latitude
    ε = 0.3
    return [lat + ε, lon - ε, lat - ε, lon + ε]
end
