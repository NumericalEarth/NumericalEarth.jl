module NumericalEarthCDSAPIExt

using CDSAPI: CDSAPI
using Downloads: Downloads

using Dates: Dates
using Oceananigans: Oceananigans
using Oceananigans.DistributedComputations: @root

using NCDatasets: NCDatasets, name, path

using NumericalEarth: NumericalEarth
using NumericalEarth.DataWrangling: Metadatum, MetadataSet, default_download_directory, metadata_path
using NumericalEarth.DataWrangling.ERA5: ERA5Dataset, ERA5Metadata, ERA5Metadatum,
                                         ERA5_dataset_variable_names, ERA5_netcdf_variable_names,
                                         ERA5PressureLevelsDataset,
                                         ERA5PressureMetadata, ERA5PressureMetadatum,
                                         ERA5PL_dataset_variable_names, ERA5PL_netcdf_variable_names
using NumericalEarth.DataWrangling.GloFAS: GloFASDataset, GloFASMetadata, GloFASMetadatum,
                                           GloFAS_netcdf_variable_names
using NumericalEarth.DataWrangling.CopernicusLandAlbedo: CopernicusAlbedo,
                                                         albedo_cds_request_variables,
                                                         albedo_source_variable_candidates,
                                                         copernicus_albedo_variables,
                                                         albedo_satellite, find_albedo_variable,
                                                         repack_albedo_pair

#####
##### Dispatch helpers — encapsulate single-level vs pressure-level differences
#####

cds_product(::ERA5Dataset)               = "reanalysis-era5-single-levels"
cds_product(::ERA5PressureLevelsDataset) = "reanalysis-era5-pressure-levels"

cds_varnames(::ERA5Dataset)               = ERA5_dataset_variable_names
cds_varnames(::ERA5PressureLevelsDataset) = ERA5PL_dataset_variable_names

nc_varnames(::ERA5Dataset)               = ERA5_netcdf_variable_names
nc_varnames(::ERA5PressureLevelsDataset) = ERA5PL_netcdf_variable_names

# Coordinate / dimension variables to propagate into each split file
const ERA5_COORD_VARS = Set(["longitude", "latitude",
                              "time", "valid_time",
                              "expver", "number"])

const ERA5PL_COORD_VARS = Set(["longitude", "latitude",
                               "pressure_level", "level",
                               "time", "valid_time",
                               "expver", "number"])

coord_vars(::ERA5Dataset)               = ERA5_COORD_VARS
coord_vars(::ERA5PressureLevelsDataset) = ERA5PL_COORD_VARS

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
##### ZIP detection — CDS returns a ZIP when mixing step types (inst/accum/avg)
#####

const ZIP_MAGIC = UInt8[0x50, 0x4b, 0x03, 0x04]

function is_zip(path)
    open(path, "r") do io
        magic = read(io, 4)
        return length(magic) ≥ 4 && magic == ZIP_MAGIC
    end
end

"""
    foreach_nc(f, download_path, cleanup_dir)

If `download_path` is a ZIP archive (as CDS returns when mixing variable step types),
extract all NetCDF files and call `f(nc_path)` on each. Otherwise call `f` directly
on `download_path`.
"""
function foreach_nc(f, download_path, cleanup_dir)
    if is_zip(download_path)
        tmp_dir = mktempdir(cleanup_dir)
        run(`unzip -qo $download_path -d $tmp_dir`)
        nc_files = filter(p -> endswith(p, ".nc"), readdir(tmp_dir; join=true))
        for nc_file in nc_files
            f(nc_file)
        end
        rm(tmp_dir; recursive=true, force=true)
    else
        f(download_path)
    end
end

#####
##### Retry wrapper — the CDS/EWDS gateway intermittently answers valid requests
##### with a transient error (e.g. 502 Bad Gateway); retry with backoff instead of
##### failing on the first hiccup. Every `CDSAPI.retrieve` call site below goes
##### through this one wrapper.
#####

function retrieve_with_retries(product, request, path; max_retries = 3)
    for attempt in 1:max_retries
        try
            return CDSAPI.retrieve(product, request, path)
        catch e
            attempt < max_retries || rethrow(e)
            @warn "CDS retrieve attempt $attempt/$max_retries failed for $product; retrying..." exception=(e, catch_backtrace())
            sleep(5.0 * attempt)
        end
    end
end

#####
##### Single-date download
#####

"""
    download(meta::ERA5Metadatum; skip_existing=true)

Download ERA5 data for a single date/time using the CDSAPI package.

# Keyword Arguments
- `skip_existing`: Skip download if the file already exists (default: `true`).

# Environment Setup
Before downloading, you must:
1. Create an account at https://cds.climate.copernicus.eu/
2. Accept the Terms of Use for the ERA5 dataset on the dataset page
3. Set up your API credentials in `~/.cdsapirc`

See https://cds.climate.copernicus.eu/how-to-api for details.
"""
function Downloads.download(meta::ERA5Metadatum; skip_existing=true)
    output_path = metadata_path(meta)

    # Skip download if file already exists
    skip_existing && isfile(output_path) && return output_path

    mkpath(dirname(output_path))

    request = build_era5_request(meta.name, meta.dataset, meta.dates; region=meta.region)

    @root retrieve_with_retries(cds_product(meta.dataset), request, output_path)

    return output_path
end

#####
##### Multi-date download — batches by calendar month, capped by CDS cost
#####
##### CDS interprets `year`/`month`/`day`/`time` as a Cartesian product, so a
##### single request can cover many days × hours per call as long as `year`
##### and `month` stay singletons. The remaining constraint is the per-request
##### cost limit: roughly `num_vars × num_pressure_levels × num_datetimes`
##### must stay under ~7500 for pressure-level data (CDS returns HTTP 403
##### "cost limits exceeded" otherwise). We pick a conservative cap and split
##### each month into smaller contiguous chunks when needed.
#####

const CDS_MAX_FIELDS_PER_REQUEST = 5000

function Downloads.download(metadata::ERA5Metadata; skip_existing=true, cleanup=true)
    dates = metadata.dates isa AbstractVector ? metadata.dates : [metadata.dates]
    batches = batch_datetimes_for_cds(dates, metadata.dataset, 1)

    paths = String[]
    for batch in batches
        path = download_era5_month(metadata.name, metadata.dataset, batch;
                                   region = metadata.region,
                                   dir = metadata.dir,
                                   skip_existing, cleanup)
        append!(paths, path)
    end

    return paths
end

"""
    group_by_calendar_month(datetimes)

Group an iterable of `DateTime`s by `(year, month)`. Returns a `Dict` whose
keys are `Tuple{Int, Int}` `(year, month)` pairs and whose values are the
datetimes that fall in that month. The `00:00` instant of a day belongs to
that day (not the previous one).
"""
function group_by_calendar_month(datetimes)
    keys = unique([(Dates.year(dt), Dates.month(dt)) for dt in datetimes])
    return Dict(k => filter(dt -> (Dates.year(dt), Dates.month(dt)) == k, datetimes)
                for k in keys)
end

"""
    max_dts_per_cds_request(dataset, num_vars; max_fields=$(CDS_MAX_FIELDS_PER_REQUEST))

Maximum number of datetimes that can share a single CDS request before the
per-request cost limit is hit. Pressure-level datasets multiply by the number
of selected pressure levels; single-level datasets count as one level. Falls
back to `1` when a single datetime already exceeds the cap (which would force
the caller's single-datetime download path).
"""
function max_dts_per_cds_request(dataset, num_vars; max_fields=CDS_MAX_FIELDS_PER_REQUEST)
    levels = dataset isa ERA5PressureLevelsDataset ? length(dataset.pressure_levels) : 1
    return max(1, fld(max_fields, num_vars * levels))
end

"""
    batch_datetimes_for_cds(datetimes, dataset, num_vars; max_fields=$(CDS_MAX_FIELDS_PER_REQUEST))

Split `datetimes` into contiguous batches that each fit in one CDS request:
each batch shares a `(year, month)` and contains at most
`max_dts_per_cds_request(dataset, num_vars; max_fields)` datetimes. Batches
are returned sorted by their first datetime so the caller can iterate in
chronological order.
"""
function batch_datetimes_for_cds(datetimes, dataset, num_vars;
                                  max_fields=CDS_MAX_FIELDS_PER_REQUEST)
    monthly = group_by_calendar_month(datetimes)
    max_dts = max_dts_per_cds_request(dataset, num_vars; max_fields)

    batches = Vector{Dates.DateTime}[]
    for key in sort(collect(keys(monthly)))
        sorted = sort(unique(monthly[key]))
        for i in 1:max_dts:length(sorted)
            push!(batches, sorted[i:min(i + max_dts - 1, end)])
        end
    end
    return batches
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
  triples consumed by `split_era5_nc_multistep`.
"""
function plan_era5_month(name, dataset, dates; region, dir, skip_existing)
    meta_filename = NumericalEarth.DataWrangling.metadata_filename

    dt_path_pairs = [(dt, joinpath(dir, meta_filename(dataset, name, dt, region)))
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
    nc_triples = [(nc_varname, dt, path) for (dt, path) in pending]

    return (; dt_path_pairs, pending, request, tmp_path, nc_triples)
end

function download_era5_month(name, dataset, dates;
                             region, dir, skip_existing, cleanup)

    plan = plan_era5_month(name, dataset, dates; region, dir, skip_existing)
    isempty(plan.pending) && return map(dt_path -> dt_path[2], plan.dt_path_pairs)

    mkpath(dir)
    time_dimnames = Set(["time", "valid_time"])

    @root begin
        retrieve_with_retries(cds_product(dataset), plan.request, plan.tmp_path)
        foreach_nc(plan.tmp_path, dir) do nc_path
            split_era5_nc_multistep(nc_path, plan.nc_triples, coord_vars(dataset), time_dimnames)
        end
        cleanup && rm(plan.tmp_path; force=true)
    end

    return map(dt_path -> dt_path[2], plan.dt_path_pairs)
end

#####
##### Multi-variable ERA5 pressure-level download
#####

"""
    Downloads.download(names::Vector{Symbol}, metadata::ERA5PressureMetadata; kwargs...)

Download multiple ERA5 pressure-level variables for each date in `metadata`.
"""
function Downloads.download(names::Vector{Symbol}, metadata::ERA5PressureMetadata; kwargs...)
    paths = String[]
    for metadatum in metadata
        append!(paths, Downloads.download(names, metadatum; kwargs...))
    end
    return paths
end

"""
    Downloads.download(mset::MetadataSet{<:ERA5PressureLevelsDataset}; kwargs...)

Route a `MetadataSet` of ERA5 pressure-level variables through the existing
multi-variable batched CDS path, instead of falling back to per-variable
requests via the default `Downloads.download(::MetadataSet)`. Each calendar day's
variables are bundled into one CDS API request.
"""
function Downloads.download(mset::MetadataSet{<:ERA5PressureLevelsDataset}; kwargs...)
    names = collect(getfield(mset, :names))

    # Build a representative ERA5PressureMetadata at the shared scope. The
    # batched method only consults its `dataset`, `dates`, `region`, `dir` —
    # the per-variable filename(s) are recomputed internally per (name, date).
    representative = NumericalEarth.DataWrangling.Metadata(
        first(names),
        getfield(mset, :dataset),
        getfield(mset, :dates),
        getfield(mset, :region),
        getfield(mset, :dir),
        nothing,
    )

    return Downloads.download(names, representative; kwargs...)
end

"""
    Downloads.download(names::Vector{Symbol}, meta::ERA5PressureMetadatum; skip_existing=true)

Download multiple ERA5 pressure-level variables for a single date in one CDS API request.
The multi-variable NetCDF is split into individual per-variable files.
"""
function Downloads.download(names::Vector{Symbol}, meta::ERA5PressureMetadatum; skip_existing=true)
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

    pending_names = [name for (name, _) in pending]
    request = build_era5_request(pending_names, meta.dataset, meta.dates; region=meta.region)

    date  = meta.dates
    year  = string(Dates.year(date))
    month = lpad(string(Dates.month(date)), 2, '0')
    day   = lpad(string(Dates.day(date)),   2, '0')
    hour  = lpad(string(Dates.hour(date)),  2, '0') * ":00"

    mkpath(meta.dir)
    tmp_path = joinpath(meta.dir, "_tmp_multi_$(year)$(month)$(day)T$(hour[1:2]).nc")

    nc_name_path_pairs = [(nc_varnames(meta.dataset)[name], path) for (name, path) in pending]

    @root begin
        retrieve_with_retries(cds_product(meta.dataset), request, tmp_path)
        foreach_nc(tmp_path, meta.dir) do nc_path
            split_era5_nc(nc_path, nc_name_path_pairs, coord_vars(meta.dataset))
        end
        rm(tmp_path; force=true)
    end

    return map(name_path -> name_path[2], name_path_pairs)
end

"""
    Downloads.download(names, dataset::ERA5Dataset, datetime; ...)

Download one or more ERA5 variables at a single datetime.
"""
function Downloads.download(names::Vector{Symbol}, dataset::ERA5Dataset, datetime;
                            region = nothing,
                            dir = default_download_directory(dataset))
    meta = Metadatum(first(names); dataset, date=datetime, region, dir)
    return Downloads.download(names, meta)
end

function Downloads.download(name::Symbol, dataset::ERA5Dataset, datetime;
                            region = nothing,
                            dir = default_download_directory(dataset))
    return Downloads.download([name], dataset, datetime; region, dir)
end

"""
    Downloads.download(names, dataset::ERA5Dataset, datetimes::AbstractVector; ...)

Download one or more ERA5 variables for multiple datetimes, batching by calendar day.
"""
function Downloads.download(names::Vector{Symbol},
                            dataset::ERA5Dataset,
                            datetimes::AbstractVector;
                            region = nothing,
                            dir = default_download_directory(dataset),
                            skip_existing = true,
                            cleanup = true)

    batches = batch_datetimes_for_cds(datetimes, dataset, length(names))

    paths = String[]
    for batch in batches
        path = download_era5_multivar_month(names, dataset, batch;
                                            region, dir, skip_existing, cleanup)
        append!(paths, path)
    end

    return paths
end

function Downloads.download(name::Symbol,
                            dataset::ERA5Dataset,
                            datetimes::AbstractVector;
                            region = nothing,
                            dir = default_download_directory(dataset),
                            skip_existing = true,
                            cleanup = true)
    return Downloads.download([name], dataset, datetimes; region, dir, skip_existing, cleanup)
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
  triples consumed by `split_era5_nc_multistep`.
"""
function plan_era5_multivar_month(names, dataset, dates; region, dir, skip_existing)
    meta_filename = NumericalEarth.DataWrangling.metadata_filename

    name_dt_paths = [(name, dt, joinpath(dir, meta_filename(dataset, name, dt, region)))
                     for name in names for dt in dates]

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
    nc_triples = [(nc_varnames(dataset)[name], dt, path)
                  for (name, dt, path) in pending]

    return (; name_dt_paths, pending, request, tmp_path, nc_triples)
end

function download_era5_multivar_month(names, dataset, dates;
                                      region, dir, skip_existing, cleanup)

    plan = plan_era5_multivar_month(names, dataset, dates; region, dir, skip_existing)
    isempty(plan.pending) && return map(name_dt_path -> name_dt_path[3], plan.name_dt_paths)

    mkpath(dir)
    time_dimnames = Set(["time", "valid_time"])

    @root begin
        retrieve_with_retries(cds_product(dataset), plan.request, plan.tmp_path)
        foreach_nc(plan.tmp_path, dir) do nc_path
            split_era5_nc_multistep(nc_path, plan.nc_triples, coord_vars(dataset), time_dimnames)
        end
        cleanup && rm(plan.tmp_path; force=true)
    end

    return map(name_dt_path -> name_dt_path[3], plan.name_dt_paths)
end

#####
##### NetCDF splitting utilities
#####

"""
    split_era5_nc(src_path, nc_name_path_pairs, coord_vars)

Split a multi-variable NetCDF into individual per-variable files (single time step).
"""
function split_era5_nc(src_path, nc_name_path_pairs, coord_vars)
    NCDatasets.Dataset(src_path, "r") do src
        src_varnames = Set(keys(src))
        for (nc_varname, dst_path) in nc_name_path_pairs
            nc_varname in src_varnames || continue
            NCDatasets.Dataset(dst_path, "c") do dst
                unlimited = NCDatasets.unlimited(src)
                for (dname, dlen) in src.dim
                    NCDatasets.defDim(dst, dname, dname in unlimited ? Inf : dlen)
                end

                for (k, v) in src.attrib
                    dst.attrib[k] = v
                end

                for (vname, var) in src
                    (vname in coord_vars || vname == nc_varname) || continue
                    ncvar_copy!(dst, var, vname)
                end
            end
        end
    end
end

"""
    split_era5_nc_multistep(src_path, triples, coord_vars, time_dimnames)

Split a multi-timestep NetCDF into individual per-variable, per-timestep files.
`triples` is a vector of `(nc_varname, datetime, dst_path)`.

Each `datetime`'s timestep is located by matching it against `src`'s time coordinate, NOT by its
position in the request. CDS expands `day`/`time` into a Cartesian product, so a request whose
datetimes span more than one day with differing hours (e.g. a window crossing midnight) comes back
with extra, sorted timesteps that no longer line up positionally with the requested datetimes.
"""
function split_era5_nc_multistep(src_path, nc_varname_datetime_path_triples, coord_vars, time_dimnames)
    NCDatasets.Dataset(src_path, "r") do src
        src_varnames = Set(keys(src))
        unlimited = NCDatasets.unlimited(src)

        # Index this file's timesteps by their valid time (see the note above).
        time_coord = "valid_time" in src_varnames ? "valid_time" :
                     "time"       in src_varnames ? "time"       :
                     error("split_era5_nc_multistep: no time coordinate variable in $src_path")
        tidx_of = Dict(t => i for (i, t) in enumerate(src[time_coord][:]))

        for (nc_varname, datetime, dst_path) in nc_varname_datetime_path_triples
            nc_varname in src_varnames || continue
            haskey(tidx_of, datetime) ||
                error("split_era5_nc_multistep: $datetime absent from $src_path")
            tidx = tidx_of[datetime]
            NCDatasets.Dataset(dst_path, "c") do dst
                for (dname, dlen) in src.dim
                    out_len = dname in time_dimnames ? 1 :
                              dname in unlimited     ? Inf : dlen
                    NCDatasets.defDim(dst, dname, out_len)
                end

                for (k, v) in src.attrib
                    dst.attrib[k] = v
                end

                for (vname, var) in src
                    (vname in coord_vars || vname == nc_varname) || continue
                    ncvar_copy_tslice!(dst, var, vname, tidx, time_dimnames)
                end
            end
        end
    end
end

function ncvar_copy!(dst, src_var, vname)
    dims     = NCDatasets.dimnames(src_var)
    T        = eltype(src_var.var)
    attribs  = src_var.attrib
    fill_val = haskey(attribs, "_FillValue") ? attribs["_FillValue"] : nothing

    dst_var = isnothing(fill_val) ?
        NCDatasets.defVar(dst, vname, T, dims) :
        NCDatasets.defVar(dst, vname, T, dims; fillvalue=fill_val)

    for (k, v) in attribs
        k == "_FillValue" && continue
        dst_var.attrib[k] = v
    end

    dst_var.var[:] = src_var.var[:]
    return nothing
end

function ncvar_copy_tslice!(dst, src_var, vname, tidx, time_dimnames)
    dims     = NCDatasets.dimnames(src_var)
    T        = eltype(src_var.var)
    attribs  = src_var.attrib
    fill_val = haskey(attribs, "_FillValue") ? attribs["_FillValue"] : nothing

    dst_var = isnothing(fill_val) ?
        NCDatasets.defVar(dst, vname, T, dims) :
        NCDatasets.defVar(dst, vname, T, dims; fillvalue=fill_val)

    for (k, v) in attribs
        k == "_FillValue" && continue
        dst_var.attrib[k] = v
    end

    has_time = any(d -> d in time_dimnames, dims)
    if has_time
        idx = ntuple(ndims(src_var.var)) do i
            dims[i] in time_dimnames ? (tidx:tidx) : Colon()
        end
        dst_var.var[:] = src_var.var[idx...]
    else
        dst_var.var[:] = src_var.var[:]
    end

    return nothing
end

#####
##### Area/bounding box utilities
#####

build_era5_area(::Nothing) = nothing

const BBOX = NumericalEarth.DataWrangling.BoundingBox
const COL  = NumericalEarth.DataWrangling.Column
const LIN  = NumericalEarth.DataWrangling.Linear
const NR   = NumericalEarth.DataWrangling.Nearest

# Columns and unbounded regions: the area is a pure function of the region.
era5_request_area(region, dataset, name) = build_era5_area(region)

# Bounding box: the native grid is built by center-bracketing `restrict`, which
# can reach one cell past a boundary-aligned edge. Fetch two native cells of
# margin (in the bbox's own longitude convention) so the downloaded file always
# covers the grid the data is interpolated onto — otherwise downscaling leaves
# NaNs at the domain edges. Over-fetching is harmless: `restrict` selects the
# exact cells from the larger file.
function era5_request_area(bbox::BBOX, dataset, name)
    (isnothing(bbox.longitude) || isnothing(bbox.latitude)) && return nothing
    Nx, Ny, _ = size(dataset, name)
    Δλ = 360 / Nx
    Δφ = 180 / Ny
    lon = bbox.longitude
    lat = bbox.latitude
    padded = BBOX(longitude = (lon[1] - 2Δλ, lon[2] + 2Δλ),
                  latitude  = (max(lat[1] - 2Δφ, -90), min(lat[2] + 2Δφ, 90)))
    return build_era5_area(padded)
end

function build_era5_area(bbox::BBOX)
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
function build_era5_area(col::COL{<:Any, <:Any, <:Any, <:NR})
    lon, lat = col.longitude, col.latitude
    ε = 1e-3
    return [lat + ε, lon - ε, lat - ε, lon + ε]  # [N, W, S, E]
end

# Column with Linear interpolation: pad by slightly more than ERA5's native
# 0.25° spacing so the file contains the 2x2 stencil bilinear interp needs.
function build_era5_area(col::COL{<:Any, <:Any, <:Any, <:LIN})
    lon, lat = col.longitude, col.latitude
    ε = 0.3
    return [lat + ε, lon - ε, lat - ε, lon + ε]
end

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

#####
##### Copernicus land surface albedo (C3S `satellite-albedo` catalogue entry)
#####
##### One CDS request per calendar month fetches the black-sky (`albb_dh`) and
##### white-sky (`albb_bh`) products for all of that month's dekads; each dekad's
##### pair is repacked into one compact local file by the CopernicusLandAlbedo module.
#####

const ALBEDO_CDS_PRODUCT = "satellite-albedo"

const AlbedoMetadata = NumericalEarth.DataWrangling.Metadata{<:CopernicusAlbedo}

"""
    build_albedo_request(name, dates) -> Dict{String, Any}

Construct the CDS request for the 1 km v2 black-sky/white-sky albedo pair covering
`dates`, which must share a `(year, month)` (CDS interprets `year`/`month`/
`nominal_day` as a Cartesian product, and the valid nominal days differ by month).
"""
function build_albedo_request(name, dates)
    dts = dates isa AbstractVector ? dates : [dates]

    length(unique((Dates.year(dt), Dates.month(dt)) for dt in dts)) == 1 ||
        error("build_albedo_request expects dates within one calendar month; got $(dts).")

    years  = unique(string.(Dates.year.(dts)))
    months = unique(lpad.(string.(Dates.month.(dts)), 2, '0'))
    days   = unique(lpad.(string.(Dates.day.(dts)), 2, '0'))
    satellites = unique([albedo_satellite(dt) for dt in dts])

    return Dict{String, Any}(
        "variable"              => collect(albedo_cds_request_variables[name]),
        "satellite"             => satellites,
        "sensor"                => ["vgt"],
        "product_version"       => ["v2"],
        "horizontal_resolution" => ["1km"],
        "year"                  => years,
        "month"                 => months,
        "nominal_day"           => days,
    )
end

# The delivery is either a ZIP of per-variable NetCDF files or a single NetCDF.
function extract_albedo_files(download_path, extraction_dir)
    if is_zip(download_path)
        run(`unzip -qo $download_path -d $extraction_dir`)
    else
        cp(download_path, joinpath(extraction_dir, "albedo.nc"); force=true)
    end
    return filter(p -> endswith(p, ".nc"), readdir(extraction_dir; join=true))
end

# The dekad a delivered file belongs to, from its time coordinate (authoritative)
# or the timestamp in its filename.
function albedo_file_date(path)
    date = NCDatasets.Dataset(path) do ds
        haskey(ds, "time") || return nothing
        t = ds["time"][1]
        return Dates.DateTime(Dates.year(t), Dates.month(t), Dates.day(t))
    end
    isnothing(date) || return date

    stamp = match(r"_(\d{8})\d{0,6}_", basename(path))
    isnothing(stamp) &&
        error("Cannot determine the date of the delivered albedo file $(basename(path)).")
    return Dates.DateTime(stamp[1], Dates.dateformat"yyyymmdd")
end

function repack_albedo_batch(nc_files, batch, path_of, destination_names, expected_size)
    members = Dict{Dates.DateTime, Dict{Symbol, Tuple{String, String}}}()
    for path in nc_files
        date = albedo_file_date(path)
        entry = get!(members, date, Dict{Symbol, Tuple{String, String}}())
        blacksky_name = find_albedo_variable(path, albedo_source_variable_candidates.blacksky)
        whitesky_name = find_albedo_variable(path, albedo_source_variable_candidates.whitesky)
        isnothing(blacksky_name) || (entry[:blacksky] = (path, blacksky_name))
        isnothing(whitesky_name) || (entry[:whitesky] = (path, whitesky_name))
    end

    for date in batch
        entry = get(members, date, nothing)
        if isnothing(entry) || !haskey(entry, :blacksky) || !haskey(entry, :whitesky)
            error("The CDS delivery is missing the black-sky/white-sky albedo pair for $date; ",
                  "it contained dates $(sort!(collect(keys(members)))).")
        end
        repack_albedo_pair(entry[:blacksky], entry[:whitesky], destination_names,
                           path_of[date], expected_size)
    end
    return nothing
end

"""
    download(metadata::Metadata{<:CopernicusAlbedo}; skip_existing=true, cleanup=true)

Download the dekadal black-sky and white-sky broadband albedo for every date of
`metadata` from the C3S `satellite-albedo` catalogue entry, one CDS request per
calendar month, and repack each dekad's pair into a single compact local NetCDF.
Requires `~/.cdsapirc` credentials and acceptance of the Copernicus Global Land
product licence on the CDS portal.
"""
function Downloads.download(metadata::AlbedoMetadata; skip_existing=true, cleanup=true)
    meta_filename = NumericalEarth.DataWrangling.metadata_filename
    dates = metadata.dates isa AbstractVector ? metadata.dates : [metadata.dates]
    dir = metadata.dir
    mkpath(dir)

    dt_path_pairs = [(dt, joinpath(dir, meta_filename(metadata.dataset, metadata.name, dt, metadata.region)))
                     for dt in dates]
    pending = skip_existing ? filter(dt_path -> !isfile(dt_path[2]), dt_path_pairs) : dt_path_pairs
    isempty(pending) && return metadata_path(metadata)

    expected_size = size(metadata.dataset, metadata.name)[1:2]
    destination_names = copernicus_albedo_variables[metadata.name]
    path_of = Dict(dt => path for (dt, path) in pending)
    monthly = group_by_calendar_month([dt for (dt, _) in pending])

    @root for key in sort(collect(keys(monthly)))
        batch = sort(unique(monthly[key]))
        request = build_albedo_request(metadata.name, batch)
        tmp_download = joinpath(dir, "_tmp_albedo_$(key[1])$(lpad(key[2], 2, '0')).download")
        extraction_dir = mktempdir(dir)
        try
            retrieve_with_retries(ALBEDO_CDS_PRODUCT, request, tmp_download)
            nc_files = extract_albedo_files(tmp_download, extraction_dir)
            repack_albedo_batch(nc_files, batch, path_of, destination_names, expected_size)
        finally
            # Gate both on `cleanup` so `cleanup=false` keeps the raw download and the
            # extracted NetCDFs for debugging.
            cleanup && rm(tmp_download; force=true)
            cleanup && rm(extraction_dir; recursive=true, force=true)
        end
    end

    return metadata_path(metadata)
end

end # module NumericalEarthCDSAPIExt
