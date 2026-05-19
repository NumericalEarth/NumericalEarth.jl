module NumericalEarthCDSAPIExt

using CDSAPI: CDSAPI

using Dates: Dates
using Oceananigans: Oceananigans
using Oceananigans.DistributedComputations: @root

using NCDatasets: NCDatasets, name, path

using NumericalEarth: NumericalEarth
using NumericalEarth.DataWrangling: Metadatum, default_download_directory, metadata_path, download_dataset
using NumericalEarth.DataWrangling.ERA5: ERA5Dataset, ERA5Metadata, ERA5Metadatum,
                                         ERA5_dataset_variable_names, ERA5_netcdf_variable_names,
                                         ERA5PressureLevelsDataset,
                                         ERA5PressureMetadata, ERA5PressureMetadatum,
                                         ERA5PL_dataset_variable_names, ERA5PL_netcdf_variable_names

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

    area = build_era5_area(region)
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
        return length(magic) >= 4 && magic == ZIP_MAGIC
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
##### Single-date download
#####

"""
    download_dataset(meta::ERA5Metadatum; skip_existing=true)

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
function NumericalEarth.DataWrangling.download_dataset(meta::ERA5Metadatum; skip_existing=true)
    output_path = metadata_path(meta)

    # Skip download if file already exists
    skip_existing && isfile(output_path) && return output_path

    mkpath(dirname(output_path))

    request = build_era5_request(meta.name, meta.dataset, meta.dates; region=meta.region)

    @root CDSAPI.retrieve(cds_product(meta.dataset), request, output_path)

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

function NumericalEarth.DataWrangling.download_dataset(metadata::ERA5Metadata; skip_existing=true, cleanup=true)
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
    dt_to_tidx = Dict(dt => i for (i, dt) in enumerate(sorted_dts))

    request = build_era5_request(name, dataset, sorted_dts; region)

    dt0   = first(sorted_dts)
    year  = string(Dates.year(dt0))
    month = lpad(string(Dates.month(dt0)), 2, '0')
    day   = lpad(string(Dates.day(dt0)),   2, '0')

    tmp_path   = joinpath(dir, "_tmp_$(year)$(month)$(day).nc")
    nc_varname = nc_varnames(dataset)[name]
    nc_triples = [(nc_varname, dt_to_tidx[dt], path) for (dt, path) in pending]

    return (; dt_path_pairs, pending, request, tmp_path, nc_triples)
end

function download_era5_month(name, dataset, dates;
                             region, dir, skip_existing, cleanup)

    plan = plan_era5_month(name, dataset, dates; region, dir, skip_existing)
    isempty(plan.pending) && return map(dt_path -> dt_path[2], plan.dt_path_pairs)

    mkpath(dir)
    time_dimnames = Set(["time", "valid_time"])

    @root begin
        CDSAPI.retrieve(cds_product(dataset), plan.request, plan.tmp_path)
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
    download_dataset(names::Vector{Symbol}, metadata::ERA5PressureMetadata; kwargs...)

Download multiple ERA5 pressure-level variables for each date in `metadata`.
"""
function NumericalEarth.DataWrangling.download_dataset(names::Vector{Symbol}, metadata::ERA5PressureMetadata; kwargs...)
    paths = String[]
    for metadatum in metadata
        append!(paths, download_dataset(names, metadatum; kwargs...))
    end
    return paths
end

"""
    download_dataset(names::Vector{Symbol}, meta::ERA5PressureMetadatum; skip_existing=true)

Download multiple ERA5 pressure-level variables for a single date in one CDS API request.
The multi-variable NetCDF is split into individual per-variable files.
"""
function NumericalEarth.DataWrangling.download_dataset(names::Vector{Symbol}, meta::ERA5PressureMetadatum; skip_existing=true)
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
        CDSAPI.retrieve(cds_product(meta.dataset), request, tmp_path)
        foreach_nc(tmp_path, meta.dir) do nc_path
            split_era5_nc(nc_path, nc_name_path_pairs, coord_vars(meta.dataset))
        end
        rm(tmp_path; force=true)
    end

    return map(name_path -> name_path[2], name_path_pairs)
end

"""
    download_dataset(names, dataset::ERA5Dataset, datetime; ...)

Download one or more ERA5 variables at a single datetime.
"""
function NumericalEarth.DataWrangling.download_dataset(names::Vector{Symbol}, dataset::ERA5Dataset, datetime;
                                                       region = nothing,
                                                       dir = default_download_directory(dataset))
    meta = Metadatum(first(names); dataset, date=datetime, region, dir)
    return download_dataset(names, meta)
end

function NumericalEarth.DataWrangling.download_dataset(name::Symbol, dataset::ERA5Dataset, datetime;
                                                       region = nothing,
                                                       dir = default_download_directory(dataset))
    return download_dataset([name], dataset, datetime; region, dir)
end

"""
    download_dataset(names, dataset::ERA5Dataset, datetimes::AbstractVector; ...)

Download one or more ERA5 variables for multiple datetimes, batching by calendar day.
"""
function NumericalEarth.DataWrangling.download_dataset(names::Vector{Symbol},
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

function NumericalEarth.DataWrangling.download_dataset(name::Symbol,
                                                       dataset::ERA5Dataset,
                                                       datetimes::AbstractVector;
                                                       region = nothing,
                                                       dir = default_download_directory(dataset),
                                                       skip_existing = true,
                                                       cleanup = true)
    return download_dataset([name], dataset, datetimes; region, dir, skip_existing, cleanup)
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
    dt_to_tidx    = Dict(dt => i for (i, dt) in enumerate(sorted_dts))

    request = build_era5_request(pending_names, dataset, sorted_dts; region)

    dt0   = first(sorted_dts)
    year  = string(Dates.year(dt0))
    month = lpad(string(Dates.month(dt0)), 2, '0')
    day   = lpad(string(Dates.day(dt0)),   2, '0')

    tmp_path   = joinpath(dir, "_tmp_multi_$(year)$(month)$(day).nc")
    nc_triples = [(nc_varnames(dataset)[name], dt_to_tidx[dt], path)
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
        CDSAPI.retrieve(cds_product(dataset), plan.request, plan.tmp_path)
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
`triples` is a vector of `(nc_varname, time_index, dst_path)`.
"""
function split_era5_nc_multistep(src_path, nc_varname_tidx_path_triples, coord_vars, time_dimnames)
    NCDatasets.Dataset(src_path, "r") do src
        src_varnames = Set(keys(src))
        unlimited = NCDatasets.unlimited(src)

        for (nc_varname, tidx, dst_path) in nc_varname_tidx_path_triples
            nc_varname in src_varnames || continue
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

end # module NumericalEarthCDSAPIExt
