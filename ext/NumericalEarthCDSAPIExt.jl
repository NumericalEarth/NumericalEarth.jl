module NumericalEarthCDSAPIExt

using NumericalEarth
using CDSAPI
using NCDatasets

using Oceananigans
using Oceananigans.DistributedComputations: @root

using Dates
using NumericalEarth.DataWrangling.ERA5: ERA5Dataset, ERA5Metadata, ERA5Metadatum,
                                         ERA5_dataset_variable_names, ERA5_netcdf_variable_names
using NumericalEarth.DataWrangling.ERA5: ERA5PressureLevelsDataset,
                                         ERA5PressureMetadata, ERA5PressureMetadatum,
                                         ERA5PL_dataset_variable_names, ERA5PL_netcdf_variable_names

import NumericalEarth.DataWrangling: download_dataset

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
function download_dataset(meta::ERA5Metadatum; skip_existing=true)
    output_path = NumericalEarth.DataWrangling.metadata_path(meta)

    # Skip download if file already exists
    skip_existing && isfile(output_path) && return output_path

    mkpath(dirname(output_path))

    date = meta.dates
    request = Dict(
        "product_type"    => ["reanalysis"],
        "variable"        => [cds_varnames(meta.dataset)[meta.name]],
        "year"            => [string(Dates.year(date))],
        "month"           => [lpad(string(Dates.month(date)), 2, '0')],
        "day"             => [lpad(string(Dates.day(date)), 2, '0')],
        "time"            => [lpad(string(Dates.hour(date)), 2, '0') * ":00"],
        "data_format"     => "netcdf",
        "download_format" => "unarchived",
    )

    extra_request_keys!(request, meta.dataset)
    area = build_era5_area(meta.region)
    isnothing(area) || (request["area"] = area)

    @root CDSAPI.retrieve(cds_product(meta.dataset), request, output_path)

    return output_path
end

#####
##### Multi-date download — batches by calendar day
#####

function download_dataset(metadata::ERA5Metadata; skip_existing=true, cleanup=true)
    dates = metadata.dates isa AbstractVector ? metadata.dates : [metadata.dates]
    grouped = _group_by_calendar_day(dates)

    for day in sort(collect(keys(grouped)))
        download_era5_day(metadata.name, metadata.dataset, grouped[day];
                           region = metadata.region,
                           dir = metadata.dir,
                           skip_existing, cleanup)
    end
end

"""
    _group_by_calendar_day(datetimes)

Group an iterable of `DateTime`s by calendar day. Returns a `Dict{Date, Vector}`
where each value is the subset of `datetimes` whose calendar day equals the key.
The `00:00` instant of a day belongs to that day (not the previous one).
"""
function _group_by_calendar_day(datetimes)
    return Dict(d => filter(dt -> Dates.Date(dt) == d, datetimes)
                for d in unique(Dates.Date.(datetimes)))
end

function download_era5_day(name, dataset, day_dates;
                            region, dir, skip_existing, cleanup)

    MDatum    = NumericalEarth.DataWrangling.Metadatum
    meta_path = NumericalEarth.DataWrangling.metadata_path

    all_pairs = [(dt, meta_path(MDatum(name; dataset, date=dt, region, dir)))
                 for dt in day_dates]

    pending = skip_existing ? filter(((_, p),) -> !isfile(p), all_pairs) : all_pairs
    isempty(pending) && return nothing

    sorted_dts = sort(unique([dt for (dt, _) in pending]))
    hours_str  = [lpad(string(Dates.hour(dt)), 2, '0') * ":00" for dt in sorted_dts]
    dt_to_tidx = Dict(dt => i for (i, dt) in enumerate(sorted_dts))

    dt    = first(sorted_dts)
    year  = string(Dates.year(dt))
    month = lpad(string(Dates.month(dt)), 2, '0')
    day   = lpad(string(Dates.day(dt)),   2, '0')

    request = Dict(
        "product_type"    => ["reanalysis"],
        "variable"        => [cds_varnames(dataset)[name]],
        "year"            => [year],
        "month"           => [month],
        "day"             => [day],
        "time"            => hours_str,
        "data_format"     => "netcdf",
        "download_format" => "unarchived",
    )

    extra_request_keys!(request, dataset)
    area = build_era5_area(region)
    isnothing(area) || (request["area"] = area)

    mkpath(dir)
    tmp_path   = joinpath(dir, "_tmp_$(year)$(month)$(day).nc")
    nc_varname = nc_varnames(dataset)[name]
    nc_triples = [(nc_varname, dt_to_tidx[dt], path) for (dt, path) in pending]

    time_dimnames = Set(["time", "valid_time"])

    @root begin
        CDSAPI.retrieve(cds_product(dataset), request, tmp_path)
        foreach_nc(tmp_path, dir) do nc_path
            split_era5_nc_multistep(nc_path, nc_triples, coord_vars(dataset), time_dimnames)
        end
        cleanup && rm(tmp_path; force=true)
    end

    return nothing
end

#####
##### Multi-variable ERA5 pressure-level download
#####

"""
    download_dataset(names::Vector{Symbol}, metadata::ERA5PressureMetadata; kwargs...)

Download multiple ERA5 pressure-level variables for each date in `metadata`.
"""
function download_dataset(names::Vector{Symbol}, metadata::ERA5PressureMetadata; kwargs...)
    for metadatum in metadata
        download_dataset(names, metadatum; kwargs...)
    end
    return nothing
end

"""
    download_dataset(names::Vector{Symbol}, meta::ERA5PressureMetadatum; skip_existing=true)

Download multiple ERA5 pressure-level variables for a single date in one CDS API request.
The multi-variable NetCDF is split into individual per-variable files.
"""
function download_dataset(names::Vector{Symbol}, meta::ERA5PressureMetadatum; skip_existing=true)
    name_path_pairs = []
    for name in names
        metadatum = NumericalEarth.DataWrangling.Metadatum(name;
                                                           dataset      = meta.dataset,
                                                           region = meta.region,
                                                           date         = meta.dates,
                                                           dir          = meta.dir)
        path = NumericalEarth.DataWrangling.metadata_path(metadatum)
        push!(name_path_pairs, (name, path))
    end

    pending = if skip_existing
        [(n, p) for (n, p) in name_path_pairs if !isfile(p)]
    else
        name_path_pairs
    end

    isempty(pending) && return [path for (_, path) in name_path_pairs]

    cds_vars = unique([cds_varnames(meta.dataset)[name] for (name, _) in pending])

    date  = meta.dates
    year  = string(Dates.year(date))
    month = lpad(string(Dates.month(date)), 2, '0')
    day   = lpad(string(Dates.day(date)),   2, '0')
    hour  = lpad(string(Dates.hour(date)),  2, '0') * ":00"

    request = Dict(
        "product_type"    => ["reanalysis"],
        "variable"        => cds_vars,
        "year"            => [year],
        "month"           => [month],
        "day"             => [day],
        "time"            => [hour],
        "data_format"     => "netcdf",
        "download_format" => "unarchived",
    )

    extra_request_keys!(request, meta.dataset)
    area = build_era5_area(meta.region)
    isnothing(area) || (request["area"] = area)

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

    return [path for (_, path) in name_path_pairs]
end

"""
    download_dataset(names, dataset::ERA5Dataset, datetime; ...)

Download one or more ERA5 variables at a single datetime.
"""
function download_dataset(names::Vector{Symbol}, dataset::ERA5Dataset, datetime;
                          region = nothing,
                          dir = NumericalEarth.DataWrangling.default_download_directory(dataset))
    meta = NumericalEarth.DataWrangling.Metadatum(first(names); dataset, date=datetime, region, dir)
    return download_dataset(names, meta)
end

function download_dataset(name::Symbol, dataset::ERA5Dataset, datetime;
                          region = nothing,
                          dir = NumericalEarth.DataWrangling.default_download_directory(dataset))
    return download_dataset([name], dataset, datetime; region, dir)
end

"""
    download_dataset(names, dataset::ERA5Dataset, datetimes::AbstractVector; ...)

Download one or more ERA5 variables for multiple datetimes, batching by calendar day.
"""
function download_dataset(names::Vector{Symbol},
                          dataset::ERA5Dataset,
                          datetimes::AbstractVector;
                          region = nothing,
                          dir = NumericalEarth.DataWrangling.default_download_directory(dataset),
                          skip_existing = true,
                          cleanup = true)

    grouped = _group_by_calendar_day(datetimes)

    for day in sort(collect(keys(grouped)))
        download_era5_multivar_day(names, dataset, grouped[day]; region, dir, skip_existing, cleanup)
    end

    return nothing
end

function download_dataset(name::Symbol,
                          dataset::ERA5Dataset,
                          datetimes::AbstractVector;
                          region = nothing,
                          dir = NumericalEarth.DataWrangling.default_download_directory(dataset),
                          skip_existing = true,
                          cleanup = true)
    return download_dataset([name], dataset, datetimes; region, dir, skip_existing, cleanup)
end

function download_era5_multivar_day(names, dataset, day_dates;
                                     region, dir, skip_existing, cleanup)

    MDatum    = NumericalEarth.DataWrangling.Metadatum
    meta_path = NumericalEarth.DataWrangling.metadata_path

    all_triples = [(name, dt, meta_path(MDatum(name; dataset, date=dt, region, dir)))
                   for name in names for dt in day_dates]

    pending = skip_existing ? filter(((_, _, p),) -> !isfile(p), all_triples) : all_triples
    isempty(pending) && return nothing

    cds_vars   = unique([cds_varnames(dataset)[name] for (name, _, _) in pending])
    sorted_dts = sort(unique([dt for (_, dt, _) in pending]))
    hours_str  = [lpad(string(Dates.hour(dt)), 2, '0') * ":00" for dt in sorted_dts]
    dt_to_tidx = Dict(dt => i for (i, dt) in enumerate(sorted_dts))

    dt    = first(sorted_dts)
    year  = string(Dates.year(dt))
    month = lpad(string(Dates.month(dt)), 2, '0')
    day   = lpad(string(Dates.day(dt)),   2, '0')

    request = Dict(
        "product_type"    => ["reanalysis"],
        "variable"        => cds_vars,
        "year"            => [year],
        "month"           => [month],
        "day"             => [day],
        "time"            => hours_str,
        "data_format"     => "netcdf",
        "download_format" => "unarchived",
    )

    extra_request_keys!(request, dataset)
    area = build_era5_area(region)
    isnothing(area) || (request["area"] = area)

    mkpath(dir)
    tmp_path   = joinpath(dir, "_tmp_multi_$(year)$(month)$(day).nc")
    nc_triples = [(nc_varnames(dataset)[name], dt_to_tidx[dt], path)
                  for (name, dt, path) in pending]

    time_dimnames = Set(["time", "valid_time"])

    @root begin
        CDSAPI.retrieve(cds_product(dataset), request, tmp_path)
        foreach_nc(tmp_path, dir) do nc_path
            split_era5_nc_multistep(nc_path, nc_triples, coord_vars(dataset), time_dimnames)
        end
        cleanup && rm(tmp_path; force=true)
    end

    return nothing
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
