module NumericalEarthCDSAPIExt

using NumericalEarth
using CDSAPI
using NCDatasets

using Oceananigans
using Oceananigans.DistributedComputations: @root

using Dates
using NumericalEarth.DataWrangling.ERA5: ERA5Metadata, ERA5Metadatum, ERA5_dataset_variable_names
using NumericalEarth.DataWrangling.ERA5: ERA5PressureDataset,
                                         ERA5PressureMetadata, ERA5PressureMetadatum,
                                         ERA5PL_dataset_variable_names, ERA5PL_netcdf_variable_names

import NumericalEarth.DataWrangling: download_dataset

# Coordinate / dimension variables to propagate into each split file
const ERA5PL_COORD_VARS = Set(["longitude", "latitude",
                               "pressure_level", "level",
                               "time", "valid_time",
                               "expver",  # experiment version: ==1 for final data, ==5 preliminary 5-day data
                               "number"]) # ensemble member

"""
    download_dataset(metadata::ERA5Metadata; kwargs...)

Download ERA5 data for each date in the metadata, returning paths to downloaded files.
"""
function download_dataset(metadata::ERA5Metadata; kwargs...)
    paths = Array{String}(undef, length(metadata))
    for (m, metadatum) in enumerate(metadata)
        paths[m] = download_dataset(metadatum; kwargs...)
    end
    return paths
end

"""
    download_dataset(meta::ERA5Metadatum; skip_existing=true, kwargs...)

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

    output_directory = meta.dir
    output_filename = NumericalEarth.DataWrangling.metadata_filename(meta)
    output_path = joinpath(output_directory, output_filename)

    # Skip if file already exists
    if skip_existing && isfile(output_path)
        return output_path
    end

    # Ensure output directory exists
    mkpath(output_directory)

    # Get the ERA5 variable name
    variable_name = ERA5_dataset_variable_names[meta.name]

    # Extract date information
    date = meta.dates
    year  = string(Dates.year(date))
    month = lpad(string(Dates.month(date)), 2, '0')
    day   = lpad(string(Dates.day(date)), 2, '0')
    hour  = lpad(string(Dates.hour(date)), 2, '0') * ":00"

    # Build request parameters
    request = Dict(
        "product_type"    => ["reanalysis"],
        "variable"        => [variable_name],
        "year"            => [year],
        "month"           => [month],
        "day"             => [day],
        "time"            => [hour],
        "data_format"     => "netcdf",
        "download_format" => "unarchived",
    )

    # Add area constraint from bounding box
    area = build_era5_area(meta.bounding_box)
    if !isnothing(area)
        request["area"] = area
    end

    # Perform the download using CDSAPI
    @root begin
        CDSAPI.retrieve("reanalysis-era5-single-levels", request, output_path)
    end

    return output_path
end

#####
##### ERA5 pressure-level download
#####

function download_dataset(metadata::ERA5PressureMetadata; kwargs...)
    paths = Array{String}(undef, length(metadata))
    for (m, metadatum) in enumerate(metadata)
        paths[m] = download_dataset(metadatum; kwargs...)
    end
    return paths
end

function download_dataset(meta::ERA5PressureMetadatum; skip_existing=true)
    output_directory = meta.dir
    output_filename  = NumericalEarth.DataWrangling.metadata_filename(meta)
    output_path      = joinpath(output_directory, output_filename)

    skip_existing && isfile(output_path) && return output_path
    mkpath(output_directory)

    variable_name = ERA5PL_dataset_variable_names[meta.name]
    date  = meta.dates
    year  = string(Dates.year(date))
    month = lpad(string(Dates.month(date)), 2, '0')
    day   = lpad(string(Dates.day(date)),   2, '0')
    hour  = lpad(string(Dates.hour(date)),  2, '0') * ":00"

    request = Dict(
        "product_type"    => ["reanalysis"],
        "variable"        => [variable_name],
        "pressure_level"  => [string(p) for p in meta.dataset.levels],
        "year"            => [year],
        "month"           => [month],
        "day"             => [day],
        "time"            => [hour],
        "data_format"     => "netcdf",
        "download_format" => "unarchived",
    )

    area = build_era5_area(meta.bounding_box)
    isnothing(area) || (request["area"] = area)

    @root begin
        CDSAPI.retrieve("reanalysis-era5-pressure-levels", request, output_path)
    end

    return output_path
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

The multi-variable NetCDF returned by CDS is split into individual per-variable files, each
stored at the path returned by `metadata_path(Metadatum(name; dataset=meta.dataset, ...))`.
This keeps subsequent calls to `Field` and `retrieve_data` unchanged.
"""
function download_dataset(names::Vector{Symbol}, meta::ERA5PressureMetadatum; skip_existing=true)
    # Build (symbol → output path) pairs using the standard single-variable filename scheme
    name_path_pairs = [(name, NumericalEarth.DataWrangling.metadata_path(
                            NumericalEarth.DataWrangling.Metadatum(name;
                                dataset      = meta.dataset,
                                bounding_box = meta.bounding_box,
                                date         = meta.dates,
                                dir          = meta.dir)))
                       for name in names]

    # Determine which output files are missing
    pending = if skip_existing
        [(n, p) for (n, p) in name_path_pairs if !isfile(p)]
    else
        name_path_pairs
    end

    isempty(pending) && return [path for (_, path) in name_path_pairs]

    # Unique CDS variable names (:geopotential and :geopotential_height share one CDS field)
    cds_vars = unique([ERA5PL_dataset_variable_names[name] for (name, _) in pending])

    date  = meta.dates
    year  = string(Dates.year(date))
    month = lpad(string(Dates.month(date)), 2, '0')
    day   = lpad(string(Dates.day(date)),   2, '0')
    hour  = lpad(string(Dates.hour(date)),  2, '0') * ":00"

    request = Dict(
        "product_type"    => ["reanalysis"],
        "variable"        => cds_vars,
        "pressure_level"  => [string(p) for p in meta.dataset.levels],
        "year"            => [year],
        "month"           => [month],
        "day"             => [day],
        "time"            => [hour],
        "data_format"     => "netcdf",
        "download_format" => "unarchived",
    )

    area = build_era5_area(meta.bounding_box)
    isnothing(area) || (request["area"] = area)

    mkpath(meta.dir)
    tmp_path = joinpath(meta.dir, "_tmp_multi_$(year)$(month)$(day)T$(hour[1:2]).nc")

    nc_name_path_pairs = [(ERA5PL_netcdf_variable_names[name], path) for (name, path) in pending]

    @root begin
        CDSAPI.retrieve("reanalysis-era5-pressure-levels", request, tmp_path)
        _split_era5pl_nc(tmp_path, nc_name_path_pairs)
        rm(tmp_path; force=true)
    end

    return [path for (_, path) in name_path_pairs]
end

"""
    download_dataset(names::Union{Symbol, Vector{Symbol}},
                     dataset::ERA5PressureDataset,
                     datetime;
                     bounding_box=nothing,
                     dir=default_download_directory(dataset))

Download one or more ERA5 pressure-level variables at a single datetime, without requiring a dummy
`Metadatum`.

# Example
```julia
ds = ERA5HourlyPressureLevels(levels=[850, 500])
download_dataset([:temperature, :geopotential_height, :eastward_velocity], ds;
                 date=DateTime(2020, 6, 15, 0), bounding_box=bbox)
```
"""
function download_dataset(names::Vector{Symbol}, dataset::ERA5PressureDataset, datetime;
                          bounding_box = nothing,
                          dir = NumericalEarth.DataWrangling.default_download_directory(dataset))
    meta = NumericalEarth.DataWrangling.Metadatum(first(names); dataset, datetime, bounding_box, dir)
    return download_dataset(names, meta)
end

function download_dataset(name::Symbol, dataset::ERA5PressureDataset, datetime;
                          bounding_box = nothing,
                          dir = NumericalEarth.DataWrangling.default_download_directory(dataset))
    return download_dataset([name], dataset, datetime; bounding_box, dir)
end

"""
    _split_era5pl_nc(src_path, nc_name_path_pairs)

Split a multi-variable ERA5 pressure-level NetCDF at `src_path` into individual
per-variable files. `nc_name_path_pairs` is a vector of `(nc_varname, dst_path)` tuples.

Each output file contains all coordinate/dimension variables plus the one data variable.
Global attributes and dimension definitions are preserved. Raw data (including scale/offset
encoding) is copied verbatim so that downstream readers decode values identically.
"""
function _split_era5pl_nc(src_path, nc_name_path_pairs)
    NCDatasets.Dataset(src_path, "r") do src
        for (nc_varname, dst_path) in nc_name_path_pairs
            NCDatasets.Dataset(dst_path, "c") do dst
                # Define dimensions
                unlimited = NCDatasets.unlimited(src)
                for (dname, dlen) in src.dim
                    NCDatasets.defDim(dst, dname, dname in unlimited ? Inf : dlen)
                end

                # Copy global attributes
                for (k, v) in src.attrib
                    dst.attrib[k] = v
                end

                # Copy coordinate variables and the one requested data variable
                for (vname, var) in src
                    (vname in ERA5PL_COORD_VARS || vname == nc_varname) || continue
                    _ncvar_copy!(dst, var, vname)
                end
            end
        end
    end
end

function _ncvar_copy!(dst, src_var, vname)
    dims     = NCDatasets.dimnames(src_var)
    T        = eltype(src_var.var)  # raw storage type; eltype(src_var) gives CF-decoded DateTime
    attribs  = src_var.attrib
    fill_val = haskey(attribs, "_FillValue") ? attribs["_FillValue"] : nothing

    dst_var = if isnothing(fill_val)
        NCDatasets.defVar(dst, vname, T, dims)
    else
        NCDatasets.defVar(dst, vname, T, dims; fillvalue=fill_val)
    end

    # Copy variable attributes (skip _FillValue — handled by defVar above)
    for (k, v) in attribs
        k == "_FillValue" && continue
        dst_var.attrib[k] = v
    end

    # Copy raw encoded data, bypassing scale/offset transforms
    dst_var.var[:] = src_var.var[:]

    return nothing
end

"""
    download_dataset(names::Union{Symbol, Vector{Symbol}},
                     dataset::ERA5PressureDataset,
                     datetimes::AbstractVector;
                     bounding_box=nothing,
                     dir=default_download_directory(dataset),
                     skip_existing=true)

Download one or more ERA5 pressure-level variables for a vector of `datetimes`, issuing one CDS API
request per unique calendar day (which may contain multiple hours).

# Example
```julia
ds    = ERA5HourlyPressureLevels(levels=[850, 500])
dates = DateTime(2020, 6, 15, 0) : Hour(6) : DateTime(2020, 6, 16, 18)
download_dataset([:temperature, :geopotential_height], ds, collect(dates); bounding_box=bbox)
```
"""
function download_dataset(names::Vector{Symbol},
                          dataset::ERA5PressureDataset,
                          datetimes::AbstractVector;
                          bounding_box = nothing,
                          dir = NumericalEarth.DataWrangling.default_download_directory(dataset),
                          skip_existing = true,
                          cleanup = true)

    grouped = Dict(d => filter(dt -> Dates.Date(dt) == d, datetimes)
                   for d in unique(Dates.Date.(datetimes)))

    for day in sort(collect(keys(grouped)))
        _download_era5pl_day(names, dataset, grouped[day]; bounding_box, dir, skip_existing, cleanup)
    end

    return nothing
end

function download_dataset(name::Symbol,
                          dataset::ERA5PressureDataset,
                          datetimes::AbstractVector;
                          bounding_box = nothing,
                          dir = NumericalEarth.DataWrangling.default_download_directory(dataset),
                          skip_existing = true,
                          cleanup = true)
    return download_dataset([name], dataset, datetimes; bounding_box, dir, skip_existing, cleanup)
end

function _download_era5pl_day(names, dataset, day_dates;
                              bounding_box, dir, skip_existing, cleanup)

    MDatum    = NumericalEarth.DataWrangling.Metadatum
    meta_path = NumericalEarth.DataWrangling.metadata_path

    # All (name, dt, output path) triples for this day
    all_triples = [(name, dt, meta_path(MDatum(name; dataset, date=dt, bounding_box, dir)))
                   for name in names for dt in day_dates]

    pending = skip_existing ? filter(((_, _, p),) -> !isfile(p), all_triples) : all_triples
    isempty(pending) && return nothing

    # Unique CDS variable names and hours needed
    cds_vars   = unique([ERA5PL_dataset_variable_names[name] for (name, _, _) in pending])
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
        "pressure_level"  => [string(p) for p in dataset.levels],
        "year"            => [year],
        "month"           => [month],
        "day"             => [day],
        "time"            => hours_str,
        "data_format"     => "netcdf",
        "download_format" => "unarchived",
    )

    area = build_era5_area(bounding_box)
    isnothing(area) || (request["area"] = area)

    mkpath(dir)
    tmp_path   = joinpath(dir, "_tmp_multi_$(year)$(month)$(day).nc")
    nc_triples = [(ERA5PL_netcdf_variable_names[name], dt_to_tidx[dt], path)
                  for (name, dt, path) in pending]

    @root begin
        CDSAPI.retrieve("reanalysis-era5-pressure-levels", request, tmp_path)
        _split_era5pl_nc_multistep(tmp_path, nc_triples)
        cleanup && rm(tmp_path; force=true)
    end

    return nothing
end

function _split_era5pl_nc_multistep(src_path, nc_varname_tidx_path_triples)
    time_dimnames = Set(["time", "valid_time"])

    NCDatasets.Dataset(src_path, "r") do src
        unlimited = NCDatasets.unlimited(src)

        for (nc_varname, tidx, dst_path) in nc_varname_tidx_path_triples
            NCDatasets.Dataset(dst_path, "c") do dst
                # Collapse the time dimension to 1; keep all others
                for (dname, dlen) in src.dim
                    out_len = dname in time_dimnames ? 1 :
                              dname in unlimited     ? Inf : dlen
                    NCDatasets.defDim(dst, dname, out_len)
                end

                for (k, v) in src.attrib
                    dst.attrib[k] = v
                end

                for (vname, var) in src
                    (vname in ERA5PL_COORD_VARS || vname == nc_varname) || continue
                    _ncvar_copy_tslice!(dst, var, vname, tidx, time_dimnames)
                end
            end
        end
    end
end

function _ncvar_copy_tslice!(dst, src_var, vname, tidx, time_dimnames)
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

    # Slice time dimension; keep all others intact
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

function build_era5_area(bbox::BBOX)
    # CDS API uses [north, west, south, east] ordering
    # BoundingBox has longitude = (west, east), latitude = (south, north)

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

end # module NumericalEarthCDSAPIExt
