#####
##### Shared machinery for batched ERA5 downloads
#####
##### Every download backend (CDSAPI.jl, era5cli via CopernicusClimateDataStore.jl) uses the
##### same strategy: fetch many datetimes in one CDS request per variable — batched by calendar
##### month and capped by the CDS per-request cost limit — then split the returned multi-step
##### NetCDF into the per-variable, per-datetime files the readers expect. The batching and
##### splitting utilities live here; each backend only builds its own request from a batch.
#####

##### CDS interprets `year`/`month`/`day`/`time` as a Cartesian product, so a
##### single request can cover many days × hours per call as long as `year`
##### and `month` stay singletons. The remaining constraint is the per-request
##### cost limit: roughly `num_vars × num_pressure_levels × num_datetimes`
##### must stay under ~7500 for pressure-level data (CDS returns HTTP 403
##### "cost limits exceeded" otherwise). We pick a conservative cap and split
##### each month into smaller contiguous chunks when needed.

const CDS_MAX_FIELDS_PER_REQUEST = 5000

const ERA5_TIME_DIMNAMES = Set(["time", "valid_time"])

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

nc_varnames(::ERA5Dataset)               = ERA5_netcdf_variable_names
nc_varnames(::ERA5PressureLevelsDataset) = ERA5PL_netcdf_variable_names

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

fields_per_datetime(dataset::ERA5Dataset)               = 1
fields_per_datetime(dataset::ERA5PressureLevelsDataset) = length(dataset.pressure_levels)

"""
    max_dts_per_cds_request(dataset, num_vars; max_fields=$(CDS_MAX_FIELDS_PER_REQUEST))

Maximum number of datetimes that can share a single CDS request before the
per-request cost limit is hit. Pressure-level datasets multiply by the number
of selected pressure levels; single-level datasets count as one level. Falls
back to `1` when a single datetime already exceeds the cap (which would force
the caller's single-datetime download path).
"""
max_dts_per_cds_request(dataset, num_vars; max_fields=CDS_MAX_FIELDS_PER_REQUEST) =
    max(1, fld(max_fields, num_vars * fields_per_datetime(dataset)))

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

    batches = Vector{DateTime}[]
    for key in sort(collect(Base.keys(monthly)))
        sorted = sort(unique(monthly[key]))
        for i in 1:max_dts:length(sorted)
            push!(batches, sorted[i:min(i + max_dts - 1, end)])
        end
    end
    return batches
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
##### NetCDF splitting utilities
#####

"""
    split_era5_nc(src_path, nc_name_path_pairs, coordinate_vars)

Split a multi-variable NetCDF into individual per-variable files (single time step).
"""
function split_era5_nc(src_path, nc_name_path_pairs, coordinate_vars)
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
                    (vname in coordinate_vars || vname == nc_varname) || continue
                    ncvar_copy!(dst, var, vname)
                end
            end
        end
    end
end

"""
    split_era5_nc_multistep(src, nc_varname_tidx_path_triples, coordinate_vars, time_dimnames)

Split a multi-timestep NetCDF into individual per-variable, per-timestep files.
`nc_varname_tidx_path_triples` is a vector of `(nc_varname, time_index, dst_path)`;
`src` is a path or an already-open `NCDatasets.NCDataset`.
"""
split_era5_nc_multistep(src_path::AbstractString, nc_varname_tidx_path_triples, coordinate_vars, time_dimnames) =
    NCDatasets.Dataset(src -> split_era5_nc_multistep(src, nc_varname_tidx_path_triples, coordinate_vars, time_dimnames),
                       src_path, "r")

function split_era5_nc_multistep(src::NCDatasets.NCDataset, nc_varname_tidx_path_triples, coordinate_vars, time_dimnames)
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
                (vname in coordinate_vars || vname == nc_varname) || continue
                ncvar_copy_tslice!(dst, var, vname, tidx, time_dimnames)
            end
        end
    end
end

"""
    split_era5_nc_by_datetime(src_path, nc_varname_dt_path_triples, coordinate_vars, time_dimnames)

Split a multi-timestep NetCDF into individual per-variable, per-timestep files, locating each
requested `DateTime` in the file's own time coordinate. `nc_varname_dt_path_triples` is a
vector of `(nc_varname, datetime, dst_path)`. Because CDS expands `year`/`month`/`day`/`time`
into a Cartesian product, the file may contain more datetimes than were asked for; matching by
time value (not by position) extracts exactly the requested ones, and errors loudly if a
requested datetime is absent.
"""
function split_era5_nc_by_datetime(src_path, nc_varname_dt_path_triples, coordinate_vars, time_dimnames)
    NCDatasets.Dataset(src_path, "r") do src
        candidate_names = ("valid_time", "time")
        n = findfirst(name -> haskey(src, name), candidate_names)
        isnothing(n) &&
            error("Cannot split $src_path: no time coordinate ($(join(candidate_names, ", "))) found.")
        file_times = [DateTime(t) for t in src[candidate_names[n]][:]]
        tidx_of_dt = Dict(t => i for (i, t) in enumerate(file_times))

        triples = map(nc_varname_dt_path_triples) do (nc_varname, dt, dst_path)
            tidx = get(tidx_of_dt, DateTime(dt), nothing)
            isnothing(tidx) &&
                error("Cannot split $src_path: it does not contain $dt ",
                      "(file times span $(first(file_times)) — $(last(file_times))).")
            return (nc_varname, tidx, dst_path)
        end

        split_era5_nc_multistep(src, triples, coordinate_vars, time_dimnames)
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
