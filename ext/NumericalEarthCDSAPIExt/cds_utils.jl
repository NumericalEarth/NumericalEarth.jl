const BBOX = NumericalEarth.DataWrangling.BoundingBox
const COL  = NumericalEarth.DataWrangling.Column
const LIN  = NumericalEarth.DataWrangling.Linear
const NR   = NumericalEarth.DataWrangling.Nearest

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
##### NetCDF splitting utilities
#####

"""
    split_nc_multistep(src_path, triples, coord_vars, time_dimnames)

Split a multi-timestep NetCDF into individual per-variable, per-timestep files.
`triples` is a vector of `(nc_varname, datetime, dst_path)`.

Each `datetime`'s timestep is located by matching it against `src`'s time coordinate, NOT by its
position in the request. CDS expands `day`/`time` into a Cartesian product, so a request whose
datetimes span more than one day with differing hours (e.g. a window crossing midnight) comes back
with extra, sorted timesteps that no longer line up positionally with the requested datetimes.
"""
function split_nc_multistep(src_path, nc_varname_datetime_path_triples, coord_vars, time_dimnames)
    NCDatasets.Dataset(src_path, "r") do src
        src_varnames = Set(keys(src))
        unlimited = NCDatasets.unlimited(src)

        # Index this file's timesteps by their valid time (see the note above).
        time_coord = "valid_time" in src_varnames ? "valid_time" :
                     "time"       in src_varnames ? "time"       :
                     error("split_nc_multistep: no time coordinate variable in $src_path")
        tidx_of = Dict(t => i for (i, t) in enumerate(src[time_coord][:]))

        for (nc_varname, datetime, dst_path) in nc_varname_datetime_path_triples
            nc_varname in src_varnames || continue
            haskey(tidx_of, datetime) ||
                error("split_nc_multistep: $datetime absent from $src_path")
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

"""
    concatenate_era5_nc(src_paths, nc_name_path_pairs, coord_vars, time_dimnames)

Concatenate NetCDF files holding consecutive time windows of the same variables on the
same grid into one file per variable, appending along the time dimension. `src_paths`
must be ordered chronologically; `nc_name_path_pairs` is a vector of
`(nc_varname, dst_path)` pairs. Non-time coordinate variables are copied from the first
source.
"""
function concatenate_era5_nc(src_paths, nc_name_path_pairs, coord_vars, time_dimnames)
    srcs = [NCDatasets.Dataset(src_path, "r") for src_path in src_paths]
    try
        src1 = first(srcs)
        src_varnames = Set(keys(src1))

        time_dims = [dname for (dname, _) in src1.dim if dname in time_dimnames]
        time_dim = isempty(time_dims) ?
            error("concatenate_era5_nc: no time dimension in $(first(src_paths))") :
            only(time_dims)
        total_time = sum(src.dim[time_dim] for src in srcs)

        for (nc_varname, dst_path) in nc_name_path_pairs
            nc_varname in src_varnames || continue
            NCDatasets.Dataset(dst_path, "c") do dst
                for (dname, dlen) in src1.dim
                    NCDatasets.defDim(dst, dname, dname == time_dim ? total_time : dlen)
                end

                for (k, v) in src1.attrib
                    dst.attrib[k] = v
                end

                for (vname, var) in src1
                    (vname in coord_vars || vname == nc_varname) || continue
                    dims     = NCDatasets.dimnames(var)
                    T        = eltype(var.var)
                    attribs  = var.attrib
                    fill_val = haskey(attribs, "_FillValue") ? attribs["_FillValue"] : nothing

                    dst_var = isnothing(fill_val) ?
                        NCDatasets.defVar(dst, vname, T, dims) :
                        NCDatasets.defVar(dst, vname, T, dims; fillvalue=fill_val)

                    for (k, v) in attribs
                        k == "_FillValue" && continue
                        dst_var.attrib[k] = v
                    end

                    if time_dim in dims
                        offset = 0
                        for src in srcs
                            n = src.dim[time_dim]
                            dst_idx = ntuple(i -> dims[i] == time_dim ? (offset+1:offset+n) : Colon(),
                                             length(dims))
                            src_idx = ntuple(Returns(Colon()), length(dims))
                            dst_var.var[dst_idx...] = src[vname].var[src_idx...]
                            offset += n
                        end
                    else
                        dst_var.var[:] = var.var[:]
                    end
                end
            end
        end
    finally
        foreach(close, srcs)
    end
    return nothing
end
