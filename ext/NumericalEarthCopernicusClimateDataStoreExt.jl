module NumericalEarthCopernicusClimateDataStoreExt

using NumericalEarth
using CopernicusClimateDataStore: CopernicusClimateDataStore
using Downloads: Downloads
using Dates: Dates
using Oceananigans.DistributedComputations: @root

using NCDatasets: NCDatasets

using NumericalEarth.DataWrangling: MetadataSet, available_variables, metadata_filename, metadata_path
using NumericalEarth.DataWrangling.ERA5: ERA5Dataset, ERA5PressureLevelsDataset,
                                         ERA5Metadata, ERA5Metadatum, hPa,
                                         ERA5_dataset_variable_names, ERA5PL_dataset_variable_names,
                                         ERA5YearlySingleLevel, ERA5MonthlySingleLevel,
                                         ERA5HourlyPressureLevels, ERA5MonthlyPressureLevels,
                                         ERA5HourlyLand, ERA5MonthlyLand, ERA5LandDataset,
                                         ERA5Land_dataset_variable_names,
                                         batch_datetimes_for_cds, coord_vars, nc_varnames,
                                         split_era5_nc_by_datetime, ERA5_TIME_DIMNAMES

#####
##### era5cli credential bootstrap
#####
##### era5cli reads credentials only from ~/.config/era5cli/cds_key.txt (its ~/.cdsapirc
##### fallback needs a TTY), so non-interactive runs fail even with valid CDSAPI_URL/CDSAPI_KEY
##### env vars. Write its config from those env vars when absent; never overwrite an existing one.
#####

const ERA5CLI_CONFIG_PATH = joinpath(homedir(), ".config", "era5cli", "cds_key.txt")

function ensure_era5cli_credentials()
    isfile(ERA5CLI_CONFIG_PATH) && return nothing
    url = get(ENV, "CDSAPI_URL", "")
    key = get(ENV, "CDSAPI_KEY", "")
    (isempty(url) || isempty(key)) && return nothing
    mkpath(dirname(ERA5CLI_CONFIG_PATH))
    write(ERA5CLI_CONFIG_PATH, "url: $url\nkey: $key\n")
    return nothing
end

#####
##### Batched downloads — same strategy as NumericalEarthCDSAPIExt
#####
##### One era5cli invocation per calendar-month batch: one CDS request per variable, expanded
##### server-side into a `months` × `days` × `hours` product, then split locally into the
##### per-datetime files the readers expect (matched against the file's own time coordinate,
##### so the product's over-fetch is harmless).
#####

"""
    Downloads.download(metadata::ERA5Metadata; skip_existing=true, cleanup=true, threads=nothing, kwargs...)

Download ERA5 data for every date in `metadata` using `era5cli` through the
CopernicusClimateDataStore package, one CDS request per calendar-month batch,
returning the paths of the per-datetime files.

# Keyword Arguments
- `skip_existing`: Skip datetimes whose files already exist (default: `true`).
- `cleanup`: Remove the temporary multi-step NetCDF after splitting (default: `true`).
- `threads`: Number of era5cli download threads (default: one per requested variable).
- Additional keyword arguments are passed to `CopernicusClimateDataStore.hourly`.

# Environment Setup
Before downloading, you must:
1. Create an account at https://cds.climate.copernicus.eu/
2. Accept the Terms of Use for the ERA5 dataset on the dataset page
3. Set up your API credentials in `~/.cdsapirc`

See https://cds.climate.copernicus.eu/how-to-api for details.
"""
function Downloads.download(metadata::ERA5Metadata; kwargs...)
    dates = metadata.dates isa AbstractVector ? metadata.dates : [metadata.dates]
    return download_era5cli([metadata.name], metadata.dataset, dates;
                            region = metadata.region, dir = metadata.dir, kwargs...)
end

"""
    Downloads.download(meta::ERA5Metadatum; skip_existing=true, kwargs...)

Download ERA5 data for a single date/time using `era5cli` through the
CopernicusClimateDataStore package, returning the path of the downloaded file.
"""
function Downloads.download(meta::ERA5Metadatum; kwargs...)
    download_era5cli([meta.name], meta.dataset, [meta.dates];
                     region = meta.region, dir = meta.dir, kwargs...)
    return metadata_path(meta)
end

"""
    Downloads.download(names::Vector{Symbol}, metadata::ERA5Metadata; kwargs...)

Download multiple ERA5 variables for every date in `metadata`, bundling variables
and datetimes into month-batched era5cli invocations.
"""
function Downloads.download(names::Vector{Symbol}, metadata::ERA5Metadata; kwargs...)
    dates = metadata.dates isa AbstractVector ? metadata.dates : [metadata.dates]
    return download_era5cli(names, metadata.dataset, dates;
                            region = metadata.region, dir = metadata.dir, kwargs...)
end

"""
    Downloads.download(mset::MetadataSet{<:ERA5Dataset}; kwargs...)

Download every variable of `mset` together: one era5cli invocation per
calendar-month batch covers all pending variables, and era5cli submits one CDS
request per variable, downloading them with concurrent threads — so the whole
bundle waits in the Copernicus queue at once instead of one variable at a time.
"""
function Downloads.download(mset::MetadataSet{<:ERA5Dataset}; kwargs...)
    names = collect(getfield(mset, :names))
    dates = getfield(mset, :dates)
    dates = dates isa AbstractVector ? dates : [dates]

    return download_era5cli(names, getfield(mset, :dataset), dates;
                            region = getfield(mset, :region),
                            dir = getfield(mset, :dir),
                            kwargs...)
end

function download_era5cli(names, dataset, dates;
                          region, dir,
                          skip_existing = true,
                          cleanup = true,
                          threads = nothing,
                          additional_kw...)

    # era5cli submits one CDS request per variable, so batch sizing is per variable
    batches = batch_datetimes_for_cds(dates, dataset, 1)

    paths = String[]
    for batch in batches
        append!(paths, download_era5cli_month(names, dataset, batch;
                                              region, dir, skip_existing, cleanup, threads,
                                              additional_kw...))
    end

    return paths
end

# era5cli silently returns surface fields unless `--levels` is given, so pass the dataset's
# pressure levels (stored in Pa; CDS wants hPa) for 3-D datasets, and `:surface` to disambiguate
# `geopotential`, which exists on both products.
era5cli_levels(dataset::ERA5PressureLevelsDataset, variable_name) = Int.(dataset.pressure_levels) .÷ hPa
era5cli_levels(dataset::ERA5Dataset, variable_name) = variable_name == "geopotential" ? :surface : nothing

# era5cli rejects `--threads` above 6
const ERA5CLI_MAX_THREADS = 6

function download_era5cli_month(names, dataset, dates;
                                region, dir,
                                skip_existing = true,
                                cleanup = true,
                                threads = nothing,
                                additional_kw...)

    name_dt_paths = [(name, dt, joinpath(dir, metadata_filename(dataset, name, dt, region)))
                     for name in names for dt in dates]

    pending = if skip_existing
        filter(name_dt_path -> !isfile(name_dt_path[3]), name_dt_paths)
    else
        name_dt_paths
    end

    isempty(pending) && return map(name_dt_path -> name_dt_path[3], name_dt_paths)

    mkpath(dir)

    pending_names = unique(map(name_dt_path -> name_dt_path[1], pending))
    sorted_dts    = sort(unique(map(name_dt_path -> name_dt_path[2], pending)))
    dt0 = first(sorted_dts)
    outputprefix = "_tmp_era5cli_$(Dates.year(dt0))$(lpad(Dates.month(dt0), 2, '0'))"

    # era5cli takes one `--levels` flag per invocation, so variables that need different
    # levels (e.g. the ambiguous single-level geopotential's `:surface` next to ordinary
    # single-level variables) go in separate invocations.
    variable_names(group) = [available_variables(dataset)[name] for name in group]
    levels_of(name) = era5cli_levels(dataset, available_variables(dataset)[name])
    levels_values = unique(map(levels_of, pending_names))

    # Each per-variable file era5cli delivers carries only its own variable, and the
    # splitter skips triples whose variable is absent, so every file is split against
    # the full pending set — no filename parsing needed.
    nc_triples = [(nc_varnames(dataset)[name], dt, path) for (name, dt, path) in pending]

    @root begin
        ensure_era5cli_credentials()

        # Stale files left by an interrupted run would hide the fresh downloads from
        # `hourly`'s before/after directory diff — clear matching leftovers first.
        for leftover in filter(f -> startswith(f, outputprefix), readdir(dir))
            rm(joinpath(dir, leftover); force=true)
        end

        for levels in levels_values
            group = filter(name -> isequal(levels_of(name), levels), pending_names)
            group_variable_names = variable_names(group)

            downloaded_files = CopernicusClimateDataStore.hourly(;
                variables = group_variable_names,
                startyear = Dates.year(dt0),
                months = unique(Dates.month.(sorted_dts)),
                days = unique(Dates.day.(sorted_dts)),
                hours = unique(Dates.hour.(sorted_dts)),
                levels,
                area = era5cli_request_area(region, dataset, group),
                format = "netcdf",
                outputprefix,
                overwrite = true,
                threads = min(something(threads, length(group_variable_names)), ERA5CLI_MAX_THREADS),
                splitmonths = false,
                directory = dir,
                additional_kw...)

            for file in downloaded_files
                split_era5_nc_by_datetime(file, nc_triples, coord_vars(dataset), ERA5_TIME_DIMNAMES)
                cleanup && rm(file; force=true)
            end
        end

        undelivered = filter(name_dt_path -> !isfile(name_dt_path[3]), pending)
        isempty(undelivered) ||
            error("The era5cli delivery is missing $(length(undelivered)) of $(length(pending)) ",
                  "requested files, e.g. $(basename(undelivered[1][3])).")
    end

    return map(name_dt_path -> name_dt_path[3], name_dt_paths)
end

#####
##### Helper functions for generic ERA5 download
#####

"""
    variable_name_mapping(dataset)

Return the appropriate variable name dictionary for the dataset type.
"""
variable_name_mapping(::Union{ERA5YearlySingleLevel, ERA5MonthlySingleLevel}) = ERA5_dataset_variable_names
variable_name_mapping(::Union{ERA5HourlyPressureLevels, ERA5MonthlyPressureLevels}) = ERA5PL_dataset_variable_names

"""
    pressure_levels(dataset)

Extract pressure levels from dataset if applicable, otherwise return nothing.
"""
pressure_levels(::Union{ERA5YearlySingleLevel, ERA5MonthlySingleLevel}) = nothing
pressure_levels(dataset::Union{ERA5HourlyPressureLevels, ERA5MonthlyPressureLevels}) = dataset.pressure_levels

"""
    date_keywords(dataset, date)

Build date keyword arguments for CDS API based on dataset granularity.
"""
function date_keywords(::ERA5YearlySingleLevel, date)
    return (; years = Dates.year(date))
end

function date_keywords(::ERA5MonthlySingleLevel, date)
    return (; year = Dates.year(date), month = Dates.month(date))
end

function date_keywords(::ERA5HourlyPressureLevels, date)
    return (;
        startyear = Dates.year(date),
        months = Dates.month(date),
        days = Dates.day(date),
        hours = Dates.hour(date)
    )
end

function date_keywords(::ERA5MonthlyPressureLevels, date)
    return (; year = Dates.year(date), month = Dates.month(date))
end

"""
    cds_download_function(dataset)

Select the appropriate CopernicusClimateDataStore download function.
"""
cds_download_function(::ERA5YearlySingleLevel) = CopernicusClimateDataStore.yearly
cds_download_function(::Union{ERA5MonthlySingleLevel, ERA5MonthlyPressureLevels}) = CopernicusClimateDataStore.monthly
cds_download_function(::ERA5HourlyPressureLevels) = CopernicusClimateDataStore.hourly

"""
    cds_dataset_keyword(dataset)

Select the `dataset` keyword (`:era5` or `:era5_land`) passed to the
CopernicusClimateDataStore download function, which determines the CDS product id
and whether `product_type` is included in the request.
"""
cds_dataset_keyword(::Union{ERA5HourlySingleLevel, ERA5YearlySingleLevel, ERA5MonthlySingleLevel,
                             ERA5HourlyPressureLevels, ERA5MonthlyPressureLevels}) = :era5

#####
##### Generic download implementation
#####

# The hourly datasets use the batched path above. Yearly and monthly datasets instead
# use the package's native `yearly` / `monthly` entry points because their metadata
# filenames represent an entire year or month rather than one hourly timestep.
const NativeGranularityERA5Dataset = Union{ERA5YearlySingleLevel,
                                           ERA5MonthlySingleLevel,
                                           ERA5MonthlyPressureLevels}

const NativeGranularityERA5Metadata =
    NumericalEarth.DataWrangling.Metadata{<:NativeGranularityERA5Dataset}

const NativeGranularityERA5MetadataSet =
    MetadataSet{<:NativeGranularityERA5Dataset}

function Downloads.download(metadata::NativeGranularityERA5Metadata; kwargs...)
    paths = Array{String}(undef, length(metadata))
    for (m, metadatum) in enumerate(metadata)
        paths[m] = Downloads.download(metadatum; kwargs...)
    end
    return paths
end

function Downloads.download(mset::NativeGranularityERA5MetadataSet; kwargs...)
    paths = String[]
    for metadata in mset
        downloaded = Downloads.download(metadata; kwargs...)
        if downloaded isa AbstractVector
            append!(paths, downloaded)
        else
            push!(paths, downloaded)
        end
    end
    return paths
end

"""
    Downloads.download(meta::NumericalEarth.DataWrangling.Metadatum{<:Union{ERA5YearlySingleLevel,
                                                                             ERA5MonthlySingleLevel,
                                                                             ERA5MonthlyPressureLevels}};
                      skip_existing=true, threads=Threads.nthreads(), additional_kw...)

Generic ERA5 download supporting yearly, monthly, and pressure-level datasets.
(Land datasets have their own yearly-file download method, below.)

Downloads are optimized based on dataset granularity:
- Yearly: 8760-8784 hours in single file
- Monthly: ~720-744 hours in single file

Multiple metadata pointing to the same temporal unit (year/month) share one file.
"""
function Downloads.download(meta::NumericalEarth.DataWrangling.Metadatum{<:Union{ERA5YearlySingleLevel,
                                                                                   ERA5MonthlySingleLevel,
                                                                                   ERA5MonthlyPressureLevels}};
                            skip_existing = true,
                            threads = Threads.nthreads(),
                            additional_kw...)

    # Common setup
    output_directory = meta.dir
    output_filename = NumericalEarth.DataWrangling.metadata_filename(meta)
    output_path = joinpath(output_directory, output_filename)

    # Skip if file already exists
    if skip_existing && isfile(output_path)
        return output_path
    end

    # Ensure output directory exists
    mkpath(output_directory)

    # Get dataset-specific mappings and parameters
    dataset = meta.dataset
    var_mapping = variable_name_mapping(dataset)
    variable_name = var_mapping[meta.name]
    date_kw = date_keywords(dataset, meta.dates)
    pl = pressure_levels(dataset)
    download_fn = cds_download_function(dataset)

    # Convert pressure levels from Pa to hPa if present
    pl_hPa = isnothing(pl) ? nothing : [round(Int, p * 1e-2) for p in pl]

    # Build area constraint from region
    area = era5_request_area(meta.region, meta.dataset, meta.name)

    # Build output prefix (filename without extension)
    output_prefix = first(splitext(output_filename))

    # Download using the appropriate CDS function
    @root begin
        downloaded_files = download_fn(;
            variables = variable_name,
            date_kw...,  # Splat dataset-specific date keywords
            area = area,
            pressure_levels = pl_hPa,
            dataset = cds_dataset_keyword(dataset),
            format = "netcdf",
            outputprefix = output_prefix,
            directory = output_directory,
            overwrite = !skip_existing,
            threads = threads,
            additional_kw...
        )

        # Handle potential filename mismatch
        if !isempty(downloaded_files)
            downloaded_file = first(downloaded_files)
            if downloaded_file != output_path && isfile(downloaded_file)
                mv(downloaded_file, output_path; force=true)
            end
        end
    end

    return output_path
end

#####
##### ERA5-Land yearly-file download
#####
##### One file per (variable, year), like ERA5YearlySingleLevel. A year of monthly
##### means fits in a single CDS request; a year of hourly data exceeds ERA5-Land's
##### per-request cost limit, so it is fetched one calendar month at a time and the
##### chunks are concatenated locally into the year's file.
#####

cds_land_product(::ERA5HourlyLand)  = "reanalysis-era5-land"
cds_land_product(::ERA5MonthlyLand) = "reanalysis-era5-land-monthly-means"

function group_by_calendar_month(datetimes)
    keys = unique([(Dates.year(dt), Dates.month(dt)) for dt in datetimes])
    return Dict(k => filter(dt -> (Dates.year(dt), Dates.month(dt)) == k, datetimes) for k in keys)
end

era5_land_year_batches(::ERA5MonthlyLand, dates) = [dates]

function era5_land_year_batches(::ERA5HourlyLand, dates)
    monthly = group_by_calendar_month(dates)
    return [sort(monthly[key]) for key in sort(collect(keys(monthly)))]
end

function era5_land_request(variable_name, ::ERA5HourlyLand, dts, area)
    request = Dict{String, Any}(
        "variable" => [variable_name],
        "year"     => unique(string.(Dates.year.(dts))),
        "month"    => unique(lpad.(string.(Dates.month.(dts)), 2, '0')),
        "day"      => unique(lpad.(string.(Dates.day.(dts)), 2, '0')),
        "time"     => unique([lpad(string(Dates.hour(dt)), 2, '0') * ":00" for dt in dts]),
        "data_format"     => "netcdf",
        "download_format" => "unarchived",
    )
    isnothing(area) || (request["area"] = area)
    return request
end

function era5_land_request(variable_name, ::ERA5MonthlyLand, dts, area)
    request = Dict{String, Any}(
        "variable"     => [variable_name],
        "product_type" => ["monthly_averaged_reanalysis"],
        "year"         => unique(string.(Dates.year.(dts))),
        "month"        => unique(lpad.(string.(Dates.month.(dts)), 2, '0')),
        "time"         => ["00:00"],
        "data_format"     => "netcdf",
        "download_format" => "unarchived",
    )
    isnothing(area) || (request["area"] = area)
    return request
end

function retrieve_with_retries(product, request, path; max_retries=3)
    for attempt in 1:max_retries
        try
            return CopernicusClimateDataStore.retrieve(product, request, path)
        catch e
            attempt < max_retries || rethrow(e)
            @warn "CDS retrieve attempt $attempt/$max_retries failed for $product; retrying..." exception=(e, catch_backtrace())
            sleep(5.0 * attempt)
        end
    end
end

# Append single-variable NetCDF chunks along the time dimension into one yearly file.
function concatenate_era5_nc(src_paths, dst_path)
    srcs = [NCDatasets.Dataset(p, "r") for p in src_paths]
    try
        src1 = first(srcs)
        time_dim = haskey(src1, "valid_time") ? "valid_time" : "time"
        total_time = sum(src.dim[time_dim] for src in srcs)

        NCDatasets.Dataset(dst_path, "c") do dst
            for (dname, dlen) in src1.dim
                NCDatasets.defDim(dst, dname, dname == time_dim ? total_time : dlen)
            end

            for (k, v) in src1.attrib
                dst.attrib[k] = v
            end

            for (vname, var) in src1
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
                        dst_idx = ntuple(i -> dims[i] == time_dim ? (offset+1:offset+n) : Colon(), length(dims))
                        src_idx = ntuple(Returns(Colon()), length(dims))
                        dst_var.var[dst_idx...] = src[vname].var[src_idx...]
                        offset += n
                    end
                else
                    dst_var.var[:] = var.var[:]
                end
            end
        end
    finally
        foreach(close, srcs)
    end
    return dst_path
end

# CDS expects [north, west, south, east]; `era5_request_area` returns [south, west, north, east].
cds_area(::Nothing) = nothing
cds_area(area) = [area[3], area[2], area[1], area[4]]

"""
    NumericalEarth.DataWrangling.ERA5.download_era5_land(meta::NumericalEarth.DataWrangling.Metadatum{<:ERA5LandDataset};
                                                         skip_existing=true, additional_kw...)

Download a whole year of ERA5-Land data for one variable into a single yearly file,
the first time any date within that year is requested; later dates in the same year
find the file already on disk and skip (see `skip_existing`).

Implements the `download_era5_land` stub declared in `src/DataWrangling/ERA5/ERA5_land.jl`,
which owns `Downloads.download` for ERA5-Land metadata under any extension load order.
"""
function NumericalEarth.DataWrangling.ERA5.download_era5_land(meta::NumericalEarth.DataWrangling.Metadatum{<:ERA5LandDataset};
                                                              skip_existing = true,
                                                              additional_kw...)

    output_directory = meta.dir
    output_filename = NumericalEarth.DataWrangling.metadata_filename(meta)
    output_path = joinpath(output_directory, output_filename)

    if skip_existing && isfile(output_path)
        return output_path
    end

    mkpath(output_directory)

    dataset = meta.dataset
    variable_name = ERA5Land_dataset_variable_names[meta.name]
    year = Dates.year(meta.dates)
    year_dates = filter(dt -> Dates.year(dt) == year,
                         NumericalEarth.DataWrangling.all_dates(dataset, meta.name))
    area = cds_area(era5_request_area(meta.region, meta.dataset, meta.name))
    batches = era5_land_year_batches(dataset, year_dates)

    @root begin
        tmp_dir = mktempdir(output_directory)
        chunk_paths = [joinpath(tmp_dir, "chunk_$(lpad(i, 2, '0')).nc") for i in eachindex(batches)]

        for (batch, chunk_path) in zip(batches, chunk_paths)
            request = era5_land_request(variable_name, dataset, batch, area)
            retrieve_with_retries(cds_land_product(dataset), request, chunk_path)
        end

        concatenate_era5_nc(chunk_paths, output_path)
        rm(tmp_dir; recursive=true, force=true)
    end

    return output_path
end

#####
##### Area/bounding box utilities
#####

const BBOX = NumericalEarth.DataWrangling.BoundingBox

padded_era5_region(region, dataset, name::Symbol) = padded_era5_region(region, dataset, [name])
padded_era5_region(::Nothing, dataset, names::Vector{Symbol}) = nothing

# Pad by two native cells of the coarsest requested variable (waves live on 0.5°, the rest
# 0.25°): center-bracketing `restrict` can reach one cell past a boundary-aligned edge, so an
# unpadded request comes up a cell short on each edge and the read is rejected. The margin also
# absorbs era5cli's two-decimal rounding. Over-fetch is harmless because the reader selects the
# exact cells from the file.
function padded_era5_region(bbox::BBOX, dataset, names::Vector{Symbol})
    (isnothing(bbox.longitude) || isnothing(bbox.latitude)) && return nothing
    Δλ = maximum(360 / size(dataset, name)[1] for name in names)
    Δφ = maximum(180 / size(dataset, name)[2] for name in names)
    lon = bbox.longitude
    lat = bbox.latitude
    return BBOX(longitude = (lon[1] - 2Δλ, lon[2] + 2Δλ),
                latitude  = (max(lat[1] - 2Δφ, -90), min(lat[2] + 2Δφ, 90)))
end

# `hourly` wants the padded area as (lat, lon) tuples; `monthly`, `yearly`, and the land
# products want it as a 4-element array.
era5cli_request_area(region, dataset, names) = build_era5cli_area(padded_era5_region(region, dataset, names))
era5_request_area(region, dataset, names) = build_era5_area(padded_era5_region(region, dataset, names))

build_era5cli_area(::Nothing) = nothing
build_era5_area(::Nothing) = nothing

function era5_area_extrema(bbox::BBOX)
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

    return (; west, east, south, north)
end

function build_era5cli_area(bbox::BBOX)
    extrema = era5_area_extrema(bbox)
    isnothing(extrema) && return nothing
    return (lat = (extrema.south, extrema.north),
            lon = (extrema.west, extrema.east))
end

function build_era5_area(bbox::BBOX)
    extrema = era5_area_extrema(bbox)
    isnothing(extrema) && return nothing
    # CopernicusClimateDataStore.monthly / yearly take [south, west, north, east].
    return [extrema.south, extrema.west, extrema.north, extrema.east]
end

#####
##### Copernicus land surface albedo, through the package's native CDS client (≥ 0.2).
##### This backend only supplies `retrieve`; the request construction, extraction, and
##### repacking live in the CopernicusLandAlbedo module.
#####

using NumericalEarth.DataWrangling.CopernicusLandAlbedo: ALBEDO_CDS_PRODUCT,
                                                         CopernicusAlbedoDatasetMetadata,
                                                         download_ten_day_albedo!

function Downloads.download(metadata::CopernicusAlbedoDatasetMetadata; kwargs...)
    isdefined(CopernicusClimateDataStore, :retrieve) || throw(ArgumentError(
        "Downloading the Copernicus land albedo needs CopernicusClimateDataStore ≥ 0.2, " *
        "whose native CDS client provides `retrieve`."))

    return download_ten_day_albedo!(metadata; kwargs...) do request, path
        CopernicusClimateDataStore.retrieve(ALBEDO_CDS_PRODUCT, request, path)
    end
end

end # module NumericalEarthCopernicusClimateDataStoreExt
