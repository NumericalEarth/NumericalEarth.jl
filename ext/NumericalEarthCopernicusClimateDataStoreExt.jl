module NumericalEarthCopernicusClimateDataStoreExt

using NumericalEarth
using CopernicusClimateDataStore: CopernicusClimateDataStore
using Downloads: Downloads
using Dates: Dates
using Oceananigans.DistributedComputations: @root

using NCDatasets: NCDatasets

using NumericalEarth.DataWrangling.ERA5: ERA5Metadata, ERA5Metadatum,
                                          ERA5_dataset_variable_names, ERA5PL_dataset_variable_names,
                                          ERA5YearlySingleLevel, ERA5MonthlySingleLevel,
                                          ERA5HourlyPressureLevels, ERA5MonthlyPressureLevels,
                                          ERA5HourlyLand, ERA5MonthlyLand, ERA5LandDataset,
                                          ERA5Land_dataset_variable_names

"""
    Downloads.download(metadata::ERA5Metadata; kwargs...)

Download ERA5 data for each date in the metadata, returning paths to downloaded files.
"""
function Downloads.download(metadata::ERA5Metadata; kwargs...)
    paths = Array{String}(undef, length(metadata))
    for (m, metadatum) in enumerate(metadata)
        paths[m] = Downloads.download(metadatum; kwargs...)
    end
    return paths
end

"""
    Downloads.download(meta::ERA5Metadatum; skip_existing=true, kwargs...)

Download ERA5 data for a single date/time using the CopernicusClimateDataStore package.

The download is performed using `era5cli` through the CopernicusClimateDataStore package.

# Keyword Arguments
- `skip_existing`: Skip download if the file already exists (default: `true`).
- `threads`: Number of download threads (default: `1`).
- Additional keyword arguments are passed to `CopernicusClimateDataStore.hourly`.

# Environment Setup
Before downloading, you must:
1. Create an account at https://cds.climate.copernicus.eu/
2. Accept the Terms of Use for the ERA5 dataset on the dataset page
3. Set up your API credentials in `~/.cdsapirc`

See https://cds.climate.copernicus.eu/how-to-api for details.
"""
function Downloads.download(meta::ERA5Metadatum;
                            skip_existing = true,
                            threads = Threads.nthreads(),
                            additional_kw...)

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
    year = Dates.year(date)
    month = Dates.month(date)
    day = Dates.day(date)
    hour = Dates.hour(date)

    # Build area constraint from region
    area = build_era5_area(meta.region, meta.dataset, meta.name)

    # Build output prefix (filename without extension)
    output_prefix = first(splitext(output_filename))

    # Perform the download using era5cli via CopernicusClimateDataStore
    @root begin
        downloaded_files = CopernicusClimateDataStore.hourly(;
            variables = variable_name,
            startyear = year,
            months = month,
            days = day,
            hours = hour,
            area = area,
            format = "netcdf",
            outputprefix = output_prefix,
            overwrite = !skip_existing,
            threads = threads,
            splitmonths = false,
            directory = output_directory,
            additional_kw...
        )

        # era5cli generates its own filename suffix, so rename to our expected name
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

"""
    Downloads.download(meta::NumericalEarth.DataWrangling.Metadatum{<:Union{ERA5YearlySingleLevel,
                                                                             ERA5MonthlySingleLevel,
                                                                             ERA5HourlyPressureLevels,
                                                                             ERA5MonthlyPressureLevels}};
                      skip_existing=true, threads=Threads.nthreads(), additional_kw...)

Generic ERA5 download supporting yearly, monthly, and pressure-level datasets.
(Land datasets have their own yearly-file download method, below.)

Downloads are optimized based on dataset granularity:
- Yearly: 8760-8784 hours in single file
- Monthly: ~720-744 hours in single file
- Pressure levels: 3D atmospheric data at specified vertical levels

Multiple metadata pointing to the same temporal unit (year/month) share one file.
"""
function Downloads.download(meta::NumericalEarth.DataWrangling.Metadatum{<:Union{ERA5YearlySingleLevel,
                                                                                   ERA5MonthlySingleLevel,
                                                                                   ERA5HourlyPressureLevels,
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
    area = build_era5_area(meta.region, meta.dataset, meta.name)

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
            threads = threads
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

# CDS expects [north, west, south, east]; build_era5_area returns [south, west, north, east].
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
    area = cds_area(build_era5_area(meta.region, meta.dataset, meta.name))
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

build_era5_area(::Nothing) = nothing

const BBOX = NumericalEarth.DataWrangling.BoundingBox

# Pad the request by two native cells on each side so the delivered file covers every
# native cell a regional read of `region` needs (the read is rejected otherwise);
# over-fetching is harmless because the reader selects the exact cells from the file.
build_era5_area(::Nothing, dataset, name) = nothing

function build_era5_area(region::BBOX, dataset, name)
    (isnothing(region.longitude) || isnothing(region.latitude)) && return build_era5_area(region)
    Nx, Ny, _ = size(dataset, name)
    Δλ = 360 / Nx
    Δφ = 180 / Ny
    lon = region.longitude
    lat = region.latitude
    padded = BBOX(longitude = (lon[1] - 2Δλ, lon[2] + 2Δλ),
                  latitude  = (max(lat[1] - 2Δφ, -90), min(lat[2] + 2Δφ, 90)))
    return build_era5_area(padded)
end

function build_era5_area(bbox::BBOX)
    # CDS API / yearly() uses [south, west, north, east] ordering (4-element array)
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

    # Return as 4-element array: [south, west, north, east]
    return [south, west, north, east]
end

end # module NumericalEarthCopernicusClimateDataStoreExt
