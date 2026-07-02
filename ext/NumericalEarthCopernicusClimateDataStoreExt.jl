module NumericalEarthCopernicusClimateDataStoreExt

using NumericalEarth
using CopernicusClimateDataStore
using Downloads: Downloads
using Dates
using Oceananigans.DistributedComputations: @root

using NumericalEarth.DataWrangling.ERA5: ERA5Metadata, ERA5Metadatum,
                                          ERA5_dataset_variable_names, ERA5PL_dataset_variable_names,
                                          ERA5YearlySingleLevel, ERA5MonthlySingleLevel,
                                          ERA5HourlyPressureLevels, ERA5MonthlyPressureLevels

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
    area = build_era5_area(meta.region)

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
    area = build_era5_area(meta.region)

    # Build output prefix (filename without extension)
    output_prefix = first(splitext(output_filename))

    # Download using the appropriate CDS function
    @root begin
        downloaded_files = download_fn(;
            variables = variable_name,
            date_kw...,  # Splat dataset-specific date keywords
            area = area,
            pressure_levels = pl_hPa,
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
##### Area/bounding box utilities
#####

build_era5_area(::Nothing) = nothing

const BBOX = NumericalEarth.DataWrangling.BoundingBox

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
