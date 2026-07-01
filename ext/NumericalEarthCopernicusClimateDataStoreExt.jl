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

"""
    Downloads.download(meta::NumericalEarth.DataWrangling.Metadatum{<:ERA5YearlySingleLevel};
                      skip_existing=true, threads=1, additional_kw...)

Download yearly ERA5 file if needed. Multiple metadata pointing to the same year
will result in only one download (file is shared across all hours in that year).

Yearly files contain all 8760-8784 hours for one variable for one year in a single NetCDF file.
This is 8760× more efficient than downloading individual hourly files.
"""
function Downloads.download(meta::NumericalEarth.DataWrangling.Metadatum{<:ERA5YearlySingleLevel};
                            skip_existing = true,
                            threads = 1,
                            additional_kw...)

    output_directory = meta.dir
    output_filename = NumericalEarth.DataWrangling.metadata_filename(meta)
    output_path = joinpath(output_directory, output_filename)

    # Skip if yearly file already exists
    if skip_existing && isfile(output_path)
        # Silently skip (avoid logging once per hour for yearly files)
        return output_path
    end

    # Ensure output directory exists
    mkpath(output_directory)

    # Extract metadata fields
    variable_name = ERA5_dataset_variable_names[meta.name]
    year = Dates.year(meta.dates)

    # Build area constraint from region
    area = build_era5_area(meta.region)

    # Build output prefix (filename without extension)
    output_prefix = first(splitext(output_filename))

    # Download full year using CopernicusClimateDataStore.yearly()
    # NOTE: Do NOT pass additional_kw - it may contain date restrictions
    #       that would limit download to less than a full year
    @root begin
        downloaded_files = CopernicusClimateDataStore.yearly(;
            variables = variable_name,
            years = year,
            area = area,
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

"""
    Downloads.download(meta::NumericalEarth.DataWrangling.Metadatum{<:ERA5MonthlySingleLevel};
                      skip_existing=true, threads=1, additional_kw...)

Download monthly ERA5 file if needed. Multiple metadata pointing to the same month
will result in only one download (file is shared across all hours in that month).

Monthly files contain all ~720-744 hours for one variable for one month in a single NetCDF file.
This is 720× more efficient than downloading individual hourly files.
"""
function Downloads.download(meta::NumericalEarth.DataWrangling.Metadatum{<:ERA5MonthlySingleLevel};
                           skip_existing = true,
                           threads = Threads.nthreads(),
                           additional_kw...)

    output_directory = meta.dir
    output_filename = NumericalEarth.DataWrangling.metadata_filename(meta)
    output_path = joinpath(output_directory, output_filename)

    # Skip if monthly file already exists
    if skip_existing && isfile(output_path)
        return output_path
    end

    # Ensure output directory exists
    mkpath(output_directory)

    # Extract metadata fields
    variable_name = ERA5_dataset_variable_names[meta.name]
    year = Dates.year(meta.dates)
    month = Dates.month(meta.dates)

    # Build area constraint from region
    area = build_era5_area(meta.region)

    # Build output prefix (filename without extension)
    output_prefix = first(splitext(output_filename))

    # Download full month using CopernicusClimateDataStore.monthly()
    # NOTE: Do NOT pass additional_kw - it may contain date restrictions
    #       that would limit download to less than a full month
    @root begin
        downloaded_files = CopernicusClimateDataStore.monthly(;
            variables = variable_name,
            year = year,
            month = month,
            area = area,
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

"""
    Downloads.download(meta::NumericalEarth.DataWrangling.Metadatum{<:ERA5HourlyPressureLevels};
                      skip_existing=true, threads=1, additional_kw...)

Download hourly ERA5 pressure-level data using pure Julia CDS client.

Pressure levels are extracted from the dataset and passed to the CDS API.
This downloads 3D atmospheric data at specified pressure levels.
"""
function Downloads.download(meta::NumericalEarth.DataWrangling.Metadatum{<:ERA5HourlyPressureLevels};
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

    # Extract metadata fields
    variable_name = ERA5PL_dataset_variable_names[meta.name]
    date = meta.dates
    year = Dates.year(date)
    month = Dates.month(date)
    day = Dates.day(date)
    hour = Dates.hour(date)

    # Extract pressure levels from dataset (in Pa, convert to hPa)
    pressure_levels_Pa = meta.dataset.pressure_levels
    pressure_levels_hPa = [round(Int, p * 1e-2) for p in pressure_levels_Pa]

    # Build area constraint from region
    area = build_era5_area(meta.region)

    # Build output prefix (filename without extension)
    output_prefix = first(splitext(output_filename))

    # Download using pure Julia CDS client with pressure levels
    @root begin
        downloaded_files = CopernicusClimateDataStore.hourly(;
            variables = variable_name,
            startyear = year,
            months = month,
            days = day,
            hours = hour,
            pressure_levels = pressure_levels_hPa,
            area = area,
            format = "netcdf",
            outputprefix = output_prefix,
            overwrite = !skip_existing,
            threads = threads,
            splitmonths = false,
            directory = output_directory,
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

"""
    Downloads.download(meta::NumericalEarth.DataWrangling.Metadatum{<:ERA5MonthlyPressureLevels};
                      skip_existing=true, threads=1, additional_kw...)

Download monthly ERA5 pressure-level data using pure Julia CDS client.

Downloads full month (all days, all hours) at specified pressure levels.
Multiple metadata pointing to the same month will result in only one download.
"""
function Downloads.download(meta::NumericalEarth.DataWrangling.Metadatum{<:ERA5MonthlyPressureLevels};
                           skip_existing = true,
                           threads = Threads.nthreads(),
                           additional_kw...)

    output_directory = meta.dir
    output_filename = NumericalEarth.DataWrangling.metadata_filename(meta)
    output_path = joinpath(output_directory, output_filename)

    # Skip if monthly file already exists
    if skip_existing && isfile(output_path)
        # Silently skip (avoid logging once per hour for monthly files)
        return output_path
    end

    # Ensure output directory exists
    mkpath(output_directory)

    # Extract metadata fields
    variable_name = ERA5PL_dataset_variable_names[meta.name]
    year = Dates.year(meta.dates)
    month = Dates.month(meta.dates)

    # Extract pressure levels from dataset (in Pa, convert to hPa)
    pressure_levels_Pa = meta.dataset.pressure_levels
    pressure_levels_hPa = [round(Int, p * 1e-2) for p in pressure_levels_Pa]

    # Build area constraint from region
    area = build_era5_area(meta.region)

    # Build output prefix (filename without extension)
    output_prefix = first(splitext(output_filename))

    # Download full month using pure Julia CDS client with pressure levels
    @root begin
        downloaded_files = CopernicusClimateDataStore.monthly(;
            variables = variable_name,
            year = year,
            month = month,
            pressure_levels = pressure_levels_hPa,
            area = area,
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
