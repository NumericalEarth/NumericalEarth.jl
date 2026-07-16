module NumericalEarthCopernicusClimateDataStoreExt

using NumericalEarth
using CopernicusClimateDataStore: CopernicusClimateDataStore
using Downloads: Downloads
using Dates: Dates
using Oceananigans.DistributedComputations: @root

using NumericalEarth.DataWrangling: is_three_dimensional, available_variables
using NumericalEarth.DataWrangling.ERA5: ERA5Metadata, ERA5Metadatum, hPa

#####
##### era5cli credential bootstrap
#####
##### era5cli reads CDS credentials ONLY from ~/.config/era5cli/cds_key.txt; it ignores the
##### CDSAPI_URL/CDSAPI_KEY env vars, and its ~/.cdsapirc fallback needs an interactive TTY, so in a
##### non-interactive/env-var environment (e.g. CI) it raises InvalidLoginError even when valid CDS
##### credentials are present. Write era5cli's config file from the CDSAPI_URL/CDSAPI_KEY env vars
##### (the same variables the CDSAPI backend reads) when it is absent — never overwriting an existing
##### config. Mirrors what `era5cli config` writes.
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
                            threads = 1,
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

    # The CDS catalog name is dataset-dependent: `eastward_velocity` is `u_component_of_wind` on
    # pressure levels but `10m_u_component_of_wind` on single levels. Dispatch on the dataset so a
    # pressure-level request doesn't silently fetch the surface field (which returned `u10`).
    variable_name = available_variables(meta.dataset)[meta.name]

    # era5cli defaults to the single-levels (surface) product unless `--levels` is given — so a
    # pressure-level request without levels silently returns a surface field (e.g. `u10` for
    # `u_component_of_wind`). Pass the dataset's pressure levels for 3-D datasets; disambiguate the
    # single-level `geopotential`/`topography` (surface geopotential, exists on both) with `:surface`;
    # ordinary single-level variables keep era5cli's default (`nothing`). `pressure_levels` is stored
    # in Pa, but the CDS API expects hPa (`[1, …, 1000]`), so convert with `÷ hPa`.
    levels = if is_three_dimensional(meta)
        Int.(meta.dataset.pressure_levels) .÷ hPa
    elseif variable_name == "geopotential"
        :surface
    else
        nothing
    end

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
        ensure_era5cli_credentials()
        downloaded_files = CopernicusClimateDataStore.hourly(;
            variables = variable_name,
            startyear = year,
            months = month,
            days = day,
            hours = hour,
            levels = levels,
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
##### Area/bounding box utilities
#####

build_era5_area(::Nothing) = nothing

const BBOX = NumericalEarth.DataWrangling.BoundingBox

function build_era5_area(bbox::BBOX)
    # ERA5/era5cli uses (lat_max, lon_min, lat_min, lon_max) ordering
    # BoundingBox has longitude = (west, east), latitude = (south, north)

    lon = bbox.longitude
    lat = bbox.latitude

    if isnothing(lon) || isnothing(lat)
        return nothing
    end

    lon_min = lon[1]  # west
    lon_max = lon[2]  # east
    lat_min = lat[1]  # south
    lat_max = lat[2]  # north

    # Return in era5cli order: (lat_max, lon_min, lat_min, lon_max)
    return (lat = (lat_min, lat_max), lon = (lon_min, lon_max))
end

end # module NumericalEarthCopernicusClimateDataStoreExt
