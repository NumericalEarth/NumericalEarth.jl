module NumericalEarthCopernicusClimateDataStoreExt

using NumericalEarth
using CopernicusClimateDataStore: CopernicusClimateDataStore
using Downloads: Downloads
using Dates: Dates
using Oceananigans.DistributedComputations: @root

using NumericalEarth.DataWrangling: available_variables, metadata_filename, metadata_path
using NumericalEarth.DataWrangling.ERA5: ERA5Dataset, ERA5PressureLevelsDataset,
                                         ERA5Metadata, ERA5Metadatum, hPa,
                                         batch_datetimes_for_cds, coord_vars, nc_varnames,
                                         split_era5_nc_by_datetime, ERA5_TIME_DIMNAMES

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

#####
##### Batched downloads — same strategy as NumericalEarthCDSAPIExt
#####
##### One era5cli invocation (= one CDS request) per variable per calendar-month batch covers
##### every pending datetime at once; era5cli expands `months`/`days`/`hours` into a Cartesian
##### product server-side, and the returned multi-step NetCDF is split locally into the
##### per-datetime files the readers expect. Splitting matches datetimes against the file's own
##### time coordinate, so the product's over-fetch is harmless.
#####

"""
    Downloads.download(metadata::ERA5Metadata; skip_existing=true, cleanup=true, threads=1, kwargs...)

Download ERA5 data for every date in `metadata` using `era5cli` through the
CopernicusClimateDataStore package, one CDS request per calendar-month batch,
returning the paths of the per-datetime files.

# Keyword Arguments
- `skip_existing`: Skip datetimes whose files already exist (default: `true`).
- `cleanup`: Remove the temporary multi-step NetCDF after splitting (default: `true`).
- `threads`: Number of era5cli download threads (default: `1`).
- Additional keyword arguments are passed to `CopernicusClimateDataStore.hourly`.

# Environment Setup
Before downloading, you must:
1. Create an account at https://cds.climate.copernicus.eu/
2. Accept the Terms of Use for the ERA5 dataset on the dataset page
3. Set up your API credentials in `~/.cdsapirc`

See https://cds.climate.copernicus.eu/how-to-api for details.
"""
function Downloads.download(metadata::ERA5Metadata;
                            skip_existing = true,
                            cleanup = true,
                            threads = 1,
                            additional_kw...)

    dates = metadata.dates isa AbstractVector ? metadata.dates : [metadata.dates]
    batches = batch_datetimes_for_cds(dates, metadata.dataset, 1)

    paths = String[]
    for batch in batches
        append!(paths, download_era5cli_month(metadata.name, metadata.dataset, batch;
                                              region = metadata.region,
                                              dir = metadata.dir,
                                              skip_existing, cleanup, threads,
                                              additional_kw...))
    end

    return paths
end

"""
    Downloads.download(meta::ERA5Metadatum; skip_existing=true, kwargs...)

Download ERA5 data for a single date/time using `era5cli` through the
CopernicusClimateDataStore package, returning the path of the downloaded file.
"""
function Downloads.download(meta::ERA5Metadatum;
                            skip_existing = true,
                            cleanup = true,
                            threads = 1,
                            additional_kw...)

    download_era5cli_month(meta.name, meta.dataset, [meta.dates];
                           region = meta.region,
                           dir = meta.dir,
                           skip_existing, cleanup, threads,
                           additional_kw...)

    return metadata_path(meta)
end

# era5cli defaults to the single-levels (surface) product unless `--levels` is given — so a
# pressure-level request without levels silently returns a surface field (e.g. `u10` for
# `u_component_of_wind`). Pass the dataset's pressure levels for 3-D datasets; disambiguate the
# single-level `geopotential`/`topography` (surface geopotential, exists on both) with `:surface`;
# ordinary single-level variables keep era5cli's default (`nothing`). `pressure_levels` is stored
# in Pa, but the CDS API expects hPa (`[1, …, 1000]`), so convert with `÷ hPa`.
era5cli_levels(dataset::ERA5PressureLevelsDataset, variable_name) = Int.(dataset.pressure_levels) .÷ hPa
era5cli_levels(dataset::ERA5Dataset, variable_name) = variable_name == "geopotential" ? :surface : nothing

function download_era5cli_month(name, dataset, dates;
                                region, dir,
                                skip_existing = true,
                                cleanup = true,
                                threads = 1,
                                additional_kw...)

    dt_path_pairs = [(dt, joinpath(dir, metadata_filename(dataset, name, dt, region)))
                     for dt in dates]

    pending = skip_existing ? filter(dt_path -> !isfile(dt_path[2]), dt_path_pairs) : dt_path_pairs
    isempty(pending) && return map(dt_path -> dt_path[2], dt_path_pairs)

    mkpath(dir)

    variable_name = available_variables(dataset)[name]
    sorted_dts = sort(unique([dt for (dt, _) in pending]))
    dt0 = first(sorted_dts)
    outputprefix = "_tmp_era5cli_$(variable_name)_$(Dates.year(dt0))$(lpad(Dates.month(dt0), 2, '0'))"

    @root begin
        ensure_era5cli_credentials()

        # A stale file left by an interrupted run would hide the fresh download from
        # `hourly`'s before/after directory diff — clear matching leftovers first.
        for leftover in filter(f -> startswith(f, outputprefix), readdir(dir))
            rm(joinpath(dir, leftover); force=true)
        end

        downloaded_files = CopernicusClimateDataStore.hourly(;
            variables = variable_name,
            startyear = Dates.year(dt0),
            months = unique(Dates.month.(sorted_dts)),
            days = unique(Dates.day.(sorted_dts)),
            hours = unique(Dates.hour.(sorted_dts)),
            levels = era5cli_levels(dataset, variable_name),
            area = era5cli_request_area(region, dataset, name),
            format = "netcdf",
            outputprefix,
            overwrite = true,
            threads,
            splitmonths = false,
            directory = dir,
            additional_kw...)

        # One variable, one year, splitmonths=false → era5cli produces exactly one file
        length(downloaded_files) == 1 ||
            error("Expected era5cli to deliver one NetCDF for $variable_name; ",
                  "got $(length(downloaded_files)): $downloaded_files")

        file = first(downloaded_files)
        nc_triples = [(nc_varnames(dataset)[name], dt, path) for (dt, path) in pending]
        split_era5_nc_by_datetime(file, nc_triples, coord_vars(dataset), ERA5_TIME_DIMNAMES)
        cleanup && rm(file; force=true)
    end

    return map(dt_path -> dt_path[2], dt_path_pairs)
end

#####
##### Area/bounding box utilities
#####

const BBOX = NumericalEarth.DataWrangling.BoundingBox

era5cli_request_area(::Nothing, dataset, name) = nothing

# The native grid is built by center-bracketing `restrict`, which can reach one cell past a
# boundary-aligned edge. Fetch two native cells of margin so the downloaded file always covers
# the grid the data is interpolated onto (cf. `era5_request_area` in NumericalEarthCDSAPIExt);
# the margin also absorbs era5cli's rounding of area coordinates to two decimals. Over-fetching
# is harmless: `restrict` selects the exact cells from the larger file.
function era5cli_request_area(bbox::BBOX, dataset, name)
    (isnothing(bbox.longitude) || isnothing(bbox.latitude)) && return nothing
    Nx, Ny, _ = size(dataset, name)
    Δλ = 360 / Nx
    Δφ = 180 / Ny
    lon = bbox.longitude
    lat = bbox.latitude
    padded = BBOX(longitude = (lon[1] - 2Δλ, lon[2] + 2Δλ),
                  latitude  = (max(lat[1] - 2Δφ, -90), min(lat[2] + 2Δφ, 90)))
    return build_era5_area(padded)
end

build_era5_area(::Nothing) = nothing

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
