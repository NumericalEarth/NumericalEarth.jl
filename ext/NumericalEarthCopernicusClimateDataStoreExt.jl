module NumericalEarthCopernicusClimateDataStoreExt

using NumericalEarth
using CopernicusClimateDataStore: CopernicusClimateDataStore
using Downloads: Downloads
using Dates: Dates
using Oceananigans.DistributedComputations: @root

using NumericalEarth.DataWrangling: MetadataSet, available_variables, metadata_filename, metadata_path
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
##### One era5cli invocation per calendar-month batch covers every pending variable and
##### datetime at once: era5cli submits one CDS request per variable (downloaded with
##### concurrent threads) and expands `months`/`days`/`hours` into a Cartesian product
##### server-side. Each returned per-variable multi-step NetCDF is split locally into the
##### per-datetime files the readers expect; splitting matches datetimes against the file's
##### own time coordinate, so the product's over-fetch is harmless.
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

# era5cli defaults to the single-levels (surface) product unless `--levels` is given — so a
# pressure-level request without levels silently returns a surface field (e.g. `u10` for
# `u_component_of_wind`). Pass the dataset's pressure levels for 3-D datasets; disambiguate the
# single-level `geopotential`/`topography` (surface geopotential, exists on both) with `:surface`;
# ordinary single-level variables keep era5cli's default (`nothing`). `pressure_levels` is stored
# in Pa, but the CDS API expects hPa (`[1, …, 1000]`), so convert with `÷ hPa`.
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
##### Area/bounding box utilities
#####

const BBOX = NumericalEarth.DataWrangling.BoundingBox

era5cli_request_area(region, dataset, name::Symbol) = era5cli_request_area(region, dataset, [name])
era5cli_request_area(::Nothing, dataset, names::Vector{Symbol}) = nothing

# The native grid is built by center-bracketing `restrict`, which can reach one cell past a
# boundary-aligned edge. Fetch two native cells of margin so the downloaded file always covers
# the grid the data is interpolated onto (cf. `era5_request_area` in NumericalEarthCDSAPIExt);
# the margin also absorbs era5cli's rounding of area coordinates to two decimals. Over-fetching
# is harmless: `restrict` selects the exact cells from the larger file. The coarsest native
# grid among the requested variables sets the padding (wave variables live on 0.5°, the rest 0.25°).
function era5cli_request_area(bbox::BBOX, dataset, names::Vector{Symbol})
    (isnothing(bbox.longitude) || isnothing(bbox.latitude)) && return nothing
    Δλ = maximum(360 / size(dataset, name)[1] for name in names)
    Δφ = maximum(180 / size(dataset, name)[2] for name in names)
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

#####
##### Copernicus land surface albedo, through the package's native CDS client (≥ 0.2).
##### This backend only supplies `retrieve`; the request construction, extraction, and
##### repacking live in the CopernicusLandAlbedo module.
#####

using NumericalEarth.DataWrangling.CopernicusLandAlbedo: ALBEDO_CDS_PRODUCT,
                                                         CopernicusAlbedoDatasetMetadata,
                                                         download_albedo_dekads!

function Downloads.download(metadata::CopernicusAlbedoDatasetMetadata; kwargs...)
    isdefined(CopernicusClimateDataStore, :retrieve) || throw(ArgumentError(
        "Downloading the Copernicus land albedo needs CopernicusClimateDataStore ≥ 0.2, " *
        "whose native CDS client provides `retrieve`."))

    return download_albedo_dekads!(metadata; kwargs...) do request, path
        CopernicusClimateDataStore.retrieve(ALBEDO_CDS_PRODUCT, request, path)
    end
end

end # module NumericalEarthCopernicusClimateDataStoreExt

