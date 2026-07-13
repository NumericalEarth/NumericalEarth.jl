#####
##### FLUXNETSite dataset type
#####

"""
    FLUXNETSite(site; grouping = "FULLSET",
                      resolution = :halfhourly,
                      longitude = 0.0,
                      latitude = 0.0,
                      dir = download_FLUXNET_cache)

A handle to the [FLUXNET Shuttle](https://data.fluxnet.org) data product for a
single flux-tower `site` (e.g. `"US-Var"`), used as the `dataset` of a
[`Metadata`](@ref). The Shuttle federates the ONEFlux-processed FLUXNET products
of the regional networks (AmeriFlux, ICOS, TERN, …); each is one comma-separated
file per site and time `resolution`, named
`<PUBLISHER>_<site>_<PROCESSING-VERSION>_<grouping>_<HH|HR|DD>_<years>_<version>.csv`
(e.g. `AMF_US-Var_FLUXNET_FULLSET_HH_…`). Files are discovered in `dir` by matching
on the stable `grouping`/`resolution` tokens, so the publisher prefix and processing
version — which differ across hubs — are irrelevant.

If no matching file is present, it is downloaded from the Shuttle via the
`fluxnet-shuttle` command-line tool (see [`download_from_shuttle`](@ref)). Site
coordinates are not stored in the flux file; pass `longitude`/`latitude` if a
downstream consumer needs them.

Keyword Arguments
=================
- `grouping`: `"FULLSET"` (all variables) or `"SUBSET"` (core variables).
- `resolution`: `:halfhourly` (`HH`), `:hourly` (`HR`), or `:daily` (`DD`).
- `longitude`, `latitude`: site coordinates in degrees.
- `dir`: directory holding (or to download) the site CSV.
"""
struct FLUXNETSite
    site :: String
    grouping :: String
    resolution :: Symbol
    longitude :: Float64
    latitude :: Float64
    dir :: String
end

function FLUXNETSite(site; grouping = "FULLSET",
                           resolution = :halfhourly,
                           longitude = 0.0,
                           latitude = 0.0,
                           dir = download_FLUXNET_cache)

    resolution_token(resolution) # validate early
    return FLUXNETSite(String(site), String(grouping), Symbol(resolution),
                       Float64(longitude), Float64(latitude), String(dir))
end

const FLUXNETMetadata{D} = Metadata{<:FLUXNETSite, D}
const FLUXNETMetadatum   = Metadatum{<:FLUXNETSite}

#####
##### Variable name mapping: NumericalEarth names → ONEFlux FLUXNET column names
#####

const FLUXNET_variable_names = Dict{Symbol, String}(
    # Meteorological forcing (gap-filled `_F`)
    :air_temperature              => "TA_F",       # °C → K
    :surface_pressure             => "PA_F",       # kPa → Pa
    :wind_speed                   => "WS_F",       # m s⁻¹
    :relative_humidity            => "RH",         # %
    :vapor_pressure_deficit       => "VPD_F",      # hPa
    :precipitation                => "P_F",        # mm per averaging interval
    :incoming_shortwave_radiation => "SW_IN_F",    # W m⁻²
    :incoming_longwave_radiation  => "LW_IN_F",    # W m⁻²
    :carbon_dioxide_mole_fraction => "CO2_F_MDS",  # µmol mol⁻¹
    # Turbulent-flux targets (gap-filled `_F_MDS`; `_CORR` are energy-balance-corrected)
    :sensible_heat_flux           => "H_F_MDS",    # W m⁻²
    :latent_heat_flux             => "LE_F_MDS",   # W m⁻²
    :sensible_heat_flux_corrected => "H_CORR",     # W m⁻²
    :latent_heat_flux_corrected   => "LE_CORR",    # W m⁻²
    :friction_velocity            => "USTAR",      # m s⁻¹
    :net_radiation                => "NETRAD",     # W m⁻²
    :ground_heat_flux             => "G_F_MDS",    # W m⁻²
    # Soil state (shallowest gap-filled level)
    :soil_temperature             => "TS_F_MDS_1", # °C → K
    :soil_water_content           => "SWC_F_MDS_1",# %
)

#####
##### File discovery + cached access
#####

# Publisher- and processing-version-agnostic: match only the stable ONEFlux
# `<grouping>_<resolution>` tokens, so `AMF_…_FLUXNET_…`, `FLX_…_FLUXNET2015_…`,
# and other hubs' products are all found.
fluxnet_glob(ds::FLUXNETSite) =
    string("*_", ds.site, "_*_", ds.grouping, "_",
           resolution_token(ds.resolution), "_*.csv")

function fluxnet_path(ds::FLUXNETSite)
    pattern = fluxnet_glob(ds)
    matches = glob(pattern, ds.dir)

    if isempty(matches)
        download_from_shuttle(ds)            # fetch via the FLUXNET Shuttle
        matches = glob(pattern, ds.dir)
    end

    isempty(matches) &&
        error("No FLUXNET file matching \"$pattern\" found in \"$(ds.dir)\" after download.")

    sorted = sort(matches)
    length(sorted) > 1 &&
        @warn "Multiple FLUXNET files match \"$pattern\" in \"$(ds.dir)\"; using $(basename(first(sorted)))"

    return first(sorted)
end

fluxnet_filename(ds::FLUXNETSite) = basename(fluxnet_path(ds))

function fluxnet_file(ds::FLUXNETSite)
    path = fluxnet_path(ds)
    return get!(() -> read_fluxnet_csv(path), FLUXNET_FILE_CACHE, path)
end

fluxnet_timestamps(ds::FLUXNETSite) = first(fluxnet_file(ds))
fluxnet_columns(ds::FLUXNETSite)    = last(fluxnet_file(ds))

function fluxnet_time_index(ds::FLUXNETSite, date)
    path = fluxnet_path(ds)
    index = get!(FLUXNET_TIME_INDEX_CACHE, path) do
        Dict(t => i for (i, t) in enumerate(fluxnet_timestamps(ds)))
    end
    return get(index, DateTime(date), nothing)
end

#####
##### Metadata interface
#####

DataWrangling.metaprefix(::FLUXNETMetadata) = "FLUXNETMetadata"
DataWrangling.default_download_directory(ds::FLUXNETSite) = mkpath(ds.dir)
DataWrangling.available_variables(::FLUXNETSite) = FLUXNET_variable_names
DataWrangling.dataset_variable_name(md::FLUXNETMetadata) = FLUXNET_variable_names[md.name]

Oceananigans.location(::FLUXNETMetadata) = (Center, Center, Center)
DataWrangling.is_three_dimensional(::FLUXNETMetadata) = false
DataWrangling.default_inpainting(::FLUXNETMetadata) = nothing

Base.size(::FLUXNETSite, variable) = (1, 1, 1)

DataWrangling.metadata_epoch(ds::FLUXNETSite) = first(fluxnet_timestamps(ds))
DataWrangling.metadata_time_step(ds::FLUXNETSite) = resolution_seconds(ds.resolution)
DataWrangling.all_dates(ds::FLUXNETSite, variable) = fluxnet_timestamps(ds)

DataWrangling.longitude_interfaces(ds::FLUXNETSite) = (ds.longitude, ds.longitude)
DataWrangling.latitude_interfaces(ds::FLUXNETSite)  = (ds.latitude, ds.latitude)

# A tower is a single point: a `Flat, Flat, Flat` column carries the time series.
DataWrangling.native_grid(::FLUXNETMetadata, arch=CPU(); halo=(3, 3, 3)) =
    RectilinearGrid(arch; size=(), topology=(Flat, Flat, Flat))

DataWrangling.metadata_filename(ds::FLUXNETSite, name, date, region) = fluxnet_filename(ds)
DataWrangling.build_filename(ds::FLUXNETSite, name, dates::AbstractArray, region) = fluxnet_filename(ds)

# Resolve the site file, downloading it from the FLUXNET Shuttle if not present.
Downloads.download(md::FLUXNETMetadata) = fluxnet_path(md.dataset)

#####
##### Unit conversions
#####

struct Kilopascal end # atmospheric pressure kPa → Pa
DataWrangling.convert_units(P::FT, ::Kilopascal) where FT = P * convert(FT, 1000)

function DataWrangling.conversion_units(md::FLUXNETMetadatum)
    md.name in (:air_temperature, :soil_temperature) && return Celsius()
    md.name == :surface_pressure && return Kilopascal()
    return nothing
end

#####
##### Data retrieval — read a single sample from the parsed CSV
#####

function DataWrangling.retrieve_data(metadata::FLUXNETMetadatum)
    columns = fluxnet_columns(metadata.dataset)
    name = DataWrangling.dataset_variable_name(metadata)
    haskey(columns, name) ||
        error("FLUXNET site $(metadata.dataset.site) has no column \"$name\" for variable $(metadata.name)")

    index = fluxnet_time_index(metadata.dataset, metadata.dates)
    isnothing(index) &&
        error("Date $(metadata.dates) not found in FLUXNET site $(metadata.dataset.site)")

    return reshape([columns[name][index]], 1, 1, 1)
end

# The generic `set!`/`Field` read coordinates from a NetCDF file; FLUXNET is a CSV,
# so fill the single column directly, applying the unit conversion in the same step.
function Oceananigans.Fields.set!(target::Field, metadata::FLUXNETMetadatum; kw...)
    arch = child_architecture(target.grid)
    FT = eltype(target)
    raw = DataWrangling.retrieve_data(metadata)
    value = DataWrangling.convert_units(convert(FT, raw[1, 1, 1]), DataWrangling.conversion_units(metadata))
    interior(target) .= on_architecture(arch, reshape([value], 1, 1, 1))
    return target
end

function Oceananigans.Fields.Field(metadata::FLUXNETMetadatum, arch=CPU(); kw...)
    grid = DataWrangling.native_grid(metadata, arch)
    LX, LY, LZ = Oceananigans.location(metadata)
    field = Field{LX, LY, LZ}(grid)
    Oceananigans.Fields.set!(field, metadata; kw...)
    return field
end
