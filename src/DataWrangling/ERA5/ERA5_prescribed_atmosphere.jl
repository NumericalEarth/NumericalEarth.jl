using ...Atmospheres: Atmospheres, PrescribedPrecipitationFlux, AtmosphereThermodynamicsParameters
using ...EarthSystemModels.InterfaceComputations: saturation_specific_humidity
using Adapt: Adapt
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: on_architecture
using Oceananigans.Fields: CenterField, interior
using Oceananigans.OutputReaders: OutputReaders, AbstractInMemoryBackend, InMemory
using Thermodynamics: Liquid

const ERA5PrescribedAtmosphere = Atmospheres.PrescribedAtmosphere{<:ERA5Dataset}

ERA5PrescribedAtmosphere(arch::Distributed; kw...) = ERA5PrescribedAtmosphere(child_architecture(arch); kw...)

# ERA5 carries the 2 m dewpoint temperature, not specific humidity. The air is
# saturated at its dewpoint, so qᵛ = qᵛ⁺(Tᵈ, pˢ).
@inline function specific_humidity_from_dewpoint(i, j, k, grid, dewpoint, pressure, ℂ, phase)
    @inbounds Tᵈ = dewpoint[i, j, k]
    @inbounds pˢ = pressure[i, j, k]
    return saturation_specific_humidity(ℂ, Tᵈ, pˢ, phase)
end

# A `FieldTimeSeries` backend that derives each in-memory window of specific humidity from the
# same window of the dewpoint and pressure series, so qᵛ holds `length` snapshots like the
# other ERA5 fields rather than the whole record.
struct DewpointHumidityBackend{D, P, C} <: AbstractInMemoryBackend{Int}
    start :: Int
    length :: Int
    dewpoint :: D
    pressure :: P
    thermodynamics_parameters :: C
end

OutputReaders.new_backend(b::DewpointHumidityBackend, start, length) =
    DewpointHumidityBackend(start, length, b.dewpoint, b.pressure, b.thermodynamics_parameters)

Adapt.adapt_structure(to, b::DewpointHumidityBackend) = InMemory(b.start, b.length)

const DewpointHumidityFTS = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:DewpointHumidityBackend}

function Oceananigans.Fields.set!(qᵛ::DewpointHumidityFTS, backend = qᵛ.backend)
    Tᵈ, p, ℂ = backend.dewpoint, backend.pressure, backend.thermodynamics_parameters
    for n in time_indices(qᵛ)
        q = KernelFunctionOperation{Center, Center, Nothing}(specific_humidity_from_dewpoint, qᵛ.grid, Tᵈ[n], p[n], ℂ, Liquid())
        set!(qᵛ[n], q)
    end
    return nothing
end

"""
    ERA5PrescribedAtmosphere([architecture = CPU()];
                             dataset = ERA5HourlySingleLevel(),
                             start_date = first_date(dataset, :temperature),
                             end_date = last_date(dataset, :temperature),
                             dir = download_ERA5_cache,
                             time_indices_in_memory = 24,
                             time_indexing = Cyclical(),
                             surface_layer_height = 10,    # meters
                             boundary_layer_height = nothing,
                             thermodynamics_parameters = nothing,
                             region = nothing,
                             other_kw...)

Return a [`PrescribedAtmosphere`](@ref) representing ERA5 single-level reanalysis, suitable for regional hindcast forcing.
Eastward/northward 10 m winds, 2 m temperature, and surface pressure are loaded directly; specific humidity is derived from
the 2 m dewpoint and surface pressure (`qᵛ = qᵛ⁺(Tᵈ, pˢ)`) one in-memory window at a time; total precipitation is
converted from hourly-accumulated depth (m) to a mass flux (kg m⁻² s⁻¹) at load time and wrapped in a
`PrescribedPrecipitationFlux`. The boundary-layer height is ERA5's diagnosed `boundary_layer_height` unless a number (m)
is passed.

`region` (a `BoundingBox`) restricts the download and the native grid to a sub-domain; the coupled model interpolates the
native-resolution atmosphere onto the exchange grid. Pass `thermodynamics_parameters` to share a specific thermodynamics with
the rest of the model (defaults to `AtmosphereThermodynamicsParameters` at the data's float type).
"""
function ERA5PrescribedAtmosphere(architecture = CPU();
                                  dataset = ERA5HourlySingleLevel(),
                                  start_date = first_date(dataset, :temperature),
                                  end_date = last_date(dataset, :temperature),
                                  dir = download_ERA5_cache,
                                  time_indices_in_memory = 24,
                                  time_indexing = Cyclical(),
                                  surface_layer_height = 10,
                                  boundary_layer_height = nothing,
                                  thermodynamics_parameters = nothing,
                                  region = nothing,
                                  tracers = NamedTuple(),
                                  other_kw...)

    kw = (; time_indexing, time_indices_in_memory)
    kw = merge(kw, other_kw)

    # Download every variable up front in one bundle: backends batch a `MetadataSet`
    # across variables (one CDS request, or one era5cli invocation with concurrent
    # per-variable requests), so the `FieldTimeSeries` below find their files cached.
    mset = MetadataSet(:eastward_velocity, :northward_velocity, :temperature,
                       :dewpoint_temperature, :surface_pressure, :total_precipitation,
                       :boundary_layer_height;
                       dataset, start_date, end_date, dir, region)
    Downloads.download(mset)

    era5_fts(name) = FieldTimeSeries(Metadata(name; dataset, start_date, end_date, dir, region), architecture; kw...)

    u    = era5_fts(:eastward_velocity)
    v    = era5_fts(:northward_velocity)
    T    = era5_fts(:temperature)
    Tᵈ   = era5_fts(:dewpoint_temperature)
    p    = era5_fts(:surface_pressure)
    rain = era5_fts(:total_precipitation)
    h_bℓ = isnothing(boundary_layer_height) ? era5_fts(:boundary_layer_height) : boundary_layer_height

    grid  = u.grid
    times = u.times
    FT    = eltype(u)

    ℂ = isnothing(thermodynamics_parameters) ? AtmosphereThermodynamicsParameters(FT) :
                                                thermodynamics_parameters

    backend = DewpointHumidityBackend(1, length(Tᵈ.backend), Tᵈ, p, ℂ)
    qᵛ = FieldTimeSeries{Center, Center, Nothing}(grid, times; backend, time_indexing = Tᵈ.time_indexing)
    set!(qᵛ)

    precipitation_flux = PrescribedPrecipitationFlux(; rain)

    return Atmospheres.PrescribedAtmosphere(grid, times;
                                            source = dataset,
                                            velocities = (; u, v),
                                            temperature = T,
                                            specific_humidity = qᵛ,
                                            pressure = p,
                                            precipitation_flux,
                                            thermodynamics_parameters = ℂ,
                                            surface_layer_height  = convert(FT, surface_layer_height),
                                            boundary_layer_height = h_bℓ,
                                            tracers)
end

# Pressure on a `PressureLevelGrid` is the level coordinate (Pa), constant in space and time
function pressure_level_field(grid, dataset, architecture)
    FT = eltype(grid)
    pˡᵉᵛᵉˡ = on_architecture(architecture, FT.(dataset.pressure_levels))
    pressure = CenterField(grid)
    Nz = length(dataset.pressure_levels)
    interior(pressure) .= reshape(pˡᵉᵛᵉˡ, 1, 1, Nz)
    fill_halo_regions!(pressure)
    return pressure
end

"""
    ERA5PrescribedAtmosphere(bounding_box::BoundingBox, dates;
                             architecture = CPU(),
                             dataset = ERA5HourlyPressureLevels(),
                             dir = download_ERA5_cache,
                             time_indices_in_memory = nothing,
                             thermodynamics_parameters = nothing,
                             other_kw...)

Return a 3-D [`PrescribedAtmosphere`](@ref) built from ERA5 **pressure-level** reanalysis over `bounding_box` at the 
requested `dates` — a range or vector of dates, or a `(start_date, end_date)` tuple that expands to the dataset's native
(hourly or monthly) cadence — on ERA5's **native grid**: a `PressureLevelGrid` at the reanalysis' native horizontal resolution
with a **time-varying** geopotential-height vertical (each pressure level's height follows the reanalysis as the atmosphere's
clock advances). Each variable loads natively and raw (per pressure level, no vertical remap); a downstream model (e.g. a
`NestedSimulation` child) interpolates the parent onto its own grid on the fly, at the current heights.
`time_indices_in_memory` defaults to all dates.

The atmosphere holds eastward/northward `velocities`, `temperature`, `specific_humidity`, `microphysical_variables = (; qᶜˡ, qʳ, qᶜⁱ, qˢ)`
(cloud liquid/ice + rain/snow water content), and `pressure` (the level coordinate, via [`pressure_level_field`](@ref)).
Use as the parent of a [`NestedSimulation`](@ref).
"""
function ERA5PrescribedAtmosphere(bounding_box::BoundingBox, dates;
                                  architecture = CPU(),
                                  dataset = ERA5HourlyPressureLevels(),
                                  dir = download_ERA5_cache,
                                  time_indices_in_memory = nothing,
                                  thermodynamics_parameters = nothing,
                                  other_kw...)

    region = bounding_box
    dates = DataWrangling.expand_dates(dataset, :temperature, dates)
    time_indices_in_memory = something(time_indices_in_memory, length(dates))

    # One up-front bundle download (including the geopotential the vertical discretization
    # needs) so the per-variable loads below find their files cached.
    mset = MetadataSet(:temperature, :eastward_velocity, :northward_velocity,
                       :specific_humidity, :specific_cloud_liquid_water_content,
                       :specific_rain_water_content, :specific_cloud_ice_water_content,
                       :specific_snow_water_content, :geopotential;
                       dataset, dates, region, dir)
    Downloads.download(mset)

    # One clock drives both the atmosphere's own time and the pressure levels' time-varying
    # geopotential heights. `time_step!(atmosphere, Δt)` advances it, so the grid geometry a child
    # interpolates over follows the reanalysis in time.
    temperature_metadata = Metadata(:temperature; dataset, dates, region, dir)
    FT = eltype(temperature_metadata)
    clock = Clock{FT}(time = zero(FT))

    # Build the native PressureLevelGrid once, with a geopotential `TimeSeriesInterpolation` bound to
    # `clock`. `_with_z` carries the fully built vertical only into this transient grid construction —
    # the atmosphere's `source` stays the plain dataset (product identity only).
    plvd = per_column_geopotential_discretization(temperature_metadata; clock)
    grid = native_grid(Metadata(:temperature; dataset = _with_z(dataset, plvd), dates, region, dir), architecture)

    era5_fts(name) = era5_native_pressure_fts(Metadata(name; dataset, dates, region, dir), grid; time_indices_in_memory, other_kw...)

    u   = era5_fts(:eastward_velocity)
    v   = era5_fts(:northward_velocity)
    T   = era5_fts(:temperature)
    qᵛ  = era5_fts(:specific_humidity)
    qᶜˡ = era5_fts(:specific_cloud_liquid_water_content)
    qʳ  = era5_fts(:specific_rain_water_content)
    qᶜⁱ = era5_fts(:specific_cloud_ice_water_content)
    qˢ  = era5_fts(:specific_snow_water_content)

    times = T.times

    return Atmospheres.PrescribedAtmosphere(grid, times;
                                            clock,
                                            source = dataset,
                                            velocities = (; u, v),
                                            temperature = T,
                                            specific_humidity = qᵛ,
                                            microphysical_variables = (; qᶜˡ, qʳ, qᶜⁱ, qˢ),
                                            pressure = pressure_level_field(grid, dataset, architecture),
                                            thermodynamics_parameters)
end

"""
    PrescribedAtmosphere(bounding_box, dates, dataset::ERA5PressureLevelsDataset; kw...)

Dataset-dispatched constructor: build an [`ERA5PrescribedAtmosphere`](@ref) over `bounding_box`
at `dates` on `dataset`'s native grid. Keyword arguments flow to `ERA5PrescribedAtmosphere`.
"""
Atmospheres.PrescribedAtmosphere(bounding_box::BoundingBox, dates, dataset::ERA5PressureLevelsDataset; kw...) =
    ERA5PrescribedAtmosphere(bounding_box, dates; dataset, kw...)
