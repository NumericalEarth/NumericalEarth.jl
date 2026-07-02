using ...Atmospheres: PrescribedAtmosphere, PrescribedPrecipitationFlux,
                      AtmosphereThermodynamicsParameters
using ...EarthSystemModels.InterfaceComputations: saturation_specific_humidity
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: on_architecture
using Oceananigans.Fields: CenterField, interior
using Thermodynamics: Liquid

# `ERA5PrescribedAtmosphere` is a type alias for a `PrescribedAtmosphere` whose `source` is any ERA5
# dataset; the constructor methods below build one (the alias is both the dispatch type and the builder).
const ERA5PrescribedAtmosphere = PrescribedAtmosphere{<:ERA5Dataset}

ERA5PrescribedAtmosphere(arch::Distributed; kw...) =
    ERA5PrescribedAtmosphere(child_architecture(arch); kw...)

# ERA5 carries the 2 m dewpoint temperature, not specific humidity. The air is
# saturated at its dewpoint, so qᵛ = qᵛ⁺(Tᵈ, pˢ). Evaluated
# pointwise (GPU-safe, via a `KernelFunctionOperation`) with the same
# thermodynamics the flux solver uses, so the prescribed q is self-consistent.
@inline function specific_humidity_from_dewpoint(i, j, k, grid, dewpoint, pressure, ℂ, phase)
    @inbounds Tᵈ = dewpoint[i, j, k]
    @inbounds pˢ = pressure[i, j, k]
    return saturation_specific_humidity(ℂ, Tᵈ, pˢ, phase)
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
                             boundary_layer_height = 512,  # meters
                             thermodynamics_parameters = nothing,
                             region = nothing,
                             other_kw...)

Return a [`PrescribedAtmosphere`](@ref) representing ERA5 single-level reanalysis,
suitable for regional hindcast forcing. Eastward/northward 10 m winds, 2 m
temperature, and surface pressure are loaded directly; specific humidity is derived
from the 2 m dewpoint and surface pressure (`qᵛ = qᵛ⁺(Tᵈ, pˢ)`); total
precipitation is converted from hourly-accumulated depth (m) to a mass flux
(kg m⁻² s⁻¹) at load time and wrapped in a `PrescribedPrecipitationFlux`.

`region` (a `BoundingBox`) restricts the download and the native grid to a
sub-domain; the coupled model interpolates the native-resolution atmosphere onto
the exchange grid. Pass `thermodynamics_parameters` to share a specific
thermodynamics with the rest of the model (defaults to
`AtmosphereThermodynamicsParameters` at the data's float type).
"""
function ERA5PrescribedAtmosphere(architecture = CPU();
                                  dataset = ERA5HourlySingleLevel(),
                                  start_date = first_date(dataset, :temperature),
                                  end_date = last_date(dataset, :temperature),
                                  dir = download_ERA5_cache,
                                  time_indices_in_memory = 24,
                                  time_indexing = Cyclical(),
                                  surface_layer_height = 10,
                                  boundary_layer_height = 512,
                                  thermodynamics_parameters = nothing,
                                  region = nothing,
                                  other_kw...)

    kw = (; time_indexing, time_indices_in_memory)
    kw = merge(kw, other_kw)

    era5_fts(name) = FieldTimeSeries(Metadata(name; dataset, start_date, end_date, dir, region), architecture; kw...)

    u    = era5_fts(:eastward_velocity)
    v    = era5_fts(:northward_velocity)
    T    = era5_fts(:temperature)
    Tᵈ   = era5_fts(:dewpoint_temperature)
    p    = era5_fts(:surface_pressure)
    rain = era5_fts(:total_precipitation)

    grid  = u.grid
    times = u.times
    FT    = eltype(u)

    ℂ = isnothing(thermodynamics_parameters) ? AtmosphereThermodynamicsParameters(FT) :
                                                thermodynamics_parameters
    phase = Liquid()

    qᵛ = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    for n in eachindex(times)
        q = KernelFunctionOperation{Center, Center, Nothing}(specific_humidity_from_dewpoint,
                                                             grid, Tᵈ[n], p[n], ℂ, phase)
        set!(qᵛ[n], q)
    end

    precipitation_flux = PrescribedPrecipitationFlux(; rain)

    return PrescribedAtmosphere(grid, times;
                                source = dataset,
                                velocities = (; u, v),
                                temperature = T,
                                specific_humidity = qᵛ,
                                pressure = p,
                                precipitation_flux,
                                thermodynamics_parameters = ℂ,
                                surface_layer_height  = convert(FT, surface_layer_height),
                                boundary_layer_height = convert(FT, boundary_layer_height))
end

# Pressure on a `PressureLevelGrid` is the level coordinate (Pa), constant in space and time:
# `pressure[i, j, k] = pˡᵉᵛᵉˡ[k]`. Holding it as a field lets a downstream consumer interpolate it
# (in log space) at arbitrary heights via the grid's per-column geopotential — i.e. faithful ln(p)
# interpolation of the native ERA5 pressure, no hydrostatic reconstruction.
function pressure_level_field(grid, dataset, architecture)
    FT = eltype(grid)
    # `dataset.pressure_levels` is sorted descending (hPa) ⇒ k=1 is the bottom (highest pressure).
    pˡᵉᵛᵉˡ = on_architecture(architecture, FT.(dataset.pressure_levels) .* 100)
    pressure = CenterField(grid)
    interior(pressure) .= reshape(pˡᵉᵛᵉˡ, 1, 1, Nz)
    fill_halo_regions!(pressure)
    return pressure
end

"""
    ERA5PrescribedAtmosphere(bounding_box::BoundingBox, dates;
                             architecture = CPU(),
                             dataset = ERA5HourlyPressureLevels(),
                             dir = download_ERA5_cache,
                             time_indices_in_memory = length(dates),
                             thermodynamics_parameters = nothing,
                             other_kw...)

Return a 3-D [`PrescribedAtmosphere`](@ref) built from ERA5 **pressure-level** reanalysis over
`bounding_box` at the requested `dates`, on ERA5's **native grid** — a `PressureLevelGrid` at the
reanalysis' native horizontal resolution with a geopotential-height-aware pressure-level vertical.
Each variable loads natively (no pre-regridding); a downstream model (e.g. a `NestedSimulation`
child) interpolates the parent onto its own grid on the fly.

The atmosphere holds eastward/northward `velocities`, `temperature`, `specific_humidity`,
`microphysical_variables = (; qᶜˡ, qʳ, qᶜⁱ, qˢ)` (cloud liquid/ice + rain/snow water content), and
`pressure` (the level coordinate, via [`pressure_level_field`](@ref)). Use as the parent of a
[`NestedSimulation`](@ref).
"""
function ERA5PrescribedAtmosphere(bounding_box::BoundingBox, dates;
                                  architecture = CPU(),
                                  dataset = ERA5HourlyPressureLevels(),
                                  dir = download_ERA5_cache,
                                  time_indices_in_memory = length(dates),
                                  thermodynamics_parameters = nothing,
                                  other_kw...)

    region = bounding_box
    kw = merge((; time_indices_in_memory), other_kw)

    # Each loads on ERA5's native PressureLevelGrid (per-snapshot geopotential ⇒ true per-column heights).
    era5_fts(name) = FieldTimeSeries(Metadata(name; dataset, dates, region, dir), architecture; kw...)

    u   = era5_fts(:eastward_velocity)
    v   = era5_fts(:northward_velocity)
    T   = era5_fts(:temperature)
    qᵛ  = era5_fts(:specific_humidity)
    qᶜˡ = era5_fts(:specific_cloud_liquid_water_content)
    qʳ  = era5_fts(:specific_rain_water_content)
    qᶜⁱ = era5_fts(:specific_cloud_ice_water_content)
    qˢ  = era5_fts(:specific_snow_water_content)

    grid  = T.grid    # native ERA5 PressureLevelGrid
    times = T.times

    return PrescribedAtmosphere(grid, times;
                                source = dataset,
                                velocities = (; u, v),
                                temperature = T,
                                specific_humidity = qᵛ,
                                microphysical_variables = (; qᶜˡ, qʳ, qᶜⁱ, qˢ),
                                pressure = pressure_level_field(grid, dataset, architecture),
                                thermodynamics_parameters)
end
