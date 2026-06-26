using ...Atmospheres: PrescribedAtmosphere, PrescribedPrecipitationFlux,
                      AtmosphereThermodynamicsParameters
using ...EarthSystemModels.InterfaceComputations: saturation_specific_humidity
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: on_architecture
using Oceananigans.Fields: CenterField, interior
using Thermodynamics: Liquid

ERA5PrescribedAtmosphere(arch::Distributed; kw...) =
    ERA5PrescribedAtmosphere(child_architecture(arch); kw...)

# ERA5 carries the 2 m dewpoint temperature, not specific humidity. The air is
# saturated at its dewpoint, so q = qˢᵃᵗ(T_dewpoint, p_surface). Evaluated
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
from the 2 m dewpoint and surface pressure (`q = qˢᵃᵗ(T_dewpoint, p)`); total
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

    ua_meta = Metadata(:eastward_velocity;    dataset, start_date, end_date, dir, region)
    va_meta = Metadata(:northward_velocity;   dataset, start_date, end_date, dir, region)
    Ta_meta = Metadata(:temperature;          dataset, start_date, end_date, dir, region)
    Td_meta = Metadata(:dewpoint_temperature; dataset, start_date, end_date, dir, region)
    pa_meta = Metadata(:surface_pressure;     dataset, start_date, end_date, dir, region)
    Fr_meta = Metadata(:total_precipitation;  dataset, start_date, end_date, dir, region)

    ua = FieldTimeSeries(ua_meta, architecture; kw...)
    va = FieldTimeSeries(va_meta, architecture; kw...)
    Ta = FieldTimeSeries(Ta_meta, architecture; kw...)
    Td = FieldTimeSeries(Td_meta, architecture; kw...)
    pa = FieldTimeSeries(pa_meta, architecture; kw...)
    Fr = FieldTimeSeries(Fr_meta, architecture; kw...)

    grid  = ua.grid
    times = ua.times
    FT    = eltype(ua)

    ℂ = isnothing(thermodynamics_parameters) ? AtmosphereThermodynamicsParameters(FT) :
                                                thermodynamics_parameters
    phase = Liquid()

    qa = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    for n in eachindex(times)
        q = KernelFunctionOperation{Center, Center, Nothing}(specific_humidity_from_dewpoint,
                                                             grid, Td[n], pa[n], ℂ, phase)
        set!(qa[n], q)
    end

    freshwater_flux = PrescribedPrecipitationFlux(rain = Fr)

    return PrescribedAtmosphere(grid, times;
                                velocities = (u = ua, v = va),
                                temperature = Ta,
                                specific_humidity = qa,
                                pressure   = pa,
                                freshwater_flux,
                                thermodynamics_parameters = ℂ,
                                surface_layer_height  = convert(FT, surface_layer_height),
                                boundary_layer_height = convert(FT, boundary_layer_height))
end

# Pressure on a `PressureLevelGrid` is the level coordinate (Pa), constant in space and time:
# `pressure[i, j, k] = pˡᵉᵛᵉˡ[k]`. Holding it as a field lets a downstream consumer interpolate it
# (in log space) at arbitrary heights via the grid's per-column geopotential — i.e. faithful ln(p)
# interpolation of the native ERA5 pressure, no hydrostatic reconstruction.
function pressure_level_field(grid, dataset, architecture)
    Nx, Ny, Nz = size(grid)
    FT = eltype(grid)
    # `dataset.pressure_levels` is sorted descending (hPa) ⇒ k=1 is the bottom (highest pressure).
    levels_Pa = on_architecture(architecture, FT.(dataset.pressure_levels) .* 100)
    pressure = CenterField(grid)
    interior(pressure) .= reshape(levels_Pa, 1, 1, Nz)
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
    pressure_level(name) = FieldTimeSeries(Metadata(name; dataset, dates, region, dir), architecture; kw...)

    ua  = pressure_level(:eastward_velocity)
    va  = pressure_level(:northward_velocity)
    Ta  = pressure_level(:temperature)
    qva = pressure_level(:specific_humidity)
    qcl = pressure_level(:specific_cloud_liquid_water_content)
    qrw = pressure_level(:specific_rain_water_content)
    qci = pressure_level(:specific_cloud_ice_water_content)
    qsw = pressure_level(:specific_snow_water_content)

    grid  = Ta.grid    # native ERA5 PressureLevelGrid
    times = Ta.times

    return PrescribedAtmosphere(grid, times;
                                velocities = (u = ua, v = va),
                                temperature = Ta,
                                specific_humidity = qva,
                                microphysical_variables = (qᶜˡ = qcl, qʳ = qrw, qᶜⁱ = qci, qˢ = qsw),
                                pressure = pressure_level_field(grid, dataset, architecture),
                                freshwater_flux = nothing,
                                thermodynamics_parameters)
end
