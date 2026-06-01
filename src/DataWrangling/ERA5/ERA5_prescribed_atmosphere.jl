using ...Atmospheres: PrescribedAtmosphere, PrescribedPrecipitationFlux,
                      AtmosphereThermodynamicsParameters
using ...EarthSystemModels.InterfaceComputations: saturation_specific_humidity
using Oceananigans.AbstractOperations: KernelFunctionOperation
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
                                tracers    = (T = Ta, q = qa),
                                pressure   = pa,
                                freshwater_flux,
                                thermodynamics_parameters = ℂ,
                                surface_layer_height  = convert(FT, surface_layer_height),
                                boundary_layer_height = convert(FT, boundary_layer_height))
end
