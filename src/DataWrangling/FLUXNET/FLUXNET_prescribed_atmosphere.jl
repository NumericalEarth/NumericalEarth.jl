"""
    fluxnet_specific_humidity_fts(site, Ta, pa, ℂ; start_date, end_date, max_gap)

Build a `FieldTimeSeries` of specific humidity (kg kg⁻¹) for `site` from air
temperature `Ta` (K) and pressure `pa` (Pa). Relative humidity is read directly
(`RH`, %) when present; otherwise it is recovered from the vapor pressure deficit
`VPD_F` (hPa) as `RH = 1 − VPD / qᵛ⁺(Ta)`. Specific humidity then follows from
`Thermodynamics.q_vap_from_RH` with the `Liquid()` saturation curve.
"""
function fluxnet_specific_humidity_fts(site, Ta, pa, ℂ; start_date, end_date, max_gap)
    grid = Ta.grid
    columns = fluxnet_columns(site)
    qa = FieldTimeSeries{Center, Center, Nothing}(grid, Ta.times)

    has_relative_humidity = haskey(columns, "RH") && any(!isnan, columns["RH"])

    if has_relative_humidity
        RHa = fluxnet_field_time_series(site, :relative_humidity, grid; start_date, end_date, max_gap)
        RH = parent(RHa) ./ 100
    else
        VPDa = fluxnet_field_time_series(site, :vapor_pressure_deficit, grid; start_date, end_date, max_gap)
        esat = saturation_vapor_pressure.(Ref(ℂ), parent(Ta), Ref(Liquid()))
        RH = clamp.(1 .- (parent(VPDa) .* 100) ./ esat, 0, 1) # VPD hPa → Pa
    end

    parent(qa) .= q_vap_from_RH.(Ref(ℂ), parent(pa), parent(Ta), RH, Ref(Liquid()))
    max_gap > 0 && fill_gaps!(qa; max_gap)
    return qa
end

"""
    FLUXNETPrescribedAtmosphere(site::FLUXNETSite, architecture = CPU(), FT = Float64;
                                start_date = first_date(site, :air_temperature),
                                end_date = last_date(site, :air_temperature),
                                surface_layer_height = 10,
                                max_gap = 48,
                                thermodynamics_parameters = nothing)

Construct a [`PrescribedAtmosphere`](@ref) from a FLUXNET flux tower's
meteorological record on a single-column `Flat, Flat, Flat` grid, suitable for
forcing a single-column land model.

Air temperature (`TA_F`), pressure (`PA_F`), and wind speed (`WS_F`) are loaded
directly; specific humidity is derived from `RH`/`VPD_F` (see
[`fluxnet_specific_humidity_fts`](@ref)); precipitation (`P_F`, mm per averaging
interval) is converted to a mass flux (kg m⁻² s⁻¹). Because towers report a scalar
wind *speed*, it is placed in the eastward component with `v = 0`.

Keyword Arguments
=================
- `start_date`, `end_date`: time range to load.
- `surface_layer_height`: tower measurement height in meters.
- `max_gap`: maximum gap length (in time steps) to fill by linear interpolation.
- `thermodynamics_parameters`: shared thermodynamics (defaults to
  `AtmosphereThermodynamicsParameters` at float type `FT`).
"""
function FLUXNETPrescribedAtmosphere(site::FLUXNETSite, architecture = CPU(), FT = Float64;
                                     start_date = first_date(site, :air_temperature),
                                     end_date = last_date(site, :air_temperature),
                                     surface_layer_height = 10,
                                     max_gap = 48,
                                     thermodynamics_parameters = nothing)

    grid = RectilinearGrid(architecture, FT; size=(), topology=(Flat, Flat, Flat))
    fts(name) = fluxnet_field_time_series(site, name, grid; start_date, end_date, max_gap)

    Ta = fts(:air_temperature) # K
    pa = fts(:surface_pressure) # Pa
    ua = fts(:wind_speed)       # m s⁻¹, placed in the eastward component

    va = FieldTimeSeries{Center, Center, Nothing}(grid, ua.times)
    parent(va) .= 0

    ℂ = isnothing(thermodynamics_parameters) ? AtmosphereThermodynamicsParameters(FT) :
                                                thermodynamics_parameters
    qa = fluxnet_specific_humidity_fts(site, Ta, pa, ℂ; start_date, end_date, max_gap)

    rain = fts(:precipitation) # mm per averaging interval
    parent(rain) ./= resolution_seconds(site.resolution) # mm/interval → kg m⁻² s⁻¹

    return PrescribedAtmosphere(grid, ua.times;
                                velocities = (u = ua, v = va),
                                tracers = (T = Ta, q = qa),
                                pressure = pa,
                                freshwater_flux = PrescribedPrecipitationFlux(; rain),
                                thermodynamics_parameters = ℂ,
                                surface_layer_height = convert(FT, surface_layer_height))
end
