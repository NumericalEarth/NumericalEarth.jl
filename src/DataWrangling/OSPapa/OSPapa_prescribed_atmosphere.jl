"""
    ospapa_specific_humidity_fts(RHa, Ta, Pa, params)

Build a `FieldTimeSeries` of specific humidity (kg/kg) from OS Papa relative humidity
(in %), air temperature (K), and pressure (Pa), using `Thermodynamics.q_vap_from_RH`
with the `Liquid()` saturation curve.
"""
function ospapa_specific_humidity_fts(RHa, Ta, Pa, params)
    LX, LY, LZ = location(Ta)
    qa = FieldTimeSeries{LX, LY, LZ}(Ta.grid, Ta.times)
    pqa, pPa, pTa, pRHa = parent(qa), parent(Pa), parent(Ta), parent(RHa)
    pqa .= q_vap_from_RH.(Ref(params), pPa, pTa, pRHa ./ 100, Ref(Liquid()))
    return qa
end

"""
    OSPapaPrescribedAtmosphere(architecture = CPU(), FT = Float32;
                                start_date = first_date(OSPapaHourly(), :air_temperature),
                                end_date   = last_date(OSPapaHourly(), :air_temperature),
                                dir = download_OSPapa_cache,
                                surface_layer_height = 2.5,
                                max_gap_hours = 72)

Construct a `PrescribedAtmosphere` from Ocean Station Papa buoy observations.

Data is automatically downloaded from the NOAA/PMEL AWS S3 bucket if not
already cached locally.

!!! note "Radiation and albedo"
    The buoy `SW` and `LW` variables are **downwelling** fluxes. When this
    atmosphere is used with `OceanOnlyModel`, ClimaOcean applies its own
    ocean albedo (default α = 0.05) to compute net absorbed shortwave, and
    computes upwelling longwave from the model SST via Stefan-Boltzmann. This
    means the resulting net heat flux will differ from the COARE-computed
    `QNET` available via [`os_papa_prescribed_fluxes`](@ref). If you need the
    exact observed net fluxes, use [`os_papa_prescribed_flux_boundary_conditions`](@ref)
    instead.

Keyword Arguments
=================
- `start_date`: start of the time range
- `end_date`: end of the time range
- `dir`: directory for cached data files
- `surface_layer_height`: measurement height in meters (default: 2.5, matching
  the buoy's temperature/humidity instruments)
- `max_gap_hours`: maximum gap size (in hours) to fill by linear interpolation
  (default: 72)
"""
function OSPapaPrescribedAtmosphere(architecture = CPU(), FT = Float32;
                                    start_date = first_date(OSPapaHourly(), :air_temperature),
                                    end_date   = last_date(OSPapaHourly(), :air_temperature),
                                    dir = download_OSPapa_cache,
                                    surface_layer_height = 2.5,
                                    max_gap_hours = 72)

    mdkw = (; dataset = OSPapaHourly(), start_date, end_date, dir)

    surface_grid = RectilinearGrid(architecture, FT; size=(), topology=(Flat, Flat, Flat))

    function ospapa_fts(name)
        md = Metadata(name; mdkw...)
        download_dataset(md)
        fts = FieldTimeSeries(md, surface_grid; time_indices_in_memory = length(md))
        fill_gaps!(fts; max_gap = max_gap_hours)
        return fts
    end

    ua   = ospapa_fts(:eastward_wind)
    va   = ospapa_fts(:northward_wind)
    Ta   = ospapa_fts(:air_temperature)     # K  (Celsius conversion)
    Pa   = ospapa_fts(:sea_level_pressure)  # Pa (Millibar conversion)
    swa  = ospapa_fts(:shortwave_radiation)
    lwa  = ospapa_fts(:longwave_radiation)
    rain = ospapa_fts(:rain)                # kg/m²/s (MillimetersPerHour conversion)

    thermo_params = AtmosphereThermodynamicsParameters(FT)
    RHa = ospapa_fts(:relative_humidity)
    qa  = ospapa_specific_humidity_fts(RHa, Ta, Pa, thermo_params)

    return PrescribedAtmosphere(ua.grid, ua.times;
                                velocities = (u=ua, v=va),
                                tracers = (T=Ta, q=qa),
                                pressure = Pa,
                                freshwater_flux = (; rain),
                                downwelling_radiation = TwoBandDownwellingRadiation(shortwave=swa, longwave=lwa),
                                thermodynamics_parameters = thermo_params,
                                surface_layer_height = convert(FT, surface_layer_height))
end
