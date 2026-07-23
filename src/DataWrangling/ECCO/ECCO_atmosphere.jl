using Oceananigans.OutputReaders

using ...Atmospheres: PrescribedAtmosphere, PrescribedPrecipitationFlux

"""
    ECCOPrescribedAtmosphere([architecture = CPU()];
                              dataset = ECCO4Monthly(),
                              start_date = first_date(dataset, :air_temperature),
                              end_date = last_date(dataset, :air_temperature),
                              dir = default_download_directory(dataset),
                              time_indices_in_memory = 10,
                              time_indexing = Cyclical(),
                              surface_layer_height = 2,  # meters
                              other_kw...)

Return a [`PrescribedAtmosphere`](@ref) representing ECCO state estimate data.
The atmospheric data will be held in `FieldTimeSeries` objects containing
- velocities: u, v
- air temperature and humidity: T, q
- surface pressure: p
- freshwater flux: rain

Note: downwelling shortwave / longwave radiation is now part of the
top-level `radiation` component. Use [`ECCOPrescribedRadiation`](@ref) to
load ECCO SW/LW into a `PrescribedRadiation`.
"""
function ECCOPrescribedAtmosphere(architecture = CPU();
                                  dataset = ECCO4Monthly(),
                                  start_date = first_date(dataset, :air_temperature),
                                  end_date = last_date(dataset, :air_temperature),
                                  dir = default_download_directory(dataset),
                                  time_indexing = Cyclical(),
                                  prefetch = false,
                                  time_indices_in_memory = 10,
                                  surface_layer_height = 2,  # meters
                                  other_kw...)

    kw = (; time_indexing, time_indices_in_memory, prefetch)
    kw = merge(kw, other_kw)

    ua_meta = Metadata(:eastward_wind;         dataset, start_date, end_date, dir)
    va_meta = Metadata(:northward_wind;        dataset, start_date, end_date, dir)
    Ta_meta = Metadata(:air_temperature;       dataset, start_date, end_date, dir)
    qa_meta = Metadata(:air_specific_humidity; dataset, start_date, end_date, dir)
    pa_meta = Metadata(:sea_level_pressure;    dataset, start_date, end_date, dir)
    Fr_meta = Metadata(:rain_freshwater_flux;  dataset, start_date, end_date, dir)

    ua = FieldTimeSeries(ua_meta, architecture; kw...)
    va = FieldTimeSeries(va_meta, architecture; kw...)
    Ta = FieldTimeSeries(Ta_meta, architecture; kw...)
    qa = FieldTimeSeries(qa_meta, architecture; kw...)
    pa = FieldTimeSeries(pa_meta, architecture; kw...)
    Fr = FieldTimeSeries(Fr_meta, architecture; kw...)

    precipitation_flux = PrescribedPrecipitationFlux(rain = Fr)

    times = ua.times
    grid  = ua.grid

    velocities = (u = ua, v = va)
    pressure = pa

    FT = eltype(ua)
    surface_layer_height = convert(FT, surface_layer_height)

    atmosphere = PrescribedAtmosphere(grid, times;
                                      velocities,
                                      precipitation_flux,
                                      temperature = Ta,
                                      specific_humidity = qa,
                                      surface_layer_height,
                                      pressure)

    return atmosphere
end
