const AA = Oceananigans.Architectures.AbstractArchitecture

JRA55PrescribedAtmosphere(arch::Distributed; kw...) =
    JRA55PrescribedAtmosphere(child_architecture(arch); kw...)

"""
    JRA55PrescribedAtmosphere([architecture = CPU()];
                              dataset = RepeatYearJRA55(),
                              start_date = first_date(dataset, :temperature),
                              end_date = last_date(dataset, :temperature),
                              dir = download_JRA55_cache,
                              time_indices_in_memory = 10,
                              time_indexing = Cyclical(),
                              surface_layer_height = 10,  # meters
                              region = nothing,
                              other_kw...)

Return a [`PrescribedAtmosphere`](@ref) representing JRA55 reanalysis data. Each atmospheric field is constructed via
`FieldTimeSeries(::JRA55Metadata)`, which uses a `DatasetBackend` parameterised by JRA55 metadata so that the JRA55-specific
`set!` (chunked-yearly NetCDF) is dispatched.
The `region` keyword restricts the atmosphere to a sub-domain of the global JRA55 grid.

Note: downwelling shortwave / longwave radiation is now part of the
top-level `radiation` component. Use [`JRA55PrescribedRadiation`](@ref) to
load JRA55 SW/LW into a `PrescribedRadiation`.
"""
function JRA55PrescribedAtmosphere(architecture = CPU();
                                   dataset = RepeatYearJRA55(),
                                   start_date = first_date(dataset, :temperature),
                                   end_date = last_date(dataset, :temperature),
                                   dir = download_JRA55_cache,
                                   time_indices_in_memory = 10,
                                   time_indexing = Cyclical(),
                                   surface_layer_height = 10,  # meters
                                   region = nothing,
                                   other_kw...)

    kw = (; time_indexing, time_indices_in_memory)
    kw = merge(kw, other_kw)

    ua_meta  = Metadata(:eastward_velocity;    dataset, start_date, end_date, dir, region)
    va_meta  = Metadata(:northward_velocity;   dataset, start_date, end_date, dir, region)
    Ta_meta  = Metadata(:temperature;          dataset, start_date, end_date, dir, region)
    qa_meta  = Metadata(:specific_humidity;    dataset, start_date, end_date, dir, region)
    pa_meta  = Metadata(:sea_level_pressure;   dataset, start_date, end_date, dir, region)
    Fra_meta = Metadata(:rain_freshwater_flux; dataset, start_date, end_date, dir, region)
    Fsn_meta = Metadata(:snow_freshwater_flux; dataset, start_date, end_date, dir, region)

    ua  = FieldTimeSeries(ua_meta,  architecture; kw...)
    va  = FieldTimeSeries(va_meta,  architecture; kw...)
    Ta  = FieldTimeSeries(Ta_meta,  architecture; kw...)
    qa  = FieldTimeSeries(qa_meta,  architecture; kw...)
    pa  = FieldTimeSeries(pa_meta,  architecture; kw...)
    Fra = FieldTimeSeries(Fra_meta, architecture; kw...)
    Fsn = FieldTimeSeries(Fsn_meta, architecture; kw...)

    freshwater_flux = PrescribedPrecipitationFlux(rain = Fra, snow = Fsn)

    times = ua.times
    grid  = ua.grid

    velocities = (u = ua,
                  v = va)

    tracers = (T = Ta,
               q = qa)

    pressure = pa

    FT = eltype(ua)
    surface_layer_height = convert(FT, surface_layer_height)

    atmosphere = PrescribedAtmosphere(grid, times;
                                      velocities,
                                      freshwater_flux,
                                      tracers,
                                      surface_layer_height,
                                      pressure)

    return atmosphere
end
