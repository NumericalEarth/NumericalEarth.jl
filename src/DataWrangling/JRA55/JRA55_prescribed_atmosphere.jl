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
                              include_rivers_and_icebergs = false,
                              region = nothing,
                              other_kw...)

Return a [`PrescribedAtmosphere`](@ref) representing JRA55 reanalysis data. Each atmospheric field is constructed via 
`FieldTimeSeries(::JRA55Metadata)`, which uses a `DatasetBackend` parameterised by JRA55 metadata so that the JRA55-specific 
`set!` (chunked-yearly NetCDF) is dispatched.
The `region` keyword restricts the atmosphere to a sub-domain of the global JRA55 grid.
"""
function JRA55PrescribedAtmosphere(architecture = CPU();
                                   dataset = RepeatYearJRA55(),
                                   start_date = first_date(dataset, :temperature),
                                   end_date = last_date(dataset, :temperature),
                                   dir = download_JRA55_cache,
                                   time_indices_in_memory = 10,
                                   time_indexing = Cyclical(),
                                   surface_layer_height = 10,  # meters
                                   include_rivers_and_icebergs = false,
                                   region = nothing,
                                   other_kw...)

    kw = (; time_indexing, time_indices_in_memory)
    kw = merge(kw, other_kw)

    JRA55FieldTimeSeries(name) = FieldTimeSeries(Metadata(name; dataset, start_date, end_date, dir, region), architecture; kw...)

    ua   = JRA55FieldTimeSeries(:eastward_velocity)
    va   = JRA55FieldTimeSeries(:northward_velocity)
    Ta   = JRA55FieldTimeSeries(:temperature)
    qa   = JRA55FieldTimeSeries(:specific_humidity)
    pa   = JRA55FieldTimeSeries(:sea_level_pressure)
    Fra  = JRA55FieldTimeSeries(:rain_freshwater_flux)
    Fsn  = JRA55FieldTimeSeries(:snow_freshwater_flux)
    ℐꜜˡʷ = JRA55FieldTimeSeries(:downwelling_longwave_radiation)
    ℐꜜˢʷ = JRA55FieldTimeSeries(:downwelling_shortwave_radiation)

    freshwater_flux = (rain = Fra,
                       snow = Fsn)

    # Rivers and icebergs are on a different grid and have a different
    # frequency than the rest of the JRA55 data. We use the
    # PrescribedAtmosphere `auxiliary_freshwater_flux` feature for them.
    if include_rivers_and_icebergs
        Fri = JRA55FieldTimeSeries(:river_freshwater_flux)
        Fic = JRA55FieldTimeSeries(:iceberg_freshwater_flux)
        auxiliary_freshwater_flux = (rivers = Fri, icebergs = Fic)
    else
        auxiliary_freshwater_flux = nothing
    end

    times = ua.times
    grid  = ua.grid

    velocities = (u = ua,
                  v = va)

    tracers = (T = Ta,
               q = qa)

    pressure = pa

    downwelling_radiation = TwoBandDownwellingRadiation(shortwave=ℐꜜˢʷ, longwave=ℐꜜˡʷ)

    FT = eltype(ua)
    surface_layer_height = convert(FT, surface_layer_height)

    atmosphere = PrescribedAtmosphere(grid, times;
                                      velocities,
                                      freshwater_flux,
                                      auxiliary_freshwater_flux,
                                      tracers,
                                      downwelling_radiation,
                                      surface_layer_height,
                                      pressure)

    return atmosphere
end
