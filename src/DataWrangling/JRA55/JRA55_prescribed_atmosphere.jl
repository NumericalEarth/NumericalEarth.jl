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
                              other_kw...)

Return a [`PrescribedAtmosphere`](@ref) representing JRA55 reanalysis data.
Each atmospheric field is constructed via `FieldTimeSeries(::JRA55Metadata)`,
which uses a `DatasetBackend` parameterised by JRA55 metadata so that the
JRA55-specific `set!` (chunked-yearly NetCDF) is dispatched.
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
                                   other_kw...)

    kw = (; time_indexing, time_indices_in_memory)
    kw = merge(kw, other_kw)

    jra55_fts(name) = FieldTimeSeries(Metadata(name; dataset, start_date, end_date, dir), architecture; kw...)

    ua   = jra55_fts(:eastward_velocity)
    va   = jra55_fts(:northward_velocity)
    Ta   = jra55_fts(:temperature)
    qa   = jra55_fts(:specific_humidity)
    pa   = jra55_fts(:sea_level_pressure)
    Fra  = jra55_fts(:rain_freshwater_flux)
    Fsn  = jra55_fts(:snow_freshwater_flux)
    ℐꜜˡʷ = jra55_fts(:downwelling_longwave_radiation)
    ℐꜜˢʷ = jra55_fts(:downwelling_shortwave_radiation)

    freshwater_flux = (rain = Fra,
                       snow = Fsn)

    # Rivers and icebergs are on a different grid and have a different
    # frequency than the rest of the JRA55 data. We use the
    # PrescribedAtmosphere `auxiliary_freshwater_flux` feature for them.
    if include_rivers_and_icebergs
        Fri = jra55_fts(:river_freshwater_flux)
        Fic = jra55_fts(:iceberg_freshwater_flux)
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
