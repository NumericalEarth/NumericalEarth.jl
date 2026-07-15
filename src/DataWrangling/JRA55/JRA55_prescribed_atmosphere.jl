using ...Atmospheres: PrescribedAtmosphere, PrescribedPrecipitationFlux

# `JRA55PrescribedAtmosphere` is a type alias for a `PrescribedAtmosphere` whose `source` is any JRA55
# dataset; the constructor methods below build one (the alias is both the dispatch type and the builder).
const JRA55PrescribedAtmosphere = PrescribedAtmosphere{<:JRA55Dataset}

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

    jra55_fts(name) = FieldTimeSeries(Metadata(name; dataset, start_date, end_date, dir, region), architecture; kw...)

    u    = jra55_fts(:eastward_velocity)
    v    = jra55_fts(:northward_velocity)
    T    = jra55_fts(:temperature)
    qᵛ   = jra55_fts(:specific_humidity)
    p    = jra55_fts(:sea_level_pressure)
    rain = jra55_fts(:rain_freshwater_flux)
    snow = jra55_fts(:snow_freshwater_flux)

    precipitation_flux = PrescribedPrecipitationFlux(; rain, snow)

    grid  = u.grid
    times = u.times
    FT    = eltype(u)

    return PrescribedAtmosphere(grid, times;
                                source = dataset,
                                velocities = (; u, v),
                                temperature = T,
                                specific_humidity = qᵛ,
                                pressure = p,
                                precipitation_flux,
                                surface_layer_height = convert(FT, surface_layer_height))
end
