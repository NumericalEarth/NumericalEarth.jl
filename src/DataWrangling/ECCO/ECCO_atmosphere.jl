using NumericalEarth.DataWrangling: DatasetBackend
using Oceananigans.OutputReaders
using NumericalEarth.Atmospheres: PrescribedAtmosphere, TwoBandDownwellingRadiation

"""
    ECCOPrescribedAtmosphere([architecture = CPU(), FT = Float32];
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
- downwelling radiation: ℐꜜˢʷ, ℐꜜˡʷ
"""
function ECCOPrescribedAtmosphere(architecture = CPU(), FT = Float32;
                                  dataset = ECCO4Monthly(),
                                  start_date = first_date(dataset, :air_temperature),
                                  end_date = last_date(dataset, :air_temperature),
                                  dir = default_download_directory(dataset),
                                  time_indexing = Cyclical(),
                                  time_indices_in_memory = 10,
                                  surface_layer_height = 2,  # meters
                                  other_kw...)

    kw = (; time_indexing, time_indices_in_memory)
    kw = merge(kw, other_kw)

    ecco_fts(name) = FieldTimeSeries(Metadata(name; dataset, start_date, end_date, dir), architecture; kw...)

    ua   = ecco_fts(:eastward_wind)
    va   = ecco_fts(:northward_wind)
    Ta   = ecco_fts(:air_temperature)
    qa   = ecco_fts(:air_specific_humidity)
    pa   = ecco_fts(:sea_level_pressure)
    ℐꜜˡʷ = ecco_fts(:downwelling_longwave)
    ℐꜜˢʷ = ecco_fts(:downwelling_shortwave)
    Fr   = ecco_fts(:rain_freshwater_flux)
    
    auxiliary_freshwater_flux = nothing
    freshwater_flux = (; rain = Fr)

    times = ua.times
    grid  = ua.grid

    velocities = (u = ua, v = va)
    tracers = (T = Ta, q = qa)
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
