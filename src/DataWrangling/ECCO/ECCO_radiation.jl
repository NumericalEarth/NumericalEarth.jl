using NumericalEarth.Radiations: PrescribedRadiation, SurfaceRadiationProperties

"""
    ECCOPrescribedRadiation([architecture = CPU(), FT = Float32];
                              dataset = ECCO4Monthly(),
                              start_date = first_date(dataset, :downwelling_shortwave),
                              end_date = last_date(dataset, :downwelling_shortwave),
                              dir = default_download_directory(dataset),
                              time_indices_in_memory = 10,
                              time_indexing = Cyclical(),
                              ocean_surface = SurfaceRadiationProperties(0.05, 0.97),
                              sea_ice_surface = SurfaceRadiationProperties(0.7, 1.0),
                              stefan_boltzmann_constant = 5.67e-8,
                              other_kw...)

Return a [`PrescribedRadiation`](@ref) backed by ECCO downwelling shortwave
and longwave fields.
"""
function ECCOPrescribedRadiation(architecture = CPU(), FT = Float32;
                                 dataset = ECCO4Monthly(),
                                 start_date = first_date(dataset, :downwelling_shortwave),
                                 end_date = last_date(dataset, :downwelling_shortwave),
                                 dir = default_download_directory(dataset),
                                 time_indexing = Cyclical(),
                                 time_indices_in_memory = 10,
                                 ocean_surface = SurfaceRadiationProperties(0.05, 0.97),
                                 sea_ice_surface = SurfaceRadiationProperties(0.7, 1.0),
                                 stefan_boltzmann_constant = 5.67e-8,
                                 other_kw...)

    ℐꜜˢʷ_meta = Metadata(:downwelling_shortwave; dataset, start_date, end_date, dir)
    ℐꜜˡʷ_meta = Metadata(:downwelling_longwave;  dataset, start_date, end_date, dir)

    kw = (; time_indices_in_memory, time_indexing)
    kw = merge(kw, other_kw)

    ℐꜜˢʷ = FieldTimeSeries(ℐꜜˢʷ_meta, architecture; kw...)
    ℐꜜˡʷ = FieldTimeSeries(ℐꜜˡʷ_meta, architecture; kw...)

    return PrescribedRadiation(ℐꜜˢʷ, ℐꜜˡʷ;
                               ocean_surface,
                               sea_ice_surface,
                               stefan_boltzmann_constant)
end
