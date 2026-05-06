JRA55PrescribedRadiation(arch::Distributed; kw...) =
    JRA55PrescribedRadiation(child_architecture(arch); kw...)

"""
    JRA55PrescribedRadiation([architecture = CPU()];
                             dataset = RepeatYearJRA55(),
                             start_date = first_date(dataset, :downwelling_shortwave_radiation),
                             end_date = last_date(dataset, :downwelling_shortwave_radiation),
                             dir = download_JRA55_cache,
                             time_indices_in_memory = 10,
                             time_indexing = Cyclical(),
                             ocean_surface = SurfaceRadiationProperties(0.05, 0.97),
                             sea_ice_surface = SurfaceRadiationProperties(0.7, 1.0),
                             stefan_boltzmann_constant = default_stefan_boltzmann_constant,
                             region = nothing,
                             other_kw...)

Return a [`PrescribedRadiation`](@ref) backed by JRA55 downwelling shortwave
and longwave `FieldTimeSeries`. Surface radiative properties (albedo,
emissivity) for ocean and sea-ice surfaces default to standard values; pass
`ocean_surface = nothing` (or `sea_ice_surface = nothing`) to omit a surface.
"""
function JRA55PrescribedRadiation(architecture = CPU();
                                  dataset = RepeatYearJRA55(),
                                  start_date = first_date(dataset, :downwelling_shortwave_radiation),
                                  end_date = last_date(dataset, :downwelling_shortwave_radiation),
                                  dir = download_JRA55_cache,
                                  time_indices_in_memory = 10,
                                  time_indexing = Cyclical(),
                                  ocean_surface = SurfaceRadiationProperties(0.05, 0.97),
                                  sea_ice_surface = SurfaceRadiationProperties(0.7, 1.0),
                                  stefan_boltzmann_constant = default_stefan_boltzmann_constant,
                                  region = nothing,
                                  other_kw...)

    kw = (; time_indexing, time_indices_in_memory)
    kw = merge(kw, other_kw)

    JRA55FieldTimeSeries(name) = FieldTimeSeries(Metadata(name; dataset, start_date, end_date, dir, region), architecture; kw...)

    ℐꜜˢʷ = JRA55FieldTimeSeries(:downwelling_shortwave_radiation)
    ℐꜜˡʷ = JRA55FieldTimeSeries(:downwelling_longwave_radiation)

    return PrescribedRadiation(ℐꜜˢʷ, ℐꜜˡʷ;
                               ocean_surface,
                               sea_ice_surface,
                               stefan_boltzmann_constant)
end
