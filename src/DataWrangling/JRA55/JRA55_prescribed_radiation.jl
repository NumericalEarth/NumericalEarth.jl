JRA55PrescribedRadiation(arch::Distributed, FT = Float32; kw...) =
    JRA55PrescribedRadiation(child_architecture(arch); kw...)

"""
    JRA55PrescribedRadiation([architecture = CPU(), FT = Float32];
                             dataset = RepeatYearJRA55(),
                             start_date = first_date(dataset, :downwelling_shortwave_radiation),
                             end_date = last_date(dataset, :downwelling_shortwave_radiation),
                             backend = JRA55NetCDFBackend(10),
                             time_indexing = Cyclical(),
                             ocean_surface = SurfaceRadiationProperties(0.05, 0.97),
                             sea_ice_surface = SurfaceRadiationProperties(0.7, 1.0),
                             stefan_boltzmann_constant = default_stefan_boltzmann_constant,
                             other_kw...)

Return a [`PrescribedRadiation`](@ref) backed by JRA55 downwelling shortwave
and longwave `JRA55FieldTimeSeries`. Surface radiative properties (albedo,
emissivity) for ocean and sea-ice surfaces default to standard values; pass
`ocean_surface = nothing` (or `sea_ice_surface = nothing`) to omit a surface.
"""
function JRA55PrescribedRadiation(architecture = CPU(), FT = Float32;
                                  dataset = RepeatYearJRA55(),
                                  start_date = first_date(dataset, :downwelling_shortwave_radiation),
                                  end_date = last_date(dataset, :downwelling_shortwave_radiation),
                                  backend = JRA55NetCDFBackend(10),
                                  time_indexing = Cyclical(),
                                  ocean_surface = SurfaceRadiationProperties(0.05, 0.97),
                                  sea_ice_surface = SurfaceRadiationProperties(0.7, 1.0),
                                  stefan_boltzmann_constant = default_stefan_boltzmann_constant,
                                  other_kw...)

    kw = (; time_indexing, backend, start_date, end_date, dataset)
    kw = merge(kw, other_kw)

    ℐꜜˢʷ = JRA55FieldTimeSeries(:downwelling_shortwave_radiation, architecture, FT; kw...)
    ℐꜜˡʷ = JRA55FieldTimeSeries(:downwelling_longwave_radiation,  architecture, FT; kw...)

    return PrescribedRadiation(ℐꜜˢʷ, ℐꜜˡʷ;
                               ocean_surface,
                               sea_ice_surface,
                               stefan_boltzmann_constant)
end
