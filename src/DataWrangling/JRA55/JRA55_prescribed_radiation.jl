JRA55PrescribedRadiation(arch::Distributed; kw...) =
    JRA55PrescribedRadiation(child_architecture(arch); kw...)

"""
    JRA55PrescribedRadiation([architecture = CPU()]; dataset = RepeatYearJRA55(),
                             time_indices_in_memory = 10, other_kw...)

Return a [`PrescribedRadiation`](@ref NumericalEarth.Radiations.PrescribedRadiation)
backed by JRA55 downwelling shortwave and longwave `FieldTimeSeries`. Surface radiative
properties (albedo, emissivity) for ocean and sea-ice surfaces default to standard values;
pass `*_surface = nothing` to omit a surface or supply your own `SurfaceRadiationProperties`
(e.g. for `land_surface` when running land-only simulations).
"""
JRA55PrescribedRadiation(architecture = CPU(); dataset = RepeatYearJRA55(),
                         time_indices_in_memory = 10, kw...) =
    prescribed_radiation(dataset, architecture; time_indices_in_memory, kw...)
