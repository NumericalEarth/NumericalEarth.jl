ERA5PrescribedRadiation(arch::Distributed; kw...) =
    ERA5PrescribedRadiation(child_architecture(arch); kw...)

"""
    ERA5PrescribedRadiation([architecture = CPU()]; dataset = ERA5HourlySingleLevel(),
                            time_indices_in_memory = 24, other_kw...)

ERA5 downwelling radiation, suitable for regional hindcast forcing. ERA5 stores these as energy
accumulated over the previous hour (J m⁻²); the load-time `conversion_units` divides by the accumulation
interval to recover the mean flux (W m⁻²). Remaining keyword arguments go to
[`prescribed_radiation`](@ref).
"""
ERA5PrescribedRadiation(architecture = CPU(); dataset = ERA5HourlySingleLevel(),
                        time_indices_in_memory = 24, kw...) =
    prescribed_radiation(dataset, architecture; time_indices_in_memory, kw...)
