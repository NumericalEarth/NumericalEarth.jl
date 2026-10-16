JRA55PrescribedRadiation(arch::Distributed; kw...) =
    JRA55PrescribedRadiation(child_architecture(arch); kw...)

"""
$(MethodSignatures(defaults = true))

JRA55 downwelling radiation. Remaining keyword arguments go to [`prescribed_radiation`](@ref).
"""
JRA55PrescribedRadiation(architecture = CPU(); dataset = RepeatYearJRA55(),
                         time_indices_in_memory = 10, kw...) =
    prescribed_radiation(dataset, architecture; time_indices_in_memory, kw...)
