"""
    AtmosphereOceanModel(atmosphere, ocean; kw...)

Convenience constructor for [`EarthSystemModel`](@ref) with an atmosphere and ocean
but no sea ice. All keyword arguments are forwarded to `EarthSystemModel`.
"""
AtmosphereOceanModel(atmosphere, ocean; kw...) = EarthSystemModel(atmosphere, ocean, nothing; kw...)
