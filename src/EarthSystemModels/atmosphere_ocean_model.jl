"""
    AtmosphereOceanModel(atmosphere, ocean; kw...)

Convenience constructor for [`EarthSystemModel`](@ref) with an atmosphere and ocean
but no sea ice. All keyword arguments are forwarded to `EarthSystemModel`.

Package extensions (e.g. `NumericalEarthBreezeExt`) provide specialized methods
that automatically configure the exchange grid and component interfaces.
"""
AtmosphereOceanModel(atmosphere, ocean; kw...) = EarthSystemModel(atmosphere, ocean, nothing; kw...)
