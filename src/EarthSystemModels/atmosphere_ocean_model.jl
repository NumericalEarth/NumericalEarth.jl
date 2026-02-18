"""
    AtmosphereOceanModel(atmosphere, ocean; kw...)

Convenience constructor for [`EarthSystemModel`](@ref) with an atmosphere and ocean
but no sea ice.

All keyword arguments are forwarded to `EarthSystemModel`. When using a `SlabOcean`,
the `exchange_grid` keyword must be provided (or use a package extension that provides
a specialized constructor, e.g. `NumericalEarthBreezeExt`).

Example
=======

```julia
model = AtmosphereOceanModel(atmosphere, ocean; exchange_grid=ocean.grid)
```
"""
AtmosphereOceanModel(atmosphere, ocean; kw...) = EarthSystemModel(atmosphere, ocean, nothing; kw...)
