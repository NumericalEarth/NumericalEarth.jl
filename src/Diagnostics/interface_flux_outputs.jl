"""
    heat_fluxes(coupled_model::EarthSystemModel;
                separate_sea_ice = false)

Return two-dimensional heat fluxes, `ρᵒᶜ cᵒᶜ Jᵀ` (J s⁻¹ m⁻²), where `Jᵀ` is the
temperature flux, from the ocean--sea ice model's top tracer boundary condition.

Arguments
=========

* `coupled_model`: An `EarthSystemModel`.


Keyword Arguments
=================

* `separate_sea_ice`: If set to `true`, then returns separate fluxes for the ocean and
                      sea ice model components. If `false` (default), the sum of the
                      tracer fluxes for the ocean and sea ice model components are output.


Examples
========

A very basic coupled model and a demo of how we can get its `interface_flux_outputs`.

```jldoctest interface_flux_outputs
using NumericalEarth
using Oceananigans

grid = RectilinearGrid(size = (4, 4, 2), extent = (1, 1, 1))

ocean = ocean_simulation(grid;
                         momentum_advection = nothing,
                         tracer_advection = nothing,
                         closure = nothing,
                         coriolis = nothing)
sea_ice = sea_ice_simulation(grid, ocean)
atmosphere = PrescribedAtmosphere(grid, [0.0])
coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation = Radiation())

outputs = heat_fluxes(coupled_model)

# output

NamedTuple with 1 Fields on 4×4×2 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×2 halo:
└── heat_flux: 4×4×1 Field{Center, Center, Nothing} reduced over dims = (3,) on RectilinearGrid on CPU
```

If we want to get the sea ice fluxes separate then

```jldoctest interface_flux_outputs
outputs = heat_fluxes(coupled_model, separate_sea_ice=true)

# output

NamedTuple with 3 Fields on 4×4×2 RectilinearGrid{Float64, Periodic, Periodic, Bounded} on CPU with 3×3×2 halo:
├── heat_flux: 4×4×1 Field{Center, Center, Nothing} reduced over dims = (3,) on RectilinearGrid on CPU
├── ocean_heat_flux: 4×4×1 Field{Center, Center, Nothing} reduced over dims = (3,) on RectilinearGrid on CPU
└── sea_ice_heat_flux: 4×4×1 Field{Center, Center, Nothing} reduced over dims = (3,) on RectilinearGrid on CPU
```
"""
function heat_fluxes(coupled_model::EarthSystemModel; separate_sea_ice = false)
    Jᵀ = coupled_model.ocean.model.tracers.T.boundary_conditions.top.condition
    ρᵒᶜ = coupled_model.interfaces.ocean_properties.reference_density
    cᵒᶜ = coupled_model.interfaces.ocean_properties.heat_capacity

    ice_ocean_fluxes = coupled_model.interfaces.sea_ice_ocean_interface.fluxes

    for name in (:frazil_heat, :interface_heat)
        hasproperty(ice_ocean_fluxes, name) ||
            throw(ArgumentError("Missing required interface flux field: $(name)."))
    end

    frazil_temperature_flux = getfield(ice_ocean_fluxes, :frazil_heat)
    frazil_heat_flux = Field(ρᵒᶜ * cᵒᶜ * frazil_temperature_flux)
    heat_flux = Field(ρᵒᶜ * cᵒᶜ * Jᵀ + frazil_heat_flux)

    outputs = (; heat_flux)

    if separate_sea_ice
        sea_ice_temperature_flux = getfield(ice_ocean_fluxes, :interface_heat)
        sea_ice_heat_flux = Field(ρᵒᶜ * cᵒᶜ * sea_ice_temperature_flux + frazil_heat_flux)
        ocean_heat_flux = Field(heat_flux - sea_ice_heat_flux)

        outputs = merge(outputs, (; ocean_heat_flux, sea_ice_heat_flux))
    end

    return outputs
end

"""
    temperature_fluxes(coupled_model::EarthSystemModel;
                       separate_sea_ice = false)

Return two-dimensional temperature fluxes, `Jᵀ` (K m s⁻¹) from the ocean--sea ice
model's top tracer boundary condition.

Arguments
=========

* `coupled_model`: An `EarthSystemModel`.


Keyword Arguments
=================

* `separate_sea_ice`: If set to `true`, then returns separate fluxes for the ocean and
                      sea ice model components. If `false` (default), the sum of the
                      tracer fluxes for the ocean and sea ice model components are output.

See [`heat_fluxes`](@ref) for examples.
"""
function temperature_fluxes(coupled_model::EarthSystemModel; separate_sea_ice = false)
    Jᵀ = coupled_model.ocean.model.tracers.T.boundary_conditions.top.condition

    ice_ocean_fluxes = coupled_model.interfaces.sea_ice_ocean_interface.fluxes

    for name in (:frazil_heat, :interface_heat)
        hasproperty(ice_ocean_fluxes, name) ||
            throw(ArgumentError("Missing required interface flux field: $(name)."))
    end

    frazil_temperature_flux = getfield(ice_ocean_fluxes, :frazil_heat)
    temperature_flux = Field(Jᵀ + frazil_temperature_flux)

    outputs = (; temperature_flux)

    if separate_sea_ice
        sea_ice_temperature_flux = Field(getfield(ice_ocean_fluxes, :interface_heat) + frazil_temperature_flux)
        ocean_temperature_flux = Field(temperature_flux - sea_ice_temperature_flux)

        outputs = merge(outputs, (; ocean_temperature_flux, sea_ice_temperature_flux))
    end

    return outputs
end

"""
    freshwater_fluxes(coupled_model::EarthSystemModel;
                      separate_sea_ice = false,
                      reference_salinity = 35)

Return two-dimensional freshwater mass fluxes, `-ρᵒᶜ Jˢ / S₀` (kg s⁻¹ m⁻²) from the ocean--sea ice
model's top tracer boundary condition.


Arguments
=========

* `coupled_model`: An `EarthSystemModel`.


Keyword Arguments
=================

* `separate_sea_ice`: If set to `true`, then returns separate fluxes for the ocean and
                      sea ice model components. If `false` (default), the sum of the
                      tracer fluxes for the ocean and sea ice model components are output.

* `reference_salinity`: Reference salinity ``S₀`` used to convert the salt fluxes to freshwater
                        mass fluxes, i.e., ``-ρᵒᶜ Jˢ / S₀``, where ``Jˢ`` is the salt fluxes.
                        Default: 35 g/kg.

See [`heat_fluxes`](@ref) for examples.
"""
function freshwater_fluxes(coupled_model::EarthSystemModel;
                           separate_sea_ice = false,
                           reference_salinity = 35)

    Jˢ = coupled_model.ocean.model.tracers.S.boundary_conditions.top.condition

    ocean_properties = coupled_model.interfaces.ocean_properties
    ρᵒᶜ = ocean_properties.reference_density
    S₀ = convert(typeof(ρᵒᶜ), reference_salinity)

    ice_ocean_fluxes = coupled_model.interfaces.sea_ice_ocean_interface.fluxes

    for name in (:salt,)
        hasproperty(ice_ocean_fluxes, name) ||
            throw(ArgumentError("Missing required interface flux field: $(name)."))
    end

    freshwater_flux = Field(-ρᵒᶜ * Jˢ / S₀)
    outputs = (; freshwater_flux)

    if separate_sea_ice
        sea_ice_salinity_flux = getfield(ice_ocean_fluxes, :salt)
        sea_ice_freshwater_flux = Field(-ρᵒᶜ * sea_ice_salinity_flux / S₀)
        ocean_freshwater_flux = Field(freshwater_flux - sea_ice_freshwater_flux)

        outputs = merge(outputs, (; ocean_freshwater_flux,
                                    sea_ice_freshwater_flux))
    end

    return outputs
end


"""
    salinity_fluxes(coupled_model::EarthSystemModel;
                    separate_sea_ice = false,
                    reference_salinity = 35)

Return two-dimensional salinity fluxes, `Jˢ` (psu m s⁻¹) from the ocean--sea ice
model's top tracer boundary condition.


Arguments
=========

* `coupled_model`: An `EarthSystemModel`.


Keyword Arguments
=================

* `separate_sea_ice`: If set to `true`, then returns separate fluxes for the ocean and
                      sea ice model components. If `false` (default), the sum of the
                      tracer fluxes for the ocean and sea ice model components are output.

See [`heat_fluxes`](@ref) for examples.
"""
function salinity_fluxes(coupled_model::EarthSystemModel; separate_sea_ice = false)
    salinity_flux = Jˢ = coupled_model.ocean.model.tracers.S.boundary_conditions.top.condition

    ice_ocean_fluxes = coupled_model.interfaces.sea_ice_ocean_interface.fluxes

    for name in (:salt,)
        hasproperty(ice_ocean_fluxes, name) ||
            throw(ArgumentError("Missing required interface flux field: $(name)."))
    end

    outputs = (; salinity_flux)

    if separate_sea_ice
        sea_ice_salinity_flux = getfield(ice_ocean_fluxes, :salt)
        ocean_salinity_flux = Field(salinity_flux - sea_ice_salinity_flux)

        outputs = merge(outputs, (; ocean_salinity_flux,
                                    sea_ice_salinity_flux))
    end

    return outputs
end
