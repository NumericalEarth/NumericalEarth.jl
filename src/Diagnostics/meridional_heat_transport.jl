using ..EarthSystemModels: EarthSystemModel, reference_density, heat_capacity

"""
    meridional_heat_transport(esm::EarthSystemModel;
                              reference_temperature = 0)

Return the meridional heat transport for the coupled `esm::EarthSystemModel` by computing
the meridional heat flux.

The meridional heat transport is computed via:

```math
\\mathrm{MHT} ≡ ρᵒᶜ cᵒᶜ ∫  v (T - T_{\\rm ref}) \\, \\mathrm{d}x \\, \\mathrm{d}z
```

Above, ``T_{\\rm ref}`` is a reference temperature and ``ρᵒᶜ`` and ``cᵒᶜ`` are the
ocean reference density and specific heat capacity respectively.

!!! warning "Only works on LatitudeLongitudeGrid"

    The `meridional_heat_transport` diagnostic currently is only supported only on
    `LongitudeLatitudeGrid`s.

Arguments
=========

* `esm`: An EarthSystemModel.


Keyword Arguments
=================

* `reference_temperature`: The reference temperature (in ᵒC) used for the calculation; default: 0 ᵒC.

  !!! info "Reference temperature"

      The reference temperature is only relevant when we compute the meridional heat transport over a section
      where there is a net volume transport. If we are computing the diagnostic globally, i.e., around a whole
      latitude circle, then by necessity there is no net volume transport and thus the reference temperature
      value is irrelevant. Section-averaged transport could also be considered as a reference temperature to
      remove residual barotropic volume fluxes in basin-scale/regional analyses where a net volume transport
      is present.

Example
=======

```jldoctest
using NumericalEarth
using Oceananigans

grid = RectilinearGrid(size = (4, 5, 2), extent = (1, 1, 1),
                       topology = (Periodic, Bounded, Bounded))

ocean = ocean_simulation(grid;
                         momentum_advection = nothing,
                         tracer_advection = nothing,
                         closure = nothing,
                         coriolis = nothing)

sea_ice = sea_ice_simulation(grid, ocean)

atmosphere = PrescribedAtmosphere(grid, [0.0])

esm = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation = Radiation())

mht = meridional_heat_transport(esm)

# output

Integral of BinaryOperation at (Center, Face, Center) over dims (1, 3)
└── operand: BinaryOperation at (Center, Face, Center)
    └── grid: 4×5×2 RectilinearGrid{Float64, Periodic, Bounded, Bounded} on CPU with 3×3×2 halo
```
"""
function meridional_heat_transport(esm::EarthSystemModel; reference_temperature=0)

    grid = esm.ocean.model.grid

    validation_grid = grid isa ImmersedBoundaryGrid ? grid.underlying_grid : grid

    grid isa OrthogonalSphericalShellGrid &&
        throw(ArgumentError("meridional_heat_transport diagnostic does not work on OrthogonalSphericalShellGrid at the moment; use LatitudeLongitudeGrid."))

    FT = eltype(esm)
    reference_temperature = convert(FT, reference_temperature)

    ρᵒᶜ = reference_density(esm.ocean)
    cᵒᶜ = heat_capacity(esm.ocean)

    T = esm.ocean.model.tracers.T
    v = esm.ocean.model.velocities.v

    MHT = Integral(ρᵒᶜ * cᵒᶜ * v * (T - reference_temperature), dims=(1, 3))
    return MHT
end
