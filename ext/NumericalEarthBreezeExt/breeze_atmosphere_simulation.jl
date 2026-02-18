using Breeze: ThermodynamicConstants, ReferenceState, AnelasticDynamics,
             SaturationAdjustment, WarmPhaseEquilibrium,
             AtmosphereModel, PolynomialCoefficient,
             BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux

"""
    atmosphere_simulation(grid;
                          sea_surface_temperature = nothing,
                          surface_pressure = 101325,
                          potential_temperature = 285,
                          roughness_length = 1.5e-4,
                          gustiness = 1e-2,
                          thermodynamic_constants = ThermodynamicConstants(eltype(grid)),
                          dynamics = ...,
                          microphysics = ...,
                          momentum_advection = WENO(order=9),
                          scalar_advection = WENO(order=5),
                          boundary_conditions = NamedTuple(),
                          kw...)

Construct a Breeze `AtmosphereModel` with sensible defaults for coupled simulations.

If `sea_surface_temperature` is provided and `boundary_conditions` is empty,
bulk surface exchange boundary conditions (drag, sensible heat, vapor flux) are
automatically constructed using `PolynomialCoefficient` with the given
`roughness_length` and `gustiness`.

Arguments
=========

- `grid`: An Oceananigans grid for the atmosphere domain.

Keyword Arguments
=================

- `sea_surface_temperature`: An Oceananigans `Field` for SST-dependent surface BCs.
  If `nothing`, no automatic BCs are added. Default: `nothing`.
- `surface_pressure`: Reference surface pressure in Pa. Default: 101325.
- `potential_temperature`: Reference potential temperature in K. Default: 285.
- `roughness_length`: Surface roughness length in meters (for automatic BCs). Default: 1.5e-4.
- `gustiness`: Gustiness parameter in m/s (for automatic BCs). Default: 1e-2.
- `thermodynamic_constants`: Breeze `ThermodynamicConstants`. Default: `ThermodynamicConstants(eltype(grid))`.
- `dynamics`: Dynamics formulation. Default: `AnelasticDynamics` with the given reference state.
- `microphysics`: Microphysics scheme. Default: `SaturationAdjustment(equilibrium=WarmPhaseEquilibrium())`.
- `momentum_advection`: Advection scheme for momentum. Default: `WENO(order=9)`.
- `scalar_advection`: Advection scheme for scalars. Default: `WENO(order=5)`.
- `boundary_conditions`: Named tuple of boundary conditions. Default: `NamedTuple()`.
  Overrides automatic SST-based BCs if non-empty.

All additional keyword arguments are forwarded to `Breeze.AtmosphereModel`.
"""
function atmosphere_simulation(grid;
                               sea_surface_temperature = nothing,
                               surface_pressure = 101325,
                               potential_temperature = 285,
                               roughness_length = 1.5e-4,
                               gustiness = 1e-2,
                               thermodynamic_constants = ThermodynamicConstants(eltype(grid)),
                               dynamics = AnelasticDynamics(ReferenceState(grid, thermodynamic_constants;
                                                                           surface_pressure, potential_temperature)),
                               microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()),
                               momentum_advection = Oceananigans.WENO(order=9),
                               scalar_advection = Oceananigans.WENO(order=5),
                               boundary_conditions = NamedTuple(),
                               kw...)

    # If SST is provided and no explicit BCs given, set up bulk surface exchange
    if !isnothing(sea_surface_temperature) && isempty(boundary_conditions)
        coef = PolynomialCoefficient(; roughness_length)
        ρu_bcs = Oceananigans.FieldBoundaryConditions(bottom=BulkDrag(; coefficient=coef, gustiness, surface_temperature=sea_surface_temperature))
        ρv_bcs = Oceananigans.FieldBoundaryConditions(bottom=BulkDrag(; coefficient=coef, gustiness, surface_temperature=sea_surface_temperature))
        ρe_bcs = Oceananigans.FieldBoundaryConditions(bottom=BulkSensibleHeatFlux(; coefficient=coef, gustiness, surface_temperature=sea_surface_temperature))
        ρqᵗ_bcs = Oceananigans.FieldBoundaryConditions(bottom=BulkVaporFlux(; coefficient=coef, gustiness, surface_temperature=sea_surface_temperature))
        boundary_conditions = (ρu=ρu_bcs, ρv=ρv_bcs, ρe=ρe_bcs, ρqᵗ=ρqᵗ_bcs)
    end

    return AtmosphereModel(grid; dynamics, microphysics,
                           momentum_advection, scalar_advection,
                           boundary_conditions, kw...)
end
