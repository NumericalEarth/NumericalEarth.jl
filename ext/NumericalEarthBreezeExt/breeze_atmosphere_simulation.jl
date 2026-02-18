using Breeze: ThermodynamicConstants, ReferenceState, AnelasticDynamics,
             SaturationAdjustment, WarmPhaseEquilibrium,
             AtmosphereModel

"""
    atmosphere_simulation(grid;
                          surface_pressure = 101325,
                          potential_temperature = 285,
                          thermodynamic_constants = ThermodynamicConstants(eltype(grid)),
                          dynamics = AnelasticDynamics(ReferenceState(grid, thermodynamic_constants;
                                                                      surface_pressure, potential_temperature)),
                          microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()),
                          momentum_advection = WENO(order=9),
                          scalar_advection = WENO(order=5),
                          boundary_conditions = NamedTuple(),
                          kw...)

Construct a Breeze `AtmosphereModel` with sensible defaults for coupled simulations.

Surface fluxes are handled by the `EarthSystemModel` coupling framework (via similarity
theory), not by Breeze's own boundary conditions.

Arguments
=========

- `grid`: An Oceananigans grid for the atmosphere domain.

Keyword Arguments
=================

- `surface_pressure`: Reference surface pressure in Pa. Default: 101325.
- `potential_temperature`: Reference potential temperature in K. Default: 285.
- `thermodynamic_constants`: Breeze `ThermodynamicConstants`. Default: `ThermodynamicConstants(eltype(grid))`.
- `dynamics`: Dynamics formulation. Default: `AnelasticDynamics` with the given reference state.
- `microphysics`: Microphysics scheme. Default: `SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())`.
- `momentum_advection`: Advection scheme for momentum. Default: `WENO(order=9)`.
- `scalar_advection`: Advection scheme for scalars. Default: `WENO(order=5)`.
- `boundary_conditions`: Named tuple of boundary conditions. Default: `NamedTuple()`.

All additional keyword arguments are forwarded to `Breeze.AtmosphereModel`.
"""
function atmosphere_simulation(grid;
                               surface_pressure = 101325,
                               potential_temperature = 285,
                               thermodynamic_constants = ThermodynamicConstants(eltype(grid)),
                               dynamics = AnelasticDynamics(ReferenceState(grid, thermodynamic_constants;
                                                                           surface_pressure, potential_temperature)),
                               microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()),
                               momentum_advection = Oceananigans.WENO(order=9),
                               scalar_advection = Oceananigans.WENO(order=5),
                               boundary_conditions = NamedTuple(),
                               kw...)

    return AtmosphereModel(grid; dynamics, microphysics,
                           momentum_advection, scalar_advection,
                           boundary_conditions, kw...)
end
