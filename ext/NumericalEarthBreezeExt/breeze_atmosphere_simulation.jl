using Oceananigans.Fields: Field
using Oceananigans.Grids: Center
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, DefaultBoundaryCondition

using Breeze: ThermodynamicConstants, ReferenceState, AnelasticDynamics,
              SaturationAdjustment, WarmPhaseEquilibrium,
              AtmosphereModel, moisture_prognostic_name

# Per-side merge for FieldBoundaryConditions: user's non-default sides override
# coupling's; coupling's non-default sides survive where the user leaves the
# default. Plain `merge(NamedTuple, NamedTuple)` replaces whole `FieldBoundaryConditions`
# objects, which clobbers the coupling's bottom-flux BCs whenever a user adds
# lateral BCs on the same prognostic.
pick(coupling_bc, user_bc) = user_bc isa DefaultBoundaryCondition ? coupling_bc : user_bc

function merge_fbcs(coupling::FieldBoundaryConditions, user::FieldBoundaryConditions)
    return FieldBoundaryConditions(west     = pick(coupling.west,     user.west),
                                   east     = pick(coupling.east,     user.east),
                                   south    = pick(coupling.south,    user.south),
                                   north    = pick(coupling.north,    user.north),
                                   bottom   = pick(coupling.bottom,   user.bottom),
                                   top      = pick(coupling.top,      user.top),
                                   immersed = pick(coupling.immersed, user.immersed))
end

function merge_boundary_conditions(coupling_bcs::NamedTuple, user_bcs::NamedTuple)
    all_keys = (keys(coupling_bcs)..., (k for k in keys(user_bcs) if !(k in keys(coupling_bcs)))...)
    pairs = (k => haskey(coupling_bcs, k) && haskey(user_bcs, k) ?
                  merge_fbcs(getproperty(coupling_bcs, k), getproperty(user_bcs, k)) :
                  (haskey(user_bcs, k) ? getproperty(user_bcs, k) : getproperty(coupling_bcs, k))
             for k in all_keys)
    return NamedTuple(pairs)
end

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
function NumericalEarth.Atmospheres.atmosphere_simulation(grid;
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

    # Create 2D flux fields for ESM coupling
    ρτˣ = Field{Center, Center, Nothing}(grid)
    ρτʸ = Field{Center, Center, Nothing}(grid)
    Jᵉ = Field{Center, Center, Nothing}(grid)
    Jᵛ = Field{Center, Center, Nothing}(grid)

    moisture_key = moisture_prognostic_name(microphysics)
    moisture_bc = NamedTuple{tuple(moisture_key)}(tuple(FieldBoundaryConditions(bottom = FluxBoundaryCondition(Jᵛ))))
    energy_bc = (; ρe = FieldBoundaryConditions(bottom = FluxBoundaryCondition(Jᵉ)))

    momentum_bcs = (
        ρu = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρτˣ)),
        ρv = FieldBoundaryConditions(bottom = FluxBoundaryCondition(ρτʸ)),
    )

    coupling_bcs = merge(momentum_bcs, energy_bc, moisture_bc)

    # User BCs override coupling defaults — per-side, so the user's lateral
    # BCs combine with the coupling's bottom-flux BCs rather than replacing
    # the whole `FieldBoundaryConditions`.
    boundary_conditions = merge_boundary_conditions(coupling_bcs, NamedTuple(boundary_conditions))

    return AtmosphereModel(grid; dynamics, microphysics,
                           momentum_advection, scalar_advection,
                           boundary_conditions, kw...)
end
