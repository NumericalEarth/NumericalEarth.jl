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
    atmosphere_model(grid;
                          surface_pressure = 101325,
                          potential_temperature = 285,
                          thermodynamic_constants = ThermodynamicConstants(eltype(grid)),
                          dynamics = AnelasticDynamics(ReferenceState(grid, thermodynamic_constants;
                                                                      surface_pressure, potential_temperature)),
                          microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()),
                          momentum_advection = WENO(order=9),
                          scalar_advection = WENO(order=5),
                          boundary_conditions = NamedTuple(),
                          coriolis = nothing,
                          forcing = NamedTuple(),
                          closure = nothing,
                          clock = Clock{eltype(grid)}(time=0))

Construct a Breeze `AtmosphereModel` with sensible defaults for coupled simulations.
[`atmosphere_simulation`](@ref) wraps this in an Oceananigans `Simulation` (mirroring
the role of [`ocean_simulation`](@ref)).

Surface fluxes are handled by the `EarthSystemModel` coupling framework (via
similarity theory), not by Breeze's own boundary conditions, so the bottom
BCs on the prognostic momentum, energy, and moisture fields are pre-wired to
2D coupling fields that the coupler fills each step.

Radiation is wired in by the coupled-model constructor through
[`materialize_earth_system_radiation!`](@ref). The atmosphere is built with a
skeleton `CoupledRadiation` proxy (no allocation, no tendency contribution),
which is replaced inside `EarthSystemModel` with a materialized proxy that
aliases `coupled_model.radiation.flux_divergence`. Passing a
`Breeze.RadiativeTransferModel` directly as `radiation` here is rejected — use
`AtmosphereLandModel(atmosphere, land; radiation = rtm)` instead.

To nest this child in a coarser parent atmosphere, use [`nested_atmosphere_model`](@ref) (Breeze
extension), which derives the lateral BCs and Davies relaxation from the parent and wraps the result
in a `NestedModel`.
"""
function NumericalEarth.Atmospheres.atmosphere_model(grid;
                                                          surface_pressure = 101325,
                                                          potential_temperature = 285,
                                                          thermodynamic_constants = ThermodynamicConstants(eltype(grid)),
                                                          dynamics = AnelasticDynamics(ReferenceState(grid, thermodynamic_constants;
                                                                                                      surface_pressure, potential_temperature)),
                                                          microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()),
                                                          momentum_advection = Oceananigans.WENO(order=9),
                                                          scalar_advection = Oceananigans.WENO(order=5),
                                                          boundary_conditions = NamedTuple(),
                                                          coriolis = nothing,
                                                          forcing = NamedTuple(),
                                                          closure = nothing,
                                                          clock = Oceananigans.TimeSteppers.Clock{eltype(grid)}(time = 0),
                                                          radiation = CoupledRadiation())

    if radiation isa Breeze.RadiativeTransferModel
        throw(ArgumentError("`atmosphere_simulation` does not accept a `Breeze.RadiativeTransferModel`. " *
                            "Pass the RTM to the coupled-model constructor instead, e.g. " *
                            "`AtmosphereLandModel(atmos, land; radiation = rtm)`."))
    end

    # Create 2D coupling-flux fields populated by the ESM coupler each step.
    ρτˣ = Field{Center, Center, Nothing}(grid)
    ρτʸ = Field{Center, Center, Nothing}(grid)
    Jᵉ  = Field{Center, Center, Nothing}(grid)
    Jᵛ  = Field{Center, Center, Nothing}(grid)

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

    return AtmosphereModel(grid;
                           dynamics, microphysics,
                           momentum_advection, scalar_advection,
                           boundary_conditions,
                           coriolis, forcing, closure, clock,
                           radiation)
end

"""
$(TYPEDSIGNATURES)

Wrap [`atmosphere_model`](@ref) in an Oceananigans `Simulation`. `Δt` defaults to `Inf` because the
coupled `Simulation` / `NestedModel` owns the time step in coupled use; pass a finite `Δt` to `run!`
this `Simulation` directly. All other keyword arguments forward to `atmosphere_model`.
"""
NumericalEarth.Atmospheres.atmosphere_simulation(grid; Δt = Inf, kw...) =
    Simulation(NumericalEarth.Atmospheres.atmosphere_model(grid; kw...); Δt, verbose = false)
