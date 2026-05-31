using Oceananigans.Fields: Field
using Oceananigans.Grids: Center

using Breeze: ThermodynamicConstants, ReferenceState, AnelasticDynamics,
              SaturationAdjustment, WarmPhaseEquilibrium,
              AtmosphereModel, moisture_prognostic_name

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
                          coriolis = nothing,
                          forcing = NamedTuple(),
                          closure = nothing,
                          clock = Clock{eltype(grid)}(time=0),
                          Δt = Inf)

Construct an Oceananigans `Simulation` wrapping a Breeze `AtmosphereModel`,
with sensible defaults for coupled simulations. Mirrors the role of
[`ocean_simulation`](@ref).

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

Returns the `Simulation` so callers can attach output writers, callbacks, or
later wrap inside a coupled `EarthSystemModel`. The inner `Δt` defaults to
`Inf` since the *coupled* `Simulation` (around an `EarthSystemModel`) owns
the time step in coupled use; if you wrap this `Simulation` directly in a
`run!`, pass a finite `Δt`.
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
                                                          coriolis = nothing,
                                                          forcing = NamedTuple(),
                                                          closure = nothing,
                                                          clock = Oceananigans.TimeSteppers.Clock{eltype(grid)}(time = 0),
                                                          Δt = Inf,
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

    # User BCs override coupling defaults.
    boundary_conditions = merge(coupling_bcs, boundary_conditions)

    atmosphere_model = AtmosphereModel(grid;
                                       dynamics, microphysics,
                                       momentum_advection, scalar_advection,
                                       boundary_conditions,
                                       coriolis, forcing, closure, clock,
                                       radiation)

    return Simulation(atmosphere_model; Δt, verbose = false)
end
