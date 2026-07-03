using KernelAbstractions: @kernel, @index
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: Field
using Oceananigans.Grids: znode, Center, Face
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, DefaultBoundaryCondition
using Oceananigans.Utils: launch!

using Breeze: ThermodynamicConstants, CompressibleDynamics,
              SaturationAdjustment, WarmPhaseEquilibrium,
              AtmosphereModel, moisture_prognostic_name, HydrostaticallyBalancedDensity

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
                     dynamics = CompressibleDynamics(; surface_pressure,
                                                     reference_potential_temperature = potential_temperature),
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
                                                     dynamics = CompressibleDynamics(; surface_pressure,
                                                                                     reference_potential_temperature = potential_temperature),
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

    model = AtmosphereModel(grid;
                            dynamics, microphysics, thermodynamic_constants,
                            momentum_advection, scalar_advection,
                            boundary_conditions,
                            coriolis, forcing, closure, clock,
                            radiation)

    # Breeze ≥0.7 no longer derives the prognostic density from the reference state inside a bare
    # `set!(θ, …)` — an uninitialized model carries ρ ≡ 0, which NaNs anything that divides by density
    # (e.g. the MOST surface-flux coupling). Initialize to a resting, hydrostatically balanced state at
    # the reference θ so the returned atmosphere is valid; a later user `set!` of θ/velocities (without
    # a `ρ` entry) preserves this density.
    set!(model; θ = potential_temperature,
                ρ = HydrostaticallyBalancedDensity(; surface_pressure))

    return model
end

"""
$(TYPEDSIGNATURES)

Wrap [`atmosphere_model`](@ref) in an Oceananigans `Simulation`. `Δt` defaults to `Inf` because the
coupled `Simulation` / `NestedModel` owns the time step in coupled use; pass a finite `Δt` to `run!`
this `Simulation` directly. All other keyword arguments forward to `atmosphere_model`.
"""
NumericalEarth.Atmospheres.atmosphere_simulation(grid; Δt = Inf, kw...) =
    Simulation(NumericalEarth.Atmospheres.atmosphere_model(grid; kw...); Δt, verbose = false)

#####
##### bulk_drag: bulk neutral log-law surface stress into the coupling ρτˣ/ρτʸ fields
#####

@kernel function _drag_coefficient!(Cᵈ, grid, κ, z₀)
    i, j = @index(Global, NTuple)
    @inbounds begin
        z₁ = znode(i, j, 1, grid, Center(), Center(), Center()) - znode(i, j, 1, grid, Center(), Center(), Face())
        Cᵈ[i, j, 1] = (κ / log(z₁ / z₀))^2
    end
end

@kernel function _bulk_drag!(ρτˣ, ρτʸ, u, v, ρ, Cᵈ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        uᶜ = (u[i, j, 1] + u[i+1, j, 1]) / 2
        vᶜ = (v[i, j, 1] + v[i, j+1, 1]) / 2
        U = sqrt(uᶜ^2 + vᶜ^2)
        ρτˣ[i, j, 1] = -ρ[i, j, 1] * Cᵈ[i, j, 1] * U * uᶜ
        ρτʸ[i, j, 1] = -ρ[i, j, 1] * Cᵈ[i, j, 1] * U * vᶜ
    end
end

function NumericalEarth.Atmospheres.bulk_drag(model::AtmosphereModel; roughness_length = 0.1, von_karman_constant = 0.4)
    grid = model.grid
    FT = eltype(grid)
    Cᵈ = Field{Center, Center, Nothing}(grid)
    launch!(architecture(grid), grid, :xy, _drag_coefficient!,
            Cᵈ, grid, convert(FT, von_karman_constant), convert(FT, roughness_length))

    ρτˣ = model.momentum.ρu.boundary_conditions.bottom.condition
    ρτʸ = model.momentum.ρv.boundary_conditions.bottom.condition
    u, v = model.velocities.u, model.velocities.v
    ρ = model.dynamics.total_density

    function apply_bulk_drag!(simulation)
        launch!(architecture(grid), grid, :xy, _bulk_drag!, ρτˣ, ρτʸ, u, v, ρ, Cᵈ)
        return nothing
    end

    return apply_bulk_drag!
end
