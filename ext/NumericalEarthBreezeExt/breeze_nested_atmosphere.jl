#####
##### Nested-atmosphere model: a Breeze child driven by a parent `PrescribedAtmosphere`.
#####
#
# The child's prognostic variables (dry density `ρᵈ`, momentum densities `ρu`/`ρv`, potential-temperature
# density `ρθ`, vapor density `ρqᵛ`) are precomputed from the parent's raw state ON THE PARENT GRID and
# stored as `FieldTimeSeries` (see `breeze_state_exchanger.jl`). The child's lateral boundary conditions
# and interior Davies relaxation then just interpolate those precomputed prognostics in space + time —
# there is no thermodynamic combine inside the tendency/halo kernels. Both layers reuse the generic
# FTS-driven `parent_boundary_conditions` / `parent_forcings` builders, so a child forcing/BC specializes
# on a plain `FieldTimeSeries` (the same type Breeze compiles for any FTS forcing).

using NumericalEarth.Atmospheres: PrescribedAtmosphere
using NumericalEarth.NestedModels: NestedModel, parent_boundary_conditions, parent_forcings
using Oceananigans: WENO
using Oceananigans.BoundaryConditions: ValueBoundaryCondition
using Oceananigans.Coriolis: SphericalCoriolis
using Oceananigans.Forcings: Relaxation
using Oceananigans.Grids: znode, Center, Face
using GPUArraysCore: @allowscalar
using Breeze: CompressibleDynamics, SplitExplicitTimeDiscretization, UpperSponge, NoDivergenceDamping,
              MixedPhaseEquilibrium, materialize_terrain!, moisture_prognostic_name

# Default child microphysics: 1-moment bulk mixed-phase (rain + snow) precipitation with
# saturation-adjustment cloud formation when Breeze's `CloudMicrophysics` extension is loaded,
# else the Breeze-native warm-phase saturation-adjustment scheme. Resolved at call time, so a
# caller that `using CloudMicrophysics` gets `OneMomentCloudMicrophysics` automatically.
function default_nested_microphysics()
    ext = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
    isnothing(ext) && return SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
    return ext.OneMomentCloudMicrophysics(cloud_formation = SaturationAdjustment(equilibrium = MixedPhaseEquilibrium()))
end

# Cubic-ramp (smoothstep) Rayleigh mask over the top `depth` metres of the domain, for the ρw lid sponge.
# `z_top` is read once host-side under `@allowscalar`: a `znode` on a terrain-following GPU grid
# indexes the (device) terrain arrays, which is otherwise disallowed.
function lid_sponge_mask(grid, depth)
    z_top = @allowscalar znode(1, 1, size(grid, 3) + 1, grid, Center(), Center(), Face())
    d = convert(eltype(grid), depth)
    return (λ, φ, z) -> (s = clamp((z - (z_top - d)) / d, zero(z), one(z)); s * s * (3 - 2s))
end

# Default child dynamics: compressible with split-explicit acoustic substepping, an `UpperSponge`
# Rayleigh layer over the top `damping_depth` metres at `damping_rate`, and no divergence damping
# (its (ρθ)′-proxy damper injects a spurious force on an unbalanced cold start). When given,
# `surface_pressure`/`reference_potential_temperature` anchor the hydrostatic reference and the
# perturbation-form pressure-gradient reference profile.
function default_nested_dynamics(grid; surface_pressure, reference_potential_temperature, damping_rate, damping_depth)
    time_discretization = SplitExplicitTimeDiscretization(sponge = UpperSponge(; damping_rate, depth = damping_depth),
                                                          damping = NoDivergenceDamping())
    kw = (;)
    isnothing(surface_pressure)                || (kw = merge(kw, (; surface_pressure)))
    isnothing(reference_potential_temperature) || (kw = merge(kw, (; reference_potential_temperature)))
    return CompressibleDynamics(time_discretization; kw...)
end

"""
$(TYPEDSIGNATURES)

Build a Breeze child atmosphere over `child_grid` nested in `parent_atmosphere`, wrapped in a
`NestedModel`. The child's prognostics (`ρᵈ, ρu, ρv, ρθ, <moisture>`) are precomputed from the parent's
raw state on the parent grid as `FieldTimeSeries` (see `child_prognostic_field_time_series`); the child's
lateral boundary conditions — and, when `relaxation_rate` (s⁻¹) is given, its interior Davies relaxation
over `relaxation_mask` — interpolate those precomputed prognostics (via `parent_boundary_conditions` /
`parent_forcings`).

Cloud/ice inputs to the combine default to the parent's `qᶜˡ`/`qᶜⁱ` but may be supplied from any source
(another dataset, or `nothing` ⇒ omitted, so `qᵗ = qᵛ`) via `qᶜˡ_source`/`qᶜⁱ_source`.

Provides sensible, overridable physics defaults: `microphysics` (1-moment mixed-phase when
`CloudMicrophysics` is loaded), `momentum_advection = WENO(order=9)`, `coriolis = SphericalCoriolis()`,
and a compressible split-explicit `dynamics` with an `UpperSponge` over the top `damping_depth` m at
`damping_rate`; a matching ρw Rayleigh lid sponge (`Relaxation` toward zero) is added to `forcing`. Pass
`surface_pressure`/`reference_potential_temperature` to anchor the default dynamics. Any
`boundary_conditions`/`forcing` the caller passes are merged with the parent-derived ones (caller wins).

When `terrain` is given (anything `materialize_terrain!` accepts), the child grid's terrain-following
coordinate is materialized in place before the model is built.
"""
function NumericalEarth.NestedModels.nested_atmosphere_model(
            parent_atmosphere::PrescribedAtmosphere, child_grid;
            relaxation_rate = nothing,
            relaxation_mask = 1,
            sides = (:west, :east, :south, :north),
            thermodynamic_constants = ThermodynamicConstants(eltype(child_grid)),
            surface_pressure = nothing,
            reference_potential_temperature = nothing,
            terrain = nothing,
            qᶜˡ_source = parent_atmosphere.microphysical_variables.qᶜˡ,
            qᶜⁱ_source = parent_atmosphere.microphysical_variables.qᶜⁱ,
            microphysics = default_nested_microphysics(),
            momentum_advection = WENO(order = 9),
            coriolis = SphericalCoriolis(),
            damping_rate = 1/5,
            damping_depth = 3000,
            dynamics = default_nested_dynamics(child_grid; surface_pressure, reference_potential_temperature,
                                               damping_rate, damping_depth),
            boundary_conditions = NamedTuple(),
            forcing = NamedTuple(),
            kw...)

    isnothing(terrain) || materialize_terrain!(child_grid, terrain)

    moisture_name = moisture_prognostic_name(microphysics)
    pˢᵗ = dynamics.standard_pressure

    # Precompute the child prognostics on the parent grid (combine-then-interpolate); the exchanger owns
    # them and refreshes them from the parent each step via `exchange_state!`.
    exchanger  = state_exchanger(parent_atmosphere, pˢᵗ, thermodynamic_constants; qᶜˡ_source, qᶜⁱ_source)
    prognostic = exchanger.prognostic

    ρqᵛ = prognostic.ρqᵛ

    # Lateral BCs: interpolate the precomputed prognostics at the boundary face. `ρu`/`ρv` are Face-normal
    # (default `NormalFlowBoundaryCondition`); `ρᵈ`/energy/moisture are Center scalars (`ValueBoundaryCondition`,
    # since `NormalFlowBC` overwrites the first interior cell asymmetrically for Center fields). The energy
    # BC is keyed `ρe` (Breeze's energy-BC interface): it merges with the coupling's bottom energy-flux BC on
    # the same field, and for a potential-temperature formulation Breeze routes the (Value) `ρθ` boundary
    # values through unchanged. `ρθ` and `ρe` must not both carry BCs.
    bc_variables = merge((; ρᵈ = prognostic.ρᵈ, ρu = prognostic.ρu, ρv = prognostic.ρv, ρe = prognostic.ρθ),
                         NamedTuple{tuple(moisture_name)}(tuple(ρqᵛ)))
    bc_types = merge((; ρᵈ = ValueBoundaryCondition, ρe = ValueBoundaryCondition),
                     NamedTuple{tuple(moisture_name)}(tuple(ValueBoundaryCondition)))
    nested_bcs = parent_boundary_conditions(child_grid; variables = bc_variables, sides, bc_types)

    # Interior Davies relaxation toward the precomputed (density-weighted) prognostics. Oceananigans'
    # FTS `Relaxation` calls `mask(x, y, z)`, so wrap a scalar mask in a callable.
    relax_mask = relaxation_mask isa Number ? Returns(relaxation_mask) : relaxation_mask
    davies = isnothing(relaxation_rate) ? (;) :
             parent_forcings(variables = merge((; ρθ = prognostic.ρθ, ρu = prognostic.ρu, ρv = prognostic.ρv),
                                               NamedTuple{tuple(moisture_name)}(tuple(ρqᵛ))),
                             rate = relaxation_rate, mask = relax_mask)

    # ρw Rayleigh lid sponge: relax vertical momentum toward zero over the top `damping_depth` metres.
    lid_sponge = (; ρw = Relaxation(rate = damping_rate, mask = lid_sponge_mask(child_grid, damping_depth)))

    child = NumericalEarth.Atmospheres.atmosphere_model(child_grid;
                thermodynamic_constants, microphysics, momentum_advection, coriolis, dynamics,
                boundary_conditions = merge_boundary_conditions(nested_bcs, NamedTuple(boundary_conditions)),
                forcing = merge(lid_sponge, davies, NamedTuple(forcing)),
                kw...)

    return NestedModel(parent_atmosphere, child, exchanger)
end
