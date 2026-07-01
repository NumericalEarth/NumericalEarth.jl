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

using NumericalEarth: regrid_topography, surface_elevation
using NumericalEarth.Atmospheres: PrescribedAtmosphere
using NumericalEarth.NestedModels: NestedModel, parent_boundary_conditions, parent_forcings,
                                   blend_parent_terrain!
using Oceananigans: WENO
using Oceananigans.BoundaryConditions: ValueBoundaryCondition
using Oceananigans.Coriolis: SphericalCoriolis
using Oceananigans.Fields: AbstractField, Field, interpolate!
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

# Child terrain for the nested LAM: an elevation `Field` passes through; anything else is treated
# as a topography dataset and regridded onto the child grid. When the parent knows its surface
# elevation (e.g. an ERA5 `PressureLevelGrid`), the child elevation is blended toward it over the
# outermost `blend_width` cells, so the terrain at the open boundaries is consistent with the
# orography the parent state was produced with.
function materialize_nested_terrain!(child_grid, terrain, parent_atmosphere, blend_width)
    elevation = terrain isa AbstractField ? terrain : regrid_topography(child_grid; dataset = terrain)
    parent_surface = surface_elevation(parent_atmosphere)
    if !isnothing(parent_surface) && blend_width > 0
        parent_elevation = Field{Center, Center, Nothing}(child_grid)
        interpolate!(parent_elevation, parent_surface)
        blend_parent_terrain!(elevation, parent_elevation; width = blend_width)
    end
    return materialize_terrain!(child_grid, elevation)
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
via `parent_condensates`, a `NamedTuple` with `qᶜˡ`/`qᶜⁱ` entries. Either entry — or the whole
`parent_condensates` — may be `nothing` (⇒ omitted, so `qᵗ = qᵛ`).

Provides sensible, overridable physics defaults: `microphysics` (1-moment mixed-phase when
`CloudMicrophysics` is loaded), `momentum_advection = WENO(order=9)`, `coriolis = SphericalCoriolis()`,
and a compressible split-explicit `dynamics` with an `UpperSponge` over the top `damping_depth` m at
`damping_rate`; a matching ρw Rayleigh lid sponge (`Relaxation` toward zero) is added to `forcing`. Pass
`surface_pressure`/`reference_potential_temperature` to anchor the default dynamics. Any
`boundary_conditions`/`forcing` the caller passes are merged with the parent-derived ones (caller wins).

When `terrain` is given — an elevation `Field`, or a topography dataset (e.g. `ETOPO2022()`) that is
regridded onto the child grid — the child grid's terrain-following coordinate is materialized in place
before the model is built. If the parent knows its surface elevation ([`surface_elevation`](@ref)), the
child elevation is first blended toward the parent's over the outermost `terrain_blend_width` cells,
so the terrain at the open boundaries matches the orography the parent state was produced with.
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
            terrain_blend_width = 5,
            parent_condensates = (qᶜˡ = parent_atmosphere.microphysical_variables.qᶜˡ,
                                  qᶜⁱ = parent_atmosphere.microphysical_variables.qᶜⁱ),
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

    isnothing(terrain) || materialize_nested_terrain!(child_grid, terrain, parent_atmosphere, terrain_blend_width)

    moisture_name = moisture_prognostic_name(microphysics)
    pˢᵗ = dynamics.standard_pressure

    # Precompute the child prognostics on the parent grid (combine-then-interpolate); the exchanger owns
    # them and refreshes them from the parent each step via `exchange_state!`.
    condensates = isnothing(parent_condensates) ? (qᶜˡ = nothing, qᶜⁱ = nothing) : parent_condensates
    exchanger  = state_exchanger(parent_atmosphere, pˢᵗ, thermodynamic_constants; condensates)
    prognostic = exchanger.prognostic

    ρqᵛ = prognostic.ρqᵛ

    # Lateral BCs: interpolate the precomputed prognostics at the boundary face. `ρu`/`ρv` are Face-normal
    # (default `NormalFlowBoundaryCondition`); `ρᵈ`/energy/moisture are Center scalars (`ValueBoundaryCondition`,
    # since `NormalFlowBC` overwrites the first interior cell asymmetrically for Center fields). The energy
    # BC is keyed `ρe` (Breeze's energy-BC interface): it merges with the coupling's bottom energy-flux BC on
    # the same field, and for a potential-temperature formulation Breeze routes the (Value) `ρθ` boundary
    # values through unchanged. `ρθ` and `ρe` must not both carry BCs.
    dry_bc_variables = (ρᵈ = prognostic.ρᵈ, ρu = prognostic.ρu, ρv = prognostic.ρv, ρe = prognostic.ρθ)
    moist_bc_variables = NamedTuple{tuple(moisture_name)}(tuple(ρqᵛ))
    bc_variables = merge(dry_bc_variables, moist_bc_variables)

    density_and_energy_types = (ρᵈ = ValueBoundaryCondition, ρe = ValueBoundaryCondition)
    moist_types = NamedTuple{tuple(moisture_name)}(tuple(ValueBoundaryCondition))
    bc_types = merge(density_and_energy_types, moist_types)

    nested_bcs = parent_boundary_conditions(child_grid; variables = bc_variables, sides, bc_types)

    # Interior Davies relaxation toward the precomputed (density-weighted) prognostics. Oceananigans'
    # FTS `Relaxation` calls `mask(x, y, z)`, so wrap a scalar mask in a callable.
    relax_mask = relaxation_mask isa Number ? Returns(relaxation_mask) : relaxation_mask
    davies = if isnothing(relaxation_rate)
        NamedTuple()
    else
        dry_forcing_variables = (ρθ = prognostic.ρθ, ρu = prognostic.ρu, ρv = prognostic.ρv)
        moist_forcing_variables = NamedTuple{tuple(moisture_name)}(tuple(ρqᵛ))
        variables = merge(dry_forcing_variables, moist_forcing_variables)
        parent_forcings(; variables, rate = relaxation_rate, mask = relax_mask)
    end

    # ρw Rayleigh lid sponge: relax vertical momentum toward zero over the top `damping_depth` metres.
    lid_sponge = (; ρw = Relaxation(rate = damping_rate, mask = lid_sponge_mask(child_grid, damping_depth)))

    child = NumericalEarth.Atmospheres.atmosphere_model(child_grid;
                thermodynamic_constants, microphysics, momentum_advection, coriolis, dynamics,
                boundary_conditions = merge_boundary_conditions(nested_bcs, NamedTuple(boundary_conditions)),
                forcing = merge(lid_sponge, davies, NamedTuple(forcing)),
                kw...)

    return NestedModel(parent_atmosphere, child, exchanger)
end
