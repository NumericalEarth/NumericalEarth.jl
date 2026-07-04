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

using NumericalEarth: BoundingBox, Metadatum, regrid_topography, surface_elevation
using NumericalEarth.Atmospheres: PrescribedAtmosphere
using NumericalEarth.DataWrangling: default_download_directory, matching_single_level_dataset,
                                    native_resolution
using NumericalEarth.NestedModels: NestedModel, parent_boundary_conditions, parent_forcings,
                                   blend_parent_terrain!
using Oceananigans: Oceananigans, WENO
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: ValueBoundaryCondition, NormalFlowBoundaryCondition
using Oceananigans.Coriolis: SphericalCoriolis
using Oceananigans.Fields: AbstractField, CenterField, Field, XFaceField, YFaceField,
                           compute!, interior, interpolate!, set!
using Oceananigans.Forcings: Relaxation
using Oceananigans.Grids: znode, λnodes, φnodes, Bounded, Center, Face, LatitudeLongitudeGrid
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Units: Time
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

# Cosine-ramp Davies mask: 1 at the lateral walls, ramping to 0 over the outermost `width` cells.
# Captures plain numbers so the closure is GPU-compilable.
function davies_relaxation_mask(grid, width)
    λ₁, λ₂ = extrema(λnodes(grid, Face(), Center(), Center()))
    φ₁, φ₂ = extrema(φnodes(grid, Center(), Face(), Center()))
    Nx, Ny, _ = size(grid)
    w = width * max((λ₂ - λ₁) / Nx, (φ₂ - φ₁) / Ny)
    return (λ, φ, z) -> begin
        d = min(λ - λ₁, λ₂ - λ, φ - φ₁, φ₂ - φ)
        d >= w && return zero(λ)
        return oftype(λ, (1 + cos(π * d / w)) / 2)
    end
end

# Cubic-ramp (smoothstep) Rayleigh mask over the top `depth` metres of the domain, for the ρw lid sponge.
# `z_top` is read once host-side under `@allowscalar`: a `znode` on a terrain-following GPU grid
# indexes the (device) terrain arrays, which is otherwise disallowed.
function lid_sponge_mask(grid, depth)
    z_top = @allowscalar znode(1, 1, size(grid, 3) + 1, grid, Center(), Center(), Face())
    d = convert(eltype(grid), depth)
    return (λ, φ, z) -> (s = clamp((z - (z_top - d)) / d, zero(z), one(z)); s * s * (3 - 2s))
end

# Default lid-sponge depth: the top ~25% of the vertical extent (≈5 km for a ~19.5 km column, base ≈15 km).
# Thin enough to leave the deep-convective layer (~12–16 km) undamped, yet deep enough to absorb genuine
# vertically-propagating gravity/acoustic reflection off the rigid top — the wall-mode source is removed
# by the consistent (exchanger-derived) IC, so the sponge no longer needs to be strong/deep. Uses the grid's
# vertical extent `Lz` (not the top node height, which on a terrain-following grid includes the terrain
# offset) so it scales correctly regardless of where the bottom sits.
default_lid_depth(grid) = convert(eltype(grid), grid.Lz / 4)

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
over `relaxation_mask` (default: a cosine ramp over the outermost `relaxation_width` cells) — interpolate
those precomputed prognostics (via `parent_boundary_conditions` / `parent_forcings`).

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
            relaxation_width = 5,
            relaxation_mask = davies_relaxation_mask(child_grid, relaxation_width),
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
            damping_depth = default_lid_depth(child_grid),
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

    # Lateral BCs: interpolate the precomputed prognostics at the boundary face. Momentum is prescribed on
    # every side, but the BC *type* is per-side: `NormalFlowBoundaryCondition` on the wall-normal side
    # (where the velocity's face coincides with the boundary), `ValueBoundaryCondition` on the tangential
    # side (prescribing the parent's tangential velocity in the halo — `NormalFlowBC` there leaves it
    # under-constrained and injects spurious near-boundary convergence). `ρᵈ`/energy/moisture are Center
    # scalars (`ValueBoundaryCondition` on all sides, since `NormalFlowBC` overwrites the first interior cell
    # asymmetrically for Center fields). The energy BC is keyed `ρe` (Breeze's energy-BC interface): it merges
    # with the coupling's bottom energy-flux BC on the same field, and for a potential-temperature formulation
    # Breeze routes the (Value) `ρθ` boundary values through unchanged. `ρθ` and `ρe` must not both carry BCs.
    dry_bc_variables = (ρᵈ = prognostic.ρᵈ, ρu = prognostic.ρu, ρv = prognostic.ρv, ρe = prognostic.ρθ)
    moist_bc_variables = NamedTuple{tuple(moisture_name)}(tuple(ρqᵛ))
    bc_variables = merge(dry_bc_variables, moist_bc_variables)

    density_and_energy_types = (ρᵈ = ValueBoundaryCondition, ρe = ValueBoundaryCondition)
    momentum_types = (ρu = (west = NormalFlowBoundaryCondition, east = NormalFlowBoundaryCondition,
                            south = ValueBoundaryCondition, north = ValueBoundaryCondition),
                      ρv = (west = ValueBoundaryCondition, east = ValueBoundaryCondition,
                            south = NormalFlowBoundaryCondition, north = NormalFlowBoundaryCondition))
    moist_types = NamedTuple{tuple(moisture_name)}(tuple(ValueBoundaryCondition))
    bc_types = merge(density_and_energy_types, momentum_types, moist_types)

    nested_bcs = parent_boundary_conditions(child_grid; variables = bc_variables, sides, bc_types)

    # Interior Davies relaxation toward the precomputed (density-weighted) prognostics. Oceananigans'
    # FTS `Relaxation` calls `mask(x, y, z)`, so wrap a scalar mask in a callable. The density `ρᵈ` is
    # relaxed alongside the momentum/energy/moisture — the mass field, following WRF (nudges dry mass μ)
    # and MPAS (nudges ρ); without it the un-relaxed near-wall density drives a persistent lateral-wall
    # residual (ρw creep) that a top sponge cannot damp.
    relax_mask = relaxation_mask isa Number ? Returns(relaxation_mask) : relaxation_mask
    davies = if isnothing(relaxation_rate)
        NamedTuple()
    else
        dry_forcing_variables = (ρᵈ = prognostic.ρᵈ, ρθ = prognostic.ρθ, ρu = prognostic.ρu, ρv = prognostic.ρv)
        moist_forcing_variables = NamedTuple{tuple(moisture_name)}(tuple(ρqᵛ))
        variables = merge(dry_forcing_variables, moist_forcing_variables)
        parent_forcings(; variables, rate = relaxation_rate, mask = relax_mask)
    end

    # ρw Rayleigh lid sponge: relax vertical momentum toward zero over the top `damping_depth` metres.
    lid_sponge = (; ρw = Relaxation(rate = damping_rate, mask = lid_sponge_mask(child_grid, damping_depth)))

    # initialize = false: the resting-state construction default would survive into
    # `initialize_nested_child!` and destabilize the adiabatic balance twin — the child's full
    # state (and reference) is derived from the parent instead.
    child = NumericalEarth.Atmospheres.atmosphere_model(child_grid;
                thermodynamic_constants, microphysics, momentum_advection, coriolis, dynamics,
                boundary_conditions = merge_boundary_conditions(nested_bcs, NamedTuple(boundary_conditions)),
                forcing = merge(lid_sponge, davies, NamedTuple(forcing)),
                initialize = false,
                kw...)

    return NestedModel(parent_atmosphere, child, exchanger)
end

# Domain-mean dataset surface pressure at `date`, regridded onto the child grid — anchors the
# default compressible dynamics' hydrostatic reference to the parent state.
function mean_surface_pressure(dataset, child_grid, date, dir)
    single_level_dataset = matching_single_level_dataset(dataset)
    p₀ = Field{Center, Center, Nothing}(child_grid)
    set!(p₀, Metadatum(:surface_pressure; dataset = single_level_dataset, date,
                       region = BoundingBox(child_grid), dir))
    return sum(interior(p₀)) / length(interior(p₀))
end

"""
    nested_atmosphere_model(child_grid, parent_dataset; dates, kw...)

Build the parent `PrescribedAtmosphere`, nest a Breeze child in it, and initialize the child from
`parent_dataset` at `first(dates)` — the returned model is ready to step. The parent spans
`child_grid`'s bounding box padded by `parent_padding` (default two of `parent_dataset`'s native
cells, margin for the lateral-BC interpolation stencils) at `dates`, on `parent_dataset`'s native
grid. Unless given, the default dynamics' `surface_pressure` anchor is the domain-mean dataset surface
pressure over the child at `first(dates)`. Remaining keyword arguments flow to
`nested_atmosphere_model(parent, child_grid; kw...)`.
"""
function NumericalEarth.NestedModels.nested_atmosphere_model(child_grid, parent_dataset;
            dates,
            dir = default_download_directory(parent_dataset),
            parent_padding = 2 * native_resolution(parent_dataset),
            surface_pressure = nothing,
            kw...)

    parent_region = BoundingBox(child_grid; padding = parent_padding)
    parent_atmosphere = PrescribedAtmosphere(parent_region, dates, parent_dataset;
                                             architecture = architecture(child_grid), dir)

    if isnothing(surface_pressure)
        surface_pressure = mean_surface_pressure(parent_dataset, child_grid, first(dates), dir)
    end

    nested_model = NumericalEarth.NestedModels.nested_atmosphere_model(parent_atmosphere, child_grid;
                                                                       surface_pressure, kw...)
    initialize_nested_child!(nested_model, parent_dataset, first(dates), dir)
    return nested_model
end

NumericalEarth.Atmospheres.bulk_drag(model::NestedModel; kw...) =
    NumericalEarth.Atmospheres.bulk_drag(model.child; kw...)

# Initialize the nested child from the exchanger's parent-derived prognostics (the SAME state that drives
# the lateral boundaries), interpolated to the child interior — so the interior IC and the prescribed
# boundary agree at the walls (no standing pressure/density jump). Recompute the Exner reference from the
# domain-mean state, graft ρw ← ρw − ρw̃ so the flow follows the terrain, and spin ρw into nonhydrostatic
# balance. `set!(…; balancer = true)` runs Breeze's adiabatic (FV3 `na_init`) balance on a stripped,
# memory-sharing twin (no microphysics/sponge/forcing) at an automatically-derived acoustic-CFL step.
function initialize_nested_child!(nested_model, dataset, date, dir)
    child = nested_model.child
    child_grid = child.grid
    prognostic = nested_model.exchanger.prognostic
    t₀ = first(prognostic.ρᵈ.times)

    # Interpolate each exchanger prognostic (parent grid, initial time) to the child interior. Using the
    # SAME parent-derived prognostics that drive the lateral boundaries — via the same `interpolate!` —
    # makes the interior IC and the prescribed boundary agree at the walls, so there is no standing
    # density/pressure jump to force spurious vertical velocity. The adiabatic balancer below then spins
    # up ρw from this consistent state.
    to_child(fts) = (field = CenterField(child_grid); interpolate!(field, fts[Time(t₀)]); field)
    ρᵈ  = to_child(prognostic.ρᵈ)
    ρθ  = to_child(prognostic.ρθ)
    ρqᵛ = to_child(prognostic.ρqᵛ)
    ρu  = to_child(prognostic.ρu)
    ρv  = to_child(prognostic.ρv)

    # Recover the specific state from the density-weighted prognostics (dry-weighted momentum/energy,
    # total-weighted vapor); `ρ` is the total density set! expects.
    ρ   = compute!(Field(ρᵈ + ρqᵛ))
    qᵗ  = compute!(Field(ρqᵛ / ρ))
    θˡⁱ = compute!(Field(ρθ / ρᵈ))
    u   = XFaceField(child_grid); interpolate!(u, compute!(Field(ρu / ρᵈ)))
    v   = YFaceField(child_grid); interpolate!(v, compute!(Field(ρv / ρᵈ)))

    set!(nested_model; ρ, u, v, qᵗ, θˡⁱ, compute_reference_state = true)

    # Consistent-w: graft ρw ← ρw − ρw̃ so the contravariant w̃ ≈ 0 (the initial flow follows the ground).
    update_state!(nested_model)
    if !isnothing(child.dynamics.contravariant_vertical_momentum)
        interior(child.momentum.ρw) .-= interior(child.dynamics.contravariant_vertical_momentum)
        update_state!(nested_model)
    end

    set!(nested_model; balancer = true)   # adiabatic (DFI) balance at Breeze's auto acoustic-CFL step
    return nested_model
end
