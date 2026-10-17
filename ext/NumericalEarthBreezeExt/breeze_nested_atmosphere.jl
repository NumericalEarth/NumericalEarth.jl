#####
##### Nested-atmosphere model: a Breeze child driven by a parent `PrescribedAtmosphere`.
#####

# The child's prognostic variables (dry density `ρᵈ`, momentum densities `ρu`/`ρv`, potential-temperature
# density `ρθ`, moisture density `ρqᵛᵉ`) are precomputed from the parent's raw state ON THE PARENT GRID and
# stored as `FieldTimeSeries` (see `breeze_state_exchanger.jl`). The child's lateral boundary conditions
# and interior Davies relaxation then just interpolate those precomputed prognostics in space + time —
# there is no thermodynamic combine inside the tendency/halo kernels. Both layers reuse the generic
# FTS-driven `parent_boundary_conditions` / `parent_forcings` builders, so a child forcing/BC specializes
# on a plain `FieldTimeSeries` (the same type Breeze compiles for any FTS forcing).

using NumericalEarth:
    BoundingBox,
    Metadatum,
    regrid_topography,
    smooth_topography!,
    surface_elevation

using NumericalEarth.Atmospheres: PrescribedAtmosphere
using NumericalEarth.EarthSystemModels: BoundaryPrescribedComponent, boundary_strips, relaxation_zone_width
using NumericalEarth.DataWrangling: default_download_directory, default_horizontal_padding, matching_single_level_dataset,
                                    expand_dates
using NumericalEarth.NestedModels: NestedModel, parent_boundary_conditions, parent_forcings, blend_parent_terrain!

using Oceananigans:
    Oceananigans,
    WENO,
    ValueBoundaryCondition,
    NormalFlowBoundaryCondition,
    Field,
    Center, Face,
    TendencyCallsite,
    set!

using Oceananigans.Architectures: architecture
using Oceananigans.DistributedComputations: all_reduce
using Oceananigans.Coriolis: SphericalCoriolis
using Oceananigans.Fields: AbstractField, interior, interpolate!
using Oceananigans.Forcings: Relaxation, FieldTimeSeriesTarget
using Oceananigans.Grids: znode, minimum_xspacing, x_domain, y_domain
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Units: Time
using Oceananigans.Utils: KernelParameters
using Oceananigans.Simulations: Callback

using GPUArraysCore: @allowscalar

using Breeze:
    BulkDrag,
    CompressibleDynamics,
    SplitExplicitTimeDiscretization,
    UpperSponge,
    NoDivergenceDamping,
    MixedPhaseEquilibrium,
    SpecificForcing,
    materialize_terrain!,
    moisture_prognostic_name,
    moisture_specific_name

using Breeze.AtmosphereModels: prognostic_field_names, dynamics_density

# Default child microphysics: 1-moment bulk mixed-phase (rain + snow) precipitation with
# saturation-adjustment cloud formation when Breeze's `CloudMicrophysics` extension is loaded,
# else the Breeze-native warm-phase saturation-adjustment scheme. Resolved at call time, so a
# caller that `using CloudMicrophysics` gets `OneMomentCloudMicrophysics` automatically.
function default_nested_microphysics()
    ext = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
    isnothing(ext) && return SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
    return ext.OneMomentCloudMicrophysics(cloud_formation = SaturationAdjustment(equilibrium = MixedPhaseEquilibrium()))
end

const default_relaxation_width = 5

# Ramp shapes (isbits callables) for a nudging zone: weight vs. normalized distance from the wall, s ∈ [0, 1].
# Contract: ramp(0)=1, ramp(1)=0, monotone between.
struct CosineRamp end
struct SmoothStepRamp end

@inline (::CosineRamp)(s)     = (1 + cos(π * s)) / 2
@inline (::SmoothStepRamp)(s) = 1 - s^2 * (3 - 2s)

# Davies mask: 1 at the lateral walls, ramping to 0 over the outermost `width` cells.
# The mask is evaluated per cell per stage inside the forcing kernel, so the default ramp
# is the smoothstep polynomial (2 multiplies) rather than the ~1%-different raised cosine.
function davies_relaxation_mask(grid, width; ramp = SmoothStepRamp())
    # `w` may use local values (spacing is rank-uniform) but the extents must be the GLOBAL
    # domain's, or partitioned ranks would relax at their seams (`all_reduce` is the identity
    # on serial architectures). Each capture is assigned exactly once: reassignment boxes it,
    # and a boxed capture is not isbits — GPU kernels reject it.
    λ₁ˡ, λ₂ˡ = x_domain(grid)
    φ₁ˡ, φ₂ˡ = y_domain(grid)
    w = relaxation_zone_width(grid, width)
    arch = architecture(grid)
    λ₁ = all_reduce(min, λ₁ˡ, arch)
    λ₂ = all_reduce(max, λ₂ˡ, arch)
    φ₁ = all_reduce(min, φ₁ˡ, arch)
    φ₂ = all_reduce(max, φ₂ˡ, arch)
    return (λ, φ, z) -> begin
        d = min(λ - λ₁, λ₂ - λ, φ - φ₁, φ₂ - φ)
        s = clamp(d / w, zero(d), one(d))
        return oftype(d, ramp(s))
    end
end

# Index ranges of the boundary strips: every cell within the relaxation zone of a side. The
# corners belong to the west and east strips, so no cell is relaxed twice.
function boundary_strip_regions(grid, width, sides)
    λ₁, λ₂ = x_domain(grid)
    φ₁, φ₂ = y_domain(grid)
    Nx, Ny, Nz = size(grid)
    w = relaxation_zone_width(grid, width)
    Wx = ceil(Int, w / ((λ₂ - λ₁) / Nx))
    Wy = ceil(Int, w / ((φ₂ - φ₁) / Ny))
    i₁ = :west in sides ? Wx + 1 : 1
    i₂ = :east in sides ? Nx - Wx : Nx
    regions = (west  = KernelParameters(1:Wx, 1:Ny, 1:Nz),
               east  = KernelParameters(Nx-Wx+1:Nx, 1:Ny, 1:Nz),
               south = KernelParameters(i₁:i₂, 1:Wy, 1:Nz),
               north = KernelParameters(i₁:i₂, Ny-Wy+1:Ny, 1:Nz))
    return NamedTuple{sides}(map(side -> regions[side], sides))
end

@kernel function _add_forcing!(G, grid, forcing, clock, model_fields)
    i, j, k = @index(Global, NTuple)
    @inbounds G[i, j, k] += forcing(i, j, k, grid, clock, model_fields)
end

function relax_boundary_strips!(model, strips)
    grid = model.grid
    arch = architecture(grid)
    model_fields = Oceananigans.fields(model)
    for strip in strips, name in keys(strip.forcings)
        launch!(arch, grid, strip.region, _add_forcing!,
                model.timestepper.Gⁿ[name], grid, strip.forcings[name], model.clock, model_fields)
    end
    return nothing
end

# Davies relaxation toward the strips' derived prognostics, applied over each strip's region by a
# tendency callback. Momentum and energy relax the specific state weighted by the child's own coupling
# density; moisture relaxes its density.
function boundary_relaxation(child, prognostic, rate, mask, width)
    child_fields = Oceananigans.prognostic_fields(child)
    moisture_name = moisture_prognostic_name(child.microphysics)
    ρᵈ = dynamics_density(child.dynamics)
    θ = Oceananigans.fields(child.formulation).θ

    relaxation(field, target) = Relaxation(rate, field, mask, FieldTimeSeriesTarget(target, target.grid),
                                           Oceananigans.instantiated_location(field), nothing)
    specific(name, field, target) = SpecificForcing(relaxation(field, target), ρᵈ,
                                                    Oceananigans.instantiated_location(child_fields[name]))

    sides = keys(prognostic.θ)
    regions = boundary_strip_regions(child.grid, width, sides)
    names = (:ρθ, :ρu, :ρv, moisture_name)
    strips = NamedTuple{sides}(
        (forcings = NamedTuple{names}((specific(:ρθ, θ, prognostic.θ[side]),
                                       specific(:ρu, child.velocities.u, prognostic.u[side]),
                                       specific(:ρv, child.velocities.v, prognostic.v[side]),
                                       relaxation(child_fields[moisture_name], prognostic.ρqᵛᵉ[side]))),
         region = regions[side])
        for side in sides)

    return Callback(relax_boundary_strips!; callsite = TendencyCallsite(), parameters = strips)
end

# Cubic-ramp (smoothstep) Rayleigh mask over the top `depth` metres of the domain, for the ρw lid sponge.
# `z_top` is read once host-side under `@allowscalar`: a `znode` on a terrain-following GPU grid
# indexes the (device) terrain arrays, which is otherwise disallowed. `s` is the normalized distance
# below the lid (0 at the top), so the shared ramp contract (1 at s=0) puts the strongest damping at
# the model top.
function lid_sponge_mask(grid, depth; ramp = SmoothStepRamp())
    z_top = @allowscalar znode(1, 1, size(grid, 3) + 1, grid, Center(), Center(), Face())
    d = convert(eltype(grid), depth)
    return (λ, φ, z) -> (s = clamp((z_top - z) / d, zero(z), one(z)); ramp(s))
end

# Default lid-sponge depth: the top ~25% of the vertical extent (≈5 km for a ~19.5 km column, base ≈15 km).
# Thin enough to leave the deep-convective layer (~12–16 km) undamped, yet deep enough to absorb genuine
# vertically-propagating gravity/acoustic reflection off the rigid top — the wall-mode source is removed
# by the consistent (exchanger-derived) IC, so the sponge no longer needs to be strong/deep. Uses the grid's
# vertical extent `Lz` (not the top node height, which on a terrain-following grid includes the terrain
# offset) so it scales correctly regardless of where the bottom sits.
default_lid_depth(grid) = convert(eltype(grid), grid.Lz / 4)

# Default child dynamics: compressible with split-explicit acoustic substepping, an `UpperSponge`
# Rayleigh layer over the top `damping_depth` meters at `damping_rate`, and no divergence damping
# (its (ρθ)′-proxy damper injects a spurious force on an unbalanced cold start). When given,
# `base_pressure`/`reference_potential_temperature` anchor the hydrostatic reference and the
# perturbation-form pressure-gradient reference profile.
function default_nested_dynamics(grid; base_pressure, reference_potential_temperature, damping_rate, damping_depth)
    time_discretization = SplitExplicitTimeDiscretization(sponge = UpperSponge(; damping_rate, depth = damping_depth),
                                                          damping = NoDivergenceDamping())
    kw = (;)
    isnothing(base_pressure)                   || (kw = merge(kw, (; base_pressure)))
    isnothing(reference_potential_temperature) || (kw = merge(kw, (; reference_potential_temperature)))
    return CompressibleDynamics(time_discretization; kw...)
end

# Default child scalar advection: WENO(5) for the energy density `ρθ` and positivity-bounded WENO(5)
# for the moisture + precipitation densities. `atmosphere_model`'s scalar default is `Centered(order=2)`,
# which is oscillatory on the sharp moist fronts of a downscaled convective case — it overshoots the
# moisture density into the density/saturation coupling and blows the cold start up within the first
# minute. Bounding mirrors Breeze's own moist-convection examples (`ρqᵉ = WENO(order=5, bounds=(0, 1))`).
# The energy density is unbounded (`ρθ` is not confined to `[0, 1]`). Names are derived from the
# microphysics so the default tracks whichever moisture/precipitation prognostics it carries.
function default_nested_scalar_advection(microphysics)
    bounded = WENO(order = 5, bounds = (0, 1))
    moist_names = (moisture_prognostic_name(microphysics), prognostic_field_names(microphysics)...)
    return merge((ρθ = WENO(order = 5),), NamedTuple{moist_names}(map(_ -> bounded, moist_names)))
end

# Blend-zone width in cells from a physical length: a fixed cell count steepens the parent→child
# terrain transition ~1/Δx as resolution increases, and the steeper σ-surface tilt destabilizes
# high-resolution runs at the boundary corner.
default_terrain_blend_width(grid, blend_length) =
    max(1, round(Int, blend_length / minimum_xspacing(grid, Center(), Center(), Center())))

# Child terrain: an elevation `Field` passes through; anything else is regridded from a
# topography dataset. Smoothing damps the grid-scale orographic roughness that excites standing
# near-surface noise on the terrain-following coordinate; blending toward the parent's surface
# elevation (after smoothing) keeps the open-boundary terrain consistent with the parent state.
function materialize_nested_terrain!(child_grid, terrain, parent_surface, blend_width, smoothing_passes)
    elevation = terrain isa AbstractField ? terrain : regrid_topography(child_grid; dataset = terrain)
    smoothing_passes > 0 && smooth_topography!(elevation; passes = smoothing_passes)
    if !isnothing(parent_surface) && blend_width > 0
        parent_elevation = Field{Center, Center, Nothing}(child_grid)
        interpolate!(parent_elevation, parent_surface)
        blend_parent_terrain!(elevation, parent_elevation; width = blend_width)
    end
    return materialize_terrain!(child_grid, elevation)
end

function default_parent_condensates(parent_atmosphere::PrescribedAtmosphere)
    return (qᶜˡ = parent_atmosphere.microphysical_variables.qᶜˡ,
            qʳ  = parent_atmosphere.microphysical_variables.qʳ,
            qᶜⁱ = parent_atmosphere.microphysical_variables.qᶜⁱ,
            qˢ  = parent_atmosphere.microphysical_variables.qˢ)
end

default_parent_condensates(parent_atmosphere::BoundaryPrescribedComponent) =
    map(default_parent_condensates, boundary_strips(parent_atmosphere))

"""
$(TYPEDSIGNATURES)

Build a Breeze child atmosphere over `child_grid` nested in `parent_atmosphere`, wrapped in a
`NestedModel`. The child's prognostics (`ρᵈ, ρu, ρv, ρθ, <moisture>`) are precomputed from the parent's
raw state on the parent grid as `FieldTimeSeries` (see `child_prognostic_field_time_series`); the child's
lateral boundary conditions — and, when `relaxation_rate` (s⁻¹) is given, its interior Davies relaxation
over `relaxation_mask` (default: a cosine ramp over the outermost `relaxation_width` cells) — interpolate
those precomputed prognostics (via `parent_boundary_conditions` / `parent_forcings`).

When `parent_atmosphere` is a `BoundaryPrescribedComponent` of `PrescribedAtmosphere`s, each side's boundary conditions
interpolate that side's strip, and the Davies relaxation acts only on the cells of each strip's
relaxation zone, through a tendency callback the `NestedModel` adds to the child's at every time step.
`parent_condensates` is then keyed by side.

Liquid/ice inputs to the combine default to the parent's hydrometeors — total liquid `qᶜˡ + qʳ`
(cloud liquid + rain) and total ice `qᶜⁱ + qˢ` (cloud ice + snow) — but may be supplied from any
source via `parent_condensates`, a `NamedTuple` with `qᶜˡ`/`qʳ`/`qᶜⁱ`/`qˢ` entries. Any missing or
`nothing` entry — or the whole `parent_condensates` — is treated as absent (⇒ omitted; with all four
absent, `qᵗ = qᵛ`).

Provides sensible, overridable physics defaults: `microphysics` (1-moment mixed-phase when
`CloudMicrophysics` is loaded), `momentum_advection = WENO(order=9)`, `coriolis = SphericalCoriolis()`,
and a compressible split-explicit `dynamics` with an `UpperSponge` over the top `damping_depth` m at
`damping_rate`; a matching ρw Rayleigh lid sponge (`Relaxation` toward zero) is added to `forcing`. Pass
`base_pressure`/`reference_potential_temperature` to anchor the default dynamics. Any
`boundary_conditions`/`forcing` the caller passes are merged with the parent-derived ones (caller wins).

When `bottom_drag_coefficient` is given — a constant drag coefficient or a `Breeze.PolynomialCoefficient`
— Breeze `BulkDrag` flux boundary conditions are applied at the bottom of the momentum densities
`ρu`/`ρv`, computing the surface stress from the face-located near-surface velocity.
`drag_surface_temperature` (a `Field`, function, or number) enters the drag's surface density and, for
polynomial coefficients, its stability correction; compressible dynamics has no default surface
temperature, so it must be supplied (the parent-dataset method defaults it to the dataset's skin
temperature).

When `terrain` is given — an elevation `Field`, or a topography dataset (e.g. `ETOPO2022()`) that is
regridded onto the child grid — the child grid's terrain-following coordinate is materialized in place
before the model is built. The elevation is first smoothed with `terrain_smoothing_passes` applications
of a binomial filter ([`smooth_topography!`](@ref); `0` disables): point-sampled regridding otherwise
leaves grid-scale orographic roughness that excites standing grid-scale noise in the near-surface flow
on the terrain-following coordinate. If the parent knows its surface elevation
([`surface_elevation`](@ref)), the child elevation is then blended toward the parent's over an outer
frame of physical width `terrain_blend_length` (meters; converted to a resolution-invariant cell count,
or overridden directly with `terrain_blend_width`), so the terrain at the open boundaries matches the
orography the parent state was produced with — and the blend slope stays fixed across resolutions
rather than steepening. `parent_surface_elevation` replaces the parent's own surface elevation in that blend.
"""
function NumericalEarth.NestedModels.nested_atmosphere_model(parent_atmosphere::Union{PrescribedAtmosphere, BoundaryPrescribedComponent},
                                                              child_grid;
    relaxation_rate = nothing,
    relaxation_width = default_relaxation_width,
    relaxation_mask = davies_relaxation_mask(child_grid, relaxation_width),
    sides = (:west, :east, :south, :north),
    thermodynamic_constants = ThermodynamicConstants(eltype(child_grid)),
    base_pressure = nothing,
    reference_potential_temperature = nothing,
    terrain = nothing,
    terrain_blend_length = 60_000,   # meters; physical blend width → resolution-invariant slope
    terrain_blend_width = nothing,    # explicit cell-count override; derived from length if `nothing`
    terrain_smoothing_passes = 2,     # binomial-filter passes on the child elevation; 0 disables
    parent_surface_elevation = nothing, # elevation the terrain blends toward; `nothing` ⇒ the parent's
    bottom_drag_coefficient = nothing,  # constant Cᴰ or a Breeze `PolynomialCoefficient`; `nothing` disables
    drag_surface_temperature = nothing, # surface temperature entering the drag's surface density
    parent_condensates = default_parent_condensates(parent_atmosphere),
    microphysics = default_nested_microphysics(),
    momentum_advection = WENO(order = 9),
    scalar_advection = default_nested_scalar_advection(microphysics),
    coriolis = SphericalCoriolis(),
    damping_rate = 1/5,
    damping_depth = default_lid_depth(child_grid),
    dynamics = default_nested_dynamics(child_grid; base_pressure, reference_potential_temperature, damping_rate, damping_depth),
    boundary_conditions = NamedTuple(),
    forcing = NamedTuple(),
    kw...)

    if !isnothing(terrain)
        blend_width = something(terrain_blend_width, default_terrain_blend_width(child_grid, terrain_blend_length))
        parent_surface = isnothing(parent_surface_elevation) ? surface_elevation(parent_atmosphere) : parent_surface_elevation
        materialize_nested_terrain!(child_grid, terrain, parent_surface, blend_width, terrain_smoothing_passes)
    end

    moisture_name = moisture_prognostic_name(microphysics)
    pˢᵗ = dynamics.standard_pressure

    # Precompute the child prognostics on the parent grid (combine-then-interpolate); the exchanger owns
    # its own 3-level moving window and refreshes it from the parent each step via `exchange_state!`.
    exchanger  = state_exchanger(parent_atmosphere, pˢᵗ, thermodynamic_constants; condensates = parent_condensates, moisture_name)
    prognostic = parent_prognostic(exchanger)

    ρqᵛᵉ = prognostic.ρqᵛᵉ
    moist_variables = NamedTuple{tuple(moisture_name)}(tuple(ρqᵛᵉ))

    # Lateral BCs: interpolate the precomputed prognostics at the boundary face. Momentum is prescribed on
    # every side, but the BC *type* is per-side: `NormalFlowBoundaryCondition` on the wall-normal side
    # (where the velocity's face coincides with the boundary), `ValueBoundaryCondition` on the tangential
    # side (prescribing the parent's tangential velocity in the halo — `NormalFlowBC` there leaves it
    # under-constrained and injects spurious near-boundary convergence). `ρᵈ`/energy/moisture are Center
    # scalars (`ValueBoundaryCondition` on all sides, since `NormalFlowBC` overwrites the first interior cell
    # asymmetrically for Center fields).
    energy_key = energy_bc_key()
    dry_bc_variables = merge((ρᵈ = prognostic.ρᵈ, ρu = prognostic.ρu, ρv = prognostic.ρv),
                             NamedTuple{(energy_key,)}((prognostic.ρθ,)))
    bc_variables = merge(dry_bc_variables, moist_variables)

    density_and_energy_types = merge((ρᵈ = ValueBoundaryCondition,),
                                     NamedTuple{(energy_key,)}((ValueBoundaryCondition,)))
    momentum_types = (ρu = (west = NormalFlowBoundaryCondition, east = NormalFlowBoundaryCondition, south = ValueBoundaryCondition, north = ValueBoundaryCondition),
                      ρv = (west = ValueBoundaryCondition, east = ValueBoundaryCondition, south = NormalFlowBoundaryCondition, north = NormalFlowBoundaryCondition))
    moist_types = NamedTuple{tuple(moisture_name)}(tuple(ValueBoundaryCondition))
    bc_types = merge(density_and_energy_types, momentum_types, moist_types)

    nested_bcs = parent_boundary_conditions(child_grid; variables = bc_variables, sides, bc_types)

    # Bulk-drag bottom stress on the momentum densities: `BulkDrag` reads the dragged velocity
    # at its own face (so a two-grid-length mode feels the drag) and infers direction from each
    # field's location; the same unmaterialized condition serves both components.
    drag_bcs = if isnothing(bottom_drag_coefficient)
        NamedTuple()
    else
        drag = BulkDrag(coefficient = bottom_drag_coefficient, surface_temperature = drag_surface_temperature)
        (ρu = FieldBoundaryConditions(bottom = drag), ρv = FieldBoundaryConditions(bottom = drag))
    end

    child_bcs = merge_boundary_conditions(nested_bcs, drag_bcs)

    # Interior Davies relaxation toward the precomputed prognostics. Oceananigans' FTS `Relaxation`
    # calls `mask(x, y, z)`, so wrap a scalar mask in a callable. Momentum and energy relax toward the
    # parent's SPECIFIC state, which `SpecificForcing` weights by the child's own `ρᵈ` at kernel time;
    # relaxing toward the parent's `ρθ`/`ρu`/`ρv` instead equilibrates at `θ = θₚ ρᵈₚ / ρᵈ`, an absolute
    # error `θ Δρᵈ / ρᵈ` (≈3 K per 1% density mismatch: the lateral-boundary cold rim). `ρᵈ` itself is
    # absent because Breeze's compressible continuity kernels overwrite `Gⁿ.ρᵈ` with `-∇·m` and never
    # read `forcing.ρᵈ`, so a mass-nudging entry is silently discarded; were that to change, the
    # specific form would gain a `θ Δρᵈ / ρᵈ` cross-term and the density-weighted form would be unbiased.
    # The wrap is explicit and keyed by the density-weighted prognostic rather than left to Breeze's
    # specific-key dispatch, so a caller's own `θ`/`u`/`v` forcing combines with the relaxation instead
    # of replacing it in the `merge` below.
    relax_mask = relaxation_mask isa Number ? Returns(relaxation_mask) : relaxation_mask
    boundary_parent = parent_atmosphere isa BoundaryPrescribedComponent
    davies = if isnothing(relaxation_rate) || boundary_parent
        NamedTuple()
    else
        specific_targets = (ρθ = prognostic.θ, ρu = prognostic.u, ρv = prognostic.v)
        specific = parent_forcings(; variables = specific_targets, rate = relaxation_rate, mask = relax_mask)
        moist = parent_forcings(; variables = moist_variables, rate = relaxation_rate, mask = relax_mask)
        merge(map(SpecificForcing, specific), moist)
    end

    # ρw Rayleigh sponge over BOTH the top `damping_depth` meters AND the lateral relaxation zone. The
    # horizontal Davies nudging drives a persistent vertical-velocity wave up the (inflow) lateral walls
    # (nudging the horizontal mass/momentum harder makes it worse); `ρw` is otherwise undamped there, so
    # the wave amplifies up the wall column until a top-only sponge catches it too late — at the wall/lid
    # corner, where ρ collapses and the run NaNs past ~2 h. Damping `ρw` toward zero across the wall zone,
    # where the parent's large-scale `w` is negligible, absorbs the creep at its source. One `Relaxation`
    # on `ρw` with the pointwise max of the lid and lateral (Davies) masks.
    lid_mask = lid_sponge_mask(child_grid, damping_depth)
    sponge_mask = (λ, φ, z) -> max(lid_mask(λ, φ, z), relax_mask(λ, φ, z))
    lid_sponge = (; ρw = Relaxation(rate = damping_rate, mask = sponge_mask))

    # initialize = false: the resting-state construction default would survive into
    # `initialize_nested_child!` and destabilize the adiabatic balance twin — the child's full
    # state (and reference) is derived from the parent instead.
    child = NumericalEarth.Atmospheres.atmosphere_model(child_grid;
        thermodynamic_constants, microphysics, momentum_advection, scalar_advection, coriolis, dynamics,
        boundary_conditions = merge_boundary_conditions(child_bcs, NamedTuple(boundary_conditions)),
        forcing = merge(lid_sponge, davies, NamedTuple(forcing)),
        initialize = false,
        kw...)

    child_callbacks = isnothing(relaxation_rate) || !boundary_parent ? () :
                      (boundary_relaxation(child, prognostic, relaxation_rate, relax_mask, relaxation_width),)

    return NestedModel(parent_atmosphere, child, exchanger, child_callbacks)
end

# Domain-mean dataset mean-sea-level pressure at `date`, regridded onto the child grid.
function mean_sea_level_pressure(dataset, child_grid, date, dir)
    single_level_dataset = matching_single_level_dataset(dataset)
    p₀ = Field{Center, Center, Nothing}(child_grid)
    set!(p₀, Metadatum(:mean_sea_level_pressure; dataset = single_level_dataset, date,
                       region = BoundingBox(child_grid), dir))
    # Reduce across ranks so every rank anchors the same hydrostatic reference
    # (`all_reduce` is the identity on serial architectures).
    # TODO: this belongs in Oceananigans — `sum`/`mean` on a distributed `Field` should
    # perform the global reduction themselves (Oceananigans.jl's reductions are rank-local).
    arch = architecture(child_grid)
    return all_reduce(+, sum(interior(p₀)), arch) / all_reduce(+, length(interior(p₀)), arch)
end

# Dataset skin temperature at `date`, regridded onto the child grid — the surface temperature
# entering the bulk drag's surface density (and its stability correction, for polynomial
# coefficients). Static in time: one snapshot, not the dataset's diurnal cycle.
function dataset_skin_temperature(dataset, child_grid, date, dir)
    single_level_dataset = matching_single_level_dataset(dataset)
    T₀ = Field{Center, Center, Nothing}(child_grid)
    set!(T₀, Metadatum(:skin_temperature; dataset = single_level_dataset, date,
                       region = BoundingBox(child_grid), dir))
    return T₀
end

"""
    nested_atmosphere_model(child_grid, parent_dataset; dates, kw...)

Build the parent `BoundaryPrescribedComponent` of `PrescribedAtmosphere`s, nest a Breeze child in it, and initialize the child from
`parent_dataset` at `first(dates)` — the returned model is ready to step. The parent's strips hold
`parent_dataset` at `dates` on its native grid, along each side of `child_grid` and `relaxation_width`
cells inward, padded by `parent_padding` (default `parent_dataset`'s `default_horizontal_padding`,
margin for the interpolation stencils). A full-domain snapshot of `parent_dataset` at the first two
`dates`, with the same padding, initializes the child and supplies the surface elevation the `terrain` blends
toward. Unless given, the default dynamics' `base_pressure` anchor is the domain-mean dataset
mean-sea-level pressure over the child at `first(dates)`. When `bottom_drag_coefficient` is given,
`drag_surface_temperature` defaults to the dataset's skin temperature at `first(dates)` regridded onto
the child grid (a static snapshot, not the dataset's diurnal cycle). `balancer` controls the
post-initialization adiabatic (DFI) balance: `true` (default) runs it, `false` skips it, and an
`AdiabaticBalancer(Δt=…)` runs a custom (e.g. gentler) excursion. Remaining keyword arguments flow to
`nested_atmosphere_model(parent, child_grid; kw...)`.
"""
function NumericalEarth.NestedModels.nested_atmosphere_model(child_grid, parent_dataset; dates,
    dir = default_download_directory(parent_dataset),
    parent_padding = default_horizontal_padding(parent_dataset),
    parent_time_indices_in_memory = nothing,   # nothing ⇒ every date resident; ≥3 streams a moving window
    relaxation_width = default_relaxation_width,
    base_pressure = nothing,
    bottom_drag_coefficient = nothing,
    drag_surface_temperature = nothing,
    balancer = true,
    kw...)

    parent_atmosphere = BoundaryPrescribedComponent(PrescribedAtmosphere, child_grid, dates, parent_dataset;
                                                    width = relaxation_width, padding = parent_padding, dir,
                                                    time_indices_in_memory = parent_time_indices_in_memory)

    snapshot_dates = expand_dates(parent_dataset, :temperature, dates)[1:2]
    snapshot = PrescribedAtmosphere(BoundingBox(child_grid; padding = parent_padding), snapshot_dates, parent_dataset;
                                    architecture = architecture(child_grid), dir)

    if isnothing(base_pressure)
        base_pressure = mean_sea_level_pressure(parent_dataset, child_grid, first(dates), dir)
    end

    if !isnothing(bottom_drag_coefficient) && isnothing(drag_surface_temperature)
        drag_surface_temperature = dataset_skin_temperature(parent_dataset, child_grid, first(dates), dir)
    end

    nested_model = NumericalEarth.NestedModels.nested_atmosphere_model(parent_atmosphere, child_grid;
                                                                       relaxation_width, base_pressure,
                                                                       bottom_drag_coefficient, drag_surface_temperature,
                                                                       parent_surface_elevation = surface_elevation(snapshot),
                                                                       kw...)

    initial_state = state_exchanger(snapshot, first(nested_model.exchanger)).prognostic
    initialize_nested_child!(nested_model, parent_dataset, first(dates), dir; balancer, prognostic = initial_state)
    return nested_model
end

NumericalEarth.Atmospheres.bulk_drag(model::NestedModel; kw...) =
    NumericalEarth.Atmospheres.bulk_drag(model.child; kw...)

function interpolate_to_child(fts, child_grid, t₀, loc = (Center, Center, Center))
    field = Field{loc...}(child_grid)
    interpolate!(field, fts[Time(t₀)])
    return field
end

# Initialize the nested child from the parent-derived `prognostic` (by default the exchanger's, the SAME
# state that drives the lateral boundaries), interpolated to the child interior — so the interior IC and the prescribed
# boundary agree at the walls (no standing pressure/density jump). Recompute the Exner reference from the
# domain-mean state, graft ρw ← ρw − ρw̃ so the flow follows the terrain, and spin ρw into nonhydrostatic
# balance. `set!(…; balancer = true)` runs Breeze's adiabatic (FV3 `na_init`) balance on a stripped,
# memory-sharing twin (no microphysics/sponge/forcing) at an automatically-derived acoustic-CFL step.
function initialize_nested_child!(nested_model, dataset, date, dir; balancer = true,
                                  prognostic = nested_model.exchanger.prognostic)
    child = nested_model.child
    child_grid = child.grid
    t₀ = first(prognostic.ρᵈ.times)

    ρᵈ   = interpolate_to_child(prognostic.ρᵈ, child_grid, t₀)
    ρθ   = interpolate_to_child(prognostic.ρθ, child_grid, t₀)
    ρqᵛᵉ = interpolate_to_child(prognostic.ρqᵛᵉ, child_grid, t₀)
    ρu   = interpolate_to_child(prognostic.ρu, child_grid, t₀, (Face, Center, Center))
    ρv   = interpolate_to_child(prognostic.ρv, child_grid, t₀, (Center, Face, Center))

    ρ   = Field(ρᵈ + ρqᵛᵉ)
    qᵛᵉ = Field(ρqᵛᵉ / ρ)
    θˡⁱ = Field(ρθ / ρᵈ)

    moisture = NamedTuple{(moisture_specific_name(child.microphysics),)}((qᵛᵉ,))
    set!(nested_model; ρ, ρu, ρv, θˡⁱ, moisture..., compute_reference_state = true)

    # Consistent-w: graft ρw ← ρw − ρw̃ so the contravariant w̃ ≈ 0 (the initial flow follows the ground).
    update_state!(nested_model)

    if !isnothing(child.dynamics.contravariant_vertical_momentum)
        interior(child.momentum.ρw) .-= interior(child.dynamics.contravariant_vertical_momentum)
        update_state!(nested_model)
    end

    # Adiabatic (DFI) balance at Breeze's auto acoustic-CFL step. `balancer=false` skips it (to isolate
    # whether the interpolated IC steps stably on its own); pass an `AdiabaticBalancer(Δt=…)` for a
    # gentler excursion when the default 0.85·Δz/c DFI drives a pathological IC cell's pressure negative.
    set!(nested_model; balancer)

    return nested_model
end
