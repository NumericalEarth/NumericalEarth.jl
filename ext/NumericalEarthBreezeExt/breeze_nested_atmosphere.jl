#####
##### Nested-atmosphere model: a Breeze child driven by a parent `PrescribedAtmosphere`.
#####

# The child's prognostic variables (dry density `ρᵈ`, momentum densities `ρu`/`ρv`, potential-temperature
# density `ρθ`, vapor density `ρqᵛ`) are precomputed from the parent's raw state ON THE PARENT GRID and
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
using NumericalEarth.DataWrangling: default_download_directory, default_horizontal_padding, matching_single_level_dataset
using NumericalEarth.NestedModels: NestedModel, parent_boundary_conditions, parent_forcings, blend_parent_terrain!

using Oceananigans:
    Oceananigans,
    WENO,
    ValueBoundaryCondition,
    NormalFlowBoundaryCondition,
    Field,
    CenterField,
    Center, Face,
    set!

using Oceananigans.Architectures: architecture, on_architecture
using Oceananigans.DistributedComputations: all_reduce
using Oceananigans.Coriolis: SphericalCoriolis
using Oceananigans.Fields: AbstractField, interior, interpolate!, compute!
using Oceananigans.Forcings: Relaxation, materialize_forcing
using Oceananigans.Grids: znode, minimum_xspacing, x_domain, y_domain
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Units: Time

using GPUArraysCore: @allowscalar
using Adapt: Adapt, adapt

using Breeze:
    BulkDrag,
    CompressibleDynamics,
    SplitExplicitTimeDiscretization,
    UpperSponge,
    NoDivergenceDamping,
    MixedPhaseEquilibrium,
    SpecificForcing,
    materialize_terrain!,
    moisture_prognostic_name

using Breeze.AtmosphereModels: AtmosphereModels, prognostic_field_names

# Default child microphysics: 1-moment bulk mixed-phase (rain + snow) precipitation with
# saturation-adjustment cloud formation when Breeze's `CloudMicrophysics` extension is loaded,
# else the Breeze-native warm-phase saturation-adjustment scheme. Resolved at call time, so a
# caller that `using CloudMicrophysics` gets `OneMomentCloudMicrophysics` automatically.
function default_nested_microphysics()
    ext = Base.get_extension(Breeze, :BreezeCloudMicrophysicsExt)
    isnothing(ext) && return SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
    return ext.OneMomentCloudMicrophysics(cloud_formation = SaturationAdjustment(equilibrium = MixedPhaseEquilibrium()))
end

# Davies relaxation for a density-weighted scalar whose physical target is a specific quantity.
# The skeleton carries an ordinary `Relaxation`; materialization replaces its relaxed field by
# ρϕ / ρ and the kernel multiplies the resulting specific tendency by the child's current total ρ.
struct ParentSpecificRelaxation{R, D}
    relaxation :: R
    density :: D
end

ParentSpecificRelaxation(; rate, mask, target) =
    ParentSpecificRelaxation(Relaxation(; rate, mask, target), nothing)

Adapt.adapt_structure(to, forcing::ParentSpecificRelaxation) =
    ParentSpecificRelaxation(adapt(to, forcing.relaxation), adapt(to, forcing.density))

@inline function (forcing::ParentSpecificRelaxation)(i, j, k, grid, clock, model_fields)
    @inbounds ρ = forcing.density[i, j, k]
    return ρ * forcing.relaxation(i, j, k, grid, clock, model_fields)
end

function AtmosphereModels.materialize_atmosphere_model_forcing(forcing::ParentSpecificRelaxation,
                                                                field, name, model_field_names,
                                                                context::NamedTuple)
    density = context.total_density
    specific_field = Field(field / density)
    relaxation = materialize_forcing(forcing.relaxation, specific_field, name, model_field_names)

    # A non-`nothing` marker normally tells Oceananigans to refresh a transformed relaxed field.
    # Our compute hook below performs that refresh directly, including after GPU adaptation drops
    # this host-only marker.
    relaxation = Relaxation(relaxation.rate, relaxation.relaxed, relaxation.mask,
                            relaxation.target, relaxation.location, Val(:specific_composition))
    return ParentSpecificRelaxation(relaxation, density)
end

function AtmosphereModels.compute_forcing!(forcing::ParentSpecificRelaxation)
    compute!(forcing.relaxation.relaxed)
    return nothing
end

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
    Nx, Ny, _ = size(grid)
    w = width * max((λ₂ˡ - λ₁ˡ) / Nx, (φ₂ˡ - φ₁ˡ) / Ny)
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

# Child scalar advection by field class: water masses (`ρq*`) bounds-preserving on `(0, 1)`, number and
# volume moments (`ρn*`, `ρb*`) positivity-preserving, the energy density `ρθ` unbounded.
function default_nested_scalar_advection(microphysics)
    names = (:ρθ, moisture_prognostic_name(microphysics), prognostic_field_names(microphysics)...)
    schemes = map(names) do name
        prefix = first(string(name), 2)
        prefix == "ρq" ? WENO(order = 5, bounds = (0, 1)) :
        prefix == "ρn" || prefix == "ρb" ? WENO(order = 5, bounds = (0.0, Inf)) :
        WENO(order = 5)
    end
    return NamedTuple{names}(schemes)
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
function materialize_nested_terrain!(child_grid, terrain, parent_atmosphere, blend_width, smoothing_passes)
    elevation = terrain isa AbstractField ? terrain : regrid_topography(child_grid; dataset = terrain)
    smoothing_passes > 0 && smooth_topography!(elevation; passes = smoothing_passes)
    parent_surface = surface_elevation(parent_atmosphere)
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

"""
$(TYPEDSIGNATURES)

Build a Breeze child atmosphere over `child_grid` nested in `parent_atmosphere`, wrapped in a
`NestedModel`. The child's prognostics (`ρᵈ, ρu, ρv, ρθ, <moisture>`) are precomputed from the parent's
raw state on the parent grid as `FieldTimeSeries` (see `child_prognostic_field_time_series`); the child's
lateral boundary conditions — and, when `relaxation_rate` (s⁻¹) is given, its interior Davies relaxation
over `relaxation_mask` (default: a cosine ramp over the outermost `relaxation_width` cells) — interpolate
those precomputed prognostics (via `parent_boundary_conditions` / `parent_forcings`).

Liquid/ice inputs to the combine default to the parent's hydrometeors — total liquid `qᶜˡ + qʳ`
(cloud liquid + rain) and total ice `qᶜⁱ + qˢ` (cloud ice + snow) — but may be supplied from any
source via `parent_condensates`, a `NamedTuple` with `qᶜˡ`/`qʳ`/`qᶜⁱ`/`qˢ` entries. Any missing or
`nothing` entry — or the whole `parent_condensates` — is treated as absent (⇒ omitted; with all four
absent, `qᵗ = qᵛ`).

For P3, the same parent condensates are also mapped into every P3 prognostic carried by the child and
used consistently for its initial condition, lateral boundary values, and Davies targets. Cloud liquid
maps directly; ERA5 cloud ice + snow become initially unrimed, uncoated P3 ice. ERA5 supplies no number
moments, so rain uses a Marshall--Palmer exponential distribution with
`p3_rain_intercept_parameter = 8e6` m⁻⁴, while ice uses a compact-particle mean-mass diameter
`p3_ice_mean_mass_diameter = 100e-6` m; both moments are clipped through the P3 scheme's configured
slope limits. With prognostic aerosol, cloudy cells start at `p3.cloud.number_concentration` and the
remaining aerosol follows the LASSO `diagCCN` partition `ρnᵃ = max(0, ρnᵃ₀ - ρnᶜˡ - ρnʳ)`.
P3 initialization skips the current adiabatic DFI balancer: its process-free twin omits the P3
condensate load and is not thermodynamically equivalent to the mapped parent state.

Provides sensible, overridable physics defaults: `microphysics` (1-moment mixed-phase when
`CloudMicrophysics` is loaded), `momentum_advection = WENO(order=9)`, `coriolis = SphericalCoriolis()`,
and a compressible split-explicit `dynamics` with an `UpperSponge` over the top `damping_depth` m at
`damping_rate`; a matching ρw Rayleigh lid sponge (`Relaxation` toward zero) is added to `forcing`. Pass
`surface_pressure`/`reference_potential_temperature` to anchor the default dynamics. Any
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
rather than steepening.
"""
function NumericalEarth.NestedModels.nested_atmosphere_model(parent_atmosphere::PrescribedAtmosphere, child_grid;
    relaxation_rate = nothing,
    relaxation_width = 5,
    relaxation_mask = davies_relaxation_mask(child_grid, relaxation_width),
    sides = (:west, :east, :south, :north),
    thermodynamic_constants = ThermodynamicConstants(eltype(child_grid)),
    surface_pressure = nothing,
    reference_potential_temperature = nothing,
    terrain = nothing,
    terrain_blend_length = 60_000,   # meters; physical blend width → resolution-invariant slope
    terrain_blend_width = nothing,    # explicit cell-count override; derived from length if `nothing`
    terrain_smoothing_passes = 2,     # binomial-filter passes on the child elevation; 0 disables
    bottom_drag_coefficient = nothing,  # constant Cᴰ or a Breeze `PolynomialCoefficient`; `nothing` disables
    drag_surface_temperature = nothing, # surface temperature entering the drag's surface density
    parent_condensates = default_parent_condensates(parent_atmosphere),
    microphysics = default_nested_microphysics(),
    p3_rain_intercept_parameter = 8e6,  # Marshall--Palmer N₀ [m⁻⁴]
    p3_ice_mean_mass_diameter = 100e-6, # ERA5 cloud ice + snow mapping [m]
    momentum_advection = WENO(order = 9),
    scalar_advection = default_nested_scalar_advection(microphysics),
    coriolis = SphericalCoriolis(),
    damping_rate = 1/5,
    damping_depth = default_lid_depth(child_grid),
    dynamics = default_nested_dynamics(child_grid; surface_pressure, reference_potential_temperature, damping_rate, damping_depth),
    boundary_conditions = NamedTuple(),
    forcing = NamedTuple(),
    kw...)

    if !isnothing(terrain)
        blend_width = something(terrain_blend_width, default_terrain_blend_width(child_grid, terrain_blend_length))
        materialize_nested_terrain!(child_grid, terrain, parent_atmosphere, blend_width, terrain_smoothing_passes)
    end

    child_microphysics = on_architecture(architecture(child_grid), microphysics)
    moisture_name = moisture_prognostic_name(child_microphysics)
    pˢᵗ = dynamics.standard_pressure

    # Precompute the child prognostics on the parent grid (combine-then-interpolate); the exchanger owns
    # its own 3-level moving window and refreshes it from the parent each step via `exchange_state!`.
    condensates = isnothing(parent_condensates) ? (qᶜˡ = nothing, qʳ = nothing, qᶜⁱ = nothing, qˢ = nothing) : parent_condensates
    exchanger  = state_exchanger(parent_atmosphere, pˢᵗ, thermodynamic_constants;
                                  condensates, microphysics = child_microphysics,
                                  rain_intercept_parameter = p3_rain_intercept_parameter,
                                  ice_mean_mass_diameter = p3_ice_mean_mass_diameter)
    prognostic = exchanger.prognostic

    ρqᵛ = prognostic.ρqᵛ
    moist_variables = NamedTuple{tuple(moisture_name)}(tuple(ρqᵛ))
    microphysical_names = parent_microphysics_names(exchanger.microphysics_mapping)
    specific_microphysical_names = parent_microphysics_specific_names(exchanger.microphysics_mapping)
    microphysical_variables =
        NamedTuple{microphysical_names}(map(name -> prognostic[name], microphysical_names))
    moist_specific_variables = NamedTuple{tuple(moisture_name)}((prognostic.qᵛ,))
    microphysical_specific_variables = NamedTuple{microphysical_names}(
        map(name -> prognostic[name], specific_microphysical_names))

    # Lateral BCs: interpolate the precomputed prognostics at the boundary face. Momentum is prescribed on
    # every side, but the BC *type* is per-side: `NormalFlowBoundaryCondition` on the wall-normal side
    # (where the velocity's face coincides with the boundary), `ValueBoundaryCondition` on the tangential
    # side (prescribing the parent's tangential velocity in the halo — `NormalFlowBC` there leaves it
    # under-constrained and injects spurious near-boundary convergence). `ρᵈ`/energy/moisture are Center
    # scalars (`ValueBoundaryCondition` on all sides, since `NormalFlowBC` overwrites the first interior cell
    # asymmetrically for Center fields). The energy BC uses Breeze's energy-BC interface key (`ρs` on
    # Breeze ≥0.10, `ρe` before): it merges with the coupling's bottom energy-flux BC on the same field,
    # and for a potential-temperature formulation Breeze routes the (Value) `ρθ` boundary values through
    # unchanged. `ρθ` and the energy key must not both carry BCs.
    energy_key = energy_bc_key()
    dry_bc_variables = merge((ρᵈ = prognostic.ρᵈ, ρu = prognostic.ρu, ρv = prognostic.ρv),
                             NamedTuple{(energy_key,)}((prognostic.ρθ,)))
    bc_variables = merge(dry_bc_variables, moist_variables, microphysical_variables)

    density_and_energy_types = merge((ρᵈ = ValueBoundaryCondition,),
                                     NamedTuple{(energy_key,)}((ValueBoundaryCondition,)))
    momentum_types = (ρu = (west = NormalFlowBoundaryCondition, east = NormalFlowBoundaryCondition, south = ValueBoundaryCondition, north = ValueBoundaryCondition),
                      ρv = (west = ValueBoundaryCondition, east = ValueBoundaryCondition, south = NormalFlowBoundaryCondition, north = NormalFlowBoundaryCondition))
    moist_types = NamedTuple{tuple(moisture_name)}(tuple(ValueBoundaryCondition))
    microphysical_types = NamedTuple{microphysical_names}(map(name -> ValueBoundaryCondition,
                                                               microphysical_names))
    bc_types = merge(density_and_energy_types, momentum_types, moist_types, microphysical_types)

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
    # Moisture and P3 moments are conserved per unit total-air mass. `ParentSpecificRelaxation` therefore
    # relaxes ρϕ/ρ toward the parent-specific FTS and multiplies by the child's current total ρ. Directly
    # relaxing toward the parent density-weighted target would instead equilibrate at
    # ϕ = ρ_parent ϕ_parent / ρ_child because dry density is not nudged.
    #
    # The dynamics wrap is explicit and keyed by the density-weighted prognostic rather than left to Breeze's
    # specific-key dispatch, so a caller's own `θ`/`u`/`v` forcing combines with the relaxation instead
    # of replacing it in the `merge` below.
    relax_mask = relaxation_mask isa Number ? Returns(relaxation_mask) : relaxation_mask
    davies = if isnothing(relaxation_rate)
        NamedTuple()
    else
        specific_targets = (ρθ = prognostic.θ, ρu = prognostic.u, ρv = prognostic.v)
        specific = parent_forcings(; variables = specific_targets, rate = relaxation_rate, mask = relax_mask)
        moist = map(target -> ParentSpecificRelaxation(rate = relaxation_rate,
                                                       mask = relax_mask, target = target),
                    moist_specific_variables)
        microphysical = map(target -> ParentSpecificRelaxation(rate = relaxation_rate,
                                                                mask = relax_mask, target = target),
                            microphysical_specific_variables)
        merge(map(SpecificForcing, specific), moist, microphysical)
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
        thermodynamic_constants, microphysics = child_microphysics,
        momentum_advection, scalar_advection, coriolis, dynamics,
        boundary_conditions = merge_boundary_conditions(child_bcs, NamedTuple(boundary_conditions)),
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

Build the parent `PrescribedAtmosphere`, nest a Breeze child in it, and initialize the child from
`parent_dataset` at `first(dates)` — the returned model is ready to step. The parent spans
`child_grid`'s bounding box padded by `parent_padding` (default `parent_dataset`'s
`default_horizontal_padding`, margin for the lateral-BC interpolation stencils) at `dates`, on
`parent_dataset`'s native grid. Unless given, the default dynamics' `surface_pressure` anchor is the domain-mean dataset surface
pressure over the child at `first(dates)`. When `bottom_drag_coefficient` is given,
`drag_surface_temperature` defaults to the dataset's skin temperature at `first(dates)` regridded onto
the child grid (a static snapshot, not the dataset's diurnal cycle). `balancer` controls the
post-initialization adiabatic (DFI) balance: `true` (default) runs it, `false` skips it, and an
`AdiabaticBalancer(Δt=…)` runs a custom (e.g. gentler) excursion. Remaining keyword arguments flow to
`nested_atmosphere_model(parent, child_grid; kw...)`.

`reconstruct_near_surface=true` inserts ERA5 2 m thermodynamics, 10 m winds, and hydrostatically
adjusted surface pressure in the parent column's lowest slot, shifts the pressure-level state upward,
and drops its topmost level before child initialization, lateral-boundary interpolation, and Davies
relaxation.
"""
function NumericalEarth.NestedModels.nested_atmosphere_model(child_grid, parent_dataset; dates,
    dir = default_download_directory(parent_dataset),
    parent_padding = default_horizontal_padding(parent_dataset),
    parent_time_indices_in_memory = nothing,   # nothing ⇒ every date resident; ≥3 streams a moving window
    reconstruct_near_surface = false,
    near_surface_reference_height = 10,
    surface_pressure = nothing,
    bottom_drag_coefficient = nothing,
    drag_surface_temperature = nothing,
    balancer = true,
    kw...)

    parent_region = BoundingBox(child_grid; padding = parent_padding)
    if reconstruct_near_surface
        parent_atmosphere = PrescribedAtmosphere(parent_region, dates, parent_dataset;
                                                 architecture = architecture(child_grid), dir,
                                                 time_indices_in_memory = parent_time_indices_in_memory,
                                                 reconstruct_near_surface,
                                                 near_surface_reference_height)
    else
        parent_atmosphere = PrescribedAtmosphere(parent_region, dates, parent_dataset;
                                                 architecture = architecture(child_grid), dir,
                                                 time_indices_in_memory = parent_time_indices_in_memory)
    end

    if isnothing(surface_pressure)
        surface_pressure = mean_surface_pressure(parent_dataset, child_grid, first(dates), dir)
    end

    if !isnothing(bottom_drag_coefficient) && isnothing(drag_surface_temperature)
        drag_surface_temperature = dataset_skin_temperature(parent_dataset, child_grid, first(dates), dir)
    end

    nested_model = NumericalEarth.NestedModels.nested_atmosphere_model(parent_atmosphere, child_grid; surface_pressure,
                                                                       bottom_drag_coefficient, drag_surface_temperature, kw...)
    initialize_nested_child!(nested_model, parent_dataset, first(dates), dir; balancer)
    return nested_model
end

NumericalEarth.Atmospheres.bulk_drag(model::NestedModel; kw...) =
    NumericalEarth.Atmospheres.bulk_drag(model.child; kw...)

function set_parent_initial_state!(nested_model, ::NoParentMicrophysicsMapping, prognostic,
                                   to_child, ρᵈ, ρθ, ρqᵛ, ρu, ρv)
    # The pre-P3 path retains the existing equilibrium-moisture initialization convention.
    ρ   = Field(ρᵈ + ρqᵛ)
    qᵗ  = Field(ρqᵛ / ρ)
    θˡⁱ = Field(ρθ / ρᵈ)
    u   = Field(ρu / ρᵈ)
    v   = Field(ρv / ρᵈ)
    set!(nested_model; ρ, u, v, qᵗ, θˡⁱ, compute_reference_state = true)
    return nothing
end

function balance_parent_initial_state!(nested_model, mapping, balancer)
    balancer === false && return nothing
    return set!(nested_model; balancer)
end

@static if isdefined(Breeze.Microphysics, :PredictedParticleProperties)
    function set_parent_initial_state!(nested_model, mapping::P3ParentMicrophysicsMapping, prognostic,
                                       to_child, ρᵈ, ρθ, ρqᵛ, ρu, ρv)
        names = parent_microphysics_names(mapping)
        microphysical = NamedTuple{names}(map(name -> to_child(prognostic[name]), names))
        θˡⁱ = Field(ρθ / ρᵈ)
        u   = Field(ρu / ρᵈ)
        v   = Field(ρv / ρᵈ)
        initial_state = merge((; ρᵈ, ρqᵛ), microphysical, (; u, v, θˡⁱ))
        set!(nested_model; initial_state..., compute_reference_state = true)
        return nothing
    end

    function balance_parent_initial_state!(nested_model, mapping::P3ParentMicrophysicsMapping, balancer)
        # Breeze's current process-free adiabatic twin drops every P3 condensate and moment. Running
        # it would omit condensate from total density and interpret θˡⁱ as a vapor-only state, so the
        # mapped P3 parent state is safer and more faithful without DFI until that twin is P3-aware.
        return nothing
    end
end

# Initialize the nested child from the exchanger's parent-derived prognostics (the SAME state that drives
# the lateral boundaries), interpolated to the child interior — so the interior IC and the prescribed
# boundary agree at the walls (no standing pressure/density jump). Recompute the Exner reference from the
# domain-mean state, graft ρw ← ρw − ρw̃ so the flow follows the terrain, and spin ρw into nonhydrostatic
# balance. `set!(…; balancer = true)` runs Breeze's adiabatic (FV3 `na_init`) balance on a stripped,
# memory-sharing twin (no microphysics/sponge/forcing) at an automatically-derived acoustic-CFL step.
function initialize_nested_child!(nested_model, dataset, date, dir; balancer = true)
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

    # P3 receives the mapped vapor, condensate masses, and moments directly. Other schemes retain the
    # existing equilibrium-moisture path. Dispatch here prevents a vapor-only qᵗ reconstruction from
    # discarding ERA5 condensate before the P3 child takes its first step.
    set_parent_initial_state!(nested_model, nested_model.exchanger.microphysics_mapping,
                              prognostic, to_child, ρᵈ, ρθ, ρqᵛ, ρu, ρv)

    # Consistent-w: graft ρw ← ρw − ρw̃ so the contravariant w̃ ≈ 0 (the initial flow follows the ground).
    update_state!(nested_model)

    if !isnothing(child.dynamics.contravariant_vertical_momentum)
        interior(child.momentum.ρw) .-= interior(child.dynamics.contravariant_vertical_momentum)
        update_state!(nested_model)
    end

    # Adiabatic (DFI) balance at Breeze's auto acoustic-CFL step for schemes whose process-free twin
    # retains the complete thermodynamic state. P3 dispatch skips DFI because its current twin omits
    # every P3 condensate and moment; `balancer=false` skips it explicitly for all other schemes.
    balance_parent_initial_state!(nested_model, nested_model.exchanger.microphysics_mapping, balancer)

    return nested_model
end
