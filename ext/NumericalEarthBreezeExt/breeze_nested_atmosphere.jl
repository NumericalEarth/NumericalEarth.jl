#####
##### Nested-atmosphere lateral boundary conditions: derive a Breeze child's prognostic lateral BCs
##### on the fly from a parent `PrescribedAtmosphere`'s raw state — no materialized parent prognostic
##### series. Each density-weighted prognostic (ρ, ρu, ρv, ρe, ρqᵗ) is driven by a `ParentStateBoundary`
##### that interpolates the parent's `(velocity, T, qᵛ, qᶜˡ, qᶜⁱ, p)` at the boundary face and applies
##### the matching prognostic transform (`AirDensity` / `MomentumDensity` / `EnergyDensity` /
##### `MoistureDensity`). Strictly-positive thermodynamic quantities (`p`, `T`) interpolate in log space
##### (faithful ln-p / EOS-consistent ln-ρ); velocities and humidities interpolate linearly.
#####

using NumericalEarth.Atmospheres: PrescribedAtmosphere
using NumericalEarth.EarthSystemModels.NestedSimulations: ParentBoundary, NestedModel,
                                                          nested_lateral_boundary_conditions, z_clamp
using Oceananigans: WENO
using Oceananigans.BoundaryConditions: NormalFlowBoundaryCondition, ValueBoundaryCondition
using Oceananigans.Coriolis: SphericalCoriolis
using Oceananigans.Grids: znode, node, Center, Face
using Oceananigans.Fields: interpolate
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.Units: Time
using Adapt: Adapt, adapt
using GPUArraysCore: @allowscalar
using Breeze: CompressibleDynamics, SplitExplicitTimeDiscretization, UpperSponge, NoDivergenceDamping,
              MixedPhaseEquilibrium, materialize_terrain!, moisture_prognostic_name, moisture_specific_name

# `ρu`/`ρv` are Face-staggered ⇒ `NormalFlowBoundaryCondition`; `ρ`/`ρe`/`ρqᵗ` are Center-located,
# where `NormalFlowBC` overwrites the first interior cell asymmetrically ⇒ `ValueBoundaryCondition`.
@inline _sided(BCType, condition, sides) =
    FieldBoundaryConditions(; (side => BCType(condition) for side in sides)...)

@inline _normal_flow(condition, sides, scheme) =
    FieldBoundaryConditions(; (side => NormalFlowBoundaryCondition(condition; scheme) for side in sides)...)

#####
##### Shared parent-state samplers: interpolate the raw parent fields at a node and combine into a
##### physical quantity. Each `z_clamp`s internally (so the child may extend below the parent in z),
##### interpolates `T`/`p` in log space and the rest linearly (`interpolate(func, …)`, Oceananigans
##### #5726), and reuses the pointwise combines from `breeze_prognostic_state.jl`. Used by BOTH the
##### lateral BCs (density-weighted) and the interior Davies relaxation (specific), so the two layers
##### share one definition of "sample the parent."
#####

@inline function air_density_at(X, t, grid, T, qᵛ, p, constants)
    c = Center();  loc = (c, c, c);  Xc = z_clamp(X, grid)
    return air_density(exp(interpolate(log, Xc, Time(t), T, loc, grid)),
                           interpolate(identity, Xc, Time(t), qᵛ, loc, grid),
                       exp(interpolate(log, Xc, Time(t), p, loc, grid)),
                       dry_air_gas_constant(constants), vapor_gas_constant(constants))
end

@inline function liquid_ice_θ_at(X, t, grid, T, qᶜˡ, qᶜⁱ, p, constants)
    c = Center();  loc = (c, c, c);  Xc = z_clamp(X, grid)
    Rᵈ = dry_air_gas_constant(constants);  cᵖᵈ = constants.dry_air.heat_capacity
    ℒˡ = constants.liquid.reference_latent_heat;  ℒⁱ = constants.ice.reference_latent_heat
    return liquid_ice_potential_temperature(exp(interpolate(log, Xc, Time(t), T, loc, grid)),
                                                interpolate(identity, Xc, Time(t), qᶜˡ, loc, grid),
                                                interpolate(identity, Xc, Time(t), qᶜⁱ, loc, grid),
                                            exp(interpolate(log, Xc, Time(t), p, loc, grid)),
                                            Rᵈ, cᵖᵈ, ℒˡ, ℒⁱ)
end

@inline function total_water_at(X, t, grid, qᵛ, qᶜˡ, qᶜⁱ)
    c = Center();  loc = (c, c, c);  Xc = z_clamp(X, grid)
    return total_water_specific_humidity(interpolate(identity, Xc, Time(t), qᵛ,  loc, grid),
                                         interpolate(identity, Xc, Time(t), qᶜˡ, loc, grid),
                                         interpolate(identity, Xc, Time(t), qᶜⁱ, loc, grid))
end

@inline velocity_at(X, t, grid, vel) =
    interpolate(identity, z_clamp(X, grid), Time(t), vel, (Center(), Center(), Center()), grid)

#####
##### Concrete per-variable lateral-BC samplers: `value(X, t) -> density-weighted boundary value`,
##### each holding the parent fields it needs. Wrapped in `ParentBoundary` (which supplies the
##### boundary-face node); mirror the discrete relaxation forcings below and share the `*_at` samplers.
#####

struct AirDensityBoundary{G, FT, FQV, FP, C};  grid::G; T::FT; qᵛ::FQV; p::FP; constants::C;  end
@inline (s::AirDensityBoundary)(X, t) = air_density_at(X, t, s.grid, s.T, s.qᵛ, s.p, s.constants)
Adapt.adapt_structure(to, s::AirDensityBoundary) =
    AirDensityBoundary(adapt(to, s.grid), adapt(to, s.T), adapt(to, s.qᵛ), adapt(to, s.p), s.constants)

struct MomentumDensityBoundary{G, FV, FT, FQV, FP, C};  grid::G; velocity::FV; T::FT; qᵛ::FQV; p::FP; constants::C;  end
@inline (s::MomentumDensityBoundary)(X, t) =
    air_density_at(X, t, s.grid, s.T, s.qᵛ, s.p, s.constants) * velocity_at(X, t, s.grid, s.velocity)
Adapt.adapt_structure(to, s::MomentumDensityBoundary) =
    MomentumDensityBoundary(adapt(to, s.grid), adapt(to, s.velocity), adapt(to, s.T), adapt(to, s.qᵛ), adapt(to, s.p), s.constants)

struct EnergyDensityBoundary{G, FT, QL, QI, FQV, FP, C};  grid::G; T::FT; qᶜˡ::QL; qᶜⁱ::QI; qᵛ::FQV; p::FP; constants::C;  end
@inline (s::EnergyDensityBoundary)(X, t) =
    air_density_at(X, t, s.grid, s.T, s.qᵛ, s.p, s.constants) * liquid_ice_θ_at(X, t, s.grid, s.T, s.qᶜˡ, s.qᶜⁱ, s.p, s.constants)
Adapt.adapt_structure(to, s::EnergyDensityBoundary) =
    EnergyDensityBoundary(adapt(to, s.grid), adapt(to, s.T), adapt(to, s.qᶜˡ), adapt(to, s.qᶜⁱ), adapt(to, s.qᵛ), adapt(to, s.p), s.constants)

struct MoistureDensityBoundary{G, FQV, QL, QI, FT, FP, C};  grid::G; qᵛ::FQV; qᶜˡ::QL; qᶜⁱ::QI; T::FT; p::FP; constants::C;  end
@inline (s::MoistureDensityBoundary)(X, t) =
    air_density_at(X, t, s.grid, s.T, s.qᵛ, s.p, s.constants) * total_water_at(X, t, s.grid, s.qᵛ, s.qᶜˡ, s.qᶜⁱ)
Adapt.adapt_structure(to, s::MoistureDensityBoundary) =
    MoistureDensityBoundary(adapt(to, s.grid), adapt(to, s.qᵛ), adapt(to, s.qᶜˡ), adapt(to, s.qᶜⁱ), adapt(to, s.T), adapt(to, s.p), s.constants)

"""
$(TYPEDSIGNATURES)

Build the child's lateral `FieldBoundaryConditions` for the Breeze prognostics
`(ρ, ρu, ρv, ρe, <moisture>)`, deriving each on the fly from the `parent_atmosphere`
[`PrescribedAtmosphere`](@ref)'s raw state. Each is a concrete per-variable sampler
(`AirDensityBoundary`, …) wrapped in a `ParentBoundary` that supplies the boundary-face node;
the samplers share the `*_at` parent-state samplers with the Davies-relaxation forcings.
"""
function NumericalEarth.EarthSystemModels.NestedSimulations.nested_lateral_boundary_conditions(
            parent_atmosphere::PrescribedAtmosphere, constants, moisture_name;
            sides = (:west, :east, :south, :north),
            momentum_scheme = nothing)

    u   = parent_atmosphere.velocities.u
    v   = parent_atmosphere.velocities.v
    T   = parent_atmosphere.temperature
    qᵛ  = parent_atmosphere.specific_humidity
    qᶜˡ = parent_atmosphere.microphysical_variables.qᶜˡ
    qᶜⁱ = parent_atmosphere.microphysical_variables.qᶜⁱ
    p   = parent_atmosphere.pressure
    grid = u.grid

    bc(value) = ParentBoundary(value, grid)
    ρ_bc  = bc(AirDensityBoundary(grid, T, qᵛ, p, constants))
    ρu_bc = bc(MomentumDensityBoundary(grid, u, T, qᵛ, p, constants))
    ρv_bc = bc(MomentumDensityBoundary(grid, v, T, qᵛ, p, constants))
    ρe_bc = bc(EnergyDensityBoundary(grid, T, qᶜˡ, qᶜⁱ, qᵛ, p, constants))
    ρq_bc = bc(MoistureDensityBoundary(grid, qᵛ, qᶜˡ, qᶜⁱ, T, p, constants))

    lateral = (ρ  = _sided(ValueBoundaryCondition, ρ_bc, sides),
               ρu = _normal_flow(ρu_bc, sides, momentum_scheme),
               ρv = _normal_flow(ρv_bc, sides, momentum_scheme),
               ρe = _sided(ValueBoundaryCondition, ρe_bc, sides))

    return merge(lateral, NamedTuple{tuple(moisture_name)}(tuple(_sided(ValueBoundaryCondition, ρq_bc, sides))))
end

#####
##### Discrete Davies-relaxation forcings: one concrete kernel-callable per density-weighted prognostic.
##### Each returns the *full* density-weighted tendency `ρᵈ · rate · mask · (parent − child)` at `(i,j,k)`
##### — `ρᵈ` (interpolated to the prognostic's stagger) and the child value read from `fields`, the parent
##### value interpolated on the fly from raw parent state at the `z_clamp`ed node (so the child may extend
##### below the parent in z). Keyed by density names (`ρθ, ρu, ρv, ρqᵗ`), so Breeze applies them directly
##### in the tendency kernel — no `ContinuousForcing`/`SpecificForcing`/`Relaxation` wrapping.
#####

# Each forcing below is its own discrete kernel function `(i, j, k, grid, clock, fields)` returning a
# density-weighted tendency. They are supplied directly (not wrapped in `Forcing`): the
# `materialize_atmosphere_model_forcing` method here returns them as-is, so Breeze calls them directly
# in the tendency kernel — bypassing `ContinuousForcing` (wrong signature) and `DiscreteForcing` (which
# only accepts `func <: Function`). `is_density_tendency_forcing` marks that they already include `ρᵈ`.
abstract type NestedForcing end

Breeze.AtmosphereModels.materialize_atmosphere_model_forcing(f::NestedForcing, field, name, model_field_names, context) = f
Breeze.AtmosphereModels.is_density_tendency_forcing(::NestedForcing) = true

# Spatial relaxation mask: a function evaluated at the node, or a constant.
@inline mask_value(mask, x, y, z) = mask(x, y, z)
@inline mask_value(mask::Number, x, y, z) = mask

# θˡⁱ → ρθ. Relax the child's `fields.θ` toward the parent's `liquid_ice_θ_at`, weighted by child ρᵈ.
struct PotentialTemperatureRelaxation{G, FT, QL, QI, FP, R, M, C} <: NestedForcing
    grid :: G;  T :: FT;  qᶜˡ :: QL;  qᶜⁱ :: QI;  p :: FP
    rate :: R;  mask :: M;  constants :: C
end

@inline function (f::PotentialTemperatureRelaxation)(i, j, k, grid, clock, fields)
    c = Center();  X = node(i, j, k, grid, c, c, c)
    θ★ = liquid_ice_θ_at(X, clock.time, f.grid, f.T, f.qᶜˡ, f.qᶜⁱ, f.p, f.constants)
    @inbounds return fields.ρᵈ[i, j, k] * f.rate * mask_value(f.mask, X[1], X[2], X[3]) * (θ★ - fields.θ[i, j, k])
end

Adapt.adapt_structure(to, f::PotentialTemperatureRelaxation) =
    PotentialTemperatureRelaxation(adapt(to, f.grid), adapt(to, f.T), adapt(to, f.qᶜˡ), adapt(to, f.qᶜⁱ),
                                   adapt(to, f.p), f.rate, adapt(to, f.mask), f.constants)

# qᵗ → moisture density. Relax the child's specific moisture (`fields[current_name]`) toward `total_water_at`.
struct MoistureRelaxation{G, QV, QL, QI, R, M, V} <: NestedForcing
    grid :: G;  qᵛ :: QV;  qᶜˡ :: QL;  qᶜⁱ :: QI
    rate :: R;  mask :: M;  current_name :: V
end

@inline field_name(::Val{N}) where N = N

@inline function (f::MoistureRelaxation)(i, j, k, grid, clock, fields)
    c = Center();  X = node(i, j, k, grid, c, c, c)
    qᵗ★ = total_water_at(X, clock.time, f.grid, f.qᵛ, f.qᶜˡ, f.qᶜⁱ)
    @inbounds qᵗ = getproperty(fields, field_name(f.current_name))[i, j, k]
    @inbounds return fields.ρᵈ[i, j, k] * f.rate * mask_value(f.mask, X[1], X[2], X[3]) * (qᵗ★ - qᵗ)
end

Adapt.adapt_structure(to, f::MoistureRelaxation) =
    MoistureRelaxation(adapt(to, f.grid), adapt(to, f.qᵛ), adapt(to, f.qᶜˡ), adapt(to, f.qᶜⁱ),
                       f.rate, adapt(to, f.mask), f.current_name)

# u → ρu (Face, Center, Center). Child `fields.u` toward the parent `velocity_at`, ρᵈ at the x-face.
struct URelaxation{G, FU, R, M} <: NestedForcing
    grid :: G;  u :: FU;  rate :: R;  mask :: M
end

@inline function (f::URelaxation)(i, j, k, grid, clock, fields)
    c = Center();  X = node(i, j, k, grid, Face(), c, c)
    u★ = velocity_at(X, clock.time, f.grid, f.u)
    @inbounds return ℑxᶠᵃᵃ(i, j, k, grid, fields.ρᵈ) * f.rate * mask_value(f.mask, X[1], X[2], X[3]) * (u★ - fields.u[i, j, k])
end

Adapt.adapt_structure(to, f::URelaxation) =
    URelaxation(adapt(to, f.grid), adapt(to, f.u), f.rate, adapt(to, f.mask))

# v → ρv (Center, Face, Center). Child `fields.v` toward the parent `velocity_at`, ρᵈ at the y-face.
struct VRelaxation{G, FV, R, M} <: NestedForcing
    grid :: G;  v :: FV;  rate :: R;  mask :: M
end

@inline function (f::VRelaxation)(i, j, k, grid, clock, fields)
    c = Center();  X = node(i, j, k, grid, c, Face(), c)
    v★ = velocity_at(X, clock.time, f.grid, f.v)
    @inbounds return ℑyᵃᶠᵃ(i, j, k, grid, fields.ρᵈ) * f.rate * mask_value(f.mask, X[1], X[2], X[3]) * (v★ - fields.v[i, j, k])
end

Adapt.adapt_structure(to, f::VRelaxation) =
    VRelaxation(adapt(to, f.grid), adapt(to, f.v), f.rate, adapt(to, f.mask))

# ρw Rayleigh lid sponge: damp the vertical momentum toward zero over the top of the domain. Discrete
# forcing on ρw (Center, Center, Face); `mask` is the smoothstep ramp evaluated at the node.
struct LidSponge{R, M} <: NestedForcing
    rate :: R;  mask :: M
end

@inline function (f::LidSponge)(i, j, k, grid, clock, fields)
    c = Center()
    X = node(i, j, k, grid, c, c, Face())
    @inbounds return - f.rate * mask_value(f.mask, X[1], X[2], X[3]) * fields.ρw[i, j, k]
end

Adapt.adapt_structure(to, f::LidSponge) = LidSponge(f.rate, adapt(to, f.mask))

"""
$(TYPEDSIGNATURES)

Build the interior Davies-relaxation forcings nudging the child toward the `parent_atmosphere`'s state
at `rate` over `mask`. Returns a `NamedTuple` keyed by the child's *density-weighted* prognostic names
(`ρθ, ρu, ρv`, and the moisture density `moisture_name`), each a discrete kernel-callable that returns
the full `ρᵈ`-weighted tendency directly — applied by Breeze with no `Relaxation`/`SpecificForcing`
wrapper. `θ`/`qᵗ` targets are derived on the fly from raw parent state; `u`/`v` interpolate the parent
velocities. The query is `z_clamp`ed into the parent's z-extent (the child may reach below it).
"""
function nested_relaxation_forcings(parent_atmosphere::PrescribedAtmosphere, constants, microphysics; rate, mask = 1)
    u   = parent_atmosphere.velocities.u
    v   = parent_atmosphere.velocities.v
    T   = parent_atmosphere.temperature
    qᵛ  = parent_atmosphere.specific_humidity
    qᶜˡ = parent_atmosphere.microphysical_variables.qᶜˡ
    qᶜⁱ = parent_atmosphere.microphysical_variables.qᶜⁱ
    p   = parent_atmosphere.pressure
    grid = u.grid

    # Supplied directly (no `Forcing` wrapper) — `materialize_atmosphere_model_forcing(::NestedForcing)`
    # returns each as-is, so Breeze calls them straight in the tendency kernel.
    ρθ = PotentialTemperatureRelaxation(grid, T, qᶜˡ, qᶜⁱ, p, rate, mask, constants)
    ρu = URelaxation(grid, u, rate, mask)
    ρv = VRelaxation(grid, v, rate, mask)
    ρq = MoistureRelaxation(grid, qᵛ, qᶜˡ, qᶜⁱ, rate, mask, Val(moisture_specific_name(microphysics)))

    moisture_name = moisture_prognostic_name(microphysics)
    return merge((; ρθ, ρu, ρv), NamedTuple{tuple(moisture_name)}(tuple(ρq)))
end

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
`NestedModel`. The child's lateral boundary conditions (`ρ, ρu, ρv, ρe, <moisture>`) — and, when
`relaxation_rate` (s⁻¹) is given, its interior Davies relaxation over `relaxation_mask` — are derived
on the fly from the parent's raw state (see [`nested_lateral_boundary_conditions`] /
[`nested_relaxation_forcings`]); no materialized parent prognostic series is needed.

Provides sensible, overridable physics defaults for a nested limited-area atmosphere: `microphysics`
(1-moment mixed-phase when `CloudMicrophysics` is loaded, see `default_nested_microphysics`),
`momentum_advection = WENO(order=9)`, `coriolis = SphericalCoriolis()`, and a compressible
split-explicit `dynamics` with an `UpperSponge` over the top `damping_depth` m at `damping_rate`
(see `default_nested_dynamics`); a matching ρw Rayleigh lid sponge is added to `forcing`. Pass
`surface_pressure`/`reference_potential_temperature` (e.g. from the initial state) to anchor the
default dynamics. Any `boundary_conditions`/`forcing` the caller passes are merged with the
parent-derived ones (caller wins).

When `terrain` is given (anything `materialize_terrain!` accepts — a `topography(λ, φ)` function or a
surface-elevation `Field`), the child grid's terrain-following coordinate is materialized in place
before the model is built, so the dynamics picks up the slope metrics and any subsequent
height-aware `set!` regrids onto the deformed coordinate. Requires a `TerrainFollowingVerticalDiscretization`
grid; `nothing` (default) leaves the grid untouched.
"""
function NumericalEarth.EarthSystemModels.NestedSimulations.nested_atmosphere_model(
            parent_atmosphere::PrescribedAtmosphere, child_grid;
            relaxation_rate = nothing,
            relaxation_mask = 1,
            sides = (:west, :east, :south, :north),
            thermodynamic_constants = ThermodynamicConstants(eltype(child_grid)),
            surface_pressure = nothing,
            reference_potential_temperature = nothing,
            terrain = nothing,
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

    nested_bcs = nested_lateral_boundary_conditions(parent_atmosphere, thermodynamic_constants, moisture_name; sides)
    davies = isnothing(relaxation_rate) ? (;) :
             nested_relaxation_forcings(parent_atmosphere, thermodynamic_constants, microphysics;
                                        rate = relaxation_rate, mask = relaxation_mask)

    lid_sponge = (; ρw = LidSponge(damping_rate, lid_sponge_mask(child_grid, damping_depth)))

    child = NumericalEarth.Atmospheres.atmosphere_model(child_grid;
                thermodynamic_constants, microphysics, momentum_advection, coriolis, dynamics,
                boundary_conditions = merge_boundary_conditions(nested_bcs, NamedTuple(boundary_conditions)),
                forcing = merge(lid_sponge, davies, NamedTuple(forcing)),
                kw...)

    return NestedModel(parent_atmosphere, child)
end
