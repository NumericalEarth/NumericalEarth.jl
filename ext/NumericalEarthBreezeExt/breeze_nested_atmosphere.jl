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
using NumericalEarth.EarthSystemModels.NestedSimulations: ParentStateBoundary, ParentStateTarget, NestedModel,
                                                          nested_lateral_boundary_conditions
using Oceananigans: Relaxation
using Oceananigans.BoundaryConditions: NormalFlowBoundaryCondition, ValueBoundaryCondition

# `ρu`/`ρv` are Face-staggered ⇒ `NormalFlowBoundaryCondition`; `ρ`/`ρe`/`ρqᵗ` are Center-located,
# where `NormalFlowBC` overwrites the first interior cell asymmetrically ⇒ `ValueBoundaryCondition`.
@inline _sided(BCType, condition, sides) =
    FieldBoundaryConditions(; (side => BCType(condition) for side in sides)...)

@inline _normal_flow(condition, sides, scheme) =
    FieldBoundaryConditions(; (side => NormalFlowBoundaryCondition(condition; scheme) for side in sides)...)

"""
$(TYPEDSIGNATURES)

Build the child's lateral `FieldBoundaryConditions` for the Breeze prognostics
`(ρ, ρu, ρv, ρe, <moisture>)`, deriving each on the fly from the `parent_atmosphere`
[`PrescribedAtmosphere`](@ref)'s raw state via [`ParentStateBoundary`]. `constants`
are the child's `ThermodynamicConstants`; `moisture_name` is the moisture prognostic
key (`moisture_prognostic_name(microphysics)`). Replaces the materialized
`breeze_prognostic_state` parent series + single-source `Interpolated` BCs.
"""
function NumericalEarth.EarthSystemModels.NestedSimulations.nested_lateral_boundary_conditions(
            parent_atmosphere::PrescribedAtmosphere, constants, moisture_name;
            sides = (:west, :east, :south, :north),
            momentum_scheme = nothing)

    u  = parent_atmosphere.velocities.u
    v  = parent_atmosphere.velocities.v
    T  = parent_atmosphere.temperature
    qᵛ = parent_atmosphere.specific_humidity
    qᶜˡ = parent_atmosphere.microphysical_variables.qᶜˡ
    qᶜⁱ = parent_atmosphere.microphysical_variables.qᶜⁱ
    p  = parent_atmosphere.pressure

    # log-space for the strictly-positive thermodynamic quantities; linear (identity) otherwise.
    ρ_psb  = ParentStateBoundary((; T, qᵛ, p),
                                 (T = log, qᵛ = identity, p = log), AirDensity(constants))

    momentum_interp = (velocity = identity, T = log, qᵛ = identity, p = log)
    ρu_psb = ParentStateBoundary((; velocity = u, T, qᵛ, p), momentum_interp, MomentumDensity(constants))
    ρv_psb = ParentStateBoundary((; velocity = v, T, qᵛ, p), momentum_interp, MomentumDensity(constants))

    ρe_psb = ParentStateBoundary((; T, qᶜˡ, qᶜⁱ, qᵛ, p),
                                 (T = log, qᶜˡ = identity, qᶜⁱ = identity, qᵛ = identity, p = log),
                                 EnergyDensity(constants))

    ρq_psb = ParentStateBoundary((; qᵛ, qᶜˡ, qᶜⁱ, T, p),
                                 (qᵛ = identity, qᶜˡ = identity, qᶜⁱ = identity, T = log, p = log),
                                 MoistureDensity(constants))

    lateral = (ρ  = _sided(ValueBoundaryCondition, ρ_psb, sides),
               ρu = _normal_flow(ρu_psb, sides, momentum_scheme),
               ρv = _normal_flow(ρv_psb, sides, momentum_scheme),
               ρe = _sided(ValueBoundaryCondition, ρe_psb, sides))

    return merge(lateral, NamedTuple{(moisture_name,)}((_sided(ValueBoundaryCondition, ρq_psb, sides),)))
end

"""
$(TYPEDSIGNATURES)

Build the interior Davies-relaxation forcings (`NamedTuple` of Oceananigans `Relaxation`) nudging the
child toward the `parent_atmosphere`'s state at `rate` over `mask`. Keyed under Breeze's
`SpecificForcing` names `(u, v, θ, qᵉ)`: `u`/`v` relax toward the raw parent velocities directly;
`θ`/`qᵉ` toward the *specific* `θˡⁱ`/`qᵗ` derived on the fly from raw parent state via
[`ParentStateTarget`] (Breeze applies the ρ-weight at kernel time). Replaces the materialized
`breeze_prognostic_state` relaxation targets.
"""
function nested_relaxation_forcings(parent_atmosphere::PrescribedAtmosphere, constants; rate, mask = 1)
    u  = parent_atmosphere.velocities.u
    v  = parent_atmosphere.velocities.v
    T  = parent_atmosphere.temperature
    qᵛ = parent_atmosphere.specific_humidity
    qᶜˡ = parent_atmosphere.microphysical_variables.qᶜˡ
    qᶜⁱ = parent_atmosphere.microphysical_variables.qᶜⁱ
    p  = parent_atmosphere.pressure

    θ_target  = ParentStateTarget((; T, qᶜˡ, qᶜⁱ, p),
                                  (T = log, qᶜˡ = identity, qᶜⁱ = identity, p = log),
                                  LiquidIcePotentialTemperature(constants))
    qᵗ_target = ParentStateTarget((; qᵛ, qᶜˡ, qᶜⁱ),
                                  (qᵛ = identity, qᶜˡ = identity, qᶜⁱ = identity),
                                  TotalWater())

    relax(target) = Relaxation(; rate, mask, target)
    return (u = relax(u), v = relax(v), θ = relax(θ_target), qᵉ = relax(qᵗ_target))
end

"""
$(TYPEDSIGNATURES)

Build a Breeze child atmosphere over `child_grid` nested in `parent_atmosphere`, wrapped in a
`NestedModel`. The child's lateral boundary conditions (`ρ, ρu, ρv, ρe, <moisture>`) — and, when
`relaxation_rate` (s⁻¹) is given, its interior Davies relaxation over `relaxation_mask` — are derived
on the fly from the parent's raw state (see [`nested_lateral_boundary_conditions`] /
[`nested_relaxation_forcings`]); no materialized parent prognostic series is needed. Remaining keyword
arguments forward to [`atmosphere_simulation`](@ref) (`dynamics`, `microphysics`, advection, …); any
`boundary_conditions` / `forcing` the caller passes are merged per-side with the parent-derived ones
(caller wins). This is the combined `parent → child(parent) → NestedModel` constructor.
"""
function NumericalEarth.EarthSystemModels.NestedSimulations.nested_atmosphere_model(
            parent_atmosphere::PrescribedAtmosphere, child_grid;
            relaxation_rate = nothing,
            relaxation_mask = 1,
            sides = (:west, :east, :south, :north),
            thermodynamic_constants = ThermodynamicConstants(eltype(child_grid)),
            microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()),
            boundary_conditions = NamedTuple(),
            forcing = NamedTuple(),
            kw...)

    moisture_name = moisture_prognostic_name(microphysics)

    nested_bcs = nested_lateral_boundary_conditions(parent_atmosphere, thermodynamic_constants, moisture_name; sides)
    davies = isnothing(relaxation_rate) ? (;) :
             nested_relaxation_forcings(parent_atmosphere, thermodynamic_constants;
                                        rate = relaxation_rate, mask = relaxation_mask)

    child = NumericalEarth.Atmospheres.atmosphere_model(child_grid;
                thermodynamic_constants, microphysics,
                boundary_conditions = merge_boundary_conditions(nested_bcs, NamedTuple(boundary_conditions)),
                forcing = merge(davies, NamedTuple(forcing)),
                kw...)

    return NestedModel(parent_atmosphere, child)
end
