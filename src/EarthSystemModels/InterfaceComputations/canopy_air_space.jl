#####
##### `CanopyAirSpace` — a two-source canopy with a diagnostic canopy-air node.
#####
##### The canopy and the soil surface exchange with a massless **canopy-air node**
##### `(Tᵃᶜ, qᵃᶜ)` that drains to the atmosphere through the aerodynamic conductance.
##### Three diagnostic scalars are solved inside the Monin–Obukhov fixed point:
#####
#####   Tᵛ  — leaf temperature      (massless leaf: Rₙᵛ = Hᵛ + LEᵛ)
#####   Tᵍ — soil-skin temperature (Rₙᵍ = Hᵍ + LEᵍ + Λᵍ(Tᵍ − Tˡᵃ), conducts to the bulk)
#####   Tᵃᶜ — canopy-air node       (Kirchhoff flux continuity; what MOST sees)
#####
##### and the paired humidity node `qᵃᶜ`. The leaf sees the *shaded soil skin* `Tᵍ`,
##### not the bulk reservoir `Tˡᵃ`; the slab is driven only by the skin conduction.
#####
##### Reuse: `canopy_conductance_terms` (stomatal conductance `gᶜ`, put in series with
##### the leaf boundary layer, and `qᵛ⁺(Tᵛ)`) and `dry_layer_terms` (soil vapor
##### conductance `gᵍʷ = Gᵉ` and the front humidity `qᵉ`) are the *same* helpers the
##### standalone/composite humidity formulations use.
##### Based on ClimaLand (Deck et al. 2026, App. D2/D5, E3).
#####
##### A `CanopyAirSpace` is a *combined* formulation: pass the same object as both
##### `atmosphere_land_interface_temperature` and `atmosphere_land_interface_specific_humidity`.
##### `compute_interface_temperature` returns `Tᵃᶜ`; `compute_interface_humidity` returns
##### `qᵃᶜ`; both run the shared `canopy_air_space_solve`.
#####

# Sensible-heat analogue of `atmospheric_vapor_flux`: the atmospheric sensible flux
# `𝒬ᵀ = -ρᵃᵗ cᵖ u★ θ★` (positive upward) from the previous iterate, and the
# node-to-air temperature increment `Δθ = Tᵃᶜ⁻ − θᵃᵗ`. Together they close the
# temperature node in the same `Δ`-multiplied form the humidity node uses.
@inline function atmospheric_sensible_flux(Ψₛ, Ψₐ, θᵃᵗ, ℂᵃᵗ)
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Ψₐ.T, Ψₐ.p, Ψₐ.q)
    cᵖ  = AtmosphericThermodynamics.cp_m(ℂᵃᵗ, Ψₐ.q)
    𝒬ᵀ  = - ρᵃᵗ * cᵖ * Ψₛ.fluxes.u★ * Ψₛ.fluxes.θ★
    Δθ  = Ψₛ.temperature - θᵃᵗ
    return 𝒬ᵀ, Δθ
end

"""
    struct CanopyInterception

Marker enabling the wet-canopy (interception) vapor branch of a [`CanopyAirSpace`](@ref). A wet
canopy evaporates intercepted water at the *potential* (stomata-free) rate through
the leaf boundary layer only, so the leaf vapor conductance blends the dry path
(stomata in series with the boundary layer) with the wet `gʷ = ρᵃᵗ · LAI · gᵇ`
by the wet fraction

```math
f_{wet} = (Wᶜ / Wᶜᵐᵃˣ)^{2/3}, \\qquad Wᶜᵐᵃˣ = c · LAI
```

([Deardorff, 1978](@cite deardorff1978)). The store `Wᶜ` and its capacity `Wᶜᵐᵃˣ = c·LAI`
are owned by the [`InterceptingHydrology`](@ref) wrapping the soil; the interface reads both
and normalizes `fʷ` by the store's *own* capacity. The leaf boundary conductance `gᵇ` is the
`leaf_boundary_conductance` on the [`CanopyAirSpace`](@ref).
"""
struct CanopyInterception end

Base.summary(::CanopyInterception) = "CanopyInterception"

# Deardorff (1978) wet fraction fʷ = (Wᶜ/Wᶜᵐᵃˣ)^(2/3), normalized by the store's own
# capacity Wᶜᵐᵃˣ (published by `InterceptingHydrology`). No interception ⇒ 0, recovering the
# dry CAS bit-for-bit; a zero capacity (no store, or a bare tile) also gives 0.
@inline wet_canopy_fraction(::Nothing, hydrology, LAI) = zero(LAI)
@inline function wet_canopy_fraction(::CanopyInterception, hydrology, LAI)
    FT    = typeof(LAI)
    Wᶜ    = convert(FT, hydrology.canopy_water_storage)
    Wᶜᵐᵃˣ = convert(FT, hydrology.canopy_water_capacity)
    return ifelse(Wᶜᵐᵃˣ > zero(FT),
                  clamp((max(Wᶜ, zero(FT)) / Wᶜᵐᵃˣ)^convert(FT, 2//3), zero(FT), one(FT)),
                  zero(FT))
end

#####
##### Undercanopy conductance closures — the ground ↔ canopy-air sensible/vapor coupling.
#####

abstract type AbstractUndercanopyConductance end

"""
    ConstantUndercanopyConductance(conductance)

Constant ground↔canopy-air aerodynamic conductance `gᵘᶜ` (m s⁻¹), independent of canopy
density and wind. A bare `Number` passed to [`CanopyAirSpace`](@ref)'s
`undercanopy_conductance` is wrapped in this closure.

```jldoctest
using NumericalEarth

ConstantUndercanopyConductance(0.013)

# output
ConstantUndercanopyConductance(gᵘᶜ=0.013)
```
"""
struct ConstantUndercanopyConductance{FT} <: AbstractUndercanopyConductance
    conductance :: FT
end

"""
    AreaIndexUndercanopyConductance(FT = Oceananigans.defaults.FloatType;
                                    drag_coefficient = 0.006,
                                    stem_area_index = 0,
                                    minimum_shielding = 0.1)

Ground↔canopy-air aerodynamic conductance that responds to canopy density and wind
(the PALADYN form; [Willeit and Ganopolski (2016)](@cite willeit2016)):

```math
gᵘᶜ = \\frac{C \\, Vₐ}{\\max\\!\\left(1 - e^{-(LAI + SAI)}, ε\\right)},
```

with `drag_coefficient` `C`, the surface wind speed `Vₐ`, and the canopy shielding
`1 − e^{−(LAI+SAI)}` (`stem_area_index` `SAI` counts the leafless woody area). A denser
canopy shields the ground more strongly and decouples it from the canopy air; a sparse
canopy (`LAI → 0`) leaves the ground ventilating at the aerodynamic limit, so the result
is capped at the aerodynamic transfer velocity `u★²/Vₐ` — the ground cannot ventilate to
the canopy air faster than the canopy air ventilates to the atmosphere. `minimum_shielding`
(`ε`) floors the shielding so the sparse-canopy limit stays finite. Both guards are
additions to the PALADYN form, which leaves `gᵘᶜ` unbounded as the canopy vanishes and
relies on the series aerodynamic resistance instead.

```jldoctest
using NumericalEarth

AreaIndexUndercanopyConductance()

# output
AreaIndexUndercanopyConductance(C=0.006, SAI=0.0, ε=0.1)
```
"""
struct AreaIndexUndercanopyConductance{FT} <: AbstractUndercanopyConductance
    drag_coefficient  :: FT
    stem_area_index   :: FT
    minimum_shielding :: FT
end

AreaIndexUndercanopyConductance(FT::Type = Oceananigans.defaults.FloatType;
                                drag_coefficient = 0.006,
                                stem_area_index = 0,
                                minimum_shielding = 0.1) =
    AreaIndexUndercanopyConductance(convert(FT, drag_coefficient),
                                    convert(FT, stem_area_index),
                                    convert(FT, minimum_shielding))

"""
    FrictionVelocityUndercanopyConductance(FT = Oceananigans.defaults.FloatType;
                                           dense_canopy_coefficient = 0.004,
                                           ground_roughness_length = 0.01,
                                           stem_area_index = 0,
                                           kinematic_viscosity = 1.5e-5)

Ground↔canopy-air aerodynamic conductance of CLM5, after
[Zeng et al. (2005)](@cite zeng2005undercanopy): the velocity scale is the friction
velocity `u★` (canopy-top shear drives the undercanopy turbulence), and the transfer
coefficient interpolates between a dense-canopy constant and a bare-ground
roughness-Reynolds-number law,

```math
gᵘᶜ = Cₛ u★, \\qquad Cₛ = Cₛᵇ W + Cₛᵈ (1 - W), \\qquad W = e^{-(LAI + SAI)},
\\qquad Cₛᵇ = \\frac{k}{0.13} \\left(\\frac{z₀ᵍ u★}{ν}\\right)^{-0.45},
```

with the dense-canopy coefficient `Cₛᵈ` (`dense_canopy_coefficient`, 0.004 after
Dickinson et al. 1993), the ground roughness length `z₀ᵍ` (`ground_roughness_length`, m),
the kinematic viscosity of air `ν` (`kinematic_viscosity`, m² s⁻¹), and the von Kármán
constant `k = 0.4` (CLM5 Tech Note Eqs. 2.5.116–2.5.121). Unlike
[`AreaIndexUndercanopyConductance`](@ref), the sparse-canopy limit is closed physically
by the bare-ground law rather than floored, and calm air (`u★ → 0`) decouples the ground.

```jldoctest
using NumericalEarth

FrictionVelocityUndercanopyConductance()

# output
FrictionVelocityUndercanopyConductance(Cₛᵈ=0.004, z₀ᵍ=0.01, SAI=0.0)
```
"""
struct FrictionVelocityUndercanopyConductance{FT} <: AbstractUndercanopyConductance
    dense_canopy_coefficient :: FT
    ground_roughness_length  :: FT
    stem_area_index          :: FT
    kinematic_viscosity      :: FT
end

FrictionVelocityUndercanopyConductance(FT::Type = Oceananigans.defaults.FloatType;
                                       dense_canopy_coefficient = 0.004,
                                       ground_roughness_length = 0.01,
                                       stem_area_index = 0,
                                       kinematic_viscosity = 1.5e-5) =
    FrictionVelocityUndercanopyConductance(convert(FT, dense_canopy_coefficient),
                                           convert(FT, ground_roughness_length),
                                           convert(FT, stem_area_index),
                                           convert(FT, kinematic_viscosity))

Base.summary(u::ConstantUndercanopyConductance) =
    string("ConstantUndercanopyConductance(gᵘᶜ=", prettysummary(u.conductance), ")")
Base.show(io::IO, u::ConstantUndercanopyConductance) = print(io, summary(u))

Base.summary(u::AreaIndexUndercanopyConductance) =
    string("AreaIndexUndercanopyConductance(C=", prettysummary(u.drag_coefficient),
           ", SAI=", prettysummary(u.stem_area_index),
           ", ε=", prettysummary(u.minimum_shielding), ")")
Base.show(io::IO, u::AreaIndexUndercanopyConductance) = print(io, summary(u))

Base.summary(u::FrictionVelocityUndercanopyConductance) =
    string("FrictionVelocityUndercanopyConductance(Cₛᵈ=", prettysummary(u.dense_canopy_coefficient),
           ", z₀ᵍ=", prettysummary(u.ground_roughness_length),
           ", SAI=", prettysummary(u.stem_area_index), ")")
Base.show(io::IO, u::FrictionVelocityUndercanopyConductance) = print(io, summary(u))

@inline undercanopy_conductance(u::ConstantUndercanopyConductance, LAI, Vₐ, u★) =
    convert(typeof(LAI), u.conductance)

@inline function undercanopy_conductance(u::AreaIndexUndercanopyConductance, LAI, Vₐ, u★)
    FT = typeof(LAI)
    C  = convert(FT, u.drag_coefficient)
    ε  = convert(FT, u.minimum_shielding)
    Λ  = LAI + convert(FT, u.stem_area_index)
    gᵘ = C * Vₐ / max(1 - exp(-Λ), ε)
    gᵃ = u★^2 / max(Vₐ, eps(FT))
    return min(gᵘ, gᵃ)
end

@inline function undercanopy_conductance(u::FrictionVelocityUndercanopyConductance, LAI, Vₐ, u★)
    FT  = typeof(LAI)
    W   = exp(-(LAI + convert(FT, u.stem_area_index)))
    Cᵈ  = convert(FT, u.dense_canopy_coefficient)
    z₀ᵍ = convert(FT, u.ground_roughness_length)
    ν   = convert(FT, u.kinematic_viscosity)
    k   = convert(FT, 2//5)
    a   = convert(FT, 13//100)
    n   = convert(FT, 9//20)
    # Cₛᵇ u★ grouped as u★^(1−n) so u★ → 0 gives 0 rather than Inf·0.
    gᵇ = (k / a) * (z₀ᵍ / ν)^(-n) * u★^(1 - n)
    return W * gᵇ + (1 - W) * Cᵈ * u★
end

# A bare number is the constant closure; a closure passes through.
undercanopy_conductance_model(x::Number, FT) = ConstantUndercanopyConductance(convert(FT, x))
undercanopy_conductance_model(x::AbstractUndercanopyConductance, FT) = x

"""
    struct CanopyAirSpace

Two-source canopy + soil surface with a diagnostic canopy-air node. Solves the
leaf temperature `Tᵛ`, the soil-skin temperature `Tᵍ`, and the canopy-air node
`(Tᵃᶜ, qᵃᶜ)` inside the Monin–Obukhov fixed point. Use the same object in both the
temperature and specific-humidity interface slots.

Fields:
- `soil`   : the soil vapor branch (a [`DryLayerHumidity`](@ref)).
- `canopy` : the leaf vapor/photosynthesis branch (a [`CanopyConductanceHumidity`](@ref)).
- `soil_skin_flux` : skin↔bulk conduction `Λᵍ = κᵀ/ℓᵀ` (a [`SoilConductiveFlux`](@ref)).
- `leaf_albedo`, `ground_albedo` : broadband shortwave albedos.
- `canopy_emissivity_max`, `ground_emissivity` : longwave emissivities (`εᵛ = εᵐᵃˣ(1 − e^{−LAI})`).
- `extinction`, `clumping` : Beer–Lambert `K`, `Ω` for the shortwave split.
- `leaf_boundary_conductance` : per-leaf boundary-layer conductance `gᵇ` (m s⁻¹) →
  sensible `gˡʰ = ρcₚ·LAI·gᵇ`, vapor `gʷ = ρ·LAI·gᵇ` (in series with the stomata
  when dry, alone over the wetted fraction).
- `undercanopy_conductance` : ground↔canopy-air conductance closure → `gᵍʰ = ρcₚ·gᵘᶜ`;
  a `Number` (m s⁻¹, wrapped as [`ConstantUndercanopyConductance`](@ref)), an
  [`AreaIndexUndercanopyConductance`](@ref) (PALADYN: canopy density and wind), or a
  [`FrictionVelocityUndercanopyConductance`](@ref) (CLM5: canopy density and `u★`).
- `inner_iterations`, `relaxation` : damped-Newton settings for the coupled solve.
- `interception` : wet-canopy vapor branch parameters (a [`CanopyInterception`](@ref)),
  or `nothing` for a dry canopy (the default; recovers the current CAS bit-for-bit).
- `phase` : saturation phase (Liquid).
"""
struct CanopyAirSpace{S, C, RF, FT, U, I, Φ}
    soil                      :: S
    canopy                    :: C
    soil_skin_flux             :: RF
    leaf_albedo               :: FT
    ground_albedo             :: FT
    canopy_emissivity_max     :: FT
    ground_emissivity         :: FT
    extinction                :: FT
    clumping                  :: FT
    leaf_boundary_conductance :: FT
    undercanopy_conductance   :: U
    inner_iterations          :: Int
    relaxation                :: FT
    interception              :: I
    phase                     :: Φ
end

function CanopyAirSpace(FT=Oceananigans.defaults.FloatType;
                        soil,
                        canopy                    = CanopyConductanceHumidity(FT),
                        soil_skin_flux             = SoilConductiveFlux(1.5, 0.05),
                        leaf_albedo               = 0.15,
                        ground_albedo             = 0.15,
                        canopy_emissivity_max     = 0.98,
                        ground_emissivity         = 0.96,
                        extinction                = 0.5,
                        clumping                  = 1,
                        leaf_boundary_conductance = 0.02,
                        undercanopy_conductance   = 0.013,
                        inner_iterations          = 40,
                        relaxation                = 1//2,
                        interception              = nothing,
                        phase                     = AtmosphericThermodynamics.Liquid())

    return CanopyAirSpace(soil, canopy, soil_skin_flux,
                          convert(FT, leaf_albedo), convert(FT, ground_albedo),
                          convert(FT, canopy_emissivity_max), convert(FT, ground_emissivity),
                          convert(FT, extinction), convert(FT, clumping),
                          convert(FT, leaf_boundary_conductance),
                          undercanopy_conductance_model(undercanopy_conductance, FT),
                          inner_iterations, convert(FT, relaxation), interception, phase)
end

Base.summary(::CanopyAirSpace) = "CanopyAirSpace"
Base.show(io::IO, c::CanopyAirSpace) =
    print(io, "CanopyAirSpace(soil=", summary(c.soil), ", canopy=", summary(c.canopy), ")")

Adapt.adapt_structure(to, c::CanopyAirSpace) =
    CanopyAirSpace(adapt(to, c.soil), adapt(to, c.canopy), c.soil_skin_flux,
                   c.leaf_albedo, c.ground_albedo, c.canopy_emissivity_max, c.ground_emissivity,
                   c.extinction, c.clumping, c.leaf_boundary_conductance,
                   c.undercanopy_conductance, c.inner_iterations, c.relaxation,
                   c.interception, c.phase)

# Materialization / identity — delegate to the sub-models so the per-cell interface
# state carries the soil saturation, bulk temperature, and LAI the branches read.
@inline interface_phase(c::CanopyAirSpace) = interface_phase(c.soil)
# The soil branch always publishes the saturation 𝒮; a canopy with interception
# additionally pulls the prognostic canopy water store Wᶜ (→ fʷ).
@inline interface_hydrology_state(i, j, grid, c::CanopyAirSpace, land_state) =
    canopy_air_space_hydrology_state(c.interception, i, j, grid, c, land_state)
@inline canopy_air_space_hydrology_state(::Nothing, i, j, grid, c, land_state) =
    interface_hydrology_state(i, j, grid, c.soil, land_state)
@inline canopy_air_space_hydrology_state(::CanopyInterception, i, j, grid, c, land_state) =
    merge(interface_hydrology_state(i, j, grid, c.soil, land_state),
          (canopy_water_storage  = state2dindex(land_state.canopy_water_storage, i, j),
           canopy_water_capacity = state2dindex(land_state.canopy_water_capacity, i, j)))
@inline interface_energy_state(i, j, grid, c::CanopyAirSpace, land_state) =
    interface_energy_state(i, j, grid, c.soil, land_state)
@inline canopy_leaf_area_index(c::CanopyAirSpace) = canopy_leaf_area_index(c.canopy)
@inline interface_vegetation_state(i, j, grid, c::CanopyAirSpace, vegetation, time_interpolator) =
    interface_vegetation_state(i, j, grid, c.canopy, vegetation, time_interpolator)

# Leaf vapor conductance: on the transpiring fraction the stomata act in series with
# the leaf boundary layer (ClimaLand Eq E17); the wetted fraction `fʷ` bypasses the
# stomata and evaporates through the boundary layer alone. `LAI → 0` sends both to zero.
@inline function leaf_vapor_conductance(gᶜ, gʷ, fʷ)
    gᵈ = ifelse(gᶜ + gʷ > 0, gᶜ * gʷ / (gᶜ + gʷ), zero(gᶜ))
    return (1 - fʷ) * gᵈ + fʷ * gʷ
end

# dqᵛ⁺/dT by centered difference — the Newton derivative of each balance's latent term.
@inline function saturation_humidity_slope(ℂᵃᵗ, T, pᵃᵗ, phase)
    δ = convert(typeof(T), 1//100)
    q⁺ = saturation_specific_humidity(ℂᵃᵗ, T + δ, pᵃᵗ, phase)
    q⁻ = saturation_specific_humidity(ℂᵃᵗ, T - δ, pᵃᵗ, phase)
    return (q⁺ - q⁻) / 2δ
end

"""
    canopy_air_space_solve(c::CanopyAirSpace, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)

Solve the coupled diagnostic state `(Tᵛ, Tᵍ, Tᵃᶜ, qᵃᶜ)` for one cell. `Ψₛ` is the
previous fixed-point iterate (carrying the MO scales and the previous node values),
`Ψᵢ.T` is the bulk reservoir `Tˡᵃ`, and `Ψᵣ` the interface radiation state. A short
damped-Newton inner loop advances the two skin balances against the node; the node
uses the `Δ`-multiplied Kirchhoff form so it stays finite as the flux vanishes.
"""
@inline function canopy_air_space_solve(c::CanopyAirSpace, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    FT  = eltype(Ψₛ)
    pᵃᵗ = Ψₐ.p
    qᵃᵗ = Ψₐ.q
    Tᵃᵗ = Ψₐ.T
    ℒ   = AtmosphericThermodynamics.latent_heat_vapor(ℂᵃᵗ, Tᵃᵗ)
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)
    cᵖ  = AtmosphericThermodynamics.cp_m(ℂᵃᵗ, qᵃᵗ)
    θᵃᵗ = surface_atmosphere_temperature(Ψₐ, ℙₐ)

    Tˡᵃ = Ψᵢ.T
    LAI = Ψₛ.vegetation.leaf_area_index

    # Aerodynamic drivers from the previous outer iterate (held fixed through the inner loop).
    𝒬ᵀ, Δθᵃ = atmospheric_sensible_flux(Ψₛ, Ψₐ, θᵃᵗ, ℂᵃᵗ)
    Jᵃ, Δqᵃ = atmospheric_vapor_flux(Ψₛ, Ψₐ, ℂᵃᵗ)

    # Land surface velocities are zero, so the surface wind speed is the atmospheric one.
    Vₐ  = sqrt(Ψₐ.u^2 + Ψₐ.v^2)
    gᵘᶜ = undercanopy_conductance(c.undercanopy_conductance, LAI, Vₐ, Ψₛ.fluxes.u★)

    gˡʰ = ρᵃᵗ * cᵖ * LAI * c.leaf_boundary_conductance
    gᵍʰ = ρᵃᵗ * cᵖ * gᵘᶜ
    gᵍʷ = ρᵃᵗ * gᵘᶜ   # undercanopy vapor conductance (wet-soil limit)
    Λ   = convert(FT, skin_conductance(c.soil_skin_flux))

    # Leaf boundary-layer vapor mass conductance `gʷ = ρᵃᵗ·LAI·gᵇ`: in series with the
    # stomata on the dry (transpiring) fraction, alone on the wetted fraction `fʷ`
    # (Deardorff 1978), so intercepted water evaporates at the potential rate.
    fʷ = wet_canopy_fraction(c.interception, Ψₛ.hydrology, LAI)
    gʷ = ρᵃᵗ * LAI * c.leaf_boundary_conductance

    # Shortwave split + longwave emissivities (broadband).
    σ   = Ψᵣ.σ
    SW  = Ψᵣ.ℐꜜˢʷ
    LWd = Ψᵣ.ℐꜜˡʷ
    εᵛ = c.canopy_emissivity_max * (1 - exp(-LAI))
    εᵍ = c.ground_emissivity
    ftrans    = exp(-c.extinction * LAI * c.clumping)
    SWᵛ = (1 - c.leaf_albedo) * (1 - ftrans) * SW
    SWᵍ = ftrans * (1 - c.ground_albedo) * SW

    Tᵛ  = Tˡᵃ
    Tᵍ = Tˡᵃ
    Tᵃᶜ = Ψₛ.temperature
    qᵃᶜ = Ψₛ.specific_humidity
    relaxation  = c.relaxation
    max_ΔT = convert(FT, 25)   # per-iterate step cap: keeps the damped Newton in-range
    Tₗₒ, Tₕᵢ = convert(FT, 180), convert(FT, 340)  # physical band; guards qˢᵃᵗ against transient overshoot
    tiny = eps(FT)

    for _ in 1:c.inner_iterations
        gᶜ, qᵛ   = canopy_conductance_terms(c.canopy, Tᵛ, Ψₛ, Ψₐ, Ψᵣ, ℙₐ)
        Gᵉ, qᵉ, fᵈ, qᵍ⁺ = dry_layer_terms(c.soil, Tᵍ, Ψₛ, Ψₐ, ℙₐ)

        # Blend the dry-layer series soil branch (front qᵉ through Gᵉ) with the
        # saturated-skin wet branch (qᵍ⁺ through the undercanopy conductance gᵍʷ),
        # weight `fᵈ` from the soil model.
        Gᵉ⁺ = fᵈ * Gᵉ + (1 - fᵈ) * gᵍʷ
        qᵉ  = ifelse(Gᵉ⁺ > tiny, (fᵈ * Gᵉ * qᵉ + (1 - fᵈ) * gᵍʷ * qᵍ⁺) / Gᵉ⁺, qᵍ⁺)
        Gᵉ  = Gᵉ⁺

        gˡʷ = leaf_vapor_conductance(gᶜ, gʷ, fʷ)

        # Δ-multiplied Kirchhoff node (as the humidity node in CompositeSurfaceHumidity);
        # guard the transient case where the aerodynamic and surface conductances cancel
        # (Dᵀ ≈ 0) before the outer MO loop is consistent, keeping the node finite.
        Dᵀ  = (gᵍʰ + gˡʰ) * Δθᵃ + 𝒬ᵀ
        Tᵃᶜ★ = ((gᵍʰ * Tᵍ + gˡʰ * Tᵛ) * Δθᵃ + 𝒬ᵀ * θᵃᵗ) / Dᵀ
        Tᵃᶜ = ifelse((Dᵀ == 0) | !isfinite(Tᵃᶜ★), Tᵃᶜ, Tᵃᶜ★)
        Dᵠ  = (Gᵉ + gˡʷ) * Δqᵃ + Jᵃ
        qᵃᶜ★ = ((Gᵉ * qᵉ + gˡʷ * qᵛ) * Δqᵃ + Jᵃ * qᵃᵗ) / Dᵠ
        qᵃᶜ = ifelse((Dᵠ == 0) | !isfinite(qᵃᶜ★), qᵃᶜ, qᵃᶜ★)

        # At the fixed point the node is a conductance-weighted mean of its sources, so
        # it lies in their hull; the Δ-multiplied transient form loses convexity when
        # Δqᵃ or Jᵃ flips sign (calm transitions) and can shoot the node far outside
        # any physical humidity. Bound it by the source states.
        Tᵃᶜ = clamp(Tᵃᶜ, min(θᵃᵗ, Tᵛ, Tᵍ), max(θᵃᵗ, Tᵛ, Tᵍ))
        qᵃᶜ = clamp(qᵃᶜ, min(qᵃᵗ, qᵛ, qᵉ), max(qᵃᵗ, qᵛ, qᵉ))

        LWꜜᵍ     = (1 - εᵛ) * LWd + εᵛ * σ * Tᵛ^4
        LWꜛᵍ     = εᵍ * σ * Tᵍ^4 + (1 - εᵍ) * LWꜜᵍ
        LWᵛ = εᵛ * (LWd + LWꜛᵍ) - 2 * εᵛ * σ * Tᵛ^4
        LWᵍ = εᵍ * (LWꜜᵍ - σ * Tᵍ^4)

        Rᵥ   = SWᵛ + LWᵛ
        resᵥ = Rᵥ - gˡʰ * (Tᵛ - Tᵃᶜ) - ℒ * gˡʷ * (qᵛ - qᵃᶜ)
        dRᵥ  = -8 * εᵛ * σ * Tᵛ^3 - gˡʰ - ℒ * gˡʷ * saturation_humidity_slope(ℂᵃᵗ, Tᵛ, pᵃᵗ, c.phase)
        Tᵛ   = ifelse(abs(dRᵥ) < tiny, Tᵃᶜ, Tᵛ - clamp(relaxation * resᵥ / dRᵥ, -max_ΔT, max_ΔT))
        Tᵛ   = clamp(Tᵛ, Tₗₒ, Tₕᵢ)

        Rᵍ   = SWᵍ + LWᵍ
        resᵍ = Rᵍ - gᵍʰ * (Tᵍ - Tᵃᶜ) - ℒ * Gᵉ * (qᵉ - qᵃᶜ) - Λ * (Tᵍ - Tˡᵃ)
        dRᵍ  = -4 * εᵍ * σ * Tᵍ^3 - gᵍʰ - Λ - ℒ * Gᵉ * saturation_humidity_slope(ℂᵃᵗ, Tᵍ, pᵃᵗ, c.phase)
        Tᵍ  = Tᵍ - clamp(relaxation * resᵍ / dRᵍ, -max_ΔT, max_ΔT)
        Tᵍ  = clamp(Tᵍ, Tₗₒ, Tₕᵢ)
    end

    # Converged diagnostics: per-surface flux shares, the skin→slab conduction, and
    # the effective radiating (LST) temperature σ Teff⁴ ≡ LWu (upwelling to space).
    gᶜ, qᵛ   = canopy_conductance_terms(c.canopy, Tᵛ, Ψₛ, Ψₐ, Ψᵣ, ℙₐ)
    Gᵉ, qᵉ, fᵈ, qᵍ⁺ = dry_layer_terms(c.soil, Tᵍ, Ψₛ, Ψₐ, ℙₐ)
    Gᵉ⁺ = fᵈ * Gᵉ + (1 - fᵈ) * gᵍʷ
    qᵉ  = ifelse(Gᵉ⁺ > tiny, (fᵈ * Gᵉ * qᵉ + (1 - fᵈ) * gᵍʷ * qᵍ⁺) / Gᵉ⁺, qᵍ⁺)
    Gᵉ  = Gᵉ⁺
    gˡʷ = leaf_vapor_conductance(gᶜ, gʷ, fʷ)

    # Re-solve the node against the final skins: inside the loop the node update
    # precedes the skin updates, so it exits one iterate stale and the flux shares
    # below would miss closure against the atmospheric flux.
    Dᵀ  = (gᵍʰ + gˡʰ) * Δθᵃ + 𝒬ᵀ
    Tᵃᶜ★ = ((gᵍʰ * Tᵍ + gˡʰ * Tᵛ) * Δθᵃ + 𝒬ᵀ * θᵃᵗ) / Dᵀ
    Tᵃᶜ = ifelse((Dᵀ == 0) | !isfinite(Tᵃᶜ★), Tᵃᶜ, Tᵃᶜ★)
    Dᵠ  = (Gᵉ + gˡʷ) * Δqᵃ + Jᵃ
    qᵃᶜ★ = ((Gᵉ * qᵉ + gˡʷ * qᵛ) * Δqᵃ + Jᵃ * qᵃᵗ) / Dᵠ
    qᵃᶜ = ifelse((Dᵠ == 0) | !isfinite(qᵃᶜ★), qᵃᶜ, qᵃᶜ★)

    # Same hull bound as in the loop.
    Tᵃᶜ = clamp(Tᵃᶜ, min(θᵃᵗ, Tᵛ, Tᵍ), max(θᵃᵗ, Tᵛ, Tᵍ))
    qᵃᶜ = clamp(qᵃᶜ, min(qᵃᵗ, qᵛ, qᵉ), max(qᵃᵗ, qᵛ, qᵉ))

    # Damp the node across outer iterates: the aerodynamic drivers (𝒬ᵀ, Δθᵃ, Jᵃ, Δqᵃ)
    # are one outer iterate stale, and a node that commits fully to them each pass
    # sustains a period-2 limit cycle against the MOST scales in the calm
    # free-convection limit. The converged solution is unchanged.
    Tᵃᶜ = (1 - relaxation) * Ψₛ.temperature + relaxation * Tᵃᶜ
    qᵃᶜ = (1 - relaxation) * Ψₛ.specific_humidity + relaxation * qᵃᶜ

    LWꜜᵍ = (1 - εᵛ) * LWd + εᵛ * σ * Tᵛ^4
    LWꜛᵍ = εᵍ * σ * Tᵍ^4 + (1 - εᵍ) * LWꜜᵍ
    LWu   = (1 - εᵛ) * LWꜛᵍ + εᵛ * σ * Tᵛ^4
    Teff  = ifelse(σ > 0, (LWu / σ)^convert(FT, 1//4), Tᵃᶜ)

    Hᵛ    = gˡʰ * (Tᵛ - Tᵃᶜ)
    Hᵍ    = gᵍʰ * (Tᵍ - Tᵃᶜ)
    LEᵛ   = ℒ * gˡʷ * (qᵛ - qᵃᶜ)              # total leaf latent (transpiration + wet-canopy)
    LEᵍ   = ℒ * Gᵉ * (qᵉ - qᵃᶜ)
    Gᶜ = Λ * (Tᵍ - Tˡᵃ)
    Eʷ = fʷ * gʷ * (qᵛ - qᵃᶜ)           # wet-canopy evaporation, mass flux (kg m⁻² s⁻¹, up)
    LEʷ = ℒ * Eʷ                           # wet-canopy latent heat (W m⁻², up); LEᵛ − LEʷ = transpiration

    return (; Tᵛ = convert(FT, Tᵛ), Tᵍ = convert(FT, Tᵍ),
              Tᵃᶜ = convert(FT, Tᵃᶜ), qᵃᶜ = convert(FT, qᵃᶜ),
              Teff = convert(FT, Teff),
              Hᵛ = convert(FT, Hᵛ), Hᵍ = convert(FT, Hᵍ),
              LEᵛ = convert(FT, LEᵛ), LEᵍ = convert(FT, LEᵍ),
              Gᶜ = convert(FT, Gᶜ), Eʷ = convert(FT, Eʷ),
              LEʷ = convert(FT, LEʷ))
end

@inline compute_interface_temperature(c::CanopyAirSpace,
                                      interface_state, atmosphere_state, interior_state,
                                      radiation_state, interface_properties,
                                      atmosphere_properties, interior_properties) =
    canopy_air_space_solve(c, interface_state, atmosphere_state, interior_state,
                           radiation_state, atmosphere_properties).Tᵃᶜ

@inline compute_interface_humidity(c::CanopyAirSpace, Tₛ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ) =
    canopy_air_space_solve(c, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ).qᵃᶜ

# Combined temperature + humidity: one shared solve returns both the canopy-air node
# temperature Tᵃᶜ and humidity qᵃᶜ, so the per-iterate inner solve runs once, not twice.
@inline function interface_temperature_and_humidity(c::CanopyAirSpace, ::CanopyAirSpace,
                                                    Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₛ, ℙₐ, ℙᵢ)
    sol = canopy_air_space_solve(c, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    return sol.Tᵃᶜ, sol.qᵃᶜ
end

"""
    struct CanopyAirSpaceDiagnostics

The atmosphere-facing `temperature` slot of a [`CanopyAirSpace`](@ref) interface: the
canopy-air node temperature the atmosphere sees, the per-source diagnostic temperatures,
and the two-source flux shares of the coupled solve. Downstream consumers dispatch on
this type — it signals that radiation is internalized in the soil-skin balance and the
slab is driven by the skin→bulk conduction (`ground_heat_flux`) rather than by a
separately added radiative flux.
"""
struct CanopyAirSpaceDiagnostics{F}
    interface              :: F   # canopy-air node Tᵃᶜ (what MOST sees)
    canopy                 :: F   # leaf temperature Tᵛ
    soil_skin              :: F   # soil-skin temperature Tᵍ
    effective              :: F   # radiating (LST) temperature Teff
    ground_heat_flux       :: F   # skin→bulk conduction Gᶜ
    canopy_latent_heat     :: F   # leaf transpiration LEᵛ
    soil_latent_heat       :: F   # soil evaporation LEᵍ
    canopy_sensible_heat   :: F   # leaf sensible Hᵛ
    soil_sensible_heat     :: F   # ground sensible Hᵍ
    canopy_evaporation     :: F   # wet-canopy evaporation Eʷ (kg m⁻² s⁻¹, up)
    canopy_wet_latent_heat :: F   # wet-canopy latent heat ℒ·Eʷ (W m⁻², up)
end

CanopyAirSpaceDiagnostics(grid) =
    CanopyAirSpaceDiagnostics(ntuple(_ -> Field{Center, Center, Nothing}(grid),
                                     Val(fieldcount(CanopyAirSpaceDiagnostics)))...)

Adapt.@adapt_structure CanopyAirSpaceDiagnostics

Base.summary(::CanopyAirSpaceDiagnostics) = "CanopyAirSpaceDiagnostics"
Base.show(io::IO, d::CanopyAirSpaceDiagnostics) = print(io, summary(d))
