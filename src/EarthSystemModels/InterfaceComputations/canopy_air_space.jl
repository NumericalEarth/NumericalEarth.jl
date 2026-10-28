#####
##### `CanopyInterface` — a two-source canopy (leaf + shaded ground) exchanging with a
##### canopy-air node that drains to the atmosphere through the aerodynamic conductance.
#####

#####
##### Undercanopy transfer velocities — the ground ↔ canopy-air sensible/vapor coupling.
#####

"""
$(TYPEDSIGNATURES)

Ground↔canopy-air transfer velocity that responds to canopy density and wind
(the PALADYN form; [Willeit and Ganopolski (2016)](@cite willeit2016)):

```math
wᵘᶜᵃⁿ = \\frac{Cᴰ \\, ΔU}{\\max\\!\\left(1 - e^{-(\\lambda^{\\mathrm{leaf}} + \\lambda^{\\mathrm{stem}})}, ε\\right)},
```

with `drag_coefficient` `Cᴰ`, the surface wind speed `ΔU`, and the canopy shielding
`1 − e^{−(λˡᵉᵃᶠ+λˢᵗᵉᵐ)}` (`stem_area_index` `λˢᵗᵉᵐ` counts the leafless woody area). A denser
canopy shields the ground more strongly and decouples it from the canopy air; a sparse
canopy (`λˡᵉᵃᶠ → 0`) leaves the ground ventilating at the aerodynamic limit, so the result
is capped at the aerodynamic transfer velocity `u★²/ΔU` — the ground cannot ventilate to
the canopy air faster than the canopy air ventilates to the atmosphere.

```jldoctest
using NumericalEarth

AreaIndexUndercanopyVelocity()

# output
AreaIndexUndercanopyVelocity(Cᴰ=0.006, λˢᵗᵉᵐ=0.0)
```
"""
struct AreaIndexUndercanopyVelocity{FT}
    drag_coefficient :: FT
    stem_area_index  :: FT
end

AreaIndexUndercanopyVelocity(FT::Type = Oceananigans.defaults.FloatType;
                             drag_coefficient = 0.006,
                             stem_area_index = 0) =
    AreaIndexUndercanopyVelocity(convert(FT, drag_coefficient),
                                 convert(FT, stem_area_index))

"""
$(TYPEDSIGNATURES)

Ground↔canopy-air transfer velocity of CLM5, after
[Zeng et al. (2005)](@cite zeng2005undercanopy): the velocity scale is the friction
velocity `u★` (canopy-top shear drives the undercanopy turbulence), and the transfer
coefficient interpolates between a dense-canopy constant and a bare-ground
roughness-Reynolds-number law,

```math
wᵘᶜᵃⁿ = Cₛ u★, \\qquad Cₛ = Cₛᵇ W + Cₛᵈ (1 - W), \\qquad W = e^{-(\\lambda^{\\mathrm{leaf}} + \\lambda^{\\mathrm{stem}})},
\\qquad Cₛᵇ = \\frac{ϰ}{0.13} \\left(\\frac{ℓᵍʳⁿᵈ u★}{ν}\\right)^{-0.45},
```

with the dense-canopy coefficient `Cₛᵈ` (`dense_canopy_coefficient`, 0.004 after
Dickinson et al. 1993), the ground roughness length `ℓᵍʳⁿᵈ` (`ground_roughness_length`, m),
the kinematic viscosity of air `ν` (`kinematic_viscosity`, m² s⁻¹), and the von Kármán
constant `ϰ = 0.4` (CLM5 Tech Note Eqs. 2.5.116–2.5.121). Unlike
[`AreaIndexUndercanopyVelocity`](@ref), the sparse-canopy limit is closed physically
by the bare-ground law rather than floored, and calm air (`u★ → 0`) decouples the ground.

```jldoctest
using NumericalEarth

FrictionVelocityUndercanopyVelocity()

# output
FrictionVelocityUndercanopyVelocity(Cₛᵈ=0.004, ℓᵍʳⁿᵈ=0.01, λˢᵗᵉᵐ=0.0)
```
"""
struct FrictionVelocityUndercanopyVelocity{FT}
    dense_canopy_coefficient :: FT
    ground_roughness_length  :: FT
    stem_area_index          :: FT
    kinematic_viscosity      :: FT
end

FrictionVelocityUndercanopyVelocity(FT::Type = Oceananigans.defaults.FloatType;
                                    dense_canopy_coefficient = 0.004,
                                    ground_roughness_length = 0.01,
                                    stem_area_index = 0,
                                    kinematic_viscosity = 1.5e-5) =
    FrictionVelocityUndercanopyVelocity(convert(FT, dense_canopy_coefficient),
                                        convert(FT, ground_roughness_length),
                                        convert(FT, stem_area_index),
                                        convert(FT, kinematic_viscosity))

Base.summary(u::AreaIndexUndercanopyVelocity) =
    string("AreaIndexUndercanopyVelocity(Cᴰ=", prettysummary(u.drag_coefficient),
           ", λˢᵗᵉᵐ=", prettysummary(u.stem_area_index), ")")
Base.show(io::IO, u::AreaIndexUndercanopyVelocity) = print(io, summary(u))

Base.summary(u::FrictionVelocityUndercanopyVelocity) =
    string("FrictionVelocityUndercanopyVelocity(Cₛᵈ=", prettysummary(u.dense_canopy_coefficient),
           ", ℓᵍʳⁿᵈ=", prettysummary(u.ground_roughness_length),
           ", λˢᵗᵉᵐ=", prettysummary(u.stem_area_index), ")")
Base.show(io::IO, u::FrictionVelocityUndercanopyVelocity) = print(io, summary(u))

# A `Number` is a constant transfer velocity (m s⁻¹).
@inline undercanopy_transfer_velocity(wᵘᶜᵃⁿ::Number, λˡᵉᵃᶠ, ΔU, u★) = convert(typeof(λˡᵉᵃᶠ), wᵘᶜᵃⁿ)

@inline function undercanopy_transfer_velocity(u::AreaIndexUndercanopyVelocity, λˡᵉᵃᶠ, ΔU, u★)
    FT = typeof(λˡᵉᵃᶠ)
    Cᴰ = convert(FT, u.drag_coefficient)
    area_index = λˡᵉᵃᶠ + convert(FT, u.stem_area_index)
    wᵘᶜᵃⁿ = Cᴰ * ΔU / max(1 - exp(-area_index), eps(FT))
    aerodynamic_limit = u★^2 / max(ΔU, eps(FT))
    return min(wᵘᶜᵃⁿ, aerodynamic_limit)
end

@inline function undercanopy_transfer_velocity(u::FrictionVelocityUndercanopyVelocity, λˡᵉᵃᶠ, ΔU, u★)
    FT   = typeof(λˡᵉᵃᶠ)
    W    = exp(-(λˡᵉᵃᶠ + convert(FT, u.stem_area_index)))
    Cᵈ   = convert(FT, u.dense_canopy_coefficient)
    ℓᵍʳⁿᵈ = convert(FT, u.ground_roughness_length)
    ν    = convert(FT, u.kinematic_viscosity)
    ϰ    = convert(FT, 2//5)
    a    = convert(FT, 13//100)
    n    = convert(FT, 9//20)
    # Cₛᵇ u★ grouped as u★^(1−n) so u★ → 0 gives 0 rather than Inf·0.
    wᵇ = (ϰ / a) * (ℓᵍʳⁿᵈ / ν)^(-n) * u★^(1 - n)
    return W * wᵇ + (1 - W) * Cᵈ * u★
end

"""
$(TYPEDSIGNATURES)

Empirical ground-surface resistance for the moist-soil (vanishing dry layer) evaporation
branch of a [`CanopyInterface`](@ref),

```math
rˢᵒⁱ = e^{a - b 𝒮},
```

with the current surface saturation `𝒮` supplied by land. The interface coefficients
`a` and `b` were fit by [Sellers et al. (1992)](@cite sellers1992) to the FIFE prairie flux
stations (`rˢᵒⁱ(1) ≈ 52` s m⁻¹). The FIFE sites carried litter, so the fit is an effective
soil-plus-litter resistance, an alternative to a
[`GroundSurfaceResistance`](@ref) rather than a companion to one.

```jldoctest
using NumericalEarth

ExponentialSoilResistance()

# output
ExponentialSoilResistance(a=8.206, b=4.255)
```
"""
struct ExponentialSoilResistance{FT}
    intercept :: FT
    slope     :: FT
end

ExponentialSoilResistance(FT::Type = Oceananigans.defaults.FloatType;
                      intercept = 8.206, slope = 4.255) =
    ExponentialSoilResistance(convert(FT, intercept), convert(FT, slope))

Base.summary(r::ExponentialSoilResistance) =
    string("ExponentialSoilResistance(a=", prettysummary(r.intercept),
           ", b=", prettysummary(r.slope), ")")
Base.show(io::IO, r::ExponentialSoilResistance) = print(io, summary(r))

@inline soil_surface_resistance(::Nothing, 𝒮) = zero(𝒮)
@inline function soil_surface_resistance(r::ExponentialSoilResistance, 𝒮)
    FT = typeof(𝒮)
    a  = convert(FT, r.intercept)
    b  = convert(FT, r.slope)
    return exp(a - b * clamp(𝒮, zero(FT), one(FT)))
end

"""
    GroundSurfaceResistance(FT = Oceananigans.defaults.FloatType; transfer_coefficient = 0.00633)

Bulk resistance of the ground surface to evaporation, in series with the undercanopy path
on both ground vapor branches of a [`CanopyInterface`](@ref),

```math
rᵍʳⁿᵈ = \\frac{1}{Cᵍʳⁿᵈ u★},
```

with a bulk transfer coefficient `Cᵍʳⁿᵈ` and the friction velocity `u★` standing in for the
wind speed near the ground. It lumps whatever covers the soil (plant litter, mulch, a
crust) into one parameter. The default `Cᵍʳⁿᵈ = 0.00633` reproduces the litter resistance of
[Sakaguchi and Zeng (2009)](@cite sakaguchi2009), Eq. 13, at unit litter area index,
`Cᵍʳⁿᵈ = 0.004 / (1 - e^{-1})`. A `CanopyInterface` also takes a `Number`, a constant
resistance in s m⁻¹.

```jldoctest
using NumericalEarth

GroundSurfaceResistance()

# output
GroundSurfaceResistance(C=0.00633)
```
"""
struct GroundSurfaceResistance{C}
    transfer_coefficient :: C
end

GroundSurfaceResistance(FT::Type = Oceananigans.defaults.FloatType; transfer_coefficient = 0.00633) =
    GroundSurfaceResistance(convert_if_number(FT, transfer_coefficient))

Adapt.@adapt_structure GroundSurfaceResistance

Base.summary(r::GroundSurfaceResistance) =
    string("GroundSurfaceResistance(C=", prettysummary(r.transfer_coefficient), ")")
Base.show(io::IO, r::GroundSurfaceResistance) = print(io, summary(r))

@inline ground_surface_resistance(::Nothing, u★) = zero(u★)
@inline ground_surface_resistance(r::Number, u★) = convert(typeof(u★), r)
@inline function ground_surface_resistance(r::GroundSurfaceResistance, u★)
    FT = typeof(u★)
    C  = convert(FT, r.transfer_coefficient)
    return 1 / max(C * u★, eps(FT))
end

"""
    struct CanopyInterface

Two-source canopy + ground surface with a canopy-air node, for forests: tall vegetation
whose crowns enclose a canopy air space with the ground as a second source beneath it.
Low vegetation (grass, crops) has no separable canopy air layer and belongs to a
single-source [`CanopyConductanceHumidity`](@ref) over the ground surface. Solves the
leaf temperature `Tˡᵉᵃᶠ`, the ground temperature `Tᵍʳⁿᵈ`, and the canopy-air node
`(Tᵃᶜᵃⁿ, qᵃᶜᵃⁿ)` inside the Monin–Obukhov fixed point (diagnostic node, the default),
or couples to the land component's prognostic canopy-air reservoir. Pass it as the interface state
formulation to solve temperature and specific humidity together. Over an
[`InterceptingHydrology`](@ref) the wetted fraction `fʷᵉᵗ = (Mᶜⁿᵖʸ/Mᶜⁿᵖʸ⁺)^{2/3}` of the leaf
([Deardorff, 1978](@cite deardorff1978)) evaporates intercepted water at the potential
rate through the leaf boundary layer alone; a dry canopy (`Mᶜⁿᵖʸ = 0`) has no wet branch.

Fields:
- `soil_humidity`   : the soil vapor branch (a [`DryLayerHumidity`](@ref)).
- `canopy_humidity` : the leaf vapor branch (a [`CanopyConductanceHumidity`](@ref)).
- `ground_heat_flux` : ground↔bulk conduction `Λᵍʳⁿᵈ = κᵀ/ℓᵀ` (a [`SoilConductiveFlux`](@ref),
  whose `κᵀ` and `ℓᵀ` may be per-cell `Field`s).
- `leaf_albedo`, `ground_albedo` : broadband shortwave albedos. A `Number`, or a
  `Field{Center, Center, Nothing}` of per-cell values (see [`atmosphere_land_interface`](@ref)).
- `maximum_canopy_emissivity`, `ground_emissivity` : longwave emissivities (`ϵˡᵉᵃᶠ = ϵᵐᵃˣ(1 − e^{−λˡᵉᵃᶠ})`),
  each a `Number` or a per-cell `Field`.
- `extinction`, `clumping` : Beer–Lambert extinction `K` and foliage clumping index `Ω` of the
  shortwave transmittance `e^{−KΩL}` (`Ω = 1` random foliage, `Ω < 1` foliage gathered into
  crowns that leave gaps).
- `leaf_boundary_layer_velocity` : constant per-leaf transfer velocity through the leaf
  boundary layer `wˡᵇˡ` (m s⁻¹) → sensible `gˡᵉᵃᶠᵀ = ρcᵖᵐ·λˡᵉᵃᶠ·wˡᵇˡ`, vapor `gˡᵇˡ = ρ·λˡᵉᵃᶠ·wˡᵇˡ` (in
  series with the stomata when dry, alone over the wetted fraction).
- `undercanopy_transfer_velocity` : ground↔canopy-air transfer velocity `wᵘᶜᵃⁿ` → `gᵍʳⁿᵈᵀ = ρcᵖᵐ·wᵘᶜᵃⁿ`;
  a constant `Number` (m s⁻¹), an [`AreaIndexUndercanopyVelocity`](@ref), or a
  [`FrictionVelocityUndercanopyVelocity`](@ref).
- `wet_soil_resistance` : soil surface resistance on the moist-soil (vanishing dry layer)
  vapor branch (a [`ExponentialSoilResistance`](@ref)), or `nothing` (the default: above the
  dry-layer onset the soil itself does not limit evaporation, and the ground-surface
  resistance and undercanopy path carry it).
- `ground_surface_resistance` : bulk ground-surface resistance in series on both ground
  vapor branches (a [`GroundSurfaceResistance`](@ref), the default, whose `Cᵍʳⁿᵈ` may be a
  per-cell `Field`, or a constant `Number` in s m⁻¹), or `nothing` for none.
- `inner_iterations` : quasi-Newton iterations of the coupled leaf and ground temperature solve.

The four optics slots accept a per-cell `Field{Center, Center, Nothing}` alongside a
`Number`, so a satellite albedo product reaches the two-source radiation balance cell by
cell. So do the ground conduction and the ground-surface resistance. The land flux kernel
localizes them before the canopy solve.

```jldoctest
using NumericalEarth
using Oceananigans
using NumericalEarth.EarthSystemModels.InterfaceComputations: local_interface_formulation

grid = LatitudeLongitudeGrid(size = (2, 1, 1), latitude = (10, 11), longitude = (10, 12),
                             z = (-1, 0), topology = (Bounded, Bounded, Bounded))

soil_humidity = DryLayerHumidity(Float64;
                                 dry_layer_depth = StorageBasedDryLayerDepth(Float64;
                                     maximum_dry_layer_depth = 0.015, dry_layer_onset_saturation = 0.5),
                                 vapor_exchange = DryLayerVaporPistonVelocity(Float64;
                                     minimum_dry_layer_depth = 1e-3, molecular_diffusivity = 2.4e-5))

leaf_albedo = Field{Center, Center, Nothing}(grid)
set!(leaf_albedo, (λ, φ) -> ifelse(λ < 11, 0.12, 0.35))

surface = CanopyInterface(Float64; soil_humidity, leaf_albedo)

local_interface_formulation(surface, 2, 1).leaf_albedo

# output
0.35
```
"""
struct CanopyInterface{S, C, RF, LA, GA, CE, GE, FT, U, W, L}
    soil_humidity                 :: S
    canopy_humidity               :: C
    ground_heat_flux              :: RF
    leaf_albedo                   :: LA
    ground_albedo                 :: GA
    maximum_canopy_emissivity     :: CE
    ground_emissivity             :: GE
    extinction                    :: FT
    clumping                      :: FT
    leaf_boundary_layer_velocity  :: FT
    undercanopy_transfer_velocity :: U
    wet_soil_resistance           :: W
    ground_surface_resistance     :: L
    inner_iterations              :: Int
end

function CanopyInterface(FT=Oceananigans.defaults.FloatType;
                             soil_humidity,
                             canopy_humidity               = CanopyConductanceHumidity(FT),
                             ground_heat_flux              = SoilConductiveFlux(1.5, 0.05),
                             leaf_albedo                   = 0.15,
                             ground_albedo                 = 0.2,
                             maximum_canopy_emissivity     = 0.98,
                             ground_emissivity             = 0.96,
                             extinction                    = 0.5,
                             clumping                      = 1,
                             leaf_boundary_layer_velocity  = 0.02,
                             undercanopy_transfer_velocity = 0.013,
                             wet_soil_resistance           = nothing,
                             ground_surface_resistance     = GroundSurfaceResistance(FT),
                             inner_iterations              = 10)

    # Convert only a scalar optics slot; a `Field` passes through and is materialized
    # per cell in the flux kernel.
    return CanopyInterface(soil_humidity, canopy_humidity, ground_heat_flux,
                               convert_if_number(FT, leaf_albedo),
                               convert_if_number(FT, ground_albedo),
                               convert_if_number(FT, maximum_canopy_emissivity),
                               convert_if_number(FT, ground_emissivity),
                               convert(FT, extinction), convert(FT, clumping),
                               convert(FT, leaf_boundary_layer_velocity),
                               convert_if_number(FT, undercanopy_transfer_velocity),
                               wet_soil_resistance, ground_surface_resistance,
                               inner_iterations)
end

@inline temperature_formulation(state::CanopyInterface) = state
@inline specific_humidity_formulation(state::CanopyInterface) = state

#####
##### Per-cell localization
#####

# Collapse `Field`-valued slots to cell (i, j) before the index-free canopy solve;
# `state2dindex` returns a `Number` slot unchanged.
@inline local_interface_formulation(formulation, i, j) = formulation

@inline local_interface_formulation(F::SoilConductiveFlux, i, j) =
    SoilConductiveFlux(state2dindex(F.conductivity, i, j), state2dindex(F.thickness, i, j))

@inline local_interface_formulation(r::GroundSurfaceResistance, i, j) =
    GroundSurfaceResistance(state2dindex(r.transfer_coefficient, i, j))

@inline local_interface_formulation(c::CanopyInterface, i, j) =
    CanopyInterface(c.soil_humidity, c.canopy_humidity, local_interface_formulation(c.ground_heat_flux, i, j),
                        state2dindex(c.leaf_albedo, i, j),
                        state2dindex(c.ground_albedo, i, j),
                        state2dindex(c.maximum_canopy_emissivity, i, j),
                        state2dindex(c.ground_emissivity, i, j),
                        c.extinction, c.clumping, c.leaf_boundary_layer_velocity,
                        c.undercanopy_transfer_velocity, c.wet_soil_resistance,
                        local_interface_formulation(c.ground_surface_resistance, i, j),
                        c.inner_iterations)

Base.summary(::CanopyInterface) = "CanopyInterface"
Base.show(io::IO, c::CanopyInterface) =
    print(io, "CanopyInterface(soil_humidity=", summary(c.soil_humidity),
          ", canopy_humidity=", summary(c.canopy_humidity), ")")

Adapt.@adapt_structure CanopyInterface

# Materialization / identity — delegate to the sub-models so the per-cell interface
# state carries the soil saturation, bulk temperature, and λˡᵉᵃᶠ the branches read.
@inline interface_phase(c::CanopyInterface) = interface_phase(c.soil_humidity)
@inline ground_conductance(c::CanopyInterface) = ground_conductance(c.ground_heat_flux)
# The soil branch reads the saturation 𝒮, the canopy branch its stress state, and the
# canopy water store Mᶜⁿᵖʸ with its capacity gives the wet fraction fʷᵉᵗ.
land_state_accessors(c::CanopyInterface) =
    merge(land_state_accessors(c.soil_humidity),
          land_state_accessors(c.canopy_humidity),
          (canopy_water_storage = land_canopy_water_storage, canopy_water_capacity = land_canopy_water_capacity))
@inline canopy_leaf_area_index(c::CanopyInterface) = canopy_leaf_area_index(c.canopy_humidity)
@inline interface_vegetation_state(i, j, grid, c::CanopyInterface, vegetation, time_interpolator) =
    interface_vegetation_state(i, j, grid, c.canopy_humidity, vegetation, time_interpolator)

# Wetted fraction of the canopy, fʷᵉᵗ = (Mᶜⁿᵖʸ/Mᶜⁿᵖʸ⁺)^(2/3) (Deardorff 1978).
@inline function wetted_fraction(Mᶜⁿᵖʸ, Mᶜⁿᵖʸ⁺)
    FT = typeof(Mᶜⁿᵖʸ)
    return ifelse(Mᶜⁿᵖʸ⁺ > 0, min(Mᶜⁿᵖʸ / Mᶜⁿᵖʸ⁺, one(FT))^convert(FT, 2//3), zero(FT))
end

# Leaf vapor conductance `gˡᵉᵃᶠᵛ` and its wet-canopy part `gʷᵉᵗ`: stomata in series with
# the leaf boundary layer on the dry fraction 1 − fʷᵉᵗ, the leaf boundary layer alone on the
# wet fraction fʷᵉᵗ. The wet path is capped so the store cannot evaporate more than it holds
# over the step, Jʷᵉᵗ = gʷᵉᵗ Δq ≤ Mᶜⁿᵖʸ/Δt. λˡᵉᵃᶠ enters through `gᶜⁿᵖʸ` and `gˡᵇˡ`, which
# both scale with it, so a leafless canopy has no leaf path.
@inline function leaf_vapor_conductance(gᶜⁿᵖʸ, gˡᵇˡ, fʷᵉᵗ, Mᶜⁿᵖʸ, Δq, Δt)
    gᵈʳʸ = ifelse(gᶜⁿᵖʸ + gˡᵇˡ > 0, gᶜⁿᵖʸ * gˡᵇˡ / (gᶜⁿᵖʸ + gˡᵇˡ), zero(gᶜⁿᵖʸ))
    gʷᵉᵗ = fʷᵉᵗ * gˡᵇˡ
    gʷᵉᵗ = ifelse((Δt > 0) & (Δq > 0), min(gʷᵉᵗ, Mᶜⁿᵖʸ / (Δt * Δq)), gʷᵉᵗ)
    return (1 - fʷᵉᵗ) * gᵈʳʸ + gʷᵉᵗ, gʷᵉᵗ
end

# Leaf vapor conductance `gˡᵉᵃᶠᵛ`, its wet-canopy part `gʷᵉᵗ`, and the leaf-saturated
# humidity `qˡᵉᵃᶠ` at the leaf temperature; the stomata and the wet cap see the node
# humidity `qᵃᶜᵃⁿ`.
@inline function leaf_vapor_terms(c, Tˡᵉᵃᶠ, qᵃᶜᵃⁿ, gˡᵇˡ, Δt, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, transmittance)
    FT = typeof(gˡᵇˡ)
    gᶜⁿᵖʸ, qˡᵉᵃᶠ = canopy_conductance_terms(c.canopy_humidity, Tˡᵉᵃᶠ, qᵃᶜᵃⁿ, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, transmittance)
    Mᶜⁿᵖʸ  = convert(FT, Ψˡᵃ.canopy_water_storage)
    Mᶜⁿᵖʸ⁺ = convert(FT, Ψˡᵃ.canopy_water_capacity)
    fʷᵉᵗ  = wetted_fraction(Mᶜⁿᵖʸ, Mᶜⁿᵖʸ⁺)
    gˡᵉᵃᶠᵛ, gʷᵉᵗ = leaf_vapor_conductance(gᶜⁿᵖʸ, gˡᵇˡ, fʷᵉᵗ, Mᶜⁿᵖʸ, qˡᵉᵃᶠ - qᵃᶜᵃⁿ, Δt)
    return gˡᵉᵃᶠᵛ, gʷᵉᵗ, qˡᵉᵃᶠ
end

# Ground vapor conductance `gᵍʳⁿᵈᵛ`, source humidity `qᵍʳⁿᵈ`, and the source's slope `dqᵍʳⁿᵈ` in the
# ground temperature `Tᵍʳⁿᵈ`: the dry-layer branch (front humidity qᵉ through Gᵉ in series with the
# ground-surface + undercanopy path `gᵈᵇʳ`) blended with the saturated-skin branch (qᵍʳⁿᵈ⁺ through gᵍʳⁿᵈᵛ⁺)
# by the soil model's weight `fᵈᵇʳ`.
@inline function soil_vapor_terms(soil, Tᵍʳⁿᵈ, gᵍʳⁿᵈᵛ⁺, gᵈᵇʳ, Ψⁱⁿ, Ψₐ, Ψˡᵃ, ℙₐ)
    Gᵉ, qᵉ, fᵈᵇʳ, qᵍʳⁿᵈ⁺, dqᵉ, dqᵍʳⁿᵈ⁺ = dry_layer_terms(soil, Tᵍʳⁿᵈ, Ψⁱⁿ, Ψₐ, Ψˡᵃ, ℙₐ)
    Gᵉ = ifelse(gᵈᵇʳ + Gᵉ > 0, Gᵉ * gᵈᵇʳ / (gᵈᵇʳ + Gᵉ), zero(Gᵉ))
    gᵍʳⁿᵈᵛ = fᵈᵇʳ * Gᵉ + (1 - fᵈᵇʳ) * gᵍʳⁿᵈᵛ⁺
    blended = gᵍʳⁿᵈᵛ > eps(eltype(Ψⁱⁿ))
    qᵍʳⁿᵈ  = ifelse(blended, (fᵈᵇʳ * Gᵉ * qᵉ  + (1 - fᵈᵇʳ) * gᵍʳⁿᵈᵛ⁺ * qᵍʳⁿᵈ⁺)  / gᵍʳⁿᵈᵛ, qᵍʳⁿᵈ⁺)
    dqᵍʳⁿᵈ = ifelse(blended, (fᵈᵇʳ * Gᵉ * dqᵉ + (1 - fᵈᵇʳ) * gᵍʳⁿᵈᵛ⁺ * dqᵍʳⁿᵈ⁺) / gᵍʳⁿᵈᵛ, dqᵍʳⁿᵈ⁺)
    return gᵍʳⁿᵈᵛ, qᵍʳⁿᵈ, dqᵍʳⁿᵈ
end

# Fraction of the start-of-step node value surviving in the step mean of the node
# balance C dx/dt = Σg (x^{eq} − x): m = (τ/Δt)(1 − e^{−Δt/τ}) with τ = C/Σg. A massless
# node (C = 0) or Δt = 0 gives 0, the diagnostic equilibrium.
@inline function node_memory(Σg, C, Δt)
    FT = typeof(Σg)
    τ  = C / max(Σg, eps(FT))
    m  = -(τ / max(Δt, eps(FT))) * expm1(-Δt / τ)
    return ifelse((Δt > 0) & (C > 0), m, zero(FT))
end

# Canopy-air node value: the conductance-weighted mean of its sources, blended with the
# start-of-step value `x⁻` by the memory `m`; also returns that equilibrium mean.
@inline function canopy_air_node(x⁻, conductances, sources, m)
    equilibrium = conductance_weighted_node(x⁻, conductances, sources)
    return equilibrium + (x⁻ - equilibrium) * m, equilibrium
end

# Longwave reaching the ground under the canopy, and leaving it upward.
@inline function ground_longwave(downwelling_longwave, Tˡᵉᵃᶠ, Tᵍʳⁿᵈ, ϵˡᵉᵃᶠ, ϵᵍʳⁿᵈ, σ)
    ground_downwelling_longwave = (1 - ϵˡᵉᵃᶠ) * downwelling_longwave + ϵˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^4
    ℐꜛˡʷᵍʳⁿᵈ = ϵᵍʳⁿᵈ * σ * Tᵍʳⁿᵈ^4 + (1 - ϵᵍʳⁿᵈ) * ground_downwelling_longwave
    return ground_downwelling_longwave, ℐꜛˡʷᵍʳⁿᵈ
end

"""
$(TYPEDSIGNATURES)

Advance a prognostic canopy-air node value `x` toward its conductance-weighted
`equilibrium` ``x^{eq}`` over `Δt`, the exact solution of ``C \\, dx/dt = Σg (x^{eq} - x)``
at fixed conductances: ``x ← x^{eq} + (x - x^{eq}) e^{-Δt Σg / C}``. `Δt = 0` or
`C = 0` return the equilibrium.
"""
@inline function advance_canopy_air(x, equilibrium, Σg, C, Δt)
    FT = typeof(x)
    decay = ifelse((Δt > 0) & (C > 0), exp(-Δt * Σg / max(C, eps(FT))), zero(FT))
    return equilibrium + (x - equilibrium) * decay
end

@inline canopy_air_depth(::Nothing) = 0
@inline canopy_air_depth(storage) = storage.layer_depth

"""
$(TYPEDSIGNATURES)

Solve the leaf and ground energy balances by quasi-Newton iteration, using the analytical
step-mean canopy-air temperature and humidity over `Δt` at fixed effective conductances
and sources. `Δt = 0` computes diagnostic equilibrium.
Without canopy-air storage, the node balances its heat and vapor fluxes instantaneously.
The leaf-conductance derivative is evaluated by a one-sided difference.
"""
@inline function canopy_air_space_solve(c::CanopyInterface, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, Δt)
    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    FT  = eltype(Ψⁱⁿ)
    pᵃᵗ = Ψₐ.p
    qᵃᵗ = Ψₐ.q
    Tᵃᵗ = Ψₐ.T
    ℒ   = AtmosphericThermodynamics.latent_heat_vapor(ℂᵃᵗ, Tᵃᵗ)
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)
    cᵖᵐ = AtmosphericThermodynamics.cp_m(ℂᵃᵗ, qᵃᵗ)
    θᵃᵗ = surface_atmosphere_temperature(Ψₐ, ℙₐ)
    Tˡᵃ = Ψˡᵃ.T
    λˡᵉᵃᶠ = Ψˡᵃ.leaf_area_index
    u★  = Ψⁱⁿ.fluxes.u★

    gᵃᵀ = aerodynamic_heat_conductance(Ψⁱⁿ, Ψₐ, ℂᵃᵗ)
    gᵃᵛ = aerodynamic_vapor_conductance(Ψⁱⁿ, Ψₐ, ℂᵃᵗ)

    # Land surface velocities are zero, so the surface wind speed is the atmospheric one.
    ΔU    = sqrt(Ψₐ.u^2 + Ψₐ.v^2)
    wᵘᶜᵃⁿ = undercanopy_transfer_velocity(c.undercanopy_transfer_velocity, λˡᵉᵃᶠ, ΔU, u★)
    gˡᵉᵃᶠᵀ = ρᵃᵗ * cᵖᵐ * λˡᵉᵃᶠ * c.leaf_boundary_layer_velocity
    gˡᵇˡ  = ρᵃᵗ * λˡᵉᵃᶠ * c.leaf_boundary_layer_velocity
    gᵍʳⁿᵈᵀ = ρᵃᵗ * cᵖᵐ * wᵘᶜᵃⁿ
    # Ground vapor path: the ground-surface resistance `rᵍʳⁿᵈ` and, on the moist-soil branch, the soil
    # surface resistance `rˢᵒⁱ` in series with the undercanopy path (Sakaguchi & Zeng 2009).
    rᵍʳⁿᵈ   = ground_surface_resistance(c.ground_surface_resistance, u★)
    rˢᵒⁱ   = soil_surface_resistance(c.wet_soil_resistance, Ψˡᵃ.𝒮)
    gᵈᵇʳ   = ρᵃᵗ * wᵘᶜᵃⁿ / (1 + wᵘᶜᵃⁿ * rᵍʳⁿᵈ)
    gᵍʳⁿᵈᵛ⁺ = ρᵃᵗ * wᵘᶜᵃⁿ / (1 + wᵘᶜᵃⁿ * (rᵍʳⁿᵈ + rˢᵒⁱ))
    Λᵍʳⁿᵈ   = convert(FT, ground_conductance(c.ground_heat_flux))

    # Canopy-air heat and vapor capacities (zero for a massless node) and the
    # start-of-step node.
    hᵃᶜᵃⁿ = convert(FT, canopy_air_depth(Ψˡᵃ.surface_storage))
    Cᵀ    = ρᵃᵗ * cᵖᵐ * hᵃᶜᵃⁿ
    Cᵛ    = ρᵃᵗ * hᵃᶜᵃⁿ
    T⁻    = Ψⁱⁿ.temperature
    q⁻    = Ψⁱⁿ.specific_humidity

    # Shortwave split and longwave emissivities (broadband), on downwelling magnitudes.
    σ = Ψᵣ.σ
    downwelling_shortwave = - Ψᵣ.ℐꜜˢʷ
    downwelling_longwave  = - Ψᵣ.ℐꜜˡʷ
    αˡᵉᵃᶠ = convert(FT, c.leaf_albedo)
    αᵍʳⁿᵈ = convert(FT, c.ground_albedo)
    ϵˡᵉᵃᶠ = convert(FT, c.maximum_canopy_emissivity) * (1 - exp(-λˡᵉᵃᶠ))
    ϵᵍʳⁿᵈ = convert(FT, c.ground_emissivity)
    transmittance = canopy_transmittance(c.extinction, c.clumping, λˡᵉᵃᶠ)
    leaf_absorbed_shortwave   = (1 - αˡᵉᵃᶠ) * (1 - transmittance) * downwelling_shortwave
    ground_absorbed_shortwave = transmittance * (1 - αᵍʳⁿᵈ) * downwelling_shortwave
    effective_albedo = (1 - transmittance) * αˡᵉᵃᶠ + transmittance * αᵍʳⁿᵈ

    # Thermal node: its conductances are fixed through the temperature iteration. The node is the
    # equilibrium mean of its sources blended with the start-of-step node by the memory `m`,
    # so its sensitivity to source `i` is (1 − m) gᵢ/Σg.
    Σgᵀ = gᵃᵀ + gˡᵉᵃᶠᵀ + gᵍʳⁿᵈᵀ
    mᵀ  = node_memory(Σgᵀ, Cᵀ, Δt)
    ∂Tᵃᶜᵃⁿ∂Tˡᵉᵃᶠ = (1 - mᵀ) * gˡᵉᵃᶠᵀ / max(Σgᵀ, eps(FT))
    ∂Tᵃᶜᵃⁿ∂Tᵍʳⁿᵈ = (1 - mᵀ) * gᵍʳⁿᵈᵀ / max(Σgᵀ, eps(FT))

    Tˡᵉᵃᶠ  = Tˡᵃ
    Tᵍʳⁿᵈ  = Tˡᵃ
    Tᵃᶜᵃⁿ = T⁻
    qᵃᶜᵃⁿ = q⁻
    maximum_temperature_step = convert(FT, 25)

    for _ in 1:c.inner_iterations
        gˡᵉᵃᶠᵛ, gʷᵉᵗ, qˡᵉᵃᶠ = leaf_vapor_terms(c, Tˡᵉᵃᶠ, qᵃᶜᵃⁿ, gˡᵇˡ, Δt, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, transmittance)
        # The leaf vapor conductance depends on the leaf temperature through the stomata and the
        # wet-canopy cap; its slope is a one-sided difference at the same node humidity.
        δT = sqrt(eps(FT)) * Tˡᵉᵃᶠ
        gˡᵉᵃᶠᵛ⁺, _, _ = leaf_vapor_terms(c, Tˡᵉᵃᶠ + δT, qᵃᶜᵃⁿ, gˡᵇˡ, Δt, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, transmittance)
        ∂gˡᵉᵃᶠᵛ∂Tˡᵉᵃᶠ  = (gˡᵉᵃᶠᵛ⁺ - gˡᵉᵃᶠᵛ) / δT
        gᵍʳⁿᵈᵛ, qᵍʳⁿᵈ, dqᵍʳⁿᵈ = soil_vapor_terms(c.soil_humidity, Tᵍʳⁿᵈ, gᵍʳⁿᵈᵛ⁺, gᵈᵇʳ, Ψⁱⁿ, Ψₐ, Ψˡᵃ, ℙₐ)

        Σgᵛ = gᵃᵛ + gˡᵉᵃᶠᵛ + gᵍʳⁿᵈᵛ
        mᵛ  = node_memory(Σgᵛ, Cᵛ, Δt)
        Tᵃᶜᵃⁿ, _ = canopy_air_node(T⁻, (gᵍʳⁿᵈᵀ, gˡᵉᵃᶠᵀ, gᵃᵀ), (Tᵍʳⁿᵈ, Tˡᵉᵃᶠ, θᵃᵗ), mᵀ)
        qᵃᶜᵃⁿ, equilibrium_humidity = canopy_air_node(q⁻, (gᵍʳⁿᵈᵛ, gˡᵉᵃᶠᵛ, gᵃᵛ), (qᵍʳⁿᵈ, qˡᵉᵃᶠ, qᵃᵗ), mᵛ)
        ∂qᵃᶜᵃⁿ∂qˡᵉᵃᶠ = (1 - mᵛ) * gˡᵉᵃᶠᵛ / max(Σgᵛ, eps(FT))
        ∂qᵃᶜᵃⁿ∂qᵍʳⁿᵈ = (1 - mᵛ) * gᵍʳⁿᵈᵛ / max(Σgᵛ, eps(FT))
        ∂qᵃᶜᵃⁿ∂gˡᵉᵃᶠᵛ = (1 - mᵛ) * (qˡᵉᵃᶠ - equilibrium_humidity) / max(Σgᵛ, eps(FT))

        ground_downwelling_longwave, ℐꜛˡʷᵍʳⁿᵈ = ground_longwave(downwelling_longwave, Tˡᵉᵃᶠ, Tᵍʳⁿᵈ, ϵˡᵉᵃᶠ, ϵᵍʳⁿᵈ, σ)
        leaf_net_longwave   = ϵˡᵉᵃᶠ * (downwelling_longwave + ℐꜛˡʷᵍʳⁿᵈ) - 2 * ϵˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^4
        ground_net_longwave = ϵᵍʳⁿᵈ * (ground_downwelling_longwave - σ * Tᵍʳⁿᵈ^4)
        dqˡᵉᵃᶠ = saturation_humidity_slope(ℂᵃᵗ, Tˡᵉᵃᶠ, pᵃᵗ, c.canopy_humidity.phase, qˡᵉᵃᶠ)

        # Leaf and ground energy residuals and their Jacobian, including each balance's
        # dependence on the other surface temperature through the node and the longwave exchange.
        Rˡᵉᵃᶠ = leaf_absorbed_shortwave + leaf_net_longwave - gˡᵉᵃᶠᵀ * (Tˡᵉᵃᶠ - Tᵃᶜᵃⁿ) - ℒ * gˡᵉᵃᶠᵛ * (qˡᵉᵃᶠ - qᵃᶜᵃⁿ)
        Rᵍʳⁿᵈ = ground_absorbed_shortwave + ground_net_longwave - gᵍʳⁿᵈᵀ * (Tᵍʳⁿᵈ - Tᵃᶜᵃⁿ) - ℒ * gᵍʳⁿᵈᵛ * (qᵍʳⁿᵈ - qᵃᶜᵃⁿ) - Λᵍʳⁿᵈ * (Tᵍʳⁿᵈ - Tˡᵃ)
        ∂Rˡᵉᵃᶠ∂Tˡᵉᵃᶠ = 4 * ϵˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^3 * (ϵˡᵉᵃᶠ * (1 - ϵᵍʳⁿᵈ) - 2) - gˡᵉᵃᶠᵀ * (1 - ∂Tᵃᶜᵃⁿ∂Tˡᵉᵃᶠ) - ℒ * gˡᵉᵃᶠᵛ * (1 - ∂qᵃᶜᵃⁿ∂qˡᵉᵃᶠ) * dqˡᵉᵃᶠ -
                       ℒ * ∂gˡᵉᵃᶠᵛ∂Tˡᵉᵃᶠ * (qˡᵉᵃᶠ - qᵃᶜᵃⁿ - gˡᵉᵃᶠᵛ * ∂qᵃᶜᵃⁿ∂gˡᵉᵃᶠᵛ)
        ∂Rˡᵉᵃᶠ∂Tᵍʳⁿᵈ = 4 * ϵˡᵉᵃᶠ * ϵᵍʳⁿᵈ * σ * Tᵍʳⁿᵈ^3 + gˡᵉᵃᶠᵀ * ∂Tᵃᶜᵃⁿ∂Tᵍʳⁿᵈ + ℒ * gˡᵉᵃᶠᵛ * ∂qᵃᶜᵃⁿ∂qᵍʳⁿᵈ * dqᵍʳⁿᵈ
        ∂Rᵍʳⁿᵈ∂Tˡᵉᵃᶠ = 4 * ϵᵍʳⁿᵈ * ϵˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^3 + gᵍʳⁿᵈᵀ * ∂Tᵃᶜᵃⁿ∂Tˡᵉᵃᶠ + ℒ * gᵍʳⁿᵈᵛ * ∂qᵃᶜᵃⁿ∂qˡᵉᵃᶠ * dqˡᵉᵃᶠ +
                    ℒ * gᵍʳⁿᵈᵛ * ∂qᵃᶜᵃⁿ∂gˡᵉᵃᶠᵛ * ∂gˡᵉᵃᶠᵛ∂Tˡᵉᵃᶠ
        ∂Rᵍʳⁿᵈ∂Tᵍʳⁿᵈ = -4 * ϵᵍʳⁿᵈ * σ * Tᵍʳⁿᵈ^3 - gᵍʳⁿᵈᵀ * (1 - ∂Tᵃᶜᵃⁿ∂Tᵍʳⁿᵈ) - ℒ * gᵍʳⁿᵈᵛ * (1 - ∂qᵃᶜᵃⁿ∂qᵍʳⁿᵈ) * dqᵍʳⁿᵈ - Λᵍʳⁿᵈ
        # A canopy-free cell (λˡᵉᵃᶠ = 0) has no leaf balance: Rˡᵉᵃᶠ = ∂Rˡᵉᵃᶠ/∂Tˡᵉᵃᶠ = 0.
        ∂Rˡᵉᵃᶠ∂Tˡᵉᵃᶠ = ifelse(∂Rˡᵉᵃᶠ∂Tˡᵉᵃᶠ < 0, ∂Rˡᵉᵃᶠ∂Tˡᵉᵃᶠ, -one(FT))
        determinant = ∂Rˡᵉᵃᶠ∂Tˡᵉᵃᶠ * ∂Rᵍʳⁿᵈ∂Tᵍʳⁿᵈ - ∂Rˡᵉᵃᶠ∂Tᵍʳⁿᵈ * ∂Rᵍʳⁿᵈ∂Tˡᵉᵃᶠ
        ΔTˡᵉᵃᶠ = (∂Rᵍʳⁿᵈ∂Tᵍʳⁿᵈ * Rˡᵉᵃᶠ - ∂Rˡᵉᵃᶠ∂Tᵍʳⁿᵈ * Rᵍʳⁿᵈ) / determinant
        ΔTᵍʳⁿᵈ = (∂Rˡᵉᵃᶠ∂Tˡᵉᵃᶠ * Rᵍʳⁿᵈ - ∂Rᵍʳⁿᵈ∂Tˡᵉᵃᶠ * Rˡᵉᵃᶠ) / determinant
        Tˡᵉᵃᶠ -= clamp(ΔTˡᵉᵃᶠ, -maximum_temperature_step, maximum_temperature_step)
        Tᵍʳⁿᵈ    -= clamp(ΔTᵍʳⁿᵈ, -maximum_temperature_step, maximum_temperature_step)
    end

    # The node and the flux partition at the final leaf and ground temperatures.
    gˡᵉᵃᶠᵛ, gʷᵉᵗ, qˡᵉᵃᶠ = leaf_vapor_terms(c, Tˡᵉᵃᶠ, qᵃᶜᵃⁿ, gˡᵇˡ, Δt, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, transmittance)
    gᵍʳⁿᵈᵛ, qᵍʳⁿᵈ, _ = soil_vapor_terms(c.soil_humidity, Tᵍʳⁿᵈ, gᵍʳⁿᵈᵛ⁺, gᵈᵇʳ, Ψⁱⁿ, Ψₐ, Ψˡᵃ, ℙₐ)
    Σgᵛ = gᵃᵛ + gˡᵉᵃᶠᵛ + gᵍʳⁿᵈᵛ
    Tᵃᶜᵃⁿ, equilibrium_temperature = canopy_air_node(T⁻, (gᵍʳⁿᵈᵀ, gˡᵉᵃᶠᵀ, gᵃᵀ), (Tᵍʳⁿᵈ, Tˡᵉᵃᶠ, θᵃᵗ), mᵀ)
    qᵃᶜᵃⁿ, equilibrium_humidity    = canopy_air_node(q⁻, (gᵍʳⁿᵈᵛ, gˡᵉᵃᶠᵛ, gᵃᵛ), (qᵍʳⁿᵈ, qˡᵉᵃᶠ, qᵃᵗ), node_memory(Σgᵛ, Cᵛ, Δt))

    # Effective radiating temperature: σ T⁴ = total upwelling longwave (emission + reflection).
    _, ℐꜛˡʷᵍʳⁿᵈ = ground_longwave(downwelling_longwave, Tˡᵉᵃᶠ, Tᵍʳⁿᵈ, ϵˡᵉᵃᶠ, ϵᵍʳⁿᵈ, σ)
    ℐꜛˡʷ = (1 - ϵˡᵉᵃᶠ) * ℐꜛˡʷᵍʳⁿᵈ + ϵˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^4
    effective_temperature = ifelse(σ > 0, (ℐꜛˡʷ / σ)^convert(FT, 1//4), Tᵃᶜᵃⁿ)

    𝒬ᵀˡᵉᵃᶠ = gˡᵉᵃᶠᵀ * (Tˡᵉᵃᶠ - Tᵃᶜᵃⁿ)
    𝒬ᵀᵍʳⁿᵈ = gᵍʳⁿᵈᵀ * (Tᵍʳⁿᵈ - Tᵃᶜᵃⁿ)
    𝒬ᵛˡᵉᵃᶠ = ℒ * gˡᵉᵃᶠᵛ * (qˡᵉᵃᶠ - qᵃᶜᵃⁿ)
    𝒬ᵛᵍʳⁿᵈ = ℒ * gᵍʳⁿᵈᵛ * (qᵍʳⁿᵈ - qᵃᶜᵃⁿ)
    𝒬ᵍʳⁿᵈ  = Λᵍʳⁿᵈ * (Tᵍʳⁿᵈ - Tˡᵃ)
    Jʷᵉᵗ  = gʷᵉᵗ * (qˡᵉᵃᶠ - qᵃᶜᵃⁿ)

    return (; Tˡᵉᵃᶠ = convert(FT, Tˡᵉᵃᶠ), Tᵍʳⁿᵈ = convert(FT, Tᵍʳⁿᵈ),
              Tᵃᶜᵃⁿ = convert(FT, Tᵃᶜᵃⁿ), qᵃᶜᵃⁿ = convert(FT, qᵃᶜᵃⁿ),
              effective_temperature = convert(FT, effective_temperature),
              effective_albedo = convert(FT, effective_albedo),
              𝒬ᵀˡᵉᵃᶠ = convert(FT, 𝒬ᵀˡᵉᵃᶠ), 𝒬ᵀᵍʳⁿᵈ = convert(FT, 𝒬ᵀᵍʳⁿᵈ),
              𝒬ᵛˡᵉᵃᶠ = convert(FT, 𝒬ᵛˡᵉᵃᶠ), 𝒬ᵛᵍʳⁿᵈ = convert(FT, 𝒬ᵛᵍʳⁿᵈ),
              𝒬ᵍʳⁿᵈ = convert(FT, 𝒬ᵍʳⁿᵈ), Jʷᵉᵗ = convert(FT, Jʷᵉᵗ), ℒ = convert(FT, ℒ),
              Σgᵀ = convert(FT, Σgᵀ), Σgᵛ = convert(FT, Σgᵛ),
              equilibrium_temperature = convert(FT, equilibrium_temperature), equilibrium_humidity = convert(FT, equilibrium_humidity),
              Cᵀ = convert(FT, Cᵀ), Cᵛ = convert(FT, Cᵛ))
end

# A diagnostic node is re-solved every iterate; one shared solve returns both node values.
@inline function interface_temperature_and_humidity(c::CanopyInterface,
                                                    Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙⁱⁿ, ℙₐ, ℙᵢ, Δt,
                                                    storage::Nothing = Ψˡᵃ.surface_storage)
    solution = canopy_air_space_solve(c, Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, Δt)
    return solution.Tᵃᶜᵃⁿ, solution.qᵃᶜᵃⁿ
end

# A prognostic node is model state, frozen through the fixed point.
@inline interface_temperature_and_humidity(::CanopyInterface,
                                           Ψⁱⁿ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙⁱⁿ, ℙₐ, ℙᵢ, Δt, storage) =
    (Ψⁱⁿ.temperature, Ψⁱⁿ.specific_humidity)

"""
    struct CanopyInterfaceDiagnostics

What a [`CanopyInterface`](@ref) stores in the interface `temperature` slot: the
canopy-air node temperature the atmosphere sees, the leaf, ground, and effective radiating
temperatures, and the two-source flux partition the land and the radiation read.
"""
struct CanopyInterfaceDiagnostics{F}
    interface              :: F   # canopy-air node Tᵃᶜᵃⁿ (what MOST sees)
    canopy                 :: F   # leaf temperature Tˡᵉᵃᶠ
    ground                 :: F   # ground temperature Tᵍʳⁿᵈ
    effective              :: F   # blackbody-equivalent radiating temperature Tʳᵃᵈ
    effective_albedo       :: F   # broadband shortwave albedo of the canopy + ground column
    ground_heat_flux       :: F   # ground→bulk conduction 𝒬ᵍʳⁿᵈ
    canopy_latent_heat     :: F   # leaf latent heat 𝒬ᵛˡᵉᵃᶠ (transpiration + wet canopy)
    ground_latent_heat     :: F   # soil evaporation 𝒬ᵛᵍʳⁿᵈ
    canopy_sensible_heat   :: F   # leaf sensible 𝒬ᵀˡᵉᵃᶠ
    ground_sensible_heat   :: F   # ground sensible 𝒬ᵀᵍʳⁿᵈ
    canopy_evaporation     :: F   # wet-canopy evaporation Jʷᵉᵗ (kg m⁻² s⁻¹, up)
    wet_canopy_latent_heat :: F   # wet-canopy latent heat ℒ·Jʷᵉᵗ (W m⁻², up)
    land_vapor_flux        :: F   # soil evaporation + transpiration, drawn from the land water store
    transpiration          :: F   # leaf transpiration 𝒬ᵛˡᵉᵃᶠ/ℒ − Jʷᵉᵗ, drawn from the soil
end

CanopyInterfaceDiagnostics(grid) =
    CanopyInterfaceDiagnostics(ntuple(_ -> Field{Center, Center, Nothing}(grid),
                                     Val(fieldcount(CanopyInterfaceDiagnostics)))...)

Adapt.@adapt_structure CanopyInterfaceDiagnostics

Base.summary(::CanopyInterfaceDiagnostics) = "CanopyInterfaceDiagnostics"
Base.show(io::IO, d::CanopyInterfaceDiagnostics) = print(io, summary(d))
