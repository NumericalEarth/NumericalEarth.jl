#####
##### `CanopyGroundSurface` — a two-source canopy (leaf + shaded ground) exchanging with a
##### canopy-air node that drains to the atmosphere through the aerodynamic conductance.
#####

#####
##### Undercanopy transfer velocities — the ground ↔ canopy-air sensible/vapor coupling.
#####

"""
    AreaIndexUndercanopyVelocity(FT = Oceananigans.defaults.FloatType;
                                    drag_coefficient = 0.006,
                                    stem_area_index = 0)

Ground↔canopy-air transfer velocity that responds to canopy density and wind
(the PALADYN form; [Willeit and Ganopolski (2016)](@cite willeit2016)):

```math
wᵘᶜᵃⁿ = \\frac{Cᴰ \\, ΔU}{\\max\\!\\left(1 - e^{-(LAI + SAI)}, ε\\right)},
```

with `drag_coefficient` `Cᴰ`, the surface wind speed `ΔU`, and the canopy shielding
`1 − e^{−(LAI+SAI)}` (`stem_area_index` `SAI` counts the leafless woody area). A denser
canopy shields the ground more strongly and decouples it from the canopy air; a sparse
canopy (`LAI → 0`) leaves the ground ventilating at the aerodynamic limit, so the result
is capped at the aerodynamic transfer velocity `u★²/ΔU` — the ground cannot ventilate to
the canopy air faster than the canopy air ventilates to the atmosphere.

```jldoctest
using NumericalEarth

AreaIndexUndercanopyVelocity()

# output
AreaIndexUndercanopyVelocity(Cᴰ=0.006, SAI=0.0)
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
    FrictionVelocityUndercanopyVelocity(FT = Oceananigans.defaults.FloatType;
                                           dense_canopy_coefficient = 0.004,
                                           ground_roughness_length = 0.01,
                                           stem_area_index = 0,
                                           kinematic_viscosity = 1.5e-5)

Ground↔canopy-air transfer velocity of CLM5, after
[Zeng et al. (2005)](@cite zeng2005undercanopy): the velocity scale is the friction
velocity `u★` (canopy-top shear drives the undercanopy turbulence), and the transfer
coefficient interpolates between a dense-canopy constant and a bare-ground
roughness-Reynolds-number law,

```math
wᵘᶜᵃⁿ = Cₛ u★, \\qquad Cₛ = Cₛᵇ W + Cₛᵈ (1 - W), \\qquad W = e^{-(LAI + SAI)},
\\qquad Cₛᵇ = \\frac{ϰ}{0.13} \\left(\\frac{ℓᵍʳᵒ u★}{ν}\\right)^{-0.45},
```

with the dense-canopy coefficient `Cₛᵈ` (`dense_canopy_coefficient`, 0.004 after
Dickinson et al. 1993), the ground roughness length `ℓᵍʳᵒ` (`ground_roughness_length`, m),
the kinematic viscosity of air `ν` (`kinematic_viscosity`, m² s⁻¹), and the von Kármán
constant `ϰ = 0.4` (CLM5 Tech Note Eqs. 2.5.116–2.5.121). Unlike
[`AreaIndexUndercanopyVelocity`](@ref), the sparse-canopy limit is closed physically
by the bare-ground law rather than floored, and calm air (`u★ → 0`) decouples the ground.

```jldoctest
using NumericalEarth

FrictionVelocityUndercanopyVelocity()

# output
FrictionVelocityUndercanopyVelocity(Cₛᵈ=0.004, ℓᵍʳᵒ=0.01, SAI=0.0)
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
           ", SAI=", prettysummary(u.stem_area_index), ")")
Base.show(io::IO, u::AreaIndexUndercanopyVelocity) = print(io, summary(u))

Base.summary(u::FrictionVelocityUndercanopyVelocity) =
    string("FrictionVelocityUndercanopyVelocity(Cₛᵈ=", prettysummary(u.dense_canopy_coefficient),
           ", ℓᵍʳᵒ=", prettysummary(u.ground_roughness_length),
           ", SAI=", prettysummary(u.stem_area_index), ")")
Base.show(io::IO, u::FrictionVelocityUndercanopyVelocity) = print(io, summary(u))

# A `Number` is a constant transfer velocity (m s⁻¹).
@inline undercanopy_transfer_velocity(wᵘᶜᵃⁿ::Number, LAI, ΔU, u★) = convert(typeof(LAI), wᵘᶜᵃⁿ)

@inline function undercanopy_transfer_velocity(u::AreaIndexUndercanopyVelocity, LAI, ΔU, u★)
    FT = typeof(LAI)
    Cᴰ = convert(FT, u.drag_coefficient)
    area_index = LAI + convert(FT, u.stem_area_index)
    wᵘᶜᵃⁿ = Cᴰ * ΔU / max(1 - exp(-area_index), eps(FT))
    aerodynamic_limit = u★^2 / max(ΔU, eps(FT))
    return min(wᵘᶜᵃⁿ, aerodynamic_limit)
end

@inline function undercanopy_transfer_velocity(u::FrictionVelocityUndercanopyVelocity, LAI, ΔU, u★)
    FT   = typeof(LAI)
    W    = exp(-(LAI + convert(FT, u.stem_area_index)))
    Cᵈ   = convert(FT, u.dense_canopy_coefficient)
    ℓᵍʳᵒ = convert(FT, u.ground_roughness_length)
    ν    = convert(FT, u.kinematic_viscosity)
    ϰ    = convert(FT, 2//5)
    a    = convert(FT, 13//100)
    n    = convert(FT, 9//20)
    # Cₛᵇ u★ grouped as u★^(1−n) so u★ → 0 gives 0 rather than Inf·0.
    wᵇ = (ϰ / a) * (ℓᵍʳᵒ / ν)^(-n) * u★^(1 - n)
    return W * wᵇ + (1 - W) * Cᵈ * u★
end

"""
    SellersSoilResistance(FT = Oceananigans.defaults.FloatType;
                          intercept = 8.206, slope = 4.255)

Empirical ground-surface resistance for the moist-soil (vanishing dry layer) evaporation
branch of a [`CanopyGroundSurface`](@ref),

```math
rˢᵒⁱ = e^{a - b 𝒮},
```

with the surface saturation `𝒮`, fit by [Sellers et al. (1992)](@cite sellers1992) to the
FIFE prairie flux stations (`rˢᵒⁱ(1) ≈ 52` s m⁻¹). The FIFE sites carried litter, so the fit
is an effective soil-plus-litter resistance, an alternative to a
[`GroundSurfaceResistance`](@ref) rather than a companion to one.

```jldoctest
using NumericalEarth

SellersSoilResistance()

# output
SellersSoilResistance(a=8.206, b=4.255)
```
"""
struct SellersSoilResistance{FT}
    intercept :: FT
    slope     :: FT
end

SellersSoilResistance(FT::Type = Oceananigans.defaults.FloatType;
                      intercept = 8.206, slope = 4.255) =
    SellersSoilResistance(convert(FT, intercept), convert(FT, slope))

Base.summary(r::SellersSoilResistance) =
    string("SellersSoilResistance(a=", prettysummary(r.intercept),
           ", b=", prettysummary(r.slope), ")")
Base.show(io::IO, r::SellersSoilResistance) = print(io, summary(r))

@inline soil_surface_resistance(::Nothing, 𝒮) = zero(𝒮)
@inline function soil_surface_resistance(r::SellersSoilResistance, 𝒮)
    FT = typeof(𝒮)
    a  = convert(FT, r.intercept)
    b  = convert(FT, r.slope)
    return exp(a - b * clamp(𝒮, zero(FT), one(FT)))
end

"""
    GroundSurfaceResistance(FT = Oceananigans.defaults.FloatType; transfer_coefficient = 0.00633)

Bulk resistance of the ground surface to evaporation, in series with the undercanopy path
on both ground vapor branches of a [`CanopyGroundSurface`](@ref),

```math
rᵍʳᵒ = \\frac{1}{Cᵍʳᵒ u★},
```

with a bulk transfer coefficient `Cᵍʳᵒ` and the friction velocity `u★` standing in for the
wind speed near the ground. It lumps whatever covers the soil (plant litter, mulch, a
crust) into one parameter. The default `Cᵍʳᵒ = 0.00633` reproduces the litter resistance of
[Sakaguchi and Zeng (2009)](@cite sakaguchi2009), Eq. 13, at unit litter area index,
`Cᵍʳᵒ = 0.004 / (1 - e^{-1})`. A `CanopyGroundSurface` also takes a `Number`, a constant
resistance in s m⁻¹.

```jldoctest
using NumericalEarth

GroundSurfaceResistance()

# output
GroundSurfaceResistance(C=0.00633)
```
"""
struct GroundSurfaceResistance{FT}
    transfer_coefficient :: FT
end

GroundSurfaceResistance(FT::Type = Oceananigans.defaults.FloatType; transfer_coefficient = 0.00633) =
    GroundSurfaceResistance(convert(FT, transfer_coefficient))

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
    struct CanopyGroundSurface

Two-source canopy + ground surface with a canopy-air node, for forests: tall vegetation
whose crowns enclose a canopy air space with the ground as a second source beneath it.
Low vegetation (grass, crops) has no separable canopy air layer and belongs to a
single-source [`CanopyConductanceHumidity`](@ref) over a soil skin. Solves the
leaf temperature `Tˡᵉᵃ`, the ground surface temperature `Tᵍʳᵒ`, and the canopy-air node
`(Tᵃᶜᵃⁿ, qᵃᶜᵃⁿ)` inside the Monin–Obukhov fixed point (diagnostic node, the default),
or couples to the land component's prognostic canopy-air reservoir. Pass it as the interface state
formulation to solve temperature and specific humidity together. Over an
[`InterceptingHydrology`](@ref) the wetted fraction `fʷᵉᵗ = (Mᶜᵃⁿ/Mᶜᵃⁿ⁺)^{2/3}` of the leaf
([Deardorff, 1978](@cite deardorff1978)) evaporates intercepted water at the potential
rate through the leaf boundary layer alone; a dry canopy (`Mᶜᵃⁿ = 0`) has no wet branch.

Fields:
- `soil_humidity`   : the soil vapor branch (a [`DryLayerHumidity`](@ref)).
- `canopy_humidity` : the leaf vapor branch (a [`CanopyConductanceHumidity`](@ref)).
- `ground_heat_flux` : ground↔bulk conduction `Λᵍʳᵒ = κᵀ/ℓᵀ` (a [`SoilConductiveFlux`](@ref)).
- `leaf_albedo`, `ground_albedo` : broadband shortwave albedos. A `Number`, or a
  `Field{Center, Center, Nothing}` of per-cell values (see [`atmosphere_land_interface`](@ref)).
- `maximum_canopy_emissivity`, `ground_emissivity` : longwave emissivities (`ϵˡᵉᵃ = ϵᵐᵃˣ(1 − e^{−LAI})`),
  each a `Number` or a per-cell `Field`.
- `extinction`, `clumping` : Beer–Lambert extinction `K` and foliage clumping index `Ω` of the
  shortwave transmittance `e^{−KΩL}` (`Ω = 1` random foliage, `Ω < 1` foliage gathered into
  crowns that leave gaps).
- `leaf_boundary_layer_velocity` : constant per-leaf transfer velocity through the leaf
  boundary layer `wˡᵇˡ` (m s⁻¹) → sensible `gˡᵉᵃᵀ = ρcᵖᵐ·LAI·wˡᵇˡ`, vapor `gˡᵇˡ = ρ·LAI·wˡᵇˡ` (in
  series with the stomata when dry, alone over the wetted fraction).
- `undercanopy_transfer_velocity` : ground↔canopy-air transfer velocity `wᵘᶜᵃⁿ` → `gᵍʳᵒᵀ = ρcᵖᵐ·wᵘᶜᵃⁿ`;
  a constant `Number` (m s⁻¹), an [`AreaIndexUndercanopyVelocity`](@ref), or a
  [`FrictionVelocityUndercanopyVelocity`](@ref).
- `wet_soil_resistance` : soil surface resistance on the moist-soil (vanishing dry layer)
  vapor branch (a [`SellersSoilResistance`](@ref)), or `nothing` (the default: above the
  dry-layer onset the soil itself does not limit evaporation, and the ground-surface
  resistance and undercanopy path carry it).
- `ground_surface_resistance` : bulk ground-surface resistance in series on both ground
  vapor branches (a [`GroundSurfaceResistance`](@ref), the default, or a constant `Number`
  in s m⁻¹), or `nothing` for none.
- `inner_iterations` : quasi-Newton iterations of the coupled skin solve.

The four optics slots accept a per-cell `Field{Center, Center, Nothing}` alongside a
`Number`, so a satellite albedo product reaches the two-source radiation balance cell by
cell. The land flux kernel localizes them before the canopy solve.

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

surface = CanopyGroundSurface(Float64; soil_humidity, leaf_albedo)

local_interface_formulation(surface, 2, 1).leaf_albedo

# output
0.35
```
"""
struct CanopyGroundSurface{S, C, RF, LA, GA, CE, GE, FT, U, W, L}
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

function CanopyGroundSurface(FT=Oceananigans.defaults.FloatType;
                             soil_humidity,
                             canopy_humidity               = CanopyConductanceHumidity(FT),
                             ground_heat_flux              = SoilConductiveFlux(1.5, 0.05),
                             leaf_albedo                   = 0.15,
                             ground_albedo                 = 0.15,
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
    return CanopyGroundSurface(soil_humidity, canopy_humidity, ground_heat_flux,
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

@inline temperature_formulation(state::CanopyGroundSurface) = state
@inline specific_humidity_formulation(state::CanopyGroundSurface) = state

#####
##### Per-cell optics: localization
#####

# Collapse `Field`-valued slots to cell (i, j) before the index-free canopy solve;
# `state2dindex` returns a `Number` slot unchanged.
@inline local_interface_formulation(formulation, i, j) = formulation

@inline local_interface_formulation(c::CanopyGroundSurface, i, j) =
    CanopyGroundSurface(c.soil_humidity, c.canopy_humidity, c.ground_heat_flux,
                        state2dindex(c.leaf_albedo, i, j),
                        state2dindex(c.ground_albedo, i, j),
                        state2dindex(c.maximum_canopy_emissivity, i, j),
                        state2dindex(c.ground_emissivity, i, j),
                        c.extinction, c.clumping, c.leaf_boundary_layer_velocity,
                        c.undercanopy_transfer_velocity, c.wet_soil_resistance, c.ground_surface_resistance,
                        c.inner_iterations)

Base.summary(::CanopyGroundSurface) = "CanopyGroundSurface"
Base.show(io::IO, c::CanopyGroundSurface) =
    print(io, "CanopyGroundSurface(soil_humidity=", summary(c.soil_humidity),
          ", canopy_humidity=", summary(c.canopy_humidity), ")")

Adapt.@adapt_structure CanopyGroundSurface

# Materialization / identity — delegate to the sub-models so the per-cell interface
# state carries the soil saturation, bulk temperature, and LAI the branches read.
@inline interface_phase(c::CanopyGroundSurface) = interface_phase(c.soil_humidity)
@inline ground_conductance(c::CanopyGroundSurface) = ground_conductance(c.ground_heat_flux)
# The soil branch reads the saturation 𝒮, the canopy branch its stress state, and the
# canopy water store Mᶜᵃⁿ with its capacity gives the wet fraction fʷᵉᵗ.
land_state_accessors(c::CanopyGroundSurface) =
    merge(land_state_accessors(c.soil_humidity),
          land_state_accessors(c.canopy_humidity),
          (canopy_water_storage = land_canopy_water_storage, canopy_water_capacity = land_canopy_water_capacity))
@inline canopy_leaf_area_index(c::CanopyGroundSurface) = canopy_leaf_area_index(c.canopy_humidity)
@inline interface_vegetation_state(i, j, grid, c::CanopyGroundSurface, vegetation, time_interpolator) =
    interface_vegetation_state(i, j, grid, c.canopy_humidity, vegetation, time_interpolator)

# Leaf vapor conductance: stomata in series with the leaf boundary layer on the dry
# fraction, the wet-canopy conductance `gʷᵉᵗ` in parallel. LAI enters through the
# arguments: `gᶜᵃⁿ`, `gˡᵇˡ`, and `gʷᵉᵗ` all scale with it, so a leafless canopy has no leaf path.
@inline function leaf_vapor_conductance(gᶜᵃⁿ, gˡᵇˡ, fʷᵉᵗ, gʷᵉᵗ)
    gᵈ = ifelse(gᶜᵃⁿ + gˡᵇˡ > 0, gᶜᵃⁿ * gˡᵇˡ / (gᶜᵃⁿ + gˡᵇˡ), zero(gᶜᵃⁿ))
    return (1 - fʷᵉᵗ) * gᵈ + gʷᵉᵗ
end

# Wet-canopy vapor conductance: the wetted fraction fʷᵉᵗ = (Mᶜᵃⁿ/Mᶜᵃⁿ⁺)^(2/3) (Deardorff 1978)
# of the leaf boundary-layer conductance, capped so the store cannot evaporate more than
# it holds over the step, Jʷᵉᵗ = gʷᵉᵗ Δq ≤ Mᶜᵃⁿ/Δt. A dry canopy (Mᶜᵃⁿ = 0) has no wet path.
@inline function wet_canopy_conductance(Ψˡᵃ, gˡᵇˡ, Δq, Δt)
    FT    = typeof(gˡᵇˡ)
    Mᶜᵃⁿ  = convert(FT, Ψˡᵃ.canopy_water_storage)
    Mᶜᵃⁿ⁺ = convert(FT, Ψˡᵃ.canopy_water_capacity)
    fʷᵉᵗ  = ifelse(Mᶜᵃⁿ⁺ > 0, min(Mᶜᵃⁿ / Mᶜᵃⁿ⁺, one(FT))^convert(FT, 2//3), zero(FT))
    gʷᵉᵗ  = fʷᵉᵗ * gˡᵇˡ
    gʷᵉᵗ  = ifelse((Δt > 0) & (Δq > 0), min(gʷᵉᵗ, Mᶜᵃⁿ / (Δt * Δq)), gʷᵉᵗ)
    return fʷᵉᵗ, gʷᵉᵗ
end

# Leaf vapor conductance `gˡᵉᵃᵛ`, its wet-canopy part `gʷᵉᵗ`, and the leaf-saturated
# humidity `qˡᵉᵃ` at the leaf temperature; the stomata and the wet cap see the node
# humidity `qᵃᶜᵃⁿ`.
@inline function leaf_vapor_terms(c, Tˡᵉᵃ, qᵃᶜᵃⁿ, gˡᵇˡ, Δt, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, transmittance)
    gᶜᵃⁿ, qˡᵉᵃ = canopy_conductance_terms(c.canopy_humidity, Tˡᵉᵃ, qᵃᶜᵃⁿ, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, transmittance)
    fʷᵉᵗ, gʷᵉᵗ = wet_canopy_conductance(Ψˡᵃ, gˡᵇˡ, qˡᵉᵃ - qᵃᶜᵃⁿ, Δt)
    return leaf_vapor_conductance(gᶜᵃⁿ, gˡᵇˡ, fʷᵉᵗ, gʷᵉᵗ), gʷᵉᵗ, qˡᵉᵃ
end

# Ground vapor conductance `gᵍʳᵒᵛ`, source humidity `qᵍʳᵒ`, and the source's slope `dqᵍʳᵒ` in the
# ground surface temperature `Tᵍʳᵒ`: the dry-layer branch (front humidity qᵉ through Gᵉ in series with the
# ground-surface + undercanopy path `gᵈᵇʳ`) blended with the saturated-skin branch (qᵍʳᵒ⁺ through gᵍʳᵒᵛ⁺)
# by the soil model's weight `fᵈᵇʳ`.
@inline function soil_vapor_terms(soil, Tᵍʳᵒ, gᵍʳᵒᵛ⁺, gᵈᵇʳ, Ψₛ, Ψₐ, Ψˡᵃ, ℙₐ)
    Gᵉ, qᵉ, fᵈᵇʳ, qᵍʳᵒ⁺, dqᵉ, dqᵍʳᵒ⁺ = dry_layer_terms(soil, Tᵍʳᵒ, Ψₛ, Ψₐ, Ψˡᵃ, ℙₐ)
    Gᵉ = ifelse(gᵈᵇʳ + Gᵉ > 0, Gᵉ * gᵈᵇʳ / (gᵈᵇʳ + Gᵉ), zero(Gᵉ))
    gᵍʳᵒᵛ = fᵈᵇʳ * Gᵉ + (1 - fᵈᵇʳ) * gᵍʳᵒᵛ⁺
    blended = gᵍʳᵒᵛ > eps(eltype(Ψₛ))
    qᵍʳᵒ  = ifelse(blended, (fᵈᵇʳ * Gᵉ * qᵉ  + (1 - fᵈᵇʳ) * gᵍʳᵒᵛ⁺ * qᵍʳᵒ⁺)  / gᵍʳᵒᵛ, qᵍʳᵒ⁺)
    dqᵍʳᵒ = ifelse(blended, (fᵈᵇʳ * Gᵉ * dqᵉ + (1 - fᵈᵇʳ) * gᵍʳᵒᵛ⁺ * dqᵍʳᵒ⁺) / gᵍʳᵒᵛ, dqᵍʳᵒ⁺)
    return gᵍʳᵒᵛ, qᵍʳᵒ, dqᵍʳᵒ
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
@inline function ground_longwave(downwelling_longwave, Tˡᵉᵃ, Tᵍʳᵒ, ϵˡᵉᵃ, ϵᵍʳᵒ, σ)
    ground_downwelling_longwave = (1 - ϵˡᵉᵃ) * downwelling_longwave + ϵˡᵉᵃ * σ * Tˡᵉᵃ^4
    ℐꜛˡʷᵍʳᵒ = ϵᵍʳᵒ * σ * Tᵍʳᵒ^4 + (1 - ϵᵍʳᵒ) * ground_downwelling_longwave
    return ground_downwelling_longwave, ℐꜛˡʷᵍʳᵒ
end

"""
    advance_canopy_air(x, equilibrium, Σg, C, Δt)

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
    canopy_air_space_solve(c::CanopyGroundSurface, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ)

Solve the leaf temperature `Tˡᵉᵃ` and ground surface temperature `Tᵍʳᵒ` of one cell by a
quasi-Newton iteration on their energy balances, with the canopy-air node `(Tᵃᶜᵃⁿ, qᵃᶜᵃⁿ)`
eliminated as the conductance-weighted mean of its sources. The aerodynamic conductances
are frozen at the similarity scales of `Ψₛ` (the Monin–Obukhov fixed point outside updates
them), and the Jacobian's dependence on the leaf vapor conductance `gˡᵉᵃᵛ(Tˡᵉᵃ)` (stomata
and wet-canopy cap) is a one-sided difference, with the canopy-air node memory held at its
iterate value; the rest of the Jacobian is exact. The storage enters only through the node
capacities `Cᵀ = ρcᵖᵐhᵃᶜᵃⁿ` and `Cᵛ = ρhᵃᶜᵃⁿ`: zero without land surface storage, so the
node is that mean; with canopy-air storage the node relaxes from its
start-of-step value toward the mean over the step. `Ψₛ` is the previous fixed-point
iterate, whose similarity scales set the aerodynamic conductances and whose node values
are the start-of-step node; `Ψˡᵃ` is the land state (`Ψˡᵃ.T` the bulk reservoir `Tˡᵃ`), and `Ψᵣ` the interface
radiation state.
"""
@inline function canopy_air_space_solve(c::CanopyGroundSurface, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ)
    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    FT  = eltype(Ψₛ)
    pᵃᵗ = Ψₐ.p
    qᵃᵗ = Ψₐ.q
    Tᵃᵗ = Ψₐ.T
    ℒ   = AtmosphericThermodynamics.latent_heat_vapor(ℂᵃᵗ, Tᵃᵗ)
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)
    cᵖᵐ = AtmosphericThermodynamics.cp_m(ℂᵃᵗ, qᵃᵗ)
    θᵃᵗ = surface_atmosphere_temperature(Ψₐ, ℙₐ)
    Tˡᵃ = Ψˡᵃ.T
    Δt  = Ψˡᵃ.time_step
    LAI = Ψˡᵃ.leaf_area_index
    u★  = Ψₛ.fluxes.u★

    gᵃᵀ = aerodynamic_heat_conductance(Ψₛ, Ψₐ, ℂᵃᵗ)
    gᵃᵛ = aerodynamic_vapor_conductance(Ψₛ, Ψₐ, ℂᵃᵗ)

    # Land surface velocities are zero, so the surface wind speed is the atmospheric one.
    ΔU    = sqrt(Ψₐ.u^2 + Ψₐ.v^2)
    wᵘᶜᵃⁿ = undercanopy_transfer_velocity(c.undercanopy_transfer_velocity, LAI, ΔU, u★)
    gˡᵉᵃᵀ = ρᵃᵗ * cᵖᵐ * LAI * c.leaf_boundary_layer_velocity
    gˡᵇˡ  = ρᵃᵗ * LAI * c.leaf_boundary_layer_velocity
    gᵍʳᵒᵀ = ρᵃᵗ * cᵖᵐ * wᵘᶜᵃⁿ
    # Ground vapor path: the ground-surface resistance `rᵍʳᵒ` and, on the moist-soil branch, the soil
    # surface resistance `rˢᵒⁱ` in series with the undercanopy path (Sakaguchi & Zeng 2009).
    rᵍʳᵒ   = ground_surface_resistance(c.ground_surface_resistance, u★)
    rˢᵒⁱ   = soil_surface_resistance(c.wet_soil_resistance, Ψˡᵃ.𝒮)
    gᵈᵇʳ   = ρᵃᵗ * wᵘᶜᵃⁿ / (1 + wᵘᶜᵃⁿ * rᵍʳᵒ)
    gᵍʳᵒᵛ⁺ = ρᵃᵗ * wᵘᶜᵃⁿ / (1 + wᵘᶜᵃⁿ * (rᵍʳᵒ + rˢᵒⁱ))
    Λᵍʳᵒ   = convert(FT, ground_conductance(c.ground_heat_flux))

    # Canopy-air heat and vapor capacities (zero for a massless node) and the
    # start-of-step node.
    hᵃᶜᵃⁿ = convert(FT, canopy_air_depth(Ψˡᵃ.surface_storage))
    Cᵀ    = ρᵃᵗ * cᵖᵐ * hᵃᶜᵃⁿ
    Cᵛ    = ρᵃᵗ * hᵃᶜᵃⁿ
    T⁻    = Ψₛ.temperature
    q⁻    = Ψₛ.specific_humidity

    # Shortwave split and longwave emissivities (broadband), on downwelling magnitudes.
    σ = Ψᵣ.σ
    downwelling_shortwave = - Ψᵣ.ℐꜜˢʷ
    downwelling_longwave  = - Ψᵣ.ℐꜜˡʷ
    αˡᵉᵃ = convert(FT, c.leaf_albedo)
    αᵍʳᵒ = convert(FT, c.ground_albedo)
    ϵˡᵉᵃ = convert(FT, c.maximum_canopy_emissivity) * (1 - exp(-LAI))
    ϵᵍʳᵒ = convert(FT, c.ground_emissivity)
    transmittance = canopy_transmittance(c.extinction, c.clumping, LAI)
    leaf_absorbed_shortwave   = (1 - αˡᵉᵃ) * (1 - transmittance) * downwelling_shortwave
    ground_absorbed_shortwave = transmittance * (1 - αᵍʳᵒ) * downwelling_shortwave
    effective_albedo = (1 - transmittance) * αˡᵉᵃ + transmittance * αᵍʳᵒ

    # Thermal node: its conductances are fixed through the skin iteration. The node is the
    # equilibrium mean of its sources blended with the start-of-step node by the memory `m`,
    # so its sensitivity to source `i` is (1 − m) gᵢ/Σg.
    Σgᵀ = gᵃᵀ + gˡᵉᵃᵀ + gᵍʳᵒᵀ
    mᵀ  = node_memory(Σgᵀ, Cᵀ, Δt)
    ∂Tᵃᶜᵃⁿ∂Tˡᵉᵃ = (1 - mᵀ) * gˡᵉᵃᵀ / max(Σgᵀ, eps(FT))
    ∂Tᵃᶜᵃⁿ∂Tᵍʳᵒ = (1 - mᵀ) * gᵍʳᵒᵀ / max(Σgᵀ, eps(FT))

    Tˡᵉᵃ  = Tˡᵃ
    Tᵍʳᵒ  = Tˡᵃ
    Tᵃᶜᵃⁿ = T⁻
    qᵃᶜᵃⁿ = q⁻
    maximum_temperature_step = convert(FT, 25)

    for _ in 1:c.inner_iterations
        gˡᵉᵃᵛ, gʷᵉᵗ, qˡᵉᵃ = leaf_vapor_terms(c, Tˡᵉᵃ, qᵃᶜᵃⁿ, gˡᵇˡ, Δt, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, transmittance)
        # The leaf vapor conductance depends on the leaf temperature through the stomata and the
        # wet-canopy cap; its slope is a one-sided difference at the same node humidity.
        δT = sqrt(eps(FT)) * Tˡᵉᵃ
        gˡᵉᵃᵛ⁺, _, _ = leaf_vapor_terms(c, Tˡᵉᵃ + δT, qᵃᶜᵃⁿ, gˡᵇˡ, Δt, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, transmittance)
        ∂gˡᵉᵃᵛ∂Tˡᵉᵃ  = (gˡᵉᵃᵛ⁺ - gˡᵉᵃᵛ) / δT
        gᵍʳᵒᵛ, qᵍʳᵒ, dqᵍʳᵒ = soil_vapor_terms(c.soil_humidity, Tᵍʳᵒ, gᵍʳᵒᵛ⁺, gᵈᵇʳ, Ψₛ, Ψₐ, Ψˡᵃ, ℙₐ)

        Σgᵛ = gᵃᵛ + gˡᵉᵃᵛ + gᵍʳᵒᵛ
        mᵛ  = node_memory(Σgᵛ, Cᵛ, Δt)
        Tᵃᶜᵃⁿ, _ = canopy_air_node(T⁻, (gᵍʳᵒᵀ, gˡᵉᵃᵀ, gᵃᵀ), (Tᵍʳᵒ, Tˡᵉᵃ, θᵃᵗ), mᵀ)
        qᵃᶜᵃⁿ, equilibrium_humidity = canopy_air_node(q⁻, (gᵍʳᵒᵛ, gˡᵉᵃᵛ, gᵃᵛ), (qᵍʳᵒ, qˡᵉᵃ, qᵃᵗ), mᵛ)
        ∂qᵃᶜᵃⁿ∂qˡᵉᵃ = (1 - mᵛ) * gˡᵉᵃᵛ / max(Σgᵛ, eps(FT))
        ∂qᵃᶜᵃⁿ∂qᵍʳᵒ = (1 - mᵛ) * gᵍʳᵒᵛ / max(Σgᵛ, eps(FT))
        ∂qᵃᶜᵃⁿ∂gˡᵉᵃᵛ = (1 - mᵛ) * (qˡᵉᵃ - equilibrium_humidity) / max(Σgᵛ, eps(FT))

        ground_downwelling_longwave, ℐꜛˡʷᵍʳᵒ = ground_longwave(downwelling_longwave, Tˡᵉᵃ, Tᵍʳᵒ, ϵˡᵉᵃ, ϵᵍʳᵒ, σ)
        leaf_net_longwave   = ϵˡᵉᵃ * (downwelling_longwave + ℐꜛˡʷᵍʳᵒ) - 2 * ϵˡᵉᵃ * σ * Tˡᵉᵃ^4
        ground_net_longwave = ϵᵍʳᵒ * (ground_downwelling_longwave - σ * Tᵍʳᵒ^4)
        dqˡᵉᵃ = saturation_humidity_slope(ℂᵃᵗ, Tˡᵉᵃ, pᵃᵗ, c.canopy_humidity.phase, qˡᵉᵃ)

        # Leaf and ground energy residuals and their Jacobian, including each balance's
        # dependence on the other skin through the node and the longwave exchange.
        Rˡᵉᵃ = leaf_absorbed_shortwave + leaf_net_longwave - gˡᵉᵃᵀ * (Tˡᵉᵃ - Tᵃᶜᵃⁿ) - ℒ * gˡᵉᵃᵛ * (qˡᵉᵃ - qᵃᶜᵃⁿ)
        Rᵍʳᵒ = ground_absorbed_shortwave + ground_net_longwave - gᵍʳᵒᵀ * (Tᵍʳᵒ - Tᵃᶜᵃⁿ) - ℒ * gᵍʳᵒᵛ * (qᵍʳᵒ - qᵃᶜᵃⁿ) - Λᵍʳᵒ * (Tᵍʳᵒ - Tˡᵃ)
        ∂Rˡᵉᵃ∂Tˡᵉᵃ = 4 * ϵˡᵉᵃ * σ * Tˡᵉᵃ^3 * (ϵˡᵉᵃ * (1 - ϵᵍʳᵒ) - 2) - gˡᵉᵃᵀ * (1 - ∂Tᵃᶜᵃⁿ∂Tˡᵉᵃ) - ℒ * gˡᵉᵃᵛ * (1 - ∂qᵃᶜᵃⁿ∂qˡᵉᵃ) * dqˡᵉᵃ -
                       ℒ * ∂gˡᵉᵃᵛ∂Tˡᵉᵃ * (qˡᵉᵃ - qᵃᶜᵃⁿ - gˡᵉᵃᵛ * ∂qᵃᶜᵃⁿ∂gˡᵉᵃᵛ)
        ∂Rˡᵉᵃ∂Tᵍʳᵒ = 4 * ϵˡᵉᵃ * ϵᵍʳᵒ * σ * Tᵍʳᵒ^3 + gˡᵉᵃᵀ * ∂Tᵃᶜᵃⁿ∂Tᵍʳᵒ + ℒ * gˡᵉᵃᵛ * ∂qᵃᶜᵃⁿ∂qᵍʳᵒ * dqᵍʳᵒ
        ∂Rᵍʳᵒ∂Tˡᵉᵃ = 4 * ϵᵍʳᵒ * ϵˡᵉᵃ * σ * Tˡᵉᵃ^3 + gᵍʳᵒᵀ * ∂Tᵃᶜᵃⁿ∂Tˡᵉᵃ + ℒ * gᵍʳᵒᵛ * ∂qᵃᶜᵃⁿ∂qˡᵉᵃ * dqˡᵉᵃ +
                    ℒ * gᵍʳᵒᵛ * ∂qᵃᶜᵃⁿ∂gˡᵉᵃᵛ * ∂gˡᵉᵃᵛ∂Tˡᵉᵃ
        ∂Rᵍʳᵒ∂Tᵍʳᵒ = -4 * ϵᵍʳᵒ * σ * Tᵍʳᵒ^3 - gᵍʳᵒᵀ * (1 - ∂Tᵃᶜᵃⁿ∂Tᵍʳᵒ) - ℒ * gᵍʳᵒᵛ * (1 - ∂qᵃᶜᵃⁿ∂qᵍʳᵒ) * dqᵍʳᵒ - Λᵍʳᵒ
        # A canopy-free cell (LAI = 0) has no leaf balance: Rˡᵉᵃ = ∂Rˡᵉᵃ/∂Tˡᵉᵃ = 0.
        ∂Rˡᵉᵃ∂Tˡᵉᵃ = ifelse(∂Rˡᵉᵃ∂Tˡᵉᵃ < 0, ∂Rˡᵉᵃ∂Tˡᵉᵃ, -one(FT))
        determinant = ∂Rˡᵉᵃ∂Tˡᵉᵃ * ∂Rᵍʳᵒ∂Tᵍʳᵒ - ∂Rˡᵉᵃ∂Tᵍʳᵒ * ∂Rᵍʳᵒ∂Tˡᵉᵃ
        ΔTˡᵉᵃ = (∂Rᵍʳᵒ∂Tᵍʳᵒ * Rˡᵉᵃ - ∂Rˡᵉᵃ∂Tᵍʳᵒ * Rᵍʳᵒ) / determinant
        ΔTᵍʳᵒ = (∂Rˡᵉᵃ∂Tˡᵉᵃ * Rᵍʳᵒ - ∂Rᵍʳᵒ∂Tˡᵉᵃ * Rˡᵉᵃ) / determinant
        Tˡᵉᵃ -= clamp(ΔTˡᵉᵃ, -maximum_temperature_step, maximum_temperature_step)
        Tᵍʳᵒ    -= clamp(ΔTᵍʳᵒ, -maximum_temperature_step, maximum_temperature_step)
    end

    # The node and the flux partition at the final skins.
    gˡᵉᵃᵛ, gʷᵉᵗ, qˡᵉᵃ = leaf_vapor_terms(c, Tˡᵉᵃ, qᵃᶜᵃⁿ, gˡᵇˡ, Δt, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, transmittance)
    gᵍʳᵒᵛ, qᵍʳᵒ, _ = soil_vapor_terms(c.soil_humidity, Tᵍʳᵒ, gᵍʳᵒᵛ⁺, gᵈᵇʳ, Ψₛ, Ψₐ, Ψˡᵃ, ℙₐ)
    Σgᵛ = gᵃᵛ + gˡᵉᵃᵛ + gᵍʳᵒᵛ
    Tᵃᶜᵃⁿ, equilibrium_temperature = canopy_air_node(T⁻, (gᵍʳᵒᵀ, gˡᵉᵃᵀ, gᵃᵀ), (Tᵍʳᵒ, Tˡᵉᵃ, θᵃᵗ), mᵀ)
    qᵃᶜᵃⁿ, equilibrium_humidity    = canopy_air_node(q⁻, (gᵍʳᵒᵛ, gˡᵉᵃᵛ, gᵃᵛ), (qᵍʳᵒ, qˡᵉᵃ, qᵃᵗ), node_memory(Σgᵛ, Cᵛ, Δt))

    # Effective radiating temperature: σ T⁴ = upwelling longwave above the canopy.
    _, ℐꜛˡʷᵍʳᵒ = ground_longwave(downwelling_longwave, Tˡᵉᵃ, Tᵍʳᵒ, ϵˡᵉᵃ, ϵᵍʳᵒ, σ)
    ℐꜛˡʷ = (1 - ϵˡᵉᵃ) * ℐꜛˡʷᵍʳᵒ + ϵˡᵉᵃ * σ * Tˡᵉᵃ^4
    effective_temperature = ifelse(σ > 0, (ℐꜛˡʷ / σ)^convert(FT, 1//4), Tᵃᶜᵃⁿ)

    𝒬ᵀˡᵉᵃ = gˡᵉᵃᵀ * (Tˡᵉᵃ - Tᵃᶜᵃⁿ)
    𝒬ᵀᵍʳᵒ = gᵍʳᵒᵀ * (Tᵍʳᵒ - Tᵃᶜᵃⁿ)
    𝒬ᵛˡᵉᵃ = ℒ * gˡᵉᵃᵛ * (qˡᵉᵃ - qᵃᶜᵃⁿ)
    𝒬ᵛᵍʳᵒ = ℒ * gᵍʳᵒᵛ * (qᵍʳᵒ - qᵃᶜᵃⁿ)
    𝒬ᵍʳᵒ  = Λᵍʳᵒ * (Tᵍʳᵒ - Tˡᵃ)
    Jʷᵉᵗ  = gʷᵉᵗ * (qˡᵉᵃ - qᵃᶜᵃⁿ)

    return (; Tˡᵉᵃ = convert(FT, Tˡᵉᵃ), Tᵍʳᵒ = convert(FT, Tᵍʳᵒ),
              Tᵃᶜᵃⁿ = convert(FT, Tᵃᶜᵃⁿ), qᵃᶜᵃⁿ = convert(FT, qᵃᶜᵃⁿ),
              effective_temperature = convert(FT, effective_temperature),
              effective_albedo = convert(FT, effective_albedo),
              𝒬ᵀˡᵉᵃ = convert(FT, 𝒬ᵀˡᵉᵃ), 𝒬ᵀᵍʳᵒ = convert(FT, 𝒬ᵀᵍʳᵒ),
              𝒬ᵛˡᵉᵃ = convert(FT, 𝒬ᵛˡᵉᵃ), 𝒬ᵛᵍʳᵒ = convert(FT, 𝒬ᵛᵍʳᵒ),
              𝒬ᵍʳᵒ = convert(FT, 𝒬ᵍʳᵒ), Jʷᵉᵗ = convert(FT, Jʷᵉᵗ), ℒ = convert(FT, ℒ),
              Σgᵀ = convert(FT, Σgᵀ), Σgᵛ = convert(FT, Σgᵛ),
              equilibrium_temperature = convert(FT, equilibrium_temperature), equilibrium_humidity = convert(FT, equilibrium_humidity),
              Cᵀ = convert(FT, Cᵀ), Cᵛ = convert(FT, Cᵛ))
end

# A diagnostic node is re-solved every iterate; one shared solve returns both node values.
@inline function interface_temperature_and_humidity(c::CanopyGroundSurface,
                                                    Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₛ, ℙₐ, ℙᵢ,
                                                    storage::Nothing = Ψˡᵃ.surface_storage)
    solution = canopy_air_space_solve(c, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ)
    return solution.Tᵃᶜᵃⁿ, solution.qᵃᶜᵃⁿ
end

# A prognostic node is model state, frozen through the fixed point.
@inline interface_temperature_and_humidity(::CanopyGroundSurface,
                                           Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₛ, ℙₐ, ℙᵢ, storage) =
    (Ψₛ.temperature, Ψₛ.specific_humidity)

"""
    struct CanopyGroundSurfaceDiagnostics

What a [`CanopyGroundSurface`](@ref) stores in the interface `temperature` slot: the
canopy-air node temperature the atmosphere sees, the leaf, ground, and effective radiating
temperatures, and the two-source flux partition the land and the radiation read.
"""
struct CanopyGroundSurfaceDiagnostics{F}
    interface              :: F   # canopy-air node Tᵃᶜᵃⁿ (what MOST sees)
    canopy                 :: F   # leaf temperature Tˡᵉᵃ
    ground                 :: F   # ground surface temperature Tᵍʳᵒ
    effective              :: F   # skin (effective radiating) temperature Tʳᵃᵈ
    effective_albedo       :: F   # broadband shortwave albedo of the canopy + ground column
    ground_heat_flux       :: F   # skin→bulk conduction 𝒬ᵍʳᵒ
    canopy_latent_heat     :: F   # leaf latent heat 𝒬ᵛˡᵉᵃ (transpiration + wet canopy)
    ground_latent_heat     :: F   # soil evaporation 𝒬ᵛᵍʳᵒ
    canopy_sensible_heat   :: F   # leaf sensible 𝒬ᵀˡᵉᵃ
    ground_sensible_heat   :: F   # ground sensible 𝒬ᵀᵍʳᵒ
    canopy_evaporation     :: F   # wet-canopy evaporation Jʷᵉᵗ (kg m⁻² s⁻¹, up)
    wet_canopy_latent_heat :: F   # wet-canopy latent heat ℒ·Jʷᵉᵗ (W m⁻², up)
    land_vapor_flux        :: F   # soil evaporation + transpiration, drawn from the land water store
    transpiration          :: F   # leaf transpiration 𝒬ᵛˡᵉᵃ/ℒ − Jʷᵉᵗ, drawn from the soil
end

CanopyGroundSurfaceDiagnostics(grid) =
    CanopyGroundSurfaceDiagnostics(ntuple(_ -> Field{Center, Center, Nothing}(grid),
                                     Val(fieldcount(CanopyGroundSurfaceDiagnostics)))...)

Adapt.@adapt_structure CanopyGroundSurfaceDiagnostics

Base.summary(::CanopyGroundSurfaceDiagnostics) = "CanopyGroundSurfaceDiagnostics"
Base.show(io::IO, d::CanopyGroundSurfaceDiagnostics) = print(io, summary(d))
