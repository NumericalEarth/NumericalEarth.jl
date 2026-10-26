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
wᵘᶜ = \\frac{Cᴰ \\, ΔU}{\\max\\!\\left(1 - e^{-(LAI + SAI)}, ε\\right)},
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
wᵘᶜ = Cₛ u★, \\qquad Cₛ = Cₛᵇ W + Cₛᵈ (1 - W), \\qquad W = e^{-(LAI + SAI)},
\\qquad Cₛᵇ = \\frac{ϰ}{0.13} \\left(\\frac{ℓᵍ u★}{ν}\\right)^{-0.45},
```

with the dense-canopy coefficient `Cₛᵈ` (`dense_canopy_coefficient`, 0.004 after
Dickinson et al. 1993), the ground roughness length `ℓᵍ` (`ground_roughness_length`, m),
the kinematic viscosity of air `ν` (`kinematic_viscosity`, m² s⁻¹), and the von Kármán
constant `ϰ = 0.4` (CLM5 Tech Note Eqs. 2.5.116–2.5.121). Unlike
[`AreaIndexUndercanopyVelocity`](@ref), the sparse-canopy limit is closed physically
by the bare-ground law rather than floored, and calm air (`u★ → 0`) decouples the ground.

```jldoctest
using NumericalEarth

FrictionVelocityUndercanopyVelocity()

# output
FrictionVelocityUndercanopyVelocity(Cₛᵈ=0.004, ℓᵍ=0.01, SAI=0.0)
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
           ", ℓᵍ=", prettysummary(u.ground_roughness_length),
           ", SAI=", prettysummary(u.stem_area_index), ")")
Base.show(io::IO, u::FrictionVelocityUndercanopyVelocity) = print(io, summary(u))

# A `Number` is a constant transfer velocity (m s⁻¹).
@inline undercanopy_transfer_velocity(wᵘᶜ::Number, LAI, ΔU, u★) = convert(typeof(LAI), wᵘᶜ)

@inline function undercanopy_transfer_velocity(u::AreaIndexUndercanopyVelocity, LAI, ΔU, u★)
    FT = typeof(LAI)
    Cᴰ = convert(FT, u.drag_coefficient)
    area_index = LAI + convert(FT, u.stem_area_index)
    wᵘᶜ = Cᴰ * ΔU / max(1 - exp(-area_index), eps(FT))
    aerodynamic_limit = u★^2 / max(ΔU, eps(FT))
    return min(wᵘᶜ, aerodynamic_limit)
end

@inline function undercanopy_transfer_velocity(u::FrictionVelocityUndercanopyVelocity, LAI, ΔU, u★)
    FT = typeof(LAI)
    W  = exp(-(LAI + convert(FT, u.stem_area_index)))
    Cᵈ = convert(FT, u.dense_canopy_coefficient)
    ℓᵍ = convert(FT, u.ground_roughness_length)
    ν  = convert(FT, u.kinematic_viscosity)
    ϰ  = convert(FT, 2//5)
    a  = convert(FT, 13//100)
    n  = convert(FT, 9//20)
    # Cₛᵇ u★ grouped as u★^(1−n) so u★ → 0 gives 0 rather than Inf·0.
    wᵇ = (ϰ / a) * (ℓᵍ / ν)^(-n) * u★^(1 - n)
    return W * wᵇ + (1 - W) * Cᵈ * u★
end

"""
    SellersSoilResistance(FT = Oceananigans.defaults.FloatType;
                          intercept = 8.206, slope = 4.255)

Empirical ground-surface resistance for the moist-soil (vanishing dry layer) evaporation
branch of a [`CanopyGroundSurface`](@ref),

```math
rᵍ = e^{a - b 𝒮},
```

with the surface saturation `𝒮`, fit by [Sellers et al. (1992)](@cite sellers1992) to the
FIFE prairie flux stations (`rᵍ(1) ≈ 52` s m⁻¹). The FIFE sites carried litter, so the fit
is an effective soil-plus-litter resistance, an alternative to an explicit
[`LitterResistance`](@ref) rather than a companion to one.

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
    LitterResistance(FT = Oceananigans.defaults.FloatType;
                     litter_area_index = 1, transfer_coefficient = 0.004)

Plant-litter resistance to ground evaporation
([Sakaguchi and Zeng (2009)](@cite sakaguchi2009), Eq. 13),

```math
rˡⁱᵗ = \\frac{1 - e^{-Lˡⁱᵗ}}{C u★},
```

with the litter area index `Lˡⁱᵗ` (m² m⁻²), a turbulent transfer coefficient `C`, and the
friction velocity `u★` standing in for the wind speed in the canopy air space. The
resistance vanishes without litter (`Lˡⁱᵗ = 0`) and saturates at `1/(C u★)` under a thick
blanket. The default `Lˡⁱᵗ = 1` is the minimum stem-plus-litter area of the global surface
dataset behind the fit; the snow-burial reduction of the original scheme is omitted.

```jldoctest
using NumericalEarth

LitterResistance()

# output
LitterResistance(Lˡⁱᵗ=1.0, C=0.004)
```
"""
struct LitterResistance{FT}
    litter_area_index    :: FT
    transfer_coefficient :: FT
end

LitterResistance(FT::Type = Oceananigans.defaults.FloatType;
                 litter_area_index = 1, transfer_coefficient = 0.004) =
    LitterResistance(convert(FT, litter_area_index), convert(FT, transfer_coefficient))

Base.summary(r::LitterResistance) =
    string("LitterResistance(Lˡⁱᵗ=", prettysummary(r.litter_area_index),
           ", C=", prettysummary(r.transfer_coefficient), ")")
Base.show(io::IO, r::LitterResistance) = print(io, summary(r))

@inline litter_resistance(::Nothing, u★) = zero(u★)
@inline function litter_resistance(r::LitterResistance, u★)
    FT   = typeof(u★)
    Lˡⁱᵗ = convert(FT, r.litter_area_index)
    C    = convert(FT, r.transfer_coefficient)
    return (1 - exp(-Lˡⁱᵗ)) / max(C * u★, eps(FT))
end

#####
##### Canopy-air storage: diagnostic (massless) or prognostic (the node carries the
##### heat and moisture capacity of the canopy air layer).
#####

"""
    DiagnosticCanopyAir()

Massless canopy-air node (the default): `(Tᵃᶜ, qᵃᶜ)` is the conductance-weighted mean
of the leaf, ground, and atmosphere states every iterate of the Monin–Obukhov fixed point.
"""
struct DiagnosticCanopyAir end

Base.summary(::DiagnosticCanopyAir) = "DiagnosticCanopyAir"
Base.show(io::IO, s::DiagnosticCanopyAir) = print(io, summary(s))

"""
    PrognosticCanopyAir(FT = Oceananigans.defaults.FloatType; layer_depth = 10)

Prognostic canopy-air storage: the node `(Tᵃᶜ, qᵃᶜ)` carries the heat and moisture
capacity of the canopy air layer of depth ``hᵃᶜ`` (`layer_depth`, m — a `Number`, or
a `Field` of canopy heights),

```math
ρ c^{pm} h^{ac} \\frac{dT^{ac}}{dt} = \\mathcal{Q}^{T\\,\\mathrm{leaf}} + \\mathcal{Q}^{Tg} - \\mathcal{Q}^T, \\qquad
ρ h^{ac} \\frac{dq^{ac}}{dt} = J^{v\\,\\mathrm{leaf}} + J^{vg} - J^v,
```

integrated with the model time step. The node is frozen while the similarity scales
iterate and advanced once per step by the exact exponential relaxation toward the
conductance-weighted equilibrium of its sources; the skins are solved, and the fluxes
exported, at the step-mean node. `layer_depth → 0` recovers the massless node.

```jldoctest
using NumericalEarth

PrognosticCanopyAir(layer_depth = 12)

# output
PrognosticCanopyAir(hᵃᶜ=12.0)
```
"""
struct PrognosticCanopyAir{H}
    layer_depth :: H
end

PrognosticCanopyAir(FT::Type = Oceananigans.defaults.FloatType; layer_depth = 10) =
    PrognosticCanopyAir(convert_if_number(FT, layer_depth))

Base.summary(s::PrognosticCanopyAir) =
    string("PrognosticCanopyAir(hᵃᶜ=", prettysummary(s.layer_depth), ")")
Base.show(io::IO, s::PrognosticCanopyAir) = print(io, summary(s))

Adapt.adapt_structure(to, s::PrognosticCanopyAir) = PrognosticCanopyAir(Adapt.adapt(to, s.layer_depth))

"""
    struct CanopyGroundSurface

Two-source canopy + ground surface with a canopy-air node, for forests: tall vegetation
whose crowns enclose a canopy air space with the ground as a second source beneath it.
Low vegetation (grass, crops) has no separable canopy air layer and belongs to a
single-source [`CanopyConductanceHumidity`](@ref) over a soil skin. Solves the
leaf temperature `Tˡᵉᵃᶠ`, the ground surface temperature `Tᵍ`, and the canopy-air node
`(Tᵃᶜ, qᵃᶜ)` inside the Monin–Obukhov fixed point (diagnostic node, the default),
or advances a prognostic node carrying the canopy-air heat and moisture capacity
(`storage = PrognosticCanopyAir(...)`). Pass it as the interface temperature
formulation; it closes the specific-humidity slot as well. Over an
[`InterceptingHydrology`](@ref) the wetted fraction `fʷᵉᵗ = (Mᶜ/Mᶜ⁺)^{2/3}` of the leaf
([Deardorff, 1978](@cite deardorff1978)) evaporates intercepted water at the potential
rate through the leaf boundary layer alone; a dry canopy (`Mᶜ = 0`) has no wet branch.

Fields:
- `soil_humidity`   : the soil vapor branch (a [`DryLayerHumidity`](@ref)).
- `canopy_humidity` : the leaf vapor/photosynthesis branch (a [`CanopyConductanceHumidity`](@ref)).
- `ground_heat_flux` : ground↔bulk conduction `Λᵍ = κᵀ/ℓᵀ` (a [`SoilConductiveFlux`](@ref)).
- `leaf_albedo`, `ground_albedo` : broadband shortwave albedos. A `Number`, or a
  `Field{Center, Center, Nothing}` of per-cell values (see [`atmosphere_land_interface`](@ref)).
- `maximum_canopy_emissivity`, `ground_emissivity` : longwave emissivities (`ϵˡᵉᵃᶠ = ϵᵐᵃˣ(1 − e^{−LAI})`),
  each a `Number` or a per-cell `Field`.
- `extinction`, `clumping` : Beer–Lambert extinction `K` and foliage clumping index `Ω` of the
  shortwave transmittance `e^{−KΩL}` (`Ω = 1` random foliage, `Ω < 1` foliage gathered into
  crowns that leave gaps).
- `leaf_boundary_layer_velocity` : constant per-leaf transfer velocity through the leaf
  boundary layer `wˡᵇ` (m s⁻¹) → sensible `gˡᵉᵃᶠᵀ = ρcᵖᵐ·LAI·wˡᵇ`, vapor `gˡᵇ = ρ·LAI·wˡᵇ` (in
  series with the stomata when dry, alone over the wetted fraction).
- `undercanopy_transfer_velocity` : ground↔canopy-air transfer velocity `wᵘᶜ` → `gᵍᵀ = ρcᵖᵐ·wᵘᶜ`;
  a constant `Number` (m s⁻¹), an [`AreaIndexUndercanopyVelocity`](@ref), or a
  [`FrictionVelocityUndercanopyVelocity`](@ref).
- `wet_soil_resistance` : soil surface resistance on the moist-soil (vanishing dry layer)
  vapor branch (a [`SellersSoilResistance`](@ref)), or `nothing` (the default: above the
  dry-layer onset the soil itself does not limit evaporation, and the litter layer and
  undercanopy path carry the resistance).
- `litter_resistance` : plant-litter resistance in series on both ground vapor branches
  (a [`LitterResistance`](@ref), the default), or `nothing` for litter-free ground.
- `inner_iterations` : Newton iterations of the coupled skin solve.
- `storage` : the canopy-air node storage — [`DiagnosticCanopyAir`](@ref) (the default,
  massless node) or [`PrognosticCanopyAir`](@ref) (the node carries the canopy-air
  heat and moisture capacity and is advanced with the model time step).

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
struct CanopyGroundSurface{ST, S, C, RF, LA, GA, CE, GE, FT, U, W, L}
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
    litter_resistance             :: L
    inner_iterations              :: Int
    storage                       :: ST
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
                             litter_resistance             = LitterResistance(FT),
                             inner_iterations              = 10,
                             storage                       = DiagnosticCanopyAir())

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
                               wet_soil_resistance, litter_resistance,
                               inner_iterations, storage)
end

const DiagnosticCanopyGroundSurface = CanopyGroundSurface{<:DiagnosticCanopyAir}
const PrognosticCanopyGroundSurface = CanopyGroundSurface{<:PrognosticCanopyAir}

#####
##### Per-cell optics: localization
#####

# Collapse `Field`-valued slots to cell (i, j) before the index-free canopy solve;
# `state2dindex` returns a `Number` slot unchanged.
@inline local_interface_formulation(formulation, i, j) = formulation

@inline local_canopy_air_storage(storage, i, j) = storage
@inline local_canopy_air_storage(s::PrognosticCanopyAir, i, j) =
    PrognosticCanopyAir(state2dindex(s.layer_depth, i, j))

@inline local_interface_formulation(c::CanopyGroundSurface, i, j) =
    CanopyGroundSurface(c.soil_humidity, c.canopy_humidity, c.ground_heat_flux,
                        state2dindex(c.leaf_albedo, i, j),
                        state2dindex(c.ground_albedo, i, j),
                        state2dindex(c.maximum_canopy_emissivity, i, j),
                        state2dindex(c.ground_emissivity, i, j),
                        c.extinction, c.clumping, c.leaf_boundary_layer_velocity,
                        c.undercanopy_transfer_velocity, c.wet_soil_resistance, c.litter_resistance,
                        c.inner_iterations, local_canopy_air_storage(c.storage, i, j))

Base.summary(::CanopyGroundSurface) = "CanopyGroundSurface"
Base.show(io::IO, c::CanopyGroundSurface) =
    print(io, "CanopyGroundSurface(soil_humidity=", summary(c.soil_humidity),
          ", canopy_humidity=", summary(c.canopy_humidity),
          ", storage=", summary(c.storage), ")")

Adapt.@adapt_structure CanopyGroundSurface

# Materialization / identity — delegate to the sub-models so the per-cell interface
# state carries the soil saturation, bulk temperature, and LAI the branches read.
@inline interface_phase(c::CanopyGroundSurface) = interface_phase(c.soil_humidity)
@inline ground_conductance(c::CanopyGroundSurface) = ground_conductance(c.ground_heat_flux)
# The soil branch reads the saturation 𝒮, the canopy branch its stress state, and the
# canopy water store Mᶜ with its capacity gives the wet fraction fʷᵉᵗ.
land_state_accessors(c::CanopyGroundSurface) =
    merge(land_state_accessors(c.soil_humidity),
          land_state_accessors(c.canopy_humidity),
          (canopy_water_storage = land_canopy_water_storage, canopy_water_capacity = land_canopy_water_capacity))
@inline canopy_leaf_area_index(c::CanopyGroundSurface) = canopy_leaf_area_index(c.canopy_humidity)
@inline interface_vegetation_state(i, j, grid, c::CanopyGroundSurface, vegetation, time_interpolator) =
    interface_vegetation_state(i, j, grid, c.canopy_humidity, vegetation, time_interpolator)

# Leaf vapor conductance: stomata in series with the leaf boundary layer on the dry
# fraction, the wet-canopy conductance `gʷᵉᵗ` in parallel. LAI enters through the
# arguments: `gᶜ`, `gˡᵇ`, and `gʷᵉᵗ` all scale with it, so a leafless canopy has no leaf path.
@inline function leaf_vapor_conductance(gᶜ, gˡᵇ, fʷᵉᵗ, gʷᵉᵗ)
    gᵈ = ifelse(gᶜ + gˡᵇ > 0, gᶜ * gˡᵇ / (gᶜ + gˡᵇ), zero(gᶜ))
    return (1 - fʷᵉᵗ) * gᵈ + gʷᵉᵗ
end

# Wet-canopy vapor conductance: the wetted fraction fʷᵉᵗ = (Mᶜ/Mᶜ⁺)^(2/3) (Deardorff 1978)
# of the leaf boundary-layer conductance, capped so the store cannot evaporate more than
# it holds over the step, Jʷᵉᵗ = gʷᵉᵗ Δq ≤ Mᶜ/Δt. A dry canopy (Mᶜ = 0) has no wet path.
@inline function wet_canopy_conductance(Ψˡᵃ, gˡᵇ, Δq, Δt)
    FT   = typeof(gˡᵇ)
    Mᶜ   = convert(FT, Ψˡᵃ.canopy_water_storage)
    Mᶜ⁺  = convert(FT, Ψˡᵃ.canopy_water_capacity)
    fʷᵉᵗ = ifelse(Mᶜ⁺ > 0, min(Mᶜ / Mᶜ⁺, one(FT))^convert(FT, 2//3), zero(FT))
    gʷᵉᵗ = fʷᵉᵗ * gˡᵇ
    gʷᵉᵗ = ifelse((Δt > 0) & (Δq > 0), min(gʷᵉᵗ, Mᶜ / (Δt * Δq)), gʷᵉᵗ)
    return fʷᵉᵗ, gʷᵉᵗ
end

# Leaf vapor conductance `gˡᵉᵃᶠᵛ`, its wet-canopy part `gʷᵉᵗ`, and the leaf-saturated
# humidity `qˡᵉᵃᶠ` at the leaf temperature; the stomata and the wet cap see the node
# humidity `qᵃᶜ`.
@inline function leaf_vapor_terms(c, Tˡᵉᵃᶠ, qᵃᶜ, gˡᵇ, Δt, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, transmittance)
    gᶜ, qˡᵉᵃᶠ = canopy_conductance_terms(c.canopy_humidity, Tˡᵉᵃᶠ, qᵃᶜ, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, transmittance)
    fʷᵉᵗ, gʷᵉᵗ = wet_canopy_conductance(Ψˡᵃ, gˡᵇ, qˡᵉᵃᶠ - qᵃᶜ, Δt)
    return leaf_vapor_conductance(gᶜ, gˡᵇ, fʷᵉᵗ, gʷᵉᵗ), gʷᵉᵗ, qˡᵉᵃᶠ
end

# Ground vapor conductance `gᵍᵛ`, source humidity `qᵍ`, and the source's slope `dqᵍ` in the
# ground surface temperature `Tᵍ`: the dry-layer branch (front humidity qᵉ through Gᵉ in series with the
# litter + undercanopy path `gˡⁱᵗ`) blended with the saturated-skin branch (qᵍ⁺ through gᵍᵛ⁺)
# by the soil model's weight `fᵈ`.
@inline function soil_vapor_terms(soil, Tᵍ, gᵍᵛ⁺, gˡⁱᵗ, Ψₛ, Ψₐ, Ψˡᵃ, ℙₐ)
    Gᵉ, qᵉ, fᵈ, qᵍ⁺, dqᵉ, dqᵍ⁺ = dry_layer_terms(soil, Tᵍ, Ψₛ, Ψₐ, Ψˡᵃ, ℙₐ)
    Gᵉ = ifelse(gˡⁱᵗ + Gᵉ > 0, Gᵉ * gˡⁱᵗ / (gˡⁱᵗ + Gᵉ), zero(Gᵉ))
    gᵍᵛ = fᵈ * Gᵉ + (1 - fᵈ) * gᵍᵛ⁺
    blended = gᵍᵛ > eps(eltype(Ψₛ))
    qᵍ  = ifelse(blended, (fᵈ * Gᵉ * qᵉ  + (1 - fᵈ) * gᵍᵛ⁺ * qᵍ⁺)  / gᵍᵛ, qᵍ⁺)
    dqᵍ = ifelse(blended, (fᵈ * Gᵉ * dqᵉ + (1 - fᵈ) * gᵍᵛ⁺ * dqᵍ⁺) / gᵍᵛ, dqᵍ⁺)
    return gᵍᵛ, qᵍ, dqᵍ
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
@inline function ground_longwave(downwelling_longwave, Tˡᵉᵃᶠ, Tᵍ, ϵˡᵉᵃᶠ, ϵᵍ, σ)
    ground_downwelling_longwave = (1 - ϵˡᵉᵃᶠ) * downwelling_longwave + ϵˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^4
    ℐꜛˡʷᵍ = ϵᵍ * σ * Tᵍ^4 + (1 - ϵᵍ) * ground_downwelling_longwave
    return ground_downwelling_longwave, ℐꜛˡʷᵍ
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

@inline canopy_air_depth(::DiagnosticCanopyAir) = 0
@inline canopy_air_depth(s::PrognosticCanopyAir) = s.layer_depth

"""
    canopy_air_space_solve(c::CanopyGroundSurface, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ)

Solve the leaf temperature `Tˡᵉᵃᶠ` and ground surface temperature `Tᵍ` of one cell by Newton
iteration on their energy balances, with the canopy-air node `(Tᵃᶜ, qᵃᶜ)` eliminated as
the conductance-weighted mean of its sources. The storage enters only through the node
capacities `Cᵀ = ρcᵖᵐhᵃᶜ` and `Cᵛ = ρhᵃᶜ`: zero for a [`DiagnosticCanopyAir`](@ref), so the
node is that mean; for a [`PrognosticCanopyAir`](@ref) the node relaxes from its
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
    ΔU  = sqrt(Ψₐ.u^2 + Ψₐ.v^2)
    wᵘᶜ = undercanopy_transfer_velocity(c.undercanopy_transfer_velocity, LAI, ΔU, u★)
    gˡᵉᵃᶠᵀ = ρᵃᵗ * cᵖᵐ * LAI * c.leaf_boundary_layer_velocity
    gˡᵇ    = ρᵃᵗ * LAI * c.leaf_boundary_layer_velocity
    gᵍᵀ    = ρᵃᵗ * cᵖᵐ * wᵘᶜ
    # Ground vapor path: the litter resistance `rˡⁱᵗ` and, on the moist-soil branch, the soil
    # surface resistance `rᵍ` in series with the undercanopy path (Sakaguchi & Zeng 2009).
    rˡⁱᵗ = litter_resistance(c.litter_resistance, u★)
    rᵍ   = soil_surface_resistance(c.wet_soil_resistance, Ψˡᵃ.𝒮)
    gˡⁱᵗ = ρᵃᵗ * wᵘᶜ / (1 + wᵘᶜ * rˡⁱᵗ)
    gᵍᵛ⁺ = ρᵃᵗ * wᵘᶜ / (1 + wᵘᶜ * (rˡⁱᵗ + rᵍ))
    Λᵍ   = convert(FT, ground_conductance(c.ground_heat_flux))

    # Canopy-air heat and vapor capacities (zero for a massless node) and the
    # start-of-step node.
    hᵃᶜ = convert(FT, canopy_air_depth(c.storage))
    Cᵀ  = ρᵃᵗ * cᵖᵐ * hᵃᶜ
    Cᵛ  = ρᵃᵗ * hᵃᶜ
    T⁻  = Ψₛ.temperature
    q⁻  = Ψₛ.specific_humidity

    # Shortwave split and longwave emissivities (broadband), on downwelling magnitudes.
    σ = Ψᵣ.σ
    downwelling_shortwave = - Ψᵣ.ℐꜜˢʷ
    downwelling_longwave  = - Ψᵣ.ℐꜜˡʷ
    αˡᵉᵃᶠ = convert(FT, c.leaf_albedo)
    αᵍ    = convert(FT, c.ground_albedo)
    ϵˡᵉᵃᶠ = convert(FT, c.maximum_canopy_emissivity) * (1 - exp(-LAI))
    ϵᵍ    = convert(FT, c.ground_emissivity)
    transmittance = canopy_transmittance(c.extinction, c.clumping, LAI)
    leaf_absorbed_shortwave   = (1 - αˡᵉᵃᶠ) * (1 - transmittance) * downwelling_shortwave
    ground_absorbed_shortwave = transmittance * (1 - αᵍ) * downwelling_shortwave
    effective_albedo = (1 - transmittance) * αˡᵉᵃᶠ + transmittance * αᵍ

    # Thermal node: its conductances are fixed through the skin iteration. The node is the
    # equilibrium mean of its sources blended with the start-of-step node by the memory `m`,
    # so its sensitivity to source `i` is (1 − m) gᵢ/Σg.
    Σgᵀ = gᵃᵀ + gˡᵉᵃᶠᵀ + gᵍᵀ
    mᵀ  = node_memory(Σgᵀ, Cᵀ, Δt)
    ∂Tᵃᶜ∂Tˡᵉᵃᶠ = (1 - mᵀ) * gˡᵉᵃᶠᵀ / max(Σgᵀ, eps(FT))
    ∂Tᵃᶜ∂Tᵍ    = (1 - mᵀ) * gᵍᵀ / max(Σgᵀ, eps(FT))

    Tˡᵉᵃᶠ = Tˡᵃ
    Tᵍ    = Tˡᵃ
    Tᵃᶜ   = T⁻
    qᵃᶜ   = q⁻
    maximum_temperature_step = convert(FT, 25)

    for _ in 1:c.inner_iterations
        gˡᵉᵃᶠᵛ, gʷᵉᵗ, qˡᵉᵃᶠ = leaf_vapor_terms(c, Tˡᵉᵃᶠ, qᵃᶜ, gˡᵇ, Δt, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, transmittance)
        gᵍᵛ, qᵍ, dqᵍ = soil_vapor_terms(c.soil_humidity, Tᵍ, gᵍᵛ⁺, gˡⁱᵗ, Ψₛ, Ψₐ, Ψˡᵃ, ℙₐ)

        Σgᵛ = gᵃᵛ + gˡᵉᵃᶠᵛ + gᵍᵛ
        mᵛ  = node_memory(Σgᵛ, Cᵛ, Δt)
        Tᵃᶜ, _ = canopy_air_node(T⁻, (gᵍᵀ, gˡᵉᵃᶠᵀ, gᵃᵀ), (Tᵍ, Tˡᵉᵃᶠ, θᵃᵗ), mᵀ)
        qᵃᶜ, _ = canopy_air_node(q⁻, (gᵍᵛ, gˡᵉᵃᶠᵛ, gᵃᵛ), (qᵍ, qˡᵉᵃᶠ, qᵃᵗ), mᵛ)
        ∂qᵃᶜ∂qˡᵉᵃᶠ = (1 - mᵛ) * gˡᵉᵃᶠᵛ / max(Σgᵛ, eps(FT))
        ∂qᵃᶜ∂qᵍ    = (1 - mᵛ) * gᵍᵛ / max(Σgᵛ, eps(FT))

        ground_downwelling_longwave, ℐꜛˡʷᵍ = ground_longwave(downwelling_longwave, Tˡᵉᵃᶠ, Tᵍ, ϵˡᵉᵃᶠ, ϵᵍ, σ)
        leaf_net_longwave   = ϵˡᵉᵃᶠ * (downwelling_longwave + ℐꜛˡʷᵍ) - 2 * ϵˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^4
        ground_net_longwave = ϵᵍ * (ground_downwelling_longwave - σ * Tᵍ^4)
        dqˡᵉᵃᶠ = saturation_humidity_slope(ℂᵃᵗ, Tˡᵉᵃᶠ, pᵃᵗ, c.canopy_humidity.phase, qˡᵉᵃᶠ)

        # Leaf and ground energy residuals and their Jacobian, including each balance's
        # dependence on the other skin through the node and the longwave exchange.
        Rˡᵉᵃᶠ = leaf_absorbed_shortwave + leaf_net_longwave - gˡᵉᵃᶠᵀ * (Tˡᵉᵃᶠ - Tᵃᶜ) - ℒ * gˡᵉᵃᶠᵛ * (qˡᵉᵃᶠ - qᵃᶜ)
        Rᵍ    = ground_absorbed_shortwave + ground_net_longwave - gᵍᵀ * (Tᵍ - Tᵃᶜ) - ℒ * gᵍᵛ * (qᵍ - qᵃᶜ) - Λᵍ * (Tᵍ - Tˡᵃ)
        ∂Rˡᵉᵃᶠ∂Tˡᵉᵃᶠ = 4 * ϵˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^3 * (ϵˡᵉᵃᶠ * (1 - ϵᵍ) - 2) - gˡᵉᵃᶠᵀ * (1 - ∂Tᵃᶜ∂Tˡᵉᵃᶠ) - ℒ * gˡᵉᵃᶠᵛ * (1 - ∂qᵃᶜ∂qˡᵉᵃᶠ) * dqˡᵉᵃᶠ
        ∂Rˡᵉᵃᶠ∂Tᵍ = 4 * ϵˡᵉᵃᶠ * ϵᵍ * σ * Tᵍ^3 + gˡᵉᵃᶠᵀ * ∂Tᵃᶜ∂Tᵍ + ℒ * gˡᵉᵃᶠᵛ * ∂qᵃᶜ∂qᵍ * dqᵍ
        ∂Rᵍ∂Tˡᵉᵃᶠ = 4 * ϵᵍ * ϵˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^3 + gᵍᵀ * ∂Tᵃᶜ∂Tˡᵉᵃᶠ + ℒ * gᵍᵛ * ∂qᵃᶜ∂qˡᵉᵃᶠ * dqˡᵉᵃᶠ
        ∂Rᵍ∂Tᵍ = -4 * ϵᵍ * σ * Tᵍ^3 - gᵍᵀ * (1 - ∂Tᵃᶜ∂Tᵍ) - ℒ * gᵍᵛ * (1 - ∂qᵃᶜ∂qᵍ) * dqᵍ - Λᵍ
        # A canopy-free cell (LAI = 0) has no leaf balance: Rˡᵉᵃᶠ = ∂Rˡᵉᵃᶠ/∂Tˡᵉᵃᶠ = 0.
        ∂Rˡᵉᵃᶠ∂Tˡᵉᵃᶠ = ifelse(∂Rˡᵉᵃᶠ∂Tˡᵉᵃᶠ < 0, ∂Rˡᵉᵃᶠ∂Tˡᵉᵃᶠ, -one(FT))
        determinant = ∂Rˡᵉᵃᶠ∂Tˡᵉᵃᶠ * ∂Rᵍ∂Tᵍ - ∂Rˡᵉᵃᶠ∂Tᵍ * ∂Rᵍ∂Tˡᵉᵃᶠ
        ΔTˡᵉᵃᶠ = (∂Rᵍ∂Tᵍ * Rˡᵉᵃᶠ - ∂Rˡᵉᵃᶠ∂Tᵍ * Rᵍ) / determinant
        ΔTᵍ    = (∂Rˡᵉᵃᶠ∂Tˡᵉᵃᶠ * Rᵍ - ∂Rᵍ∂Tˡᵉᵃᶠ * Rˡᵉᵃᶠ) / determinant
        Tˡᵉᵃᶠ -= clamp(ΔTˡᵉᵃᶠ, -maximum_temperature_step, maximum_temperature_step)
        Tᵍ    -= clamp(ΔTᵍ, -maximum_temperature_step, maximum_temperature_step)
    end

    # The node and the flux partition at the final skins.
    gˡᵉᵃᶠᵛ, gʷᵉᵗ, qˡᵉᵃᶠ = leaf_vapor_terms(c, Tˡᵉᵃᶠ, qᵃᶜ, gˡᵇ, Δt, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, transmittance)
    gᵍᵛ, qᵍ, _ = soil_vapor_terms(c.soil_humidity, Tᵍ, gᵍᵛ⁺, gˡⁱᵗ, Ψₛ, Ψₐ, Ψˡᵃ, ℙₐ)
    Σgᵛ = gᵃᵛ + gˡᵉᵃᶠᵛ + gᵍᵛ
    Tᵃᶜ, equilibrium_temperature = canopy_air_node(T⁻, (gᵍᵀ, gˡᵉᵃᶠᵀ, gᵃᵀ), (Tᵍ, Tˡᵉᵃᶠ, θᵃᵗ), mᵀ)
    qᵃᶜ, equilibrium_humidity    = canopy_air_node(q⁻, (gᵍᵛ, gˡᵉᵃᶠᵛ, gᵃᵛ), (qᵍ, qˡᵉᵃᶠ, qᵃᵗ), node_memory(Σgᵛ, Cᵛ, Δt))

    # Effective radiating temperature: σ T⁴ = upwelling longwave above the canopy.
    _, ℐꜛˡʷᵍ = ground_longwave(downwelling_longwave, Tˡᵉᵃᶠ, Tᵍ, ϵˡᵉᵃᶠ, ϵᵍ, σ)
    ℐꜛˡʷ = (1 - ϵˡᵉᵃᶠ) * ℐꜛˡʷᵍ + ϵˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^4
    effective_temperature = ifelse(σ > 0, (ℐꜛˡʷ / σ)^convert(FT, 1//4), Tᵃᶜ)

    𝒬ᵀˡᵉᵃᶠ = gˡᵉᵃᶠᵀ * (Tˡᵉᵃᶠ - Tᵃᶜ)
    𝒬ᵀᵍ    = gᵍᵀ * (Tᵍ - Tᵃᶜ)
    𝒬ᵛˡᵉᵃᶠ = ℒ * gˡᵉᵃᶠᵛ * (qˡᵉᵃᶠ - qᵃᶜ)
    𝒬ᵛᵍ    = ℒ * gᵍᵛ * (qᵍ - qᵃᶜ)
    𝒬ᵍ     = Λᵍ * (Tᵍ - Tˡᵃ)
    Jʷᵉᵗ   = gʷᵉᵗ * (qˡᵉᵃᶠ - qᵃᶜ)

    return (; Tˡᵉᵃᶠ = convert(FT, Tˡᵉᵃᶠ), Tᵍ = convert(FT, Tᵍ),
              Tᵃᶜ = convert(FT, Tᵃᶜ), qᵃᶜ = convert(FT, qᵃᶜ),
              effective_temperature = convert(FT, effective_temperature),
              effective_albedo = convert(FT, effective_albedo),
              𝒬ᵀˡᵉᵃᶠ = convert(FT, 𝒬ᵀˡᵉᵃᶠ), 𝒬ᵀᵍ = convert(FT, 𝒬ᵀᵍ),
              𝒬ᵛˡᵉᵃᶠ = convert(FT, 𝒬ᵛˡᵉᵃᶠ), 𝒬ᵛᵍ = convert(FT, 𝒬ᵛᵍ),
              𝒬ᵍ = convert(FT, 𝒬ᵍ), Jʷᵉᵗ = convert(FT, Jʷᵉᵗ), ℒ = convert(FT, ℒ),
              Σgᵀ = convert(FT, Σgᵀ), Σgᵛ = convert(FT, Σgᵛ),
              equilibrium_temperature = convert(FT, equilibrium_temperature), equilibrium_humidity = convert(FT, equilibrium_humidity),
              Cᵀ = convert(FT, Cᵀ), Cᵛ = convert(FT, Cᵛ))
end

# A diagnostic node is re-solved every iterate; one shared solve returns both node values.
@inline function interface_temperature_and_humidity(c::DiagnosticCanopyGroundSurface, ::CanopyGroundSurface,
                                                    Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₛ, ℙₐ, ℙᵢ)
    solution = canopy_air_space_solve(c, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ)
    return solution.Tᵃᶜ, solution.qᵃᶜ
end

# A prognostic node is model state, frozen through the fixed point.
@inline interface_temperature_and_humidity(::PrognosticCanopyGroundSurface, ::CanopyGroundSurface,
                                           Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₛ, ℙₐ, ℙᵢ) =
    (Ψₛ.temperature, Ψₛ.specific_humidity)

"""
    struct CanopyGroundSurfaceDiagnostics

What a [`CanopyGroundSurface`](@ref) stores in the interface `temperature` slot: the
canopy-air node temperature the atmosphere sees, the leaf, ground, and effective radiating
temperatures, and the two-source flux partition the land and the radiation read.
"""
struct CanopyGroundSurfaceDiagnostics{F}
    interface              :: F   # canopy-air node Tᵃᶜ (what MOST sees)
    canopy                 :: F   # leaf temperature Tˡᵉᵃᶠ
    ground                 :: F   # ground surface temperature Tᵍ
    effective              :: F   # skin (effective radiating) temperature Tᵉᶠᶠ
    effective_albedo       :: F   # broadband shortwave albedo of the canopy + ground column
    ground_heat_flux       :: F   # skin→bulk conduction 𝒬ᵍ
    canopy_latent_heat     :: F   # leaf latent heat 𝒬ᵛˡᵉᵃᶠ (transpiration + wet canopy)
    ground_latent_heat     :: F   # soil evaporation 𝒬ᵛᵍ
    canopy_sensible_heat   :: F   # leaf sensible 𝒬ᵀˡᵉᵃᶠ
    ground_sensible_heat   :: F   # ground sensible 𝒬ᵀᵍ
    canopy_evaporation     :: F   # wet-canopy evaporation Jʷᵉᵗ (kg m⁻² s⁻¹, up)
    wet_canopy_latent_heat :: F   # wet-canopy latent heat ℒ·Jʷᵉᵗ (W m⁻², up)
    land_vapor_flux        :: F   # soil evaporation + transpiration, drawn from the land water store
    transpiration          :: F   # leaf transpiration 𝒬ᵛˡᵉᵃᶠ/ℒ − Jʷᵉᵗ, drawn from the soil
end

CanopyGroundSurfaceDiagnostics(grid) =
    CanopyGroundSurfaceDiagnostics(ntuple(_ -> Field{Center, Center, Nothing}(grid),
                                     Val(fieldcount(CanopyGroundSurfaceDiagnostics)))...)

Adapt.@adapt_structure CanopyGroundSurfaceDiagnostics

Base.summary(::CanopyGroundSurfaceDiagnostics) = "CanopyGroundSurfaceDiagnostics"
Base.show(io::IO, d::CanopyGroundSurfaceDiagnostics) = print(io, summary(d))
