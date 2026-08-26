using DocStringExtensions: TYPEDSIGNATURES
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: interior, location

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
    SellersSoilResistance(FT = Oceananigans.defaults.FloatType;
                          intercept = 8.206, slope = 4.255)

Empirical ground-surface resistance for the moist-soil (vanishing dry layer) evaporation
branch of a [`CanopyAirSpace`](@ref),

```math
rˢ = e^{a - b 𝒮},
```

with the surface saturation `𝒮`, fit by [Sellers et al. (1992)](@cite sellers1992) to the
FIFE prairie flux stations (`rˢ(1) ≈ 52` s m⁻¹). The FIFE sites carried standing dead
grass and plant litter, so the moist-soil end of the fit is an *effective* soil-plus-litter
resistance rather than pore-scale soil physics: chamber measurements of litter-free wet
soil find near-zero surface resistance, and [Sakaguchi and Zeng (2009)](@cite sakaguchi2009)
attribute the moist-soil remainder to the litter layer. Use this fit as a bundled
alternative to an explicit [`LitterResistance`](@ref) (pass `litter_resistance = nothing`);
combining both double-counts the litter effect.

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
rˡ = \\frac{1 - e^{-Lˡ}}{C u★},
```

with the litter area index `Lˡ` (m² m⁻²), a turbulent transfer coefficient `C`, and the
friction velocity `u★` standing in for the wind speed in the canopy air space. Dead
leaves and stems blanket the ground under vegetation; the litter is too porous for
capillary rise to wet its top, so it evaporates little itself while blocking turbulent
and diffusive vapor exchange with the soil below. The resistance vanishes without litter
(`Lˡ = 0`) and saturates at `1/(C u★)` under a thick blanket — 500–1200 s m⁻¹ under
mid-latitude forests, larger in calm air. The default `Lˡ = 1` is the minimum stem-plus-
litter area of the global surface dataset behind the fit; the snow-burial reduction of
the original scheme is omitted (no snow model). Applies to vegetated ground: the bare
tile of a [`TiledLandInterface`](@ref) drops it (see [`bare_canopy_air_space`](@ref)).

```jldoctest
using NumericalEarth

LitterResistance()

# output
LitterResistance(Lˡ=1.0, C=0.004)
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
    string("LitterResistance(Lˡ=", prettysummary(r.litter_area_index),
           ", C=", prettysummary(r.transfer_coefficient), ")")
Base.show(io::IO, r::LitterResistance) = print(io, summary(r))

@inline litter_resistance(::Nothing, u★) = zero(u★)
@inline function litter_resistance(r::LitterResistance, u★)
    FT = typeof(u★)
    Lˡ = convert(FT, r.litter_area_index)
    C  = convert(FT, r.transfer_coefficient)
    return (1 - exp(-Lˡ)) / max(C * u★, eps(FT))
end

#####
##### Canopy-air storage: diagnostic (massless) or prognostic (the node carries the
##### heat and moisture capacity of the canopy air layer).
#####

"""
    DiagnosticCanopyAir()

Massless canopy-air node (the default): `(Tᵃᶜ, qᵃᶜ)` solve the algebraic Kirchhoff
balance between leaf, ground, and atmosphere every iterate of the Monin–Obukhov
fixed point. The node has no memory — turbulence, skins, and canopy air are asserted
to be in mutual equilibrium within each time step.
"""
struct DiagnosticCanopyAir end

Base.summary(::DiagnosticCanopyAir) = "DiagnosticCanopyAir"
Base.show(io::IO, s::DiagnosticCanopyAir) = print(io, summary(s))

"""
    PrognosticCanopyAir(FT = Oceananigans.defaults.FloatType; layer_depth = 10)

Prognostic canopy-air storage: the node `(Tᵃᶜ, qᵃᶜ)` carries the heat and moisture
capacity of the canopy air layer of depth ``h_c`` (`layer_depth`, m — a `Number`, or
a `Field` of canopy heights),

```math
ρ c^p h_c \\frac{dT^{ac}}{dt} = H^v + H^g - H, \\qquad
ρ h_c \\frac{dq^{ac}}{dt} = E^v + E^g - E,
```

integrated with the model time step instead of solved to equilibrium within it. The
node is **frozen while the similarity scales iterate** — the outer fixed point reduces
to the well-conditioned ``u_★ ↔ ζ`` bulk solve — and advanced once per step by an
exact exponential relaxation toward the conductance-weighted equilibrium the
diagnostic node computes today (see [`advance_canopy_air`](@ref)). The atmosphere
receives fluxes evaluated at the step-mean node state, so
``\\mathrm{flux} = \\mathrm{supply} - \\mathrm{storage}`` closes exactly.

`layer_depth → 0` and the first `update_state!` (``Δt = 0``) recover the diagnostic
equilibrium; daytime, where the relaxation time ``τ = ρ (c^p) h_c / Σg`` is shorter
than the time step, the node lands on the equilibrium each step and the diagnostic
physics is unchanged. At calm night the vapor node keeps memory of hours — the
regime where the massless closure has no steady state (the calm-dusk limit cycle).

```jldoctest
using NumericalEarth

PrognosticCanopyAir(layer_depth = 12)

# output
PrognosticCanopyAir(h_c=12.0)
```
"""
struct PrognosticCanopyAir{H}
    layer_depth :: H
end

PrognosticCanopyAir(FT::Type = Oceananigans.defaults.FloatType; layer_depth = 10) =
    PrognosticCanopyAir(layer_depth isa Number ? convert(FT, layer_depth) : layer_depth)

Base.summary(s::PrognosticCanopyAir) =
    string("PrognosticCanopyAir(h_c=", prettysummary(s.layer_depth), ")")
Base.show(io::IO, s::PrognosticCanopyAir) = print(io, summary(s))

Adapt.adapt_structure(to, s::PrognosticCanopyAir) = PrognosticCanopyAir(adapt(to, s.layer_depth))

"""
    struct CanopyAirSpace

Two-source canopy + soil surface with a canopy-air node. Solves the
leaf temperature `Tᵛ`, the soil-skin temperature `Tᵍ`, and the canopy-air node
`(Tᵃᶜ, qᵃᶜ)` inside the Monin–Obukhov fixed point (diagnostic node, the default),
or advances a prognostic node carrying the canopy-air heat and moisture capacity
(`storage = PrognosticCanopyAir(...)`). Use the same object in both the
temperature and specific-humidity interface slots.

Fields:
- `soil`   : the soil vapor branch (a [`DryLayerHumidity`](@ref)).
- `canopy` : the leaf vapor/photosynthesis branch (a [`CanopyConductanceHumidity`](@ref)).
- `soil_skin_flux` : skin↔bulk conduction `Λᵍ = κᵀ/ℓᵀ` (a [`SoilConductiveFlux`](@ref)).
- `leaf_albedo`, `ground_albedo` : broadband shortwave albedos. A `Number`, or a
  `Field{Center, Center, Nothing}` of per-cell values (see [`atmosphere_land_interface`](@ref)).
- `max_canopy_emissivity`, `ground_emissivity` : longwave emissivities (`εᵛ = εᵐᵃˣ(1 − e^{−LAI})`),
  each a `Number` or a per-cell `Field`.
- `extinction`, `clumping` : Beer–Lambert `K`, `Ω` for the shortwave split.
- `leaf_boundary_conductance` : per-leaf boundary-layer conductance `gᵇ` (m s⁻¹) →
  sensible `gˡʰ = ρcₚ·LAI·gᵇ`, vapor `gʷ = ρ·LAI·gᵇ` (in series with the stomata
  when dry, alone over the wetted fraction).
- `undercanopy_conductance` : ground↔canopy-air conductance closure → `gᵍʰ = ρcₚ·gᵘᶜ`;
  a `Number` (m s⁻¹, wrapped as [`ConstantUndercanopyConductance`](@ref)), an
  [`AreaIndexUndercanopyConductance`](@ref) (PALADYN: canopy density and wind), or a
  [`FrictionVelocityUndercanopyConductance`](@ref) (CLM5: canopy density and `u★`).
- `wet_soil_resistance` : soil surface resistance on the moist-soil (vanishing dry layer)
  vapor branch (a [`SellersSoilResistance`](@ref)), or `nothing` (the default: above the
  dry-layer onset the soil itself does not limit evaporation, and the litter layer and
  undercanopy path carry the resistance).
- `litter_resistance` : plant-litter resistance in series on both ground vapor branches
  (a [`LitterResistance`](@ref), the default), or `nothing` for litter-free ground
  ([`bare_canopy_air_space`](@ref) drops it on the bare tile).
- `inner_iterations`, `relaxation` : damped-Newton settings for the coupled solve.
- `interception` : wet-canopy vapor branch parameters (a [`CanopyInterception`](@ref)),
  or `nothing` for a dry canopy (the default; recovers the current CAS bit-for-bit).
- `phase` : saturation phase (Liquid).
- `storage` : the canopy-air node storage — [`DiagnosticCanopyAir`](@ref) (the default,
  massless node) or [`PrognosticCanopyAir`](@ref) (the node carries the canopy-air
  heat and moisture capacity and is advanced with the model time step).

The four optics slots accept a per-cell `Field{Center, Center, Nothing}` alongside a
`Number`, so a satellite albedo product reaches the two-source radiation balance cell by
cell. The land flux kernel localizes them before the canopy solve; the interface
constructor checks them with [`validate_canopy_optics`](@ref).

```jldoctest
using NumericalEarth
using Oceananigans
using NumericalEarth.EarthSystemModels.InterfaceComputations: local_interface_formulation

grid = LatitudeLongitudeGrid(size = (2, 1, 1), latitude = (10, 11), longitude = (10, 12),
                             z = (-1, 0), topology = (Bounded, Bounded, Bounded))

soil = DryLayerHumidity(Float64;
                        dry_layer_depth = StorageBasedDryLayerDepth(Float64;
                            maximum_dry_layer_depth = 0.015, dry_layer_onset_saturation = 0.5),
                        vapor_exchange = DryLayerVaporPistonVelocity(Float64;
                            minimum_dry_layer_depth = 1e-3, molecular_diffusivity = 2.4e-5),
                        thermal_exchange_depth = 0.05, porosity = 0.4)

leaf_albedo = Field{Center, Center, Nothing}(grid)
set!(leaf_albedo, (λ, φ) -> ifelse(λ < 11, 0.12, 0.35))

canopy = CanopyAirSpace(Float64; soil, leaf_albedo)

local_interface_formulation(canopy, 2, 1).leaf_albedo

# output
0.35
```
"""
struct CanopyAirSpace{S, C, RF, LA, GA, CE, GE, FT, U, W, L, I, Φ, ST}
    soil                      :: S
    canopy                    :: C
    soil_skin_flux             :: RF
    leaf_albedo               :: LA
    ground_albedo             :: GA
    max_canopy_emissivity     :: CE
    ground_emissivity         :: GE
    extinction                :: FT
    clumping                  :: FT
    leaf_boundary_conductance :: FT
    undercanopy_conductance   :: U
    wet_soil_resistance       :: W
    litter_resistance         :: L
    inner_iterations          :: Int
    relaxation                :: FT
    interception              :: I
    phase                     :: Φ
    storage                   :: ST
end

function CanopyAirSpace(FT=Oceananigans.defaults.FloatType;
                        soil,
                        canopy                    = CanopyConductanceHumidity(FT),
                        soil_skin_flux             = SoilConductiveFlux(1.5, 0.05),
                        leaf_albedo               = 0.15,
                        ground_albedo             = 0.15,
                        max_canopy_emissivity     = 0.98,
                        ground_emissivity         = 0.96,
                        extinction                = 0.5,
                        clumping                  = 1,
                        leaf_boundary_conductance = 0.02,
                        undercanopy_conductance   = 0.013,
                        wet_soil_resistance       = nothing,
                        litter_resistance         = LitterResistance(FT),
                        inner_iterations          = 40,
                        relaxation                = 1//2,
                        interception              = nothing,
                        phase                     = AtmosphericThermodynamics.Liquid(),
                        storage                   = DiagnosticCanopyAir())

    # Convert only a scalar optics slot; a `Field` passes through and is materialized
    # per cell in the flux kernel.
    return CanopyAirSpace(soil, canopy, soil_skin_flux,
                          convert_if_number(FT, leaf_albedo),
                          convert_if_number(FT, ground_albedo),
                          convert_if_number(FT, max_canopy_emissivity),
                          convert_if_number(FT, ground_emissivity),
                          convert(FT, extinction), convert(FT, clumping),
                          convert(FT, leaf_boundary_conductance),
                          undercanopy_conductance_model(undercanopy_conductance, FT),
                          wet_soil_resistance, litter_resistance,
                          inner_iterations, convert(FT, relaxation), interception, phase,
                          storage)
end

# The storage type selects the node treatment inside `canopy_air_space_solve` and
# the kernel data flow (frozen node + advance for the prognostic variant).
const DiagnosticCanopyAirSpace = CanopyAirSpace{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                <:DiagnosticCanopyAir}
const PrognosticCanopyAirSpace = CanopyAirSpace{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                <:PrognosticCanopyAir}

#####
##### Per-cell optics: localization (kernel) and validation (setup)
#####

# Collapse `Field`-valued optics slots to cell (i, j) before the index-free canopy solve;
# `state2dindex` returns a `Number` slot unchanged.
@inline local_interface_formulation(formulation, i, j) = formulation

@inline local_interface_formulation(c::CanopyAirSpace, i, j) =
    CanopyAirSpace(c.soil, c.canopy, c.soil_skin_flux,
                   state2dindex(c.leaf_albedo, i, j),
                   state2dindex(c.ground_albedo, i, j),
                   state2dindex(c.max_canopy_emissivity, i, j),
                   state2dindex(c.ground_emissivity, i, j),
                   c.extinction, c.clumping, c.leaf_boundary_conductance,
                   c.undercanopy_conductance, c.wet_soil_resistance, c.litter_resistance,
                   c.inner_iterations, c.relaxation, c.interception, c.phase, c.storage)

@inline evaluable_albedo(α) = isfinite(α) & (α ≥ 0) & (α < 1)
@inline evaluable_emissivity(ε) = isfinite(ε) & (ε > 0) & (ε ≤ 1)

# The four slots that may carry per-cell values, each with the predicate its values must
# satisfy and the requirement to quote when they do not.
@inline canopy_optics_slots(c::CanopyAirSpace) =
    (("leaf_albedo",           c.leaf_albedo,           evaluable_albedo,     "in [0, 1)"),
     ("ground_albedo",         c.ground_albedo,         evaluable_albedo,     "in [0, 1)"),
     ("max_canopy_emissivity", c.max_canopy_emissivity, evaluable_emissivity, "in (0, 1]"),
     ("ground_emissivity",     c.ground_emissivity,     evaluable_emissivity, "in (0, 1]"))

# Setup-time only, so the host copy is fine.
optics_slot_values(slot::Number) = (slot,)
optics_slot_values(slot::AbstractField) = Array(interior(slot))

# `state2dindex` reads `slot[i, j, 1]`, so a slot must be a horizontally-reduced field on
# *this* grid: a `(Center, Center, Center)` field would silently contribute its deepest
# level, and a field from another grid would read the wrong cell — the kernel reads it
# `@inbounds`.
function validate_optics_slot_layout(slot::AbstractField, name, grid)
    if location(slot) !== (Center, Center, Nothing)
        LX, LY, LZ = location(slot)
        throw(ArgumentError("$name must be a horizontally-reduced Field at " *
                            "(Center, Center, Nothing), got ($LX, $LY, $LZ). " *
                            "Build it with Field{Center, Center, Nothing}(grid)."))
    end

    # Grids compare with `==` (topology and nodes): two references to one grid are not
    # guaranteed to be egal, so identity would reject a field built on the interface's grid.
    if architecture(slot) !== architecture(grid) || slot.grid != grid
        throw(ArgumentError("$name is built on a different grid than the interface " *
                            "($(summary(slot.grid)) vs $(summary(grid))). Per-cell canopy " *
                            "optics must be built on the grid the interface is constructed " *
                            "with, or they index the wrong cell."))
    end

    return nothing
end

validate_optics_slot_layout(slot::Number, name, grid) = nothing

# A bare array carries no location or grid, so nothing pins its rows and columns to the
# exchange grid's cells; require a Field so the layout above can be checked. Anything else
# has no per-cell values for `optics_slot_values` to read.
validate_optics_slot_layout(slot, name, grid) =
    throw(ArgumentError("$name must be a Number or a Field, got a $(summary(slot)). " *
                        "Wrap per-cell values in a Field{Center, Center, Nothing}(grid)."))

"""
$(TYPEDSIGNATURES)

Check that the optics slots of `formulation` can be localized on `grid` and evaluated by
the canopy radiation balance, and throw an `ArgumentError` naming the offending slot
otherwise.

Kernels can neither throw nor report, so the conditions the two-source radiation balance
depends on are checked here: a `Field` slot is horizontally reduced and sized to `grid`,
every albedo lies in `[0, 1)` so the absorbed shortwave `(1 - α) SW` is a real fraction,
and every emissivity lies in `(0, 1]` so the longwave balance stays invertible. A gap in
a satellite albedo product would otherwise propagate `NaN` through the leaf and ground
skin temperatures into the coupled state.
"""
function validate_canopy_optics(c::CanopyAirSpace, grid)
    for (name, slot, evaluable, requirement) in canopy_optics_slots(c)
        validate_optics_slot_layout(slot, name, grid)

        values = optics_slot_values(slot)
        all(evaluable, values) && continue

        throw(ArgumentError("$name has $(count(!evaluable, values)) cells that are not " *
                            "$requirement (minimum $(minimum(values)), maximum " *
                            "$(maximum(values))). A bad cell propagates NaN through the " *
                            "leaf and ground skin temperatures into the coupled state. " *
                            "Gap-fill the field first."))
    end

    return nothing
end

validate_canopy_optics(formulation, grid) = nothing

Base.summary(::CanopyAirSpace) = "CanopyAirSpace"
Base.show(io::IO, c::CanopyAirSpace) =
    print(io, "CanopyAirSpace(soil=", summary(c.soil), ", canopy=", summary(c.canopy),
          ", storage=", summary(c.storage), ")")

# `local_interface_formulation` rebuilds this struct positionally inside a kernel, where a
# name-keyed rebuild would not be type-stable, so a reordered or inserted field would
# silently mis-wire the closure rather than fail to compile. `test_canopy_air_space.jl`
# pins the field order.
Adapt.@adapt_structure CanopyAirSpace

# Materialization / identity — delegate to the sub-models so the per-cell interface
# state carries the soil saturation, bulk temperature, and LAI the branches read.
@inline interface_phase(c::CanopyAirSpace) = interface_phase(c.soil)
# The soil branch always publishes the saturation 𝒮 and the canopy branch its stress
# state; a canopy with interception additionally pulls the prognostic canopy water
# store Wᶜ (→ fʷ).
@inline interface_hydrology_state(i, j, grid, c::CanopyAirSpace, land_state) =
    canopy_air_space_hydrology_state(c.interception, i, j, grid, c, land_state)
@inline requires_retention_curve(c::CanopyAirSpace) =
    requires_retention_curve(c.soil) || requires_retention_curve(c.canopy)
@inline canopy_air_space_hydrology_state(::Nothing, i, j, grid, c, land_state) =
    merge(interface_hydrology_state(i, j, grid, c.soil, land_state),
          interface_hydrology_state(i, j, grid, c.canopy, land_state))
@inline canopy_air_space_hydrology_state(::CanopyInterception, i, j, grid, c, land_state) =
    merge(interface_hydrology_state(i, j, grid, c.soil, land_state),
          interface_hydrology_state(i, j, grid, c.canopy, land_state),
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

# Leaf vapor conductance `gˡʷ` and leaf-saturated humidity `qᵛ` at the leaf temperature `Tᵛ`.
@inline function leaf_vapor_terms(canopy, Tᵛ, gʷ, fʷ, Ψₛ, Ψₐ, Ψᵣ, ℙₐ, ftrans)
    gᶜ, qᵛ = canopy_conductance_terms(canopy, Tᵛ, Ψₛ, Ψₐ, Ψᵣ, ℙₐ, ftrans)
    return leaf_vapor_conductance(gᶜ, gʷ, fʷ), qᵛ
end

# Soil vapor conductance and front humidity at the soil-skin temperature `Tᵍ`: the dry-layer
# branch (front qᵉ through Gᵉ, in series with the litter + undercanopy path `gᵖ`; Sakaguchi &
# Zeng 2009, Eq. 18b, matching the sensible path which already crosses gᵘᶜ via gᵍʰ) blended
# with the saturated-skin wet branch (qᵍ⁺ through gᵍʷ), weight `fᵈ` from the soil model.
@inline function soil_vapor_terms(soil, Tᵍ, gᵍʷ, gᵖ, Ψₛ, Ψₐ, ℙₐ)
    Gᵉ, qᵉ, fᵈ, qᵍ⁺ = dry_layer_terms(soil, Tᵍ, Ψₛ, Ψₐ, ℙₐ)
    Gᵉ = Gᵉ * gᵖ / (gᵖ + Gᵉ)
    Gᵉ⁺ = fᵈ * Gᵉ + (1 - fᵈ) * gᵍʷ
    qᵉ⁺ = ifelse(Gᵉ⁺ > eps(eltype(Ψₛ)), (fᵈ * Gᵉ * qᵉ + (1 - fᵈ) * gᵍʷ * qᵍ⁺) / Gᵉ⁺, qᵍ⁺)
    return Gᵉ⁺, qᵉ⁺
end

# Kirchhoff node (as the humidity node in CompositeSurfaceHumidity): the ground branch
# `(gᵍ, xᵍ)` and the leaf branch `(gᵛ, xᵛ)` in parallel behind the aerodynamic branch
# `(gᵃ, xᵃᵗ)` — a conductance-weighted mean, within the hull of its sources by
# construction. The all-decoupled corner (every conductance zero) keeps the previous
# iterate `x⁻`.
@inline function canopy_air_node(gᵍ, xᵍ, gᵛ, xᵛ, gᵃ, xᵃᵗ, x⁻)
    D = gᵍ + gᵛ + gᵃ
    return ifelse(D > 0, (gᵍ * xᵍ + gᵛ * xᵛ + gᵃ * xᵃᵗ) / D, x⁻)
end

# Node treatment per storage: the diagnostic node re-balances every iterate; the
# prognostic node is model state, frozen through the solve (the skins equilibrate
# against it, and the node advances once per step in `advance_interface_state!`).
@inline node_value(::DiagnosticCanopyAir, gᵍ, xᵍ, gᵛ, xᵛ, gᵃ, xᵃᵗ, x⁻) =
    canopy_air_node(gᵍ, xᵍ, gᵛ, xᵛ, gᵃ, xᵃᵗ, x⁻)
@inline node_value(::PrognosticCanopyAir, gᵍ, xᵍ, gᵛ, xᵛ, gᵃ, xᵃᵗ, x⁻) = x⁻

"""
    advance_canopy_air(x, x_eq, Σg, C, Δt)

Advance a prognostic canopy-air node value `x` toward its conductance-weighted
equilibrium `x_eq` over `Δt` — the exact solution of the linearized node ODE
``C \\, dx/dt = Σg (x_{eq} - x)`` at fixed conductances,

```math
x ← x_{eq} + (x - x_{eq}) e^{-Δt Σg / C}.
```

A convex blend of the old node and its equilibrium: unconditionally stable and
hull-bounded at any `Δt`. `Δt = 0` (the first `update_state!`, before any time
step) or `C = 0` return the equilibrium — the diagnostic limit.
"""
@inline function advance_canopy_air(x, x_eq, Σg, C, Δt)
    FT = typeof(x)
    w = ifelse((Δt > 0) & (C > 0), exp(-Δt * Σg / max(C, eps(FT))), zero(FT))
    return x_eq + (x - x_eq) * w
end

# Time-mean node value over the step, ⟨x⟩ = x_eq + (x₀ − x_eq) (τ/Δt)(1 − e^{−Δt/τ})
# with τ = C/Σg. Fluxes are linear in the node, so evaluating them here closes the
# step budget exactly: flux to the atmosphere = Kirchhoff supply − storage tendency.
# `expm1` keeps the τ ≫ Δt limit accurate; Δt = 0 or C = 0 return the equilibrium.
@inline function step_mean_canopy_air(x, x_eq, Σg, C, Δt)
    FT = typeof(x)
    τ = C / max(Σg, eps(FT))
    m = -(τ / max(Δt, eps(FT))) * expm1(-Δt / τ)
    return ifelse((Δt > 0) & (C > 0), x_eq + (x - x_eq) * m, x_eq)
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
is the Kirchhoff conductance-weighted mean of its sources, with the aerodynamic
branch read off the previous iterate's transfer coefficients (`ρ cᵖ u★ χθ`,
`ρ u★ χq`), so it stays within the hull of its source states by construction. On
exit the node is re-solved against the final skins so the returned flux partition
closes.
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

    # Aerodynamic conductances from the previous outer iterate's similarity solution
    # (held fixed through the inner loop): heat `gᵃʰ = ρ cᵖ u★ χθ` and vapor
    # `gᵃʷ = ρ u★ χq`, floored at zero against transient unphysical profiles.
    u★  = Ψₛ.fluxes.u★
    gᵃʰ = max(0, ρᵃᵗ * cᵖ * u★ * Ψₛ.fluxes.χθ)
    gᵃʷ = max(0, ρᵃᵗ * u★ * Ψₛ.fluxes.χq)

    # Land surface velocities are zero, so the surface wind speed is the atmospheric one.
    Vₐ  = sqrt(Ψₐ.u^2 + Ψₐ.v^2)
    gᵘᶜ = undercanopy_conductance(c.undercanopy_conductance, LAI, Vₐ, u★)

    gˡʰ = ρᵃᵗ * cᵖ * LAI * c.leaf_boundary_conductance
    gᵍʰ = ρᵃᵗ * cᵖ * gᵘᶜ
    # Ground vapor path (Sakaguchi & Zeng 2009, Eq. 18): the litter resistance rˡ and — on
    # the moist-soil branch — the soil surface resistance rˢ sit in series with the
    # undercanopy aerodynamic path, so a vanishing dry layer does not evaporate like open
    # water. The dry branch crosses the same litter + undercanopy path `gᵖ`.
    rˡ  = litter_resistance(c.litter_resistance, Ψₛ.fluxes.u★)
    rˢ  = soil_surface_resistance(c.wet_soil_resistance, Ψₛ.hydrology.saturation)
    gᵖ  = ρᵃᵗ * gᵘᶜ / (1 + gᵘᶜ * rˡ)
    gᵍʷ = ρᵃᵗ * gᵘᶜ / (1 + gᵘᶜ * (rˡ + rˢ))
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
    αˡᶠ = convert(FT, c.leaf_albedo)
    αᵍ  = convert(FT, c.ground_albedo)
    εᵛ = convert(FT, c.max_canopy_emissivity) * (1 - exp(-LAI))
    εᵍ = convert(FT, c.ground_emissivity)
    ftrans    = exp(-c.extinction * LAI * c.clumping)
    SWᵛ = (1 - αˡᶠ) * (1 - ftrans) * SW
    SWᵍ = ftrans * (1 - αᵍ) * SW

    # The column's shortwave albedo: the complement of what the two-source split absorbs.
    αeff = 1 - ((1 - αˡᶠ) * (1 - ftrans) + ftrans * (1 - αᵍ))

    Tᵛ  = Tˡᵃ
    Tᵍ = Tˡᵃ
    Tᵃᶜ = Ψₛ.temperature
    qᵃᶜ = Ψₛ.specific_humidity
    relaxation  = c.relaxation
    max_temperature_step = convert(FT, 25)   # damped-Newton trust region, per iterate
    Tₗₒ, Tₕᵢ = convert(FT, 180), convert(FT, 340)  # physical band; guards qˢᵃᵗ against transient overshoot
    tiny = eps(FT)

    for _ in 1:c.inner_iterations
        gˡʷ, qᵛ = leaf_vapor_terms(c.canopy, Tᵛ, gʷ, fʷ, Ψₛ, Ψₐ, Ψᵣ, ℙₐ, ftrans)
        Gᵉ, qᵉ  = soil_vapor_terms(c.soil, Tᵍ, gᵍʷ, gᵖ, Ψₛ, Ψₐ, ℙₐ)

        Tᵃᶜ = node_value(c.storage, gᵍʰ, Tᵍ, gˡʰ, Tᵛ, gᵃʰ, θᵃᵗ, Tᵃᶜ)
        qᵃᶜ = node_value(c.storage, Gᵉ, qᵉ, gˡʷ, qᵛ, gᵃʷ, qᵃᵗ, qᵃᶜ)

        LWꜜᵍ     = (1 - εᵛ) * LWd + εᵛ * σ * Tᵛ^4
        LWꜛᵍ     = εᵍ * σ * Tᵍ^4 + (1 - εᵍ) * LWꜜᵍ
        LWᵛ = εᵛ * (LWd + LWꜛᵍ) - 2 * εᵛ * σ * Tᵛ^4
        LWᵍ = εᵍ * (LWꜜᵍ - σ * Tᵍ^4)

        Rᵥ   = SWᵛ + LWᵛ
        resᵥ = Rᵥ - gˡʰ * (Tᵛ - Tᵃᶜ) - ℒ * gˡʷ * (qᵛ - qᵃᶜ)
        dRᵥ  = -8 * εᵛ * σ * Tᵛ^3 - gˡʰ - ℒ * gˡʷ * saturation_humidity_slope(ℂᵃᵗ, Tᵛ, pᵃᵗ, c.phase)
        Tᵛ   = ifelse(abs(dRᵥ) < tiny, Tᵃᶜ, Tᵛ - clamp(relaxation * resᵥ / dRᵥ, -max_temperature_step, max_temperature_step))
        Tᵛ   = clamp(Tᵛ, Tₗₒ, Tₕᵢ)

        Rᵍ   = SWᵍ + LWᵍ
        resᵍ = Rᵍ - gᵍʰ * (Tᵍ - Tᵃᶜ) - ℒ * Gᵉ * (qᵉ - qᵃᶜ) - Λ * (Tᵍ - Tˡᵃ)
        dRᵍ  = -4 * εᵍ * σ * Tᵍ^3 - gᵍʰ - Λ - ℒ * Gᵉ * saturation_humidity_slope(ℂᵃᵗ, Tᵍ, pᵃᵗ, c.phase)
        Tᵍ  = Tᵍ - clamp(relaxation * resᵍ / dRᵍ, -max_temperature_step, max_temperature_step)
        Tᵍ  = clamp(Tᵍ, Tₗₒ, Tₕᵢ)
    end

    # Converged diagnostics: per-surface flux shares, the skin→slab conduction, and
    # the effective radiating (LST) temperature σ Teff⁴ ≡ LWu (upwelling to space).
    # The diagnostic node is re-solved against the final skins: inside the loop it
    # updates ahead of them, so the loop exits one iterate stale and the shares below
    # would miss closure. The prognostic node stays frozen.
    gˡʷ, qᵛ = leaf_vapor_terms(c.canopy, Tᵛ, gʷ, fʷ, Ψₛ, Ψₐ, Ψᵣ, ℙₐ, ftrans)
    Gᵉ, qᵉ  = soil_vapor_terms(c.soil, Tᵍ, gᵍʷ, gᵖ, Ψₛ, Ψₐ, ℙₐ)
    Tᵃᶜ = node_value(c.storage, gᵍʰ, Tᵍ, gˡʰ, Tᵛ, gᵃʰ, θᵃᵗ, Tᵃᶜ)
    qᵃᶜ = node_value(c.storage, Gᵉ, qᵉ, gˡʷ, qᵛ, gᵃʷ, qᵃᵗ, qᵃᶜ)

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

    # Node balance ingredients for the prognostic advance: conductance sums per side
    # and the equilibria the node relaxes toward (the diagnostic node's own values).
    Σgᵀ = gᵃʰ + gˡʰ + gᵍʰ
    Σgᵛ = gᵃʷ + gˡʷ + Gᵉ
    T_eq = canopy_air_node(gᵍʰ, Tᵍ, gˡʰ, Tᵛ, gᵃʰ, θᵃᵗ, Tᵃᶜ)
    q_eq = canopy_air_node(Gᵉ, qᵉ, gˡʷ, qᵛ, gᵃʷ, qᵃᵗ, qᵃᶜ)

    return (; Tᵛ = convert(FT, Tᵛ), Tᵍ = convert(FT, Tᵍ),
              Tᵃᶜ = convert(FT, Tᵃᶜ), qᵃᶜ = convert(FT, qᵃᶜ),
              Teff = convert(FT, Teff), αeff = convert(FT, αeff),
              Hᵛ = convert(FT, Hᵛ), Hᵍ = convert(FT, Hᵍ),
              LEᵛ = convert(FT, LEᵛ), LEᵍ = convert(FT, LEᵍ),
              Gᶜ = convert(FT, Gᶜ), Eʷ = convert(FT, Eʷ),
              LEʷ = convert(FT, LEʷ),
              Σgᵀ = convert(FT, Σgᵀ), Σgᵛ = convert(FT, Σgᵛ),
              T_eq = convert(FT, T_eq), q_eq = convert(FT, q_eq),
              gˡʰ = convert(FT, gˡʰ), gᵍʰ = convert(FT, gᵍʰ),
              gˡʷ = convert(FT, gˡʷ), Gᵉ = convert(FT, Gᵉ),
              qᵛ = convert(FT, qᵛ), qᵉ = convert(FT, qᵉ),
              ρᵃᵗ = convert(FT, ρᵃᵗ), cᵖ = convert(FT, cᵖ), ℒ = convert(FT, ℒ))
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

# Prognostic node: the node is model state, frozen through the outer fixed point —
# the similarity scales iterate against fixed surface values (the well-conditioned
# u★ ↔ ζ bulk solve), and the skins are solved once per step at exit
# (`advance_interface_state!`) rather than every iterate.
@inline interface_temperature_and_humidity(::PrognosticCanopyAirSpace, ::CanopyAirSpace,
                                           Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₛ, ℙₐ, ℙᵢ) =
    (Ψₛ.temperature, Ψₛ.specific_humidity)

"""
    struct CanopyAirState

Prognostic canopy-air node state of a [`PrognosticCanopyAir`](@ref) storage: the node
temperature `Tᵃᶜ` and specific humidity `qᵃᶜ` carried across time steps. Model state
(unlike the pure-output [`CanopyAirSpaceDiagnostics`](@ref) fields): it enters
checkpointing and is read back as the interface state at the start of each flux
computation. Initialized to the diagnostic equilibrium on the first `update_state!`
(`Δt = 0`), so a fresh start degrades gracefully to a diagnostic first step.
"""
struct CanopyAirState{F}
    temperature       :: F   # Tᵃᶜ (Field{Center, Center, Nothing})
    specific_humidity :: F   # qᵃᶜ
end

CanopyAirState(grid) = CanopyAirState(Field{Center, Center, Nothing}(grid),
                                      Field{Center, Center, Nothing}(grid))

Adapt.@adapt_structure CanopyAirState

Base.summary(::CanopyAirState) = "CanopyAirState"
Base.show(io::IO, s::CanopyAirState) = print(io, summary(s))

@inline build_canopy_air_state(::DiagnosticCanopyAir, grid) = nothing
@inline build_canopy_air_state(::PrognosticCanopyAir, grid) = CanopyAirState(grid)

"""
    struct CanopyAirSpaceDiagnostics

The atmosphere-facing `temperature` slot of a [`CanopyAirSpace`](@ref) interface: the
canopy-air node temperature the atmosphere sees, the per-source diagnostic temperatures,
and the two-source flux shares of the coupled solve. Downstream consumers dispatch on
this type — it signals that radiation is internalized in the soil-skin balance and the
slab is driven by the skin→bulk conduction (`ground_heat_flux`) rather than by a
separately added radiative flux.
"""
struct CanopyAirSpaceDiagnostics{F, S}
    interface              :: F   # canopy-air node Tᵃᶜ (what MOST sees)
    canopy                 :: F   # leaf temperature Tᵛ
    soil_skin              :: F   # soil-skin temperature Tᵍ
    effective              :: F   # radiating (LST) temperature Teff
    effective_albedo       :: F   # broadband shortwave albedo of the canopy + ground column
    ground_heat_flux       :: F   # skin→bulk conduction Gᶜ
    canopy_latent_heat     :: F   # leaf transpiration LEᵛ
    soil_latent_heat       :: F   # soil evaporation LEᵍ
    canopy_sensible_heat   :: F   # leaf sensible Hᵛ
    soil_sensible_heat     :: F   # ground sensible Hᵍ
    canopy_evaporation     :: F   # wet-canopy evaporation Eʷ (kg m⁻² s⁻¹, up)
    canopy_wet_latent_heat :: F   # wet-canopy latent heat ℒ·Eʷ (W m⁻², up)
    state                  :: S   # prognostic CanopyAirState, or nothing (diagnostic node)
end

CanopyAirSpaceDiagnostics(grid, storage = DiagnosticCanopyAir()) =
    CanopyAirSpaceDiagnostics(ntuple(_ -> Field{Center, Center, Nothing}(grid),
                                     Val(fieldcount(CanopyAirSpaceDiagnostics) - 1))...,
                              build_canopy_air_state(storage, grid))

Adapt.@adapt_structure CanopyAirSpaceDiagnostics

Base.summary(::CanopyAirSpaceDiagnostics) = "CanopyAirSpaceDiagnostics"
Base.show(io::IO, d::CanopyAirSpaceDiagnostics) = print(io, summary(d))
