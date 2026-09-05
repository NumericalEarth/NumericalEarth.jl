#####
##### `CanopyAirSpace` — a two-source canopy (leaf + shaded soil skin) exchanging with a
##### canopy-air node that drains to the atmosphere through the aerodynamic conductance.
#####

"""
    struct CanopyInterception

Marker enabling the wet-canopy (interception) vapor branch of a [`CanopyAirSpace`](@ref). A wet
canopy evaporates intercepted water at the *potential* (stomata-free) rate through
the leaf boundary layer only, so the leaf vapor conductance blends the dry path
(stomata in series with the boundary layer) with the wet `gˡᵇ = ρᵃᵗ · LAI · gᵇ`
by the wet fraction

```math
f_{wet} = (Wᶜ / Wᶜᵐᵃˣ)^{2/3}, \\qquad Wᶜᵐᵃˣ = c · LAI
```

([Deardorff, 1978](@cite deardorff1978)). The store `Wᶜ` and its capacity `Wᶜᵐᵃˣ = c·LAI`
are owned by the [`InterceptingHydrology`](@ref) wrapping the soil; the interface reads both
and normalizes `fʷᵉᵗ` by the store's *own* capacity. The leaf boundary conductance `gᵇ` is the
`leaf_boundary_conductance` on the [`CanopyAirSpace`](@ref).
"""
struct CanopyInterception end

Base.summary(::CanopyInterception) = "CanopyInterception"

#####
##### Undercanopy conductance closures — the ground ↔ canopy-air sensible/vapor coupling.
#####

"""
    AreaIndexUndercanopyConductance(FT = Oceananigans.defaults.FloatType;
                                    drag_coefficient = 0.006,
                                    stem_area_index = 0)

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
the canopy air faster than the canopy air ventilates to the atmosphere.

```jldoctest
using NumericalEarth

AreaIndexUndercanopyConductance()

# output
AreaIndexUndercanopyConductance(C=0.006, SAI=0.0)
```
"""
struct AreaIndexUndercanopyConductance{FT}
    drag_coefficient :: FT
    stem_area_index  :: FT
end

AreaIndexUndercanopyConductance(FT::Type = Oceananigans.defaults.FloatType;
                                drag_coefficient = 0.006,
                                stem_area_index = 0) =
    AreaIndexUndercanopyConductance(convert(FT, drag_coefficient),
                                    convert(FT, stem_area_index))

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
struct FrictionVelocityUndercanopyConductance{FT}
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

Base.summary(u::AreaIndexUndercanopyConductance) =
    string("AreaIndexUndercanopyConductance(C=", prettysummary(u.drag_coefficient),
           ", SAI=", prettysummary(u.stem_area_index), ")")
Base.show(io::IO, u::AreaIndexUndercanopyConductance) = print(io, summary(u))

Base.summary(u::FrictionVelocityUndercanopyConductance) =
    string("FrictionVelocityUndercanopyConductance(Cₛᵈ=", prettysummary(u.dense_canopy_coefficient),
           ", z₀ᵍ=", prettysummary(u.ground_roughness_length),
           ", SAI=", prettysummary(u.stem_area_index), ")")
Base.show(io::IO, u::FrictionVelocityUndercanopyConductance) = print(io, summary(u))

# A `Number` is a constant conductance (m s⁻¹).
@inline undercanopy_conductance(gᵘᶜ::Number, LAI, Vₐ, u★) = convert(typeof(LAI), gᵘᶜ)

@inline function undercanopy_conductance(u::AreaIndexUndercanopyConductance, LAI, Vₐ, u★)
    FT = typeof(LAI)
    C  = convert(FT, u.drag_coefficient)
    Λ  = LAI + convert(FT, u.stem_area_index)
    gᵘ = C * Vₐ / max(1 - exp(-Λ), eps(FT))
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

"""
    SellersSoilResistance(FT = Oceananigans.defaults.FloatType;
                          intercept = 8.206, slope = 4.255)

Empirical ground-surface resistance for the moist-soil (vanishing dry layer) evaporation
branch of a [`CanopyAirSpace`](@ref),

```math
rˢ = e^{a - b 𝒮},
```

with the surface saturation `𝒮`, fit by [Sellers et al. (1992)](@cite sellers1992) to the
FIFE prairie flux stations (`rˢ(1) ≈ 52` s m⁻¹). Those sites carried litter, so the fit is
an effective soil-plus-litter resistance: use it as an alternative to an explicit
[`LitterResistance`](@ref) (pass `litter_resistance = nothing`), not alongside one.

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
the original scheme is omitted (no snow model).

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

Massless canopy-air node (the default): `(Tᵃᶜ, qᵃᶜ)` is the conductance-weighted mean
of the leaf, ground, and atmosphere states every iterate of the Monin–Obukhov fixed point.
"""
struct DiagnosticCanopyAir end

Base.summary(::DiagnosticCanopyAir) = "DiagnosticCanopyAir"
Base.show(io::IO, s::DiagnosticCanopyAir) = print(io, summary(s))

"""
    PrognosticCanopyAir(FT = Oceananigans.defaults.FloatType; layer_depth = 10)

Prognostic canopy-air storage: the node `(Tᵃᶜ, qᵃᶜ)` carries the heat and moisture
capacity of the canopy air layer of depth ``hᶜ`` (`layer_depth`, m — a `Number`, or
a `Field` of canopy heights),

```math
ρ c^p h^c \\frac{dT^{ac}}{dt} = H^v + H^g - H, \\qquad
ρ h^c \\frac{dq^{ac}}{dt} = E^v + E^g - E,
```

integrated with the model time step. The node is frozen while the similarity scales
iterate and advanced once per step by the exact exponential relaxation toward the
conductance-weighted equilibrium of its sources; the skins are solved, and the fluxes
exported, at the step-mean node. `layer_depth → 0` recovers the massless node.

```jldoctest
using NumericalEarth

PrognosticCanopyAir(layer_depth = 12)

# output
PrognosticCanopyAir(hᶜ=12.0)
```
"""
struct PrognosticCanopyAir{H}
    layer_depth :: H
end

PrognosticCanopyAir(FT::Type = Oceananigans.defaults.FloatType; layer_depth = 10) =
    PrognosticCanopyAir(layer_depth isa Number ? convert(FT, layer_depth) : layer_depth)

Base.summary(s::PrognosticCanopyAir) =
    string("PrognosticCanopyAir(hᶜ=", prettysummary(s.layer_depth), ")")
Base.show(io::IO, s::PrognosticCanopyAir) = print(io, summary(s))

Adapt.adapt_structure(to, s::PrognosticCanopyAir) = PrognosticCanopyAir(Adapt.adapt(to, s.layer_depth))

"""
    struct CanopyAirSpace

Two-source canopy + soil surface with a canopy-air node. Solves the
leaf temperature `Tˡᵉᵃᶠ`, the soil-skin temperature `Tᵍ`, and the canopy-air node
`(Tᵃᶜ, qᵃᶜ)` inside the Monin–Obukhov fixed point (diagnostic node, the default),
or advances a prognostic node carrying the canopy-air heat and moisture capacity
(`storage = PrognosticCanopyAir(...)`). Pass it as the interface temperature
formulation; it closes the specific-humidity slot as well.

Fields:
- `soil`   : the soil vapor branch (a [`DryLayerHumidity`](@ref)).
- `canopy` : the leaf vapor/photosynthesis branch (a [`CanopyConductanceHumidity`](@ref)).
- `soil_skin_flux` : skin↔bulk conduction `Λᵍ = κᵀ/ℓᵀ` (a [`SoilConductiveFlux`](@ref)).
- `leaf_albedo`, `ground_albedo` : broadband shortwave albedos. A `Number`, or a
  `Field{Center, Center, Nothing}` of per-cell values (see [`atmosphere_land_interface`](@ref)).
- `max_canopy_emissivity`, `ground_emissivity` : longwave emissivities (`εˡᵉᵃᶠ = εᵐᵃˣ(1 − e^{−LAI})`),
  each a `Number` or a per-cell `Field`.
- `extinction`, `clumping` : Beer–Lambert `K`, `Ω` for the shortwave split.
- `leaf_boundary_conductance` : per-leaf boundary-layer conductance `gᵇ` (m s⁻¹) →
  sensible `gˡᵉᵃᶠᵀ = ρcₚ·LAI·gᵇ`, vapor `gˡᵇ = ρ·LAI·gᵇ` (in series with the stomata
  when dry, alone over the wetted fraction). A constant; CTSM and ClimaLand scale it
  with the wind as `Cᵥ √(u★ / dˡᵉᵃᶠ)` (`Cᵥ = 0.01 m s⁻¹ᐟ²`, `dˡᵉᵃᶠ = 0.04 m`), which a
  wind-dependent closure in this slot would reproduce.
- `undercanopy_conductance` : ground↔canopy-air conductance `gᵘᶜ` → `gᵍᵀ = ρcₚ·gᵘᶜ`;
  a constant `Number` (m s⁻¹), an [`AreaIndexUndercanopyConductance`](@ref), or a
  [`FrictionVelocityUndercanopyConductance`](@ref).
- `wet_soil_resistance` : soil surface resistance on the moist-soil (vanishing dry layer)
  vapor branch (a [`SellersSoilResistance`](@ref)), or `nothing` (the default: above the
  dry-layer onset the soil itself does not limit evaporation, and the litter layer and
  undercanopy path carry the resistance).
- `litter_resistance` : plant-litter resistance in series on both ground vapor branches
  (a [`LitterResistance`](@ref), the default), or `nothing` for litter-free ground.
- `inner_iterations` : Newton iterations of the coupled skin solve.
- `interception` : wet-canopy vapor branch parameters (a [`CanopyInterception`](@ref)),
  or `nothing` for a dry canopy (the default; recovers the current CAS bit-for-bit).
- `phase` : saturation phase (Liquid).
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
                        inner_iterations          = 10,
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
                          convert_if_number(FT, undercanopy_conductance),
                          wet_soil_resistance, litter_resistance,
                          inner_iterations, interception, phase, storage)
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
##### Per-cell optics: localization
#####

# Collapse `Field`-valued slots to cell (i, j) before the index-free canopy solve;
# `state2dindex` returns a `Number` slot unchanged.
@inline local_interface_formulation(formulation, i, j) = formulation

@inline local_canopy_air_storage(storage, i, j) = storage
@inline local_canopy_air_storage(s::PrognosticCanopyAir, i, j) =
    PrognosticCanopyAir(state2dindex(s.layer_depth, i, j))

@inline local_interface_formulation(c::CanopyAirSpace, i, j) =
    CanopyAirSpace(c.soil, c.canopy, c.soil_skin_flux,
                   state2dindex(c.leaf_albedo, i, j),
                   state2dindex(c.ground_albedo, i, j),
                   state2dindex(c.max_canopy_emissivity, i, j),
                   state2dindex(c.ground_emissivity, i, j),
                   c.extinction, c.clumping, c.leaf_boundary_conductance,
                   c.undercanopy_conductance, c.wet_soil_resistance, c.litter_resistance,
                   c.inner_iterations, c.interception, c.phase,
                   local_canopy_air_storage(c.storage, i, j))

Base.summary(::CanopyAirSpace) = "CanopyAirSpace"
Base.show(io::IO, c::CanopyAirSpace) =
    print(io, "CanopyAirSpace(soil=", summary(c.soil), ", canopy=", summary(c.canopy),
          ", storage=", summary(c.storage), ")")

Adapt.@adapt_structure CanopyAirSpace

# Materialization / identity — delegate to the sub-models so the per-cell interface
# state carries the soil saturation, bulk temperature, and LAI the branches read.
@inline interface_phase(c::CanopyAirSpace) = interface_phase(c.soil)
@inline skin_conductance(c::CanopyAirSpace) = skin_conductance(c.soil_skin_flux)
# The soil branch always publishes the saturation 𝒮 and the canopy branch its stress
# state; a canopy with interception additionally pulls the prognostic canopy water
# store Wᶜ (→ fʷᵉᵗ).
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
    merge(interface_energy_state(i, j, grid, c.soil, land_state),
          (time_step = land_state.time_step,))
@inline canopy_leaf_area_index(c::CanopyAirSpace) = canopy_leaf_area_index(c.canopy)
@inline interface_vegetation_state(i, j, grid, c::CanopyAirSpace, vegetation, time_interpolator) =
    interface_vegetation_state(i, j, grid, c.canopy, vegetation, time_interpolator)

# Leaf vapor conductance: stomata in series with the leaf boundary layer on the dry
# fraction, the wet-canopy conductance `gʷ` in parallel. `LAI → 0` sends both to zero.
@inline function leaf_vapor_conductance(gᶜ, gˡᵇ, fʷᵉᵗ, gʷ)
    gᵈ = ifelse(gᶜ + gˡᵇ > 0, gᶜ * gˡᵇ / (gᶜ + gˡᵇ), zero(gᶜ))
    return (1 - fʷᵉᵗ) * gᵈ + gʷ
end

# Wet-canopy vapor conductance: the wetted fraction fʷᵉᵗ = (Wᶜ/Wᶜᵐᵃˣ)^(2/3) (Deardorff 1978)
# of the leaf boundary-layer conductance, capped so the store cannot evaporate more than
# it holds over the step, Eʷᵉᵗ = gʷ Δq ≤ Wᶜ/Δt.
@inline wet_canopy_conductance(::Nothing, hydrology, gˡᵇ, Δq, Δt) = zero(gˡᵇ), zero(gˡᵇ)
@inline function wet_canopy_conductance(::CanopyInterception, hydrology, gˡᵇ, Δq, Δt)
    FT    = typeof(gˡᵇ)
    Wᶜ    = convert(FT, hydrology.canopy_water_storage)
    Wᶜᵐᵃˣ = convert(FT, hydrology.canopy_water_capacity)
    fʷᵉᵗ  = ifelse(Wᶜᵐᵃˣ > 0, min(Wᶜ / Wᶜᵐᵃˣ, one(FT))^convert(FT, 2//3), zero(FT))
    gʷ    = fʷᵉᵗ * gˡᵇ
    gʷ    = ifelse((Δt > 0) & (Δq > 0), min(gʷ, Wᶜ / (Δt * Δq)), gʷ)
    return fʷᵉᵗ, gʷ
end

# Leaf vapor conductance `gˡᵉᵃᶠᵛ`, its wet-canopy part `gʷ`, and the leaf-saturated
# humidity `qˡᵉᵃᶠ` at the leaf temperature; the stomata and the wet cap see the node
# humidity `qᵃᶜ`.
@inline function leaf_vapor_terms(c, Tˡᵉᵃᶠ, qᵃᶜ, gˡᵇ, Δt, Ψₛ, Ψₐ, Ψᵣ, ℙₐ, transmittance)
    gᶜ, qˡᵉᵃᶠ = canopy_conductance_terms(c.canopy, Tˡᵉᵃᶠ, qᵃᶜ, Ψₛ, Ψₐ, Ψᵣ, ℙₐ, transmittance)
    fʷᵉᵗ, gʷ = wet_canopy_conductance(c.interception, Ψₛ.hydrology, gˡᵇ, qˡᵉᵃᶠ - qᵃᶜ, Δt)
    return leaf_vapor_conductance(gᶜ, gˡᵇ, fʷᵉᵗ, gʷ), gʷ, qˡᵉᵃᶠ
end

# Soil vapor conductance and source humidity at the soil-skin temperature `Tᵍ`: the
# dry-layer branch (front humidity qᵉ through Gᵉ in series with the litter + undercanopy
# path `gᵖ`) blended with the saturated-skin branch (qᵍ⁺ through gᵍᵛ) by the soil
# model's weight `fᵈ`.
@inline function soil_vapor_terms(soil, Tᵍ, gᵍᵛ, gᵖ, Ψₛ, Ψₐ, ℙₐ)
    Gᵉ, qᵉ, fᵈ, qᵍ⁺ = dry_layer_terms(soil, Tᵍ, Ψₛ, Ψₐ, ℙₐ)
    Gᵉ = Gᵉ * gᵖ / (gᵖ + Gᵉ)
    Gᵉ⁺ = fᵈ * Gᵉ + (1 - fᵈ) * gᵍᵛ
    qᵉ⁺ = ifelse(Gᵉ⁺ > eps(eltype(Ψₛ)), (fᵈ * Gᵉ * qᵉ + (1 - fᵈ) * gᵍᵛ * qᵍ⁺) / Gᵉ⁺, qᵍ⁺)
    return Gᵉ⁺, qᵉ⁺
end

# Fraction of the start-of-step node value surviving in the step mean of the node
# balance C dx/dt = Σg (x_eq − x): m = (τ/Δt)(1 − e^{−Δt/τ}) with τ = C/Σg. A massless
# node (C = 0) or Δt = 0 gives 0, the diagnostic equilibrium.
@inline function node_memory(Σg, C, Δt)
    FT = typeof(Σg)
    τ  = C / max(Σg, eps(FT))
    m  = -(τ / max(Δt, eps(FT))) * expm1(-Δt / τ)
    return ifelse((Δt > 0) & (C > 0), m, zero(FT))
end

"""
    advance_canopy_air(x, x_eq, Σg, C, Δt)

Advance a prognostic canopy-air node value `x` toward its conductance-weighted
equilibrium `x_eq` over `Δt`, the exact solution of ``C \\, dx/dt = Σg (x_{eq} - x)``
at fixed conductances: ``x ← x_{eq} + (x - x_{eq}) e^{-Δt Σg / C}``. `Δt = 0` or
`C = 0` return the equilibrium.
"""
@inline function advance_canopy_air(x, x_eq, Σg, C, Δt)
    FT = typeof(x)
    w = ifelse((Δt > 0) & (C > 0), exp(-Δt * Σg / max(C, eps(FT))), zero(FT))
    return x_eq + (x - x_eq) * w
end

@inline canopy_air_depth(::DiagnosticCanopyAir) = 0
@inline canopy_air_depth(s::PrognosticCanopyAir) = s.layer_depth

"""
    canopy_air_space_solve(c::CanopyAirSpace, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)

Solve the leaf temperature `Tˡᵉᵃᶠ` and soil-skin temperature `Tᵍ` of one cell by Newton
iteration on their energy balances, with the canopy-air node `(Tᵃᶜ, qᵃᶜ)` eliminated as
the conductance-weighted mean of its sources (relaxed toward the start-of-step node over
the step for a [`PrognosticCanopyAir`](@ref) storage). `Ψₛ` is the previous fixed-point
iterate, whose similarity scales set the aerodynamic conductances and whose node values
are the start-of-step node; `Ψᵢ.T` is the bulk reservoir `Tˡᵃ`, and `Ψᵣ` the interface
radiation state.
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
    Δt  = Ψₛ.energy.time_step
    LAI = Ψₛ.vegetation.leaf_area_index
    u★  = Ψₛ.fluxes.u★

    gᵃᵀ = aerodynamic_heat_conductance(Ψₛ, Ψₐ, ℂᵃᵗ)
    gᵃᵛ = aerodynamic_vapor_conductance(Ψₛ, Ψₐ, ℂᵃᵗ)

    # Land surface velocities are zero, so the surface wind speed is the atmospheric one.
    Vₐ  = sqrt(Ψₐ.u^2 + Ψₐ.v^2)
    gᵘᶜ = undercanopy_conductance(c.undercanopy_conductance, LAI, Vₐ, u★)
    gˡᵉᵃᶠᵀ = ρᵃᵗ * cᵖ * LAI * c.leaf_boundary_conductance
    gˡᵇ    = ρᵃᵗ * LAI * c.leaf_boundary_conductance
    gᵍᵀ    = ρᵃᵗ * cᵖ * gᵘᶜ
    # Ground vapor path: the litter resistance `rˡ` and, on the moist-soil branch, the soil
    # surface resistance `rˢ` in series with the undercanopy path (Sakaguchi & Zeng 2009).
    rˡ  = litter_resistance(c.litter_resistance, u★)
    rˢ  = soil_surface_resistance(c.wet_soil_resistance, Ψₛ.hydrology.saturation)
    gᵖ  = ρᵃᵗ * gᵘᶜ / (1 + gᵘᶜ * rˡ)
    gᵍᵛ = ρᵃᵗ * gᵘᶜ / (1 + gᵘᶜ * (rˡ + rˢ))
    Λ   = convert(FT, skin_conductance(c.soil_skin_flux))

    # Canopy-air heat and vapor capacities (zero for a massless node) and the
    # start-of-step node.
    hᶜ = convert(FT, canopy_air_depth(c.storage))
    Cᵀ  = ρᵃᵗ * cᵖ * hᶜ
    Cᵛ  = ρᵃᵗ * hᶜ
    T⁻  = Ψₛ.temperature
    q⁻  = Ψₛ.specific_humidity

    # Shortwave split and longwave emissivities (broadband).
    σ   = Ψᵣ.σ
    SW  = Ψᵣ.ℐꜜˢʷ
    LWꜜ = Ψᵣ.ℐꜜˡʷ
    αˡᵉᵃᶠ = convert(FT, c.leaf_albedo)
    αᵍ    = convert(FT, c.ground_albedo)
    εˡᵉᵃᶠ = convert(FT, c.max_canopy_emissivity) * (1 - exp(-LAI))
    εᵍ    = convert(FT, c.ground_emissivity)
    transmittance = canopy_transmittance(c.extinction, c.clumping, LAI)
    SWˡᵉᵃᶠ = (1 - αˡᵉᵃᶠ) * (1 - transmittance) * SW
    SWᵍ    = transmittance * (1 - αᵍ) * SW

    Tˡᵉᵃᶠ = Tˡᵃ
    Tᵍ    = Tˡᵃ
    Tᵃᶜ   = T⁻
    qᵃᶜ   = q⁻
    max_temperature_step = convert(FT, 25)

    for _ in 1:c.inner_iterations
        gˡᵉᵃᶠᵛ, gʷ, qˡᵉᵃᶠ = leaf_vapor_terms(c, Tˡᵉᵃᶠ, qᵃᶜ, gˡᵇ, Δt, Ψₛ, Ψₐ, Ψᵣ, ℙₐ, transmittance)
        Gᵉ, qᵉ = soil_vapor_terms(c.soil, Tᵍ, gᵍᵛ, gᵖ, Ψₛ, Ψₐ, ℙₐ)

        # Node: the equilibrium mean of its sources blended with the start-of-step node by
        # the memory `m`; its sensitivity to source `i` is (1 − m) gᵢ/Σg.
        Σgᵀ = gᵃᵀ + gˡᵉᵃᶠᵀ + gᵍᵀ
        Σgᵛ = gᵃᵛ + gˡᵉᵃᶠᵛ + Gᵉ
        mᵀ  = node_memory(Σgᵀ, Cᵀ, Δt)
        mᵛ  = node_memory(Σgᵛ, Cᵛ, Δt)
        equilibrium_temperature = conductance_weighted_node(T⁻, (gᵍᵀ, gˡᵉᵃᶠᵀ, gᵃᵀ), (Tᵍ, Tˡᵉᵃᶠ, θᵃᵗ))
        equilibrium_humidity = conductance_weighted_node(q⁻, (Gᵉ, gˡᵉᵃᶠᵛ, gᵃᵛ), (qᵉ, qˡᵉᵃᶠ, qᵃᵗ))
        Tᵃᶜ = equilibrium_temperature + (T⁻ - equilibrium_temperature) * mᵀ
        qᵃᶜ = equilibrium_humidity + (q⁻ - equilibrium_humidity) * mᵛ
        wˡᵉᵃᶠᵀ = (1 - mᵀ) * gˡᵉᵃᶠᵀ / max(Σgᵀ, eps(FT))
        wᵍᵀ = (1 - mᵀ) * gᵍᵀ / max(Σgᵀ, eps(FT))
        wˡᵉᵃᶠᵛ = (1 - mᵛ) * gˡᵉᵃᶠᵛ / max(Σgᵛ, eps(FT))
        wᵍᵛ = (1 - mᵛ) * Gᵉ / max(Σgᵛ, eps(FT))

        LWꜜᵍ   = (1 - εˡᵉᵃᶠ) * LWꜜ + εˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^4
        LWꜛᵍ   = εᵍ * σ * Tᵍ^4 + (1 - εᵍ) * LWꜜᵍ
        LWˡᵉᵃᶠ = εˡᵉᵃᶠ * (LWꜜ + LWꜛᵍ) - 2 * εˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^4
        LWᵍ    = εᵍ * (LWꜜᵍ - σ * Tᵍ^4)
        dqˡᵉᵃᶠ = saturation_humidity_slope(ℂᵃᵗ, Tˡᵉᵃᶠ, pᵃᵗ, c.phase)
        dqᵍ    = saturation_humidity_slope(ℂᵃᵗ, Tᵍ, pᵃᵗ, c.phase)

        # Leaf and ground energy residuals and their Jacobian, including each balance's
        # dependence on the other skin through the node and the longwave exchange.
        Rˡᵉᵃᶠ = SWˡᵉᵃᶠ + LWˡᵉᵃᶠ - gˡᵉᵃᶠᵀ * (Tˡᵉᵃᶠ - Tᵃᶜ) - ℒ * gˡᵉᵃᶠᵛ * (qˡᵉᵃᶠ - qᵃᶜ)
        Rᵍ    = SWᵍ + LWᵍ - gᵍᵀ * (Tᵍ - Tᵃᶜ) - ℒ * Gᵉ * (qᵉ - qᵃᶜ) - Λ * (Tᵍ - Tˡᵃ)
        ∂Rˡᵉᵃᶠ∂Tˡᵉᵃᶠ = 4 * εˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^3 * (εˡᵉᵃᶠ * (1 - εᵍ) - 2) - gˡᵉᵃᶠᵀ * (1 - wˡᵉᵃᶠᵀ) - ℒ * gˡᵉᵃᶠᵛ * (1 - wˡᵉᵃᶠᵛ) * dqˡᵉᵃᶠ
        ∂Rˡᵉᵃᶠ∂Tᵍ = 4 * εˡᵉᵃᶠ * εᵍ * σ * Tᵍ^3 + gˡᵉᵃᶠᵀ * wᵍᵀ + ℒ * gˡᵉᵃᶠᵛ * wᵍᵛ * dqᵍ
        ∂Rᵍ∂Tˡᵉᵃᶠ = 4 * εᵍ * εˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^3 + gᵍᵀ * wˡᵉᵃᶠᵀ + ℒ * Gᵉ * wˡᵉᵃᶠᵛ * dqˡᵉᵃᶠ
        ∂Rᵍ∂Tᵍ = -4 * εᵍ * σ * Tᵍ^3 - gᵍᵀ * (1 - wᵍᵀ) - ℒ * Gᵉ * (1 - wᵍᵛ) * dqᵍ - Λ
        # A canopy-free cell (LAI = 0) has no leaf balance: Rˡᵉᵃᶠ = ∂Rˡᵉᵃᶠ/∂Tˡᵉᵃᶠ = 0.
        ∂Rˡᵉᵃᶠ∂Tˡᵉᵃᶠ = ifelse(∂Rˡᵉᵃᶠ∂Tˡᵉᵃᶠ < 0, ∂Rˡᵉᵃᶠ∂Tˡᵉᵃᶠ, -one(FT))
        determinant = ∂Rˡᵉᵃᶠ∂Tˡᵉᵃᶠ * ∂Rᵍ∂Tᵍ - ∂Rˡᵉᵃᶠ∂Tᵍ * ∂Rᵍ∂Tˡᵉᵃᶠ
        ΔTˡᵉᵃᶠ = (∂Rᵍ∂Tᵍ * Rˡᵉᵃᶠ - ∂Rˡᵉᵃᶠ∂Tᵍ * Rᵍ) / determinant
        ΔTᵍ    = (∂Rˡᵉᵃᶠ∂Tˡᵉᵃᶠ * Rᵍ - ∂Rᵍ∂Tˡᵉᵃᶠ * Rˡᵉᵃᶠ) / determinant
        Tˡᵉᵃᶠ -= clamp(ΔTˡᵉᵃᶠ, -max_temperature_step, max_temperature_step)
        Tᵍ    -= clamp(ΔTᵍ, -max_temperature_step, max_temperature_step)
    end

    # The node and the flux partition at the final skins.
    gˡᵉᵃᶠᵛ, gʷ, qˡᵉᵃᶠ = leaf_vapor_terms(c, Tˡᵉᵃᶠ, qᵃᶜ, gˡᵇ, Δt, Ψₛ, Ψₐ, Ψᵣ, ℙₐ, transmittance)
    Gᵉ, qᵉ = soil_vapor_terms(c.soil, Tᵍ, gᵍᵛ, gᵖ, Ψₛ, Ψₐ, ℙₐ)
    Σgᵀ = gᵃᵀ + gˡᵉᵃᶠᵀ + gᵍᵀ
    Σgᵛ = gᵃᵛ + gˡᵉᵃᶠᵛ + Gᵉ
    equilibrium_temperature = conductance_weighted_node(T⁻, (gᵍᵀ, gˡᵉᵃᶠᵀ, gᵃᵀ), (Tᵍ, Tˡᵉᵃᶠ, θᵃᵗ))
    equilibrium_humidity = conductance_weighted_node(q⁻, (Gᵉ, gˡᵉᵃᶠᵛ, gᵃᵛ), (qᵉ, qˡᵉᵃᶠ, qᵃᵗ))
    Tᵃᶜ = equilibrium_temperature + (T⁻ - equilibrium_temperature) * node_memory(Σgᵀ, Cᵀ, Δt)
    qᵃᶜ = equilibrium_humidity + (q⁻ - equilibrium_humidity) * node_memory(Σgᵛ, Cᵛ, Δt)

    # Effective radiating temperature: σ T⁴ = upwelling longwave above the canopy.
    LWꜜᵍ = (1 - εˡᵉᵃᶠ) * LWꜜ + εˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^4
    LWꜛᵍ = εᵍ * σ * Tᵍ^4 + (1 - εᵍ) * LWꜜᵍ
    LWꜛ  = (1 - εˡᵉᵃᶠ) * LWꜛᵍ + εˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^4
    effective_temperature = ifelse(σ > 0, (LWꜛ / σ)^convert(FT, 1//4), Tᵃᶜ)

    Hˡᵉᵃᶠ  = gˡᵉᵃᶠᵀ * (Tˡᵉᵃᶠ - Tᵃᶜ)
    Hᵍ     = gᵍᵀ * (Tᵍ - Tᵃᶜ)
    LEˡᵉᵃᶠ = ℒ * gˡᵉᵃᶠᵛ * (qˡᵉᵃᶠ - qᵃᶜ)
    LEᵍ    = ℒ * Gᵉ * (qᵉ - qᵃᶜ)
    𝒬ᵍ     = Λ * (Tᵍ - Tˡᵃ)
    Eʷᵉᵗ   = gʷ * (qˡᵉᵃᶠ - qᵃᶜ)

    return (; Tˡᵉᵃᶠ = convert(FT, Tˡᵉᵃᶠ), Tᵍ = convert(FT, Tᵍ),
              Tᵃᶜ = convert(FT, Tᵃᶜ), qᵃᶜ = convert(FT, qᵃᶜ),
              effective_temperature = convert(FT, effective_temperature),
              Hˡᵉᵃᶠ = convert(FT, Hˡᵉᵃᶠ), Hᵍ = convert(FT, Hᵍ),
              LEˡᵉᵃᶠ = convert(FT, LEˡᵉᵃᶠ), LEᵍ = convert(FT, LEᵍ),
              𝒬ᵍ = convert(FT, 𝒬ᵍ), Eʷᵉᵗ = convert(FT, Eʷᵉᵗ), ℒ = convert(FT, ℒ),
              Σgᵀ = convert(FT, Σgᵀ), Σgᵛ = convert(FT, Σgᵛ),
              equilibrium_temperature = convert(FT, equilibrium_temperature), equilibrium_humidity = convert(FT, equilibrium_humidity),
              Cᵀ = convert(FT, Cᵀ), Cᵛ = convert(FT, Cᵛ))
end

# One shared solve returns both node values for the fixed point.
@inline function interface_temperature_and_humidity(c::CanopyAirSpace, ::CanopyAirSpace,
                                                    Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₛ, ℙₐ, ℙᵢ)
    sol = canopy_air_space_solve(c, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    return sol.Tᵃᶜ, sol.qᵃᶜ
end

# A prognostic node is model state, frozen through the fixed point.
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

The `temperature` slot of a [`CanopyAirSpace`](@ref) interface: the canopy-air node
temperature the atmosphere sees, the leaf and soil-skin temperatures, the two-source flux
shares, and the prognostic node state (`nothing` for a diagnostic node).
"""
struct CanopyAirSpaceDiagnostics{F, S}
    interface              :: F   # canopy-air node Tᵃᶜ (what MOST sees)
    canopy                 :: F   # leaf temperature Tˡᵉᵃᶠ
    soil_skin              :: F   # soil-skin temperature Tᵍ
    effective              :: F   # radiating (LST) temperature effective_temperature
    ground_heat_flux       :: F   # skin→bulk conduction 𝒬ᵍ
    canopy_latent_heat     :: F   # leaf transpiration LEˡᵉᵃᶠ
    soil_latent_heat       :: F   # soil evaporation LEᵍ
    canopy_sensible_heat   :: F   # leaf sensible Hˡᵉᵃᶠ
    soil_sensible_heat     :: F   # ground sensible Hᵍ
    canopy_evaporation     :: F   # wet-canopy evaporation Eʷᵉᵗ (kg m⁻² s⁻¹, up)
    canopy_wet_latent_heat :: F   # wet-canopy latent heat ℒ·Eʷᵉᵗ (W m⁻², up)
    land_vapor_flux        :: F   # soil evaporation + transpiration, drawn from the land water store
    state                  :: S   # prognostic CanopyAirState, or nothing (diagnostic node)
end

CanopyAirSpaceDiagnostics(grid, storage = DiagnosticCanopyAir()) =
    CanopyAirSpaceDiagnostics(ntuple(_ -> Field{Center, Center, Nothing}(grid),
                                     Val(fieldcount(CanopyAirSpaceDiagnostics) - 1))...,
                              build_canopy_air_state(storage, grid))

Adapt.@adapt_structure CanopyAirSpaceDiagnostics

@inline skin_temperature(Ts::CanopyAirSpaceDiagnostics, i, j) = @inbounds Ts.soil_skin[i, j, 1]

# Vapor fluxes the land closures consume: the wet-canopy evaporation `Eʷᵉᵗ` drains the canopy
# store, the rest of the leaf and ground sources drain the land water store. A plain skin
# has no canopy, and its atmospheric vapor flux `Jᵛ` is the land's loss.
@inline canopy_evaporation(Ts, i, j) = false
@inline canopy_evaporation(Ts::CanopyAirSpaceDiagnostics, i, j) = @inbounds Ts.canopy_evaporation[i, j, 1]
@inline land_vapor_flux(Ts, i, j, Jᵛ) = Jᵛ
@inline land_vapor_flux(Ts::CanopyAirSpaceDiagnostics, i, j, Jᵛ) = @inbounds Ts.land_vapor_flux[i, j, 1]

Base.summary(::CanopyAirSpaceDiagnostics) = "CanopyAirSpaceDiagnostics"
Base.show(io::IO, d::CanopyAirSpaceDiagnostics) = print(io, summary(d))
