#####
##### Moisture-availability (evaporation-efficiency) models — the β(𝒮) seam.
#####
##### `evaporation_efficiency(model, hydrology)` maps the land surface saturation 𝒮
##### to an availability factor β ∈ [0, 1]. `CriticalSaturation` is the bare-soil
##### evaporation model; `PlantAvailableWaterStress` is the transpiration model a
##### canopy reads. Both are consumed by `FractionalHumidity` and by the canopy
##### formulations (`CanopyConductanceHumidity`, `CanopyAirSpace`).
#####

"""
    struct CriticalSaturation

Evaporation efficiency after [Manabe (1969)](@cite manabe1969climate): the surface is saturated (`β = 1`) above a
critical saturation `𝒮ᶜ`, and the efficiency falls off linearly below it,

```math
β(𝒮) = \\min(𝒮 / 𝒮ᶜ, 1),   𝒮 = Mˡᵃ / Mˡᵃ⁺.
```

Used as the `efficiency` of [`FractionalHumidity`](@ref). The type declares its
land-state dependency (the saturation `𝒮`); the interface materializes exactly
that into the land interface state.
"""
struct CriticalSaturation{FT}
    critical_saturation :: FT
end

@inline function evaporation_efficiency(𝒮ᶜ::CriticalSaturation, hydrology)
    𝒮 = hydrology.saturation
    return min(𝒮 / convert(typeof(𝒮), 𝒮ᶜ.critical_saturation), one(𝒮))
end

# Constant efficiency — a uniformly sub-saturated surface; reads no land state.
@inline evaporation_efficiency(β::Number, hydrology) = β

# The van Genuchten (1980) shape relation and effective-saturation curve at
# dimensionless suction αψ ≥ 0 — one home for the algebra `VariablySaturatedHydrology`'s
# retention curve shares. Defined here (not in `Lands`) because the interface modules
# load first; `Lands` imports these.
@inline van_genuchten_m(n) = 1 - 1/n
@inline van_genuchten_saturation(αψ, n) = (1 + αψ^n)^(-van_genuchten_m(n))

const CARSEL_PARRISH_RETENTION = (
    sand            = (inverse_air_entry_head = 14.5, pore_size_uniformity = 2.68),
    loamy_sand      = (inverse_air_entry_head = 12.4, pore_size_uniformity = 2.28),
    sandy_loam      = (inverse_air_entry_head = 7.5,  pore_size_uniformity = 1.89),
    loam            = (inverse_air_entry_head = 3.6,  pore_size_uniformity = 1.56),
    silt            = (inverse_air_entry_head = 1.6,  pore_size_uniformity = 1.37),
    silt_loam       = (inverse_air_entry_head = 2.0,  pore_size_uniformity = 1.41),
    sandy_clay_loam = (inverse_air_entry_head = 5.9,  pore_size_uniformity = 1.48),
    clay_loam       = (inverse_air_entry_head = 1.9,  pore_size_uniformity = 1.31),
    silty_clay_loam = (inverse_air_entry_head = 1.0,  pore_size_uniformity = 1.23),
    sandy_clay      = (inverse_air_entry_head = 2.7,  pore_size_uniformity = 1.23),
    silty_clay      = (inverse_air_entry_head = 0.5,  pore_size_uniformity = 1.09),
    clay            = (inverse_air_entry_head = 0.8,  pore_size_uniformity = 1.09))

"""
    van_genuchten_texture_parameters(texture)

The [Carsel and Parrish (1988)](@cite carsel1988) mean van Genuchten retention parameters
`(; inverse_air_entry_head, pore_size_uniformity)` (`α` in m⁻¹ and `n`) for a USDA soil
texture class — the parameter source a plant stress closure wants, as opposed to
moisture-matched pedotransfer fits. `texture` is one of

    :sand, :loamy_sand, :sandy_loam, :loam, :silt, :silt_loam,
    :sandy_clay_loam, :clay_loam, :silty_clay_loam, :sandy_clay, :silty_clay, :clay

```jldoctest
using NumericalEarth

van_genuchten_texture_parameters(:loam)

# output
(inverse_air_entry_head = 3.6, pore_size_uniformity = 1.56)
```
"""
function van_genuchten_texture_parameters(texture::Symbol)
    haskey(CARSEL_PARRISH_RETENTION, texture) ||
        throw(ArgumentError("unknown soil texture class :$texture; expected one of " *
                            join(string.(":", keys(CARSEL_PARRISH_RETENTION)), ", ")))
    return CARSEL_PARRISH_RETENTION[texture]
end

"""
    PlantAvailableWaterStress(FT = Oceananigans.defaults.FloatType;
                              texture = nothing,
                              inverse_air_entry_head = nothing,
                              pore_size_uniformity = nothing,
                              field_capacity_head = 1,
                              wilting_point_head = 150)

Transpiration moisture stress from plant-available water: `β` rises linearly from 0 at
the permanent wilting point to 1 at field capacity,

```math
β(𝒮) = \\mathrm{clamp}\\left(\\frac{𝒮 - 𝒮ʷᵖ}{𝒮ᶠᶜ - 𝒮ʷᵖ}, 0, 1\\right),
```

where `𝒮ᶠᶜ` and `𝒮ʷᵖ` are the effective saturations at the two suction heads. The default
heads follow [Balsamo et al. (2009)](@cite balsamo2009): `ψᶠᶜ = 1` m and `ψʷᵖ = 150` m of
suction.

The retention curve the heads are evaluated on may come from two places:

  * **The closure's own curve** — pass a USDA `texture` class (`:loam`, `:silt_loam`, … —
    resolved through [`van_genuchten_texture_parameters`](@ref)) or explicit
    `inverse_air_entry_head` `α` (m⁻¹) and `pore_size_uniformity` `n`.
  * **The land's own retention curve, per cell** (the default, when neither is given).
    A hydrology carrying no retention curve (e.g. [`BucketHydrology`](@ref), whose `𝒮`
    is a fill fraction rather than a retention-curve saturation) is rejected when the
    interface is built.

!!! warning "Pedotransfer curves over-stress the canopy"
    Prefer texture-class retention parameters ([Carsel and Parrish (1988)](@cite carsel1988))
    when the hydrology's curve comes from a pedotransfer reduction: fits matched through
    the very field-capacity and wilting heads spanned here push `n` toward 1, and on such
    a curve every moderate saturation maps deep into stress (`β ≈ 0.15` across a whole
    domain is the typical symptom).

Because the stress is a ratio of effective saturations on one curve, the porosity and the
residual liquid fraction cancel; the closure needs neither.

This is the moisture stress meant for a transpiring canopy
([`CanopyConductanceHumidity`](@ref)'s `moisture_stress`): unlike
[`CriticalSaturation`](@ref) — a *bare-soil* evaporation model whose `β` reaches 1 at the
critical saturation and 0 only at the residual liquid fraction — its endpoints are the
plant ones, so stomata shut at wilting rather than at oven dryness. The clamp at wilting
is physical: a wilted plant does not respond to a perturbation.

```jldoctest
using NumericalEarth

PlantAvailableWaterStress(texture = :loam)

# output
PlantAvailableWaterStress(α=3.6, n=1.56, ψᶠᶜ=1.0, ψʷᵖ=150.0)
```
"""
struct PlantAvailableWaterStress{FT, R}
    inverse_air_entry_head :: R
    pore_size_uniformity   :: R
    field_capacity_head    :: FT
    wilting_point_head     :: FT
end

function PlantAvailableWaterStress(FT::Type = Oceananigans.defaults.FloatType;
                                   texture = nothing,
                                   inverse_air_entry_head = nothing,
                                   pore_size_uniformity = nothing,
                                   field_capacity_head = 1,
                                   wilting_point_head = 150)
    if !isnothing(texture)
        (isnothing(inverse_air_entry_head) && isnothing(pore_size_uniformity)) ||
            throw(ArgumentError("supply either a texture class or explicit retention " *
                                "parameters, not both"))
        (; inverse_air_entry_head, pore_size_uniformity) = van_genuchten_texture_parameters(texture)
    elseif isnothing(inverse_air_entry_head) ≠ isnothing(pore_size_uniformity)
        throw(ArgumentError("supply both inverse_air_entry_head and pore_size_uniformity " *
                            "(or a texture class, or neither to use the land's own curve)"))
    end
    isnothing(pore_size_uniformity) || pore_size_uniformity > 1 ||
        throw(ArgumentError("pore_size_uniformity must exceed 1"))
    0 < field_capacity_head < wilting_point_head ||
        throw(ArgumentError("heads must satisfy 0 < field_capacity_head < wilting_point_head"))
    α = isnothing(inverse_air_entry_head) ? nothing : convert(FT, inverse_air_entry_head)
    n = isnothing(pore_size_uniformity)   ? nothing : convert(FT, pore_size_uniformity)
    return PlantAvailableWaterStress(α, n,
                                     convert(FT, field_capacity_head),
                                     convert(FT, wilting_point_head))
end

# A stress with no curve of its own reads the land's, per cell.
const HydrologyCurveWaterStress = PlantAvailableWaterStress{<:Any, Nothing}

# The endpoints arrive precomputed — on the land's own curve or the closure's — see
# `interface_hydrology_state(i, j, grid, ::PlantAvailableWaterStress, land_state)`.
@inline function evaporation_efficiency(::PlantAvailableWaterStress, hydrology)
    𝒮   = hydrology.saturation
    FT  = typeof(𝒮)
    𝒮ᶠᶜ = convert(FT, hydrology.field_capacity_saturation)
    𝒮ʷᵖ = convert(FT, hydrology.wilting_saturation)
    return clamp((𝒮 - 𝒮ʷᵖ) / (𝒮ᶠᶜ - 𝒮ʷᵖ), zero(FT), one(FT))
end

Base.summary(p::HydrologyCurveWaterStress) =
    string("PlantAvailableWaterStress",
           "(ψᶠᶜ=", prettysummary(p.field_capacity_head),
           ", ψʷᵖ=", prettysummary(p.wilting_point_head), ")")

Base.summary(p::PlantAvailableWaterStress) =
    string("PlantAvailableWaterStress",
           "(α=", prettysummary(p.inverse_air_entry_head),
           ", n=", prettysummary(p.pore_size_uniformity),
           ", ψᶠᶜ=", prettysummary(p.field_capacity_head),
           ", ψʷᵖ=", prettysummary(p.wilting_point_head), ")")

Base.show(io::IO, p::PlantAvailableWaterStress) = print(io, summary(p))

#####
##### Which formulations need the land to own a retention curve.
#####
##### Checked once, when the atmosphere-land interface is built, so a stress paired with
##### a curve-less hydrology fails there instead of reading a `nothing` in a kernel.
#####

@inline requires_retention_curve(model) = false
@inline requires_retention_curve(::HydrologyCurveWaterStress) = true
