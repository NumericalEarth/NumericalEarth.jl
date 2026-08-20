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

"""
    PlantAvailableWaterStress(FT = Oceananigans.defaults.FloatType;
                              field_capacity_head = 1,
                              wilting_point_head = 150)

Transpiration moisture stress from plant-available water: `β` rises linearly from 0 at
the permanent wilting point to 1 at field capacity,

```math
β(𝒮) = \\mathrm{clamp}\\left(\\frac{𝒮 - 𝒮ʷᵖ}{𝒮ᶠᶜ - 𝒮ʷᵖ}, 0, 1\\right),
```

where `𝒮ᶠᶜ` and `𝒮ʷᵖ` are the effective saturations at the two suction heads. The
closure holds the heads, which are plant properties; the curve they are evaluated on
comes from the land's own hydrology, per cell, so the stress and the soil cannot
disagree. A hydrology carrying no retention curve (e.g. [`BucketHydrology`](@ref),
whose `𝒮` is a fill fraction rather than a retention-curve saturation) is rejected when
the interface is built.

The default heads follow [Balsamo et al. (2009)](@cite balsamo2009): `ψᶠᶜ = 1` m and
`ψʷᵖ = 150` m of suction.

Because `β` is a ratio of effective saturations on one curve, the porosity and the
residual liquid fraction cancel; the closure needs neither.

This is the moisture stress meant for a transpiring canopy
([`CanopyConductanceHumidity`](@ref)'s `moisture_stress`): unlike
[`CriticalSaturation`](@ref) — a *bare-soil* evaporation model whose `β` reaches 1 at the
critical saturation and 0 only at the residual liquid fraction — its endpoints are the
plant ones, so stomata shut at wilting rather than at oven dryness. The clamp at wilting
is physical: a wilted plant does not respond to a perturbation.

```jldoctest
using NumericalEarth

PlantAvailableWaterStress()

# output
PlantAvailableWaterStress(ψᶠᶜ=1.0, ψʷᵖ=150.0)
```
"""
struct PlantAvailableWaterStress{FT}
    field_capacity_head :: FT
    wilting_point_head  :: FT
end

function PlantAvailableWaterStress(FT::Type = Oceananigans.defaults.FloatType;
                                   field_capacity_head = 1,
                                   wilting_point_head = 150)
    0 < field_capacity_head < wilting_point_head ||
        throw(ArgumentError("heads must satisfy 0 < field_capacity_head < wilting_point_head"))
    return PlantAvailableWaterStress(convert(FT, field_capacity_head),
                                     convert(FT, wilting_point_head))
end

# The endpoints arrive precomputed on the land's own curve — see
# `interface_hydrology_state(i, j, grid, ::PlantAvailableWaterStress, land_state)`.
@inline function evaporation_efficiency(::PlantAvailableWaterStress, hydrology)
    𝒮   = hydrology.saturation
    FT  = typeof(𝒮)
    𝒮ᶠᶜ = convert(FT, hydrology.field_capacity_saturation)
    𝒮ʷᵖ = convert(FT, hydrology.wilting_saturation)
    return clamp((𝒮 - 𝒮ʷᵖ) / (𝒮ᶠᶜ - 𝒮ʷᵖ), zero(FT), one(FT))
end

Base.summary(p::PlantAvailableWaterStress) =
    string("PlantAvailableWaterStress",
           "(ψᶠᶜ=", prettysummary(p.field_capacity_head),
           ", ψʷᵖ=", prettysummary(p.wilting_point_head), ")")

Base.show(io::IO, p::PlantAvailableWaterStress) = print(io, summary(p))

#####
##### Which formulations need the land to own a retention curve.
#####
##### Checked once, when the atmosphere-land interface is built, so a stress paired with
##### a curve-less hydrology fails there instead of reading a `nothing` in a kernel.
#####

@inline requires_retention_curve(model) = false
@inline requires_retention_curve(::PlantAvailableWaterStress) = true
