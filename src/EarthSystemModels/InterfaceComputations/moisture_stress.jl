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
texture class — a nominal retention curve for [`PlantAvailableWaterStress`](@ref) when
the hydrology carries none of its own. `texture` is one of

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
                              retention_curve = nothing,
                              texture = nothing,
                              inverse_air_entry_head = nothing,
                              pore_size_uniformity = nothing,
                              field_capacity_head = 1,
                              wilting_point_head = 150)

Transpiration moisture stress from plant-available water: `β` rises linearly from 0 at
the permanent wilting point to 1 at field capacity,

```math
β(𝒮) = \\mathrm{clamp}\\left(\\frac{𝒮 - 𝒮ʷᵖ}{𝒮ᶠᶜ - 𝒮ʷᵖ}, 0, 1\\right),
\\qquad 𝒮ˣ = \\left[1 + (α ψˣ)^n\\right]^{-m}, \\quad m = 1 - 1/n,
```

with both endpoints evaluated on a van Genuchten (1980) retention curve, supplied by
exactly one of

* `retention_curve` — the hydrology's own [`VanGenuchtenRetention`](@ref): pass the same
  object given to [`VariablySaturatedHydrology`](@ref), so the stress and the hydrology
  share one curve by construction;
* `texture` — a USDA texture class (`:loam`, `:silt_loam`, … — resolved through
  [`van_genuchten_texture_parameters`](@ref)) for a hydrology that carries no retention
  curve of its own (e.g. [`BucketHydrology`](@ref)), where the stress's curve is a
  modeling choice;
* explicit `inverse_air_entry_head` `α` (m⁻¹) and `pore_size_uniformity` `n`, each a
  scalar or a per-cell `Field` — hand a heterogeneous hydrology the same parameter
  fields it reads, and every column's endpoints follow its own curve.

The default heads follow [Balsamo et al. (2009)](@cite balsamo2009): `ψᶠᶜ = 1` m and
`ψʷᵖ = 150` m of suction.

!!! warning "Evaluate the endpoints on the hydrology's own curve"
    `β` compares the hydrology's saturation with endpoints on this curve, and effective
    saturations on different curves are different quantities: at the wilting head a loam
    sits near 0.4 of one published curve family and near 0.03 of another. When the
    hydrology owns a retention curve, share it — via `retention_curve`, or by handing a
    per-cell hydrology the same parameter fields it reads; endpoints from any other
    source shift the whole stress band and pin `β` high or low regardless of the actual
    wetness. The rule covers the state too: moisture transplanted from another model's
    parameter space misreads through any curve here, so rescale it by plant-available
    fraction first.

Because the stress is a ratio of effective saturations on one shared curve, the porosity
and the residual fraction cancel; sharing the curve is the whole consistency requirement.

This is the moisture stress meant for a transpiring canopy
([`CanopyConductanceHumidity`](@ref)'s `moisture_stress`): unlike
[`CriticalSaturation`](@ref) — a *bare-soil* evaporation model whose `β` reaches 1 at the
critical saturation and 0 only at the residual liquid fraction — its endpoints are the
plant ones, so stomata shut at wilting rather than at oven dryness. The clamp at wilting
is physical: a wilted plant does not respond to a perturbation.

```jldoctest
using NumericalEarth

soil_retention = VanGenuchtenRetention(α = 1.0, n = 2.0)

PlantAvailableWaterStress(retention_curve = soil_retention)

# output
PlantAvailableWaterStress(α=1.0, n=2.0, ψᶠᶜ=1.0, ψʷᵖ=150.0)
```
"""
struct PlantAvailableWaterStress{A, N, FT}
    inverse_air_entry_head :: A
    pore_size_uniformity   :: N
    field_capacity_head    :: FT
    wilting_point_head     :: FT
end

Adapt.adapt_structure(to, p::PlantAvailableWaterStress) =
    PlantAvailableWaterStress(adapt(to, p.inverse_air_entry_head),
                              adapt(to, p.pore_size_uniformity),
                              p.field_capacity_head,
                              p.wilting_point_head)

# A stress parameter slot holds a uniform scalar (converted to `FT`) or a per-cell
# `Field`, read at `(i, j)` by `interface_hydrology_state`.
@inline stress_parameter(FT, x::Number) = convert(FT, x)
@inline stress_parameter(FT, x) = x

function PlantAvailableWaterStress(FT::Type = Oceananigans.defaults.FloatType;
                                   retention_curve = nothing,
                                   texture = nothing,
                                   inverse_air_entry_head = nothing,
                                   pore_size_uniformity = nothing,
                                   field_capacity_head = 1,
                                   wilting_point_head = 150)
    explicit = !isnothing(inverse_air_entry_head) || !isnothing(pore_size_uniformity)
    sum((!isnothing(retention_curve), !isnothing(texture), explicit)) == 1 ||
        throw(ArgumentError("supply exactly one retention-parameter source: the " *
                            "hydrology's retention_curve, a texture class, or explicit " *
                            "inverse_air_entry_head and pore_size_uniformity"))
    if !isnothing(retention_curve)
        inverse_air_entry_head = retention_curve.α
        pore_size_uniformity   = retention_curve.n
    elseif !isnothing(texture)
        (; inverse_air_entry_head, pore_size_uniformity) = van_genuchten_texture_parameters(texture)
    else
        (isnothing(inverse_air_entry_head) || isnothing(pore_size_uniformity)) &&
            throw(ArgumentError("supply both inverse_air_entry_head and pore_size_uniformity"))
    end
    if pore_size_uniformity isa Number
        pore_size_uniformity > 1 ||
            throw(ArgumentError("pore_size_uniformity must exceed 1"))
    end
    0 < field_capacity_head < wilting_point_head ||
        throw(ArgumentError("heads must satisfy 0 < field_capacity_head < wilting_point_head"))
    return PlantAvailableWaterStress(stress_parameter(FT, inverse_air_entry_head),
                                     stress_parameter(FT, pore_size_uniformity),
                                     convert(FT, field_capacity_head),
                                     convert(FT, wilting_point_head))
end

# The endpoints arrive precomputed on the closure's (possibly per-cell) curve — see
# `interface_hydrology_state(i, j, grid, ::PlantAvailableWaterStress, land_state)`.
@inline function evaporation_efficiency(p::PlantAvailableWaterStress, hydrology)
    𝒮   = hydrology.saturation
    FT  = typeof(𝒮)
    𝒮ᶠᶜ = convert(FT, hydrology.field_capacity_saturation)
    𝒮ʷᵖ = convert(FT, hydrology.wilting_saturation)
    return clamp((𝒮 - 𝒮ʷᵖ) / (𝒮ᶠᶜ - 𝒮ʷᵖ), zero(FT), one(FT))
end

Base.summary(p::PlantAvailableWaterStress) =
    string("PlantAvailableWaterStress",
           "(α=", prettysummary(p.inverse_air_entry_head),
           ", n=", prettysummary(p.pore_size_uniformity),
           ", ψᶠᶜ=", prettysummary(p.field_capacity_head),
           ", ψʷᵖ=", prettysummary(p.wilting_point_head), ")")

Base.show(io::IO, p::PlantAvailableWaterStress) = print(io, summary(p))
