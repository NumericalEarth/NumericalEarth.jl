#####
##### Urban morphometric roughness closures
#####
##### Aerodynamic momentum roughness length ℓᵐ and zero-plane displacement d for the
##### urban (built-up) surface, from the plan-area index λᵖ and mean building height h.
##### Each closure is a distinct type dispatched by `aerodynamic_parameters`, tied together by
##### `AbstractUrbanRoughness`.
#####

"""
    abstract type AbstractUrbanRoughness end

Supertype of the urban morphometric roughness closures. A closure maps the plan-area
index `λᵖ` and mean building height `h` to the momentum roughness length `ℓᵐ`
and zero-plane displacement `d` via [`aerodynamic_parameters`](@ref); it is also callable as
`closure(λᵖ, h)`.
"""
abstract type AbstractUrbanRoughness end

#####
##### Frontal-area estimators — λᶠ is not observed by GHSL, so it is estimated from (λᵖ, h)
#####

"""
    abstract type AbstractFrontalAreaEstimator end

Supertype of the estimators of the frontal-area index `λᶠ` from the plan-area index
`λᵖ` and mean height `h` (which GHSL does not observe). This estimate is the dominant
modeling uncertainty of the drag-partition route, hence it is a selectable closure field.
"""
abstract type AbstractFrontalAreaEstimator end

"""
    IsotropicFrontalArea()

Take `λᶠ ≈ λᵖ` (cubic elements).
"""
struct IsotropicFrontalArea <: AbstractFrontalAreaEstimator end

"""
$(TYPEDEF)

Take `λᶠ = λᵖ · h / Lb` for square buildings of characteristic width `Lb`
(`building_width`, m).
"""
struct CuboidFrontalArea{FT} <: AbstractFrontalAreaEstimator
    building_width :: FT
end

CuboidFrontalArea(; building_width = 10.0) = CuboidFrontalArea(building_width)

# Match the estimator's eltype to the closure's `FT` (isotropic carries none).
convert_frontal_area(FT, e::IsotropicFrontalArea) = e
convert_frontal_area(FT, e::CuboidFrontalArea) = CuboidFrontalArea(building_width = convert(FT, e.building_width))

Base.summary(::IsotropicFrontalArea) = "IsotropicFrontalArea()"
Base.summary(e::CuboidFrontalArea) = string("CuboidFrontalArea(building_width = ", e.building_width, ")")
Base.show(io::IO, e::AbstractFrontalAreaEstimator) = print(io, summary(e))

"""
$(TYPEDSIGNATURES)

Frontal-area index `λᶠ` of plan-area index `λᵖ` and mean height `h` under `estimator`.
"""
@inline frontal_area_index(::IsotropicFrontalArea, λᵖ, h) = λᵖ
@inline frontal_area_index(e::CuboidFrontalArea, λᵖ, h) =
    ifelse(e.building_width > 0, λᵖ * h / e.building_width, λᵖ)

#####
##### Height distributions — how the spread of roughness-element heights enters (ℓᵐ, d)
#####

"""
    abstract type AbstractHeightDistribution end

Supertype of the treatments of the roughness-element height distribution within a cell.
[`UniformHeight`](@ref) takes every element to be as tall as the mean; [`VariableHeight`](@ref)
parameterizes the spread `σʰ` and the tallest element `hᵐᵃˣ` as fractions of the mean,
correcting the uniform-height `(ℓᵐ, d)`.
"""
abstract type AbstractHeightDistribution end

"""
    UniformHeight()

Treat every roughness element as being of the mean height `h`, so `(ℓᵐ, d)` follow from
the plan-area and frontal-area indices alone. The idealized obstacle-array limit of
[Macdonald et al. (1998)](@cite Macdonald1998).
"""
struct UniformHeight <: AbstractHeightDistribution end

"""
$(TYPEDEF)

Correct the uniform-height `(ℓᵐ, d)` for the spread of roughness-element heights, after
[Kanda et al. (2013)](@cite Kanda2013). The tallest elements absorb the most momentum, so
`d` is referenced to `hᵐᵃˣ` rather than to `h`, and `ℓᵐ` is rescaled by a polynomial in
`λᵖ σʰ / h`. Because a mean-height dataset does not observe `σʰ` or `hᵐᵃˣ`, both are
parameterized as fractions of `h`.
"""
struct VariableHeight{FT} <: AbstractHeightDistribution
    displacement_constants :: NTuple{3, FT}
    roughness_constants    :: NTuple{3, FT}
    height_variability     :: FT
    maximum_height_ratio   :: FT
end

"""
    VariableHeight(FT = Oceananigans.defaults.FloatType; kw...)

Construct a [`VariableHeight`](@ref) distribution.

* `displacement_constants` (`a0, b0, c0`) — constants of the displacement fit.
  Default `(1.29, 0.36, -0.17)`.
* `roughness_constants` (`a1, b1, c1`) — constants of the roughness rescaling.
  Default `(0.71, 20.21, -0.77)`.
* `height_variability` (`σʰ / h`) — assumed height standard deviation as a fraction
  of the mean height. Default 0.4.
* `maximum_height_ratio` (`hᵐᵃˣ / h`) — assumed maximum-to-mean height ratio. Default 2.5.

```jldoctest
julia> using NumericalEarth.Lands

julia> VariableHeight()
VariableHeight(height_variability = 0.4, maximum_height_ratio = 2.5)
```
"""
function VariableHeight(FT = Oceananigans.defaults.FloatType;
                        displacement_constants = (1.29, 0.36, -0.17),
                        roughness_constants = (0.71, 20.21, -0.77),
                        height_variability = 0.4,
                        maximum_height_ratio = 2.5)
    return VariableHeight(convert.(FT, displacement_constants),
                          convert.(FT, roughness_constants),
                          convert(FT, height_variability),
                          convert(FT, maximum_height_ratio))
end

# Match the distribution's eltype to the closure's `FT` (uniform carries none).
convert_height_distribution(FT, d::UniformHeight) = d
convert_height_distribution(FT, d::VariableHeight) =
    VariableHeight(convert.(FT, d.displacement_constants),
                   convert.(FT, d.roughness_constants),
                   convert(FT, d.height_variability),
                   convert(FT, d.maximum_height_ratio))

Base.summary(::UniformHeight) = "UniformHeight()"
Base.summary(d::VariableHeight) =
    string("VariableHeight(height_variability = ", d.height_variability,
           ", maximum_height_ratio = ", d.maximum_height_ratio, ")")
Base.show(io::IO, d::AbstractHeightDistribution) = print(io, summary(d))

#####
##### Pure morphometric ratios (unit-testable, shared by the closures)
#####

"""
$(TYPEDSIGNATURES)

Displacement ratio `d/h = 1 + A^(−λᵖ)·(λᵖ − 1)` of an obstacle array packed to plan-area
index `λᵖ`, clamped to `[0, dᵐᵃˣ]` (`dᵐᵃˣ < 1` avoids the singular skimming limit
`λᵖ → 1`). Empirical fit of [Macdonald et al. (1998)](@cite Macdonald1998).
"""
@inline function packing_displacement_ratio(λᵖ, A, dᵐᵃˣ)
    dʰ = 1 + A^(-λᵖ) * (λᵖ - 1)
    return clamp(dʰ, zero(dʰ), dᵐᵃˣ)
end

"""
$(TYPEDSIGNATURES)

Roughness ratio `ℓᵐ/h = (1 − dʰ)·exp{ −[0.5·β·(Cᴰ/ϰ²)·(1 − dʰ)·λᶠ]^(−1/2) }`, obtained by
partitioning the surface stress between the element drag and the ground and matching the
log law over the canopy ([Macdonald et al. (1998)](@cite Macdonald1998)). Non-monotonic in
`λᵖ`: `ℓᵐ` rises then falls (isolated → wake-interference → skimming) with the frontal
area `λᶠ`.
"""
@inline function drag_partition_roughness_ratio(λᶠ, dʰ, Cᴰ, ϰ, β)
    bracket = (β * Cᴰ) / (2 * ϰ^2) * (1 - dʰ) * λᶠ
    decay = ifelse(bracket > 0, exp(-inv(sqrt(bracket))), zero(bracket))
    return (1 - dʰ) * decay
end

"""
$(TYPEDSIGNATURES)

Displacement height `d` (m) referenced to the tallest roughness element `hᵐᵃˣ`, given the
mean height `h` and the height standard deviation `σʰ`:
`d/hᵐᵃˣ = c0·X² + (a0·λᵖ^b0 − c0)·X`, with `X = (σʰ + h)/hᵐᵃˣ` in `[0, 1]`
([Kanda et al. (2013)](@cite Kanda2013)).
"""
@inline function maximum_height_displacement(λᵖ, h, σʰ, hᵐᵃˣ, a0, b0, c0)
    X = clamp(ifelse(hᵐᵃˣ > 0, (σʰ + h) / hᵐᵃˣ, zero(h)), zero(h), one(h))
    ratio = c0 * X^2 + (a0 * λᵖ^b0 - c0) * X
    return hᵐᵃˣ * max(ratio, zero(ratio))
end

"""
$(TYPEDSIGNATURES)

Rescale the uniform-height roughness length `ℓᵐ` (m) for the height spread `σʰ` by
`b1·Y² + c1·Y + a1`, with `Y = λᵖ·σʰ/h` ([Kanda et al. (2013)](@cite Kanda2013)). Reduces
to `a1·ℓᵐ` for a height-homogeneous canopy (`σʰ → 0`).
"""
@inline function height_spread_roughness(ℓᵐ, λᵖ, h, σʰ, a1, b1, c1)
    Y = ifelse(h > 0, λᵖ * σʰ / h, zero(h))
    ratio = b1 * Y^2 + c1 * Y + a1
    return ℓᵐ * max(ratio, zero(ratio))
end

"""
$(TYPEDSIGNATURES)

Apply the height `distribution` to the uniform-height parameters `(ℓᵐ, d)` of a cell with
plan-area index `λᵖ` and mean height `h`. `dᵐᵃˣ` caps the returned `d/h`.
"""
@inline apply_height_distribution(::UniformHeight, ℓᵐ, d, λᵖ, h, dᵐᵃˣ) = ℓᵐ, d

@inline function apply_height_distribution(v::VariableHeight, ℓᵐ, d, λᵖ, h, dᵐᵃˣ)
    σʰ = v.height_variability * h
    hᵐᵃˣ = v.maximum_height_ratio * h

    a0, b0, c0 = v.displacement_constants
    a1, b1, c1 = v.roughness_constants

    ℓᵐ = height_spread_roughness(ℓᵐ, λᵖ, h, σʰ, a1, b1, c1)
    d = min(maximum_height_displacement(λᵖ, h, σʰ, hᵐᵃˣ, a0, b0, c0), dᵐᵃˣ * h)

    return ℓᵐ, d
end

#####
##### Closures
#####

"""
$(TYPEDEF)

Morphometric roughness closure. Maps `(λᵖ, h)` to `(ℓᵐ, d)` from the plan-area index and
mean building height alone: `d` from the plan-area packing, `ℓᵐ` from a drag partition over
the frontal area `λᶠ` estimated by `frontal_area`, both then corrected for the spread of
element heights by `height_distribution`. `ℓᵐ` is non-monotonic in `λᵖ` (isolated →
wake-interference → skimming).

Formulated by [Macdonald et al. (1998)](@cite Macdonald1998) for idealized obstacle arrays
and extended to height-heterogeneous real cities by [Kanda et al. (2013)](@cite Kanda2013).
"""
struct MorphometricRoughness{FT, E, H} <: AbstractUrbanRoughness
    array_constant             :: FT
    drag_coefficient           :: FT
    correction_factor          :: FT
    von_karman_constant        :: FT
    frontal_area               :: E
    height_distribution        :: H
    bare_soil_roughness        :: FT
    minimum_built_fraction     :: FT
    maximum_displacement_ratio :: FT
end

"""
    MorphometricRoughness(FT = Oceananigans.defaults.FloatType; kw...)

Construct a [`MorphometricRoughness`](@ref) closure.

* `array_constant` (`A`) — default 4.43 (staggered array); use 3.59 for a square one.
* `drag_coefficient` (`Cᴰ`) — building drag coefficient. Default 1.2.
* `correction_factor` (`β`) — drag-partition correction factor. Default 1.
* `von_karman_constant` (`ϰ`) — default 0.4.
* `frontal_area` — frontal-area estimator: [`IsotropicFrontalArea`](@ref) (default) or
  [`CuboidFrontalArea`](@ref).
* `height_distribution` — height-spread treatment: [`VariableHeight`](@ref) (default) or
  [`UniformHeight`](@ref) for the idealized equal-height array.
* `bare_soil_roughness` (`ℓˢᵒⁱˡ`, m) — momentum roughness length where the built
  fraction vanishes. Default 0.03.
* `minimum_built_fraction` — built-fraction floor below which the cell reduces to
  bare soil. Default 0.01.
* `maximum_displacement_ratio` — displacement ceiling `d/h`, below 1 to avoid the
  singular skimming limit. Default 0.95.

```jldoctest
julia> using NumericalEarth.Lands

julia> MorphometricRoughness()
MorphometricRoughness{Float64} with IsotropicFrontalArea() and VariableHeight(height_variability = 0.4, maximum_height_ratio = 2.5)

julia> MorphometricRoughness(height_distribution = UniformHeight())
MorphometricRoughness{Float64} with IsotropicFrontalArea() and UniformHeight()
```
"""
function MorphometricRoughness(FT = Oceananigans.defaults.FloatType;
                               array_constant = 4.43,
                               drag_coefficient = 1.2,
                               correction_factor = 1.0,
                               von_karman_constant = 0.4,
                               frontal_area = IsotropicFrontalArea(),
                               height_distribution = VariableHeight(FT),
                               bare_soil_roughness = 0.03,
                               minimum_built_fraction = 0.01,
                               maximum_displacement_ratio = 0.95)
    return MorphometricRoughness(convert(FT, array_constant),
                                 convert(FT, drag_coefficient),
                                 convert(FT, correction_factor),
                                 convert(FT, von_karman_constant),
                                 convert_frontal_area(FT, frontal_area),
                                 convert_height_distribution(FT, height_distribution),
                                 convert(FT, bare_soil_roughness),
                                 convert(FT, minimum_built_fraction),
                                 convert(FT, maximum_displacement_ratio))
end

Base.summary(c::MorphometricRoughness{FT}) where FT =
    string("MorphometricRoughness{", FT, "} with ", summary(c.frontal_area),
           " and ", summary(c.height_distribution))

Base.show(io::IO, c::AbstractUrbanRoughness) = print(io, summary(c))

#####
##### Common interface: (λᵖ, h) → (ℓᵐ, d)
#####

# Clamp to the physical range, floor to bare soil below the built-fraction threshold, and
# return honest NaN gaps for invalid (NaN / negative-height) inputs. Shared by all closures.
@inline function finalize_aerodynamic_parameters(ℓᵐ, d, λᵖ, valid, ℓˢᵒⁱˡ, λᵐⁱⁿ)
    ℓˢᵒⁱˡ = oftype(ℓᵐ, ℓˢᵒⁱˡ)  # unify with the computed type so a narrower-FT closure stays Union-free
    bare = λᵖ < λᵐⁱⁿ
    ℓᵐ = ifelse(bare, ℓˢᵒⁱˡ, max(ℓᵐ, ℓˢᵒⁱˡ))
    d = ifelse(bare, zero(d), d)
    gap = oftype(ℓᵐ, NaN)
    return ifelse(valid, ℓᵐ, gap), ifelse(valid, d, gap)
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length `ℓᵐ` and zero-plane displacement `d` (meters) of a built-up
surface with plan-area index `λᵖ` and mean building height `h`, under `closure`.
Returns `(ℓᵐ, d)`. Endpoints reduce cleanly: `λᵖ → 0` returns the bare-soil `(ℓˢᵒⁱˡ, 0)`;
`λᵖ → 1` is the skimming limit (`d/h` capped below 1). An invalid (`NaN`/negative) input
returns `(NaN, NaN)`.
"""
@inline function aerodynamic_parameters(c::MorphometricRoughness{FT}, λᵖ, h) where FT
    valid = isfinite(λᵖ) & isfinite(h) & (h >= 0)
    λᵖ = clamp(λᵖ, zero(FT), one(FT))
    h = max(h, zero(FT))

    dʰ = packing_displacement_ratio(λᵖ, c.array_constant, c.maximum_displacement_ratio)
    λᶠ = frontal_area_index(c.frontal_area, λᵖ, h)
    ℓᵐ = h * drag_partition_roughness_ratio(λᶠ, dʰ, c.drag_coefficient, c.von_karman_constant, c.correction_factor)
    d = h * dʰ

    ℓᵐ, d = apply_height_distribution(c.height_distribution, ℓᵐ, d, λᵖ, h, c.maximum_displacement_ratio)

    return finalize_aerodynamic_parameters(ℓᵐ, d, λᵖ, valid, c.bare_soil_roughness, c.minimum_built_fraction)
end

@inline (c::AbstractUrbanRoughness)(λᵖ, h) = aerodynamic_parameters(c, λᵖ, h)

"""
$(TYPEDSIGNATURES)

Shared grid-builder contract: evaluate the closure from a per-cell `cell` NamedTuple,
reading only this closure's own keys (`plan_area_fraction`, `building_height`).
"""
@inline aerodynamic_parameters(c::AbstractUrbanRoughness, cell) =
    aerodynamic_parameters(c, cell.plan_area_fraction, cell.building_height)
