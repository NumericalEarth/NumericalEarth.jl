#####
##### Urban morphometric roughness closures
#####
##### Momentum roughness length ℓᵐ and zero-plane displacement d of a built-up surface,
##### from the plan-area index λᵖ and mean building height h.
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

Supertype of the estimators of the frontal-area index `λᶠ` from the plan-area index `λᵖ`
and mean height `h`.

A dataset that supplies only `λᵖ` leaves `λᶠ` to be assumed from a roughness-element
shape — the dominant uncertainty of the drag partition. Where `λᶠ` is measured per cell
it should be used directly instead.
"""
abstract type AbstractFrontalAreaEstimator end

"""
    IsotropicFrontalArea()

Take `λᶠ = λᵖ`, exact for cubical elements, whose frontal and plan area densities are
equal — the shape the array constant `A` of [`MorphometricRoughness`](@ref) was fitted to.
"""
struct IsotropicFrontalArea <: AbstractFrontalAreaEstimator end

"""
$(TYPEDEF)

Take `λᶠ = q·λᵖ² + l·λᵖ`, regressed by [Kanda et al. (2013)](@cite Kanda2013) on the
building databases of Tokyo, Nagoya, Toulouse, Berlin, Salt Lake City and Los Angeles
(their eq. 1, fitted over `0.05 < λᵖ < 0.45`). Unlike [`IsotropicFrontalArea`](@ref) this
assumes no element geometry, and it respects the `λᶠ < 2λᵖ` envelope they observe across
real cities. The default, since a mean-height dataset never measures `λᶠ`.
"""
struct EmpiricalFrontalArea{FT} <: AbstractFrontalAreaEstimator
    quadratic_coefficient :: FT
    linear_coefficient    :: FT
end

EmpiricalFrontalArea(FT = Oceananigans.defaults.FloatType;
                     quadratic_coefficient = 1.42,
                     linear_coefficient = 0.4) =
    EmpiricalFrontalArea(convert(FT, quadratic_coefficient), convert(FT, linear_coefficient))

convert_frontal_area(FT, e::IsotropicFrontalArea) = e
convert_frontal_area(FT, e::EmpiricalFrontalArea) =
    EmpiricalFrontalArea(convert(FT, e.quadratic_coefficient), convert(FT, e.linear_coefficient))

Base.summary(::IsotropicFrontalArea) = "IsotropicFrontalArea()"
Base.summary(::EmpiricalFrontalArea) = "EmpiricalFrontalArea()"
Base.show(io::IO, e::AbstractFrontalAreaEstimator) = print(io, summary(e))

"""
$(TYPEDSIGNATURES)

Frontal-area index `λᶠ` of plan-area index `λᵖ` and mean height `h` under `estimator`.
"""
@inline frontal_area_index(::IsotropicFrontalArea, λᵖ, h) = λᵖ
@inline frontal_area_index(e::EmpiricalFrontalArea, λᵖ, h) =
    e.quadratic_coefficient * λᵖ^2 + e.linear_coefficient * λᵖ

#####
##### Height distributions — how the spread of roughness-element heights enters (ℓᵐ, d)
#####

"""
    abstract type AbstractHeightDistribution end

Supertype of the treatments of the roughness-element height distribution within a cell.
[`UniformHeight`](@ref) takes every element to be as tall as the mean;
[`VariableHeight`](@ref) derives the spread `σʰ` and the tallest element `hᵐᵃˣ` from the
mean and corrects the uniform-height `(ℓᵐ, d)` for them.
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
`λᵖ σʰ / h`.

A mean-height dataset measures neither `σʰ` nor `hᵐᵃˣ`, so both are derived from `h` by
the regressions the same study fitted to real cities (their eqs. 2 and 3). Pass measured
per-cell statistics instead wherever a footprint-level dataset supplies them.
"""
struct VariableHeight{FT} <: AbstractHeightDistribution
    displacement_constants   :: NTuple{3, FT}
    roughness_constants      :: NTuple{3, FT}
    height_spread_constants  :: NTuple{2, FT}
    maximum_height_constants :: NTuple{2, FT}
end

"""
    VariableHeight(FT = Oceananigans.defaults.FloatType; kw...)

Construct a [`VariableHeight`](@ref) distribution. All four defaults are the Method 1
regressions of [Kanda et al. (2013)](@cite Kanda2013).

* `displacement_constants` (`a₀, b₀, c₀`) — displacement fit, their eq. 10.
  Default `(1.29, 0.36, -0.17)`; their Method 2 gives `(0.86, 0.28, -0.18)`.
* `roughness_constants` (`a₁, b₁, c₁`) — roughness rescaling, their eq. 12.
  Default `(0.71, 20.21, -0.77)`; their Method 2 gives `(0.93, 8.93, 4.68)`.
* `height_spread_constants` (`s₁, s₀`) — `σʰ = s₁·h + s₀` (m), their eq. 2.
  Default `(1.05, -3.7)`, whose zero crossing near `h = 3.5` m is about one story.
* `maximum_height_constants` (`m₁, m₂`) — `hᵐᵃˣ = m₁·(σʰ)^m₂` (m), their eq. 3.
  Default `(12.51, 0.77)`. The most scattered of the four, since a city's tallest
  building is often an outlying landmark.

```jldoctest
julia> using NumericalEarth.Lands

julia> VariableHeight()
VariableHeight()
```
"""
function VariableHeight(FT = Oceananigans.defaults.FloatType;
                        displacement_constants = (1.29, 0.36, -0.17),
                        roughness_constants = (0.71, 20.21, -0.77),
                        height_spread_constants = (1.05, -3.7),
                        maximum_height_constants = (12.51, 0.77))
    return VariableHeight(convert.(FT, displacement_constants),
                          convert.(FT, roughness_constants),
                          convert.(FT, height_spread_constants),
                          convert.(FT, maximum_height_constants))
end

convert_height_distribution(FT, d::UniformHeight) = d
convert_height_distribution(FT, d::VariableHeight) =
    VariableHeight(convert.(FT, d.displacement_constants),
                   convert.(FT, d.roughness_constants),
                   convert.(FT, d.height_spread_constants),
                   convert.(FT, d.maximum_height_constants))

Base.summary(::UniformHeight) = "UniformHeight()"
Base.summary(::VariableHeight) = "VariableHeight()"
Base.show(io::IO, d::AbstractHeightDistribution) = print(io, summary(d))

#####
##### Pure morphometric ratios (unit-testable, shared by the closures)
#####

"""
$(TYPEDSIGNATURES)

Displacement ratio `d/h = 1 + A^(−λᵖ)·(λᵖ − 1)` of an obstacle array packed to plan-area
index `λᵖ`, clamped to `[0, dᵐᵃˣ]`. Empirical fit of
[Macdonald et al. (1998)](@cite Macdonald1998), satisfying `d/h ≥ λᵖ` and running from
`0` at `λᵖ = 0` to `1` at `λᵖ = 1`. Capping below 1 holds the displacement under roof
level, and keeps `ℓᵐ` off the zero it would reach where the packed array closes into a
smooth surface again.
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
`d/hᵐᵃˣ = c₀·X² + (a₀·λᵖ^b₀ − c₀)·X`, with `X = (σʰ + h)/hᵐᵃˣ` in `[0, 1]`
([Kanda et al. (2013)](@cite Kanda2013)).
"""
@inline function maximum_height_displacement(λᵖ, h, σʰ, hᵐᵃˣ, a₀, b₀, c₀)
    X = clamp(ifelse(hᵐᵃˣ > 0, (σʰ + h) / hᵐᵃˣ, zero(h)), zero(h), one(h))
    ratio = c₀ * X^2 + (a₀ * λᵖ^b₀ - c₀) * X
    return hᵐᵃˣ * max(ratio, zero(ratio))
end

"""
$(TYPEDSIGNATURES)

Rescale the uniform-height roughness length `ℓᵐ` (m) for the height spread `σʰ` by
`b₁·Y² + c₁·Y + a₁`, with `Y = λᵖ·σʰ/h` ([Kanda et al. (2013)](@cite Kanda2013)). Reduces
to `a₁·ℓᵐ` for a height-homogeneous canopy (`σʰ → 0`).
"""
@inline function height_spread_roughness(ℓᵐ, λᵖ, h, σʰ, a₁, b₁, c₁)
    Y = ifelse(h > 0, λᵖ * σʰ / h, zero(h))
    ratio = b₁ * Y^2 + c₁ * Y + a₁
    return ℓᵐ * max(ratio, zero(ratio))
end

"""
$(TYPEDSIGNATURES)

Height standard deviation `σʰ` (m) of a cell of mean element height `h`, from the affine
regression `σʰ = s₁·h + s₀`, floored at zero below its one-story zero crossing.
"""
@inline function height_spread(v::VariableHeight, h)
    s₁, s₀ = v.height_spread_constants
    return max(s₁ * h + s₀, zero(h))
end

"""
$(TYPEDSIGNATURES)

Tallest element `hᵐᵃˣ` (m) of a cell of mean height `h` and height spread `σʰ`, from the
power-law regression `hᵐᵃˣ = m₁·(σʰ)^m₂`, floored at `σʰ + h` so the displacement
parameter `X` stays within its `[0, 1]` range.
"""
@inline function maximum_element_height(v::VariableHeight, h, σʰ)
    m₁, m₂ = v.maximum_height_constants
    return max(m₁ * σʰ^m₂, σʰ + h)
end

"""
$(TYPEDSIGNATURES)

Apply the height `distribution` to the uniform-height parameters `(ℓᵐ, d)` of a cell with
plan-area index `λᵖ` and mean height `h`.
"""
@inline apply_height_distribution(::UniformHeight, ℓᵐ, d, λᵖ, h) = ℓᵐ, d

@inline function apply_height_distribution(v::VariableHeight, ℓᵐ, d, λᵖ, h)
    σʰ = height_spread(v, h)
    hᵐᵃˣ = maximum_element_height(v, h, σʰ)
    return apply_height_distribution(v, ℓᵐ, d, λᵖ, h, σʰ, hᵐᵃˣ)
end

"""
$(TYPEDSIGNATURES)

Apply the height `distribution` to the uniform-height parameters `(ℓᵐ, d)` of a cell whose
height spread `σʰ` and tallest element `hᵐᵃˣ` are measured rather than regressed from `h`.
"""
@inline apply_height_distribution(::UniformHeight, ℓᵐ, d, λᵖ, h, σʰ, hᵐᵃˣ) = ℓᵐ, d

@inline function apply_height_distribution(v::VariableHeight, ℓᵐ, d, λᵖ, h, σʰ, hᵐᵃˣ)
    a₀, b₀, c₀ = v.displacement_constants
    a₁, b₁, c₁ = v.roughness_constants

    ℓᵐ = height_spread_roughness(ℓᵐ, λᵖ, h, σʰ, a₁, b₁, c₁)

    # `hᵐᵃˣ`, not `h`, bounds the displacement: over a height-heterogeneous city `d`
    # routinely exceeds the mean building height.
    d = min(maximum_height_displacement(λᵖ, h, σʰ, hᵐᵃˣ, a₀, b₀, c₀), hᵐᵃˣ)

    return ℓᵐ, d
end

#####
##### Closures
#####

"""
$(TYPEDEF)

Morphometric roughness closure. Maps `(λᵖ, h)` to `(ℓᵐ, d)` from the plan-area index and
mean building height alone: `d` from the plan-area packing, `ℓᵐ` from a drag partition over
the frontal area `λᶠ` estimated by `frontal_area_estimator`, both then corrected for the spread of
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
    frontal_area_estimator     :: E
    height_distribution        :: H
    bare_soil_roughness        :: FT
    minimum_built_fraction     :: FT
    maximum_displacement_ratio :: FT
end

"""
    MorphometricRoughness(FT = Oceananigans.defaults.FloatType; kw...)

Construct a [`MorphometricRoughness`](@ref) closure.

* `array_constant` (`A`) — default 4.43, the staggered-array fit. Under
  [`UniformHeight`](@ref) the square-array fit is 3.59, but it pairs with
  `correction_factor = 0.55`: sheltering is not captured by `A` alone, and 3.59 on its own
  overpredicts `ℓᵐ` by about a factor of two. Leave both at their defaults under
  [`VariableHeight`](@ref), whose constants were regressed with `A = 4.43` and `β = 1`
  held fixed for either array type.
* `drag_coefficient` (`Cᴰ`) — building drag coefficient. Default 1.2, for a sharp-edged
  cube; rounded or cylindrical elements are lower by up to a factor of two.
* `correction_factor` (`β`) — lumped drag correction, folding in the velocity-profile
  shape, turbulence intensity and length scale, incidence angle and corner rounding.
  Default 1 reproduces the staggered array; 0.55 is the square-array value.
* `von_karman_constant` (`ϰ`) — default 0.4.
* `frontal_area_estimator` — frontal-area estimator: [`EmpiricalFrontalArea`](@ref) (default),
  [`IsotropicFrontalArea`](@ref) for the idealized cube array.
* `height_distribution` — height-spread treatment: [`VariableHeight`](@ref) (default) or
  [`UniformHeight`](@ref) for the idealized equal-height array.
* `bare_soil_roughness` (`ℓˢᵒⁱˡ`, m) — momentum roughness length where the built
  fraction vanishes. Default 0.03.
* `minimum_built_fraction` — built-fraction floor below which the cell reduces to
  bare soil. Default 0.01.
* `maximum_displacement_ratio` — ceiling on the packing displacement ratio `d/h` that
  feeds the drag partition. Default 1, the value at which `d/h → 1` and `ℓᵐ → 0` as the
  array closes into a smooth surface. Capping below 1 freezes the `(1 − d/h)` suppression
  and makes `ℓᵐ` turn back up in the skimming limit. Under [`VariableHeight`](@ref) the
  returned displacement is bounded by `hᵐᵃˣ` rather than by this ratio.

```jldoctest
julia> using NumericalEarth.Lands

julia> MorphometricRoughness()
MorphometricRoughness{Float64} with EmpiricalFrontalArea() and VariableHeight()

julia> MorphometricRoughness(height_distribution = UniformHeight())
MorphometricRoughness{Float64} with EmpiricalFrontalArea() and UniformHeight()
```
"""
function MorphometricRoughness(FT = Oceananigans.defaults.FloatType;
                               array_constant = 4.43,
                               drag_coefficient = 1.2,
                               correction_factor = 1.0,
                               von_karman_constant = 0.4,
                               frontal_area_estimator = EmpiricalFrontalArea(FT),
                               height_distribution = VariableHeight(FT),
                               bare_soil_roughness = 0.03,
                               minimum_built_fraction = 0.01,
                               maximum_displacement_ratio = 1.0)
    return MorphometricRoughness(convert(FT, array_constant),
                                 convert(FT, drag_coefficient),
                                 convert(FT, correction_factor),
                                 convert(FT, von_karman_constant),
                                 convert_frontal_area(FT, frontal_area_estimator),
                                 convert_height_distribution(FT, height_distribution),
                                 convert(FT, bare_soil_roughness),
                                 convert(FT, minimum_built_fraction),
                                 convert(FT, maximum_displacement_ratio))
end

Base.summary(c::MorphometricRoughness{FT}) where FT =
    string("MorphometricRoughness{", FT, "} with ", summary(c.frontal_area_estimator),
           " and ", summary(c.height_distribution))

Base.show(io::IO, c::AbstractUrbanRoughness) = print(io, summary(c))

#####
##### Common interface: (λᵖ, h) → (ℓᵐ, d)
#####

@inline function finalize_aerodynamic_parameters(ℓᵐ, d, λᵖ, valid, ℓˢᵒⁱˡ, λᵖᵐⁱⁿ)
    ℓˢᵒⁱˡ = oftype(ℓᵐ, ℓˢᵒⁱˡ)  # unify with the computed type so a narrower-FT closure stays Union-free
    bare = λᵖ < λᵖᵐⁱⁿ
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
`λᵖ → 1` is the skimming limit. An invalid (`NaN`/negative) input returns `(NaN, NaN)`.
Under [`VariableHeight`](@ref) the displacement is bounded by the tallest element rather
than by `h`, and routinely exceeds the mean building height.
"""
@inline function aerodynamic_parameters(c::MorphometricRoughness{FT}, λᵖ, h) where FT
    valid = isfinite(λᵖ) & isfinite(h) & (h >= 0)
    λᵖ = clamp(λᵖ, zero(FT), one(FT))
    h = max(h, zero(FT))

    dʰ = packing_displacement_ratio(λᵖ, c.array_constant, c.maximum_displacement_ratio)
    λᶠ = frontal_area_index(c.frontal_area_estimator, λᵖ, h)
    ℓᵐ = h * drag_partition_roughness_ratio(λᶠ, dʰ, c.drag_coefficient, c.von_karman_constant, c.correction_factor)
    d = h * dʰ

    ℓᵐ, d = apply_height_distribution(c.height_distribution, ℓᵐ, d, λᵖ, h)

    return finalize_aerodynamic_parameters(ℓᵐ, d, λᵖ, valid, c.bare_soil_roughness, c.minimum_built_fraction)
end

@inline (c::AbstractUrbanRoughness)(λᵖ, h) = aerodynamic_parameters(c, λᵖ, h)

"""
$(TYPEDSIGNATURES)

Evaluate `closure` from a per-cell `cell` NamedTuple, reading its `plan_area_index`
and `mean_building_height` keys.
"""
@inline aerodynamic_parameters(c::AbstractUrbanRoughness, cell) =
    aerodynamic_parameters(c, cell.plan_area_index, cell.mean_building_height)

#####
##### Measured-morphometry interface: (λᵖ, h, σʰ, hᵐᵃˣ, λᶠ) → (ℓᵐ, d)
#####

"""
$(TYPEDSIGNATURES)

Momentum roughness length `ℓᵐ` and zero-plane displacement `d` (meters) from **measured**
per-cell morphometry: the plan-area index `λᵖ`, mean building height `h`, height standard
deviation `σʰ`, tallest element `hᵐᵃˣ` and frontal-area index `λᶠ`, as a footprint-level
dataset supplies them. The `frontal_area_estimator` and the `σʰ`/`hᵐᵃˣ` regressions of
[`VariableHeight`](@ref) drop out; the drag partition and the height-spread corrections
themselves remain. `hᵐᵃˣ` is floored at `h`, and the displacement parameter
`X = (σʰ + h)/hᵐᵃˣ` keeps its `[0, 1]` clamp, so a low-biased maximum height cannot invert
the correction. Under [`UniformHeight`](@ref) `σʰ` and `hᵐᵃˣ` are ignored and the result
is the obstacle-array closure with measured `λᶠ`.
"""
@inline function aerodynamic_parameters(c::MorphometricRoughness{FT}, λᵖ, h, σʰ, hᵐᵃˣ, λᶠ) where FT
    valid = isfinite(λᵖ) & isfinite(h) & isfinite(σʰ) & isfinite(hᵐᵃˣ) & isfinite(λᶠ) & (h >= 0)
    λᵖ = clamp(λᵖ, zero(FT), one(FT))
    h = max(h, zero(FT))
    σʰ = max(σʰ, zero(FT))
    hᵐᵃˣ = max(hᵐᵃˣ, h)
    λᶠ = max(λᶠ, zero(FT))

    dʰ = packing_displacement_ratio(λᵖ, c.array_constant, c.maximum_displacement_ratio)
    ℓᵐ = h * drag_partition_roughness_ratio(λᶠ, dʰ, c.drag_coefficient, c.von_karman_constant, c.correction_factor)
    d = h * dʰ

    ℓᵐ, d = apply_height_distribution(c.height_distribution, ℓᵐ, d, λᵖ, h, σʰ, hᵐᵃˣ)

    return finalize_aerodynamic_parameters(ℓᵐ, d, λᵖ, valid, c.bare_soil_roughness, c.minimum_built_fraction)
end

@inline (c::AbstractUrbanRoughness)(λᵖ, h, σʰ, hᵐᵃˣ, λᶠ) = aerodynamic_parameters(c, λᵖ, h, σʰ, hᵐᵃˣ, λᶠ)
