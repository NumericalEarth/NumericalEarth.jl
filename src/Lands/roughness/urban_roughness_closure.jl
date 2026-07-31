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
modeling uncertainty of the Macdonald route, hence it is a selectable closure field.
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
##### Pure morphometric ratios (unit-testable, shared by the closures)
#####

"""
$(TYPEDSIGNATURES)

Macdonald et al. (1998) displacement ratio `d/h = 1 + A^(−λᵖ)·(λᵖ − 1)`, clamped to
`[0, dᵐᵃˣ]` (`dᵐᵃˣ < 1` avoids the singular skimming limit `λᵖ → 1`).
"""
@inline function macdonald_displacement_ratio(λᵖ, A, dᵐᵃˣ)
    dʰ = 1 + A^(-λᵖ) * (λᵖ - 1)
    return clamp(dʰ, zero(dʰ), dᵐᵃˣ)
end

"""
$(TYPEDSIGNATURES)

Macdonald et al. (1998) roughness ratio
`ℓᵐ/h = (1 − dʰ)·exp{ −[0.5·β·(Cᴰ/ϰ²)·(1 − dʰ)·λᶠ]^(−1/2) }`. Non-monotonic in `λᵖ`:
`ℓᵐ` rises then falls (isolated → wake-interference → skimming) with the frontal area `λᶠ`.
"""
@inline function macdonald_roughness_ratio(λᶠ, dʰ, Cᴰ, ϰ, β)
    bracket = (β * Cᴰ) / (2 * ϰ^2) * (1 - dʰ) * λᶠ
    decay = ifelse(bracket > 0, exp(-inv(sqrt(bracket))), zero(bracket))
    return (1 - dʰ) * decay
end

"""
$(TYPEDSIGNATURES)

Kanda et al. (2013) displacement height `d` (m), which references the maximum building
height `hᵐᵃˣ` and the height standard deviation `σʰ`:
`d/hᵐᵃˣ = c0·X² + (a0·λᵖ^b0 − c0)·X`, with `X = (σʰ + h)/hᵐᵃˣ` in `[0, 1]`.
"""
@inline function kanda_displacement_height(λᵖ, h, σʰ, hᵐᵃˣ, a0, b0, c0)
    X = clamp(ifelse(hᵐᵃˣ > 0, (σʰ + h) / hᵐᵃˣ, zero(h)), zero(h), one(h))
    ratio = c0 * X^2 + (a0 * λᵖ^b0 - c0) * X
    return hᵐᵃˣ * max(ratio, zero(ratio))
end

"""
$(TYPEDSIGNATURES)

Kanda et al. (2013) roughness length (m), a correction of the Macdonald value `ℓᵐ`:
`b1·Y² + c1·Y + a1` times `ℓᵐ`, with `Y = λᵖ·σʰ/h`. Reduces to `a1·ℓᵐ` for a
height-homogeneous canopy (`σʰ → 0`).
"""
@inline function kanda_roughness_length(ℓᵐ, λᵖ, h, σʰ, a1, b1, c1)
    Y = ifelse(h > 0, λᵖ * σʰ / h, zero(h))
    ratio = b1 * Y^2 + c1 * Y + a1
    return ℓᵐ * max(ratio, zero(ratio))
end

#####
##### Closures
#####

"""
$(TYPEDEF)

Macdonald et al. (1998) morphometric roughness closure (staggered-array defaults). Maps
`(λᵖ, h)` to `(ℓᵐ, d)` from the plan-area index and mean building height alone, estimating
the frontal area `λᶠ` with `frontal_area`. `ℓᵐ` is non-monotonic in `λᵖ` (isolated →
wake-interference → skimming).
"""
struct MacdonaldRoughness{FT, E} <: AbstractUrbanRoughness
    array_constant             :: FT
    drag_coefficient           :: FT
    correction_factor          :: FT
    von_karman_constant        :: FT
    frontal_area               :: E
    bare_soil_roughness        :: FT
    minimum_built_fraction     :: FT
    maximum_displacement_ratio :: FT
end

"""
    MacdonaldRoughness(FT = Oceananigans.defaults.FloatType; kw...)

Construct a [`MacdonaldRoughness`](@ref) closure.

* `array_constant` (`A`) — default 4.43 (staggered array); use 3.59 for a square one.
* `drag_coefficient` (`Cᴰ`) — building drag coefficient. Default 1.2.
* `correction_factor` (`β`) — Macdonald correction factor. Default 1.
* `von_karman_constant` (`ϰ`) — default 0.4.
* `frontal_area` — frontal-area estimator: [`IsotropicFrontalArea`](@ref) (default) or
  [`CuboidFrontalArea`](@ref).
* `bare_soil_roughness` (`ℓˢᵒⁱˡ`, m) — momentum roughness length where the built
  fraction vanishes. Default 0.03.
* `minimum_built_fraction` — built-fraction floor below which the cell reduces to
  bare soil. Default 0.01.
* `maximum_displacement_ratio` — displacement ceiling `d/h`, below 1 to avoid the
  singular skimming limit. Default 0.95.

```jldoctest
julia> using NumericalEarth.Lands

julia> MacdonaldRoughness()
MacdonaldRoughness{Float64} with IsotropicFrontalArea()
```
"""
function MacdonaldRoughness(FT = Oceananigans.defaults.FloatType;
                            array_constant = 4.43,
                            drag_coefficient = 1.2,
                            correction_factor = 1.0,
                            von_karman_constant = 0.4,
                            frontal_area = IsotropicFrontalArea(),
                            bare_soil_roughness = 0.03,
                            minimum_built_fraction = 0.01,
                            maximum_displacement_ratio = 0.95)
    return MacdonaldRoughness(convert(FT, array_constant),
                              convert(FT, drag_coefficient),
                              convert(FT, correction_factor),
                              convert(FT, von_karman_constant),
                              convert_frontal_area(FT, frontal_area),
                              convert(FT, bare_soil_roughness),
                              convert(FT, minimum_built_fraction),
                              convert(FT, maximum_displacement_ratio))
end

Base.summary(c::MacdonaldRoughness{FT}) where FT =
    string("MacdonaldRoughness{", FT, "} with ", summary(c.frontal_area))

"""
$(TYPEDEF)

Kanda et al. (2013) height-heterogeneity roughness closure, and the default urban closure.
Corrects the Macdonald `ℓᵐ` of the wrapped [`MacdonaldRoughness`](@ref) with the
building-height spread, and takes the displacement height from the Kanda formula
referencing the maximum height. The bare-soil floor, built-fraction floor and displacement
ceiling are inherited from the wrapped Macdonald closure.
"""
struct KandaRoughness{FT, M} <: AbstractUrbanRoughness
    macdonald              :: M
    displacement_constants :: NTuple{3, FT}
    roughness_constants    :: NTuple{3, FT}
    height_variability     :: FT
    maximum_height_ratio   :: FT
end

"""
    KandaRoughness(FT = Oceananigans.defaults.FloatType; kw...)

Construct a [`KandaRoughness`](@ref) closure.

* `macdonald` — the [`MacdonaldRoughness`](@ref) closure supplying the base roughness
  `ℓᵐ`, the frontal-area estimator and the floors. Pass a configured one to change the
  array constant, estimator or floors. Default `MacdonaldRoughness(FT)`.
* `displacement_constants` (`a0, b0, c0`) — Kanda displacement constants.
  Default `(1.29, 0.36, -0.17)`.
* `roughness_constants` (`a1, b1, c1`) — Kanda roughness constants.
  Default `(0.71, 20.21, -0.77)`.
* `height_variability` (`σʰ / h`) — assumed height standard deviation as a fraction
  of the mean height. Default 0.4.
* `maximum_height_ratio` (`hᵐᵃˣ / h`) — assumed maximum-to-mean building height ratio.
  Default 2.5.

```jldoctest
julia> using NumericalEarth.Lands

julia> KandaRoughness()
KandaRoughness{Float64} correcting MacdonaldRoughness{Float64} with IsotropicFrontalArea()
```
"""
function KandaRoughness(FT = Oceananigans.defaults.FloatType;
                        macdonald = MacdonaldRoughness(FT),
                        displacement_constants = (1.29, 0.36, -0.17),
                        roughness_constants = (0.71, 20.21, -0.77),
                        height_variability = 0.4,
                        maximum_height_ratio = 2.5)
    return KandaRoughness(macdonald,
                          convert.(FT, displacement_constants),
                          convert.(FT, roughness_constants),
                          convert(FT, height_variability),
                          convert(FT, maximum_height_ratio))
end

Base.summary(c::KandaRoughness{FT}) where FT =
    string("KandaRoughness{", FT, "} correcting ", summary(c.macdonald))

"""
$(TYPEDEF)

Rule-of-thumb roughness closure: `ℓᵐ = ℓˢᵒⁱˡ + fz·h` and `d = fd·h`, a coarse fallback
where the morphometric routes are not warranted.
"""
struct LookupRoughness{FT} <: AbstractUrbanRoughness
    bare_soil_roughness          :: FT
    roughness_height_fraction    :: FT
    displacement_height_fraction :: FT
    minimum_built_fraction       :: FT
end

"""
    LookupRoughness(FT = Oceananigans.defaults.FloatType; kw...)

Construct a [`LookupRoughness`](@ref) closure.

* `bare_soil_roughness` (`ℓˢᵒⁱˡ`, m) — momentum roughness length where the built
  fraction vanishes. Default 0.03.
* `roughness_height_fraction` (`fz`) — default 0.1.
* `displacement_height_fraction` (`fd`) — default 0.7.
* `minimum_built_fraction` — built-fraction floor below which the cell reduces to
  bare soil. Default 0.01.

```jldoctest
julia> using NumericalEarth.Lands

julia> LookupRoughness()
LookupRoughness{Float64}
```
"""
function LookupRoughness(FT = Oceananigans.defaults.FloatType;
                         bare_soil_roughness = 0.03,
                         roughness_height_fraction = 0.1,
                         displacement_height_fraction = 0.7,
                         minimum_built_fraction = 0.01)
    return LookupRoughness(convert(FT, bare_soil_roughness),
                           convert(FT, roughness_height_fraction),
                           convert(FT, displacement_height_fraction),
                           convert(FT, minimum_built_fraction))
end

Base.summary(::LookupRoughness{FT}) where FT = string("LookupRoughness{", FT, "}")

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
@inline function aerodynamic_parameters(c::MacdonaldRoughness{FT}, λᵖ, h) where FT
    valid = isfinite(λᵖ) & isfinite(h) & (h >= 0)
    λᵖ = clamp(λᵖ, zero(FT), one(FT))
    h = max(h, zero(FT))

    dʰ = macdonald_displacement_ratio(λᵖ, c.array_constant, c.maximum_displacement_ratio)
    λᶠ = frontal_area_index(c.frontal_area, λᵖ, h)
    ℓᵐ = h * macdonald_roughness_ratio(λᶠ, dʰ, c.drag_coefficient, c.von_karman_constant, c.correction_factor)
    d = h * dʰ

    return finalize_aerodynamic_parameters(ℓᵐ, d, λᵖ, valid, c.bare_soil_roughness, c.minimum_built_fraction)
end

@inline function aerodynamic_parameters(c::KandaRoughness{FT}, λᵖ, h) where FT
    m = c.macdonald
    valid = isfinite(λᵖ) & isfinite(h) & (h >= 0)
    λᵖ = clamp(λᵖ, zero(FT), one(FT))
    h = max(h, zero(FT))

    σʰ = c.height_variability * h
    hᵐᵃˣ = c.maximum_height_ratio * h

    dʰ = macdonald_displacement_ratio(λᵖ, m.array_constant, m.maximum_displacement_ratio)
    λᶠ = frontal_area_index(m.frontal_area, λᵖ, h)
    ℓᵐ = h * macdonald_roughness_ratio(λᶠ, dʰ, m.drag_coefficient, m.von_karman_constant, m.correction_factor)

    a0, b0, c0 = c.displacement_constants
    a1, b1, c1 = c.roughness_constants
    ℓᵐ = kanda_roughness_length(ℓᵐ, λᵖ, h, σʰ, a1, b1, c1)
    d = min(kanda_displacement_height(λᵖ, h, σʰ, hᵐᵃˣ, a0, b0, c0), m.maximum_displacement_ratio * h)

    return finalize_aerodynamic_parameters(ℓᵐ, d, λᵖ, valid, m.bare_soil_roughness, m.minimum_built_fraction)
end

@inline function aerodynamic_parameters(c::LookupRoughness{FT}, λᵖ, h) where FT
    valid = isfinite(λᵖ) & isfinite(h) & (h >= 0)
    λᵖ = clamp(λᵖ, zero(FT), one(FT))
    h = max(h, zero(FT))

    ℓᵐ = c.bare_soil_roughness + c.roughness_height_fraction * h
    d = c.displacement_height_fraction * h

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
