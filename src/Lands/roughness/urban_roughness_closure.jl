#####
##### Urban morphometric roughness closures
#####
##### Aerodynamic momentum roughness length z0 and zero-plane displacement d0 for the
##### urban (built-up) surface, from the plan-area fraction λp and mean building height H.
##### Each closure is a distinct type dispatched by `aerodynamic_parameters`, tied together by
##### `AbstractUrbanRoughness`.
#####

"""
    AbstractUrbanRoughness

Supertype of the urban morphometric roughness closures. A closure maps the plan-area
built fraction `λp` and mean building height `h` to the momentum roughness length `z0`
and zero-plane displacement `d0` via [`aerodynamic_parameters`](@ref); it is also callable as
`closure(λp, h)`.
"""
abstract type AbstractUrbanRoughness end

#####
##### Frontal-area estimators — λf is not observed by GHSL, so it is estimated from (λp, h)
#####

"""
    AbstractFrontalAreaEstimator

Supertype of the estimators of the frontal-area index `λf` from the plan-area fraction
`λp` and mean height `h` (which GHSL does not observe). This estimate is the dominant
modeling uncertainty of the Macdonald route, hence it is a selectable closure field.
"""
abstract type AbstractFrontalAreaEstimator end

"""
    IsotropicFrontalArea()

Take `λf ≈ λp` (cubic elements).
"""
struct IsotropicFrontalArea <: AbstractFrontalAreaEstimator end

"""
$(TYPEDEF)

Take `λf = λp · h / Lb` for square buildings of width `Lb` (`building_width`, m).

$(TYPEDFIELDS)
"""
struct CuboidFrontalArea{FT} <: AbstractFrontalAreaEstimator
    "characteristic building width `Lb` (m)"
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

Frontal-area index `λf` of built fraction `λp` and mean height `h` under `estimator`.
"""
@inline frontal_area_index(::IsotropicFrontalArea, λp, h) = λp
@inline frontal_area_index(e::CuboidFrontalArea, λp, h) =
    ifelse(e.building_width > 0, λp * h / e.building_width, λp)

#####
##### Pure morphometric ratios (unit-testable, shared by the closures)
#####

"""
$(TYPEDSIGNATURES)

Macdonald et al. (1998) displacement ratio `d0/h = 1 + A^(−λp)·(λp − 1)`, clamped to
`[0, dₘₐₓ]` (`dₘₐₓ < 1` avoids the singular skimming limit `λp → 1`).
"""
@inline function macdonald_displacement_ratio(λp, A, dmax)
    d0h = 1 + A^(-λp) * (λp - 1)
    return clamp(d0h, zero(d0h), dmax)
end

"""
$(TYPEDSIGNATURES)

Macdonald et al. (1998) roughness ratio
`z0/h = (1 − d0h)·exp{ −[0.5·β·(Cd/κ²)·(1 − d0h)·λf]^(−1/2) }`. Non-monotonic in `λp`:
`z0` rises then falls (isolated → wake-interference → skimming) with the frontal area `λf`.
"""
@inline function macdonald_roughness_ratio(λf, d0h, Cd, κ, β)
    bracket = (β * Cd) / (2 * κ^2) * (1 - d0h) * λf
    decay = ifelse(bracket > 0, exp(-inv(sqrt(bracket))), zero(bracket))
    return (1 - d0h) * decay
end

"""
$(TYPEDSIGNATURES)

Kanda et al. (2013) displacement height `d0` (m), which references the maximum building
height `hmax` and the height standard deviation `σh`:
`d0/hmax = c0·X² + (a0·λp^b0 − c0)·X`, with `X = (σh + h)/hmax` in `[0, 1]`.
"""
@inline function kanda_displacement_height(λp, h, σh, hmax, a0, b0, c0)
    X = clamp(ifelse(hmax > 0, (σh + h) / hmax, zero(h)), zero(h), one(h))
    d0hmax = c0 * X^2 + (a0 * λp^b0 - c0) * X
    return hmax * max(d0hmax, zero(d0hmax))
end

"""
$(TYPEDSIGNATURES)

Kanda et al. (2013) roughness length `z0` (m), a correction of the Macdonald value `z0ᵐ`:
`z0/z0ᵐ = b1·Y² + c1·Y + a1`, with `Y = λp·σh/h`. Reduces to `a1·z0ᵐ` for a
height-homogeneous canopy (`σh → 0`).
"""
@inline function kanda_roughness_length(z0ᵐ, λp, h, σh, a1, b1, c1)
    Y = ifelse(h > 0, λp * σh / h, zero(h))
    ratio = b1 * Y^2 + c1 * Y + a1
    return z0ᵐ * max(ratio, zero(ratio))
end

#####
##### Closures
#####

"""
$(TYPEDEF)

Macdonald et al. (1998) morphometric roughness closure (staggered-array defaults). Maps
`(λp, h)` to `(z0, d0)` from the plan-area fraction and mean height alone, estimating the
frontal area `λf` with `frontal_area`. `z0` is non-monotonic in `λp` (isolated →
wake-interference → skimming).

$(TYPEDFIELDS)
"""
struct MacdonaldRoughness{FT, E} <: AbstractUrbanRoughness
    "Macdonald array constant `A` (4.43 staggered, 3.59 square)"
    array_constant :: FT
    "building drag coefficient `Cd`"
    drag_coefficient :: FT
    "Macdonald correction factor `β`"
    correction_factor :: FT
    "von Kármán constant `κ`"
    von_karman_constant :: FT
    "frontal-area estimator (`IsotropicFrontalArea` or `CuboidFrontalArea`)"
    frontal_area :: E
    "bare-soil momentum roughness length (m) where the built fraction vanishes"
    bare_soil_roughness :: FT
    "built-fraction floor below which the cell reduces to bare soil"
    minimum_built_fraction :: FT
    "displacement ceiling `d0/h` (< 1, avoids the singular skimming limit)"
    maximum_displacement_ratio :: FT
end

"""
    MacdonaldRoughness(FT = Oceananigans.defaults.FloatType; kw...)

Construct a [`MacdonaldRoughness`](@ref) closure. Keyword arguments override the fields
documented above.

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

Kanda et al. (2013) height-heterogeneity roughness closure. Corrects the Macdonald `z0`
of the wrapped [`MacdonaldRoughness`](@ref) with the building-height spread, and takes the
displacement height from the Kanda formula referencing the maximum height. The default
urban closure. The bare-soil floor, built-fraction floor and displacement ceiling are
inherited from the wrapped Macdonald closure.

$(TYPEDFIELDS)
"""
struct KandaRoughness{FT, M} <: AbstractUrbanRoughness
    "Macdonald closure supplying the base roughness `z0ᵐ` and the frontal-area estimator"
    macdonald :: M
    "Kanda displacement constants `(a0, b0, c0)`"
    displacement_constants :: NTuple{3, FT}
    "Kanda roughness constants `(a1, b1, c1)`"
    roughness_constants :: NTuple{3, FT}
    "assumed height standard deviation as a fraction of mean height, `σH / H`"
    height_variability :: FT
    "assumed maximum-to-mean building height ratio, `Hmax / H`"
    maximum_height_ratio :: FT
end

"""
    KandaRoughness(FT = Oceananigans.defaults.FloatType; macdonald = MacdonaldRoughness(FT), kw...)

Construct a [`KandaRoughness`](@ref) closure. Pass a configured `macdonald` closure to
change the array constant, frontal-area estimator or floors; remaining keywords override
the Kanda fields documented above.

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

Rule-of-thumb roughness closure: `z0 = z0_soil + fz·h` and `d0 = fd·h`, a coarse fallback
where the morphometric routes are not warranted.

$(TYPEDFIELDS)
"""
struct LookupRoughness{FT} <: AbstractUrbanRoughness
    "bare-soil momentum roughness length (m) where the built fraction vanishes"
    bare_soil_roughness :: FT
    "roughness height fraction `fz` (`z0 = z0_soil + fz·h`)"
    roughness_height_fraction :: FT
    "displacement height fraction `fd` (`d0 = fd·h`)"
    displacement_height_fraction :: FT
    "built-fraction floor below which the cell reduces to bare soil"
    minimum_built_fraction :: FT
end

"""
    LookupRoughness(FT = Oceananigans.defaults.FloatType; kw...)

Construct a [`LookupRoughness`](@ref) closure. Keyword arguments override the fields
documented above.

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
##### Common interface: (λp, h) → (z0, d0)
#####

# Clamp to the physical range, floor to bare soil below the built-fraction threshold, and
# return honest NaN gaps for invalid (NaN / negative-height) inputs. Shared by all closures.
@inline function finalize_aerodynamic_parameters(z0, d0, λ, valid, z0ˢ, λmin)
    z0ˢ = oftype(z0, z0ˢ)  # unify with the computed type so a narrower-FT closure stays Union-free
    bare = λ < λmin
    z0 = ifelse(bare, z0ˢ, max(z0, z0ˢ))
    d0 = ifelse(bare, zero(d0), d0)
    gap = oftype(z0, NaN)
    return ifelse(valid, z0, gap), ifelse(valid, d0, gap)
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length `z0` and zero-plane displacement `d0` (meters) of a built-up
surface with plan-area fraction `λp` and mean building height `h`, under `closure`.
Returns `(z0, d0)`. Endpoints reduce cleanly: `λp → 0` returns the bare-soil `(z0, 0)`;
`λp → 1` is the skimming limit (`d0/h` capped below 1). An invalid (`NaN`/negative) input
returns `(NaN, NaN)`.
"""
@inline function aerodynamic_parameters(c::MacdonaldRoughness{FT}, λp, h) where FT
    valid = isfinite(λp) & isfinite(h) & (h >= 0)
    λ = clamp(λp, zero(FT), one(FT))
    h = max(h, zero(FT))

    d0h = macdonald_displacement_ratio(λ, c.array_constant, c.maximum_displacement_ratio)
    λf = frontal_area_index(c.frontal_area, λ, h)
    z0 = h * macdonald_roughness_ratio(λf, d0h, c.drag_coefficient, c.von_karman_constant, c.correction_factor)
    d0 = h * d0h

    return finalize_aerodynamic_parameters(z0, d0, λ, valid, c.bare_soil_roughness, c.minimum_built_fraction)
end

@inline function aerodynamic_parameters(c::KandaRoughness{FT}, λp, h) where FT
    m = c.macdonald
    valid = isfinite(λp) & isfinite(h) & (h >= 0)
    λ = clamp(λp, zero(FT), one(FT))
    h = max(h, zero(FT))

    σh = c.height_variability * h
    hmax = c.maximum_height_ratio * h

    d0hᵐ = macdonald_displacement_ratio(λ, m.array_constant, m.maximum_displacement_ratio)
    λf = frontal_area_index(m.frontal_area, λ, h)
    z0ᵐ = h * macdonald_roughness_ratio(λf, d0hᵐ, m.drag_coefficient, m.von_karman_constant, m.correction_factor)

    a0, b0, c0 = c.displacement_constants
    a1, b1, c1 = c.roughness_constants
    z0 = kanda_roughness_length(z0ᵐ, λ, h, σh, a1, b1, c1)
    d0 = min(kanda_displacement_height(λ, h, σh, hmax, a0, b0, c0), m.maximum_displacement_ratio * h)

    return finalize_aerodynamic_parameters(z0, d0, λ, valid, m.bare_soil_roughness, m.minimum_built_fraction)
end

@inline function aerodynamic_parameters(c::LookupRoughness{FT}, λp, h) where FT
    valid = isfinite(λp) & isfinite(h) & (h >= 0)
    λ = clamp(λp, zero(FT), one(FT))
    h = max(h, zero(FT))

    z0 = c.bare_soil_roughness + c.roughness_height_fraction * h
    d0 = c.displacement_height_fraction * h

    return finalize_aerodynamic_parameters(z0, d0, λ, valid, c.bare_soil_roughness, c.minimum_built_fraction)
end

@inline (c::AbstractUrbanRoughness)(λp, h) = aerodynamic_parameters(c, λp, h)
