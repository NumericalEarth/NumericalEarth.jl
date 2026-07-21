#####
##### Urban morphometric roughness closure
#####
##### Aerodynamic momentum roughness length z0 and zero-plane displacement d0 for the
##### urban (built-up) surface, from the plan-area fraction λp and mean building height H,
##### following the morphometry of [Macdonald et al. (1998)](@cite Macdonald1998) with the
##### height-heterogeneity correction of [Kanda et al. (2013)](@cite Kanda2013). Buildings
##### are non-porous bluff bodies, so this is a separate closure from the vegetation
##### (Raupach) one. Pure, allocation-free and kernel-safe.
#####

"""
$(TYPEDEF)

Empirical parameters of the urban morphometric roughness closure. All dimensionless
unless noted. Defaults follow Macdonald et al. (1998) (staggered array) and Kanda et
al. (2013).

$(TYPEDFIELDS)
"""
struct UrbanRoughnessParameters{FT}
    "Macdonald array constant `A` (4.43 staggered, 3.59 square)"
    array_constant :: FT
    "building drag coefficient `Cd`"
    drag_coefficient :: FT
    "Macdonald correction factor `β`"
    correction_factor :: FT
    "von Kármán constant `κ`"
    von_karman_constant :: FT
    "Kanda displacement constants `(a0, b0, c0)`"
    kanda_displacement :: NTuple{3, FT}
    "Kanda roughness constants `(a1, b1, c1)`"
    kanda_roughness :: NTuple{3, FT}
    "characteristic building width `Lb` (m), used by the cuboid frontal-area estimator"
    building_width :: FT
    "assumed height standard deviation as a fraction of mean height, `σH / H`"
    height_variability :: FT
    "assumed maximum-to-mean building height ratio, `Hmax / H`"
    maximum_height_ratio :: FT
    "bare-soil momentum roughness length (m) where the built fraction vanishes"
    bare_soil_roughness :: FT
    "built-fraction floor below which the cell reduces to bare soil"
    minimum_built_fraction :: FT
    "displacement ceiling `d0/H` (< 1, avoids the singular skimming limit)"
    maximum_displacement_ratio :: FT
end

function UrbanRoughnessParameters(FT = Oceananigans.defaults.FloatType;
                                  array_constant = 4.43,
                                  drag_coefficient = 1.2,
                                  correction_factor = 1.0,
                                  von_karman_constant = 0.4,
                                  kanda_displacement = (1.29, 0.36, -0.17),
                                  kanda_roughness = (0.71, 20.21, -0.77),
                                  building_width = 10.0,
                                  height_variability = 0.4,
                                  maximum_height_ratio = 2.5,
                                  bare_soil_roughness = 0.03,
                                  minimum_built_fraction = 0.01,
                                  maximum_displacement_ratio = 0.95)
    return UrbanRoughnessParameters{FT}(convert(FT, array_constant),
                                        convert(FT, drag_coefficient),
                                        convert(FT, correction_factor),
                                        convert(FT, von_karman_constant),
                                        convert.(FT, kanda_displacement),
                                        convert.(FT, kanda_roughness),
                                        convert(FT, building_width),
                                        convert(FT, height_variability),
                                        convert(FT, maximum_height_ratio),
                                        convert(FT, bare_soil_roughness),
                                        convert(FT, minimum_built_fraction),
                                        convert(FT, maximum_displacement_ratio))
end

# Method codes carried into the kernel (Symbols are not kernel-safe).
const MACDONALD = 0
const KANDA     = 1
const LOOKUP    = 2

# Frontal-area estimator codes.
const ISOTROPIC = 0
const CUBOID    = 1

"""
$(TYPEDSIGNATURES)

Frontal-area index `λf` estimated from the plan-area fraction `λp` and mean height
`h`, which GHSL does not observe. `ISOTROPIC` takes `λf ≈ λp` (cubic elements);
`CUBOID` takes `λf = λp · h / Lb` for square buildings of width `Lb`. This estimate
is the dominant modeling uncertainty of the Macdonald route, hence it is exposed.
"""
@inline function frontal_area_index(λp, h, estimator, Lb)
    isotropic = λp
    cuboid = ifelse(Lb > 0, λp * h / Lb, λp)
    return ifelse(estimator == CUBOID, cuboid, isotropic)
end

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
`z0` rises then falls (isolated → wake-interference → skimming) with the frontal
area `λf`.
"""
@inline function macdonald_roughness_ratio(λf, d0h, Cd, κ, β)
    bracket = (β * Cd) / (2 * κ^2) * (1 - d0h) * λf
    decay = ifelse(bracket > 0, exp(-inv(sqrt(bracket))), zero(bracket))
    return (1 - d0h) * decay
end

"""
$(TYPEDSIGNATURES)

Kanda et al. (2013) displacement height `d0` (m), which references the maximum
building height `hmax` and the height standard deviation `σh`:
`d0/hmax = c0·X² + (a0·λp^b0 − c0)·X`, with `X = (σh + h)/hmax` in `[0, 1]`.
"""
@inline function kanda_displacement_height(λp, h, σh, hmax, a0, b0, c0)
    X = clamp(ifelse(hmax > 0, (σh + h) / hmax, zero(h)), zero(h), one(h))
    d0hmax = c0 * X^2 + (a0 * λp^b0 - c0) * X
    return hmax * max(d0hmax, zero(d0hmax))
end

"""
$(TYPEDSIGNATURES)

Kanda et al. (2013) roughness length `z0` (m), a correction of the Macdonald value
`z0ᵐ`: `z0/z0ᵐ = b1·Y² + c1·Y + a1`, with `Y = λp·σh/h`. Reduces to `a1·z0ᵐ` for a
height-homogeneous canopy (`σh → 0`).
"""
@inline function kanda_roughness_length(z0ᵐ, λp, h, σh, a1, b1, c1)
    Y = ifelse(h > 0, λp * σh / h, zero(h))
    ratio = b1 * Y^2 + c1 * Y + a1
    return z0ᵐ * max(ratio, zero(ratio))
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length `z0` and zero-plane displacement `d0` (metres) of a
built-up surface with plan-area fraction `λp` and mean building height `h`, using
morphometric `method` (`MACDONALD`, `KANDA`, or `LOOKUP`) and frontal-area
`estimator` (`ISOTROPIC` or `CUBOID`). Returns `(z0, d0)`.

Endpoints reduce cleanly: `λp → 0` returns the bare-soil `(z0, 0)`; `λp → 1` is the
skimming limit (`d0/h` capped below 1). An invalid (`NaN`/negative) input returns
`(NaN, NaN)` — an honest gap.
"""
@inline function urban_roughness_point(λp, h, method, estimator, p::UrbanRoughnessParameters{FT}) where FT
    valid = isfinite(λp) & isfinite(h) & (h >= 0)
    λ = clamp(λp, zero(FT), one(FT))
    h = max(h, zero(FT))

    σh = p.height_variability * h
    hmax = p.maximum_height_ratio * h
    λf = frontal_area_index(λ, h, estimator, p.building_width)

    d0hᵐ = macdonald_displacement_ratio(λ, p.array_constant, p.maximum_displacement_ratio)
    z0ᵐ = h * macdonald_roughness_ratio(λf, d0hᵐ, p.drag_coefficient, p.von_karman_constant, p.correction_factor)
    d0ᵐ = h * d0hᵐ

    a0, b0, c0 = p.kanda_displacement
    a1, b1, c1 = p.kanda_roughness
    d0ᵏ = min(kanda_displacement_height(λ, h, σh, hmax, a0, b0, c0), p.maximum_displacement_ratio * h)
    z0ᵏ = kanda_roughness_length(z0ᵐ, λ, h, σh, a1, b1, c1)

    z0ˡ = p.bare_soil_roughness + FT(0.1) * h
    d0ˡ = FT(0.7) * h

    z0 = ifelse(method == KANDA, z0ᵏ, ifelse(method == LOOKUP, z0ˡ, z0ᵐ))
    d0 = ifelse(method == KANDA, d0ᵏ, ifelse(method == LOOKUP, d0ˡ, d0ᵐ))

    # Below the built-fraction floor the cell is bare soil, not a building array.
    bare = λ < p.minimum_built_fraction
    z0 = ifelse(bare, p.bare_soil_roughness, max(z0, p.bare_soil_roughness))
    d0 = ifelse(bare, zero(FT), d0)

    gap = convert(FT, NaN)
    return ifelse(valid, z0, gap), ifelse(valid, d0, gap)
end
