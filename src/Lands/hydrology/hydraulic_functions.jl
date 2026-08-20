#####
##### Hydraulic property functions used by `VariablySaturatedHydrology`.
#####
##### Two closures: retention curve `Π_m(θˡ)` (pressure head from pore liquid
##### fraction) and hydraulic conductivity `K(𝒮)`. Both are pure `@inline`
##### functions called from per-cell kernels — type-stable, allocation-free.
#####
##### van Genuchten (1980) retention and Mualem (1976) conductivity are the
##### available models.
#####

"""
    VanGenuchtenRetention(α, n)

Empirical soil-water retention curve mapping liquid pore fraction `θˡ`
(or saturation `𝒮`) to soil matric pressure head `Π_m` (m, negative in
unsaturated soil), following [van Genuchten (1980)](@cite vangenuchten1980):

```math
\\Pi_m(\\mathcal S) = -\\frac{1}{\\alpha}\\left[\\mathcal S^{-1/m} - 1\\right]^{1/n},
\\qquad m = 1 - 1/n.
```

`α` (m⁻¹) and `n` (–) are the standard van Genuchten shape parameters.
"""
struct VanGenuchtenRetention{FT}
    α :: FT
    n :: FT
end

VanGenuchtenRetention(FT::Type = Oceananigans.defaults.FloatType; α, n) =
    VanGenuchtenRetention(convert(FT, α), convert(FT, n))

"""
$(TYPEDSIGNATURES)

The second van Genuchten shape parameter, `m = 1 - 1/n`. [Mualem (1976)](@cite mualem1976new)
fixes `m` to this relation; left free, the pore-bundle integral behind
[`VanGenuchtenConductivity`](@ref) has no closed form.
"""
@inline van_genuchten_m(n) = 1 - 1/n

"""
$(TYPEDSIGNATURES)

Effective saturation `𝒮 = [1 + (αψ)ⁿ]^(-m)` at dimensionless suction `αψ ≥ 0`, the
inverse of [`VanGenuchtenRetention`](@ref)'s `pressure_head`.
"""
@inline van_genuchten_saturation(αψ, n) = (1 + αψ^n)^(-van_genuchten_m(n))

# Per-cell endpoint evaluation for the interface's plant-stress formulations: read the
# curve's parameters at `(i, j)` — scalar or `Field` — and invert the retention relation.
@inline function EarthSystemModels.effective_saturation(i, j, grid, r::VanGenuchtenRetention, ψ)
    FT = typeof(ψ)
    α  = convert(FT, property_value(r.α, i, j))
    n  = convert(FT, property_value(r.n, i, j))
    return van_genuchten_saturation(α * ψ, n)
end

@inline function pressure_head(r::VanGenuchtenRetention, 𝒮)
    FT = typeof(𝒮)
    α = convert(FT, r.α)
    n = convert(FT, r.n)
    m = van_genuchten_m(n)
    # Clamp 𝒮 strictly inside (0, 1] to avoid singularities at endpoints.
    𝒮c = clamp(𝒮, eps(FT), one(FT))
    return ifelse(𝒮c >= one(FT),
                  zero(FT),
                  -(𝒮c^(-1/m) - one(FT))^(1/n) / α)
end

Base.summary(r::VanGenuchtenRetention) =
    string("VanGenuchtenRetention(α=", prettysummary(r.α), ", n=", prettysummary(r.n), ")")

"""
    VanGenuchtenConductivity(K_saturated, n, ℓ)

Unsaturated hydraulic conductivity as a function of saturation `𝒮`, combining
the [Mualem (1976)](@cite mualem1976new) pore-bundle model with the
[van Genuchten (1980)](@cite vangenuchten1980) retention shape:

```math
K(\\mathcal S) = K_{sat}\\,\\mathcal S^\\ell\\left[1 - (1 - \\mathcal S^{1/m})^m\\right]^2,
\\qquad m = 1 - 1/n.
```

`K_saturated` (m s⁻¹) is the saturated hydraulic conductivity, `n` matches the
retention `n`, and `ℓ` (–) is the Mualem pore-connectivity exponent (default 0.5).
"""
struct VanGenuchtenConductivity{FT}
    K_saturated :: FT
    n           :: FT
    ℓ           :: FT
end

VanGenuchtenConductivity(FT::Type = Oceananigans.defaults.FloatType;
                         K_saturated, n, ℓ = 0.5) =
    VanGenuchtenConductivity(convert(FT, K_saturated), convert(FT, n), convert(FT, ℓ))

@inline function hydraulic_conductivity(c::VanGenuchtenConductivity, 𝒮)
    FT = typeof(𝒮)
    Ksat = convert(FT, c.K_saturated)
    n = convert(FT, c.n)
    ℓ = convert(FT, c.ℓ)
    m = van_genuchten_m(n)
    𝒮c = clamp(𝒮, zero(FT), one(FT))
    # K → Kˢᵃᵗ as 𝒮 → 1.
    inner = one(FT) - (one(FT) - 𝒮c^(1/m))^m
    return Ksat * 𝒮c^ℓ * inner^2
end

Base.summary(c::VanGenuchtenConductivity) =
    string("VanGenuchtenConductivity(K_saturated=", prettysummary(c.K_saturated),
           ", n=", prettysummary(c.n), ", ℓ=", prettysummary(c.ℓ), ")")
