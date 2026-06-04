#####
##### Hydraulic property functions used by `VariablySaturatedHydrology`.
#####
##### Two closures: retention curve `Π_m(θˡ)` (pressure head from pore liquid
##### fraction) and hydraulic conductivity `K(𝒮, T)`. Both are pure `@inline`
##### functions called from per-cell kernels — type-stable, allocation-free.
#####
##### Van Genuchten (1980) parameterization is the only retention/conductivity
##### model in this PR; Brooks–Corey is a follow-up.
#####

"""
    VanGenuchtenRetention(α, n)

Van Genuchten (1980) retention curve mapping liquid pore fraction `θˡ`
(or saturation `𝒮`) to soil matric pressure head `Π_m` (m, negative in
unsaturated soil):

```math
\\Pi_m(\\mathcal S) = -\\frac{1}{\\alpha}\\left[\\mathcal S^{-1/m} - 1\\right]^{1/n},
\\qquad m = 1 - 1/n.
```

`α` (m⁻¹) and `n` (–) are the standard Van Genuchten shape parameters.
"""
struct VanGenuchtenRetention{FT}
    α :: FT
    n :: FT
end

VanGenuchtenRetention(FT::Type = Oceananigans.defaults.FloatType; α, n) =
    VanGenuchtenRetention(convert(FT, α), convert(FT, n))

@inline van_genuchten_m(n) = 1 - 1/n

@inline function pressure_head(r::VanGenuchtenRetention, 𝒮)
    FT = typeof(𝒮)
    m = van_genuchten_m(r.n)
    # Clamp 𝒮 strictly inside (0, 1] to avoid singularities at endpoints.
    𝒮c = clamp(𝒮, eps(FT), one(FT))
    return ifelse(𝒮c >= one(FT),
                  zero(FT),
                  -(𝒮c^(-1/m) - one(FT))^(1/r.n) / r.α)
end

Base.summary(r::VanGenuchtenRetention) =
    string("VanGenuchtenRetention(α=", prettysummary(r.α), ", n=", prettysummary(r.n), ")")

"""
    VanGenuchtenConductivity(K_saturated, n, ℓ)

Van Genuchten–Mualem hydraulic conductivity:

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
    m = van_genuchten_m(c.n)
    𝒮c = clamp(𝒮, zero(FT), one(FT))
    # K → K_sat as 𝒮 → 1.
    inner = one(FT) - (one(FT) - 𝒮c^(1/m))^m
    return c.K_saturated * 𝒮c^c.ℓ * inner^2
end

Base.summary(c::VanGenuchtenConductivity) =
    string("VanGenuchtenConductivity(K_saturated=", prettysummary(c.K_saturated),
           ", n=", prettysummary(c.n), ", ℓ=", prettysummary(c.ℓ), ")")
