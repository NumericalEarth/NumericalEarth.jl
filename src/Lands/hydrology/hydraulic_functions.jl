#####
##### Hydraulic property functions used by `VariablySaturatedHydrology`.
#####
##### Two closures: retention curve `Π_m(θˡ)` (pressure head from pore liquid
##### fraction) and hydraulic conductivity `K(𝒮, T)`. Both are pure `@inline`
##### functions called from per-cell kernels — type-stable, allocation-free.
#####
##### van Genuchten (1980) / Mualem (1976) is the available retention/conductivity model.
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

`α` (m⁻¹) and `n` (–) are the standard Van Genuchten shape parameters. Each may be
a scalar (uniform) or a `Field` that varies grid point by grid point, e.g. from a
pedotransfer function over a soil-texture map (see [`soil_hydraulic_properties`](@ref)).
"""
struct VanGenuchtenRetention{A, N}
    α :: A
    n :: N
end

VanGenuchtenRetention(FT::Type = Oceananigans.defaults.FloatType; α, n) =
    VanGenuchtenRetention(normalize_property(FT, α), normalize_property(FT, n))

Adapt.adapt_structure(to, r::VanGenuchtenRetention) =
    VanGenuchtenRetention(Adapt.adapt(to, r.α), Adapt.adapt(to, r.n))

@inline van_genuchten_m(n) = 1 - 1/n

@inline function pressure_head(r::VanGenuchtenRetention, 𝒮, i, j)
    FT = typeof(𝒮)
    α = convert(FT, property_value(r.α, i, j))
    n = convert(FT, property_value(r.n, i, j))
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
`K_saturated`, `n`, and `ℓ` may each be a scalar or a `Field` (see
[`soil_hydraulic_properties`](@ref)).
"""
struct VanGenuchtenConductivity{K, N, L}
    K_saturated :: K
    n           :: N
    ℓ           :: L
end

VanGenuchtenConductivity(FT::Type = Oceananigans.defaults.FloatType;
                         K_saturated, n, ℓ = 0.5) =
    VanGenuchtenConductivity(normalize_property(FT, K_saturated),
                             normalize_property(FT, n),
                             normalize_property(FT, ℓ))

Adapt.adapt_structure(to, c::VanGenuchtenConductivity) =
    VanGenuchtenConductivity(Adapt.adapt(to, c.K_saturated),
                             Adapt.adapt(to, c.n),
                             Adapt.adapt(to, c.ℓ))

@inline function hydraulic_conductivity(c::VanGenuchtenConductivity, 𝒮, i, j)
    FT = typeof(𝒮)
    Kₛ = convert(FT, property_value(c.K_saturated, i, j))
    n  = convert(FT, property_value(c.n, i, j))
    ℓ  = convert(FT, property_value(c.ℓ, i, j))
    m  = van_genuchten_m(n)
    𝒮c = clamp(𝒮, zero(FT), one(FT))
    # K → K_sat as 𝒮 → 1.
    inner = one(FT) - (one(FT) - 𝒮c^(1/m))^m
    return Kₛ * 𝒮c^ℓ * inner^2
end

Base.summary(c::VanGenuchtenConductivity) =
    string("VanGenuchtenConductivity(K_saturated=", prettysummary(c.K_saturated),
           ", n=", prettysummary(c.n), ", ℓ=", prettysummary(c.ℓ), ")")
