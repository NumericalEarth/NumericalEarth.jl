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
    VanGenuchtenRetention(inverse_air_entry_head, n)

Empirical soil-water retention curve mapping liquid pore fraction `θˡ`
(or saturation `𝒮`) to soil matric pressure head `Π_m` (m, negative in
unsaturated soil), following [van Genuchten (1980)](@cite vangenuchten1980):

```math
\\Pi_m(\\mathcal S) = -\\frac{1}{\\alpha}\\left[\\mathcal S^{-1/m} - 1\\right]^{1/n},
\\qquad m = 1 - 1/n.
```

`inverse_air_entry_head` (`α`, m⁻¹) sets the head scale at which the soil drains:
its reciprocal `1/α` is the air-entry (bubbling) pressure head, the suction at which
the largest pores begin to empty. `n` (–) is the pore-size distribution index, which
sets how sharply the curve falls away from saturation. Each may be a scalar (uniform)
or a `Field` that varies grid point by grid point, e.g. from a pedotransfer function
over a soil-texture map (see [`soil_hydraulic_properties`](@ref)).
"""
struct VanGenuchtenRetention{A, N}
    inverse_air_entry_head :: A
    n                      :: N
end

VanGenuchtenRetention(FT::Type = Oceananigans.defaults.FloatType; inverse_air_entry_head, n) =
    VanGenuchtenRetention(normalize_property(FT, inverse_air_entry_head),
                          normalize_property(FT, n))

Adapt.adapt_structure(to, r::VanGenuchtenRetention) =
    VanGenuchtenRetention(Adapt.adapt(to, r.inverse_air_entry_head),
                          Adapt.adapt(to, r.n))

@inline van_genuchten_m(n) = 1 - 1/n

@inline function pressure_head(i, j, grid, r::VanGenuchtenRetention, 𝒮)
    FT  = eltype(grid)
    α   = convert(FT, property_value(r.inverse_air_entry_head, i, j))
    n   = convert(FT, property_value(r.n, i, j))
    n⁻¹ = 1 / n
    m⁻¹ = 1 / (one(FT) - n⁻¹)
    # Clamp 𝒮 strictly inside (0, 1] to avoid singularities at endpoints.
    𝒮c = clamp(convert(FT, 𝒮), eps(FT), one(FT))
    return ifelse(𝒮c >= one(FT),
                  zero(FT),
                  -(𝒮c^(-m⁻¹) - one(FT))^n⁻¹ / α)
end

Base.summary(r::VanGenuchtenRetention) =
    string("VanGenuchtenRetention(α=", prettysummary(r.inverse_air_entry_head),
           ", n=", prettysummary(r.n), ")")

"""
    VanGenuchtenConductivity(K_saturated, n, pore_connectivity_exponent)

Unsaturated hydraulic conductivity as a function of saturation `𝒮`, combining
the [Mualem (1976)](@cite mualem1976new) pore-bundle model with the
[van Genuchten (1980)](@cite vangenuchten1980) retention shape:

```math
K(\\mathcal S) = K_{sat}\\,\\mathcal S^\\ell\\left[1 - (1 - \\mathcal S^{1/m})^m\\right]^2,
\\qquad m = 1 - 1/n.
```

`K_saturated` (m s⁻¹) is the saturated hydraulic conductivity and `n` matches the
retention `n`. `pore_connectivity_exponent` (`ℓ`, –, default 0.5) is the Mualem
exponent on saturation: it measures how well the water-filled pores stay connected
as the soil drains, so a larger value throttles conductivity more steeply. Each may
be a scalar or a `Field` (see [`soil_hydraulic_properties`](@ref)).
"""
struct VanGenuchtenConductivity{K, N, L}
    K_saturated                :: K
    n                          :: N
    pore_connectivity_exponent :: L
end

VanGenuchtenConductivity(FT::Type = Oceananigans.defaults.FloatType;
                         K_saturated, n, pore_connectivity_exponent = 0.5) =
    VanGenuchtenConductivity(normalize_property(FT, K_saturated),
                             normalize_property(FT, n),
                             normalize_property(FT, pore_connectivity_exponent))

Adapt.adapt_structure(to, c::VanGenuchtenConductivity) =
    VanGenuchtenConductivity(Adapt.adapt(to, c.K_saturated),
                             Adapt.adapt(to, c.n),
                             Adapt.adapt(to, c.pore_connectivity_exponent))

@inline function hydraulic_conductivity(i, j, grid, c::VanGenuchtenConductivity, 𝒮)
    FT   = eltype(grid)
    Ksat = convert(FT, property_value(c.K_saturated, i, j))
    n    = convert(FT, property_value(c.n, i, j))
    ℓ    = convert(FT, property_value(c.pore_connectivity_exponent, i, j))
    m    = van_genuchten_m(n)
    m⁻¹  = 1 / m
    𝒮c   = clamp(convert(FT, 𝒮), zero(FT), one(FT))
    # K → K_sat as 𝒮 → 1.
    inner = one(FT) - (one(FT) - 𝒮c^m⁻¹)^m
    return Ksat * 𝒮c^ℓ * inner^2
end

Base.summary(c::VanGenuchtenConductivity) =
    string("VanGenuchtenConductivity(K_saturated=", prettysummary(c.K_saturated),
           ", n=", prettysummary(c.n),
           ", ℓ=", prettysummary(c.pore_connectivity_exponent), ")")
