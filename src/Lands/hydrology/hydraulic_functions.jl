#####
##### Hydraulic property closures for `VariablySaturatedHydrology`: van Genuchten (1980)
##### retention `Π_m(θˡ)` and Mualem (1976) conductivity `K(𝒮)`. Both are pure `@inline`
##### functions called from per-cell kernels.
#####

"""
    VanGenuchtenRetention(FT = Oceananigans.defaults.FloatType;
                          inverse_air_entry_head, pore_size_uniformity)

Empirical soil-water retention curve mapping liquid pore fraction `θˡ`
(or saturation `𝒮`) to soil matric pressure head `Π_m` (m, negative in
unsaturated soil), following [van Genuchten (1980)](@cite vangenuchten1980):

```math
\\Pi_m(\\mathcal S) = -\\frac{1}{\\alpha}\\left[\\mathcal S^{-1/m} - 1\\right]^{1/n},
\\qquad m = 1 - 1/n.
```

`inverse_air_entry_head` (`α`, m⁻¹) is the reciprocal of the air-entry pressure head, the
suction at which the largest pores begin to empty. `pore_size_uniformity` (`n`, –) sets how
narrow the band of suctions is over which the soil drains. Each may be a scalar or a
`Field` (see [`soil_hydraulic_properties`](@ref)).
"""
struct VanGenuchtenRetention{A, N}
    inverse_air_entry_head :: A
    pore_size_uniformity   :: N
end

VanGenuchtenRetention(FT::Type = Oceananigans.defaults.FloatType;
                      inverse_air_entry_head, pore_size_uniformity) =
    VanGenuchtenRetention(normalize_property(FT, inverse_air_entry_head),
                          normalize_property(FT, pore_size_uniformity))

Adapt.adapt_structure(to, r::VanGenuchtenRetention) =
    VanGenuchtenRetention(Adapt.adapt(to, r.inverse_air_entry_head),
                          Adapt.adapt(to, r.pore_size_uniformity))

"""
$(TYPEDSIGNATURES)

The second van Genuchten shape parameter, `m = 1 - 1/n`, under the
[Mualem (1976)](@cite mualem1976new) restriction.
"""
@inline van_genuchten_m(n) = 1 - 1/n

"""
$(TYPEDSIGNATURES)

Effective saturation `𝒮 = [1 + (αψ)ⁿ]^(-m)` at dimensionless suction `αψ ≥ 0`, the
inverse of [`VanGenuchtenRetention`](@ref)'s `pressure_head`.
"""
@inline van_genuchten_saturation(αψ, n) = (1 + αψ^n)^(-van_genuchten_m(n))

"""
$(TYPEDSIGNATURES)

`log(eˣ - 1)` for `x > 0`, evaluated as `x + log(1 - e⁻ˣ)` to keep `eˣ` from overflowing.
"""
@inline logexpm1(x) = x + log(-expm1(-x))

@inline function pressure_head(i, j, grid, r::VanGenuchtenRetention, 𝒮)
    FT = typeof(𝒮)
    α  = convert(FT, property_value(r.inverse_air_entry_head, i, j))
    n  = convert(FT, property_value(r.pore_size_uniformity, i, j))
    m  = van_genuchten_m(n)
    𝒮c = clamp(𝒮, 0, 1)

    # the head is capped at √floatmax so the pole at 𝒮 = 0 stays finite
    logΠᵐᵃˣ = log(floatmax(FT)) / 2
    logΠ    = logexpm1(-log(𝒮c) / m) / n - log(α)
    return ifelse(𝒮c >= 1, zero(FT), -exp(min(logΠ, logΠᵐᵃˣ)))
end

Base.summary(r::VanGenuchtenRetention) =
    string("VanGenuchtenRetention(α=", prettysummary(r.inverse_air_entry_head),
           ", n=", prettysummary(r.pore_size_uniformity), ")")

"""
    VanGenuchtenConductivity(FT = Oceananigans.defaults.FloatType;
                             matching_point_conductivity, pore_size_uniformity,
                             pore_connectivity_exponent = 1//2,
                             water_viscosity = WaterViscosity(FT))

Unsaturated hydraulic conductivity as a function of saturation `𝒮`, combining
the [Mualem (1976)](@cite mualem1976new) pore-bundle model with the
[van Genuchten (1980)](@cite vangenuchten1980) retention shape:

```math
K(\\mathcal S) = K_0\\,\\mathcal S^{\\eta^K}\\left[1 - (1 - \\mathcal S^{1/m})^m\\right]^2,
\\qquad m = 1 - 1/n.
```

`matching_point_conductivity` (`K₀`, m s⁻¹) is the conductivity this curve reaches at
`𝒮 = 1`, `pore_size_uniformity` (`n`) must match the retention curve's, and
`pore_connectivity_exponent` (`ηᴷ`, –) is the exponent on saturation: a larger value
throttles conductivity more steeply as the soil drains. Each may be a scalar or a
`Field` (see [`soil_hydraulic_properties`](@ref)). `K₀` and `ηᴷ` belong to one fit:
[`WeynantsPedotransfer`](@ref) predicts the two together.

`water_viscosity` is a [`WaterViscosity`](@ref) scaling `K` with the slab temperature;
pass `nothing` for an isothermal conductivity.
"""
struct VanGenuchtenConductivity{K, N, L, V}
    matching_point_conductivity :: K
    pore_size_uniformity        :: N
    pore_connectivity_exponent  :: L
    water_viscosity             :: V
end

VanGenuchtenConductivity(FT::Type = Oceananigans.defaults.FloatType;
                         matching_point_conductivity, pore_size_uniformity,
                         pore_connectivity_exponent = 1//2,
                         water_viscosity = WaterViscosity(FT)) =
    VanGenuchtenConductivity(normalize_property(FT, matching_point_conductivity),
                             normalize_property(FT, pore_size_uniformity),
                             normalize_property(FT, pore_connectivity_exponent),
                             water_viscosity)

Adapt.adapt_structure(to, c::VanGenuchtenConductivity) =
    VanGenuchtenConductivity(Adapt.adapt(to, c.matching_point_conductivity),
                             Adapt.adapt(to, c.pore_size_uniformity),
                             Adapt.adapt(to, c.pore_connectivity_exponent),
                             Adapt.adapt(to, c.water_viscosity))

"""
$(TYPEDSIGNATURES)

Darcy hydraulic conductivity (m s⁻¹) of closure `c` at saturation `𝒮` and temperature `T`
(K) in cell `(i, j)`.
"""
@inline function hydraulic_conductivity(i, j, grid, c::VanGenuchtenConductivity, 𝒮, T)
    FT = typeof(𝒮)
    K₀ = convert(FT, property_value(c.matching_point_conductivity, i, j))
    n  = convert(FT, property_value(c.pore_size_uniformity, i, j))
    ηᴷ = convert(FT, property_value(c.pore_connectivity_exponent, i, j))
    m  = van_genuchten_m(n)
    𝒮c = clamp(𝒮, 0, 1)

    log𝒮 = log(𝒮c)
    logu = log𝒮 / m
    u    = exp(logu)
    # summing logarithms keeps 𝒮^ηᴷ [⋯]² finite where the direct product is `Inf * 0`
    negligible = logu < log(eps(FT))
    logbracket = ifelse(negligible, log(m) + logu, log(-expm1(m * log1p(-u))))
    K = K₀ * exp(ηᴷ * log𝒮 + 2 * logbracket) * viscosity_correction(c.water_viscosity, T)
    return ifelse(𝒮c == 0, zero(FT), K)
end

"""
    WaterViscosity(FT = Oceananigans.defaults.FloatType;
                   activation_temperature = 507.88,
                   pole_temperature = 149.3,
                   reference_temperature = 288)

Temperature dependence of the dynamic viscosity of water, as the factor by which it scales
the hydraulic conductivity of soil:

```math
\\Theta(T) = \\exp\\!\\left[\\frac{T_1}{T^{ref} - T_2} - \\frac{T_1}{T - T_2}\\right],
```

with `T₁` the `activation_temperature` and `T₂` the `pole_temperature`, the law of
[Deck et al. (2026)](@cite deck2026) (their Equations A19–A20) after
[Hansson et al. (2004)](@cite hansson2004). Read with [`viscosity_correction`](@ref).

`Θ` is unity at `reference_temperature`, the temperature a tabulated
`matching_point_conductivity` is taken to have been measured at.
"""
struct WaterViscosity{FT}
    activation_temperature :: FT
    pole_temperature       :: FT
    reference_temperature  :: FT
end

WaterViscosity(FT::Type = Oceananigans.defaults.FloatType;
               activation_temperature = 50_788//100,
               pole_temperature = 1493//10,
               reference_temperature = 288) =
    WaterViscosity(convert(FT, activation_temperature),
                   convert(FT, pole_temperature),
                   convert(FT, reference_temperature))

Base.summary(v::WaterViscosity) =
    string("WaterViscosity(activation_temperature=", prettysummary(v.activation_temperature),
           ", pole_temperature=", prettysummary(v.pole_temperature),
           ", reference_temperature=", prettysummary(v.reference_temperature), ")")

Base.show(io::IO, v::WaterViscosity) = print(io, summary(v))

"""
$(TYPEDSIGNATURES)

Factor by which [`WaterViscosity`](@ref) scales hydraulic conductivity at temperature `T`
(K), unity at `v.reference_temperature`.

```jldoctest
using NumericalEarth

v = WaterViscosity()

## warm soil conducts more than cold soil
round.([viscosity_correction(v, T) for T in (275.0, 288.0, 310.0)], digits = 3)

# output
3-element Vector{Float64}:
 0.685
 1.0
 1.651
```
"""
@inline function viscosity_correction(v::WaterViscosity, T)
    FT = float(typeof(T))
    T₁ = convert(FT, v.activation_temperature)
    T₂ = convert(FT, v.pole_temperature)
    Tᵛ = convert(FT, v.reference_temperature)
    return exp(T₁ / (Tᵛ - T₂) - T₁ / (convert(FT, T) - T₂))
end

@inline viscosity_correction(::Nothing, T) = one(float(typeof(T)))

"""
$(TYPEDSIGNATURES)

Effective saturation at which capillary flow paths to an evaporating surface become
disconnected, `𝒮ᶜ = [1 + ((n-1)/n)^(1-2n)]^(-m)` with `m = 1 - 1/n`, the saturation at the
critical head `hᶜ = α⁻¹((n-1)/n)^((1-2n)/n)` of [Lehmann et al. (2008)](@cite lehmann2008).
The retention curve depends on the product `α hᶜ`, so `α` cancels and `𝒮ᶜ` is set by `n`
alone. Bare-soil evaporation falls to half its potential rate near
`θ½ ≈ θʳ + (ν - θʳ) 𝒮ᶜ` ([Lehmann et al. (2018)](@cite lehmann2018)), the threshold
[`CriticalSaturation`](@ref) takes as `critical_saturation`.

```jldoctest
using NumericalEarth

round.(capillary_disconnect_saturation.([1.2, 1.5, 3.0]), digits = 3)

# output
3-element Vector{Float64}:
 0.65
 0.464
 0.238
```
"""
@inline function capillary_disconnect_saturation(n)
    m = van_genuchten_m(n)
    return (1 + ((n - 1) / n)^(1 - 2n))^(-m)
end

"""
    CosbyConductivity(FT = Oceananigans.defaults.FloatType;
                      intercept = -0.884,
                      sand_coefficient = 0.0153,
                      spread_intercept = 0.459,
                      spread_silt_coefficient = 0.00321)

Saturated hydraulic conductivity from sand, and its within-class spread from silt, after
[Cosby et al. (1984)](@cite cosby1984) Table 5:

```math
\\log_{10} K^{+} = a + b\\,S, \\qquad
\\sigma(\\log_{10} K^{+}) = c + d\\,S^{silt},
```

with texture in % and `K⁺` in inch hour⁻¹. Read with [`saturated_conductivity`](@ref) and
[`conductivity_spread`](@ref), which take mass fractions (kg/kg) and return m s⁻¹ and
decades respectively.

The defaults are the published fit, regressed on 1,448 US samples across 11 texture
classes; against the laboratory `K⁺` of GSHP ([Gupta et al. (2022)](@cite gupta2022)) it
runs `+1.16` high in `log₁₀ cm/day`, so treat it as a prior to calibrate.
"""
struct CosbyConductivity{FT}
    intercept               :: FT
    sand_coefficient        :: FT
    spread_intercept        :: FT
    spread_silt_coefficient :: FT
end

CosbyConductivity(FT::Type = Oceananigans.defaults.FloatType;
                  intercept = -884//1000,
                  sand_coefficient = 153//10_000,
                  spread_intercept = 459//1000,
                  spread_silt_coefficient = 321//100_000) =
    CosbyConductivity(convert(FT, intercept),
                      convert(FT, sand_coefficient),
                      convert(FT, spread_intercept),
                      convert(FT, spread_silt_coefficient))

Base.summary(c::CosbyConductivity) =
    string("CosbyConductivity(intercept=", prettysummary(c.intercept),
           ", sand_coefficient=", prettysummary(c.sand_coefficient),
           ", spread_intercept=", prettysummary(c.spread_intercept),
           ", spread_silt_coefficient=", prettysummary(c.spread_silt_coefficient), ")")

Base.show(io::IO, c::CosbyConductivity) = print(io, summary(c))

"""
$(TYPEDSIGNATURES)

Macropore-inclusive saturated hydraulic conductivity `K⁺` (m s⁻¹) from sand mass fraction
(kg/kg), for an infiltration cap such as [`InfiltrationCapacityRunoff`](@ref).

```jldoctest
using NumericalEarth

## sand, loam and clay, in mm hour⁻¹
K⁺ = [saturated_conductivity(CosbyConductivity(), sand) for sand in (0.92, 0.43, 0.20)]
round.(K⁺ .* 3.6e6, digits = 1)

# output
3-element Vector{Float64}:
 84.8
 15.1
  6.7
```
"""
@inline saturated_conductivity(c::CosbyConductivity, sand) =
    (254//36_000_000) * 10^(c.sand_coefficient * 100sand + c.intercept)  # inch hour⁻¹ → m s⁻¹

"""
$(TYPEDSIGNATURES)

Standard deviation of `log₁₀ K⁺` within a texture class, from silt mass fraction (kg/kg).
"""
@inline conductivity_spread(c::CosbyConductivity, silt) =
    c.spread_silt_coefficient * 100silt + c.spread_intercept

viscosity_summary(v::WaterViscosity) = string("Tᵛ=", prettysummary(v.reference_temperature))
viscosity_summary(::Nothing) = "isothermal"

Base.summary(c::VanGenuchtenConductivity) =
    string("VanGenuchtenConductivity(K₀=", prettysummary(c.matching_point_conductivity),
           ", n=", prettysummary(c.pore_size_uniformity),
           ", ηᴷ=", prettysummary(c.pore_connectivity_exponent),
           ", ", viscosity_summary(c.water_viscosity), ")")
