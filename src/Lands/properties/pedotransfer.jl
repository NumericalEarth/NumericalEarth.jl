#####
##### Pedotransfer functions (PTFs): soil texture + bulk density → van Genuchten
##### hydraulic parameters `(ν, θʳ, α, n, Kₛ)`.
#####
##### `soil_hydraulic_parameters(ptf, sand, silt, clay, bulk_density)` is a pure,
##### `@inline`, allocation-free function called per grid point (and per depth
##### layer) by the `soil_hydraulic_properties` reduction. Texture is a mass
##### fraction (kg/kg) and bulk density is kg/m³ — the model-side units delivered
##### by the DataWrangling soil datasets (e.g. `OpenLandMapSoilDB`, `SoilGrids2`).
#####

"""
    abstract type PedotransferFunction

A pedotransfer function maps basic soil properties (texture, bulk density) to the
van Genuchten–Mualem hydraulic parameters. Concrete subtypes implement

    soil_hydraulic_parameters(ptf, sand, silt, clay, bulk_density)
        -> (; ν, θʳ, α, n, K_saturated)

with `sand`, `silt`, `clay` mass fractions (kg/kg), `bulk_density` in kg/m³, and
outputs in model units (`α` in m⁻¹, `K_saturated` in m s⁻¹). [`ContinuousPedotransfer`](@ref)
is the analytic default.
"""
abstract type PedotransferFunction end

"""
    ContinuousPedotransfer(FT = Float64; organic_matter = 1, topsoil = true,
                                         residual_liquid_fraction = 0.01, mualem_exponent = 0.5)

A continuous pedotransfer function: closed-form regressions mapping continuous soil
texture and bulk density to continuous van Genuchten parameters — as opposed to a
*class* PTF that bins soil into a few texture classes. Implements the
[Wösten et al. (1999)](@cite wosten1999) HYPRES functions in clay %, silt %, organic
matter %, bulk density (g/cm³) and a topsoil/subsoil flag.

Organic matter and the topsoil flag are not carried by the 30 m texture datasets,
so they are uniform defaults here: `organic_matter` (%, a mineral-soil value) and
`topsoil` (`true`/`false`). The residual water content `residual_liquid_fraction`
(`θʳ`) and Mualem exponent `mualem_exponent` (`ℓ`) are fixed constants — the Wösten
`ℓ` regression is noisy and can go negative, so `ℓ = 0.5` is used by default.

Predicted `θs` is returned as the porosity `ν`. Units are converted to model units:
`α` (cm⁻¹) → m⁻¹, `K_saturated` (cm day⁻¹) → m s⁻¹.
"""
struct ContinuousPedotransfer{FT} <: PedotransferFunction
    organic_matter           :: FT
    topsoil                  :: FT
    residual_liquid_fraction :: FT
    mualem_exponent          :: FT
end

ContinuousPedotransfer(FT::Type = Oceananigans.defaults.FloatType;
           organic_matter = 1,
           topsoil = true,
           residual_liquid_fraction = 0.01,
           mualem_exponent = 0.5) =
    ContinuousPedotransfer(convert(FT, organic_matter),
               convert(FT, topsoil),
               convert(FT, residual_liquid_fraction),
               convert(FT, mualem_exponent))

Base.summary(ptf::ContinuousPedotransfer) =
    string("ContinuousPedotransfer(organic_matter=", prettysummary(ptf.organic_matter),
           ", topsoil=", prettysummary(ptf.topsoil),
           ", θʳ=", prettysummary(ptf.residual_liquid_fraction),
           ", ℓ=", prettysummary(ptf.mualem_exponent), ")")

@inline function soil_hydraulic_parameters(ptf::ContinuousPedotransfer, sand, silt, clay, bulk_density)
    FT = typeof(clay)
    floor = convert(FT, 1//10)                 # 0.1 %, and 0.1 g/cm³ for ρᵇ

    # kg/kg → %, kg/m³ → g/cm³; floor 1/x and ln x arguments away from zero.
    C = max(100 * clay, floor)
    S = max(100 * silt, floor)
    OM = max(ptf.organic_matter, floor)
    D = max(bulk_density / 1000, floor)
    T = ptf.topsoil

    θs = 0.7919 + 0.001691C - 0.29619D - 0.000001491*S^2 + 0.0000821*OM^2 +
         0.02427/C + 0.01113/S + 0.01472*log(S) - 0.0000733*OM*C -
         0.000619*D*C - 0.001183*D*OM - 0.0001664*T*S

    αstar = -14.96 + 0.03135C + 0.0351S + 0.646OM + 15.29D - 0.192T - 4.671*D^2 -
            0.000781*C^2 - 0.00687*OM^2 + 0.0449/OM + 0.0663*log(S) + 0.1482*log(OM) -
            0.04546*D*S - 0.4852*D*OM + 0.00673*T*C

    nstar = -25.23 - 0.02195C + 0.0074S - 0.1940OM + 45.5D - 7.24*D^2 + 0.0003658*C^2 +
            0.002885*OM^2 - 12.81/D - 0.1524/S - 0.01958/OM - 0.2876*log(S) -
            0.0709*log(OM) - 44.6*log(D) - 0.02264*D*C + 0.0896*D*OM + 0.00718*T*C

    Ksstar = 7.755 + 0.0352S + 0.93T - 0.967*D^2 - 0.000484*C^2 - 0.000322*S^2 +
             0.001/S - 0.0748/OM - 0.643*log(S) - 0.01398*D*C - 0.1673*D*OM +
             0.02986*T*C - 0.03305*T*S

    θʳ = ptf.residual_liquid_fraction
    ν  = clamp(θs, θʳ + convert(FT, 1//100), one(FT) - eps(FT))
    α  = 100 * exp(αstar)                          # cm⁻¹ → m⁻¹
    n  = 1 + exp(nstar)
    Kₛ = exp(Ksstar) * (0.01 / 86400)              # cm day⁻¹ → m s⁻¹

    return (ν = convert(FT, ν),
            θʳ = convert(FT, θʳ),
            α = convert(FT, α),
            n = convert(FT, n),
            K_saturated = convert(FT, Kₛ))
end
