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

Subtypes also implement `on_float_type(FT, ptf)`, which rebuilds `ptf` at float type
`FT`. Devices reject float types they do not support (Metal has no `Float64`), so
[`soil_hydraulic_properties`](@ref) converts to the grid's float type before launching.
"""
abstract type PedotransferFunction end

"""
    on_float_type(FT, ptf)

Rebuild pedotransfer function `ptf` with every parameter at float type `FT`.
"""
function on_float_type end

#####
##### Regression basis.
#####
##### The four HYPRES regressions are linear in a shared set of predictors built
##### from clay % `C`, silt % `S`, organic matter % `OM`, bulk density `D`
##### (g/cm³), and the topsoil flag `T`. Each regression is then a coefficient
##### tuple in this fixed order:
#####
#####   1, C, S, OM, D, T, D², C², OM², S², 1/C, 1/S, 1/OM, 1/D,
#####   ln S, ln OM, ln D, OM·C, D·C, D·S, D·OM, T·C, T·S
#####

@inline hypres_predictors(C, S, OM, D, T) =
    (one(C), C, S, OM, D, T, D^2, C^2, OM^2, S^2,
     1/C, 1/S, 1/OM, 1/D, log(S), log(OM), log(D),
     OM*C, D*C, D*S, D*OM, T*C, T*S)

@inline apply_regression(coefficients, predictors) = sum(map(*, coefficients, predictors))

# Wösten et al. (1999) HYPRES continuous regressions, Table 4. `porosity` predicts
# θs directly; the other three predict transformed variables (α*, n*, Kₛ*).
const HYPRES_COEFFICIENTS = (
    porosity = (0.7919, 0.001691, 0, 0, -0.29619, 0, 0, 0, 0.0000821, -0.000001491,
                0.02427, 0.01113, 0, 0, 0.01472, 0, 0,
                -0.0000733, -0.000619, 0, -0.001183, 0, -0.0001664),

    α = (-14.96, 0.03135, 0.0351, 0.646, 15.29, -0.192, -4.671, -0.000781, -0.00687, 0,
         0, 0, 0.0449, 0, 0.0663, 0.1482, 0,
         0, 0, -0.04546, -0.4852, 0.00673, 0),

    n = (-25.23, -0.02195, 0.0074, -0.1940, 45.5, 0, -7.24, 0.0003658, 0.002885, 0,
         0, -0.1524, -0.01958, -12.81, -0.2876, -0.0709, -44.6,
         0, -0.02264, 0, 0.0896, 0.00718, 0),

    K_saturated = (7.755, 0, 0.0352, 0, 0, 0.93, -0.967, -0.000484, 0, -0.000322,
                   0, 0.001, -0.0748, 0, -0.643, 0, 0,
                   0, -0.01398, 0, -0.1673, 0.02986, -0.03305))

"""
    ContinuousPedotransfer(FT = Oceananigans.defaults.FloatType;
                           organic_matter = 1,
                           topsoil = true,
                           residual_liquid_fraction = 0.01,
                           pore_connectivity_exponent = 0.5,
                           coefficients = HYPRES_COEFFICIENTS)

A continuous pedotransfer function: closed-form regressions mapping continuous soil
texture and bulk density to continuous van Genuchten parameters — as opposed to a
*class* PTF that bins soil into a few texture classes. Implements the
[Wösten et al. (1999)](@cite wosten1999) HYPRES functions in clay %, silt %, organic
matter %, bulk density (g/cm³) and a topsoil/subsoil flag.

Organic matter and the topsoil flag are not carried by the 30 m texture datasets,
so they are uniform defaults here: `organic_matter` (%, a mineral-soil value) and
`topsoil` (`true`/`false`). The residual water content `residual_liquid_fraction`
(`θʳ`) and pore-connectivity exponent `pore_connectivity_exponent` (`ℓ`, the Mualem
exponent) are fixed constants — the Wösten `ℓ` regression is noisy and can go
negative, so `ℓ = 0.5` is used by default.

`coefficients` holds one tuple per predicted parameter, each ordered by the shared
regression basis; supply your own to swap in a different calibration of the same
functional form.

Predicted `θs` is returned as the porosity `ν`. Units are converted to model units:
`α` (cm⁻¹) → m⁻¹, `K_saturated` (cm day⁻¹) → m s⁻¹.
"""
struct ContinuousPedotransfer{FT, C} <: PedotransferFunction
    organic_matter             :: FT
    topsoil                    :: FT
    residual_liquid_fraction   :: FT
    pore_connectivity_exponent :: FT
    coefficients               :: C
end

ContinuousPedotransfer(FT::Type = Oceananigans.defaults.FloatType;
                       organic_matter = 1,
                       topsoil = true,
                       residual_liquid_fraction = 0.01,
                       pore_connectivity_exponent = 0.5,
                       coefficients = HYPRES_COEFFICIENTS) =
    ContinuousPedotransfer(convert(FT, organic_matter),
                           convert(FT, topsoil),
                           convert(FT, residual_liquid_fraction),
                           convert(FT, pore_connectivity_exponent),
                           map(c -> convert.(FT, c), coefficients))

on_float_type(FT, ptf::ContinuousPedotransfer) =
    ContinuousPedotransfer(FT; organic_matter = ptf.organic_matter,
                               topsoil = ptf.topsoil,
                               residual_liquid_fraction = ptf.residual_liquid_fraction,
                               pore_connectivity_exponent = ptf.pore_connectivity_exponent,
                               coefficients = ptf.coefficients)

Base.summary(ptf::ContinuousPedotransfer) =
    string("ContinuousPedotransfer(organic_matter=", prettysummary(ptf.organic_matter),
           ", topsoil=", prettysummary(ptf.topsoil),
           ", θʳ=", prettysummary(ptf.residual_liquid_fraction),
           ", ℓ=", prettysummary(ptf.pore_connectivity_exponent), ")")

@inline function soil_hydraulic_parameters(ptf::ContinuousPedotransfer, sand, silt, clay, bulk_density)
    FT = typeof(clay)
    lower_bound = convert(FT, 1//10)           # 0.1 %, and 0.1 g/cm³ for ρᵇ

    # kg/kg → %, kg/m³ → g/cm³; bound 1/x and ln x arguments away from zero.
    C  = max(100 * clay, lower_bound)
    S  = max(100 * silt, lower_bound)
    OM = max(convert(FT, ptf.organic_matter), lower_bound)
    D  = max(bulk_density / 1000, lower_bound)
    T  = convert(FT, ptf.topsoil)

    predictors = hypres_predictors(C, S, OM, D, T)

    θs     = apply_regression(ptf.coefficients.porosity, predictors)
    αstar  = apply_regression(ptf.coefficients.α, predictors)
    nstar  = apply_regression(ptf.coefficients.n, predictors)
    Ksstar = apply_regression(ptf.coefficients.K_saturated, predictors)

    θʳ = convert(FT, ptf.residual_liquid_fraction)
    ν  = clamp(θs, θʳ + convert(FT, 1//100), one(FT) - eps(FT))
    α  = 100 * exp(αstar)                                    # cm⁻¹ → m⁻¹
    n  = 1 + exp(nstar)
    Kₛ = exp(Ksstar) * convert(FT, 1//100) / convert(FT, 86400)   # cm day⁻¹ → m s⁻¹

    return (ν = ν,
            θʳ = θʳ,
            α = α,
            n = n,
            K_saturated = Kₛ)
end
