#####
##### Pedotransfer functions (PTFs): soil texture + bulk density → van Genuchten
##### hydraulic parameters `(ν, θʳ, α, n, Kₛ)`.
#####
##### `soil_hydraulic_parameters(ptf, sand, silt, clay, bulk_density, depth)` is a
##### pure, `@inline`, allocation-free function called per grid point (and per depth
##### layer) by the `soil_hydraulic_properties` reduction. Texture is a mass
##### fraction (kg/kg), bulk density is kg/m³ — the model-side units delivered by
##### the DataWrangling soil datasets (e.g. `OpenLandMapSoilDB`, `SoilGrids2`) — and
##### `depth` is metres below the surface, which selects topsoil or subsoil.
#####
##### The default regressions are fitted to HYPRES, the database of HYdraulic
##### PRoperties of European Soils.
#####

"""
    abstract type PedotransferFunction

A pedotransfer function maps basic soil properties (texture, bulk density) to the
van Genuchten–Mualem hydraulic parameters. Concrete subtypes implement

    soil_hydraulic_parameters(ptf, sand, silt, clay, bulk_density, depth)
        -> (; porosity, residual_liquid_fraction, inverse_air_entry_head,
              pore_size_uniformity, K_saturated)

with `sand`, `silt`, `clay` mass fractions (kg/kg), `bulk_density` in kg/m³, `depth`
the soil layer's depth below the surface (m, positive down), and outputs in model
units (`α` in m⁻¹, `K_saturated` in m s⁻¹). Called without `depth`, it reports the
surface value. [`ContinuousPedotransfer`](@ref) is the analytic default.

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
##### The HYPRES regressions are all linear in a shared set of predictors built from
##### clay % `C`, silt % `S`, organic matter % `OM`, bulk density `D` (g/cm³), and the
##### topsoil flag `T`. Each regression is then a coefficient tuple in this order:
#####
#####   1, C, S, OM, D, T, D², C², OM², S², 1/C, 1/S, 1/OM, 1/D,
#####   ln S, ln OM, ln D, OM·C, D·C, D·S, D·OM, T·C, T·S
#####

@inline HYPRES_predictors(C, S, OM, D, T) =
    (one(C), C, S, OM, D, T, D^2, C^2, OM^2, S^2,
     1/C, 1/S, 1/OM, 1/D, log(S), log(OM), log(D),
     OM*C, D*C, D*S, D*OM, T*C, T*S)

@inline apply_regression(coefficients, predictors) = sum(map(*, coefficients, predictors))

"""
    HYPRESRegression(; porosity, inverse_air_entry_head, pore_size_uniformity, K_saturated)

Coefficients of the continuous pedotransfer regressions fitted to HYPRES, the
database of HYdraulic PRoperties of European Soils. One tuple per predicted
parameter, each ordered by the shared predictor basis. `porosity` predicts
`θs` directly; the other three predict the transformed parameters `α* = ln α`,
`n* = ln(n - 1)`, and `Kₛ* = ln Kₛ`, which the regression uses to enforce `α > 0`,
`n > 1`, and `Kₛ > 0`.

Construct one to swap in a different calibration of the same functional form;
[`HYPRES_REGRESSION`](@ref) holds the published fit.
"""
struct HYPRESRegression{C}
    porosity               :: C
    inverse_air_entry_head :: C
    pore_size_uniformity   :: C
    K_saturated            :: C
end

# `float` keeps the tuples homogeneous so integer zeros can be written literally.
HYPRESRegression(; porosity, inverse_air_entry_head, pore_size_uniformity, K_saturated) =
    HYPRESRegression(map(float, porosity),
                     map(float, inverse_air_entry_head),
                     map(float, pore_size_uniformity),
                     map(float, K_saturated))

on_float_type(FT, r::HYPRESRegression) =
    HYPRESRegression(convert.(FT, r.porosity),
                     convert.(FT, r.inverse_air_entry_head),
                     convert.(FT, r.pore_size_uniformity),
                     convert.(FT, r.K_saturated))

"""
    HYPRES_REGRESSION

The [Wösten et al. (1999)](@cite wosten1999) continuous pedotransfer regressions
fitted to HYPRES, the database of HYdraulic PRoperties of European Soils (their
Table 5). Each field predicts one parameter:

| field | predicts | units | R² |
|:------|:---------|:------|:---|
| `porosity` | `θs`, the saturated water content | – | 76 % |
| `inverse_air_entry_head` | `α* = ln α` | ln(cm⁻¹) | 20 % |
| `pore_size_uniformity` | `n* = ln(n - 1)` | – | 54 % |
| `K_saturated` | `Kₛ* = ln Kₛ` | ln(cm day⁻¹) | 19 % |

The three logarithmic transforms are what enforce `α > 0`, `n > 1`, and `Kₛ > 0`;
[`ContinuousPedotransfer`](@ref) inverts them and converts to model units. `θs` is
predicted directly and needs no transform.

Coefficients follow the `HYPRES_predictors` order, laid out one row per predictor
family. A zero marks a predictor the paper's subset selection dropped for that
parameter — no regression uses all 23.
"""
const HYPRES_REGRESSION = HYPRESRegression(
    porosity = (0.7919, 0.001691, 0, 0, -0.29619, 0,                     # 1, C, S, OM, D, T
                0, 0, 0.0000821, -0.000001491,                           # D², C², OM², S²
                0.02427, 0.01113, 0, 0,                                  # 1/C, 1/S, 1/OM, 1/D
                0.01472, 0, 0,                                           # ln S, ln OM, ln D
                -0.0000733, -0.000619, 0, -0.001183, 0, -0.0001664),     # OM·C, D·C, D·S, D·OM, T·C, T·S

    inverse_air_entry_head =
               (-14.96, 0.03135, 0.0351, 0.646, 15.29, -0.192,           # 1, C, S, OM, D, T
                -4.671, -0.000781, -0.00687, 0,                          # D², C², OM², S²
                0, 0, 0.0449, 0,                                         # 1/C, 1/S, 1/OM, 1/D
                0.0663, 0.1482, 0,                                       # ln S, ln OM, ln D
                0, 0, -0.04546, -0.4852, 0.00673, 0),                    # OM·C, D·C, D·S, D·OM, T·C, T·S

    pore_size_uniformity =
        (-25.23, -0.02195, 0.0074, -0.1940, 45.5, 0,                     # 1, C, S, OM, D, T
         -7.24, 0.0003658, 0.002885, 0,                                  # D², C², OM², S²
         0, -0.1524, -0.01958, -12.81,                                   # 1/C, 1/S, 1/OM, 1/D
         -0.2876, -0.0709, -44.6,                                        # ln S, ln OM, ln D
         0, -0.02264, 0, 0.0896, 0.00718, 0),                            # OM·C, D·C, D·S, D·OM, T·C, T·S

    K_saturated = (7.755, 0, 0.0352, 0, 0, 0.93,                         # 1, C, S, OM, D, T
                   -0.967, -0.000484, 0, -0.000322,                      # D², C², OM², S²
                   0, 0.001, -0.0748, 0,                                 # 1/C, 1/S, 1/OM, 1/D
                   -0.643, 0, 0,                                         # ln S, ln OM, ln D
                   0, -0.01398, 0, -0.1673, 0.02986, -0.03305))          # OM·C, D·C, D·S, D·OM, T·C, T·S

"""
    ContinuousPedotransfer(FT = Oceananigans.defaults.FloatType;
                           organic_matter = 1,
                           topsoil_depth = 0.3,
                           residual_liquid_fraction = 0,
                           pore_connectivity_exponent = 0.5,
                           regression_coefficients = HYPRES_REGRESSION)

A continuous pedotransfer function: closed-form regressions mapping continuous soil
texture and bulk density to continuous van Genuchten parameters — as opposed to a
*class* PTF that bins soil into a few texture classes. Implements the
[Wösten et al. (1999)](@cite wosten1999) regressions (their Table 5) fitted to
HYPRES, the database of HYdraulic PRoperties of European Soils, in clay %, silt %,
organic matter %, bulk density (g/cm³) and a topsoil/subsoil flag. Clay is the
< 2 μm fraction and silt the 2–50 μm fraction, matching the USDA split the soil
datasets report.

The regression distinguishes topsoil from subsoil, which it reads off the layer
`depth` passed to `soil_hydraulic_parameters`: soil shallower than `topsoil_depth`
is topsoil. The distinction is worth keeping — for a clay-rich soil it moves
`K_saturated` by a factor of 4 to 7, since a clay topsoil drains through its
aggregate structure while a clay subsoil does not.

Organic matter is not carried by the 30 m texture datasets, so `organic_matter` (%)
is a uniform mineral-soil default. The residual water content
`residual_liquid_fraction` (`θʳ`) and pore-connectivity exponent
`pore_connectivity_exponent` (`ℓ`, the Mualem exponent) are fixed constants: HYPRES
fits its retention curves with `θʳ = 0` — so anything else leaves `α` and `n`
describing a curve that was never fitted — and its `ℓ` fit is the weakest of the five
(coefficient of determination 12 %, and negative across most of the published texture
classes), so `ℓ = 0.5` is used by default.

`regression_coefficients` is a [`HYPRESRegression`](@ref); supply your own to swap
in a different calibration of the same functional form.

Predicted `θs` is returned as the porosity `ν`. Units are converted to model units:
`α` (cm⁻¹) → m⁻¹, `K_saturated` (cm day⁻¹) → m s⁻¹.

Clay, silt and organic matter enter the regressions as `1/x` and `ln x`, and bulk
density as `1/D` and `ln D`, so the predictors are held inside the range the fit
behaves in: texture and organic matter at or above 1 %, bulk density within
0.5–2.0 g/cm³. Extrapolating past those bounds is not a small error — it drives `θs`
above 1 and `n` down to 1, where the retention curve is singular.
"""
struct ContinuousPedotransfer{FT, C} <: PedotransferFunction
    organic_matter             :: FT
    topsoil_depth              :: FT
    residual_liquid_fraction   :: FT
    pore_connectivity_exponent :: FT
    regression_coefficients     :: C
end

ContinuousPedotransfer(FT::Type = Oceananigans.defaults.FloatType;
                       organic_matter = 1,
                       topsoil_depth = 0.3,
                       residual_liquid_fraction = 0,
                       pore_connectivity_exponent = 0.5,
                       regression_coefficients = HYPRES_REGRESSION) =
    ContinuousPedotransfer(convert(FT, organic_matter),
                           convert(FT, topsoil_depth),
                           convert(FT, residual_liquid_fraction),
                           convert(FT, pore_connectivity_exponent),
                           on_float_type(FT, regression_coefficients))

on_float_type(FT, ptf::ContinuousPedotransfer) =
    ContinuousPedotransfer(FT; organic_matter = ptf.organic_matter,
                               topsoil_depth = ptf.topsoil_depth,
                               residual_liquid_fraction = ptf.residual_liquid_fraction,
                               pore_connectivity_exponent = ptf.pore_connectivity_exponent,
                               regression_coefficients = ptf.regression_coefficients)

Base.summary(ptf::ContinuousPedotransfer) =
    string("ContinuousPedotransfer(organic_matter=", prettysummary(ptf.organic_matter),
           ", topsoil_depth=", prettysummary(ptf.topsoil_depth),
           ", θʳ=", prettysummary(ptf.residual_liquid_fraction),
           ", ℓ=", prettysummary(ptf.pore_connectivity_exponent), ")")

@inline soil_hydraulic_parameters(ptf::PedotransferFunction, sand, silt, clay, bulk_density) =
    soil_hydraulic_parameters(ptf, sand, silt, clay, bulk_density, zero(bulk_density))

@inline function soil_hydraulic_parameters(ptf::ContinuousPedotransfer,
                                           sand, silt, clay, bulk_density, depth)
    FT = typeof(clay)

    # The reciprocal and logarithmic predictors are what break down outside the fit:
    # as texture → 0 the `1/C` term alone lifts `θs` past 1, and as `ρᵇ` → 0 the
    # `ln D` term drives `n` to 1, where `m = 1 - 1/n` vanishes and the retention
    # curve is singular. Hold both inside the range the regression behaves in.
    minimum_percentage   = convert(FT, 1)
    minimum_bulk_density = convert(FT, 1//2)     # g/cm³
    maximum_bulk_density = convert(FT, 2)

    # kg/kg → %, kg/m³ → g/cm³.
    C  = max(100 * clay, minimum_percentage)
    S  = max(100 * silt, minimum_percentage)
    OM = max(convert(FT, ptf.organic_matter), minimum_percentage)
    D  = clamp(bulk_density / 1000, minimum_bulk_density, maximum_bulk_density)
    T  = ifelse(convert(FT, depth) < convert(FT, ptf.topsoil_depth), one(FT), zero(FT))

    predictors = HYPRES_predictors(C, S, OM, D, T)

    c      = ptf.regression_coefficients
    θs     = apply_regression(c.porosity, predictors)
    αstar  = apply_regression(c.inverse_air_entry_head, predictors)
    nstar  = apply_regression(c.pore_size_uniformity, predictors)
    Ksstar = apply_regression(c.K_saturated, predictors)

    θʳ = convert(FT, ptf.residual_liquid_fraction)
    ν  = clamp(θs, θʳ + convert(FT, 1//100), one(FT) - eps(FT))
    α  = 100 * exp(αstar)                                    # cm⁻¹ → m⁻¹
    n  = 1 + exp(nstar)
    Kₛ = exp(Ksstar) * convert(FT, 1//100) / convert(FT, 86400)   # cm day⁻¹ → m s⁻¹

    return (porosity = ν,
            residual_liquid_fraction = θʳ,
            inverse_air_entry_head = α,
            pore_size_uniformity = n,
            K_saturated = Kₛ)
end
