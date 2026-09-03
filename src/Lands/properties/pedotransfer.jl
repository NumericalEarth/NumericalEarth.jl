#####
##### Pedotransfer functions: soil texture and bulk density → van Genuchten hydraulic
##### parameters `(ν, θʳ, α, n, K₀, ηᴷ)`, evaluated per grid point and per depth layer inside
##### the `soil_hydraulic_properties` reduction.
#####

"""
    abstract type PedotransferFunction

A pedotransfer function maps basic soil properties (texture, bulk density) to the
van Genuchten–Mualem hydraulic parameters. Concrete subtypes implement

    soil_hydraulic_parameters(ptf, sand, silt, clay, bulk_density, depth = 0)
        -> (; porosity, residual_liquid_fraction, inverse_air_entry_head,
              pore_size_uniformity, matching_point_conductivity,
              pore_connectivity_exponent)

with `sand`, `silt`, `clay` mass fractions (kg/kg), `bulk_density` in kg/m³, `depth` the
soil layer's depth below the surface (m, positive down), and outputs in model units (`α` in
m⁻¹, `K₀` in m s⁻¹). The keys match the keyword arguments of
[`VariablySaturatedHydrology`](@ref), [`VanGenuchtenRetention`](@ref) and
[`VanGenuchtenConductivity`](@ref). Subtypes also implement [`convert_eltype`](@ref).

[`WeynantsPedotransfer`](@ref) is the default; [`HYPRESPedotransfer`](@ref) is the
Wösten alternative.
"""
abstract type PedotransferFunction end

"""
    convert_eltype(FT, parameters)

Rebuild a parameter set with every coefficient at float type `FT`, so that a kernel at
`FT` receives no `Float64`.
"""
function convert_eltype end

@inline apply_regression(coefficients, predictors) = sum(map(*, coefficients, predictors))

#####
##### Weynants et al. (2009) — the default.
#####

@inline weynants_predictors(C, S, D, OC) = (1, C, S, S^2, D, OC)

"""
    WeynantsRegression(; porosity, inverse_air_entry_head, pore_size_uniformity,
                         matching_point_conductivity, pore_connectivity_exponent)

Coefficients of the [Weynants et al. (2009)](@cite weynants2009) pedotransfer
regressions, one tuple per predicted parameter, each ordered by the shared predictor
basis `(1, C, S, S², D, OC)` in clay %, sand %, bulk density g/cm³ and organic
carbon %. `porosity` and `pore_connectivity_exponent` are predicted directly; the other
three predict `ln α`, `ln(n - 1)` and `ln K₀`.

Construct one to swap in a different calibration of the same functional form;
[`WEYNANTS_REGRESSION`](@ref) holds the published fit.
"""
struct WeynantsRegression{C}
    porosity                    :: C
    inverse_air_entry_head      :: C
    pore_size_uniformity        :: C
    matching_point_conductivity :: C
    pore_connectivity_exponent  :: C
end

# `float` keeps the tuples homogeneous so integer zeros can be written literally.
WeynantsRegression(; porosity, inverse_air_entry_head, pore_size_uniformity,
                     matching_point_conductivity, pore_connectivity_exponent) =
    WeynantsRegression(map(float, porosity),
                       map(float, inverse_air_entry_head),
                       map(float, pore_size_uniformity),
                       map(float, matching_point_conductivity),
                       map(float, pore_connectivity_exponent))

convert_eltype(::Type{FT}, r::WeynantsRegression) where FT =
    WeynantsRegression(convert.(FT, r.porosity),
                       convert.(FT, r.inverse_air_entry_head),
                       convert.(FT, r.pore_size_uniformity),
                       convert.(FT, r.matching_point_conductivity),
                       convert.(FT, r.pore_connectivity_exponent))

"""
    WEYNANTS_REGRESSION

The [Weynants et al. (2009)](@cite weynants2009) pedotransfer coefficients (their
Table 6), with the units of the [Weihermüller et al. (2017)](@cite weihermuller2017)
erratum. Each field predicts one parameter:

| field | predicts | units | source |
|:------|:---------|:------|:-------|
| `porosity` | `ν` (`θs` in the paper) | – | clay, `ρᵇ` |
| `inverse_air_entry_head` | `ln α` | ln(cm⁻¹) | clay, sand, OC |
| `pore_size_uniformity` | `ln(n - 1)` | – | clay, sand, sand² |
| `matching_point_conductivity` | `ln K₀` | ln(cm day⁻¹) | sand, `ρᵇ`, OC |
| `pore_connectivity_exponent` | `ηᴷ` (`λ` in the paper) | – | clay, sand |

`θʳ` is absent because Weynants fitted it and found it "not significantly different from 0".
"""
const WEYNANTS_REGRESSION = WeynantsRegression(
    #                            1,        C,       S,      S²,       D,       OC
    porosity                    = (0.6355,  0.0013,  0,       0,      -0.1631,  0),
    inverse_air_entry_head      = (-4.3003, -0.0097, 0.0138,  0,       0,      -0.0992),
    pore_size_uniformity        = (-1.0846, -0.0236, -0.0085, 0.0001,  0,       0),
    matching_point_conductivity = (1.9582,  0,       0.0308,  0,      -0.6142, -0.1566),
    pore_connectivity_exponent  = (-1.8642, -0.1317, 0.0067,  0,       0,       0))

"""
    WeynantsPedotransfer(FT = Oceananigans.defaults.FloatType;
                         organic_carbon = 0.58,
                         regression_coefficients = WEYNANTS_REGRESSION)

The [Weynants et al. (2009)](@cite weynants2009) pedotransfer function: closed-form
regressions mapping clay %, sand %, bulk density and organic carbon % to all six
Mualem–van Genuchten parameters, fitted in a single inversion against measured retention
and conductivity curves.

`organic_carbon` (% by weight, per the [Weihermüller et al. (2017)](@cite weihermuller2017)
erratum) is a uniform value, since the 30 m texture datasets do not carry it; it enters `α`
and `K₀` only. 0.58 % is the same soil as [`HYPRESPedotransfer`](@ref)'s 1 % organic matter
under the van Bemmelen convention that organic matter is 58 % carbon.
`regression_coefficients` is a [`WeynantsRegression`](@ref).

Predicted `θs` is returned as the porosity `ν` and `θʳ` is zero. Units are converted to
model units: `α` (cm⁻¹) → m⁻¹, `K₀` (cm day⁻¹) → m s⁻¹. `K₀` is a matrix value: the fit
excluded measurements wetter than 6 cm suction, so macropore flow is not in it.

Predictors are clamped to the fit's own ranges — clay ≤ 54.5 %, sand 5.6–97.8 %, `ρᵇ`
0.89–1.77 g/cm³, OC 0.01–6.6 % — so a soil past those bounds receives the parameters of
the boundary.
"""
struct WeynantsPedotransfer{FT, C} <: PedotransferFunction
    organic_carbon          :: FT
    regression_coefficients :: C
end

WeynantsPedotransfer(FT::Type = Oceananigans.defaults.FloatType;
                     organic_carbon = 0.58,
                     regression_coefficients = WEYNANTS_REGRESSION) =
    WeynantsPedotransfer(convert(FT, organic_carbon),
                         convert_eltype(FT, regression_coefficients))

convert_eltype(::Type{FT}, ptf::WeynantsPedotransfer) where FT =
    WeynantsPedotransfer(FT; organic_carbon = ptf.organic_carbon,
                             regression_coefficients = ptf.regression_coefficients)

Base.summary(ptf::WeynantsPedotransfer) =
    string("WeynantsPedotransfer(organic_carbon=", prettysummary(ptf.organic_carbon), ")")

@inline function soil_hydraulic_parameters(ptf::WeynantsPedotransfer, sand, silt, clay, bulk_density, depth = 0)
    FT = typeof(clay)

    # kg/kg → %, kg/m³ → g/cm³, each held inside the fit's range (Weynants Table 1)
    C  = clamp(100clay, 0, convert(FT, 54.5))
    S  = clamp(100sand, convert(FT, 5.6), convert(FT, 97.8))
    D  = clamp(bulk_density / 1000, convert(FT, 0.89), convert(FT, 1.77))
    OC = clamp(convert(FT, ptf.organic_carbon), convert(FT, 1//100), convert(FT, 6.6))

    predictors = weynants_predictors(C, S, D, OC)
    c = ptf.regression_coefficients

    ν  = apply_regression(c.porosity, predictors)
    α  = 100 * exp(apply_regression(c.inverse_air_entry_head, predictors))            # cm⁻¹ → m⁻¹
    n  = 1 + exp(apply_regression(c.pore_size_uniformity, predictors))
    K₀ = exp(apply_regression(c.matching_point_conductivity, predictors)) / 8_640_000  # cm day⁻¹ → m s⁻¹
    ηᴷ = apply_regression(c.pore_connectivity_exponent, predictors)

    return (porosity = ν,
            residual_liquid_fraction = zero(FT),
            inverse_air_entry_head = α,
            pore_size_uniformity = n,
            matching_point_conductivity = K₀,
            pore_connectivity_exponent = ηᴷ)
end

#####
##### Wösten et al. (1999) HYPRES — the wider European alternative. Every regression is
##### linear in a shared basis of clay % `C`, silt % `S`, organic matter % `OM`, bulk
##### density `D` (g/cm³) and the topsoil flag.
#####

@inline hypres_predictors(C, S, OM, D, topsoil) =
    (1, C, S, OM, D, topsoil, D^2, C^2, OM^2, S^2,
     1/C, 1/S, 1/OM, 1/D, log(S), log(OM), log(D),
     OM*C, D*C, D*S, D*OM, topsoil*C, topsoil*S)

"""
    HYPRESRegression(; porosity, inverse_air_entry_head, pore_size_uniformity,
                       matching_point_conductivity)

Coefficients of the continuous pedotransfer regressions fitted to HYPRES, the
database of HYdraulic PRoperties of European Soils. One tuple per predicted
parameter, each ordered by the shared predictor basis. `porosity` predicts `ν`
directly; the other three predict `ln α`, `ln(n - 1)` and `ln K₀`, the transforms that
enforce `α > 0`, `n > 1` and `K₀ > 0`.

Construct one to swap in a different calibration of the same functional form;
[`HYPRES_REGRESSION`](@ref) holds the published fit.
"""
struct HYPRESRegression{C}
    porosity                    :: C
    inverse_air_entry_head      :: C
    pore_size_uniformity        :: C
    matching_point_conductivity :: C
end

HYPRESRegression(; porosity, inverse_air_entry_head, pore_size_uniformity,
                   matching_point_conductivity) =
    HYPRESRegression(map(float, porosity),
                     map(float, inverse_air_entry_head),
                     map(float, pore_size_uniformity),
                     map(float, matching_point_conductivity))

convert_eltype(::Type{FT}, r::HYPRESRegression) where FT =
    HYPRESRegression(convert.(FT, r.porosity),
                     convert.(FT, r.inverse_air_entry_head),
                     convert.(FT, r.pore_size_uniformity),
                     convert.(FT, r.matching_point_conductivity))

"""
    HYPRES_REGRESSION

The [Wösten et al. (1999)](@cite wosten1999) continuous pedotransfer regressions
fitted to HYPRES, the database of HYdraulic PRoperties of European Soils (their
Table 5). Each field predicts one parameter:

| field | predicts | units | R² |
|:------|:---------|:------|:---|
| `porosity` | `ν` (`θs` in the paper) | – | 76 % |
| `inverse_air_entry_head` | `ln α` | ln(cm⁻¹) | 20 % |
| `pore_size_uniformity` | `ln(n - 1)` | – | 54 % |
| `matching_point_conductivity` | `ln K₀` | ln(cm day⁻¹) | 19 % |

There is no regression for `θʳ`, and Wösten's regression for `ηᴷ` (`R² = 12 %`) is not
included; [`HYPRESPedotransfer`](@ref) carries both as constants.

Coefficients follow the `hypres_predictors` order, one row per predictor family. A zero
marks a predictor the paper's subset selection dropped.
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

    matching_point_conductivity =
        (7.755, 0, 0.0352, 0, 0, 0.93,                                   # 1, C, S, OM, D, T
         -0.967, -0.000484, 0, -0.000322,                                # D², C², OM², S²
         0, 0.001, -0.0748, 0,                                           # 1/C, 1/S, 1/OM, 1/D
         -0.643, 0, 0,                                                   # ln S, ln OM, ln D
         0, -0.01398, 0, -0.1673, 0.02986, -0.03305))                    # OM·C, D·C, D·S, D·OM, T·C, T·S

"""
    HYPRESPedotransfer(FT = Oceananigans.defaults.FloatType;
                       organic_matter = 1,
                       topsoil_depth = 0.3,
                       residual_liquid_fraction = 0.01,
                       pore_connectivity_exponent = 0.5,
                       regression_coefficients = HYPRES_REGRESSION)

The [Wösten et al. (1999)](@cite wosten1999) continuous pedotransfer function (their
Table 5), fitted to HYPRES, the database of HYdraulic PRoperties of European Soils:
closed-form regressions in clay %, silt %, organic matter %, bulk density (g/cm³) and a
topsoil/subsoil flag. Clay is the < 2 μm fraction and silt the 2–50 μm fraction, matching
the USDA split the soil datasets report. Wösten's *class* regressions, which bin soil into
eleven texture classes, are not implemented.

Soil shallower than `topsoil_depth` (m) is topsoil. `organic_matter` (%) is a uniform
value. `residual_liquid_fraction` (`θʳ`) and `pore_connectivity_exponent` (`ηᴷ`) are
constants, since HYPRES supplies no usable regression for either: the defaults are Wösten's
Table 4 class fit for `θʳ` and the Mualem exponent an unreduced matching point pairs with.
`regression_coefficients` is a [`HYPRESRegression`](@ref).

Predicted `θs` is returned as the porosity `ν`. Units are converted to model units:
`α` (cm⁻¹) → m⁻¹, `K₀` (cm day⁻¹) → m s⁻¹.

Texture and organic matter are floored at 1 % and bulk density held within 0.5–2.0 g/cm³,
because they enter as `1/x` and `ln x`. Wösten warns the regressions "should not be used
for the assignment of hydraulic properties to soils outside Europe".
"""
struct HYPRESPedotransfer{FT, C} <: PedotransferFunction
    organic_matter             :: FT
    topsoil_depth              :: FT
    residual_liquid_fraction   :: FT
    pore_connectivity_exponent :: FT
    regression_coefficients    :: C
end

HYPRESPedotransfer(FT::Type = Oceananigans.defaults.FloatType;
                   organic_matter = 1,
                   topsoil_depth = 0.3,
                   residual_liquid_fraction = 1//100,
                   pore_connectivity_exponent = 1//2,
                   regression_coefficients = HYPRES_REGRESSION) =
    HYPRESPedotransfer(convert(FT, organic_matter),
                       convert(FT, topsoil_depth),
                       convert(FT, residual_liquid_fraction),
                       convert(FT, pore_connectivity_exponent),
                       convert_eltype(FT, regression_coefficients))

convert_eltype(::Type{FT}, ptf::HYPRESPedotransfer) where FT =
    HYPRESPedotransfer(FT; organic_matter = ptf.organic_matter,
                           topsoil_depth = ptf.topsoil_depth,
                           residual_liquid_fraction = ptf.residual_liquid_fraction,
                           pore_connectivity_exponent = ptf.pore_connectivity_exponent,
                           regression_coefficients = ptf.regression_coefficients)

Base.summary(ptf::HYPRESPedotransfer) =
    string("HYPRESPedotransfer(organic_matter=", prettysummary(ptf.organic_matter),
           ", topsoil_depth=", prettysummary(ptf.topsoil_depth),
           ", θʳ=", prettysummary(ptf.residual_liquid_fraction),
           ", ηᴷ=", prettysummary(ptf.pore_connectivity_exponent), ")")

@inline function soil_hydraulic_parameters(ptf::HYPRESPedotransfer, sand, silt, clay, bulk_density, depth = 0)
    FT = typeof(clay)

    # kg/kg → %, kg/m³ → g/cm³; the 1/C, 1/S and ln D predictors diverge at zero
    C  = max(100clay, 1)
    S  = max(100silt, 1)
    OM = max(convert(FT, ptf.organic_matter), 1)
    D  = clamp(bulk_density / 1000, convert(FT, 1//2), 2)
    topsoil = ifelse(depth < ptf.topsoil_depth, one(FT), zero(FT))

    predictors = hypres_predictors(C, S, OM, D, topsoil)
    c = ptf.regression_coefficients

    ν  = apply_regression(c.porosity, predictors)
    α  = 100 * exp(apply_regression(c.inverse_air_entry_head, predictors))            # cm⁻¹ → m⁻¹
    n  = 1 + exp(apply_regression(c.pore_size_uniformity, predictors))
    K₀ = exp(apply_regression(c.matching_point_conductivity, predictors)) / 8_640_000  # cm day⁻¹ → m s⁻¹

    return (porosity = ν,
            residual_liquid_fraction = convert(FT, ptf.residual_liquid_fraction),
            inverse_air_entry_head = α,
            pore_size_uniformity = n,
            matching_point_conductivity = K₀,
            pore_connectivity_exponent = convert(FT, ptf.pore_connectivity_exponent))
end
