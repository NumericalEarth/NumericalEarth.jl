#####
##### Optional melt physics: stratification feedback, convective limit, ice heat conduction
#####

using DocStringExtensions: TYPEDEF, TYPEDFIELDS, TYPEDSIGNATURES

"""
$(TYPEDEF)

Stratification feedback on the heat and salt transfer coefficients
([yung2025](@citet), following [rosevear2022](@citet)). Under melting, the
meltwater stratifies the ice-shelf boundary layer and suppresses turbulent
transfer. The suppression is measured by the viscous Obukhov scale

```math
L^+ = - \\frac{u_*^4}{k \\, ν \\, B_b} ,
\\qquad
B_b = - g \\left[ β (S - S_b) γ_S - α (T - T_b) γ_T \\right] ,
```

and the transfer coefficients become

```math
Γ_T = \\min(A_T \\, L^{+\\, n_T}, \\, Γ_T^{CC}) ,
\\qquad
Γ_S = \\min(A_S \\, L^{+\\, n_S}, \\, Γ_S^{CC}) ,
```

with ``γ_{T, S} = Γ_{T, S} u_*``. The caps ``Γ^{CC}`` are the
`heat_transfer_coefficient` and `salt_transfer_coefficient` of the interface's
`ThreeEquationHeatFlux`. The interface is first solved with the caps; if that
gives ``B_b < 0`` (melting), ``L^+`` is iterated `iterations` times, each
time relaxing it halfway (`relaxation`) toward its value from the latest
interface state. Otherwise (freezing) the caps are kept.

Defaults are the fit of [yung2025](@citet) (Table 1), whose caps are
`ThreeEquationHeatFlux(heat_transfer_coefficient = 0.012, salt_transfer_coefficient = 3.9e-4)`
with drag coefficient ``C_d = 0.0025``.

$(TYPEDFIELDS)
"""
struct StratFeedback{FT}
    "heat prefactor ``A_T``"
    heat_prefactor :: FT
    "heat exponent ``n_T``"
    heat_exponent :: FT
    "salt prefactor ``A_S``"
    salt_prefactor :: FT
    "salt exponent ``n_S``"
    salt_exponent :: FT
    "molecular kinematic viscosity ``ν`` (m² s⁻¹)"
    molecular_viscosity :: FT
    "von Kármán constant ``k``"
    von_karman_constant :: FT
    "thermal expansion coefficient ``α`` (K⁻¹)"
    thermal_expansion :: FT
    "haline contraction coefficient ``β`` (psu⁻¹)"
    haline_contraction :: FT
    "gravitational acceleration ``g`` (m s⁻²)"
    gravitational_acceleration :: FT
    "fraction of the way ``L^+`` moves toward its new value at each iteration"
    relaxation :: FT
    "number of ``L^+`` iterations"
    iterations :: Int
end

"""
$(TYPEDSIGNATURES)

Construct a [`StratFeedback`](@ref) with the coefficients of [yung2025](@citet).

```jldoctest
using NumericalEarth.IceShelves: StratFeedback

StratFeedback()

# output
StratFeedback{Float64}(A_T = 0.000616595, n_T = 0.322, A_S = 5.01187e-5, n_S = 0.223, iterations = 10)
```
"""
function StratFeedback(FT::DataType = Oceananigans.defaults.FloatType;
                       heat_prefactor = 10^(-3.21),
                       heat_exponent = 0.322,
                       salt_prefactor = 10^(-4.30),
                       salt_exponent = 0.223,
                       molecular_viscosity = 1.95e-6,
                       von_karman_constant = 0.4,
                       thermal_expansion = 3.733e-5,
                       haline_contraction = 7.843e-4,
                       gravitational_acceleration = 9.80,
                       relaxation = 0.5,
                       iterations = 10)

    return StratFeedback(convert(FT, heat_prefactor),
                         convert(FT, heat_exponent),
                         convert(FT, salt_prefactor),
                         convert(FT, salt_exponent),
                         convert(FT, molecular_viscosity),
                         convert(FT, von_karman_constant),
                         convert(FT, thermal_expansion),
                         convert(FT, haline_contraction),
                         convert(FT, gravitational_acceleration),
                         convert(FT, relaxation),
                         iterations)
end

"""
$(TYPEDEF)

Convective lower bound on the transfer coefficients for steep ice bases
([mcconnochie2018](@citet), as implemented by [yung2025](@citet)). When
melting is driven by meltwater convection rather than shear,

```math
γ_T^{conv} = χ \\, \\sin^{2/3} θ \\left( \\frac{g β (S - S_b)}{ν} \\right)^{1/3} κ_S^{1/6} κ_T^{1/2} ,
\\qquad
γ_S^{conv} = γ_T^{conv} \\sqrt{κ_S / κ_T} \\, \\frac{ρ_I S_b}{ρ_0 S} ,
```

with ``κ_T = ν / Pr``, ``κ_S = ν / Sc`` and ``θ`` the basal slope angle.
The transfer coefficients are ``\\max(γ, γ^{conv})``. The thermal contribution
to the buoyancy is neglected, as in [yung2025](@citet). Requires the
`basal_slope` keyword of [`IceShelfOceanInterface`](@ref).

$(TYPEDFIELDS)
"""
struct ConvectiveMeltLimit{FT}
    "convective prefactor ``χ``"
    coefficient :: FT
    "molecular kinematic viscosity ``ν`` (m² s⁻¹)"
    molecular_viscosity :: FT
    "Prandtl number ``ν / κ_T``"
    prandtl_number :: FT
    "Schmidt number ``ν / κ_S``"
    schmidt_number :: FT
    "haline contraction coefficient ``β`` (psu⁻¹)"
    haline_contraction :: FT
    "gravitational acceleration ``g`` (m s⁻²)"
    gravitational_acceleration :: FT
    "ice density ``ρ_I`` (kg m⁻³)"
    ice_density :: FT
    "number of interface solves when used without [`StratFeedback`](@ref)"
    iterations :: Int
end

"""
$(TYPEDSIGNATURES)

Construct a [`ConvectiveMeltLimit`](@ref) with the coefficients of [yung2025](@citet).

```jldoctest
using NumericalEarth.IceShelves: ConvectiveMeltLimit

ConvectiveMeltLimit()

# output
ConvectiveMeltLimit{Float64}(χ = 0.086, Pr = 13.8, Sc = 2432.0, iterations = 10)
```
"""
function ConvectiveMeltLimit(FT::DataType = Oceananigans.defaults.FloatType;
                             coefficient = 0.086,
                             molecular_viscosity = 1.95e-6,
                             prandtl_number = 13.8,
                             schmidt_number = 2432,
                             haline_contraction = 7.843e-4,
                             gravitational_acceleration = 9.80,
                             ice_density = 918,
                             iterations = 10)

    return ConvectiveMeltLimit(convert(FT, coefficient),
                               convert(FT, molecular_viscosity),
                               convert(FT, prandtl_number),
                               convert(FT, schmidt_number),
                               convert(FT, haline_contraction),
                               convert(FT, gravitational_acceleration),
                               convert(FT, ice_density),
                               iterations)
end

"""
$(TYPEDEF)

Turbulent transfer coefficients of [holland1999modeling](@citet) that depend on
the friction velocity and on the stability of the ice-shelf boundary layer
(the HJ99-M81 form of [yung2025](@citet), Appendix A1),

```math
γ_{T, S} = \\frac{u_*}{Γ_{turb} + Γ_{mole}^{T, S}} ,
\\qquad
Γ_{mole}^T = 12.5 \\, Pr^{2/3} - 6 ,
\\qquad
Γ_{mole}^S = 12.5 \\, Sc^{2/3} - 6 ,
```

```math
Γ_{turb} = \\frac{1}{k} \\ln \\left( \\frac{u_* ξ_N η_*^2}{|f| h_ν} \\right) + \\frac{1}{2 ξ_N η_*} - \\frac{1}{k} ,
\\qquad
h_ν = \\frac{5 ν}{u_*} ,
```

with the stability parameter

```math
η_* = \\left( 1 + \\frac{ξ_N u_*}{|f| L R_c} \\right)^{-1/2} ,
\\qquad
L = - \\frac{u_*^3}{k B_b} ,
```

where ``f`` is the Coriolis parameter of the ocean model and ``B_b`` is the
interface buoyancy flux of [`StratFeedback`](@ref). Under freezing (``B_b \\ge 0``)
``η_* = 1``; `critical_richardson_number = Inf` gives ``η_* = 1`` always (neutral).
Where ``f = 0`` the logarithm is dropped and ``η_* = 1``.

The interface is first solved with ``η_* = 1``; ``η_*`` is then iterated
`iterations` times, each time relaxing it by `relaxation` toward its value
from the latest interface state. The `heat_transfer_coefficient` and
`salt_transfer_coefficient` of the interface's `ThreeEquationHeatFlux` are not used.

$(TYPEDFIELDS)
"""
struct RotatingBoundaryLayerTransfer{FT}
    "dimensionless stability constant ``ξ_N``"
    stability_constant :: FT
    "critical flux Richardson number ``R_c``"
    critical_richardson_number :: FT
    "Prandtl number ``ν / κ_T``"
    prandtl_number :: FT
    "Schmidt number ``ν / κ_S``"
    schmidt_number :: FT
    "molecular kinematic viscosity ``ν`` (m² s⁻¹)"
    molecular_viscosity :: FT
    "von Kármán constant ``k``"
    von_karman_constant :: FT
    "thermal expansion coefficient ``α`` (K⁻¹)"
    thermal_expansion :: FT
    "haline contraction coefficient ``β`` (psu⁻¹)"
    haline_contraction :: FT
    "gravitational acceleration ``g`` (m s⁻²)"
    gravitational_acceleration :: FT
    "fraction of the way ``η_*`` moves toward its new value at each iteration"
    relaxation :: FT
    "number of ``η_*`` iterations"
    iterations :: Int
end

"""
$(TYPEDSIGNATURES)

Construct a [`RotatingBoundaryLayerTransfer`](@ref) with the constants of [yung2025](@citet) (Table A1).

```jldoctest
using NumericalEarth.IceShelves: RotatingBoundaryLayerTransfer

RotatingBoundaryLayerTransfer()

# output
RotatingBoundaryLayerTransfer{Float64}(ξ_N = 0.052, R_c = 0.2, Pr = 13.8, Sc = 2432.0, iterations = 10)
```
"""
function RotatingBoundaryLayerTransfer(FT::DataType = Oceananigans.defaults.FloatType;
                                stability_constant = 0.052,
                                critical_richardson_number = 0.2,
                                prandtl_number = 13.8,
                                schmidt_number = 2432,
                                molecular_viscosity = 1.95e-6,
                                von_karman_constant = 0.4,
                                thermal_expansion = 3.733e-5,
                                haline_contraction = 7.843e-4,
                                gravitational_acceleration = 9.80,
                                relaxation = 0.5,
                                iterations = 10)

    return RotatingBoundaryLayerTransfer(convert(FT, stability_constant),
                                  convert(FT, critical_richardson_number),
                                  convert(FT, prandtl_number),
                                  convert(FT, schmidt_number),
                                  convert(FT, molecular_viscosity),
                                  convert(FT, von_karman_constant),
                                  convert(FT, thermal_expansion),
                                  convert(FT, haline_contraction),
                                  convert(FT, gravitational_acceleration),
                                  convert(FT, relaxation),
                                  iterations)
end

struct DiffusiveIceHeatFlux end
struct AdvectiveDiffusiveIceHeatFlux end

"""
$(TYPEDEF)

Heat conduction into the ice shelf. By default it is diffusive
([holland1999modeling](@citet), eq. 21),

```math
Q_I = ρ_I c_I κ_I \\frac{T_b - T_s}{h} ,
```

which takes part of the ocean heat flux and so reduces the melt rate. The ice
thickness ``h`` is that of floating ice, ``h = ρ_0 |z_d| / ρ_I``, where ``z_d``
is the draft.

With `advective = true` the ice temperature profile is instead the steady
advective–diffusive one of a melting ice shelf ([holland1999modeling](@citet), eqs. 26 and 31),
for which the heat drawn into the ice is

```math
Q_I = ρ_I c_I \\, m \\, (T_b - T_s)
```

when the ocean is above its freezing point (``m`` the melt rate) and zero
otherwise; ``κ_I`` is then not used.

The ocean is cooled by the turbulent heat flux only.

$(TYPEDFIELDS)
"""
struct IceHeatConduction{FT, F}
    "ice density ``ρ_I`` (kg m⁻³)"
    ice_density :: FT
    "ice heat capacity ``c_I`` (J kg⁻¹ K⁻¹)"
    ice_heat_capacity :: FT
    "ice thermal diffusivity ``κ_I`` (m² s⁻¹)"
    ice_diffusivity :: FT
    "ice surface temperature ``T_s`` (°C)"
    surface_temperature :: FT
    "diffusive or advective–diffusive ice heat flux"
    flux :: F
end

"""
$(TYPEDSIGNATURES)

Construct an [`IceHeatConduction`](@ref), advective–diffusive if `advective = true`.

```jldoctest
using NumericalEarth.IceShelves: IceHeatConduction

IceHeatConduction(advective = true)

# output
IceHeatConduction{Float64}(ρ_I = 917.0, c_I = 2000.0, κ_I = 1.54e-6, T_s = -20.0, advective = true)
```
"""
function IceHeatConduction(FT::DataType = Oceananigans.defaults.FloatType;
                           ice_density = 917,
                           ice_heat_capacity = 2000,
                           ice_diffusivity = 1.54e-6,
                           surface_temperature = -20,
                           advective = false)

    flux = advective ? AdvectiveDiffusiveIceHeatFlux() : DiffusiveIceHeatFlux()

    return IceHeatConduction(convert(FT, ice_density),
                             convert(FT, ice_heat_capacity),
                             convert(FT, ice_diffusivity),
                             convert(FT, surface_temperature),
                             flux)
end

#####
##### Interface solve with the optional physics
#####

const DefaultMeltPhysics = NamedTuple{(:transfer_coefficients, :convective_limit, :ice_heat_conduction, :basal_slope),
                                      Tuple{Nothing, Nothing, Nothing, Nothing}}

# Without optional physics the shared interface solve applies.
@inline ice_ocean_interface_fluxes(::DefaultMeltPhysics, i, j, flux_formulation, ocean_state, ice_state,
                                   liquidus, zᵈ, properties, ℰ, u★) =
    ice_ocean_interface_fluxes(nothing, i, j, flux_formulation, ocean_state, ice_state, liquidus, zᵈ, properties, ℰ, u★)

@inline ice_ocean_interface_fluxes(::DefaultMeltPhysics, i, j, flux::ThreeEquationHeatFlux, ocean_state, ice_state,
                                   liquidus, zᵈ, properties, ℰ, u★) =
    ice_ocean_interface_fluxes(nothing, i, j, flux, ocean_state, ice_state, liquidus, zᵈ, properties, ℰ, u★)

@inline function ice_ocean_interface_fluxes(physics::NamedTuple, i, j, flux::ThreeEquationHeatFlux, ocean_state, ice_state,
                                            liquidus, zᵈ, properties, ℰ, u★)
    ρᵒᶜ = properties.reference_density
    cᵒᶜ = properties.heat_capacity
    Tᵒᶜ = ocean_state.T
    Sᵒᶜ = ocean_state.S
    ℵ   = ice_state.ℵ

    κ, Tˢ, ζ = conduction_parameters(physics.ice_heat_conduction, zᵈ, ρᵒᶜ, ℰ)
    sinθ = basal_slope_sine(physics.basal_slope, i, j)
    f    = coriolis_parameter(physics.transfer_coefficients, ocean_state)

    coefficients = (; ρᵒᶜ, cᵒᶜ, ℰ, κ, Tˢ, ζ, Sˢⁱ = ice_state.S, zᵈ, f)

    γᵀ = flux.heat_transfer_coefficient * u★
    γˢ = flux.salt_transfer_coefficient * u★
    Tᵦ, Sᵦ, q = solve_ice_shelf_interface(γᵀ, γˢ, Tᵒᶜ, Sᵒᶜ, Sᵒᶜ, liquidus, coefficients)

    γᵀ, γˢ, Tᵦ, Sᵦ, q = refine_transfer_coefficients(physics.transfer_coefficients, physics.convective_limit,
                                                    γᵀ, γˢ, Tᵦ, Sᵦ, q, Tᵒᶜ, Sᵒᶜ, u★, sinθ, flux,
                                                    liquidus, coefficients)

    𝒬 = ρᵒᶜ * cᵒᶜ * γᵀ * (Tᵒᶜ - Tᵦ) * ℵ

    return 𝒬, q * ℵ, Tᵦ, Sᵦ
end

# Solve with turbulent transfer velocities γᵀ, γˢ (m s⁻¹), relinearizing a TEOS-10 liquidus about Sᵦ.
@inline function solve_ice_shelf_interface(γᵀ, γˢ, Tᵒᶜ, Sᵒᶜ, Sᵦ, liquidus, c)
    η = c.ρᵒᶜ * c.cᵒᶜ * γᵀ / c.ℰ
    γ = c.ρᵒᶜ * γˢ
    return ice_interface_conditions(c.ζ, η, γ, c.κ, c.Tˢ, Tᵒᶜ, Sᵒᶜ, c.Sˢⁱ, at_depth(liquidus, c.zᵈ))
end

@inline function solve_ice_shelf_interface(γᵀ, γˢ, Tᵒᶜ, Sᵒᶜ, Sᵦ, liquidus::TEOS10Liquidus, c)
    η = c.ρᵒᶜ * c.cᵒᶜ * γᵀ / c.ℰ
    γ = c.ρᵒᶜ * γˢ
    Tᵦ = zero(Sᵦ)
    q  = zero(Sᵦ)

    for _ in 1:liquidus.iterations
        Tᵦ, Sᵦ, q = ice_interface_conditions(c.ζ, η, γ, c.κ, c.Tˢ, Tᵒᶜ, Sᵒᶜ, c.Sˢⁱ, linearize(liquidus, Sᵦ, c.zᵈ))
    end

    return Tᵦ, Sᵦ, q
end

@inline conduction_parameters(::Nothing, zᵈ, ρᵒᶜ, ℰ) = (zero(zᵈ), zero(zᵈ), nothing)

@inline function conduction_parameters(conduction::IceHeatConduction, zᵈ, ρᵒᶜ, ℰ)
    ρᴵ = conduction.ice_density
    h  = - ρᵒᶜ * zᵈ / ρᴵ
    κ  = ifelse(h > zero(h), ρᴵ * conduction.ice_heat_capacity * conduction.ice_diffusivity / (h * ℰ), zero(h))
    return κ, conduction.surface_temperature, nothing
end

@inline function conduction_parameters(conduction::IceHeatConduction{<:Any, AdvectiveDiffusiveIceHeatFlux}, zᵈ, ρᵒᶜ, ℰ)
    return zero(zᵈ), conduction.surface_temperature, conduction.ice_heat_capacity / ℰ
end

@inline ice_interface_conditions(::Nothing, η, γ, κ, Tˢ, Tᵒᶜ, Sᵒᶜ, Sˢⁱ, liquidus) =
    interface_conditions(η, γ, κ, Tˢ, Tᵒᶜ, Sᵒᶜ, Sˢⁱ, liquidus)

# Heat balance η (Tᵒᶜ - Tᵦ) + κ (Tˢ - Tᵦ) = q (1 + ζ (Tᵦ - Tˢ)), with ζ = c_I / ℰ while the ocean is above freezing.
@inline function ice_interface_conditions(ζ, η, γ, κ, Tˢ, Tᵒᶜ, Sᵒᶜ, Sˢⁱ, liquidus)
    λ₁ = - liquidus.salinity_slope
    λ₂ = liquidus.freshwater_melting_temperature

    ζ = ifelse(Tᵒᶜ > melting_temperature(liquidus, Sᵒᶜ), ζ, zero(ζ))
    A = 1 + ζ * (λ₂ - Tˢ)
    θ = η + κ
    P = η * Tᵒᶜ + κ * Tˢ

    a = λ₁ * (θ - γ * ζ)
    b = - γ * A - P + θ * (λ₂ - λ₁ * Sˢⁱ) + γ * ζ * λ₁ * Sᵒᶜ
    c = (P - θ * λ₂) * Sˢⁱ + γ * Sᵒᶜ * A

    ξ  = ifelse(a == zero(a), zero(a), one(a) / (2a))
    Δ  = max(b^2 - 4a * c, zero(a))
    Sᵦ = (-b - sqrt(Δ)) * ξ
    Sᵦ = ifelse(Sᵦ < zero(Sᵦ), (-b + sqrt(Δ)) * ξ, Sᵦ)
    Tᵦ = melting_temperature(liquidus, Sᵦ)

    q = (P - θ * Tᵦ) / (1 + ζ * (Tᵦ - Tˢ))

    return Tᵦ, Sᵦ, q
end

@inline basal_slope_sine(::Nothing, i, j) = false

@inline function basal_slope_sine(slope, i, j)
    s = @inbounds slope[i, j, 1]
    return s / sqrt(1 + s^2)
end

@inline refine_transfer_coefficients(::Nothing, ::Nothing, γᵀ, γˢ, Tᵦ, Sᵦ, q, args...) = (γᵀ, γˢ, Tᵦ, Sᵦ, q)

@inline function refine_transfer_coefficients(::Nothing, limit::ConvectiveMeltLimit, γᵀ₀, γˢ₀, Tᵦ, Sᵦ, q,
                                              Tᵒᶜ, Sᵒᶜ, u★, sinθ, flux, liquidus, c)
    γᵀ = γᵀ₀
    γˢ = γˢ₀

    for _ in 1:limit.iterations
        γᵀ, γˢ = convective_limit(limit, γᵀ₀, γˢ₀, Sᵒᶜ, Sᵦ, sinθ, c.ρᵒᶜ)
        Tᵦ, Sᵦ, q = solve_ice_shelf_interface(γᵀ, γˢ, Tᵒᶜ, Sᵒᶜ, Sᵦ, liquidus, c)
    end

    return γᵀ, γˢ, Tᵦ, Sᵦ, q
end

@inline function refine_transfer_coefficients(feedback::StratFeedback, limit, γᵀ, γˢ, Tᵦ, Sᵦ, q,
                                              Tᵒᶜ, Sᵒᶜ, u★, sinθ, flux, liquidus, c)
    Bᵦ = basal_buoyancy_flux(feedback, γᵀ, γˢ, Tᵒᶜ - Tᵦ, Sᵒᶜ - Sᵦ)

    if Bᵦ < 0
        L⁺ = viscous_obukhov_scale(feedback, u★, Bᵦ)

        for _ in 1:feedback.iterations
            Bᵦ = basal_buoyancy_flux(feedback, γᵀ, γˢ, Tᵒᶜ - Tᵦ, Sᵒᶜ - Sᵦ)
            L⁺ₙ = ifelse(Bᵦ < 0, viscous_obukhov_scale(feedback, u★, Bᵦ), L⁺)
            L⁺ += feedback.relaxation * (L⁺ₙ - L⁺)

            γᵀ = min(feedback.heat_prefactor * L⁺^feedback.heat_exponent, flux.heat_transfer_coefficient) * u★
            γˢ = min(feedback.salt_prefactor * L⁺^feedback.salt_exponent, flux.salt_transfer_coefficient) * u★
            γᵀ, γˢ = convective_limit(limit, γᵀ, γˢ, Sᵒᶜ, Sᵦ, sinθ, c.ρᵒᶜ)

            Tᵦ, Sᵦ, q = solve_ice_shelf_interface(γᵀ, γˢ, Tᵒᶜ, Sᵒᶜ, Sᵦ, liquidus, c)
        end
    end

    return γᵀ, γˢ, Tᵦ, Sᵦ, q
end

@inline basal_buoyancy_flux(f, γᵀ, γˢ, ΔT, ΔS) =
    - f.gravitational_acceleration * (f.haline_contraction * ΔS * γˢ - f.thermal_expansion * ΔT * γᵀ)

@inline viscous_obukhov_scale(f::StratFeedback, u★, Bᵦ) =
    - u★^4 / (f.von_karman_constant * f.molecular_viscosity * Bᵦ)

@inline coriolis_parameter(transfer_coefficients, ocean_state) = false
@inline coriolis_parameter(::RotatingBoundaryLayerTransfer, ocean_state) = ocean_state.f

@inline function refine_transfer_coefficients(hj::RotatingBoundaryLayerTransfer, limit, γᵀ, γˢ, Tᵦ, Sᵦ, q,
                                              Tᵒᶜ, Sᵒᶜ, u★, sinθ, flux, liquidus, c)
    η★ = one(u★)
    γᵀ, γˢ = rotating_boundary_layer_coefficients(hj, u★, η★, c.f)
    γᵀ, γˢ = convective_limit(limit, γᵀ, γˢ, Sᵒᶜ, Sᵦ, sinθ, c.ρᵒᶜ)
    Tᵦ, Sᵦ, q = solve_ice_shelf_interface(γᵀ, γˢ, Tᵒᶜ, Sᵒᶜ, Sᵦ, liquidus, c)

    for _ in 1:hj.iterations
        Bᵦ  = basal_buoyancy_flux(hj, γᵀ, γˢ, Tᵒᶜ - Tᵦ, Sᵒᶜ - Sᵦ)
        η★ += hj.relaxation * (stability_parameter(hj, u★, Bᵦ, c.f) - η★)

        γᵀ, γˢ = rotating_boundary_layer_coefficients(hj, u★, η★, c.f)
        γᵀ, γˢ = convective_limit(limit, γᵀ, γˢ, Sᵒᶜ, Sᵦ, sinθ, c.ρᵒᶜ)
        Tᵦ, Sᵦ, q = solve_ice_shelf_interface(γᵀ, γˢ, Tᵒᶜ, Sᵒᶜ, Sᵦ, liquidus, c)
    end

    return γᵀ, γˢ, Tᵦ, Sᵦ, q
end

# η★ = (1 + ξ_N u★ / (|f| L R_c))^(-1/2) with L = -u★³ / (k Bᵦ); neutral (η★ = 1) when Bᵦ ≥ 0 or f = 0.
@inline function stability_parameter(hj::RotatingBoundaryLayerTransfer, u★, Bᵦ, f)
    D = abs(f) * hj.critical_richardson_number * u★^2
    N = hj.stability_constant * hj.von_karman_constant * max(-Bᵦ, zero(Bᵦ))
    D⁻¹ = ifelse(D > zero(D), 1 / D, zero(D))
    return 1 / sqrt(1 + N * D⁻¹)
end

@inline function rotating_boundary_layer_coefficients(hj::RotatingBoundaryLayerTransfer, u★, η★, f)
    ξᴺ = hj.stability_constant
    k  = hj.von_karman_constant
    ν  = hj.molecular_viscosity

    F  = abs(f)
    F⁻¹ = ifelse(F > zero(F), 1 / F, zero(F))
    X  = max(ξᴺ * η★^2 * u★^2 * F⁻¹ / (5ν), eps(typeof(u★)))
    ln = ifelse(F > zero(F), log(X), zero(X))
    Γᵗ = ln / k + 1 / (2ξᴺ * η★) - 1 / k

    Γᵀ = 25 * cbrt(hj.prandtl_number^2) / 2 - 6
    Γˢ = 25 * cbrt(hj.schmidt_number^2) / 2 - 6

    γᵀ = max(u★ / (Γᵗ + Γᵀ), zero(u★))
    γˢ = max(u★ / (Γᵗ + Γˢ), zero(u★))

    return γᵀ, γˢ
end

@inline convective_limit(::Nothing, γᵀ, γˢ, Sᵒᶜ, Sᵦ, sinθ, ρᵒᶜ) = (γᵀ, γˢ)

@inline function convective_limit(limit::ConvectiveMeltLimit, γᵀ, γˢ, Sᵒᶜ, Sᵦ, sinθ, ρᵒᶜ)
    ν  = limit.molecular_viscosity
    κᵀ = ν / limit.prandtl_number
    κˢ = ν / limit.schmidt_number
    ΔS = max(Sᵒᶜ - Sᵦ, zero(Sᵦ))

    γᵀᶜ = limit.coefficient * cbrt(sinθ^2 * limit.gravitational_acceleration * limit.haline_contraction * ΔS / ν) *
          cbrt(sqrt(κˢ)) * sqrt(κᵀ)

    S⁻¹ = ifelse(Sᵒᶜ > 0, 1 / Sᵒᶜ, zero(Sᵒᶜ))
    γˢᶜ = γᵀᶜ * sqrt(κˢ / κᵀ) * limit.ice_density / ρᵒᶜ * Sᵦ * S⁻¹

    return max(γᵀ, γᵀᶜ), max(γˢ, γˢᶜ)
end

#####
##### Show methods
#####

Base.summary(::StratFeedback{FT}) where FT = "StratFeedback{$FT}"
Base.summary(::ConvectiveMeltLimit{FT}) where FT = "ConvectiveMeltLimit{$FT}"
Base.summary(::IceHeatConduction{FT}) where FT = "IceHeatConduction{$FT}"
Base.summary(::RotatingBoundaryLayerTransfer{FT}) where FT = "RotatingBoundaryLayerTransfer{$FT}"

Base.show(io::IO, hj::RotatingBoundaryLayerTransfer) =
    print(io, summary(hj), "(ξ_N = ", hj.stability_constant, ", R_c = ", hj.critical_richardson_number,
              ", Pr = ", hj.prandtl_number, ", Sc = ", hj.schmidt_number, ", iterations = ", hj.iterations, ")")

Base.show(io::IO, f::StratFeedback) =
    print(io, summary(f), "(A_T = ", round(f.heat_prefactor, sigdigits = 6), ", n_T = ", f.heat_exponent,
              ", A_S = ", round(f.salt_prefactor, sigdigits = 6), ", n_S = ", f.salt_exponent,
              ", iterations = ", f.iterations, ")")

Base.show(io::IO, l::ConvectiveMeltLimit) =
    print(io, summary(l), "(χ = ", l.coefficient, ", Pr = ", l.prandtl_number,
              ", Sc = ", l.schmidt_number, ", iterations = ", l.iterations, ")")

Base.show(io::IO, c::IceHeatConduction) =
    print(io, summary(c), "(ρ_I = ", c.ice_density, ", c_I = ", c.ice_heat_capacity,
              ", κ_I = ", c.ice_diffusivity, ", T_s = ", c.surface_temperature,
              ", advective = ", c.flux isa AdvectiveDiffusiveIceHeatFlux, ")")
