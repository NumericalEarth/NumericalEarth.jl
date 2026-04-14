using Oceananigans.Grids: AbstractGrid, prettysummary

using Adapt
using Printf
using Thermodynamics: Liquid
using KernelAbstractions.Extras.LoopInfo: @unroll
using Statistics: norm

import Thermodynamics as AtmosphericThermodynamics
import Thermodynamics.Parameters: Rv_over_Rd

#####
##### Bulk turbulent fluxes based on similarity theory
#####

struct SimilarityTheoryFluxes{FT, UF, R, B, S}
    von_karman_constant :: FT        # parameter
    turbulent_prandtl_number :: FT   # parameter
    gustiness_parameter :: FT        # bulk velocity parameter
    minimum_gustiness :: FT          # minimum gustiness velocity [m/s]
    stability_functions :: UF        # functions for turbulent fluxes
    roughness_lengths :: R           # parameterization for turbulent fluxes
    similarity_form :: B             # similarity profile relating atmosphere to interface state
    solver_stop_criteria :: S        # stop criteria for compute_interface_state
end

Adapt.adapt_structure(to, fluxes::SimilarityTheoryFluxes) =
    SimilarityTheoryFluxes(adapt(to, fluxes.von_karman_constant),
                           adapt(to, fluxes.turbulent_prandtl_number),
                           adapt(to, fluxes.gustiness_parameter),
                           adapt(to, fluxes.minimum_gustiness),
                           adapt(to, fluxes.stability_functions),
                           adapt(to, fluxes.roughness_lengths),
                           adapt(to, fluxes.similarity_form),
                           adapt(to, fluxes.solver_stop_criteria))


Base.summary(::SimilarityTheoryFluxes{FT}) where FT = "SimilarityTheoryFluxes{$FT}"

function Base.show(io::IO, fluxes::SimilarityTheoryFluxes)
    print(io, summary(fluxes), '\n',
          "├── von_karman_constant: ",        prettysummary(fluxes.von_karman_constant), '\n',
          "├── turbulent_prandtl_number: ",   prettysummary(fluxes.turbulent_prandtl_number), '\n',
          "├── gustiness_parameter: ",        prettysummary(fluxes.gustiness_parameter), '\n',
          "├── minimum_gustiness: ",          prettysummary(fluxes.minimum_gustiness), '\n',
          "├── stability_functions: ",        summary(fluxes.stability_functions), '\n',
          "├── roughness_lengths: ",          summary(fluxes.roughness_lengths), '\n',
          "├── similarity_form: ",            summary(fluxes.similarity_form), '\n',
          "└── solver_stop_criteria: ",       summary(fluxes.solver_stop_criteria))
end

"""
    SimilarityTheoryFluxes(FT::DataType = Float64;
                           gravitational_acceleration = 9.81,
                           von_karman_constant = 0.4,
                           turbulent_prandtl_number = 1,
                           gustiness_parameter = 1.2,
                           minimum_gustiness = 0.01,
                           stability_functions = default_stability_functions(FT),
                           roughness_lengths = default_roughness_lengths(FT),
                           similarity_form = LogarithmicSimilarityProfile(),
                           solver_stop_criteria = nothing,
                           solver_tolerance = 1e-8,
                           solver_maxiter = 100)

`SimilarityTheoryFluxes` contains parameters and settings to calculate
air-interface turbulent fluxes using Monin--Obukhov similarity theory.

Keyword Arguments
==================

- `von_karman_constant`: The von Karman constant. Default: 0.4.
- `turbulent_prandtl_number`: The turbulent Prandtl number. Default: 1.
- `gustiness_parameter`: Scaling factor for convective gustiness velocity. Default: 1.2.
- `minimum_gustiness`: Minimum gustiness velocity [m/s], used as a floor in stable conditions
                       where convective gustiness is zero. Default: 0.01.
- `stability_functions`: The stability functions. Default: `default_stability_functions(FT)` that follow the
                         formulation of [edson2013exchange](@citet).
- `roughness_lengths`: The roughness lengths used to calculate the characteristic scales for momentum, temperature and
                       water vapor. Default: `default_roughness_lengths(FT)`, formulation taken from [edson2013exchange](@citet).
- `similarity_form`: The type of similarity profile used to relate the atmospheric state to the
                             interface fluxes / characteristic scales.
- `solver_tolerance`: The tolerance for convergence. Default: 1e-8.
- `solver_maxiter`: The maximum number of iterations. Default: 100.
"""
function SimilarityTheoryFluxes(FT::DataType = Oceananigans.defaults.FloatType;
                                von_karman_constant = 0.4,
                                turbulent_prandtl_number = 1,
                                gustiness_parameter = 1.2,
                                minimum_gustiness = 0.01,
                                stability_functions = atmosphere_ocean_stability_functions(FT),
                                momentum_roughness_length = MomentumRoughnessLength(FT),
                                temperature_roughness_length = ScalarRoughnessLength(FT),
                                water_vapor_roughness_length = ScalarRoughnessLength(FT),
                                similarity_form = LogarithmicSimilarityProfile(),
                                solver_stop_criteria = nothing,
                                solver_tolerance = 1e-8,
                                solver_maxiter = 100)

    roughness_lengths = SimilarityScales(momentum_roughness_length,
                                         temperature_roughness_length,
                                         water_vapor_roughness_length)

    if isnothing(solver_stop_criteria)
        solver_tolerance = convert(FT, solver_tolerance)
        solver_stop_criteria = ConvergenceStopCriteria(solver_tolerance, solver_maxiter)
    end

    if isnothing(stability_functions)
        returns_zero = Returns(zero(FT))
        stability_functions = SimilarityScales(returns_zero, returns_zero, returns_zero)
    end

    return SimilarityTheoryFluxes(convert(FT, von_karman_constant),
                                  convert(FT, turbulent_prandtl_number),
                                  convert(FT, gustiness_parameter),
                                  convert(FT, minimum_gustiness),
                                  stability_functions,
                                  roughness_lengths,
                                  similarity_form,
                                  solver_stop_criteria)
end

#####
##### Similarity profile types
#####

"""
    LogarithmicSimilarityProfile()

Represent the classic Monin--Obukhov similarity profile, which finds that

```math
ϕ(z) = Π(z) ϕ_★ / ϰ
```

where ``ϰ`` is the Von Karman constant, ``ϕ_★`` is the characteristic scale for ``ϕ``,
and ``Π`` is the "similarity profile",

```math
Π(h) = \\log(h / ℓ) - ψ(h / L) + ψ(ℓ / L)
```

which is a logarithmic profile adjusted by the stability function ``ψ`` and dependent on
the Monin--Obukhov length ``L`` and the roughness length ``ℓ``.
"""
struct LogarithmicSimilarityProfile end
struct COARELogarithmicSimilarityProfile end

@inline function similarity_profile(::LogarithmicSimilarityProfile, stability_function, h, ℓ, L)
    ζ = h / L
    ψh = stability_profile(stability_function, ζ)
    ψℓ = stability_profile(stability_function, ℓ / L)
    return log(h / ℓ) - ψh + ψℓ
end

@inline function similarity_profile(::COARELogarithmicSimilarityProfile, stability_function, h, ℓ, L)
    ζ = h / L
    ψh = stability_profile(stability_function, ζ)
    return log(h / ℓ) - ψh
end

function iterate_interface_fluxes(flux_formulation::SimilarityTheoryFluxes,
                                  Tₛ, qₛ, Δθ, Δq, Δh,
                                  approximate_interface_state,
                                  atmosphere_state,
                                  interface_properties,
                                  atmosphere_properties)

    ℂᵃᵗ = atmosphere_properties.thermodynamics_parameters
    g  = atmosphere_properties.gravitational_acceleration
    pᵃᵗ = atmosphere_state.p

    # "initial" scales because we will recompute them
    u★ = approximate_interface_state.u★
    θ★ = approximate_interface_state.θ★
    q★ = approximate_interface_state.q★

    # Stability functions for momentum, heat, and vapor
    ψu = flux_formulation.stability_functions.momentum
    ψθ = flux_formulation.stability_functions.temperature
    ψq = flux_formulation.stability_functions.water_vapor

    # Extract roughness lengths
    ℓu = flux_formulation.roughness_lengths.momentum
    ℓθ = flux_formulation.roughness_lengths.temperature
    ℓq = flux_formulation.roughness_lengths.water_vapor
    β  = flux_formulation.gustiness_parameter

    # Compute Monin--Obukhov length scale depending on a `buoyancy flux`
    b★ = buoyancy_scale(θ★, q★, ℂᵃᵗ, Tₛ, qₛ, g)

    # Buoyancy flux characteristic scale for gustiness.
    # In unstable conditions (Jᵇ > 0), gustiness = β * (Jᵇ * h_bℓ)^(1/3).
    # In stable conditions, a baseline gustiness is used (default 0.2 m/s).
    h_bℓ = atmosphere_state.h_bℓ
    Jᵇ = - u★ * b★
    Uᴳ₀ = flux_formulation.minimum_gustiness
    Uᴳ = max(Uᴳ₀, β * cbrt(max(zero(Jᵇ), Jᵇ) * h_bℓ))

    # Velocity difference accounting for gustiness
    Δu, Δv = velocity_difference(interface_properties.velocity_formulation,
                                 atmosphere_state,
                                 approximate_interface_state)

    U = sqrt(Δu^2 + Δv^2 + Uᴳ^2)

    # Compute roughness length scales (pass surface temperature for viscosity calculation)
    ℓu₀ = roughness_length(ℓu, u★, U, ℂᵃᵗ, Tₛ)
    ℓq₀ = roughness_length(ℓq, ℓu₀, u★, U, ℂᵃᵗ, Tₛ)
    ℓθ₀ = roughness_length(ℓθ, ℓu₀, u★, U, ℂᵃᵗ, Tₛ)

    # Transfer coefficients at height `h`
    ϰ = flux_formulation.von_karman_constant
    L★ = ifelse(b★ == 0, Inf, u★^2 / (ϰ * b★))
    form = flux_formulation.similarity_form

    χu = ϰ / similarity_profile(form, ψu, Δh, ℓu₀, L★)
    χθ = ϰ / similarity_profile(form, ψθ, Δh, ℓθ₀, L★)
    χq = ϰ / similarity_profile(form, ψq, Δh, ℓq₀, L★)

    # Recompute
    u★ = χu * U
    θ★ = χθ * Δθ
    q★ = χq * Δq

    return u★, θ★, q★
end

"""
    buoyancy_scale(θ★, q★, ℂᵃᵗ, Tₛ, qₛ, g)

Return the characteristic buoyancy scale `b★` associated with
the characteristic temperature `θ★`, specific humidity scale `q★`,
surface temperature `Tₛ`, surface specific humidity `qₛ`,
atmosphere thermodynamic parameters `ℂᵃᵗ`, and gravitational acceleration `g`.

The buoyancy scale is defined in terms of the interface buoyancy flux,

```math
u★ b★ ≡ w'b',
```

where `u_★` is the friction velocity.
Using the definition of buoyancy for clear air without condensation, we find that

```math
b★ = (g / 𝒯ₛ) [θ★ (1 + δ qₛ) + δ 𝒯ₛ q★] ,
```
where ``𝒯ₛ`` is the virtual temperature at the surface, and ``δ = Rᵛ / Rᵈ - 1``,
where ``Rᵛ`` is the molar mass of water vapor and ``Rᵈ`` is the molar mass of dry air.

Note that the Monin--Obukhov characteristic length scale is defined
in terms of ``b★`` and additionally the Von Karman constant ``ϰ``,

```math
L★ = u★² / ϰ b★ .
```
"""
@inline function buoyancy_scale(θ★, q★, ℂᵃᵗ, Tₛ, qₛ, g)
    𝒯ₛ = AtmosphericThermodynamics.virtual_temperature(ℂᵃᵗ, Tₛ, qₛ)
    ε  = AtmosphericThermodynamics.Parameters.Rv_over_Rd(ℂᵃᵗ)
    δ  = ε - 1 # typically equal to 0.608

    b★ = g / 𝒯ₛ * (θ★ * (1 + δ * qₛ) + δ * 𝒯ₛ * q★)

    return b★
end

import Statistics

#####
##### Struct that represents a 3-tuple of momentum, heat, and water vapor
#####

struct SimilarityScales{U, T, Q}
    momentum :: U
    temperature :: T
    water_vapor :: Q
end

Base.summary(ss::SimilarityScales) =
    string("SimilarityScales(momentum=", prettysummary(ss.momentum),
           ", temperature=", prettysummary(ss.temperature),
           ", water_vapor=", prettysummary(ss.water_vapor), ")")

Base.show(io::IO, ss::SimilarityScales) = print(io, summary(ss))

@inline stability_profile(ψ, ζ) = ψ(ζ)

# Convenience
abstract type AbstractStabilityFunction end
@inline (ψ::AbstractStabilityFunction)(ζ) = stability_profile(ψ, ζ)

"""
    EdsonMomentumStabilityFunction{FT}

A struct representing the momentum stability function detailed by [edson2013exchange](@citet).
The formulation hinges on the definition of three different functions:
one for stable atmospheric conditions ``(ζ > 0)``, named ``ψₛ`` and two for unstable conditions,
named ``ψᵤ₁`` and ``ψᵤ₂``.
These stability functions are obtained by regression to experimental data.

The stability parameter for stable atmospheric conditions is defined as
```math
\\begin{align*}
dζ &= \\min(ζ_{\\max}, A⁺ ζ) \\\\
ψ⁺ &= - B⁺ ζ⁺ - C⁺ (ζ⁺ - D⁺) \\exp(- dζ) - C⁺ D⁺
\\end{align*}
```

While the stability parameter for unstable atmospheric conditions is calculated
as a function of the two individual stability functions as follows

```math
\\begin{align*}
f⁻₁ &= (1 - A⁻ζ)^{1/4} \\\\
ψ⁻₁ &= (B⁻ / 2) \\log[(1 + f⁻₁ + f⁻₁² + f⁻₁³) / B⁻] - √B⁻ \\mathrm{atan}(f⁻₁) - C⁻ \\\\
\\\\
f⁻₂ &= ∛(1 - D⁻ζ) \\\\
ψ⁻₂ &= (E⁻ / 2) \\log[(1 + f⁻₂ + f⁻₂²) / E⁻]- √E⁻ \\mathrm{atan}[(1 + 2f⁻₂) / √E⁻] + F⁻ \\\\
\\\\
f   &= ζ² / (1 + ζ²) \\\\
ψ⁻  &= (1 - f) ψ⁻₁ + f ψ⁻₂
\\end{align*}
```

The superscripts ``+`` and ``-`` indicate if the parameter applies to the
stability function for _stable_ or _unstable_ atmospheric conditions, respectively.
"""
@kwdef struct EdsonMomentumStabilityFunction{FT} <: AbstractStabilityFunction
    ζmax :: FT = 50.0
    A⁺   :: FT = 0.35
    B⁺   :: FT = 0.7
    C⁺   :: FT = 0.75
    D⁺   :: FT = 5/0.35
    A⁻   :: FT = 15.0
    B⁻   :: FT = 2.0
    C⁻   :: FT = π/2
    D⁻   :: FT = 10.15
    E⁻   :: FT = 3.0
    F⁻   :: FT = π / sqrt(3)
end

@inline function stability_profile(ψ::EdsonMomentumStabilityFunction, ζ)
    ζmax = ψ.ζmax
    A⁺   = ψ.A⁺
    B⁺   = ψ.B⁺
    C⁺   = ψ.C⁺
    D⁺   = ψ.D⁺
    A⁻   = ψ.A⁻
    B⁻   = ψ.B⁻
    C⁻   = ψ.C⁻
    D⁻   = ψ.D⁻
    E⁻   = ψ.E⁻
    F⁻   = ψ.F⁻

    ζ⁻ = min(zero(ζ), ζ)
    ζ⁺ = max(zero(ζ), ζ)
    dζ = min(ζmax, A⁺ * ζ⁺)

    # Stability parameter for _stable_ atmospheric conditions
    ψ⁺ = - B⁺ * ζ⁺ - C⁺ * (ζ⁺ - D⁺) * exp(- dζ) - C⁺ * D⁺

    # Stability parameter for _unstable_ atmospheric conditions
    f⁻₁ = sqrt(sqrt(1 - A⁻ * ζ⁻))
    ψ⁻₁ = B⁻ * log((1 + f⁻₁) / B⁻) + log((1 + f⁻₁^2) / B⁻) - B⁻ * atan(f⁻₁) + C⁻

    f⁻₂ = cbrt(1 - D⁻ * ζ⁻)
    ψ⁻₂ = E⁻ / 2 * log((1 + f⁻₂ + f⁻₂^2) / E⁻) - sqrt(E⁻) * atan( (1 + 2f⁻₂) / sqrt(E⁻)) + F⁻

    f  = ζ⁻^2 / (1 + ζ⁻^2)
    ψ⁻ = (1 - f) * ψ⁻₁ + f * ψ⁻₂

    return ifelse(ζ < 0, ψ⁻, ψ⁺)
end

"""
    EdsonScalarStabilityFunction{FT}

A struct representing the scalar stability function detailed by [edson2013exchange](@citet).
The formulation hinges on the definition of two different functions:
one for stable atmospheric conditions ``(ζ > 0)``, named ``ψ⁺`` and one for unstable conditions,
named ``ψ⁻``.

These stability functions are obtained by regression to experimental data.

The stability parameter for stable atmospheric conditions is defined as

```math
\\begin{align*}
dζ &= \\min(ζ_{\\max}, A⁺ζ) \\\\
ψ⁺ &= - (1 + B⁺ ζ)^{C⁺} - B⁺ (ζ - D⁺) \\exp( - dζ) - E⁺
\\end{align*}
```

While the stability parameter for unstable atmospheric conditions is calculated
as a function of the two individual stability functions as follows
```math
\\begin{align*}
f⁻₁ &= √(1 - A⁻ζ) \\\\
ψ⁻₁ &= B⁻ \\log[(1 + f⁻₁) / B⁻] + C⁻ \\\\
\\\\
f⁻₂ &= ∛(1 - D⁻ζ) \\\\
ψ⁻₂ &= (E⁻ / 2) \\log[(1 + f⁻₂ + f⁻₂²) / E⁻] - √E⁻ \\mathrm{atan}[(1 + 2f⁻₂) / √E⁻] + F⁻ \\\\
\\\\
f   &= ζ² / (1 + ζ²) \\\\
ψ⁻  &= (1 - f) ψ⁻₁ + f ψ⁻₂
\\end{align*}
```

The superscripts ``+`` and ``-`` indicate if the parameter applies to the
stability function for _stable_ or _unstable_ atmospheric conditions, respectively.
"""
@kwdef struct EdsonScalarStabilityFunction{FT} <: AbstractStabilityFunction
    ζmax :: FT = 50.0
    A⁺   :: FT = 0.35
    B⁺   :: FT = 2/3
    C⁺   :: FT = 3/2
    D⁺   :: FT = 14.28
    E⁺   :: FT = 8.525
    A⁻   :: FT = 15.0
    B⁻   :: FT = 2.0
    C⁻   :: FT = 0.0
    D⁻   :: FT = 34.15
    E⁻   :: FT = 3.0
    F⁻   :: FT = π / sqrt(3)
end

@inline function stability_profile(ψ::EdsonScalarStabilityFunction, ζ)
    ζmax = ψ.ζmax
    A⁺   = ψ.A⁺
    B⁺   = ψ.B⁺
    C⁺   = ψ.C⁺
    D⁺   = ψ.D⁺
    E⁺   = ψ.E⁺
    A⁻   = ψ.A⁻
    B⁻   = ψ.B⁻
    C⁻   = ψ.C⁻
    D⁻   = ψ.D⁻
    E⁻   = ψ.E⁻
    F⁻   = ψ.F⁻

    ζ⁻ = min(zero(ζ), ζ)
    ζ⁺ = max(zero(ζ), ζ)
    dζ = min(ζmax, A⁺ * ζ⁺)

    # stability function for stable atmospheric conditions
    ψ⁺ = - (1 + B⁺ * ζ⁺)^C⁺ - B⁺ * (ζ⁺ - D⁺) * exp(-dζ) - E⁺

    # Stability parameter for _unstable_ atmospheric conditions
    f⁻₁ = sqrt(1 - A⁻ * ζ⁻)
    ψ⁻₁ = B⁻ * log((1 + f⁻₁) / B⁻) + C⁻

    f⁻₂ = cbrt(1 - D⁻ * ζ⁻)
    ψ⁻₂ = E⁻ / 2 * log((1 + f⁻₂ + f⁻₂^2) / E⁻) - sqrt(E⁻) * atan((1 + 2f⁻₂) / sqrt(E⁻)) + F⁻

    f  = ζ⁻^2 / (1 + ζ⁻^2)
    ψ⁻ = (1 - f) * ψ⁻₁ + f * ψ⁻₂

    return ifelse(ζ < 0, ψ⁻, ψ⁺)
end

# Edson et al. (2013)
function atmosphere_ocean_stability_functions(FT=Oceananigans.defaults.FloatType)
    ψu = EdsonMomentumStabilityFunction{FT}()
    ψc = EdsonScalarStabilityFunction{FT}()
    return SimilarityScales(ψu, ψc, ψc)
end

Base.summary(::EdsonMomentumStabilityFunction{FT}) where FT = "EdsonMomentumStabilityFunction{$FT}"
Base.summary(::EdsonScalarStabilityFunction{FT}) where FT = "EdsonScalarStabilityFunction{$FT}"

Base.show(io, ::EdsonMomentumStabilityFunction{FT}) where FT = print(io, "EdsonMomentumStabilityFunction{$FT}")
Base.show(io, ::EdsonScalarStabilityFunction{FT}) where FT = print(io, "EdsonScalarStabilityFunction{$FT}")

#####
##### From Grachev et al. (2007), for stable boundary layers
#####

@kwdef struct ShebaMomentumStabilityFunction{FT} <: AbstractStabilityFunction
    a :: FT = 6.5
    b :: FT = 1.3
end

# @inline (ψ::ShebaMomentumStabilityFunction)(ζ) = 1 + ψ.a * ζ * cbrt(1 + ζ) / (ψ.b + ζ)
@inline function stability_profile(ψ::ShebaMomentumStabilityFunction, ζ)
    a = ψ.a
    b = ψ.b
    ζ⁺ = max(zero(ζ), ζ)
    z = cbrt(1 + ζ⁺)
    B = cbrt((1 - b) / b)

    rt3 = sqrt(3)
    Ψ₁ = - 3 * a * (z - 1) / b
    Ψ₂ = a * B / 2b * (2 * log((z + B) / (1 + B))
                       - log((z^2 - B * z + B^2) / (1 - B + B^2))
                       + 2 * rt3 * (atan((2z - B) / (rt3 * B)) - atan((2 - B) / (rt3 * B))))

    return Ψ₁ + Ψ₂
end

@kwdef struct ShebaScalarStabilityFunction{FT} <: AbstractStabilityFunction
    a :: FT = 5.0
    b :: FT = 5.0
    c :: FT = 3.0
end

@inline function stability_profile(ψ::ShebaScalarStabilityFunction, ζ)
    a = ψ.a
    b = ψ.b
    c = ψ.c
    B = sqrt(c^2 - 4)
    ζ⁺ = max(zero(ζ), ζ)

    Ψ₁ = - b/2 * log(1 + c * ζ⁺ + ζ⁺^2)
    Ψ₂ = (b * c / 2B - a / B) *
        (log((2ζ⁺ + c - B) / (2ζ⁺ + c + B)) - log((c - B) / (c + B)))

    return Ψ₁ + Ψ₂
end

#####
##### Linear stable stability function (ψ = -c ζ, bounded)
#####

"""
    LinearStableStabilityFunction{FT}

A simple linear stability function for stable conditions: ``ψ = -c ζ``,
bounded at ``|ζ| ≤ ζ_{max}``.

Used by the NCAR/Large-Yeager (2004) bulk formulae with ``c = 5`` and ``ζ_{max} = 10``.

References:
- Large, W.G. & Yeager, S.G. (2004): NCAR/TN-460+STR
"""
@kwdef struct LinearStableStabilityFunction{FT} <: AbstractStabilityFunction
    coefficient :: FT = 5.0
    maximum_stability_parameter :: FT = 10.0
end

@inline function stability_profile(ψ::LinearStableStabilityFunction, ζ)
    c = ψ.coefficient
    ζmax = ψ.maximum_stability_parameter
    ζ⁺ = max(zero(ζ), ζ)
    return -c * min(ζ⁺, ζmax)
end

Base.summary(::LinearStableStabilityFunction{FT}) where FT = "LinearStableStabilityFunction{$FT}"
Base.show(io::IO, ::LinearStableStabilityFunction{FT}) where FT = print(io, "LinearStableStabilityFunction{$FT}")

#####
##### From Paulson (1970), for unstable boundary layers
#####

@kwdef struct PaulsonMomentumStabilityFunction{FT} <: AbstractStabilityFunction
    a :: FT = 16.0
    b :: FT = π/2
end

@inline function stability_profile(ψ::PaulsonMomentumStabilityFunction, ζ)
    a = ψ.a
    b = ψ.b
    ζ⁻ = min(zero(ζ), ζ)
    z = sqrt(sqrt((1 - a * ζ⁻)))

    Ψ₁ = 2 * log((1 + z) / 2)
    Ψ₂ = log((1 + z^2) / 2)
    Ψ₃ = - 2 * atan(z)

    return Ψ₁ + Ψ₂ + Ψ₃ + b
end

@kwdef struct PaulsonScalarStabilityFunction{FT} <: AbstractStabilityFunction
    a :: FT = 16.0
end

@inline function stability_profile(ψ::PaulsonScalarStabilityFunction, ζ)
    a = ψ.a
    ζ⁻ = min(zero(ζ), ζ)
    z = sqrt(sqrt((1 - a * ζ⁻)))
    return 2 * log((1 + z^2) / 2)
end

struct SplitStabilityFunction{S, U}
    stable :: S
    unstable :: U
end

Base.summary(ss::SplitStabilityFunction) = "SplitStabilityFunction"
Base.show(io::IO, ss::SplitStabilityFunction) = print(io, "SplitStabilityFunction")

@inline function stability_profile(ψ::SplitStabilityFunction, ζ)
    Ψ_stable = stability_profile(ψ.stable, ζ)
    Ψ_unstable = stability_profile(ψ.unstable, ζ)
    stable = ζ > 0
    return ifelse(stable, Ψ_stable, Ψ_unstable)
end

function atmosphere_sea_ice_stability_functions(FT=Oceananigans.defaults.FloatType)
    unstable_momentum = PaulsonMomentumStabilityFunction{FT}()
    stable_momentum = ShebaMomentumStabilityFunction{FT}()
    momentum = SplitStabilityFunction(stable_momentum, unstable_momentum)

    unstable_scalar = PaulsonScalarStabilityFunction{FT}()
    stable_scalar = ShebaScalarStabilityFunction{FT}()
    scalar = SplitStabilityFunction(stable_scalar, unstable_scalar)

    return SimilarityScales(momentum, scalar, scalar)
end

function atmosphere_sea_ice_similarity_theory(FT=Oceananigans.defaults.FloatType)
    stability_functions = atmosphere_sea_ice_stability_functions(FT)
    return SimilarityTheoryFluxes(FT; stability_functions)
end

"""
    ncar_stability_functions(FT = Float64)

NCAR/Large-Yeager (2004) stability functions combining:
- Unstable: Paulson (1970) with γ = 16
- Stable: linear ψ = -5ζ, bounded at |ζ| ≤ 10

Used for OMIP-2 protocol compliance.
"""
function ncar_stability_functions(FT=Oceananigans.defaults.FloatType)
    stable   = LinearStableStabilityFunction{FT}()
    momentum = SplitStabilityFunction(stable, PaulsonMomentumStabilityFunction{FT}())
    scalar   = SplitStabilityFunction(stable, PaulsonScalarStabilityFunction{FT}())
    return SimilarityScales(momentum, scalar, scalar)
end
