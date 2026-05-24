using GPUArraysCore: @allowscalar
using Printf

import ClimaSeaIce
import Thermodynamics as AtmosphericThermodynamics
using Thermodynamics: Liquid, Ice

#####
##### Interface properties
#####

struct InterfaceProperties{Q, T, V}
    specific_humidity_formulation :: Q
    temperature_formulation :: T
    velocity_formulation :: V
end

#####
##### Interface specific humidity formulations
#####

# TODO: allow different saturation models
# struct ClasiusClapyeronSaturation end
struct ImpureSaturationSpecificHumidity{Φ, X}
    # saturation :: S
    phase :: Φ
    water_mole_fraction :: X
end

function Base.summary(q★::ImpureSaturationSpecificHumidity)
    phase_str = if q★.phase == AtmosphericThermodynamics.Ice()
        "Ice"
    elseif q★.phase == AtmosphericThermodynamics.Liquid()
        "Liquid"
    end


    return string("ImpureSaturationSpecificHumidity{$phase_str}(water_mole_fraction=",
                  prettysummary(q★.water_mole_fraction), ")") 
end

Base.show(io::IO, q★::ImpureSaturationSpecificHumidity) = print(io, summary(q★))

"""
    ImpureSaturationSpecificHumidity(phase [, water_mole_fraction=1])

Return the formulation for computing specific humidity at an interface.
"""
ImpureSaturationSpecificHumidity(phase) = ImpureSaturationSpecificHumidity(phase, nothing)

@inline compute_water_mole_fraction(::Nothing, salinity) = 1
@inline compute_water_mole_fraction(x_H₂O::Number, salinity) = x_H₂O

@inline function surface_specific_humidity(formulation::ImpureSaturationSpecificHumidity,
                                            ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ,
                                            Tₛ, Sₛ=zero(Tₛ))
    # Extrapolate air density to the surface temperature
    # following an adiabatic ideal gas transformation
    cvₘ = Thermodynamics.cv_m(ℂᵃᵗ, qᵃᵗ)
    Rᵃᵗ = Thermodynamics.gas_constant_air(ℂᵃᵗ, qᵃᵗ)
    κᵃᵗ = cvₘ / Rᵃᵗ # 1 / (γ - 1)
    ρᵃᵗ = Thermodynamics.air_density(ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)
    ρₛ = ρᵃᵗ * (Tₛ / Tᵃᵗ)^κᵃᵗ
    return surface_specific_humidity(formulation, ℂᵃᵗ, ρₛ, Tₛ, Sₛ)
end

@inline function surface_specific_humidity(formulation::ImpureSaturationSpecificHumidity, ℂᵃᵗ, ρₛ::Number, Tₛ, Sₛ=zero(Tₛ))
    FT = eltype(Tₛ)
    CT = eltype(ℂᵃᵗ)
    Tₛ = convert(CT, Tₛ)
    ρₛ = convert(CT, ρₛ)
    phase = formulation.phase
    p★ = Thermodynamics.saturation_vapor_pressure(ℂᵃᵗ, Tₛ, phase)
    q★ = Thermodynamics.q_vap_from_p_vap(ℂᵃᵗ, Tₛ, ρₛ, p★)

    # Compute saturation specific humidity according to Raoult's law
    χ_H₂O = compute_water_mole_fraction(formulation.water_mole_fraction, Sₛ)
    qₛ = χ_H₂O * q★

    return convert(FT, qₛ)
end

# A β-reduced saturation specific humidity for land surfaces:
# qₛ = qₐ + β · (surface saturation specific humidity - qₐ), where β ∈ [0, 1] is the moisture
# availability exposed by the land's `surface_wetness`. The β is threaded through
# the existing iteration pipeline by hijacking the `S` slot of `InterfaceState`,
# so no plumbing changes are needed downstream of the fixed-point solver.
struct BetaSurfaceSpecificHumidity{Φ}
    phase :: Φ
end

Base.summary(::BetaSurfaceSpecificHumidity{Φ}) where Φ =
    string("BetaSurfaceSpecificHumidity{",
           Φ === AtmosphericThermodynamics.Liquid ? "Liquid" : "Ice", "}")
Base.show(io::IO, q::BetaSurfaceSpecificHumidity) = print(io, summary(q))

@inline function surface_specific_humidity(formulation::BetaSurfaceSpecificHumidity,
                                           ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ,
                                           Tₛ, β=one(Tₛ))
    cvₘ = Thermodynamics.cv_m(ℂᵃᵗ, qᵃᵗ)
    Rᵃᵗ = Thermodynamics.gas_constant_air(ℂᵃᵗ, qᵃᵗ)
    κᵃᵗ = cvₘ / Rᵃᵗ
    ρᵃᵗ = Thermodynamics.air_density(ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)
    ρₛ = ρᵃᵗ * (Tₛ / Tᵃᵗ)^κᵃᵗ
    return surface_specific_humidity(formulation, ℂᵃᵗ, ρₛ, Tₛ, β, qᵃᵗ)
end

@inline function surface_specific_humidity(formulation::BetaSurfaceSpecificHumidity,
                                           ℂᵃᵗ, ρₛ::Number, Tₛ, β=one(Tₛ), qₐ=zero(Tₛ))
    FT = eltype(Tₛ)
    CT = eltype(ℂᵃᵗ)
    Tₛ = convert(CT, Tₛ)
    ρₛ = convert(CT, ρₛ)
    p★ = Thermodynamics.saturation_vapor_pressure(ℂᵃᵗ, Tₛ, formulation.phase)
    q★ = Thermodynamics.q_vap_from_p_vap(ℂᵃᵗ, Tₛ, ρₛ, p★)
    qₐ = convert(FT, qₐ)
    return convert(FT, qₐ + β * (q★ - qₐ))
end

struct SalinityConstituent{FT}
    molar_mass :: FT
    mass_fraction :: FT
end

struct WaterMoleFraction{FT, C}
    water_molar_mass :: FT
    salinity_constituents :: C
end

function WaterMoleFraction(FT=Oceananigans.defaults.FloatType)
    water_molar_mass = convert(FT, 18.02)

    # TODO: find reference for these
    salinity_constituents = (
        chloride  = SalinityConstituent{FT}(35.45, 0.56),
        sodium    = SalinityConstituent{FT}(22.99, 0.31),
        sulfate   = SalinityConstituent{FT}(96.06, 0.08),
        magnesium = SalinityConstituent{FT}(24.31, 0.05),
    )

    return WaterMoleFraction(water_molar_mass, salinity_constituents)
end

@inline function compute_water_mole_fraction(wmf::WaterMoleFraction, S)
    # TODO: express the concept of "ocean_salinity_units"?
    s = S / 1000 # convert g/kg to concentration

    # Molecular weights
    μ_H₂O = wmf.water_molar_mass

    # Salinity constituents: Cl⁻, Na, SO₄, Mg
    μ_Cl  = wmf.salinity_constituents.chloride.molar_mass
    μ_Na  = wmf.salinity_constituents.sodium.molar_mass
    μ_SO₄ = wmf.salinity_constituents.sulfate.molar_mass
    μ_Mg  = wmf.salinity_constituents.magnesium.molar_mass

    # Salinity constituent fractions
    ϵ_Cl  = wmf.salinity_constituents.chloride.mass_fraction
    ϵ_Na  = wmf.salinity_constituents.sodium.mass_fraction
    ϵ_SO₄ = wmf.salinity_constituents.sulfate.mass_fraction
    ϵ_Mg  = wmf.salinity_constituents.magnesium.mass_fraction

    α = μ_H₂O * (ϵ_Cl/μ_Cl + ϵ_Na/μ_Na  + ϵ_SO₄/μ_SO₄ + ϵ_Mg/μ_Mg)

    return (1 - s) / (1 - s + α * s)
end

####
#### Velocity difference formulations
####

""" The exchange fluxes depend on the atmosphere velocity but not the interface velocity """
struct WindVelocity end

""" The exchange fluxes depend on the relative velocity between the atmosphere and the interface """
struct RelativeVelocity end

@inline function velocity_difference(::RelativeVelocity, 𝒰₁, 𝒰₀)
    Δu = 𝒰₁.u - 𝒰₀.u
    Δv = 𝒰₁.v - 𝒰₀.v
    return Δu, Δv
end

@inline velocity_difference(::WindVelocity, 𝒰₁, 𝒰₀) = 𝒰₁.u, 𝒰₁.v

####
#### Atmospheric temperature
####

# Temperature increment including the ``lapse rate'' `α = g / cᵖᵐ`
function surface_atmosphere_temperature(Ψₐ, ℙₐ)
    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    g  = ℙₐ.gravitational_acceleration
    Tᵃᵗ = Ψₐ.T
    qᵃᵗ = Ψₐ.q
    zᵃᵗ = Ψₐ.z
    Δh = zᵃᵗ # Assumption! The surface is at z = 0 -> Δh = zᵃᵗ - 0
    cᵃᵗ = AtmosphericThermodynamics.cp_m(ℂᵃᵗ, qᵃᵗ)
    return Tᵃᵗ + g * Δh / cᵃᵗ
end

####
#### Interface temperature formulations
####

"""
    struct BulkTemperature

A type to represent the interface temperature used in fixed-point iteration for interface
fluxes following similarity theory. The interface temperature is not calculated but instead
provided by either the ocean or the sea ice model.
"""
struct BulkTemperature end

# Do nothing (just copy the temperature)
@inline compute_interface_temperature(::BulkTemperature, Ψₛ, args...) = Ψₛ.T

####
#### Skin interface temperature calculated as a flux balance
####

"""
    struct SkinTemperature

A type to represent the interface temperature used in the flux calculation.
The interface temperature is calculated from the flux balance at the interface.
In particular, the interface temperature ``Tₛ`` is the root of:

```math
F(Tₛ) - Jᵀ = 0
```

where ``Jᵀ`` are the fluxes at the top of the interface (turbulent + radiative), and
``F`` is the internal diffusive flux dependent on the interface temperature itself.

Note that all fluxes positive upwards.
"""
struct SkinTemperature{I, FT}
    internal_flux :: I
    max_ΔT :: FT
end

SkinTemperature(internal_flux; max_ΔT=5) = SkinTemperature(internal_flux, max_ΔT)

struct DiffusiveFlux{Z, K}
    δ :: Z # Boundary layer thickness, as a first guess we will use half the grid spacing
    κ :: K # diffusivity in m² s⁻¹
end

# The flux balance is solved by computing
#
#            κ
# Jᵃ(Tₛⁿ) + --- (Tₛⁿ⁺¹ - Tˢⁱ) = 0
#            δ
#
# where Jᵃ is the external flux impinging on the surface from above and
# Jᵢ = - κ (Tₛ - Tˢⁱ) / δ is the "internal flux" coming up from below.
# We have indicated that Jᵃ may depend on the surface temperature from the previous
# iterate. We thus find that
#
# Tₛⁿ⁺¹ = Tˢⁱ - δ * Jᵃ(Tₛⁿ) / κ
#
# Note that we could also use the fact that Jᵃ(T) = σ * ϵ * T^4 + ⋯
# to expand Jᵃ around Tⁿ⁺¹,
#
# Jᵃ(Tⁿ⁺¹) ≈ Jᵃ(Tⁿ) + (Tⁿ⁺¹ - Tⁿ) * ∂T_Jᵃ(Tⁿ)
#          ≈ Jᵃ(Tⁿ) + 4 * (Tⁿ⁺¹ - Tⁿ) σ * ϵ * Tⁿ^3 / (ρ c)
#
# which produces the alternative, semi-implicit flux balance
#
#                                      κ
# Jᵃ(Tₛⁿ) - 4 α Tₛⁿ⁴ + 4 α Tₛⁿ Tₛⁿ³ + --- (Tₛⁿ⁺¹ - Tˢⁱ) = 0
#                                      δ
#
# with α = σ ϵ / (ρ c) such that
#
# Tₛⁿ⁺¹ (κ / δ + 4 α Tₛⁿ³) = κ * Tˢⁱ / δ - Jᵃ + 4 α Tₛⁿ⁴)
#
# or
#
# Tₛⁿ⁺¹ = = (Tˢⁱ - δ / κ * (Jᵃ - 4 α Tₛⁿ⁴)) / (1 + 4 δ σ ϵ Tₛⁿ³ / ρ c κ)
#
# corresponding to a linearization of the outgoing longwave radiation term.
@inline function flux_balance_temperature(st::SkinTemperature{<:DiffusiveFlux}, Ψₛ, ℙₛ, 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd, Ψᵢ, ℙᵢ, Ψₐ, ℙₐ)
    Qa = 𝒬ᵛ + ℐꜛˡʷ + Qd # Net flux (positive out of the ocean)
    F  = st.internal_flux
    ρ  = ℙᵢ.reference_density
    c  = ℙᵢ.heat_capacity
    Qa = (𝒬ᵛ + ℐꜛˡʷ + Qd) # Net flux excluding sensible heat (positive out of the ocean)
    λ  = 1 / (ρ * c) # m³ K J⁻¹
    Jᵀ = Qa * λ

    # Calculating the atmospheric temperature
    Tᵃᵗ = surface_atmosphere_temperature(Ψₐ, ℙₐ)
    ΔT = Tᵃᵗ - Ψₛ.T

    # Flux balance: T★ = (Tᵢ κ - (Jᵀ + Ωc Tᵃᵗ) δ) / (κ - Ωc δ)
    # where Ωc = 𝒬ᵀ λ / ΔT. Multiply through by ΔT to avoid Inf when ΔT → 0.
    Ωᵀ = 𝒬ᵀ * λ  # unnormalized sensible heat coefficient (= Ωc * ΔT)
    D  = F.κ * ΔT - Ωᵀ * F.δ
    T★ = (Ψᵢ.T * F.κ * ΔT - (Jᵀ * ΔT + Ωᵀ * Tᵃᵗ) * F.δ) / D
    
    return ifelse(D == 0, Ψₛ.T, T★)
end

# Solve the surface flux balance equation:
#   Qa(Tₛ) + Ωc (Tᵃᵗ - Tₛ) + (Tₛ - Tᵦ) / R = 0
# where R is the total thermal resistance (h/k for bare ice, hₛ/kₛ + hᵢ/kᵢ with snow),
# Ωc = 𝒬ᵀ/(Tᵃᵗ-Tₛ) is the linearized sensible heat coefficient, and Qa = 𝒬ᵛ + ℐꜛˡʷ + Qd.
# The upward longwave ℐꜛˡʷ = σ ε Tₛ⁴ is strongly nonlinear in Tₛ; a pure Picard
# iteration (treating Qa constant) is unstable when 4σεTₛ³ ≳ 1/R (radiation
# dominated). We linearize: Qa(Tₛ) ≈ Qa(Tₛ⁻) + β (Tₛ − Tₛ⁻) with β = 4σεTₛ⁻³,
# yielding the Newton-like semi-implicit update:
#   Tₛ = [Tᵦ + β R Tₛ⁻ - Ωc R Tᵃᵗ - Qa R] / [1 + β R - Ωc R]
@inline function conductive_flux_balance_temperature(st, R, Ψₛ, ℙₛ, 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd, Ψᵢ, ℙᵢ, Ψₐ, ℙₐ)
    hᵢ = Ψᵢ.hi
    hc = Ψᵢ.hc

    # Bottom temperature at the melting point
    Tᵦ = ClimaSeaIce.SeaIceThermodynamics.melting_temperature(ℙᵢ.liquidus, Ψᵢ.S)
    Tᵦ = convert_to_kelvin(ℙᵢ.temperature_units, Tᵦ)
    Tₛ⁻ = Ψₛ.T

    Tᵃᵗ = surface_atmosphere_temperature(Ψₐ, ℙₐ)
    ΔT = Tᵃᵗ - Tₛ⁻
    Qa = 𝒬ᵛ + ℐꜛˡʷ + Qd

    # Sensible transfer coefficient Ωc = 𝒬ᵀ/ΔT, safely handling ΔT → 0.
    Ωc = ifelse(ΔT == zero(ΔT), zero(Tₛ⁻), 𝒬ᵀ / ΔT)

    # Newton linearization of upwelling longwave: ℐꜛˡʷ(Tₛ) ≈ ℐꜛˡʷ(Tₛ⁻) + β (Tₛ − Tₛ⁻).
    # Since ℐꜛˡʷ = σ ϵ Tₛ⁻⁴, we have β = 4 σ ϵ Tₛ⁻³ = 4 ℐꜛˡʷ / Tₛ⁻.
    β = 4 * ℐꜛˡʷ / Tₛ⁻

    # Flux balance solution with T⁴ linearization (stable even at ΔT = 0):
    D  = 1 + β * R - Ωc * R
    T★ = (Tᵦ + β * R * Tₛ⁻ - Ωc * R * Tᵃᵗ - Qa * R) / D
    T★ = ifelse(D == 0, Tₛ⁻, T★)
    T★ = ifelse(isnan(T★), Tₛ⁻, T★)

    # Cap the temperature step for iteration stability
    ΔT★ = T★ - Tₛ⁻
    max_ΔT = convert(typeof(T★), st.max_ΔT)
    Tₛ⁺ = Tₛ⁻ + clamp(ΔT★, -max_ΔT, max_ΔT)

    # Cap at melting temperature
    Tₘ = ℙᵢ.liquidus.freshwater_melting_temperature
    Tₘ = convert_to_kelvin(ℙᵢ.temperature_units, Tₘ)
    Tₛ⁺ = min(Tₛ⁺, Tₘ)

    # If ice is not consolidated, use the bottom temperature
    Tₛ⁺ = ifelse(hᵢ ≥ hc, Tₛ⁺, Tᵦ)

    return Tₛ⁺
end

# Bare ice: R = hᵢ / kᵢ
@inline function flux_balance_temperature(st::SkinTemperature{<:ClimaSeaIce.ConductiveFlux},
                                          Ψₛ, ℙₛ, 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd, Ψᵢ, ℙᵢ, Ψₐ, ℙₐ)
    k  = st.internal_flux.conductivity
    R  = Ψᵢ.hi / k
    return conductive_flux_balance_temperature(st, R, Ψₛ, ℙₛ, 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd, Ψᵢ, ℙᵢ, Ψₐ, ℙₐ)
end

# Snow + ice: R = hₛ / kₛ + hᵢ / kᵢ
@inline function flux_balance_temperature(st::SkinTemperature{<:ClimaSeaIce.SeaIceThermodynamics.IceSnowConductiveFlux},
                                          Ψₛ, ℙₛ, 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd, Ψᵢ, ℙᵢ, Ψₐ, ℙₐ)
    F  = st.internal_flux
    R  = Ψᵢ.hs / F.snow_conductivity + Ψᵢ.hi / F.ice_conductivity
    return conductive_flux_balance_temperature(st, R, Ψₛ, ℙₛ, 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd, Ψᵢ, ℙᵢ, Ψₐ, ℙₐ)
end

@inline function compute_interface_temperature(st::SkinTemperature,
                                               interface_state,
                                               atmosphere_state,
                                               interior_state,
                                               radiation_state,
                                               interface_properties,
                                               atmosphere_properties,
                                               interior_properties)

    ℂᵃᵗ = atmosphere_properties.thermodynamics_parameters
    Tᵃᵗ = atmosphere_state.T
    pᵃᵗ = atmosphere_state.p
    qᵃᵗ = atmosphere_state.q
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)
    cᵃᵗ = AtmosphericThermodynamics.cp_m(ℂᵃᵗ, qᵃᵗ) # moist heat capacity

    # TODO: this depends on the phase of the interface
    #ℰv = 0 #AtmosphericThermodynamics.latent_heat_vapor(ℂᵃᵗ, Tᵃᵗ)
    ℒⁱ = AtmosphericThermodynamics.latent_heat_sublim(ℂᵃᵗ, Tᵃᵗ)

    # upwelling radiation is calculated explicitly. radiation_state is
    # produced by `air_sea_interface_radiation_state` (or its sea-ice
    # variant) and contains zero-valued σ/α/ϵ/SW/LW when radiation is off.
    Tₛ⁻ = interface_state.T # approximate interface temperature from previous iteration
    σ = radiation_state.σ
    ϵ = radiation_state.ϵ
    α = radiation_state.α

    ℐꜜˢʷ = radiation_state.ℐꜜˢʷ
    ℐꜜˡʷ = radiation_state.ℐꜜˡʷ
    ℐꜛˡʷ = σ * ϵ * Tₛ⁻^4
    Qd = - (1 - α) * ℐꜜˢʷ - ϵ * ℐꜜˡʷ

    u★ = interface_state.u★
    θ★ = interface_state.θ★
    q★ = interface_state.q★

    # Turbulent heat fluxes, sensible + latent (positive out of the ocean)
    𝒬ᵀ = - ρᵃᵗ * cᵃᵗ * u★ * θ★ # = - ρᵃᵗ cᵃᵗ u★ Ch / sqrt(Cd) * (θᵃᵗ - Tₛ)
    𝒬ᵛ = - ρᵃᵗ * ℒⁱ * u★ * q★

    Tₛ = flux_balance_temperature(st,
                                  interface_state,
                                  interface_properties,
                                  𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd,
                                  interior_state,
                                  interior_properties,
                                  atmosphere_state,
                                  atmosphere_properties)

    return Tₛ
end

######
###### Interface state
######

"""
    InterfaceState{FT}

Interior-side state seen by the similarity-theory fixed-point solver
(`compute_interface_state`).

The `S` slot is overloaded by surface type:

* atmosphere–ocean: ocean surface salinity (used by `WaterMoleFraction`
  for `ImpureSaturationSpecificHumidity`).
* atmosphere–sea-ice: ignored (humidity is over `Ice` phase).
* atmosphere–land: moisture availability `β ∈ [0, 1]`, consumed by
  [`BetaSurfaceSpecificHumidity`](@ref) to scale the surface saturation
  humidity (`qₛ = qₐ + β·(q⁺ − qₐ)`). The land-coupling kernel writes
  `S = βₛ` in `_compute_atmosphere_land_interface_state!`.

The reuse keeps the iteration pipeline shared across surface types
without growing `InterfaceState`. Future surfaces that need an
additional scalar should add a separate field instead of re-overloading
`S`.
"""
struct InterfaceState{FT}
    u★ :: FT # friction velocity
    θ★ :: FT # flux characteristic temperature
    q★ :: FT # flux characteristic specific humidity
    u :: FT  # interface x-velocity
    v :: FT  # interface y-velocity
    T :: FT  # interface temperature
    S :: FT  # ocean: salinity; land: moisture availability β. See docstring.
    q :: FT  # interface specific humidity
    melting :: Bool
end

@inline InterfaceState(u★, θ★, q★, u, v, T, S, q) =
    InterfaceState(u★, θ★, q★, u, v, T, S, q, false)

Base.eltype(::InterfaceState{FT}) where FT = FT

function Base.show(io::IO, is::InterfaceState)
    print(io, "InterfaceState(",
          "u★=", prettysummary(is.u★), " ",
          "θ★=", prettysummary(is.θ★), " ",
          "q★=", prettysummary(is.q★), " ",
          "u=", prettysummary(is.u), " ",
          "v=", prettysummary(is.v), " ",
          "T=", prettysummary(is.T), " ",
          "S=", prettysummary(is.S), " ",
          "q=", prettysummary(is.q), ")")
end

@inline zero_interface_state(FT) = InterfaceState(zero(FT),
                                                  zero(FT),
                                                  zero(FT),
                                                  zero(FT),
                                                  zero(FT),
                                                  convert(FT, 273.15),
                                                  zero(FT),
                                                  zero(FT))
