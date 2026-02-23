using CUDA: @allowscalar
using Printf

import ClimaSeaIce
import Thermodynamics as AtmosphericThermodynamics
using Thermodynamics: Liquid, Ice

#####
##### Interface properties
#####

struct InterfaceProperties{R, Q, T, V}
    radiation :: R
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
    # We use to compute the sensible heat flux
    Tᵃᵗ = surface_atmosphere_temperature(Ψₐ, ℙₐ)
    ΔT = Tᵃᵗ - Ψₛ.T
    Ωc = ifelse(ΔT == 0, zero(ΔT), 𝒬ᵀ / ΔT * λ) # Sensible heat transfer coefficient (W/m²K)

    # Computing the flux balance temperature
    return (Ψᵢ.T * F.κ - (Jᵀ + Ωc * Tᵃᵗ) * F.δ) / (F.κ - Ωc * F.δ)
end

# 𝒬ᵛ + ℐꜛˡʷ + Qd + Ωc * (Tᵃᵗ - Tˢ) + k / h * (Tˢ - Tˢⁱ) = 0
# where Ωc (the sensible heat transfer coefficient) is given by Ωc = 𝒬ᵀ / (Tᵃᵗ - Tˢ)
# ⟹  Tₛ = (Tˢⁱ * k - (𝒬ᵛ + ℐꜛˡʷ + Qd + Ωc * Tᵃᵗ) * h / (k - Ωc * h)
@inline function flux_balance_temperature(st::SkinTemperature{<:ClimaSeaIce.ConductiveFlux}, Ψₛ, ℙₛ, 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd, Ψᵢ, ℙᵢ, Ψₐ, ℙₐ)
    F  = st.internal_flux
    k  = F.conductivity
    h  = Ψᵢ.h
    hc = Ψᵢ.hc # Critical thickness for ice consolidation

    # Bottom temperature at the melting temperature
    Tˢⁱ = ClimaSeaIce.SeaIceThermodynamics.melting_temperature(ℙᵢ.liquidus, Ψᵢ.S)
    Tˢⁱ = convert_to_kelvin(ℙᵢ.temperature_units, Tˢⁱ)
    Tₛ⁻ = Ψₛ.T

    # Calculating the atmospheric temperature
    # We use to compute the sensible heat flux
    Tᵃᵗ = surface_atmosphere_temperature(Ψₐ, ℙₐ)
    ΔT = Tᵃᵗ - Tₛ⁻
    Ωc = ifelse(ΔT == 0, zero(h), 𝒬ᵀ / ΔT) # Sensible heat transfer coefficient (W/m²K)
    Qa = (𝒬ᵛ + ℐꜛˡʷ + Qd) # Net flux excluding sensible heat (positive out of the ocean)

    # Computing the flux balance temperature
    T★ = (Tˢⁱ * k - (Qa + Ωc * Tᵃᵗ) * h) / (k - Ωc * h)

    # Fix a NaN
    T★ = ifelse(isnan(T★), Tₛ⁻, T★)

    # To prevent instabilities in the fixed point iteration
    # solver we cap the maximum temperature difference with `max_ΔT`
    ΔT★ = T★ - Tₛ⁻
    max_ΔT = convert(typeof(T★), st.max_ΔT)
    abs_ΔT = min(max_ΔT, abs(ΔT★))
    Tₛ⁺ = Tₛ⁻ + abs_ΔT * sign(ΔT★)

    # Under heating fluxes, cap surface temperature by melting temperature
    Tₘ = ℙᵢ.liquidus.freshwater_melting_temperature
    Tₘ = convert_to_kelvin(ℙᵢ.temperature_units, Tₘ)
    Tₛ⁺ = min(Tₛ⁺, Tₘ)

    # If the ice is not consolidated, use the bottom temperature
    Tₛ⁺ = ifelse(h ≥ hc, Tₛ⁺, Tˢⁱ)
    
    return Tₛ⁺
end

@inline function compute_interface_temperature(st::SkinTemperature,
                                               interface_state,
                                               atmosphere_state,
                                               interior_state,
                                               downwelling_radiation,
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

    # upwelling radiation is calculated explicitly
    Tₛ⁻ = interface_state.T # approximate interface temperature from previous iteration
    σ = interface_properties.radiation.σ
    ϵ = interface_properties.radiation.ϵ
    α = interface_properties.radiation.α

    ℐꜜˢʷ = downwelling_radiation.ℐꜜˢʷ
    ℐꜜˡʷ = downwelling_radiation.ℐꜜˡʷ
    ℐꜛˡʷ = emitted_longwave_radiation(Tₛ⁻, σ, ϵ)
    Qd = net_absorbed_interface_radiation(ℐꜜˢʷ, ℐꜜˡʷ, α, ϵ)

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

struct InterfaceState{FT}
    u★ :: FT # friction velocity
    θ★ :: FT # flux characteristic temperature
    q★ :: FT # flux characteristic specific humidity
    u :: FT  # interface x-velocity
    v :: FT  # interface y-velocity
    T :: FT  # interface temperature
    S :: FT  # interface salinity
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
