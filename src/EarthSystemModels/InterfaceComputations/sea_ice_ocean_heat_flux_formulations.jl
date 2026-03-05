using ClimaSeaIce.SeaIceThermodynamics: melting_temperature, LinearLiquidus, ConductiveFlux
using Adapt

#####
##### Ice Bath Heat Flux (bulk formulation)
#####

"""
    IceBathHeatFlux{FT, U}

Bulk formulation for sea ice-ocean heat flux.

The interface temperature is fixed at the freezing point of the surface salinity,
and the heat flux is computed using bulk transfer:
```math
Q = \\rho_o c_o \\alpha_h u_* (T - T_m)
```
where ``\\alpha_h`` is the heat transfer coefficient and ``u_*`` is the friction velocity.

Fields
======

- `heat_transfer_coefficient::FT`: turbulent heat exchange coefficient ``\\alpha_h`` (dimensionless)
- `friction_velocity::U`: friction velocity value or formulation (constant `Number` or `MomentumBasedFrictionVelocity`)

Example
=======

```jldoctest
using NumericalEarth.EarthSystemModels: IceBathHeatFlux

flux = IceBathHeatFlux(heat_transfer_coefficient = 0.006, friction_velocity = 0.002)

# output
IceBathHeatFlux{Float64}
├── heat_transfer_coefficient: 0.006
└── friction_velocity: 0.002
```

References
==========

- [holland1999modeling](@citet): Holland, D. M., & Jenkins, A. (1999). Modeling thermodynamic ice–ocean interactions
  at the base of an ice shelf. *Journal of Physical Oceanography*, 29(8), 1787-1800.
"""
struct IceBathHeatFlux{FT, U}
    heat_transfer_coefficient :: FT
    friction_velocity :: U
end

"""
    IceBathHeatFlux(FT::DataType = Oceananigans.defaults.FloatType;
                    heat_transfer_coefficient = 0.006,
                    friction_velocity = 0.02)

Construct an `IceBathHeatFlux` with the specified parameters.

Keyword Arguments
=================

- `heat_transfer_coefficient`: turbulent heat exchange coefficient. Default: 0.006.
- `friction_velocity`: friction velocity value or formulation. Default: 0.02.
"""
function IceBathHeatFlux(FT::DataType = Oceananigans.defaults.FloatType;
                         heat_transfer_coefficient = convert(FT, 0.006),
                         friction_velocity = convert(FT, 0.02))
    return IceBathHeatFlux(convert(FT, heat_transfer_coefficient), friction_velocity)
end

#####
##### Three-Equation Heat Flux (full formulation)
#####

"""
    ThreeEquationHeatFlux{FT, U}

Three-equation formulation for sea ice-ocean heat flux.

This formulation solves a coupled system for the interface temperature and salinity:
1. Heat balance: ``\\rho c_p \\gamma_T (T - T_b) = ℰ q``
2. Salt balance: ``\\gamma_S (S - S_b) = q (S_b - S_i)``
3. Freezing point: ``T_b = T_m(S_b)``

where ``T_b`` and ``S_b`` are the interface temperature and salinity,
``\\gamma_T = \\alpha_h u_*`` and ``\\gamma_S = \\alpha_s u_*`` are turbulent exchange velocities,
``L`` is the latent heat of fusion, and ``q`` is the melt rate (computed, not input).

Fields
======

- `heat_transfer_coefficient::FT`: turbulent heat exchange coefficient ``\\alpha_h`` (dimensionless)
- `salt_transfer_coefficient::FT`: turbulent salt exchange coefficient ``\\alpha_s`` (dimensionless)
- `internal_heat_flux::FT`: diffusive flux inside the sea ice (`ConductiveFlux`)
- `friction_velocity::U`: friction velocity value or formulation (constant `Number` or `MomentumBasedFrictionVelocity`)

Example
=======

```jldoctest
using NumericalEarth.EarthSystemModels: ThreeEquationHeatFlux

flux = ThreeEquationHeatFlux()

# output
ThreeEquationHeatFlux{Nothing}
├── heat_transfer_coefficient: 0.0095
├── salt_transfer_coefficient: 0.00027142857142857144
└── friction_velocity: 0.002
```

References
==========

- [holland1999modeling](@citet): Holland, D. M., & Jenkins, A. (1999). Modeling thermodynamic ice–ocean interactions
  at the base of an ice shelf. *Journal of Physical Oceanography*, 29(8), 1787-1800.
- [hieronymus2021comparison](@citet): Hieronymus, M., et al. (2021). A comparison of ocean-ice flux parametrizations.
  *Geosci. Model Dev.*, 14, 4891-4908.
"""
struct ThreeEquationHeatFlux{F, T, FT, U}
    conductive_flux :: F
    internal_temperature :: T
    heat_transfer_coefficient :: FT
    salt_transfer_coefficient :: FT
    friction_velocity :: U
end

Adapt.adapt_structure(to, f::ThreeEquationHeatFlux) = 
    ThreeEquationHeatFlux(Adapt.adapt(to, f.conductive_flux),
                          Adapt.adapt(to, f.internal_temperature),
                          f.heat_transfer_coefficient,
                          f.salt_transfer_coefficient,
                          Adapt.adapt(to, f.friction_velocity))

"""
    ThreeEquationHeatFlux(FT::DataType = Oceananigans.defaults.FloatType;
                          heat_transfer_coefficient = 0.0095,
                          salt_transfer_coefficient = heat_transfer_coefficient / 35,
                          friction_velocity = 0.002)

Construct a `ThreeEquationHeatFlux` with the specified parameters.

Default values follow [hieronymus2021comparison](@citet) with ``R = \\alpha_h / \\alpha_s = 35``.

Keyword Arguments
=================

- `heat_transfer_coefficient`: turbulent heat exchange coefficient ``\\alpha_h``. Default: 0.0095.
- `salt_transfer_coefficient`: turbulent salt exchange coefficient ``\\alpha_s``. Default: ``\\alpha_h / 35 \\approx 0.000271``.
- `friction_velocity`: friction velocity value or formulation. Default: 0.002.
"""
function ThreeEquationHeatFlux(FT::DataType = Oceananigans.defaults.FloatType;
                               heat_transfer_coefficient = 0.0095,
                               salt_transfer_coefficient = heat_transfer_coefficient / 35,
                               friction_velocity = convert(FT, 0.002))
    return ThreeEquationHeatFlux(nothing,
                                 nothing,
                                 convert(FT, heat_transfer_coefficient),
                                 convert(FT, salt_transfer_coefficient),
                                 friction_velocity)
end

# Constructor that accepts the sea-ice model
ThreeEquationHeatFlux(::Nothing, FT::DataType = Oceananigans.defaults.FloatType; kwargs...) = ThreeEquationHeatFlux(FT; kwargs...)

#####
##### Interface heat flux computation
#####

"""
    compute_interface_heat_flux(flux::IceBathHeatFlux, ocean_state, ice_state, liquidus, ocean_properties, ℰ, u★)

Compute the heat flux and melt rate at the sea ice-ocean interface using bulk formulation.
Returns `(Q, q, Tᵦ, Sᵦ)` where:
- `Q > 0` means heat flux from ocean to ice (ocean cooling)
- `q > 0` means melting (ice volume loss)
- `Tᵦ, Sᵦ` are the interface temperature and salinity
"""
@inline function compute_interface_heat_flux(flux::IceBathHeatFlux,
                                             ocean_state, ice_state,
                                             liquidus, ocean_properties, ℰ, u★)
    Tᵒᶜ = ocean_state.T
    Sᵒᶜ = ocean_state.S
    ℵ  = ice_state.ℵ

    ρᵒᶜ = ocean_properties.reference_density
    cᵒᶜ = ocean_properties.heat_capacity
    αₕ = flux.heat_transfer_coefficient

    # Interface temperature is at the freezing point of ocean surface salinity
    Tₘ = melting_temperature(liquidus, Sᵒᶜ)

    # Heat flux: Q > 0 means heat flux from ocean to ice (ocean cooling)
    Qᵢₒ = ρᵒᶜ * cᵒᶜ * αₕ * u★ * (Tᵒᶜ - Tₘ) * ℵ

    # Melt rate: q = Q / L (positive for melting)
    q = Qᵢₒ / ℰ

    # For IceBathHeatFlux, interface is at ocean surface values
    return Qᵢₒ, q, Tₘ, Sᵒᶜ
end

const NoInternalFluxTEF{FT} = ThreeEquationHeatFlux{<:Nothing, <:Nothing, FT} where FT
const ConductiveFluxTEF{FT} = ThreeEquationHeatFlux{<:ConductiveFlux, <:AbstractField, FT} where FT

# Helper for internal temperature extraction (used in kernel)
@inline extract_internal_temperature(::NoInternalFluxTEF{FT}, i, j) where FT = zero(FT)
@inline extract_internal_temperature(::IceBathHeatFlux{FT},   i, j) where FT = zero(FT)
@inline extract_internal_temperature(flux::ConductiveFluxTEF, i, j) = @inbounds flux.internal_temperature[i, j, 1]

"""
    compute_interface_heat_flux(flux::ThreeEquationHeatFlux, ocean_state, ice_state, liquidus, ocean_properties, ℰ, u★)

Compute the heat flux and melt rate at the sea ice-ocean interface using three-equation formulation.
Dispatches to the appropriate `solve_interface_conditions` based on whether the flux has internal
conductive flux or not.

Returns `(Q, q, Tᵦ, Sᵦ)` where:
- `Q > 0` means heat flux from ocean to ice (ocean cooling)
- `q > 0` means melting (ice volume loss)
- `Tᵦ, Sᵦ` are the interface temperature and salinity
"""
@inline function compute_interface_heat_flux(flux::ThreeEquationHeatFlux,
                                             ocean_state, ice_state,
                                             liquidus, ocean_properties, ℰ, u★)
    # Unpack states
    Tᵒᶜ = ocean_state.T
    Sᵒᶜ = ocean_state.S
    ℵ  = ice_state.ℵ

    ρᵒᶜ = ocean_properties.reference_density
    cᵒᶜ = ocean_properties.heat_capacity

    # Get transfer coefficients
    αₕ = flux.heat_transfer_coefficient
    αₛ = flux.salt_transfer_coefficient

    # Solve interface conditions - dispatch on flux type via ice_state
    T★, S★, q = solve_interface_conditions(flux, Tᵒᶜ, Sᵒᶜ, ice_state, αₕ, αₛ, u★, ℰ, ρᵒᶜ, cᵒᶜ, liquidus)

    # Scale by ice concentration
    q = q * ℵ
    Qᵢₒ = ℰ * q

    return Qᵢₒ, q, T★, S★
end

# Helper to get conductive flux parameters (κ, Tˢⁱ) - dispatches on flux type
@inline conductive_flux_parameters(::NoInternalFluxTEF, ice_state, ℰ) = (zero(ℰ), zero(ℰ))

@inline function conductive_flux_parameters(flux::ConductiveFluxTEF, ice_state, ℰ)
    h  = ice_state.h
    hc = ice_state.hc
    Tˢⁱ = ice_state.T
    k  = flux.conductive_flux.conductivity
    # Set κ to zero when h < hc (ice not consolidated)
    consolidated = h ≥ hc
    κ = ifelse(consolidated, k / (h * ℰ), zero(h))
    return κ, Tˢⁱ
end

"""
    solve_interface_conditions(flux::ThreeEquationHeatFlux, Tᵒᶜ, Sᵒᶜ, ice_state, αₕ, αₛ, u★, ℰ, ρᵒᶜ, cᵒᶜ, liquidus)

Solve the three-equation system for interface temperature, salinity, and melt rate.

The three equations are:
1. Heat balance: ``ρᵒᶜ cᵒᶜ αₕ u★ (Tᵒᶜ - T★) + κ (Tˢⁱ - T★) = ℰ q``
2. Salt balance: ``ρᵒᶜ αₛ u★ (Sᵒᶜ - S★) = q (S★ - Sˢⁱ)``
3. Freezing point: ``T★ = Tₘ(S★)``

where `κ = k/(h ℰ)` is the conductive heat transfer coefficient (zero for `NoInternalFluxTEF`).

Arguments
=========
- `ice_state`: NamedTuple with fields `S`, `h`, `hc`, `ℵ`, `T` (internal temperature)

Returns `(T★, S★, q)` where q is the melt rate (positive for melting).
"""
@inline function solve_interface_conditions(flux::ThreeEquationHeatFlux, Tᵒᶜ, Sᵒᶜ, ice_state,
                                            αₕ, αₛ, u★, ℰ, ρᵒᶜ, cᵒᶜ, liquidus::LinearLiquidus)
    Sˢⁱ = ice_state.S

    # Get conductive flux parameters - dispatches on flux type
    κ, Tˢⁱ = conductive_flux_parameters(flux, ice_state, ℰ)

    λ₁ = -liquidus.slope
    λ₂ = liquidus.freshwater_melting_temperature

    # Transfer coefficients
    η = ρᵒᶜ * cᵒᶜ * αₕ * u★ / ℰ  # turbulent heat
    γ = ρᵒᶜ * αₛ * u★           # turbulent salt
    θ = η + κ                  # total heat

    # Quadratic coefficients: a S★² + b S★ + c = 0
    a = θ * λ₁
    b = -γ - η * Tᵒᶜ - κ * Tˢⁱ + θ * (λ₂ - λ₁ * Sˢⁱ)
    c = γ * Sᵒᶜ + (η * Tᵒᶜ + κ * Tˢⁱ - θ * λ₂) * Sˢⁱ

    # Solve quadratic with zero-safe reciprocal (MITgcm approach)
    ξ = ifelse(a == zero(a), zero(a), one(a) / (2a))
    Δ = max(b^2 - 4a * c, zero(a))
    S★ = (-b - sqrt(Δ)) * ξ
    S★ = ifelse(S★ < zero(S★), (-b + sqrt(Δ)) * ξ, S★)

    # Interface temperature from liquidus
    T★ = melting_temperature(liquidus, S★)

    # Melt rate from heat balance
    q = η * (Tᵒᶜ - T★) + κ * (Tˢⁱ - T★)

    return T★, S★, q
end

#####
##### Show methods
#####

Base.summary(::IceBathHeatFlux{FT}) where FT = "IceBathHeatFlux{$FT}"
Base.summary(::ThreeEquationHeatFlux{FT}) where FT = "ThreeEquationHeatFlux{$FT}"

function Base.show(io::IO, flux::IceBathHeatFlux)
    print(io, summary(flux), '\n')
    print(io, "├── heat_transfer_coefficient: ", flux.heat_transfer_coefficient, '\n')
    print(io, "└── friction_velocity: ", flux.friction_velocity)
end

function Base.show(io::IO, flux::ThreeEquationHeatFlux)
    print(io, summary(flux), '\n')
    print(io, "├── heat_transfer_coefficient: ", flux.heat_transfer_coefficient, '\n')
    print(io, "├── salt_transfer_coefficient: ", flux.salt_transfer_coefficient, '\n')
    print(io, "└── friction_velocity: ", flux.friction_velocity)
end
