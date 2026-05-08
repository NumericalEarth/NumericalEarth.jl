struct MomentumRoughnessLength{FT, G, V}
    gravitational_acceleration :: FT
    air_kinematic_viscosity :: V
    wave_formulation :: G
    smooth_wall_parameter :: FT
    maximum_roughness_length :: FT
end

Base.summary(::MomentumRoughnessLength{FT}) where FT = "MomentumRoughnessLength{$FT}"
Base.show(io::IO, ::MomentumRoughnessLength{FT}) where FT = print(io, "MomentumRoughnessLength{$FT}")

struct ScalarRoughnessLength{FT, V, R}
    air_kinematic_viscosity :: V
    reynolds_number_scaling_function :: R
    maximum_roughness_length :: FT
end

Base.summary(::ScalarRoughnessLength{FT}) where FT = "ScalarRoughnessLength{$FT}"
Base.show(io::IO, ::ScalarRoughnessLength{FT}) where FT = print(io, "ScalarRoughnessLength{$FT}")

struct LandRoughnessLength{FT}
    multiplier :: FT
    minimum_roughness_length :: FT
end

"""
    LandRoughnessLength(FT = Float64; multiplier = 1, minimum_roughness_length = eps(FT))

Use the local land aerodynamic roughness field `znt` as a MOST roughness
length. `multiplier` scales `znt`, for example `0.1` for scalar roughness
lengths when heat and moisture roughness are taken as `znt / 10`.
"""
function LandRoughnessLength(FT=Oceananigans.defaults.FloatType;
                             multiplier = 1,
                             minimum_roughness_length = eps(FT))
    return LandRoughnessLength(convert(FT, multiplier),
                               convert(FT, minimum_roughness_length))
end

Base.summary(::LandRoughnessLength{FT}) where FT = "LandRoughnessLength{$FT}"
Base.show(io::IO, ::LandRoughnessLength{FT}) where FT = print(io, "LandRoughnessLength{$FT}")

struct WindDependentWaveFormulation{FT}
    Umax :: FT
    ℂ₁ :: FT
    ℂ₂ :: FT
end

"""
    WindDependentWaveFormulation(FT = Oceananigans.defaults.FloatType;
                                 Umax = 19, ℂ₁ = 0.0017, ℂ₂ = -0.005)

A gravity wave parameter based on the wind speed `ΔU` with the formula `ℂ₁ * max(ΔU, Umax) + ℂ₂`
as shown in (Edson (2013)'s)[@cite Edson2013] equation (13) and surrounding text.
"""
WindDependentWaveFormulation(FT=Oceananigans.defaults.FloatType; Umax = 19, ℂ₁ = 0.0017, ℂ₂ = -0.005) =
    WindDependentWaveFormulation(convert(FT, Umax),
                                 convert(FT, ℂ₁),
                                 convert(FT, ℂ₂))

gravity_wave_parameter(α::Number, args...) = α
gravity_wave_parameter(α::WindDependentWaveFormulation, ΔU) = max(zero(ΔU), α.ℂ₁ * min(ΔU, α.Umax) + α.ℂ₂)

"""
    ScalarRoughnessLength(FT = Float64;
                          air_kinematic_viscosity = temperature_dependent_viscosity,
                          reynolds_number_scaling_function = empirical_scaling_function,
                          maximum_roughness_length = 1.6e-4)

Construct a `ScalarRoughnessLength` object that represents the scalar roughness length
that regulates the exchange of heat and water vapor between the ocean and the atmosphere.

Keyword Arguments
=================

- `air_kinematic_viscosity::Function`: The function to compute the air kinematic viscosity.
- `reynolds_number_scaling_function::Function`: The function to compute the Reynolds number scaling factor.
- `maximum_roughness_length::Float`: The maximum roughness length value. Defaults to `1.6e-4`.
"""
function ScalarRoughnessLength(FT=Oceananigans.defaults.FloatType;
                               air_kinematic_viscosity = 1.5e-5,
                               reynolds_number_scaling_function = ReynoldsScalingFunction(FT),
                               maximum_roughness_length = 1.6e-4) # Values from COARE3.6

    return ScalarRoughnessLength(air_kinematic_viscosity,
                                 reynolds_number_scaling_function,
                                 convert(FT, maximum_roughness_length))
end

"""
    MomentumRoughnessLength(FT = Float64;
                            gravitational_acceleration = default_gravitational_acceleration,
                            maximum_roughness_length = 1.0,
                            air_kinematic_viscosity = 1.5e-5,
                            wave_formulation = 0.011,
                            smooth_wall_parameter = 0.11)

Construct a `MomentumRoughnessLength` object that represents the momentum roughness length that
regulates the exchange of momentum, heat, and water vapor between the ocean and the atmosphere.

Keyword Arguments
=================

- `gravitational_acceleration`: The gravitational acceleration. Default: `default_gravitational_acceleration`.
- `maximum_roughness_length`: The maximum roughness length. Default: 1e-1.
- `air_kinematic_viscosity`: The air kinematic viscosity. Default: 1.5e-5.
- `wave_formulation`: The gravity wave parameter. Default: 0.011.
- `smooth_wall_parameter`: The smooth_wall_parameter parameter. Default: 0.11.
"""
function MomentumRoughnessLength(FT=Oceananigans.defaults.FloatType;
                                 gravitational_acceleration = default_gravitational_acceleration,
                                 maximum_roughness_length = 1,
                                 air_kinematic_viscosity = 1.5e-5,
                                 wave_formulation = 0.02,
                                 smooth_wall_parameter = 0.11)

    if wave_formulation isa Number
        wave_formulation = convert(FT, wave_formulation)
    end

    return MomentumRoughnessLength(convert(FT, gravitational_acceleration),
                                   air_kinematic_viscosity,
                                   wave_formulation,
                                   convert(FT, smooth_wall_parameter),
                                   convert(FT, maximum_roughness_length))
end

function default_roughness_lengths(FT=Oceananigans.defaults.FloatType)
    momentum    = MomentumRoughnessLength(FT)
    temperature = ScalarRoughnessLength(FT)
    water_vapor = ScalarRoughnessLength(FT)
    return SimilarityScales(momentum, temperature, water_vapor)
end

# Temperature-dependent viscosity law
struct TemperatureDependentAirViscosity{FT}
    ℂ₀ :: FT
    ℂ₁ :: FT
    ℂ₂ :: FT
    ℂ₃ :: FT
end

"""
    TemperatureDependentAirViscosity([FT = Oceananigans.defaults.FloatType;
                                      ℂ₀ = 1.326e-5,
                                      ℂ₁ = ℂ₀ * 6.542e-3,
                                      ℂ₂ = ℂ₀ * 8.301e-6,
                                      ℂ₃ = - ℂ₀ * 4.84e-9])

Construct a `TemperatureDependentAirViscosity` object that calculates the kinematic
viscosity of air as

```math
ℂ₀ + ℂ₁ T + ℂ₂ T^2 + ℂ₃ T^3
```
"""
function TemperatureDependentAirViscosity(FT = Oceananigans.defaults.FloatType;
                                          ℂ₀ = 1.326e-5,
                                          ℂ₁ = ℂ₀ * 6.542e-3,
                                          ℂ₂ = ℂ₀ * 8.301e-6,
                                          ℂ₃ = - ℂ₀ * 4.84e-9)

    return TemperatureDependentAirViscosity(convert(FT, ℂ₀),
                                            convert(FT, ℂ₁),
                                            convert(FT, ℂ₂),
                                            convert(FT, ℂ₃))
end

@inline compute_air_kinematic_viscosity(ν::Number, ℂ, T) = ν

""" Calculate the air viscosity based on the temperature θ in Celsius. """
@inline function compute_air_kinematic_viscosity(ν::TemperatureDependentAirViscosity, ℂ, T)
    FT = eltype(ν.ℂ₀)
    T′ = convert(FT, T - celsius_to_kelvin)
    return ν.ℂ₀ + ν.ℂ₁ * T′ + ν.ℂ₂ * T′^2 + ν.ℂ₃ * T′^3
end

# Fallbacks for constant roughness length
@inline roughness_length(ℓ, u★, args...) = ℓ(u★, args...)
@inline roughness_length(ℓ::Number, args...) = ℓ

# Momentum roughness length should be different from scalar roughness length.
# Temperature and water vapor can be considered the same (Edson et al. 2013)
@inline function roughness_length(ℓ::MomentumRoughnessLength{FT}, u★, Uᵃᵗ, ℂᵃᵗ=nothing, Tᵃᵗ=nothing) where FT
    ν = compute_air_kinematic_viscosity(ℓ.air_kinematic_viscosity, ℂᵃᵗ, Tᵃᵗ)
    g = ℓ.gravitational_acceleration
    ℂg = gravity_wave_parameter(ℓ.wave_formulation, Uᵃᵗ)
    ℂν = ℓ.smooth_wall_parameter

    ℓᵂ = ℂg * u★^2 / g # gravity wave roughness length
    ℓᴿ = ifelse(ℂν == 0, zero(u★), ℂν * ν / u★) # viscous sublayer roughness length
    ℓ★ = ℓᵂ + ℓᴿ # arbitrary way of combining the two

    # Clip to ℓ_max, deals with u★ = 0
    ℓ_max = ℓ.maximum_roughness_length
    return min(ℓ★, ℓ_max)
end

struct ReynoldsScalingFunction{FT}
    A :: FT
    b :: FT
end

"""
    ReynoldsScalingFunction(FT = Oceananigans.defaults.FloatType; A = 5.85e-5, b = 0.72)

Empirical fit of the scalar roughness length with roughness Reynolds number `R_★ = u_★ ℓu / ν`.

```math
    ℓs = A / R_★ ^ b
```

See equation (28) by [edson2013exchange](@citet).
"""
ReynoldsScalingFunction(FT = Oceananigans.defaults.FloatType; A = 5.85e-5, b = 0.72) =
    ReynoldsScalingFunction(convert(FT, A), convert(FT, b))

@inline (s::ReynoldsScalingFunction)(R★, args...) = ifelse(R★ == 0, convert(eltype(R★), 0), s.A / R★ ^ s.b)

# Edson 2013 formulation of scalar roughness length in terms of momentum roughness length ℓu
@inline function roughness_length(ℓ::ScalarRoughnessLength{FT}, ℓu, u★, Uᵃᵗ, ℂᵃᵗ=nothing, Tᵃᵗ=nothing) where FT
    # Roughness Reynolds number
    ν = compute_air_kinematic_viscosity(ℓ.air_kinematic_viscosity, ℂᵃᵗ, Tᵃᵗ)
    R★ = ℓu * u★ / ν

    # implementation of scalar roughness length
    scaling_function = ℓ.reynolds_number_scaling_function
    ℓs = scaling_function(R★, ℓu, u★, ν)

    # Clip
    ℓ_max = ℓ.maximum_roughness_length
    return min(ℓs, ℓ_max)
end

# Convenience for users
@inline (ℓ::MomentumRoughnessLength{FT})(u★, Uᵃᵗ=nothing, ℂᵃᵗ=nothing, Tᵃᵗ=nothing) where FT =
    roughness_length(ℓ, u★, ℂᵃᵗ, Tᵃᵗ)

@inline function (ℓ::ScalarRoughnessLength{FT})(u★, Uᵃᵗ=nothing, ℂᵃᵗ=nothing, Tᵃᵗ=nothing) where FT
    roughness_length(ℓ, u★, ℂᵃᵗ, Tᵃᵗ)
end
