#####
##### Farquhar C3 photosynthesis and its temperature-response helpers.
#####
##### Net CO₂ assimilation `Aₙ` is the (smoothly) co-limited minimum of the
##### Rubisco- and light-limited rates, less dark respiration. Rate parameters are
##### given at 25 °C and scaled to leaf temperature. Grounded in ClimaLand
##### (Deck et al. 2026, JAMES, App. C): Farquhar co-limitation (Eqs C1–C5), the
##### peaked/plain Arrhenius scalings (Eqs C6, C11), and Heskel respiration (Eq C12).
#####

# Reference temperature of the photosynthesis rate parameters (the "25" suffix
# of Vcmax25, etc.) — distinct from the thermodynamic reference (triple point).
const physiology_reference_temperature = 298.15 # K (25 °C)

# Arrhenius temperature scaling `f(T) = exp[ΔH (T − T₂₅) / (T₂₅ R T)]`
# (ClimaLand Eq C6). Normalized to 1 at `T = T₂₅`.
@inline function arrhenius_scaling(T, ΔH)
    T₂₅ = oftype(T, physiology_reference_temperature)
    R   = oftype(T, default_gas_constant)
    return exp(ΔH * (T - T₂₅) / (T₂₅ * R * T))
end

# Deactivation exponent `(ΔS T − ΔHd)/(R T)` of the peaked Arrhenius form.
@inline function deactivation_exponent(T, ΔS, ΔHd)
    R = oftype(T, default_gas_constant)
    return clamp((T * ΔS - ΔHd) / (R * T), oftype(T, -80), oftype(T, 80))
end

# Peaked Arrhenius for `Vcmax`/`Jmax` (ClimaLand Eq C11): plain Arrhenius times a
# high-temperature deactivation term, so the capacity peaks near an optimum
# (low-to-mid 30s °C) and rolls off, rather than climbing without bound. The numerator
# normalizes `f′(T₂₅) = 1`, preserving the meaning of the 25 °C values. Activation
# `ΔHa`, entropy `ΔS`, deactivation `ΔHd` are the Kattge & Knorr (2007) peaked set.
@inline function peaked_arrhenius(T, ΔHa, ΔS, ΔHd)
    T₂₅  = oftype(T, physiology_reference_temperature)
    base = arrhenius_scaling(T, ΔHa)
    return base * (1 + exp(deactivation_exponent(T₂₅, ΔS, ΔHd))) /
                  (1 + exp(deactivation_exponent(T,   ΔS, ΔHd)))
end

# Heskel et al. (2016) leaf-respiration temperature response (ClimaLand Eq C12),
# normalized to 1 at 25 °C. Defined in Celsius (`Tᶜ = T − 273.15`): using Kelvin
# with these coefficients flips the sign and makes `Rd` fall with temperature.
@inline function heskel_respiration_scaling(T, b, c)
    Tᶜ   = T - oftype(T, celsius_to_kelvin)
    T₂₅ᶜ = oftype(T, 25)
    return exp(b * (Tᶜ - T₂₅ᶜ) + c * (Tᶜ^2 - T₂₅ᶜ^2))
end

"""
    PeakedArrheniusParameters(FT = Oceananigans.defaults.FloatType;
                              activation_energy, entropy, deactivation_energy)

Parameters of the peaked Arrhenius temperature response of a photosynthetic
capacity: `activation_energy` `ΔHᵃ` (J mol⁻¹), `entropy` `ΔS` (J mol⁻¹ K⁻¹),
and `deactivation_energy` `ΔHᵈ` (J mol⁻¹). The Kattge & Knorr (2007) values for
`Vcmax`/`Jmax` are the [`FarquharPhotosynthesis`](@ref) defaults.
"""
struct PeakedArrheniusParameters{FT}
    activation_energy   :: FT
    entropy             :: FT
    deactivation_energy :: FT
end

PeakedArrheniusParameters(FT::Type = Oceananigans.defaults.FloatType;
                          activation_energy, entropy, deactivation_energy) =
    PeakedArrheniusParameters(convert(FT, activation_energy),
                              convert(FT, entropy),
                              convert(FT, deactivation_energy))

Base.summary(p::PeakedArrheniusParameters) =
    string("PeakedArrheniusParameters(activation_energy=", prettysummary(p.activation_energy),
           ", entropy=", prettysummary(p.entropy),
           ", deactivation_energy=", prettysummary(p.deactivation_energy), ")")
Base.show(io::IO, p::PeakedArrheniusParameters) = print(io, summary(p))

"""
    HeskelParameters(FT = Oceananigans.defaults.FloatType;
                     slope = 0.1012, curvature = -0.0005)

Coefficients of the Heskel et al. (2016) leaf-respiration temperature response:
the linear `slope` (°C⁻¹) and quadratic `curvature` (°C⁻²) of the exponent, in
Celsius (ClimaLand Eq C12).
"""
struct HeskelParameters{FT}
    slope     :: FT
    curvature :: FT
end

HeskelParameters(FT::Type = Oceananigans.defaults.FloatType;
                 slope = 0.1012, curvature = -0.0005) =
    HeskelParameters(convert(FT, slope), convert(FT, curvature))

Base.summary(p::HeskelParameters) =
    string("HeskelParameters(slope=", prettysummary(p.slope),
           ", curvature=", prettysummary(p.curvature), ")")
Base.show(io::IO, p::HeskelParameters) = print(io, summary(p))

#####
##### Photosynthetic-capacity temperature response (trait). `PeakedArrhenius`
##### (default) rolls the capacities off above their optimum; `PlainArrhenius`
##### keeps a monotone response (deactivation disabled) for comparison.
##### `Rd` always uses the Heskel form; the trait toggles only `Vcmax`/`Jmax`.
#####

abstract type AbstractCapacityResponse end
struct PlainArrhenius  <: AbstractCapacityResponse end
struct PeakedArrhenius <: AbstractCapacityResponse end

@inline capacity_scaling(::PlainArrhenius,  T, p::PeakedArrheniusParameters) =
    arrhenius_scaling(T, p.activation_energy)
@inline capacity_scaling(::PeakedArrhenius, T, p::PeakedArrheniusParameters) =
    peaked_arrhenius(T, p.activation_energy, p.entropy, p.deactivation_energy)

# Smooth (θ-quadratic) minimum of two positive rates — the standard co-limitation
# smoothing (Collatz/Bonan): the smaller root of `θ x² − (a+b) x + a b = 0`.
# As `θ → 1` it approaches `min(a, b)` but stays differentiable. The discriminant
# is floored at zero to stay real under round-off.
@inline function smooth_minimum(a, b, θ)
    s = a + b
    discriminant = max(s^2 - 4θ * a * b, zero(s))
    return (s - sqrt(discriminant)) / (2θ)
end

"""
    struct FarquharPhotosynthesis

C3 photosynthesis after [Farquhar et al. (1980)](@cite farquhar1980): net CO₂
assimilation `Aₙ` is the (smoothly) co-limited minimum of the Rubisco-limited
rate `Aᶜ` and the light (RuBP-regeneration)-limited rate `Aⱼ`, less dark
respiration `Rᵈ`. Rate parameters are given at 25 °C and scaled to leaf
temperature: `Vcmax`/`Jmax` by the peaked Arrhenius factor (ClimaLand Eq C11) so
they peak near an optimum and roll off at high leaf temperature, `Rᵈ` by the
Heskel (2016) form (Eq C12), and `Γ*`/`Kc`/`Ko` by plain Arrhenius (Eq C6).
Defaults follow ClimaLand Table C1 / Kattge & Knorr (2007); `Vcmax25` is the
C3-grass value used for the ClimaLand US-Var flux-tower run.

Fields (all at 25 °C unless noted):
- `Vcmax25`      : maximum carboxylation rate (mol CO₂ m⁻² s⁻¹).
- `jmax_to_vcmax`: ratio `Jmax25 / Vcmax25` (–).
- `respiration_to_vcmax` : ratio `Rd25 / Vcmax25` (–).
- `quantum_yield`: electrons to PSII per absorbed photon (–).
- `Γ★25`         : CO₂ compensation point (Pa); `Kc25`, `Ko25`: Michaelis constants (Pa).
- `O₂`           : intercellular O₂ mole fraction (–).
- `θⱼ`, `θᶜⱼ`    : co-limitation smoothing for `J` and for `min(Aᶜ, Aⱼ)` (–).
- `capacity_response` : `PeakedArrhenius()` (default) or `PlainArrhenius()` — the
  `Vcmax`/`Jmax` temperature response.
- `vcmax_response`, `jmax_response` : the [`PeakedArrheniusParameters`](@ref) of each capacity.
- `respiration_response` : the [`HeskelParameters`](@ref) of the `Rᵈ` temperature response.
- `compensation_activation_energy`, `kc_activation_energy`, `ko_activation_energy` :
  plain-Arrhenius activation energies of `Γ★`, `Kc`, `Ko` (J mol⁻¹).
"""
struct FarquharPhotosynthesis{FT, K, PV, PJ, H}
    Vcmax25              :: FT
    jmax_to_vcmax        :: FT
    respiration_to_vcmax :: FT
    quantum_yield        :: FT
    Γ★25                 :: FT
    Kc25                 :: FT
    Ko25                 :: FT
    O₂                   :: FT
    θⱼ                   :: FT
    θᶜⱼ                  :: FT
    capacity_response    :: K
    vcmax_response       :: PV
    jmax_response        :: PJ
    respiration_response :: H
    compensation_activation_energy :: FT
    kc_activation_energy :: FT
    ko_activation_energy :: FT
end

function FarquharPhotosynthesis(FT=Oceananigans.defaults.FloatType;
                                Vcmax25              = 5e-5,
                                jmax_to_vcmax        = 1.67,
                                respiration_to_vcmax = 0.015,
                                quantum_yield        = 0.425,
                                Γ★25                 = 4.332,
                                Kc25                 = 39.97,
                                Ko25                 = 27480,
                                O₂                   = 0.209,
                                θⱼ                   = 0.9,
                                θᶜⱼ                  = 0.98,
                                capacity_response    = PeakedArrhenius(),
                                vcmax_response       = PeakedArrheniusParameters(FT;
                                                           activation_energy = 71513,
                                                           entropy = 649,
                                                           deactivation_energy = 200000),
                                jmax_response        = PeakedArrheniusParameters(FT;
                                                           activation_energy = 49884,
                                                           entropy = 646,
                                                           deactivation_energy = 200000),
                                respiration_response = HeskelParameters(FT),
                                compensation_activation_energy = 37830,
                                kc_activation_energy = 79430,
                                ko_activation_energy = 36380)

    return FarquharPhotosynthesis{FT, typeof(capacity_response), typeof(vcmax_response),
                                  typeof(jmax_response), typeof(respiration_response)}(
        convert(FT, Vcmax25), convert(FT, jmax_to_vcmax), convert(FT, respiration_to_vcmax),
        convert(FT, quantum_yield), convert(FT, Γ★25), convert(FT, Kc25),
        convert(FT, Ko25), convert(FT, O₂), convert(FT, θⱼ), convert(FT, θᶜⱼ),
        capacity_response, vcmax_response, jmax_response, respiration_response,
        convert(FT, compensation_activation_energy),
        convert(FT, kc_activation_energy), convert(FT, ko_activation_energy))
end

Base.summary(::FarquharPhotosynthesis{FT}) where FT = "FarquharPhotosynthesis{$FT}"
Base.show(io::IO, p::FarquharPhotosynthesis) = print(io, summary(p),
    "(Vcmax25=", prettysummary(p.Vcmax25), ")")

"""
    net_assimilation(photosynthesis, ci, APAR, Tₗ, P, β)

Net CO₂ assimilation `Aₙ` (mol CO₂ m⁻² s⁻¹) at intercellular CO₂ partial pressure
`ci` (Pa), absorbed PAR `APAR` (mol photon m⁻² s⁻¹), leaf temperature `Tₗ` (K),
air pressure `P` (Pa), and moisture-stress factor `β ∈ [0, 1]`. `β` multiplies the
photosynthetic capacities `Vcmax`, `Jmax` (Egea-type stress), so it propagates to
both `Aₙ` and — through the Medlyn coupling — the stomatal conductance.
"""
@inline function net_assimilation(p::FarquharPhotosynthesis, ci, APAR, Tₗ, P, β)
    Γ★ = p.Γ★25 * arrhenius_scaling(Tₗ, p.compensation_activation_energy) * P / oftype(P, 101325)
    Kc = p.Kc25 * arrhenius_scaling(Tₗ, p.kc_activation_energy)
    Ko = p.Ko25 * arrhenius_scaling(Tₗ, p.ko_activation_energy)
    Km = Kc * (1 + p.O₂ * P / Ko)

    Vcmax = β * p.Vcmax25 * capacity_scaling(p.capacity_response, Tₗ, p.vcmax_response)
    Jmax  = β * p.jmax_to_vcmax * p.Vcmax25 * capacity_scaling(p.capacity_response, Tₗ, p.jmax_response)
    Rd    = p.respiration_to_vcmax * p.Vcmax25 *
            heskel_respiration_scaling(Tₗ, p.respiration_response.slope, p.respiration_response.curvature)

    # Electron transport rate: smooth minimum of light supply and Jmax.
    J = smooth_minimum(p.quantum_yield * APAR, Jmax, p.θⱼ)

    Aᶜ = Vcmax * (ci - Γ★) / (ci + Km)      # Rubisco-limited
    Aⱼ = J / 4 * (ci - Γ★) / (ci + 2Γ★)     # light-limited
    Ag = smooth_minimum(Aᶜ, Aⱼ, p.θᶜⱼ)
    return Ag - Rd
end
