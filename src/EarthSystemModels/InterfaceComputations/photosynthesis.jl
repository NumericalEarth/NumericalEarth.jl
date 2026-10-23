#####
##### C3 photosynthesis of the Farquhar–von Caemmerer–Berry (FvCB) model and the
##### temperature responses of its rate parameters.
#####

# Reference temperature of the photosynthesis rate parameters (the "25" suffix
# of Vcmax25, etc.) — distinct from the thermodynamic reference (triple point).
const physiology_reference_temperature = 298.15 # K (25 °C)

"""
    PeakedArrheniusParameters(FT = Oceananigans.defaults.FloatType;
                              activation_energy, entropy, deactivation_energy)

Parameters of the peaked Arrhenius temperature response of a photosynthetic
capacity: `activation_energy` `ΔHᵃ` (J mol⁻¹), `entropy` `ΔS` (J mol⁻¹ K⁻¹),
and `deactivation_energy` `ΔHᵈ` (J mol⁻¹). The Kattge & Knorr (2007) values for
`Vcmax`/`Jmax` are the [`RubiscoLightLimitedPhotosynthesis`](@ref) defaults.
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

"""
    temperature_response(T, parameters)

Temperature response of a photosynthesis rate parameter given at 25 °C: the ratio of
its value at `T` (K) to its value at `T₂₅ = 298.15 K`. An activation energy `ΔH`
(J mol⁻¹) gives the Arrhenius form `exp[ΔH (T − T₂₅) / (T₂₅ R T)]` (ClimaLand Eq C6);
[`PeakedArrheniusParameters`](@ref) add the high-temperature deactivation of `Vcmax`
and `Jmax`; [`HeskelParameters`](@ref) give the leaf-respiration response, quadratic
in the Celsius temperature.
"""
@inline function temperature_response(T, ΔH::Number)
    T₂₅ = oftype(T, physiology_reference_temperature)
    R   = oftype(T, default_gas_constant)
    return exp(ΔH * (T - T₂₅) / (T₂₅ * R * T))
end

# Deactivation exponent `(ΔS T − ΔHᵈ)/(R T)` of the peaked Arrhenius form.
@inline function deactivation_exponent(T, ΔS, ΔHᵈ)
    R = oftype(T, default_gas_constant)
    return (T * ΔS - ΔHᵈ) / (R * T)
end

@inline function temperature_response(T, p::PeakedArrheniusParameters)
    ΔS, ΔHᵈ = p.entropy, p.deactivation_energy
    T₂₅  = oftype(T, physiology_reference_temperature)
    base = temperature_response(T, p.activation_energy)
    return base * (1 + exp(deactivation_exponent(T₂₅, ΔS, ΔHᵈ))) /
                  (1 + exp(deactivation_exponent(T,   ΔS, ΔHᵈ)))
end

@inline function temperature_response(T, p::HeskelParameters)
    celsius_temperature = T - oftype(T, celsius_to_kelvin)
    T₂₅ᶜ = oftype(T, 25)
    return exp(p.slope * (celsius_temperature - T₂₅ᶜ) + p.curvature * (celsius_temperature^2 - T₂₅ᶜ^2))
end

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
    struct RubiscoLightLimitedPhotosynthesis

C3 photosynthesis of the Farquhar–von Caemmerer–Berry (FvCB) model
([Farquhar et al. (1980)](@cite farquhar1980)): net CO₂ assimilation `Aₙ` is the
(smoothly) co-limited minimum of the Rubisco-limited rate `Aᶜ` and the light
(RuBP-regeneration)-limited rate `Aⱼ`, less dark respiration `Rᵈ`. Rate parameters
are given at 25 °C and scaled to leaf temperature: `Vcmax`/`Jmax` by a peaked
Arrhenius factor, `Rᵈ` by the Heskel (2016) form, and `Γ★`/`Kc`/`Ko` by plain
Arrhenius. The peaked-Arrhenius entropies and `electron_transport_ratio` are the Kattge & Knorr
(2007) dataset means, a fixed temperate canopy; the remaining constants are ClimaLand's.

Fields (all at 25 °C unless noted):
- `Vcmax25`      : maximum carboxylation rate (mol CO₂ m⁻² s⁻¹).
- `electron_transport_ratio` : ratio `Jmax25 / Vcmax25` (–); the default 1.67 is the
  Kattge & Knorr (2007) ratio acclimated to 25 °C growth temperature.
- `respiration_ratio` : ratio `Rd25 / Vcmax25` (–).
- `quantum_yield`: electrons to PSII per absorbed photon (–).
- `Γ★25`         : photorespiratory compensation point (Pa); `Kc25`, `Ko25`: Michaelis constants (Pa).
- `O₂`           : intercellular O₂ mole fraction (–).
- `θⱼ`, `θᶜⱼ`    : co-limitation smoothing for `J` and for `min(Aᶜ, Aⱼ)` (–).
- `carboxylation_response`, `electron_transport_response` : the [`PeakedArrheniusParameters`](@ref) of each capacity.
- `respiration_response` : the [`HeskelParameters`](@ref) of the `Rᵈ` temperature response.
- `compensation_activation_energy`, `carboxylation_constant_activation_energy`,
  `oxygenation_constant_activation_energy` :
  plain-Arrhenius activation energies of `Γ★`, `Kc`, `Ko` (J mol⁻¹).
"""
struct RubiscoLightLimitedPhotosynthesis{FT, PV, PJ, H}
    Vcmax25                                  :: FT
    electron_transport_ratio                 :: FT
    respiration_ratio                        :: FT
    quantum_yield                            :: FT
    Γ★25                                     :: FT
    Kc25                                     :: FT
    Ko25                                     :: FT
    O₂                                       :: FT
    θⱼ                                       :: FT
    θᶜⱼ                                      :: FT
    carboxylation_response                   :: PV
    electron_transport_response              :: PJ
    respiration_response                     :: H
    compensation_activation_energy           :: FT
    carboxylation_constant_activation_energy :: FT
    oxygenation_constant_activation_energy   :: FT
end

function RubiscoLightLimitedPhotosynthesis(FT=Oceananigans.defaults.FloatType;
                                           Vcmax25                     = 5e-5,
                                           electron_transport_ratio    = 1.67,
                                           respiration_ratio           = 0.015,
                                           quantum_yield               = 0.425,
                                           Γ★25                        = 4.332,
                                           Kc25                        = 39.97,
                                           Ko25                        = 27480,
                                           O₂                          = 0.209,
                                           θⱼ                          = 0.9,
                                           θᶜⱼ                         = 0.98,
                                           carboxylation_response      = PeakedArrheniusParameters(FT;
                                                                             activation_energy = 71513,
                                                                             entropy = 649,
                                                                             deactivation_energy = 200000),
                                           electron_transport_response = PeakedArrheniusParameters(FT;
                                                                             activation_energy = 49884,
                                                                             entropy = 646,
                                                                             deactivation_energy = 200000),
                                           respiration_response        = HeskelParameters(FT),
                                           compensation_activation_energy           = 37830,
                                           carboxylation_constant_activation_energy = 79430,
                                           oxygenation_constant_activation_energy   = 36380)

    return RubiscoLightLimitedPhotosynthesis{FT, typeof(carboxylation_response),
                                             typeof(electron_transport_response), typeof(respiration_response)}(
        convert(FT, Vcmax25), convert(FT, electron_transport_ratio), convert(FT, respiration_ratio),
        convert(FT, quantum_yield), convert(FT, Γ★25), convert(FT, Kc25),
        convert(FT, Ko25), convert(FT, O₂), convert(FT, θⱼ), convert(FT, θᶜⱼ),
        carboxylation_response, electron_transport_response, respiration_response,
        convert(FT, compensation_activation_energy),
        convert(FT, carboxylation_constant_activation_energy),
        convert(FT, oxygenation_constant_activation_energy))
end

Base.summary(::RubiscoLightLimitedPhotosynthesis{FT}) where FT = "RubiscoLightLimitedPhotosynthesis{$FT}"
Base.show(io::IO, p::RubiscoLightLimitedPhotosynthesis) = print(io, summary(p),
    "(Vcmax25=", prettysummary(p.Vcmax25), ")")

"""
    net_assimilation(photosynthesis, ci, APAR, Tˡᵉᵃᶠ, p, β)

Net CO₂ assimilation `Aₙ` (mol CO₂ m⁻² s⁻¹) at intercellular CO₂ partial pressure
`ci` (Pa), absorbed PAR `APAR` (mol photon m⁻² s⁻¹), leaf temperature `Tˡᵉᵃᶠ` (K),
air pressure `p` (Pa), and moisture-stress factor `β ∈ [0, 1]`. `β` multiplies the
photosynthetic capacities `Vcmax`, `Jmax` (Egea-type stress), so it propagates to
both `Aₙ` and — through the Medlyn coupling — the stomatal conductance.
"""
@inline function net_assimilation(photosynthesis::RubiscoLightLimitedPhotosynthesis, ci, APAR, Tˡᵉᵃᶠ, p, β)
    Γ★ = photosynthesis.Γ★25 * temperature_response(Tˡᵉᵃᶠ, photosynthesis.compensation_activation_energy) *
         p / oftype(p, 101325)
    Kc = photosynthesis.Kc25 * temperature_response(Tˡᵉᵃᶠ, photosynthesis.carboxylation_constant_activation_energy)
    Ko = photosynthesis.Ko25 * temperature_response(Tˡᵉᵃᶠ, photosynthesis.oxygenation_constant_activation_energy)
    Km = Kc * (1 + photosynthesis.O₂ * p / Ko)

    Vcmax = β * photosynthesis.Vcmax25 * temperature_response(Tˡᵉᵃᶠ, photosynthesis.carboxylation_response)
    Jmax  = β * photosynthesis.electron_transport_ratio * photosynthesis.Vcmax25 *
            temperature_response(Tˡᵉᵃᶠ, photosynthesis.electron_transport_response)
    Rᵈ    = photosynthesis.respiration_ratio * photosynthesis.Vcmax25 *
            temperature_response(Tˡᵉᵃᶠ, photosynthesis.respiration_response)

    # Electron transport rate: smooth minimum of light supply and Jmax.
    J = smooth_minimum(photosynthesis.quantum_yield * APAR, Jmax, photosynthesis.θⱼ)

    Aᶜ = Vcmax * (ci - Γ★) / (ci + Km)      # Rubisco-limited
    Aⱼ = J / 4 * (ci - Γ★) / (ci + 2Γ★)     # light-limited
    Ag = smooth_minimum(Aᶜ, Aⱼ, photosynthesis.θᶜⱼ)
    return Ag - Rᵈ
end
