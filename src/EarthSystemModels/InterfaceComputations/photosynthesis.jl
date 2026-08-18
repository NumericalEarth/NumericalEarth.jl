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

#####
##### Photosynthetic-capacity temperature response (trait). `PeakedArrhenius`
##### (default) rolls the capacities off above their optimum; `PlainArrhenius`
##### keeps a monotone response (deactivation disabled) for comparison.
##### `Rd` always uses the Heskel form; the trait toggles only `Vcmax`/`Jmax`.
#####

abstract type AbstractCapacityResponse end
struct PlainArrhenius  <: AbstractCapacityResponse end
struct PeakedArrhenius <: AbstractCapacityResponse end

@inline capacity_scaling(::PlainArrhenius,  T, ΔHa, ΔS, ΔHd) = arrhenius_scaling(T, ΔHa)
@inline capacity_scaling(::PeakedArrhenius, T, ΔHa, ΔS, ΔHd) = peaked_arrhenius(T, ΔHa, ΔS, ΔHd)

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
- `Jmax_to_Vcmax`: ratio `Jmax25 / Vcmax25` (–).
- `Rd_to_Vcmax`  : ratio `Rd25 / Vcmax25` (–).
- `quantum_yield`: electrons to PSII per absorbed photon (–).
- `Γstar25`         : CO₂ compensation point (Pa); `Kc25`, `Ko25`: Michaelis constants (Pa).
- `O2`           : intercellular O₂ mole fraction (–).
- `θⱼ`, `θ_colimit` : co-limitation smoothing for `J` and for `min(Aᶜ, Aⱼ)` (–).
- `capacity_response` : `PeakedArrhenius()` (default) or `PlainArrhenius()` — the
  `Vcmax`/`Jmax` temperature response.
- `ΔHa_*`, `ΔS_*`, `ΔHd_*` : peaked-Arrhenius activation (J mol⁻¹), entropy
  (J mol⁻¹ K⁻¹), and deactivation (J mol⁻¹) for `Vcmax`/`Jmax`.
- `heskel_b`, `heskel_c` : Heskel respiration coefficients (°C⁻¹, °C⁻²).
- `ΔH_Γstar`, `ΔH_Kc`, `ΔH_Ko` : plain-Arrhenius activation energies (J mol⁻¹).
"""
struct FarquharPhotosynthesis{FT, K}
    Vcmax25       :: FT
    Jmax_to_Vcmax :: FT
    Rd_to_Vcmax   :: FT
    quantum_yield :: FT
    Γstar25       :: FT
    Kc25          :: FT
    Ko25          :: FT
    O2            :: FT
    θⱼ            :: FT
    θ_colimit     :: FT
    capacity_response :: K
    ΔHa_Vcmax     :: FT
    ΔS_Vcmax      :: FT
    ΔHd_Vcmax     :: FT
    ΔHa_Jmax      :: FT
    ΔS_Jmax       :: FT
    ΔHd_Jmax      :: FT
    heskel_b      :: FT
    heskel_c      :: FT
    ΔH_Γstar      :: FT
    ΔH_Kc         :: FT
    ΔH_Ko         :: FT
end

function FarquharPhotosynthesis(FT=Oceananigans.defaults.FloatType;
                                Vcmax25       = 5e-5,
                                Jmax_to_Vcmax = 1.67,
                                Rd_to_Vcmax   = 0.015,
                                quantum_yield = 0.425,
                                Γstar25       = 4.332,
                                Kc25          = 39.97,
                                Ko25          = 27480,
                                O2            = 0.209,
                                θⱼ            = 0.9,
                                θ_colimit     = 0.98,
                                capacity_response = PeakedArrhenius(),
                                ΔHa_Vcmax     = 71513,
                                ΔS_Vcmax      = 649,
                                ΔHd_Vcmax     = 200000,
                                ΔHa_Jmax      = 49884,
                                ΔS_Jmax       = 646,
                                ΔHd_Jmax      = 200000,
                                heskel_b      = 0.1012,
                                heskel_c      = -0.0005,
                                ΔH_Γstar      = 37830,
                                ΔH_Kc         = 79430,
                                ΔH_Ko         = 36380)

    return FarquharPhotosynthesis{FT, typeof(capacity_response)}(
        convert(FT, Vcmax25), convert(FT, Jmax_to_Vcmax), convert(FT, Rd_to_Vcmax),
        convert(FT, quantum_yield), convert(FT, Γstar25), convert(FT, Kc25),
        convert(FT, Ko25), convert(FT, O2), convert(FT, θⱼ), convert(FT, θ_colimit),
        capacity_response,
        convert(FT, ΔHa_Vcmax), convert(FT, ΔS_Vcmax), convert(FT, ΔHd_Vcmax),
        convert(FT, ΔHa_Jmax), convert(FT, ΔS_Jmax), convert(FT, ΔHd_Jmax),
        convert(FT, heskel_b), convert(FT, heskel_c),
        convert(FT, ΔH_Γstar), convert(FT, ΔH_Kc), convert(FT, ΔH_Ko))
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
    Γstar = p.Γstar25 * arrhenius_scaling(Tₗ, p.ΔH_Γstar) * P / oftype(P, 101325)
    Kc = p.Kc25 * arrhenius_scaling(Tₗ, p.ΔH_Kc)
    Ko = p.Ko25 * arrhenius_scaling(Tₗ, p.ΔH_Ko)
    Km = Kc * (1 + p.O2 * P / Ko)

    Vcmax = β * p.Vcmax25 * capacity_scaling(p.capacity_response, Tₗ, p.ΔHa_Vcmax, p.ΔS_Vcmax, p.ΔHd_Vcmax)
    Jmax  = β * p.Jmax_to_Vcmax * p.Vcmax25 * capacity_scaling(p.capacity_response, Tₗ, p.ΔHa_Jmax, p.ΔS_Jmax, p.ΔHd_Jmax)
    Rd    = p.Rd_to_Vcmax * p.Vcmax25 * heskel_respiration_scaling(Tₗ, p.heskel_b, p.heskel_c)

    # Electron transport rate: smooth minimum of light supply and Jmax.
    J = smooth_minimum(p.quantum_yield * APAR, Jmax, p.θⱼ)

    Ac = Vcmax * (ci - Γstar) / (ci + Km)      # Rubisco-limited
    Aj = J / 4 * (ci - Γstar) / (ci + 2Γstar)     # light-limited
    Ag = smooth_minimum(Ac, Aj, p.θ_colimit)
    return Ag - Rd
end
