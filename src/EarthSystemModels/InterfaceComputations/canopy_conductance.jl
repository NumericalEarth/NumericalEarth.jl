#####
##### Single-source, resistance-only vegetation surface.
#####
##### A `CanopyConductanceHumidity` is the vegetation analogue of `SkinHumidity`:
##### it puts a *canopy (stomatal) conductance* `g_c = LAI · g_s` in series with
##### the aerodynamic conductance and solves the same surface vapor-flux balance
##### for `qˢ` inside the Monin–Obukhov fixed point. The stomatal conductance
##### `g_s` is the photosynthesis-coupled optimality conductance of
##### [Medlyn et al. (2011)](@cite medlyn2011), driven by the net CO₂ assimilation
##### `Aₙ` of the [Farquhar et al. (1980)](@cite farquhar1980) model — the dominant
##### lever on the Bowen ratio, the sensible/latent partition an atmosphere-coupled
##### simulation reads from the land.
#####
##### Grounded in ClimaLand (Deck et al. 2026, JAMES, App. C–E): the series
##### resistance network `r_stomata + r_ae` (Eqs E15–E17), the Farquhar
##### co-limitation (Eqs C1–C5), and the Medlyn conductance. The `min(A_c, A_j)`
##### co-limitation uses the smooth quadratic (θ) minimum and every `√`/division
##### is guarded, so the whole path is Enzyme/Reactant-friendly.
#####
##### The stomatal conductance is pluggable: the photosynthesis-coupled
##### [`MedlynConductance`](@ref) (default) or the empirical, closed-form
##### [`JarvisConductance`](@ref), both behind `AbstractStomatalConductance`.
##### Absorbed PAR is either prescribed ([`PrescribedAbsorbedPAR`](@ref), the
##### offline default) or recomputed each step from the downwelling shortwave in
##### the radiation state ([`InteractiveAbsorbedPAR`](@ref)), so the canopy can
##### follow the diurnal light cycle. CO₂ is prescribed, and the leaf temperature
##### is the skin temperature `Tₛ` (single-source).
#####

#####
##### Small differentiable helpers
#####

# Universal gas constant and dry-air molar mass. These mirror the model's
# constitutive thermodynamic parameters (`gas_constant`, `dry_air_molar_mass`);
# the accessors live in the later-loaded `Atmospheres` module, and the
# `Thermodynamics.Parameters` interface reachable here exposes only the specific
# constants (`R_d`, `R_v`), so the canopy solve — a scalar physics kernel with no
# thermodynamics parameter set in scope — restates them as module constants.
const GAS_CONSTANT = 8.3144598          # J mol⁻¹ K⁻¹
const MOLAR_MASS_DRY_AIR = 0.02897      # kg mol⁻¹
# Reference temperature of the photosynthesis rate parameters (the "25" subscript
# of Vcmax25, etc.) — distinct from the thermodynamic reference (triple point).
const REFERENCE_TEMPERATURE = 298.15    # K (25 °C)

# Arrhenius temperature scaling `f(T) = exp[ΔH (T − T₂₅) / (T₂₅ R T)]`
# (ClimaLand Eq C6). Normalized to 1 at `T = T₂₅`.
@inline function arrhenius_scaling(T, ΔH)
    T25 = oftype(T, REFERENCE_TEMPERATURE)
    R   = oftype(T, GAS_CONSTANT)
    return exp(ΔH * (T - T25) / (T25 * R * T))
end

# Peaked Arrhenius for `Vcmax`/`Jmax` (ClimaLand Eq C11): plain Arrhenius times a
# high-temperature deactivation term, so the capacity peaks near an optimum
# (low-to-mid 30s °C) and rolls off, rather than climbing without bound. The numerator
# normalizes `f′(T₂₅) = 1`, preserving the meaning of the 25 °C values. Activation
# `ΔHa`, entropy `ΔS`, deactivation `ΔHd` are the Kattge & Knorr (2007) peaked set.
# The deactivation exponent is clamped for Float32 / extreme-`T` autodiff safety;
# within the physical range it stays small so the clamp is inert.
@inline function peaked_arrhenius(T, ΔHa, ΔS, ΔHd)
    T25 = oftype(T, REFERENCE_TEMPERATURE)
    R   = oftype(T, GAS_CONSTANT)
    base = arrhenius_scaling(T, ΔHa)
    a25 = clamp((T25 * ΔS - ΔHd) / (R * T25), oftype(T, -80), oftype(T, 80))
    aT  = clamp((T   * ΔS - ΔHd) / (R * T  ), oftype(T, -80), oftype(T, 80))
    return base * (1 + exp(a25)) / (1 + exp(aT))
end

# Heskel et al. (2016) leaf-respiration temperature response (ClimaLand Eq C12),
# normalized to 1 at 25 °C. Defined in Celsius (`Tc = T − 273.15`): using Kelvin
# with these coefficients flips the sign and makes `Rd` fall with temperature.
@inline function heskel_respiration_scaling(T, b, c)
    Tc   = T - oftype(T, 273.15)
    T25c = oftype(T, 25)
    return exp(b * (Tc - T25c) + c * (Tc^2 - T25c^2))
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
    disc = max(s^2 - 4θ * a * b, zero(s))
    return (s - sqrt(disc)) / (2θ)
end

# Leaf-to-air vapor pressure deficit (Pa), floored to a small positive value so
# the Medlyn `√VPD` stays finite and differentiable at saturation.
@inline function vapor_pressure_deficit(ℂᵃᵗ, Tₗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ, phase)
    eₛ = AtmosphericThermodynamics.saturation_vapor_pressure(ℂᵃᵗ, Tₗ, phase)
    ε  = 1 / AtmosphericThermodynamics.Parameters.Rv_over_Rd(ℂᵃᵗ)   # Rᵈ/Rᵥ ≈ 0.622
    eₐ = pᵃᵗ * qᵃᵗ / (ε + (1 - ε) * qᵃᵗ)                            # air vapor pressure
    return max(eₛ - eₐ, oftype(Tₗ, 1))                              # ≥ 1 Pa
end

#####
##### Farquhar C3 photosynthesis
#####

"""
    struct FarquharPhotosynthesis

C3 photosynthesis after [Farquhar et al. (1980)](@cite farquhar1980): net CO₂
assimilation `Aₙ` is the (smoothly) co-limited minimum of the Rubisco-limited
rate `A_c` and the light (RuBP-regeneration)-limited rate `A_j`, less dark
respiration `R_d`. Rate parameters are given at 25 °C and scaled to leaf
temperature: `Vcmax`/`Jmax` by the peaked Arrhenius factor (ClimaLand Eq C11) so
they peak near an optimum and roll off at high leaf temperature, `R_d` by the
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
- `θⱼ`, `θ_colimit` : co-limitation smoothing for `J` and for `min(A_c, A_j)` (–).
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

#####
##### Stomatal conductance models, behind one dispatch seam. `MedlynConductance`
##### (default) is photosynthesis-coupled and solved iteratively;
##### `JarvisConductance` is a closed-form empirical multiplicative form that needs
##### no photosynthesis. `stomatal_conductance` dispatches on the model.
#####

abstract type AbstractStomatalConductance end

#####
##### Medlyn (2011) optimality stomatal conductance
#####

"""
    struct MedlynConductance

Photosynthesis-coupled optimality stomatal conductance of
[Medlyn et al. (2011)](@cite medlyn2011),

    gₛ = g₀ + 1.6 (1 + g₁/√VPD) Aₙ / cₐ ,

with `gₛ`, `g₀` in mol H₂O m⁻² s⁻¹, `Aₙ` in mol CO₂ m⁻² s⁻¹, `cₐ` the CO₂ mole
fraction at the leaf surface, VPD in Pa, and `g₁` in √Pa (ClimaLand molar form;
`D_rel = 1.6`). The `√VPD` water-use-efficiency response is *derived* from
optimality, so a single parameter `g₁` carries the humidity sensitivity. Defaults
are the ClimaLand US-Var grass values (`g₁ = 166 √Pa`).
"""
struct MedlynConductance{FT} <: AbstractStomatalConductance
    g0    :: FT   # cuticular / minimum conductance (mol m⁻² s⁻¹)
    g1    :: FT   # slope parameter (√Pa)
    D_rel :: FT   # 1.6 (H₂O/CO₂ diffusivity ratio)
end

MedlynConductance(FT=Oceananigans.defaults.FloatType; g0=1e-4, g1=166, D_rel=1.6) =
    MedlynConductance{FT}(g0, g1, D_rel)

Base.summary(::MedlynConductance{FT}) where FT = "MedlynConductance{$FT}"
Base.show(io::IO, c::MedlynConductance) = print(io, summary(c),
    "(g1=", prettysummary(c.g1), ")")

"""
    medlyn_conductance(conductance, An, VPD, ca_mole_fraction)

Leaf stomatal conductance `gₛ` (mol H₂O m⁻² s⁻¹) from net assimilation `An`
(mol CO₂ m⁻² s⁻¹), leaf-to-air VPD (Pa), and leaf-surface CO₂ mole fraction.
Assimilation is floored at zero so a respiring leaf sits at the minimum
conductance `g₀` rather than driving `gₛ` negative.
"""
@inline function medlyn_conductance(c::MedlynConductance, An, VPD, ca_mole_fraction)
    A⁺ = max(An, zero(An))
    return c.g0 + c.D_rel * (1 + c.g1 / sqrt(VPD)) * A⁺ / ca_mole_fraction
end

"""
    stomatal_conductance(conductance, photosynthesis, APAR, VPD, Tₗ, ca, P, β; iterations=12)

Leaf stomatal conductance `gₛ` (mol H₂O m⁻² s⁻¹), dispatched on the conductance
model. For [`MedlynConductance`](@ref) this solves the coupled Farquhar–Medlyn
system: photosynthesis sets `Aₙ(ci)`, Medlyn sets `gₛ(Aₙ)`, and CO₂ diffusion
closes the loop, `ci = cₐ − 1.6 Aₙ/gₛ`. A short damped fixed-point on `ci` (fixed
iteration count — allocation-free, GPU- and AD-safe) is used instead of an
implicit solve; it converges in a few iterations for the physiological range. For
[`JarvisConductance`](@ref) `gₛ` is a closed-form product of environmental
factors, `photosynthesis` is unused, and no iteration runs. `ca` is the
atmospheric CO₂ partial pressure (Pa) and `P` the air pressure (Pa). Returns
`(gₛ, Aₙ, ci)` (`Aₙ = ci = 0` for Jarvis).
"""
@inline function stomatal_conductance(c::MedlynConductance, photosynthesis,
                                      APAR, VPD, Tₗ, ca, P, β; iterations=12)
    ca_mf = ca / P                       # CO₂ mole fraction
    ci    = oftype(ca, 0.7) * ca         # initial intercellular CO₂ (Pa)
    damp  = oftype(ca, 0.5)
    An    = zero(ca)
    gs    = c.g0

    for _ in 1:iterations
        An = net_assimilation(photosynthesis, ci, APAR, Tₗ, P, β)
        gs = medlyn_conductance(c, An, VPD, ca_mf)
        ci_target_mf = ca_mf - c.D_rel * An / gs
        # Keep ci in the physical band (Γstar-ish floor, ≤ cₐ) and damp the update.
        ci_target = clamp(ci_target_mf, oftype(ca, 1e-6), ca_mf) * P
        ci = ci + damp * (ci_target - ci)
    end

    return gs, An, ci
end

#####
##### Jarvis–Stewart empirical stomatal conductance
#####

"""
    struct JarvisConductance

Empirical multiplicative stomatal conductance after Jarvis (1976) / Stewart
(1988): a maximum conductance reduced by independent environmental stress
factors,

    gₛ = gₛ,max · f_PAR(APAR) · f_VPD(VPD) · f_T(Tₗ) · β ,

with `gₛ`, `gₛ,max` in mol H₂O m⁻² s⁻¹. Unlike [`MedlynConductance`](@ref) it is
not coupled to photosynthesis, so it is closed-form (no iteration, no Farquhar
call) — cheap and a trivial reverse-mode adjoint, adequate for weather-timescale
runs. The soil-moisture factor is the same `β(𝒮)` the interface already forms.
Defaults follow the Noilhan & Planton (1989) / Noah tables (`gₛ,max ≈ 0.4`
corresponds to a minimum stomatal resistance `Rsmin ≈ 100 s m⁻¹`).

Fields:
- `gs_max`          : unstressed maximum conductance (mol m⁻² s⁻¹).
- `par_half`        : PAR half-saturation of the light factor (mol m⁻² s⁻¹).
- `vpd_sensitivity` : VPD stress coefficient (Pa⁻¹).
- `T_reference`     : optimal leaf temperature (K).
- `T_curvature`     : temperature-factor curvature (K⁻²).
- `factor_floor`    : lower clamp on each factor (numerical safety).
"""
struct JarvisConductance{FT} <: AbstractStomatalConductance
    gs_max          :: FT
    par_half        :: FT
    vpd_sensitivity :: FT
    T_reference     :: FT
    T_curvature     :: FT
    factor_floor    :: FT
end

JarvisConductance(FT=Oceananigans.defaults.FloatType;
                  gs_max          = 0.4,
                  par_half        = 1e-4,
                  vpd_sensitivity = 4e-4,
                  T_reference     = 298.15,
                  T_curvature     = 1.6e-3,
                  factor_floor    = 1e-3) =
    JarvisConductance{FT}(gs_max, par_half, vpd_sensitivity,
                          T_reference, T_curvature, factor_floor)

Base.summary(::JarvisConductance{FT}) where FT = "JarvisConductance{$FT}"
Base.show(io::IO, c::JarvisConductance) = print(io, summary(c),
    "(gs_max=", prettysummary(c.gs_max), ")")

# Light factor: saturating in absorbed PAR (0 → 1). VPD factor: hyperbolic
# decline as the air dries (1 → 0). Temperature factor: quadratic in `Tₗ` peaking
# at `T_reference`, clamped to stay positive away from the optimum.
@inline jarvis_light_factor(c::JarvisConductance, APAR) = APAR / (APAR + c.par_half)
@inline jarvis_vpd_factor(c::JarvisConductance, VPD)    = 1 / (1 + c.vpd_sensitivity * VPD)

@inline function jarvis_temperature_factor(c::JarvisConductance, T)
    f = 1 - c.T_curvature * (c.T_reference - T)^2
    return clamp(f, c.factor_floor, one(f))
end

@inline function stomatal_conductance(c::JarvisConductance, photosynthesis,
                                      APAR, VPD, Tₗ, ca, P, β; kw...)
    fPAR = jarvis_light_factor(c, APAR)
    fVPD = jarvis_vpd_factor(c, VPD)
    fT   = jarvis_temperature_factor(c, Tₗ)
    gs   = c.gs_max * fPAR * fVPD * fT * β
    z    = zero(gs)
    return gs, z, z          # (gₛ, Aₙ, ci); Aₙ, ci unused for Jarvis
end

#####
##### Beer–Lambert absorbed PAR (helper — used to derive `absorbed_par` from a
##### downwelling PAR flux; not called inside the flux solver, where APAR is
##### prescribed). ClimaLand Eqs D9, D11.
#####

"""
    beer_lambert_absorbed_fraction(leaf_area_index, leaf_albedo, extinction, clumping)

Fraction of incident shortwave a bulk canopy absorbs, `f_abs = (1 − α)(1 − e^{−K·LAI·Ω})`
(ClimaLand Eq D11). Multiply an incident PAR photon flux by this to get `absorbed_par`.
"""
@inline function beer_lambert_absorbed_fraction(leaf_area_index, leaf_albedo, extinction, clumping)
    transmitted = exp(-extinction * leaf_area_index * clumping)
    return (1 - leaf_albedo) * (1 - transmitted)
end

#####
##### Absorbed PAR spec — prescribed (offline default) or live from radiation.
##### Consumed as a per-unit-leaf-area quantity: the canopy up-scaling happens
##### downstream via `g_c = LAI · gₛ`, so `InteractiveAbsorbedPAR` divides the
##### Beer–Lambert canopy-absorbed flux by `LAI` to match the per-leaf convention
##### that `net_assimilation` and the Jarvis light factor expect.
#####

abstract type AbstractAbsorbedPAR end

"""
    PrescribedAbsorbedPAR(value)

Fixed per-leaf absorbed PAR (mol photon m⁻² s⁻¹) — the offline default. Ignores
the radiation state and reproduces the original constant-`absorbed_par` behavior.
"""
struct PrescribedAbsorbedPAR{FT} <: AbstractAbsorbedPAR
    value :: FT
end

Base.summary(p::PrescribedAbsorbedPAR) = string("PrescribedAbsorbedPAR(", prettysummary(p.value), ")")
Base.show(io::IO, p::PrescribedAbsorbedPAR) = print(io, summary(p))

"""
    InteractiveAbsorbedPAR(FT = Float64; par_fraction, photon_per_joule,
                           leaf_albedo_par, extinction, clumping, lai_min)

Per-leaf absorbed PAR recomputed each step from the downwelling shortwave `ℐꜜˢʷ`
(W m⁻²) in the radiation state,

    APAR = f_abs(LAI) · (f_par · ℐꜜˢʷ · Q_J) / max(LAI, LAI_min) ,

where `f_par` (`par_fraction`) is the PAR energy fraction of shortwave, `Q_J`
(`photon_per_joule`) converts PAR energy to a photon flux, and `f_abs` is the
Beer–Lambert canopy-absorbed fraction ([`beer_lambert_absorbed_fraction`](@ref)).
Dividing by `LAI` returns a per-leaf value (see the convention note above). With
the shortwave following the sun, the canopy conductance then follows the diurnal
light cycle — the dominant daytime driver of transpiration.

Fields:
- `par_fraction`     : PAR/shortwave by energy (≈ 0.45).
- `photon_per_joule` : mol photons per J in the PAR band (≈ 4.57e-6).
- `leaf_albedo_par`  : leaf albedo in the PAR band.
- `extinction`       : canopy extinction coefficient `K`.
- `clumping`         : foliage clumping index `Ω`.
- `lai_min`          : floor on `LAI` in the per-leaf division.
"""
struct InteractiveAbsorbedPAR{FT} <: AbstractAbsorbedPAR
    par_fraction     :: FT
    photon_per_joule :: FT
    leaf_albedo_par  :: FT
    extinction       :: FT
    clumping         :: FT
    lai_min          :: FT
end

InteractiveAbsorbedPAR(FT=Oceananigans.defaults.FloatType;
                       par_fraction     = 0.45,
                       photon_per_joule = 4.57e-6,
                       leaf_albedo_par  = 0.1,
                       extinction       = 0.5,
                       clumping         = 1,
                       lai_min          = 0.1) =
    InteractiveAbsorbedPAR{FT}(par_fraction, photon_per_joule, leaf_albedo_par,
                               extinction, clumping, lai_min)

Base.summary(::InteractiveAbsorbedPAR{FT}) where FT = "InteractiveAbsorbedPAR{$FT}"
Base.show(io::IO, p::InteractiveAbsorbedPAR) = print(io, summary(p),
    "(par_fraction=", prettysummary(p.par_fraction), ")")

# Wrap a bare number as the prescribed spec (backward-compatible constructor arg).
@inline absorbed_par_spec(x::AbstractAbsorbedPAR, FT) = x
@inline absorbed_par_spec(x::Number, FT) = PrescribedAbsorbedPAR(convert(FT, x))

@inline absorbed_par_value(p::PrescribedAbsorbedPAR, radiation, leaf_area_index) = p.value

@inline function absorbed_par_value(p::InteractiveAbsorbedPAR, radiation, leaf_area_index)
    SW   = radiation.ℐꜜˢʷ                                      # downwelling shortwave (W m⁻²)
    parQ = p.par_fraction * SW * p.photon_per_joule            # canopy-incident PAR photon flux
    fabs = beer_lambert_absorbed_fraction(leaf_area_index, p.leaf_albedo_par, p.extinction, p.clumping)
    return fabs * parQ / max(leaf_area_index, p.lai_min)       # per-leaf
end

#####
##### CanopyConductanceHumidity — the humidity-formulation slot
#####

"""
    struct CanopyConductanceHumidity

Surface specific humidity `qˢ` for a single-source (big-leaf) canopy: the
canopy conductance `g_c = LAI · gₛ` in series with the aerodynamic conductance,
solved inside the Monin–Obukhov fixed point exactly as [`SkinHumidity`](@ref)
solves a soil-resistance balance. The stomatal conductance `gₛ` comes from the
`conductance` model driven by the per-cell leaf-to-air VPD, leaf temperature
(`= Tₛ`, single-source), and absorbed PAR, with the moisture-stress factor `β(𝒮)`
read from the ground hydrology (`moisture_stress`, a `Number` or
[`CriticalSaturation`](@ref)). The conductance is either the photosynthesis-coupled
[`MedlynConductance`](@ref) (needs a `photosynthesis` model) or the empirical
[`JarvisConductance`](@ref) (needs none). Absorbed PAR is prescribed
([`PrescribedAbsorbedPAR`](@ref)) or live from the radiation state
([`InteractiveAbsorbedPAR`](@ref)); CO₂ is prescribed.

Because the canopy vapor flux *is* transpiration, the resulting reduced `qˢ`
lowers the latent-heat / vapor flux, which the existing
flux → evaporation → water-storage plumbing already routes as a sink on the ground
water store — no separate transpiration wiring is needed.

Fields:
- `leaf_area_index` : bulk LAI (–), upscales leaf `gₛ` to the canopy.
- `photosynthesis`  : a [`FarquharPhotosynthesis`](@ref), or `nothing` for Jarvis.
- `conductance`     : a [`MedlynConductance`](@ref) or [`JarvisConductance`](@ref).
- `moisture_stress` : `β(𝒮)` model — a `Number` or [`CriticalSaturation`](@ref).
- `absorbed_par`    : an [`AbstractAbsorbedPAR`](@ref) (a `Number` is wrapped as prescribed).
- `atmospheric_co2` : prescribed CO₂ partial pressure (Pa).
- `phase`           : saturation phase (Liquid).
"""
struct CanopyConductanceHumidity{L, P, C, S, A, Q, Φ}
    leaf_area_index :: L
    photosynthesis  :: P
    conductance     :: C
    moisture_stress :: S
    absorbed_par    :: A
    atmospheric_co2 :: Q
    phase           :: Φ
end

# Medlyn needs a Farquhar model; Jarvis needs none. Default `photosynthesis` per
# conductance type when the user leaves it unset (`nothing`).
@inline default_photosynthesis(photosynthesis, conductance, FT) = photosynthesis
@inline default_photosynthesis(::Nothing, ::MedlynConductance, FT) = FarquharPhotosynthesis(FT)
@inline default_photosynthesis(::Nothing, ::JarvisConductance, FT) = nothing

function CanopyConductanceHumidity(FT=Oceananigans.defaults.FloatType;
                                   leaf_area_index = 2,
                                   photosynthesis  = nothing,
                                   conductance     = MedlynConductance(FT),
                                   moisture_stress = 1,
                                   absorbed_par    = 4e-4,
                                   atmospheric_co2 = 40,
                                   phase           = AtmosphericThermodynamics.Liquid())

    photosynthesis = default_photosynthesis(photosynthesis, conductance, FT)

    return CanopyConductanceHumidity(convert(FT, leaf_area_index),
                                     photosynthesis, conductance, moisture_stress,
                                     absorbed_par_spec(absorbed_par, FT),
                                     convert(FT, atmospheric_co2), phase)
end

Base.summary(::CanopyConductanceHumidity{L, P, C, S, A, Q, Φ}) where {L, P, C, S, A, Q, Φ} =
    string("CanopyConductanceHumidity{", Φ === AtmosphericThermodynamics.Liquid ? "Liquid" : "Ice", "}")
Base.show(io::IO, q::CanopyConductanceHumidity) = print(io, summary(q))

# The canopy stress reads the ground saturation 𝒮 (as `CriticalSaturation` does),
# so the interface materializes it into the per-cell land state.
@inline interface_hydrology_state(i, j, grid, ::CanopyConductanceHumidity, land_state) =
    land_saturation(i, j, grid, land_state)

# `CanopyConductanceHumidity`: solve the surface vapor-flux balance for qˢ with a
# canopy conductance g_c in series with the turbulent transfer — the SkinHumidity
# construction with gˢ → g_c. The canopy (leaf) reservoir is saturated at the leaf
# temperature (= skin temperature Tₛ, single-source). The stomatal conductance is
# the `conductance`-model solve; g_c = LAI · gₛ · Mₐ converts the molar leaf
# conductance to the mass conductance the specific-humidity balance uses.
# Canopy flux terms, split off so the standalone formulation and the composite
# (soil + canopy) share them. Returns the bulk canopy (stomatal) mass conductance
# `g_c = LAI · gₛ · Mₐ` (kg m⁻² s⁻¹) and the leaf saturation source `qᵛ⁺(Tₗ)`.
# `Ψᵣ` is the interface radiation state (drives `InteractiveAbsorbedPAR`).
@inline function canopy_conductance_terms(q::CanopyConductanceHumidity, Tₗ, Ψₛ, Ψₐ, Ψᵣ, ℙₐ)
    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    pᵃᵗ = Ψₐ.p
    qᵃᵗ = Ψₐ.q
    Tᵃᵗ = Ψₐ.T

    qᵛ⁺  = saturation_specific_humidity(ℂᵃᵗ, Tₗ, pᵃᵗ, q.phase)
    VPD  = vapor_pressure_deficit(ℂᵃᵗ, Tₗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ, q.phase)
    β    = evaporation_efficiency(q.moisture_stress, Ψₛ.hydrology)
    APAR = absorbed_par_value(q.absorbed_par, Ψᵣ, q.leaf_area_index)

    gs, _, _ = stomatal_conductance(q.conductance, q.photosynthesis,
                                    APAR, VPD, Tₗ, q.atmospheric_co2, pᵃᵗ, β)

    # Molar leaf conductance → canopy mass conductance (kg m⁻² s⁻¹).
    g_c = q.leaf_area_index * gs * oftype(gs, MOLAR_MASS_DRY_AIR)

    return g_c, qᵛ⁺
end

@inline function compute_interface_humidity(q::CanopyConductanceHumidity, Tₛ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    FT = eltype(Ψₛ)
    g_c, qᵛ⁺ = canopy_conductance_terms(q, Tₛ, Ψₛ, Ψₐ, Ψᵣ, ℙₐ)   # leaf temperature = skin temperature Tₛ

    qˢ⁻ = Ψₛ.specific_humidity
    qᵃᵗ = Ψₐ.q
    Jᵃ, Δq = atmospheric_vapor_flux(Ψₛ, Ψₐ, ℙₐ.thermodynamics_parameters)

    D  = g_c * Δq + Jᵃ
    qˢ = (g_c * qᵛ⁺ * Δq + Jᵃ * qᵃᵗ) / D

    return convert(FT, ifelse(D == 0, qˢ⁻, qˢ))
end
