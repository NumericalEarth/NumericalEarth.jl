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
`diffusivity_ratio = 1.6`). The `√VPD` water-use-efficiency response is *derived*
from optimality, so a single parameter `g₁` carries the humidity sensitivity.
Defaults are the ClimaLand US-Var grass values (`g₁ = 166 √Pa`).
"""
struct MedlynConductance{FT} <: AbstractStomatalConductance
    g0                :: FT   # cuticular / minimum conductance (mol m⁻² s⁻¹)
    g1                :: FT   # slope parameter (√Pa)
    diffusivity_ratio :: FT   # 1.6 (H₂O/CO₂ diffusivity ratio)
end

MedlynConductance(FT=Oceananigans.defaults.FloatType; g0=1e-4, g1=166, diffusivity_ratio=1.6) =
    MedlynConductance{FT}(g0, g1, diffusivity_ratio)

Base.summary(::MedlynConductance{FT}) where FT = "MedlynConductance{$FT}"
Base.show(io::IO, c::MedlynConductance) = print(io, summary(c),
    "(g1=", prettysummary(c.g1), ")")

"""
    medlyn_conductance(conductance, An, VPD, χCO₂)

Leaf stomatal conductance `gₛ` (mol H₂O m⁻² s⁻¹) from net assimilation `An`
(mol CO₂ m⁻² s⁻¹), leaf-to-air VPD (Pa), and leaf-surface CO₂ mole fraction
`χCO₂`. Assimilation is floored at zero so a respiring leaf sits at the minimum
conductance `g₀` rather than driving `gₛ` negative.
"""
@inline function medlyn_conductance(c::MedlynConductance, An, VPD, χCO₂)
    A⁺ = max(An, zero(An))
    return c.g0 + c.diffusivity_ratio * (1 + c.g1 / sqrt(VPD)) * A⁺ / χCO₂
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
    χa      = ca / P                # ambient CO₂ mole fraction
    ci      = oftype(ca, 0.7) * ca  # initial intercellular CO₂ (Pa)
    damping = oftype(ca, 0.5)
    An      = zero(ca)
    gs      = c.g0

    for _ in 1:iterations
        An = net_assimilation(photosynthesis, ci, APAR, Tₗ, P, β)
        gs = medlyn_conductance(c, An, VPD, χa)
        χi★ = χa - c.diffusivity_ratio * An / gs
        # Keep ci in the physical band (Γstar-ish floor, ≤ cₐ) and damp the update.
        ci★ = clamp(χi★, oftype(ca, 1e-6), χa) * P
        ci = ci + damping * (ci★ - ci)
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
- `maximum_conductance`   : unstressed maximum conductance (mol m⁻² s⁻¹).
- `par_half_saturation`   : PAR half-saturation of the light factor (mol m⁻² s⁻¹).
- `vpd_sensitivity`       : VPD stress coefficient (Pa⁻¹).
- `optimal_temperature`   : optimal leaf temperature (K).
- `temperature_curvature` : temperature-factor curvature (K⁻²).
- `factor_floor`          : lower clamp on each factor (numerical safety).
"""
struct JarvisConductance{FT} <: AbstractStomatalConductance
    maximum_conductance   :: FT
    par_half_saturation   :: FT
    vpd_sensitivity       :: FT
    optimal_temperature   :: FT
    temperature_curvature :: FT
    factor_floor          :: FT
end

JarvisConductance(FT=Oceananigans.defaults.FloatType;
                  maximum_conductance   = 0.4,
                  par_half_saturation   = 1e-4,
                  vpd_sensitivity       = 4e-4,
                  optimal_temperature   = 298.15,
                  temperature_curvature = 1.6e-3,
                  factor_floor          = 1e-3) =
    JarvisConductance{FT}(maximum_conductance, par_half_saturation, vpd_sensitivity,
                          optimal_temperature, temperature_curvature, factor_floor)

Base.summary(::JarvisConductance{FT}) where FT = "JarvisConductance{$FT}"
Base.show(io::IO, c::JarvisConductance) = print(io, summary(c),
    "(maximum_conductance=", prettysummary(c.maximum_conductance), ")")

# Light factor: saturating in absorbed PAR (0 → 1). VPD factor: hyperbolic
# decline as the air dries (1 → 0). Temperature factor: quadratic in `Tₗ` peaking
# at `optimal_temperature`, clamped to stay positive away from the optimum.
@inline jarvis_light_factor(c::JarvisConductance, APAR) = APAR / (APAR + c.par_half_saturation)
@inline jarvis_vpd_factor(c::JarvisConductance, VPD)    = 1 / (1 + c.vpd_sensitivity * VPD)

@inline function jarvis_temperature_factor(c::JarvisConductance, T)
    f = 1 - c.temperature_curvature * (c.optimal_temperature - T)^2
    return clamp(f, c.factor_floor, one(f))
end

@inline function stomatal_conductance(c::JarvisConductance, photosynthesis,
                                      APAR, VPD, Tₗ, ca, P, β; kw...)
    fPAR = jarvis_light_factor(c, APAR)
    fVPD = jarvis_vpd_factor(c, VPD)
    fT   = jarvis_temperature_factor(c, Tₗ)
    gs   = c.maximum_conductance * fPAR * fVPD * fT * β
    z    = zero(gs)
    return gs, z, z          # (gₛ, Aₙ, ci); Aₙ, ci unused for Jarvis
end
