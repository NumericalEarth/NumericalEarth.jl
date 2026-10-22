#####
##### Stomatal conductance models: the empirical `JarvisConductance` and the
##### photosynthesis-coupled `MedlynConductance`.
#####

#####
##### Medlyn (2011) optimality stomatal conductance
#####

"""
    struct MedlynConductance

Photosynthesis-coupled optimality stomatal conductance of
[Medlyn et al. (2011)](@cite medlyn2011), in the corrected form of the
[2012 corrigendum](@cite medlyn2012corrigendum) (which supplies the 1.6 factor),

    gₛ = g₀ + 1.6 (1 + g₁/√VPD) Aₙ / cₐ ,

with `gₛ`, `g₀` in mol H₂O m⁻² s⁻¹, `Aₙ` in mol CO₂ m⁻² s⁻¹, `cₐ` the CO₂ mole
fraction at the leaf surface, VPD in Pa, and `g₁` in √Pa (ClimaLand molar form;
`diffusivity_ratio = 1.6`). The leaf-surface `cₐ` is taken equal to the ambient
(canopy-air) mole fraction: the CO₂ drawdown across the leaf boundary layer is
neglected. The `√VPD` water-use-efficiency response is *derived* from optimality,
so a single parameter `g₁` carries the humidity sensitivity. Defaults are the
ClimaLand US-Var grass values (`g₁ = 166 √Pa`).
"""
struct MedlynConductance{FT}
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
    stomatal_conductance(conductance, photosynthesis, APAR, VPD, Tˡᵉᵃᶠ, ca, p, β)

Leaf stomatal conductance `gₛ` (mol H₂O m⁻² s⁻¹), dispatched on the conductance
model. For [`MedlynConductance`](@ref) the photosynthesis–Medlyn system closes in
one pass: substituting the Medlyn `gₛ` into the CO₂ diffusion relation
`ci = cₐ − 1.6 Aₙ/gₛ` cancels the assimilation, leaving the optimality ratio

    ci / cₐ = g₁ / (g₁ + √VPD) ,

so photosynthesis is evaluated once at that `ci` — no fixed point, exact up to
the cuticular minimum `g₀`. The assimilation entering the Medlyn formula is floored
at zero, so a respiring leaf sits at `g₀` rather than driving `gₛ` negative. For
[`JarvisConductance`](@ref) `gₛ` is a closed-form product of environmental factors
and `photosynthesis` is unused. `ca` is the atmospheric CO₂ partial pressure (Pa),
which also stands in for the leaf-surface value, and `p` the air pressure (Pa).
Returns `(gₛ, Aₙ, ci)` (`Aₙ = ci = 0` for Jarvis).
"""
@inline function stomatal_conductance(c::MedlynConductance, photosynthesis,
                                      APAR, VPD, Tˡᵉᵃᶠ, ca, p, β)
    χₐ = ca / p                        # ambient CO₂ mole fraction, taken as the leaf-surface value
    ci = ca * c.g1 / (c.g1 + sqrt(VPD))
    Aₙ = net_assimilation(photosynthesis, ci, APAR, Tˡᵉᵃᶠ, p, β)
    A⁺ = max(Aₙ, zero(Aₙ))
    gₛ = c.g0 + c.diffusivity_ratio * (1 + c.g1 / sqrt(VPD)) * A⁺ / χₐ
    return gₛ, Aₙ, ci
end

#####
##### Jarvis–Stewart empirical stomatal conductance
#####

"""
    struct JarvisConductance

Jarvis-type multiplicative stomatal conductance: a maximum conductance reduced by
independent environmental factors,

    gₛ = gₛ,max · fᴾᴬᴿ(APAR) · fⱽᴾᴰ(VPD) · fᵀ(Tˡᵉᵃᶠ) · β ,

with `gₛ`, `gₛ,max` in mol H₂O m⁻² s⁻¹, a saturating light factor
`APAR/(APAR + APAR₁ᐟ₂)`, a hyperbolic VPD factor `1/(1 + kᴰ VPD)`, the Noah-MP
quadratic temperature factor `1 − c (Tᵒᵖᵗ − Tˡᵉᵃᶠ)²` evaluated at the leaf
temperature, and the soil-moisture factor `β(𝒮)`. `gₛ,max = 0.4` corresponds to a
minimum stomatal resistance of about 100 s m⁻¹. The light and VPD factors are this
model's own forms, not those of Jarvis (1976), Stewart (1988), or Noilhan & Planton (1989).

Fields:
- `maximum_conductance`   : unstressed maximum conductance (mol m⁻² s⁻¹).
- `par_half_saturation`   : PAR half-saturation of the light factor (mol m⁻² s⁻¹).
- `vpd_sensitivity`       : VPD stress coefficient (Pa⁻¹).
- `optimal_temperature`   : optimal leaf temperature (K).
- `temperature_curvature` : temperature-factor curvature (K⁻²).
"""
struct JarvisConductance{FT}
    maximum_conductance   :: FT
    par_half_saturation   :: FT
    vpd_sensitivity       :: FT
    optimal_temperature   :: FT
    temperature_curvature :: FT
end

JarvisConductance(FT=Oceananigans.defaults.FloatType;
                  maximum_conductance   = 0.4,
                  par_half_saturation   = 1e-4,
                  vpd_sensitivity       = 4e-4,
                  optimal_temperature   = 298.15,
                  temperature_curvature = 1.6e-3) =
    JarvisConductance{FT}(maximum_conductance, par_half_saturation, vpd_sensitivity,
                          optimal_temperature, temperature_curvature)

Base.summary(::JarvisConductance{FT}) where FT = "JarvisConductance{$FT}"
Base.show(io::IO, c::JarvisConductance) = print(io, summary(c),
    "(maximum_conductance=", prettysummary(c.maximum_conductance), ")")

@inline function stomatal_conductance(c::JarvisConductance, photosynthesis,
                                      APAR, VPD, Tˡᵉᵃᶠ, ca, p, β)
    fPAR = APAR / (APAR + c.par_half_saturation)
    fVPD = 1 / (1 + c.vpd_sensitivity * VPD)
    fT   = clamp(1 - c.temperature_curvature * (c.optimal_temperature - Tˡᵉᵃᶠ)^2, 0, 1)
    gₛ   = c.maximum_conductance * fPAR * fVPD * fT * β
    z    = zero(gₛ)
    return gₛ, z, z          # (gₛ, Aₙ, ci); Aₙ, ci unused for Jarvis
end
