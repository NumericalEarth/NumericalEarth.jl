#####
##### Jarvis–Stewart empirical stomatal conductance
#####

"""
    struct JarvisConductance

Jarvis-type multiplicative stomatal conductance: a maximum conductance reduced by
independent environmental factors,

    gₛ = gₛ,max · fᴾᴬᴿ(APAR) · fⱽᴾᴰ(VPD) · fᵀ(Tˡᵉᵃᶠ) · β ,

with `gₛ`, `gₛ,max` in mol m⁻² s⁻¹ (per unit leaf area), a saturating light factor
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

@inline function stomatal_conductance(c::JarvisConductance, APAR, VPD, Tˡᵉᵃᶠ, β)
    fPAR = APAR / (APAR + c.par_half_saturation)
    fVPD = 1 / (1 + c.vpd_sensitivity * VPD)
    fT   = clamp(1 - c.temperature_curvature * (c.optimal_temperature - Tˡᵉᵃᶠ)^2, 0, 1)
    gₛ   = c.maximum_conductance * fPAR * fVPD * fT * β
    return gₛ
end
