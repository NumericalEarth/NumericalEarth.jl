#####
##### `DryLayerHumidity` — atmosphere-facing specific humidity from a
##### vapor-flux balance across an unresolved dry surface layer.
#####
##### Physical picture (Or et al. 2013, Vadose Zone J., review): bare-soil
##### evaporation has two stages. While the surface is hydraulically connected
##### to moist soil by capillary flow ("stage 1") the skin stays effectively
##### saturated and evaporation is demand-limited. Once the soil dries, the
##### evaporating front retreats below the surface and a dry surface layer
##### grows; evaporation becomes limited by Fickian vapor diffusion through
##### that layer ("stage 2"). In-soil vapor diffusion only dominates within
##### this thin dry layer (Philip & de Vries 1957; Tang & Riley 2013), which
##### is why land models — including multi-layer Richards-equation models —
##### transport only liquid water in the soil column and represent the
##### dry-layer vapor limitation at the surface, as a soil resistance or a
##### reduced surface humidity (Vanderborght et al. 2017).
#####
##### This closure is that surface representation, written as a humidity
##### boundary condition in the spirit of Ye & Pielke (1993): the
##### atmosphere-facing humidity `qⁱⁿ` is solved (by the existing
##### `compute_interface_state` fixed point) from the balance between the
##### dry-layer Fick flux and the atmospheric turbulent flux,
#####
#####     Jᵉ→ⁱⁿ = Gᵉ (qᵉ - qⁱⁿ),         Gᵉ = ρᵃᵗ Dᵛ_eff / max(δᵛ, δᵛ_min),
#####     Jⁱⁿ→ᵃ = ρᵃᵗ wᵛ (qⁱⁿ - qᵃᵗ)    (atmospheric side, from u★ q★)
#####
##### closed by `Jᵉ→ⁱⁿ = Jⁱⁿ→ᵃ`. `Gᵉ` is the reciprocal of the dry-surface-
##### layer soil resistance `r_soil = δᵛ/Dᵛ_eff` of Yamanaka et al. (1997)
##### and Swenson & Lawrence (2014) (the CLM5 scheme); the diagnostic depth
##### `δᵛ(𝒮)` plays their DSL-thickness role, and `Dᵛ_eff` optionally carries
##### a Millington & Quirk (1961) tortuosity factor. Solving the balance
##### rather than prescribing an efficiency follows Ye & Pielke's analysis:
##### a prescribed `β(𝒮)` ignores the atmospheric state and an `α qᵛ⁺` skin
##### overestimates evaporation from unsaturated soil, while the combined
##### (flux-balance) form is accurate in both limits.
#####
##### The vapor source is saturated air at the front temperature
##### `Tᵉ = Tⁱⁿ + χ(Tˡᵃ − Tⁱⁿ)` with `χ = clip(δᵛ/ℓᵀ, 0, 1)`. The wet branch
##### (`δᵛ ≲ δᵛ_min`) collapses to `qⁱⁿ = qᵛ⁺(Tⁱⁿ)` so the saturated-surface
##### limit reproduces the existing similarity-theory behavior; it hands over
##### to the dry-layer series solution through a smooth logistic blend of
##### width `wet_transition_width` (sharp switch when 0).
#####
##### Pair this with `SkinTemperature(DiffusiveFlux(δ=ℓᵀ, κ=κᵀ))` on the
##### temperature side: the same `Λⁱⁿ = κᵀ/ℓᵀ` couples the bulk land temperature
##### `Tˡᵃ` to the skin temperature `Tⁱⁿ` and the energy fluxes
##### (`𝒬ᴿ + 𝒬ᵀ + 𝒬ᵛ`) — no separate temperature formulation is needed.
#####

using Oceananigans: Oceananigans
using Oceananigans.Utils: prettysummary
using Thermodynamics: Thermodynamics as AtmosphericThermodynamics

#####
##### Dry-layer depth diagnostics
#####

"""
    StorageBasedDryLayerDepth(maximum_dry_layer_depth, dry_layer_onset_saturation,
                                      dry_layer_exponent)

Diagnostic dry-layer depth `δᵛ` as a function of land saturation `𝒮`:

```math
\\delta^v(\\mathcal S) = \\delta^v_{max}
\\left[1 - \\min\\!\\left(\\frac{\\mathcal S}{\\mathcal S^c},\\ 1\\right)\\right]^\\eta.
```

`δᵛ = 0` when `𝒮 ≥ 𝒮ᶜ` (wet branch), growing toward `δᵛ_max` as the slab dries.

Here `dry_layer_onset_saturation` (`𝒮ᶜ`) is the slab saturation at which the
dry layer first appears.

Diagnosing the dry-surface-layer thickness from near-surface moisture follows
the CLM5 scheme of [Swenson and Lawrence (2014)](@cite swenson2014dry)
(also used by ClimaLand, with maximum depth ≈ 15 mm after Shokri and Or 2011);
this closure differs only in using the slab saturation `𝒮` as the moisture
variable and a power-law shape.
"""
struct StorageBasedDryLayerDepth{FT}
    maximum_dry_layer_depth    :: FT
    dry_layer_onset_saturation :: FT
    dry_layer_exponent         :: FT
end

StorageBasedDryLayerDepth(FT::Type = Oceananigans.defaults.FloatType;
                          maximum_dry_layer_depth,
                          dry_layer_onset_saturation,
                          dry_layer_exponent = 2) =
    StorageBasedDryLayerDepth(convert(FT, maximum_dry_layer_depth),
                              convert(FT, dry_layer_onset_saturation),
                              convert(FT, dry_layer_exponent))

@inline function dry_layer_depth(d::StorageBasedDryLayerDepth, 𝒮)
    FT = typeof(𝒮)
    s  = min(𝒮 / convert(FT, d.dry_layer_onset_saturation), one(FT))
    return convert(FT, d.maximum_dry_layer_depth) *
           max(one(FT) - s, zero(FT))^convert(FT, d.dry_layer_exponent)
end

Base.summary(d::StorageBasedDryLayerDepth) =
    string("StorageBasedDryLayerDepth(δᵛ_max=", prettysummary(d.maximum_dry_layer_depth),
           ", 𝒮ᶜ=", prettysummary(d.dry_layer_onset_saturation),
           ", η=", prettysummary(d.dry_layer_exponent), ")")

#####
##### Dry-layer vapor exchange parameters
#####

"""
    ConstantTortuosity()

Trivial tortuosity model: `Dᵛ_eff = Dᵛ₀`. Used by
[`DryLayerVaporPistonVelocity`](@ref) when the soil air space is not modeled
explicitly.
"""
struct ConstantTortuosity end

"""
    MillingtonQuirk()

Millington–Quirk tortuosity: `Dᵛ_eff = Dᵛ₀ · θᵍ^(10/3) / ν²` where
`θᵍ = ν − θˡ` is the gas-filled pore fraction. Reduces vapor diffusivity in
near-saturated soils.
"""
struct MillingtonQuirk end

Base.summary(::ConstantTortuosity) = "ConstantTortuosity"
Base.summary(::MillingtonQuirk)    = "MillingtonQuirk"

"""
    DryLayerVaporPistonVelocity(minimum_dry_layer_depth, molecular_diffusivity;
                                tortuosity_model = ConstantTortuosity(),
                                wet_transition_width = 5 * minimum_dry_layer_depth)

Parameters of the dry-layer vapor piston velocity `wᵈ = Dᵛ_eff / max(δᵛ, δᵛ_min)`,
the reciprocal of the dry-surface-layer soil resistance `r_soil = δᵛ/Dᵛ_eff` of
[Yamanaka et al. (1997)](@cite yamanaka1997surface) and
[Swenson and Lawrence (2014)](@cite swenson2014dry). The tortuosity model is a
singleton type — [`ConstantTortuosity`](@ref) or [`MillingtonQuirk`](@ref),
after [Millington and Quirk (1961)](@cite millington1961permeability) —
dispatched on by `effective_vapor_diffusivity`. The piston velocity feeds the
[`DryLayerHumidity`](@ref) flux balance.

`wet_transition_width` (m) is the width over which the saturated-skin (wet)
humidity transitions to the dry-layer series solution: the two are combined with
a logistic weight in `δᵛ` centered at `δᵛ_min + wet_transition_width/2`, so the
transition is infinitely differentiable (see [Kavetski and Kuczera (2007)](@cite kavetski2007smoothing))
and ≈99 % complete across `δᵛ ∈ [δᵛ_min, δᵛ_min + wet_transition_width]`. 
Pass `0` to recover a sharp switch at `δᵛ = δᵛ_min`.
"""
struct DryLayerVaporPistonVelocity{FT, T}
    minimum_dry_layer_depth :: FT
    molecular_diffusivity   :: FT
    wet_transition_width    :: FT
    tortuosity_model        :: T
end

DryLayerVaporPistonVelocity(FT::Type = Oceananigans.defaults.FloatType;
                            minimum_dry_layer_depth,
                            molecular_diffusivity,
                            wet_transition_width = 5 * minimum_dry_layer_depth,
                            tortuosity_model = ConstantTortuosity()) =
    DryLayerVaporPistonVelocity(convert(FT, minimum_dry_layer_depth),
                                convert(FT, molecular_diffusivity),
                                convert(FT, wet_transition_width),
                                tortuosity_model)

Base.summary(v::DryLayerVaporPistonVelocity) =
    string("DryLayerVaporPistonVelocity(δᵛ_min=", prettysummary(v.minimum_dry_layer_depth),
           ", Dᵛ₀=", prettysummary(v.molecular_diffusivity),
           ", δᵛʷ=", prettysummary(v.wet_transition_width),
           ", tortuosity=", summary(v.tortuosity_model), ")")

#####
##### DryLayerHumidity — the humidity formulation
#####

"""
    DryLayerHumidity(phase = AtmosphericThermodynamics.Liquid();
                             dry_layer_depth,
                             vapor_exchange,
                             thermal_exchange_depth,
                             porosity)

Surface specific-humidity formulation for the *dry-layer* model:
`qⁱⁿ` is solved from a vapor-flux balance between a Fick flux through an
unresolved dry surface layer and the atmospheric vapor flux, following
[Ye and Pielke (1993)](@cite yepielke1993) with the dry-layer (DSL)
resistance of [Yamanaka et al. (1997)](@cite yamanaka1997surface) and
[Swenson and Lawrence (2014)](@cite swenson2014dry). The
formulation plugs into the existing `compute_interface_state` solver exactly
where [`SkinHumidity`](@ref) does, and reduces to a wet-surface
saturated-skin BC when the slab is wet enough (`𝒮 ≥ 𝒮ᶜ`).

* `dry_layer_depth` — depth diagnostic, e.g.
  [`StorageBasedDryLayerDepth`](@ref).
* `vapor_exchange` — `δᵛ_min`, `Dᵛ₀`, tortuosity (a
  [`DryLayerVaporPistonVelocity`](@ref)).
* `thermal_exchange_depth` — `ℓᵀ` (m), the same depth used by
  `SkinTemperature(DiffusiveFlux)` on the temperature side. Controls the
  interpolation `Tᵉ = Tⁱⁿ + χ(Tˡᵃ − Tⁱⁿ)` with `χ = clip(δᵛ/ℓᵀ, 0, 1)`.
* `porosity` — `ν`, soil porosity (matches the hydrology closure; needed
  for the Millington–Quirk tortuosity).

The dry-layer source humidity is the saturation value at the front
temperature, `qᵉ = qᵛ⁺(Tᵉ)`, with no matric-suction (Kelvin-equation)
reduction of the pore relative humidity (the `hₛ` of
[Ye and Pielke (1993)](@cite yepielke1993), after Philip 1957) — that factor
departs appreciably from 1 only at extreme dryness.
"""
struct DryLayerHumidity{EFD, VEX, FT, Φ}
    dry_layer_depth        :: EFD
    vapor_exchange         :: VEX
    thermal_exchange_depth :: FT
    porosity               :: FT
    phase                  :: Φ
end

DryLayerHumidity(phase = AtmosphericThermodynamics.Liquid();
                 dry_layer_depth,
                 vapor_exchange,
                 thermal_exchange_depth,
                 porosity) =
    DryLayerHumidity(dry_layer_depth,
                     vapor_exchange,
                     convert(Oceananigans.defaults.FloatType, thermal_exchange_depth),
                     convert(Oceananigans.defaults.FloatType, porosity),
                     phase)

Base.summary(q::DryLayerHumidity{EFD, VEX, FT, Φ}) where {EFD, VEX, FT, Φ} =
    string("DryLayerHumidity{",
           Φ === AtmosphericThermodynamics.Liquid ? "Liquid" : "Ice",
           "}(depth=", summary(q.dry_layer_depth),
           ", vapor=", summary(q.vapor_exchange),
           ", ℓᵀ=", prettysummary(q.thermal_exchange_depth),
           ", ν=", prettysummary(q.porosity), ")")
Base.show(io::IO, q::DryLayerHumidity) = print(io, summary(q))

#####
##### Effective vapor diffusivity (tortuosity)
#####
##### Dispatched on the tortuosity-model singleton so the per-cell call is
##### compile-time-resolved (no runtime `if` branch inside the kernel path).
#####

@inline effective_vapor_diffusivity(v::DryLayerVaporPistonVelocity, ν, θˡ) =
    effective_vapor_diffusivity(v.tortuosity_model, v.molecular_diffusivity, ν, θˡ)

@inline effective_vapor_diffusivity(::ConstantTortuosity, D₀, ν, θˡ) =
    convert(typeof(θˡ), D₀)

@inline function effective_vapor_diffusivity(::MillingtonQuirk, D₀, ν, θˡ)
    FT = typeof(θˡ)
    νF = convert(FT, ν)
    θᵍ = max(νF - θˡ, zero(FT))
    return convert(FT, D₀) * θᵍ^(FT(10//3)) / νF^2
end

#####
##### Humidity solver
#####
##### Sign convention for fluxes here matches `SkinHumidity` exactly: every
##### flux is positive upward, and `Jᵃ = -ρᵃᵗ u★ q★` is the atmospheric vapor
##### flux the similarity solver computed from the previous Picard iterate
##### (q★ < 0 when evaporating).
#####
##### The model. The pore air at the front is saturated, so the source
##### humidity is the saturation specific humidity at the front temperature,
##### qᵉ = qᵛ⁺(Tᵉ, pᵃᵗ). The dry layer transmits a Fick flux from the front
##### up to the interface (humidity qⁱⁿ),
#####
#####     Jᵉ = Gᵉ (qᵉ - qⁱⁿ),        Gᵉ = ρᵃᵗ Dᵛ_eff / max(δᵛ, δᵛ_min),
#####
##### so a wetter front (qᵉ > qⁱⁿ) drives vapor upward, while above the
##### interface similarity theory carries vapor away at Jᵃ(Tⁱⁿ, qⁱⁿ). The
##### interface stores no vapor, so Jᵉ = Jᵃ — a nonlinear equation for qⁱⁿ.
#####
##### The solver. Over one Picard iteration we linearize the similarity flux
##### as a bulk conductance law anchored at the previous iterate qⁱⁿ⁻,
#####
#####     Jᵃ(q) ≈ Gᵃ (q - qᵃᵗ),      Gᵃ = Jᵃ / Δq,      Δq = qⁱⁿ⁻ - qᵃᵗ,
#####
##### chosen so that Jᵃ(qⁱⁿ⁻) reproduces the flux the similarity solver
##### actually returned. The linearized balance then has the
##### two-conductances-in-series solution
#####
#####     Gᵉ (qᵉ - qⁱⁿ) = Gᵃ (qⁱⁿ - qᵃᵗ)
#####
#####     ⇒  qⁱⁿ = (Gᵉ qᵉ + Gᵃ qᵃᵗ) / (Gᵉ + Gᵃ).
#####
##### This is the standard series solution of a surface flux balance —
##### eq. (12b) of Ye & Pielke (1993) with their pore relative
##### humidity hₛ = 1, and the same expression CLM5/ClimaLand evaluate with a
##### prescribed exchange coefficient in place of Gᵃ. Substituting
##### Gᵃ = Jᵃ/Δq and multiplying numerator and denominator by Δq removes the
##### 0/0 ambiguity of Gᵃ as Δq → 0:
#####
#####     qⁱⁿ = (Gᵉ qᵉ Δq + Jᵃ qᵃᵗ) / (Gᵉ Δq + Jᵃ),
#####
##### the form coded below (denominator `D = Gᵉ Δq + Jᵃ`; if `D == 0` the
##### previous iterate is returned unchanged). Note Δq → 0 means qⁱⁿ⁻ = qᵃᵗ,
##### and the update then returns qᵃᵗ — the two statements agree, so the
##### limit is continuous.
#####
##### Limits worth checking: δᵛ → 0 gives Gᵉ → ∞ and qⁱⁿ → qᵉ, the saturated
##### skin of the wet branch; a deep front gives Gᵉ → 0 and qⁱⁿ → qᵃᵗ, i.e.
##### vanishing evaporation. At the Picard fixed point qⁱⁿ = qⁱⁿ⁻ the
##### linearization is exact, so the converged humidity satisfies the true
##### nonlinear balance Gᵉ (qᵉ - qⁱⁿ) = -ρᵃᵗ u★ q★(qⁱⁿ).
#####
# Dry-layer flux terms, split off so the standalone formulation and the
# composite (soil + canopy) share them. Returns the dry-layer conductance `Gᵉ`,
# the front (dry-branch) source humidity `qᵉ = qᵛ⁺(Tᵉ)`, the wet-branch logistic
# weight `σ`, and the wet (saturated-skin) humidity `qⁱⁿ⁺ = qᵛ⁺(Tⁱⁿ)`. The full
# humidity is `(1 − σ) qⁱⁿ⁺ + σ · [Δq-series divider with (Gᵉ, qᵉ)]`.
@inline function dry_layer_terms(q::DryLayerHumidity, Tⁱⁿ, Ψₛ, Ψₐ, ℙₐ)
    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    FT  = eltype(Ψₛ)
    pᵃᵗ = Ψₐ.p
    Tᵃᵗ = Ψₐ.T
    qᵃᵗ = Ψₐ.q
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)

    Tˡᵃ = Ψₛ.energy.temperature       # bulk land
    𝒮   = Ψₛ.hydrology.saturation     # surface saturation

    # Dry-layer depth, front temperature, and front (source) humidity
    # qᵉ = qᵛ⁺(Tᵉ) — the saturation specific humidity at the front.
    δᵛ    = dry_layer_depth(q.dry_layer_depth, 𝒮)
    δᵛmin = convert(FT, q.vapor_exchange.minimum_dry_layer_depth)
    ℓᵀ    = convert(FT, q.thermal_exchange_depth)
    χ     = clamp(δᵛ / ℓᵀ, zero(FT), one(FT))
    Tᵉ    = Tⁱⁿ + χ * (Tˡᵃ - Tⁱⁿ)
    qᵉ    = saturation_specific_humidity(ℂᵃᵗ, Tᵉ, pᵃᵗ, q.phase)

    # Dry-layer conductance. The actual pore liquid fraction is
    # θˡ = 𝒮(ν − θʳ) + θʳ; we use the simpler θˡ ≈ 𝒮·ν (the residual is
    # small and θˡ only enters the tortuosity scaling).
    θˡ  = 𝒮 * convert(FT, q.porosity)
    Dᵛ  = effective_vapor_diffusivity(q.vapor_exchange, q.porosity, θˡ)
    Gᵉ  = ρᵃᵗ * Dᵛ / max(δᵛ, δᵛmin)

    # Wet branch: the front co-locates with the skin, which saturates. The wet
    # limit is not the δᵛ → 0 limit of the series solution (Millington-Quirk
    # tortuosity closes the Fick path entirely at saturation), so the branches
    # are combined with a smooth logistic weight after Kavetski & Kuczera (2007).
    qⁱⁿ⁺ = saturation_specific_humidity(ℂᵃᵗ, Tⁱⁿ, pᵃᵗ, q.phase)
    δᵛʷ  = convert(FT, q.vapor_exchange.wet_transition_width)
    z    = 10 * (δᵛ - δᵛmin - δᵛʷ / 2) / max(δᵛʷ, eps(FT))
    σ    = 1 / (1 + exp(-z))

    return Gᵉ, qᵉ, σ, qⁱⁿ⁺
end

@inline function compute_interface_humidity(q::DryLayerHumidity, Tₛ, Ψₛ, Ψₐ, Ψᵢ, ℙₐ)
    FT = eltype(Ψₛ)
    Gᵉ, qᵉ, σ, qⁱⁿ⁺ = dry_layer_terms(q, Tₛ, Ψₛ, Ψₐ, ℙₐ)

    qⁱⁿ⁻ = Ψₛ.specific_humidity
    qᵃᵗ  = Ψₐ.q
    Jᵃ, Δq = atmospheric_vapor_flux(Ψₛ, Ψₐ, ℙₐ.thermodynamics_parameters)

    # Δq-multiplied series solution qⁱⁿ = (Gᵉ qᵉ + Gᵃ qᵃᵗ)/(Gᵉ + Gᵃ);
    # see the derivation in the banner above.
    D    = Gᵉ * Δq + Jᵃ
    qⁱⁿ★ = ifelse(D == 0, qⁱⁿ⁻, (Gᵉ * qᵉ * Δq + Jᵃ * qᵃᵗ) / D)

    return convert(FT, qⁱⁿ⁺ + σ * (qⁱⁿ★ - qⁱⁿ⁺))
end
