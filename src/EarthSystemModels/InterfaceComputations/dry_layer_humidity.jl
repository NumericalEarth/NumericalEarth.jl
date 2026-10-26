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
##### atmosphere-facing humidity `qⁱⁿ` is solved (by the
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
##### `Tᵉ = Tᵍʳᵒ + χ(Tˡᵃ − Tᵍʳᵒ)` with `χ = clip(δᵛ/ℓᵀ, 0, 1)`. The wet branch
##### (`δᵛ ≲ δᵛ_min`) collapses to `qⁱⁿ = qᵛ⁺(Tᵍʳᵒ)` so the saturated-surface
##### limit reproduces the existing similarity-theory behavior; it hands over
##### to the dry-layer series solution through a smooth logistic blend of
##### width `wet_transition_width` (sharp switch when 0).
#####
##### Pair this with `SoilSkinTemperature(κᵀ, ℓᵀ)` on the
##### temperature side: the same `Λᵍʳᵒ = κᵀ/ℓᵀ` couples the bulk land temperature
##### `Tˡᵃ` to the skin temperature `Tᵍʳᵒ`, and the front-temperature interpolation
##### `Tᵉ   = Tᵍʳᵒ + χ(Tˡᵃ − Tᵍʳᵒ)` then becomes live (with `BulkTemperature`,
##### `Tᵍʳᵒ = Tˡᵃ` and the χ term vanishes).
#####

#####
##### Dry-layer depth diagnostics
#####

"""
    StorageBasedDryLayerDepth(FT = Oceananigans.defaults.FloatType;
                              maximum_dry_layer_depth,
                              dry_layer_onset_saturation,
                              dry_layer_exponent = 2)

Diagnostic dry-layer depth `δᵛ` as a function of land saturation `𝒮`:

```math
δᵛ(𝒮) = δᵛ_{max} \\left[1 - \\min(𝒮 / 𝒮ᶜ, 1)\\right]^η
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
    PowerLawTortuosity()

Millington–Quirk tortuosity: `Dᵛ_eff = Dᵛ₀ · θᵍ^(10/3) / ν²` where
`θᵍ = ν − θˡ` is the gas-filled pore fraction. Reduces vapor diffusivity in
near-saturated soils.
"""
struct PowerLawTortuosity end

Base.summary(::ConstantTortuosity) = "ConstantTortuosity"
Base.summary(::PowerLawTortuosity) = "PowerLawTortuosity"

"""
    DryLayerVaporPistonVelocity(FT = Oceananigans.defaults.FloatType;
                                minimum_dry_layer_depth,
                                molecular_diffusivity,
                                wet_transition_width = 5 * minimum_dry_layer_depth,
                                tortuosity = ConstantTortuosity())

Parameters of the dry-layer vapor piston velocity `wᵈ = Dᵛ_eff / max(δᵛ, δᵛ_min)`,
the reciprocal of the dry-surface-layer soil resistance `r_soil = δᵛ/Dᵛ_eff` of
[Yamanaka et al. (1997)](@cite yamanaka1997surface) and
[Swenson and Lawrence (2014)](@cite swenson2014dry). The tortuosity model is a
singleton type — [`ConstantTortuosity`](@ref) or [`PowerLawTortuosity`](@ref),
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
    tortuosity              :: T
end

DryLayerVaporPistonVelocity(FT::Type = Oceananigans.defaults.FloatType;
                            minimum_dry_layer_depth,
                            molecular_diffusivity,
                            wet_transition_width = 5 * minimum_dry_layer_depth,
                            tortuosity = ConstantTortuosity()) =
    DryLayerVaporPistonVelocity(convert(FT, minimum_dry_layer_depth),
                                convert(FT, molecular_diffusivity),
                                convert(FT, wet_transition_width),
                                tortuosity)

Base.summary(v::DryLayerVaporPistonVelocity) =
    string("DryLayerVaporPistonVelocity(δᵛ_min=", prettysummary(v.minimum_dry_layer_depth),
           ", Dᵛ₀=", prettysummary(v.molecular_diffusivity),
           ", δᵛʷ=", prettysummary(v.wet_transition_width),
           ", tortuosity=", summary(v.tortuosity), ")")

#####
##### DryLayerHumidity — the humidity formulation
#####

"""
$(TYPEDSIGNATURES)

Surface humidity from vapor diffusion through a dry soil layer, following
[Ye and Pielke (1993)](@cite yepielke1993) with the dry-layer resistance of
[Swenson and Lawrence (2014)](@cite swenson2014dry). `dry_layer_depth` is the
moisture-dependent depth (m), and `vapor_exchange` specifies molecular transport.
Soil diffusivity is supplied by land; the thermal exchange distance comes from the
surface heat-transfer closure. Pore air at the evaporation front is saturated;
matric-suction suppression of its vapor pressure is neglected.
"""
struct DryLayerHumidity{D, V, P}
    dry_layer_depth :: D
    vapor_exchange :: V
    phase :: P
end

DryLayerHumidity(FT::Type=Oceananigans.defaults.FloatType,
                 phase=AtmosphericThermodynamics.Liquid(); dry_layer_depth, vapor_exchange) =
    DryLayerHumidity(dry_layer_depth, vapor_exchange, phase)

Base.summary(q::DryLayerHumidity{D, V, P}) where {D, V, P} =
    string("DryLayerHumidity{", P === AtmosphericThermodynamics.Liquid ? "Liquid" : "Ice",
           "}(depth=", summary(q.dry_layer_depth), ", vapor=", summary(q.vapor_exchange), ")")
Base.show(io::IO, q::DryLayerHumidity) = print(io, summary(q))

"""
    SoilVaporDiffusivity(vapor_exchange)

Land state accessor for the effective soil vapor diffusivity `Dᵛ` of `vapor_exchange`
(a [`DryLayerVaporPistonVelocity`](@ref)) at the land's own porosity and saturation. A land
that supports [`DryLayerHumidity`](@ref) implements `(d::SoilVaporDiffusivity)(land)`.
"""
struct SoilVaporDiffusivity{V}
    vapor_exchange :: V
end

land_state_accessors(q::DryLayerHumidity) =
    (T = land_temperature, 𝒮 = land_saturation, effective_vapor_diffusivity = SoilVaporDiffusivity(q.vapor_exchange))

#####
##### Effective vapor diffusivity (tortuosity)
#####
##### Dispatched on the tortuosity-model singleton so the per-cell call is
##### compile-time-resolved (no runtime `if` branch inside the kernel path).
#####

@inline effective_vapor_diffusivity(v::DryLayerVaporPistonVelocity, ν, θˡ) =
    effective_vapor_diffusivity(v.tortuosity, v.molecular_diffusivity, ν, θˡ)

@inline effective_vapor_diffusivity(::ConstantTortuosity, D₀, ν, θˡ) =
    convert(typeof(θˡ), D₀)

@inline function effective_vapor_diffusivity(::PowerLawTortuosity, D₀, ν, θˡ)
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
##### interface similarity theory carries vapor away at Jᵃ(Tᵍʳᵒ, qⁱⁿ). The
##### interface stores no vapor, so Jᵉ = Jᵃ — a nonlinear equation for qⁱⁿ.
#####
##### The solver. Over one Picard iteration we linearize the similarity flux
##### as a bulk conductance law,
#####
#####     Jᵃ(q) ≈ Gᵃ (q - qᵃᵗ),      Gᵃ = ρᵃᵗ u★ χq,
#####
##### with the friction velocity and vapor transfer coefficient of the previous
##### similarity iterate, so `Jᵃ(qⁱⁿ⁻)` reproduces the flux `-ρᵃᵗ u★ q★` the
##### similarity solver actually returned (`q★ = χq Δq`). The linearized balance
##### then has the two-conductances-in-series solution
#####
#####     Gᵉ (qᵉ - qⁱⁿ) = Gᵃ (qⁱⁿ - qᵃᵗ)
#####
#####     ⇒  qⁱⁿ = (Gᵉ qᵉ + Gᵃ qᵃᵗ) / (Gᵉ + Gᵃ),
#####
##### the form coded below (if every conductance vanishes the previous iterate
##### is returned unchanged). This is the standard series solution of a surface
##### flux balance — eq. (12b) of Ye & Pielke (1993) with their pore relative
##### humidity hₛ = 1, and the same expression CLM5/ClimaLand evaluate with a
##### prescribed exchange coefficient in place of Gᵃ.
#####
##### Limits worth checking: δᵛ → 0 gives Gᵉ → ∞ and qⁱⁿ → qᵉ, the saturated
##### skin of the wet branch; a deep front gives Gᵉ → 0 and qⁱⁿ → qᵃᵗ, i.e.
##### vanishing evaporation. At the Picard fixed point qⁱⁿ = qⁱⁿ⁻ the
##### linearization is exact, so the converged humidity satisfies the true
##### nonlinear balance Gᵉ (qᵉ - qⁱⁿ) = -ρᵃᵗ u★ q★(qⁱⁿ).
#####
# Dry-layer flux terms: the dry-layer conductance `Gᵉ`, the front (dry-branch) source
# humidity `qᵉ = qᵛ⁺(Tᵉ)`, the dry-branch logistic weight `fᵈᵇʳ`, the wet (saturated-skin)
# humidity `qᵍʳᵒ⁺ = qᵛ⁺(Tᵍʳᵒ)`, and the slopes `dqᵉ/dTᵍʳᵒ` and `dqᵍʳᵒ⁺/dTᵍʳᵒ`. The full humidity is
# `(1 − fᵈᵇʳ) qᵍʳᵒ⁺ + fᵈᵇʳ · [Δq-series divider with (Gᵉ, qᵉ)]`.
@inline function dry_layer_terms(q::DryLayerHumidity, Tᵍʳᵒ, Ψₛ, Ψₐ, Ψˡᵃ, ℙₐ)
    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    FT  = eltype(Ψₛ)
    pᵃᵗ = Ψₐ.p
    Tᵃᵗ = Ψₐ.T
    qᵃᵗ = Ψₐ.q
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)

    Tˡᵃ = Ψˡᵃ.T  # bulk land
    𝒮   = Ψˡᵃ.𝒮  # surface saturation

    # Dry-layer depth, front temperature, and front (source) humidity
    # qᵉ = qᵛ⁺(Tᵉ) — the saturation specific humidity at the front.
    δᵛ    = dry_layer_depth(q.dry_layer_depth, 𝒮)
    δᵛmin = convert(FT, q.vapor_exchange.minimum_dry_layer_depth)
    ℓᵀ    = convert(FT, Ψˡᵃ.thermal_exchange_depth)
    χ     = clamp(δᵛ / ℓᵀ, zero(FT), one(FT))
    Tᵉ    = Tᵍʳᵒ + χ * (Tˡᵃ - Tᵍʳᵒ)
    qᵉ    = saturation_specific_humidity(ℂᵃᵗ, Tᵉ, pᵃᵗ, q.phase)

    Dᵛ  = convert(FT, Ψˡᵃ.effective_vapor_diffusivity)
    Gᵉ  = ρᵃᵗ * Dᵛ / max(δᵛ, δᵛmin)

    # Wet branch: the front co-locates with the skin, which saturates. The wet
    # limit is not the δᵛ → 0 limit of the series solution (Millington-Quirk
    # tortuosity closes the Fick path entirely at saturation), so the branches
    # are combined with a smooth logistic weight after Kavetski & Kuczera (2007).
    qᵍʳᵒ⁺ = saturation_specific_humidity(ℂᵃᵗ, Tᵍʳᵒ, pᵃᵗ, q.phase)
    δᵛʷ  = convert(FT, q.vapor_exchange.wet_transition_width)
    z    = 10 * (δᵛ - δᵛmin - δᵛʷ / 2) / max(δᵛʷ, eps(FT))
    fᵈᵇʳ = 1 / (1 + exp(-z))

    # The front moves with the skin by (1 − χ) K per K.
    dqᵉ    = (1 - χ) * saturation_humidity_slope(ℂᵃᵗ, Tᵉ, pᵃᵗ, q.phase, qᵉ)
    dqᵍʳᵒ⁺ = saturation_humidity_slope(ℂᵃᵗ, Tᵍʳᵒ, pᵃᵗ, q.phase, qᵍʳᵒ⁺)

    return Gᵉ, qᵉ, fᵈᵇʳ, qᵍʳᵒ⁺, dqᵉ, dqᵍʳᵒ⁺
end

@inline function compute_interface_humidity(q::DryLayerHumidity, Tₛ, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ)
    FT = eltype(Ψₛ)
    Gᵉ, qᵉ, fᵈᵇʳ, qᵍʳᵒ⁺, _, _ = dry_layer_terms(q, Tₛ, Ψₛ, Ψₐ, Ψˡᵃ, ℙₐ)

    Gᵃ   = aerodynamic_vapor_conductance(Ψₛ, Ψₐ, ℙₐ.thermodynamics_parameters)
    qⁱⁿ★ = conductance_weighted_node(Ψₛ.specific_humidity, (Gᵉ, Gᵃ), (qᵉ, Ψₐ.q))

    return convert(FT, qᵍʳᵒ⁺ + fᵈᵇʳ * (qⁱⁿ★ - qᵍʳᵒ⁺))
end
