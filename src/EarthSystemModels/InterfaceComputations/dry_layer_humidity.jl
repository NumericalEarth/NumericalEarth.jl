#####
##### `DryLayerHumidity` — atmosphere-facing specific humidity from a
##### dry-layer vapor balance against an unresolved dry layer.
#####
##### The atmosphere-facing humidity `qⁱⁿ` is solved by the existing
##### `compute_interface_state` fixed point. Vapor diffuses up from an
##### dry layer at depth `δᵛ(𝒮)` below the surface; the source humidity
##### is the saturation value at the *front* temperature
##### `Tᵉ = Tⁱⁿ + χ(Tˡᵃ − Tⁱⁿ)` with `χ = clip(δᵛ/ℓᵀ, 0, 1)`:
#####
#####     Jᵉ→ⁱⁿ = Gᵉ (qᵉ - qⁱⁿ),         Gᵉ = ρᵃᵗ Dᵛ_eff / max(δᵛ, δᵛ_min),
#####     Jⁱⁿ→ᵃ = ρᵃᵗ wᵛ (qⁱⁿ - qᵃᵗ)    (atmospheric side, from u★ q★)
#####
##### closed by `Jᵉ→ⁱⁿ = Jⁱⁿ→ᵃ`. The wet branch (`δᵛ ≤ δᵛ_min`) collapses to
##### `qⁱⁿ = qᵛ⁺(Tⁱⁿ)` so the saturated-surface limit reproduces the existing
##### similarity-theory behavior.
#####
##### Pair this with `SkinTemperature(DiffusiveFlux(δ=ℓᵀ, κ=κᵀ))` on the
##### temperature side: the same `Λⁱⁿ = κᵀ/ℓᵀ` couples the bulk land temperature
##### `Tˡᵃ` to the skin temperature `Tⁱⁿ` and the energy fluxes
##### (`𝒬ᴿ + 𝒬ᵀ + 𝒬ᵛ`) — this PR doesn't need a new temperature formulation.
#####

using Oceananigans: Oceananigans
using Oceananigans.Utils: prettysummary
using Thermodynamics: Thermodynamics as AtmosphericThermodynamics

#####
##### Dry-layer depth diagnostics
#####

"""
    StorageBasedDryLayerDepth(maximum_dry_layer_depth, critical_saturation,
                                      dry_layer_exponent)

Diagnostic dry-layer depth `δᵛ` as a function of land saturation `𝒮`:

```math
\\delta^v(\\mathcal S) = \\delta^v_{max}\\,
                         \\left[1 - \\min(\\mathcal S/\\mathcal S^c, 1)\\right]^\\eta.
```

`δᵛ = 0` when `𝒮 ≥ 𝒮ᶜ` (wet branch), growing toward `δᵛ_max` as the slab dries.
"""
struct StorageBasedDryLayerDepth{FT}
    maximum_dry_layer_depth   :: FT
    critical_saturation   :: FT
    dry_layer_exponent  :: FT
end

StorageBasedDryLayerDepth(FT::Type = Oceananigans.defaults.FloatType;
                                  maximum_dry_layer_depth,
                                  critical_saturation,
                                  dry_layer_exponent = 2) =
    StorageBasedDryLayerDepth(convert(FT, maximum_dry_layer_depth),
                                      convert(FT, critical_saturation),
                                      convert(FT, dry_layer_exponent))

@inline function dry_layer_depth(d::StorageBasedDryLayerDepth, 𝒮)
    FT = typeof(𝒮)
    s  = min(𝒮 / convert(FT, d.critical_saturation), one(FT))
    return convert(FT, d.maximum_dry_layer_depth) *
           max(one(FT) - s, zero(FT))^convert(FT, d.dry_layer_exponent)
end

Base.summary(d::StorageBasedDryLayerDepth) =
    string("StorageBasedDryLayerDepth(δᵛ_max=", prettysummary(d.maximum_dry_layer_depth),
           ", 𝒮ᶜ=", prettysummary(d.critical_saturation),
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
                                tortuosity_model = ConstantTortuosity())

Parameters of the dry-layer vapor piston velocity `wᵈ = Dᵛ_eff / max(δᵛ, δᵛ_min)`.
The tortuosity model is a singleton type — [`ConstantTortuosity`](@ref) or
[`MillingtonQuirk`](@ref) — dispatched on by
`effective_vapor_diffusivity`.
"""
struct DryLayerVaporPistonVelocity{FT, T}
    minimum_dry_layer_depth   :: FT
    molecular_diffusivity :: FT
    tortuosity_model      :: T
end

DryLayerVaporPistonVelocity(FT::Type = Oceananigans.defaults.FloatType;
                            minimum_dry_layer_depth,
                            molecular_diffusivity,
                            tortuosity_model = ConstantTortuosity()) =
    DryLayerVaporPistonVelocity(convert(FT, minimum_dry_layer_depth),
                                convert(FT, molecular_diffusivity),
                                tortuosity_model)

Base.summary(v::DryLayerVaporPistonVelocity) =
    string("DryLayerVaporPistonVelocity(δᵛ_min=", prettysummary(v.minimum_dry_layer_depth),
           ", Dᵛ₀=", prettysummary(v.molecular_diffusivity),
           ", tortuosity=", summary(v.tortuosity_model), ")")

#####
##### Water activity
#####

"""
    UnitWaterActivity()

Trivial water-activity model: `aᵉ ≡ 1`. The dry-layer source humidity
is the saturation value at the front temperature.
"""
struct UnitWaterActivity end

@inline water_activity(::UnitWaterActivity, Π, Tᵉ) = one(Tᵉ)

Base.summary(::UnitWaterActivity) = "UnitWaterActivity"

#####
##### DryLayerHumidity — the humidity formulation
#####

"""
    DryLayerHumidity(phase = AtmosphericThermodynamics.Liquid();
                             dry_layer_depth,
                             vapor_exchange,
                             thermal_exchange_depth,
                             porosity,
                             water_activity = UnitWaterActivity())

Surface specific-humidity formulation for the *dry-layer* model:
`qⁱⁿ` is solved from a vapor-flux balance between a dry-layer Fick flux from
an unresolved dry layer and the atmospheric vapor flux. The
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
* `water_activity` — `aᵉ` model (only [`UnitWaterActivity`](@ref) in this
  PR; a `MatricPotentialActivity` model — matric-suction-driven vapor-pressure
  reduction via the Kelvin equation, only relevant at extreme dryness — is a
  follow-up).
"""
struct DryLayerHumidity{EFD, VEX, FT, A, Φ}
    dry_layer_depth :: EFD
    vapor_exchange          :: VEX
    thermal_exchange_depth  :: FT
    porosity                :: FT
    water_activity          :: A
    phase                   :: Φ
end

DryLayerHumidity(phase = AtmosphericThermodynamics.Liquid();
                         dry_layer_depth,
                         vapor_exchange,
                         thermal_exchange_depth,
                         porosity,
                         water_activity = UnitWaterActivity()) =
    DryLayerHumidity(dry_layer_depth,
                             vapor_exchange,
                             convert(Oceananigans.defaults.FloatType, thermal_exchange_depth),
                             convert(Oceananigans.defaults.FloatType, porosity),
                             water_activity,
                             phase)

Base.summary(q::DryLayerHumidity{EFD, VEX, FT, A, Φ}) where {EFD, VEX, FT, A, Φ} =
    string("DryLayerHumidity{",
           Φ === AtmosphericThermodynamics.Liquid ? "Liquid" : "Ice",
           "}(depth=", summary(q.dry_layer_depth),
           ", vapor=", summary(q.vapor_exchange),
           ", ℓᵀ=", prettysummary(q.thermal_exchange_depth),
           ", ν=", prettysummary(q.porosity),
           ", activity=", summary(q.water_activity), ")")
Base.show(io::IO, q::DryLayerHumidity) = print(io, summary(q))

#####
##### Effective vapor diffusivity (tortuosity)
#####
##### Dispatched on the tortuosity-model singleton so the per-cell call is
##### compile-time-resolved (no runtime `if` branch inside the kernel path).
#####

@inline effective_vapor_diffusivity(v::DryLayerVaporPistonVelocity, ν, θˡ, Tᵉ, pᵃᵗ) =
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
##### Humidity solver — the headline of the closure.
#####
##### Sign convention for fluxes here matches `SkinHumidity` exactly: each
##### `Jᵃ = -ρᵃᵗ u★ q★` (a positive Jᵃ means vapor moves upward into the
##### atmosphere; q★ < 0 when evaporating). The dry-layer flux Jᵉ from the
##### front up to the interface is taken as `Gᵉ(qᵉ - qⁱⁿ)`, so a wetter front
##### (qᵉ > qⁱⁿ) drives vapor upward. The balance Jᵉ = Jᵃ closes for qⁱⁿ.
#####
@inline function compute_interface_humidity(q::DryLayerHumidity, Tₛ, Ψₛ, Ψₐ, Ψᵢ, ℙₐ)
    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    FT  = eltype(Ψₛ)
    pᵃᵗ = Ψₐ.p
    qᵃᵗ = Ψₐ.q
    Tᵃᵗ = Ψₐ.T
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)

    Tˡᵃ = Ψₛ.energy.temperature       # bulk land
    𝒮   = Ψₛ.hydrology.saturation     # surface saturation
    Tⁱⁿ = Tₛ                           # current iterate of the skin temp

    # Dry-layer depth and temperature.
    δᵛ    = dry_layer_depth(q.dry_layer_depth, 𝒮)
    δᵛmin = convert(FT, q.vapor_exchange.minimum_dry_layer_depth)
    ℓᵀ    = convert(FT, q.thermal_exchange_depth)
    χ     = clamp(δᵛ / ℓᵀ, zero(FT), one(FT))
    Tᵉ    = Tⁱⁿ + χ * (Tˡᵃ - Tⁱⁿ)

    # Wet branch: front sits at the surface, skin is saturated.
    qˢᵃᵗ = saturation_specific_humidity(ℂᵃᵗ, Tᵉ, pᵃᵗ, q.phase)
    if δᵛ <= δᵛmin
        # Saturate the *skin* (which co-locates with the front when δᵛ=0).
        qˢᵃᵗ_skin = saturation_specific_humidity(ℂᵃᵗ, Tⁱⁿ, pᵃᵗ, q.phase)
        return convert(FT, qˢᵃᵗ_skin)
    end

    # Dry branch: vapor activity at the front, dry-layer piston velocity,
    # atmospheric vapor flux from the previous iterate.
    Π   = zero(FT)                      # placeholder; UnitWaterActivity ignores it
    aᵉ  = water_activity(q.water_activity, Π, Tᵉ)
    qᵉ  = aᵉ * qˢᵃᵗ

    # Actual pore liquid fraction is θˡ = 𝒮(ν − θʳ) + θʳ; for the tortuosity
    # model we use the simpler θˡ ≈ 𝒮·ν (residual is small, this only enters
    # the tortuosity scaling).
    θˡ  = 𝒮 * convert(FT, q.porosity)
    Dᵛ  = effective_vapor_diffusivity(q.vapor_exchange, q.porosity, θˡ, Tᵉ, pᵃᵗ)
    wᵈ  = Dᵛ / max(δᵛ, δᵛmin)
    Gᵉ  = ρᵃᵗ * wᵈ

    # Atmospheric flux from previous iterate.
    u★  = Ψₛ.fluxes.u★
    q★  = Ψₛ.fluxes.q★
    qⁱⁿ⁻ = Ψₛ.specific_humidity
    Jᵃ   = -ρᵃᵗ * u★ * q★               # positive upward
    Δq   = qⁱⁿ⁻ - qᵃᵗ

    # Same Δq-multiplied form as SkinHumidity to stay finite when Δq → 0:
    #   qⁱⁿ (Gᵉ Δq + Jᵃ) = Gᵉ qᵉ Δq + Jᵃ qᵃᵗ
    D    = Gᵉ * Δq + Jᵃ
    qⁱⁿ★ = (Gᵉ * qᵉ * Δq + Jᵃ * qᵃᵗ) / D
    return convert(FT, ifelse(D == 0, qⁱⁿ⁻, qⁱⁿ★))
end
