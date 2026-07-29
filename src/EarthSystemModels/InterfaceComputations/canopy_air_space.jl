#####
##### `CanopyAirSpace` — a two-source canopy with a diagnostic canopy-air node.
#####
##### The canopy and the soil surface exchange with a massless **canopy-air node**
##### `(Tᵃᶜ, qᵃᶜ)` that drains to the atmosphere through the aerodynamic conductance,
##### exactly the network the humidity side already builds in `CompositeSurfaceHumidity`
##### — now applied symmetrically to temperature. Three diagnostic scalars are solved
##### inside the Monin–Obukhov fixed point:
#####
#####   Tᵛ  — leaf temperature      (massless leaf: Rₙᵛ = Hᵛ + LEᵛ)
#####   Tⁱⁿ — soil-skin temperature (Rₙᵍ = Hᵍ + LEᵍ + Λⁱⁿ(Tⁱⁿ − Tˡᵃ), conducts to the bulk)
#####   Tᵃᶜ — canopy-air node       (Kirchhoff flux continuity; what MOST sees)
#####
##### and the paired humidity node `qᵃᶜ`. The leaf sees the *shaded soil skin* `Tⁱⁿ`,
##### not the bulk reservoir `Tˡᵃ`; the slab is driven only by the skin conduction.
#####
##### Reuse: `canopy_conductance_terms` (leaf vapor conductance `gˡʷ = g_c` and
##### `qᵛ⁺(Tᵛ)`, the Farquhar–Medlyn stomatal path) and `dry_layer_terms` (soil vapor
##### conductance `gᵍʷ = Gᵉ` and the front humidity `qᵉ`) are the *same* helpers the
##### standalone/composite humidity formulations use — the CAS only adds the sensible
##### conductances, the two-face longwave ledger, the Beer–Lambert shortwave split, and
##### the coupled solve. Grounded in ClimaLand (Deck et al. 2026, App. D2/D5, E3).
#####
##### A `CanopyAirSpace` is a *combined* formulation: pass the same object as both
##### `atmosphere_land_interface_temperature` and `atmosphere_land_interface_specific_humidity`.
##### `compute_interface_temperature` returns `Tᵃᶜ`; `compute_interface_humidity` returns
##### `qᵃᶜ`; both run the shared `canopy_air_space_solve`.
#####

# Sensible-heat analogue of `atmospheric_vapor_flux`: the atmospheric sensible flux
# `𝒬ᵀ = -ρᵃᵗ cᵖ u★ θ★` (positive upward) from the previous iterate, and the
# node-to-air temperature increment `Δθ = Tᵃᶜ⁻ − θᵃᵗ`. Together they close the
# temperature node in the same `Δ`-multiplied form the humidity node uses.
@inline function atmospheric_sensible_flux(Ψₛ, Ψₐ, θᵃᵗ, ℂᵃᵗ)
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Ψₐ.T, Ψₐ.p, Ψₐ.q)
    cᵖ  = AtmosphericThermodynamics.cp_m(ℂᵃᵗ, Ψₐ.q)
    𝒬ᵀ  = - ρᵃᵗ * cᵖ * Ψₛ.fluxes.u★ * Ψₛ.fluxes.θ★
    Δθ  = Ψₛ.temperature - θᵃᵗ
    return 𝒬ᵀ, Δθ
end

"""
    struct CanopyInterception

Marker enabling the wet-canopy (interception) vapor branch of a [`CanopyAirSpace`](@ref). A wet
canopy evaporates intercepted water at the *potential* (stomata-free) rate through
the leaf boundary layer only, so the leaf vapor conductance blends the dry
(stomatal) `g_c` with a wet `g_wet = ρᵃᵗ · LAI · gᵇ` by the wet fraction

```math
f_{wet} = (Wᶜ / Wᶜᵐᵃˣ)^{2/3}, \\qquad Wᶜᵐᵃˣ = c · LAI
```

([Deardorff, 1978](@cite deardorff1978)). The store `Wᶜ` and its capacity `Wᶜᵐᵃˣ = c·LAI`
are owned by the [`InterceptingHydrology`](@ref) wrapping the soil; the interface reads both
and normalizes `f_wet` by the store's *own* capacity. The leaf boundary conductance `gᵇ` is the
`leaf_boundary_conductance` on the [`CanopyAirSpace`](@ref).
"""
struct CanopyInterception end

Base.summary(::CanopyInterception) = "CanopyInterception"

# Deardorff (1978) wet fraction f_wet = (Wᶜ/Wᶜᵐᵃˣ)^(2/3), normalized by the store's own
# capacity Wᶜᵐᵃˣ (published by `InterceptingHydrology`). No interception ⇒ 0, recovering the
# dry CAS bit-for-bit; a zero capacity (no store, or a bare tile) also gives 0.
@inline wet_canopy_fraction(::Nothing, hydrology, LAI) = zero(LAI)
@inline function wet_canopy_fraction(::CanopyInterception, hydrology, LAI)
    FT    = typeof(LAI)
    Wᶜ    = convert(FT, hydrology.canopy_water_storage)
    Wᶜᵐᵃˣ = convert(FT, hydrology.canopy_water_capacity)
    return ifelse(Wᶜᵐᵃˣ > zero(FT),
                  clamp((max(Wᶜ, zero(FT)) / Wᶜᵐᵃˣ)^convert(FT, 2//3), zero(FT), one(FT)),
                  zero(FT))
end

"""
    struct CanopyAirSpace

Two-source canopy + soil surface with a diagnostic canopy-air node. Solves the
leaf temperature `Tᵛ`, the soil-skin temperature `Tⁱⁿ`, and the canopy-air node
`(Tᵃᶜ, qᵃᶜ)` inside the Monin–Obukhov fixed point. Use the same object in both the
temperature and specific-humidity interface slots.

Fields:
- `soil`   : the soil vapor branch (a [`DryLayerHumidity`](@ref)).
- `canopy` : the leaf vapor/photosynthesis branch (a [`CanopyConductanceHumidity`](@ref)).
- `soil_skin_flux` : skin↔bulk conduction `Λⁱⁿ = κᵀ/ℓᵀ` (a [`SoilConductiveFlux`](@ref)).
- `leaf_albedo`, `ground_albedo` : broadband shortwave albedos.
- `canopy_emissivity_max`, `ground_emissivity` : longwave emissivities (`ε_c = ε_max(1 − e^{−LAI})`).
- `extinction`, `clumping` : Beer–Lambert `K`, `Ω` for the shortwave split.
- `leaf_boundary_conductance` : per-leaf boundary-layer conductance `gᵇ` (m s⁻¹) → `gˡʰ = ρcₚ·LAI·gᵇ`.
- `undercanopy_conductance` : ground↔canopy-air conductance (m s⁻¹) → `gᵍʰ = ρcₚ·gᵘᶜ`.
- `inner_iterations`, `relaxation` : damped-Newton settings for the coupled solve.
- `interception` : wet-canopy vapor branch parameters (a [`CanopyInterception`](@ref)),
  or `nothing` for a dry canopy (the default; recovers the current CAS bit-for-bit).
- `phase` : saturation phase (Liquid).
"""
struct CanopyAirSpace{S, C, RF, FT, I, Φ}
    soil                      :: S
    canopy                    :: C
    soil_skin_flux            :: RF
    leaf_albedo               :: FT
    ground_albedo             :: FT
    canopy_emissivity_max     :: FT
    ground_emissivity         :: FT
    extinction                :: FT
    clumping                  :: FT
    leaf_boundary_conductance :: FT
    undercanopy_conductance   :: FT
    inner_iterations          :: Int
    relaxation                :: FT
    interception              :: I
    phase                     :: Φ
end

function CanopyAirSpace(FT=Oceananigans.defaults.FloatType;
                        soil,
                        canopy                    = CanopyConductanceHumidity(FT),
                        soil_skin_flux            = SoilConductiveFlux(1.5, 0.05),
                        leaf_albedo               = 0.15,
                        ground_albedo             = 0.15,
                        canopy_emissivity_max     = 0.98,
                        ground_emissivity         = 0.96,
                        extinction                = 0.5,
                        clumping                  = 1,
                        leaf_boundary_conductance = 0.02,
                        undercanopy_conductance   = 0.013,
                        inner_iterations          = 40,
                        relaxation                = 1//2,
                        interception              = nothing,
                        phase                     = AtmosphericThermodynamics.Liquid())

    return CanopyAirSpace(soil, canopy, soil_skin_flux,
                          convert(FT, leaf_albedo), convert(FT, ground_albedo),
                          convert(FT, canopy_emissivity_max), convert(FT, ground_emissivity),
                          convert(FT, extinction), convert(FT, clumping),
                          convert(FT, leaf_boundary_conductance),
                          convert(FT, undercanopy_conductance),
                          inner_iterations, convert(FT, relaxation), interception, phase)
end

Base.summary(::CanopyAirSpace) = "CanopyAirSpace"
Base.show(io::IO, c::CanopyAirSpace) =
    print(io, "CanopyAirSpace(soil=", summary(c.soil), ", canopy=", summary(c.canopy), ")")

# Materialization / identity — delegate to the sub-models so the per-cell interface
# state carries the soil saturation, bulk temperature, and LAI the branches read.
@inline interface_phase(c::CanopyAirSpace) = interface_phase(c.soil)
# The soil branch always publishes the saturation 𝒮; a canopy with interception
# additionally pulls the prognostic canopy water store Wᶜ (→ f_wet).
@inline interface_hydrology_state(i, j, grid, c::CanopyAirSpace, land_state) =
    canopy_air_space_hydrology_state(c.interception, i, j, grid, c, land_state)
@inline canopy_air_space_hydrology_state(::Nothing, i, j, grid, c, land_state) =
    interface_hydrology_state(i, j, grid, c.soil, land_state)
@inline canopy_air_space_hydrology_state(::CanopyInterception, i, j, grid, c, land_state) =
    merge(interface_hydrology_state(i, j, grid, c.soil, land_state),
          (canopy_water_storage  = land_field_value(land_state.canopy_water_storage, i, j),
           canopy_water_capacity = land_field_value(land_state.canopy_water_capacity, i, j)))
@inline interface_energy_state(i, j, grid, c::CanopyAirSpace, land_state) =
    interface_energy_state(i, j, grid, c.soil, land_state)
@inline canopy_leaf_area_index(c::CanopyAirSpace) = canopy_leaf_area_index(c.canopy)
@inline interface_vegetation_state(i, j, grid, c::CanopyAirSpace, vegetation, time_interpolator) =
    interface_vegetation_state(i, j, grid, c.canopy, vegetation, time_interpolator)

# dqᵛ⁺/dT by centered difference — the Newton derivative of each balance's latent term.
@inline function saturation_humidity_slope(ℂᵃᵗ, T, pᵃᵗ, phase)
    δ = convert(typeof(T), 1//100)
    q⁺ = saturation_specific_humidity(ℂᵃᵗ, T + δ, pᵃᵗ, phase)
    q⁻ = saturation_specific_humidity(ℂᵃᵗ, T - δ, pᵃᵗ, phase)
    return (q⁺ - q⁻) / 2δ
end

"""
    canopy_air_space_solve(c::CanopyAirSpace, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)

Solve the coupled diagnostic state `(Tᵛ, Tⁱⁿ, Tᵃᶜ, qᵃᶜ)` for one cell. `Ψₛ` is the
previous fixed-point iterate (carrying the MO scales and the previous node values),
`Ψᵢ.T` is the bulk reservoir `Tˡᵃ`, and `Ψᵣ` the interface radiation state. A short
damped-Newton inner loop advances the two skin balances against the node; the node
uses the `Δ`-multiplied Kirchhoff form so it stays finite as the flux vanishes.
"""
@inline function canopy_air_space_solve(c::CanopyAirSpace, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    FT  = eltype(Ψₛ)
    pᵃᵗ = Ψₐ.p
    qᵃᵗ = Ψₐ.q
    Tᵃᵗ = Ψₐ.T
    ℒ   = AtmosphericThermodynamics.latent_heat_vapor(ℂᵃᵗ, Tᵃᵗ)
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)
    cᵖ  = AtmosphericThermodynamics.cp_m(ℂᵃᵗ, qᵃᵗ)
    θᵃᵗ = surface_atmosphere_temperature(Ψₐ, ℙₐ)

    Tˡᵃ = Ψᵢ.T
    LAI = Ψₛ.vegetation.leaf_area_index

    # Aerodynamic drivers from the previous outer iterate (held fixed through the inner loop).
    𝒬ᵀ, Δθᵃ = atmospheric_sensible_flux(Ψₛ, Ψₐ, θᵃᵗ, ℂᵃᵗ)
    Jᵃ, Δqᵃ = atmospheric_vapor_flux(Ψₛ, Ψₐ, ℂᵃᵗ)

    gˡʰ = ρᵃᵗ * cᵖ * LAI * c.leaf_boundary_conductance
    gᵍʰ = ρᵃᵗ * cᵖ * c.undercanopy_conductance
    gᵍʷ = ρᵃᵗ * c.undercanopy_conductance   # undercanopy vapor conductance (wet-soil limit)
    Λ   = convert(FT, skin_conductance(c.soil_skin_flux))

    # Wet-canopy vapor branch. `f_wet` (Deardorff 1978) blends the dry stomatal
    # conductance `g_c` with the stomata-free wet-leaf conductance `g_wet = ρᵃᵗ·LAI·gᵇ`
    # (the boundary-layer vapor mass conductance), so intercepted water evaporates at the
    # potential rate through the leaf boundary layer. `f_wet = 0` (no interception)
    # recovers the dry CAS bit-for-bit.
    f_wet = wet_canopy_fraction(c.interception, Ψₛ.hydrology, LAI)
    g_wet = ρᵃᵗ * LAI * c.leaf_boundary_conductance

    # Shortwave split + longwave emissivities (broadband).
    σ   = Ψᵣ.σ
    SW  = Ψᵣ.ℐꜜˢʷ
    LWd = Ψᵣ.ℐꜜˡʷ
    ε_c = c.canopy_emissivity_max * (1 - exp(-LAI))
    ε_g = c.ground_emissivity
    ftrans    = exp(-c.extinction * LAI * c.clumping)
    canopy_SW = (1 - c.leaf_albedo) * (1 - ftrans) * SW
    ground_SW = ftrans * (1 - c.ground_albedo) * SW

    Tᵛ  = Tˡᵃ
    Tⁱⁿ = Tˡᵃ
    Tᵃᶜ = Ψₛ.temperature
    qᵃᶜ = Ψₛ.specific_humidity
    relax  = c.relaxation
    max_ΔT = convert(FT, 25)   # per-iterate step cap: keeps the damped Newton in-range
    Tₗₒ, Tₕᵢ = convert(FT, 180), convert(FT, 340)  # physical band; guards qˢᵃᵗ against transient overshoot
    tiny = eps(FT)

    for _ in 1:c.inner_iterations
        g_c, qᵛ   = canopy_conductance_terms(c.canopy, Tᵛ, Ψₛ, Ψₐ, Ψᵣ, ℙₐ)
        Gᵉ, qᵉ, f_dry, qⁱⁿ⁺ = dry_layer_terms(c.soil, Tⁱⁿ, Ψₛ, Ψₐ, ℙₐ)

        # Blend the dry-layer series soil branch (front qᵉ through Gᵉ) with the
        # saturated-skin wet branch (qⁱⁿ⁺ through the undercanopy conductance gᵍʷ),
        # weight `f_dry` from the soil model.
        Gᵉ⁺ = f_dry * Gᵉ + (1 - f_dry) * gᵍʷ
        qᵉ  = ifelse(Gᵉ⁺ > tiny, (f_dry * Gᵉ * qᵉ + (1 - f_dry) * gᵍʷ * qⁱⁿ⁺) / Gᵉ⁺, qⁱⁿ⁺)
        Gᵉ  = Gᵉ⁺

        # Blended leaf vapor conductance: dry (stomatal) g_c over the transpiring
        # fraction, wet (boundary-layer) g_wet over the wetted fraction f_wet.
        g_leaf = (1 - f_wet) * g_c + f_wet * g_wet

        # Δ-multiplied Kirchhoff node (as the humidity node in CompositeSurfaceHumidity);
        # guard the transient case where the aerodynamic and surface conductances cancel
        # (Dᵀ ≈ 0) before the outer MO loop is consistent, keeping the node finite.
        Dᵀ  = (gᵍʰ + gˡʰ) * Δθᵃ + 𝒬ᵀ
        Tᵃᶜ★ = ((gᵍʰ * Tⁱⁿ + gˡʰ * Tᵛ) * Δθᵃ + 𝒬ᵀ * θᵃᵗ) / Dᵀ
        Tᵃᶜ = ifelse((Dᵀ == 0) | !isfinite(Tᵃᶜ★), Tᵃᶜ, Tᵃᶜ★)
        Dᵠ  = (Gᵉ + g_leaf) * Δqᵃ + Jᵃ
        qᵃᶜ★ = ((Gᵉ * qᵉ + g_leaf * qᵛ) * Δqᵃ + Jᵃ * qᵃᵗ) / Dᵠ
        qᵃᶜ = ifelse((Dᵠ == 0) | !isfinite(qᵃᶜ★), qᵃᶜ, qᵃᶜ★)

        LWd_c     = (1 - ε_c) * LWd + ε_c * σ * Tᵛ^4
        LWu_g     = ε_g * σ * Tⁱⁿ^4 + (1 - ε_g) * LWd_c
        canopy_lw = ε_c * (LWd + LWu_g) - 2 * ε_c * σ * Tᵛ^4
        ground_lw = ε_g * (LWd_c - σ * Tⁱⁿ^4)

        Rᵥ   = canopy_SW + canopy_lw
        resᵥ = Rᵥ - gˡʰ * (Tᵛ - Tᵃᶜ) - ℒ * g_leaf * (qᵛ - qᵃᶜ)
        dRᵥ  = -8 * ε_c * σ * Tᵛ^3 - gˡʰ - ℒ * g_leaf * saturation_humidity_slope(ℂᵃᵗ, Tᵛ, pᵃᵗ, c.phase)
        Tᵛ   = ifelse(abs(dRᵥ) < tiny, Tᵃᶜ, Tᵛ - clamp(relax * resᵥ / dRᵥ, -max_ΔT, max_ΔT))
        Tᵛ   = clamp(Tᵛ, Tₗₒ, Tₕᵢ)

        Rᵍ   = ground_SW + ground_lw
        resᵍ = Rᵍ - gᵍʰ * (Tⁱⁿ - Tᵃᶜ) - ℒ * Gᵉ * (qᵉ - qᵃᶜ) - Λ * (Tⁱⁿ - Tˡᵃ)
        dRᵍ  = -4 * ε_g * σ * Tⁱⁿ^3 - gᵍʰ - Λ - ℒ * Gᵉ * saturation_humidity_slope(ℂᵃᵗ, Tⁱⁿ, pᵃᵗ, c.phase)
        Tⁱⁿ  = Tⁱⁿ - clamp(relax * resᵍ / dRᵍ, -max_ΔT, max_ΔT)
        Tⁱⁿ  = clamp(Tⁱⁿ, Tₗₒ, Tₕᵢ)
    end

    # Converged diagnostics: per-surface flux shares, the skin→slab conduction, and
    # the effective radiating (LST) temperature σ T_eff⁴ ≡ LWu (upwelling to space).
    g_c, qᵛ   = canopy_conductance_terms(c.canopy, Tᵛ, Ψₛ, Ψₐ, Ψᵣ, ℙₐ)
    Gᵉ, qᵉ, f_dry, qⁱⁿ⁺ = dry_layer_terms(c.soil, Tⁱⁿ, Ψₛ, Ψₐ, ℙₐ)
    Gᵉ⁺ = f_dry * Gᵉ + (1 - f_dry) * gᵍʷ
    qᵉ  = ifelse(Gᵉ⁺ > tiny, (f_dry * Gᵉ * qᵉ + (1 - f_dry) * gᵍʷ * qⁱⁿ⁺) / Gᵉ⁺, qⁱⁿ⁺)
    Gᵉ  = Gᵉ⁺
    g_leaf = (1 - f_wet) * g_c + f_wet * g_wet
    LWd_c = (1 - ε_c) * LWd + ε_c * σ * Tᵛ^4
    LWu_g = ε_g * σ * Tⁱⁿ^4 + (1 - ε_g) * LWd_c
    LWu   = (1 - ε_c) * LWu_g + ε_c * σ * Tᵛ^4
    Teff  = ifelse(σ > 0, (LWu / σ)^convert(FT, 1//4), Tᵃᶜ)

    Hᵛ    = gˡʰ * (Tᵛ - Tᵃᶜ)
    Hᵍ    = gᵍʰ * (Tⁱⁿ - Tᵃᶜ)
    LEᵛ   = ℒ * g_leaf * (qᵛ - qᵃᶜ)              # total leaf latent (transpiration + wet-canopy)
    LEᵍ   = ℒ * Gᵉ * (qᵉ - qᵃᶜ)
    Gcond = Λ * (Tⁱⁿ - Tˡᵃ)
    E_wet = f_wet * g_wet * (qᵛ - qᵃᶜ)           # wet-canopy evaporation, mass flux (kg m⁻² s⁻¹, up)
    LE_wet = ℒ * E_wet                           # wet-canopy latent heat (W m⁻², up); LEᵛ − LE_wet = transpiration

    return (; Tᵛ = convert(FT, Tᵛ), Tⁱⁿ = convert(FT, Tⁱⁿ),
              Tᵃᶜ = convert(FT, Tᵃᶜ), qᵃᶜ = convert(FT, qᵃᶜ),
              Teff = convert(FT, Teff),
              Hᵛ = convert(FT, Hᵛ), Hᵍ = convert(FT, Hᵍ),
              LEᵛ = convert(FT, LEᵛ), LEᵍ = convert(FT, LEᵍ),
              Gcond = convert(FT, Gcond), E_wet = convert(FT, E_wet),
              LE_wet = convert(FT, LE_wet))
end

@inline compute_interface_temperature(c::CanopyAirSpace,
                                      interface_state, atmosphere_state, interior_state,
                                      radiation_state, interface_properties,
                                      atmosphere_properties, interior_properties) =
    canopy_air_space_solve(c, interface_state, atmosphere_state, interior_state,
                           radiation_state, atmosphere_properties).Tᵃᶜ

@inline compute_interface_humidity(c::CanopyAirSpace, Tₛ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ) =
    canopy_air_space_solve(c, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ).qᵃᶜ

# Combined temperature + humidity: one shared solve returns both the canopy-air node
# temperature Tᵃᶜ and humidity qᵃᶜ, so the per-iterate inner solve runs once, not twice.
@inline function interface_temperature_and_humidity(c::CanopyAirSpace, ::CanopyAirSpace,
                                                    Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₛ, ℙₐ, ℙᵢ)
    sol = canopy_air_space_solve(c, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    return sol.Tᵃᶜ, sol.qᵃᶜ
end
