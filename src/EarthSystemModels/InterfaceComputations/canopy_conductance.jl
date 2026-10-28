#####
##### `CanopyConductanceHumidity` — a single-source (big-leaf) canopy: the stomatal
##### conductance `gᶜᵃⁿ = λˡᵉᵃᶠ · gₛ` in series with the aerodynamic conductance.
#####

"""
    struct CanopyConductanceHumidity

Surface specific humidity `qˢ` for a single-source (big-leaf) canopy: the
canopy conductance `gᶜᵃⁿ = λˡᵉᵃᶠ · gₛ` in series with the aerodynamic conductance,
solved inside the Monin–Obukhov fixed point exactly as [`SkinHumidity`](@ref)
solves a soil-resistance balance. The stomatal conductance `gₛ` comes from the
`conductance` model driven by the per-cell leaf-to-air VPD, leaf temperature
(`= Tₛ`, single-source), and absorbed PAR, with the moisture-stress factor `β(𝒮)`
read from the ground hydrology (`moisture_stress`, a `Number`, a
[`PlantAvailableWaterStress`](@ref) — the model meant for transpiration — or a
[`CriticalSaturation`](@ref)). The default conductance is
[`JarvisConductance`](@ref). Absorbed PAR is a prescribed `Number` or live from the
radiation state ([`AbsorbedPhotosyntheticallyActiveRadiation`](@ref)).

Fields:
- `leaf_area_index` : bulk λˡᵉᵃᶠ (–), upscales leaf `gₛ` to the canopy.
- `conductance`     : a [`JarvisConductance`](@ref) by default.
- `moisture_stress` : `β(𝒮)` model — a `Number`, a [`PlantAvailableWaterStress`](@ref)
  (wilting-point/field-capacity endpoints, the model meant for a transpiring canopy), or
  a [`CriticalSaturation`](@ref) (the bare-soil evaporation model).
- `absorbed_par`    : a prescribed per-leaf absorbed PAR (`Number`, mol photon m⁻² s⁻¹) or an
  [`AbsorbedPhotosyntheticallyActiveRadiation`](@ref).
- `phase`           : saturation phase (Liquid).
"""
struct CanopyConductanceHumidity{L, C, S, A, Φ}
    leaf_area_index :: L
    conductance     :: C
    moisture_stress :: S
    absorbed_par    :: A
    phase           :: Φ
end

function CanopyConductanceHumidity(FT=Oceananigans.defaults.FloatType;
                                   leaf_area_index = 2,
                                   conductance     = JarvisConductance(FT),
                                   moisture_stress = 1,
                                   absorbed_par    = 4e-4,
                                   phase           = AtmosphericThermodynamics.Liquid())

    return CanopyConductanceHumidity(convert_if_number(FT, leaf_area_index),
                                     conductance, moisture_stress,
                                     convert_if_number(FT, absorbed_par),
                                     phase)
end

Adapt.adapt_structure(to, q::CanopyConductanceHumidity) =
    CanopyConductanceHumidity(Adapt.adapt(to, q.leaf_area_index),
                              q.conductance, q.moisture_stress,
                              q.absorbed_par, q.phase)

Base.summary(::CanopyConductanceHumidity{L, C, S, A, Φ}) where {L, C, S, A, Φ} =
    string("CanopyConductanceHumidity{", Φ === AtmosphericThermodynamics.Liquid ? "Liquid" : "Ice", "}")
Base.show(io::IO, q::CanopyConductanceHumidity) = print(io, summary(q))

# The canopy reads the land state its moisture stress declares.
land_state_accessors(q::CanopyConductanceHumidity) = land_state_accessors(q.moisture_stress)

# The bulk λˡᵉᵃᶠ (a constant, a static `Field`, or a `FieldTimeSeries`) upscales the leaf
# conductance and shades the absorbed PAR.
@inline canopy_leaf_area_index(q::CanopyConductanceHumidity) = q.leaf_area_index
@inline interface_vegetation_state(i, j, grid, ::CanopyConductanceHumidity, vegetation, time_interpolator) =
    (leaf_area_index = convert(eltype(grid), stateindex(vegetation, i, j, 1, time_interpolator)),)

# Bulk canopy (stomatal) mass conductance `gᶜᵃⁿ = λˡᵉᵃᶠ · gₛ · Mᵈ` (kg m⁻² s⁻¹) and the leaf
# saturation humidity `qᵛ⁺(Tˡᵉᵃ)`; `qᵃ` is the humidity of the air the leaf exchanges with.
@inline function canopy_conductance_terms(q::CanopyConductanceHumidity, Tˡᵉᵃ, qᵃ, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ,
                                          canopy_transmittance)
    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    pᵃᵗ = Ψₐ.p

    λˡᵉᵃᶠ = Ψˡᵃ.leaf_area_index
    qᵛ⁺ = saturation_specific_humidity(ℂᵃᵗ, Tˡᵉᵃ, pᵃᵗ, q.phase)
    # Leaf-to-air vapor pressure deficit (Pa).
    VPD  = max(AtmosphericThermodynamics.vapor_pressure_deficit(ℂᵃᵗ, Tˡᵉᵃ, pᵃᵗ, qᵃ, 0, 0, q.phase),
               oftype(Tˡᵉᵃ, 50))
    β    = evaporation_efficiency(q.moisture_stress, Ψˡᵃ)
    APAR = absorbed_par_value(q.absorbed_par, Ψᵣ, λˡᵉᵃᶠ, canopy_transmittance)

    gₛ = stomatal_conductance(q.conductance, APAR, VPD, Tˡᵉᵃ, β)

    # gₛ (mol m⁻² s⁻¹ per m² leaf) drives a molar vapor flux gₛ Δχ across a mole-fraction
    # difference Δχ ≈ (Mᵈ/Mʷ) Δq, so its mass flux is Mʷ gₛ Δχ = Mᵈ gₛ Δq. λˡᵉᵃᶠ (m² leaf m⁻² ground)
    # upscales it to the canopy, and Mᵈ = 0.02897 kg mol⁻¹ is the dry-air molar mass, giving
    # gᶜᵃⁿ in kg m⁻² s⁻¹.
    gᶜᵃⁿ = λˡᵉᵃᶠ * gₛ * oftype(gₛ, default_dry_air_molar_mass)

    return gᶜᵃⁿ, qᵛ⁺
end

@inline function compute_interface_humidity(q::CanopyConductanceHumidity, Tₛ, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ)
    FT = eltype(Ψₛ)
    gᶜᵃⁿ, qᵛ⁺ = canopy_conductance_terms(q, Tₛ, Ψₐ.q, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ, nothing)

    gᵃᵛ = aerodynamic_vapor_conductance(Ψₛ, Ψₐ, ℙₐ.thermodynamics_parameters)
    qˢ  = conductance_weighted_node(Ψₛ.specific_humidity, (gᶜᵃⁿ, gᵃᵛ), (qᵛ⁺, Ψₐ.q))

    return convert(FT, qˢ)
end
