#####
##### `CanopyConductanceHumidity` — a single-source (big-leaf) canopy: the stomatal
##### conductance `gᶜ = LAI · gₛ` in series with the aerodynamic conductance.
#####

# Leaf-to-air vapor pressure deficit (Pa), floored so the Medlyn `√VPD` stays
# differentiable at saturation.
@inline function vapor_pressure_deficit(ℂᵃᵗ, Tˡᵉᵃᶠ, Tᵃᵗ, pᵃᵗ, qᵃᵗ, phase)
    eₛ = AtmosphericThermodynamics.saturation_vapor_pressure(ℂᵃᵗ, Tˡᵉᵃᶠ, phase)
    ε  = 1 / AtmosphericThermodynamics.Parameters.Rv_over_Rd(ℂᵃᵗ)   # Rᵈ/Rᵛ ≈ 0.622
    eₐ = pᵃᵗ * qᵃᵗ / (ε + (1 - ε) * qᵃᵗ)                            # air vapor pressure
    return max(eₛ - eₐ, oftype(Tˡᵉᵃᶠ, 1))                              # ≥ 1 Pa
end

"""
    struct CanopyConductanceHumidity

Surface specific humidity `qˢ` for a single-source (big-leaf) canopy: the
canopy conductance `gᶜ = LAI · gₛ` in series with the aerodynamic conductance,
solved inside the Monin–Obukhov fixed point exactly as [`SkinHumidity`](@ref)
solves a soil-resistance balance. The stomatal conductance `gₛ` comes from the
`conductance` model driven by the per-cell leaf-to-air VPD, leaf temperature
(`= Tₛ`, single-source), and absorbed PAR, with the moisture-stress factor `β(𝒮)`
read from the ground hydrology (`moisture_stress`, a `Number`, a
[`PlantAvailableWaterStress`](@ref) — the model meant for transpiration — or a
[`CriticalSaturation`](@ref)). The conductance is either the empirical
[`JarvisConductance`](@ref) (default; needs no `photosynthesis` model) or the
photosynthesis-coupled [`MedlynConductance`](@ref). Absorbed PAR is a prescribed
`Number` or live from the radiation state ([`InteractiveAbsorbedPAR`](@ref)); CO₂ is
prescribed.

Fields:
- `leaf_area_index` : bulk LAI (–), upscales leaf `gₛ` to the canopy.
- `photosynthesis`  : a [`FarquharPhotosynthesis`](@ref), or `nothing` for Jarvis.
- `conductance`     : a [`JarvisConductance`](@ref) (default) or [`MedlynConductance`](@ref).
- `moisture_stress` : `β(𝒮)` model — a `Number`, a [`PlantAvailableWaterStress`](@ref)
  (wilting-point/field-capacity endpoints, the model meant for a transpiring canopy), or
  a [`CriticalSaturation`](@ref) (the bare-soil evaporation model).
- `absorbed_par`    : a prescribed per-leaf absorbed PAR (`Number`, mol photon m⁻² s⁻¹) or an
  [`InteractiveAbsorbedPAR`](@ref).
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

function CanopyConductanceHumidity(FT=Oceananigans.defaults.FloatType;
                                   leaf_area_index = 2,
                                   photosynthesis  = nothing,
                                   conductance     = JarvisConductance(FT),
                                   moisture_stress = 1,
                                   absorbed_par    = 4e-4,
                                   atmospheric_co2 = 40,
                                   phase           = AtmosphericThermodynamics.Liquid())

    photosynthesis = default_photosynthesis(photosynthesis, conductance, FT)

    return CanopyConductanceHumidity(convert_if_number(FT, leaf_area_index),
                                     photosynthesis, conductance, moisture_stress,
                                     convert_if_number(FT, absorbed_par),
                                     convert(FT, atmospheric_co2), phase)
end

Adapt.adapt_structure(to, q::CanopyConductanceHumidity) =
    CanopyConductanceHumidity(Adapt.adapt(to, q.leaf_area_index),
                              q.photosynthesis, q.conductance, q.moisture_stress,
                              q.absorbed_par, q.atmospheric_co2, q.phase)

Base.summary(::CanopyConductanceHumidity{L, P, C, S, A, Q, Φ}) where {L, P, C, S, A, Q, Φ} =
    string("CanopyConductanceHumidity{", Φ === AtmosphericThermodynamics.Liquid ? "Liquid" : "Ice", "}")
Base.show(io::IO, q::CanopyConductanceHumidity) = print(io, summary(q))

# The canopy stress reads the ground saturation 𝒮 and, for a `PlantAvailableWaterStress`,
# its per-cell stress endpoints, so the interface materializes the stress's own state.
@inline interface_hydrology_state(i, j, grid, q::CanopyConductanceHumidity, land_state) =
    merge(land_saturation(i, j, grid, land_state),
          interface_hydrology_state(i, j, grid, q.moisture_stress, land_state))
@inline requires_retention_curve(q::CanopyConductanceHumidity) = requires_retention_curve(q.moisture_stress)

# The bulk LAI upscales the leaf conductance and shades the absorbed PAR. It is a
# prescribed vegetation input (constant, static `Field`, or `FieldTimeSeries`),
# materialized per-cell here so the fixed-point solve reads a plain scalar.
@inline canopy_leaf_area_index(q::CanopyConductanceHumidity) = q.leaf_area_index
# Convert to the grid float type: time-interpolating a `FieldTimeSeries` blends
# the data with the (often `Float64`) times, so the raw value may not be `FT`.
@inline interface_vegetation_state(i, j, grid, ::CanopyConductanceHumidity, vegetation, time_interpolator) =
    (leaf_area_index = convert(eltype(grid), surface_field_value(vegetation, i, j, time_interpolator)),)

# Bulk canopy (stomatal) mass conductance `gᶜ = LAI · gₛ · Mᵈ` (kg m⁻² s⁻¹) and the leaf
# saturation humidity `qᵛ⁺(Tˡᵉᵃᶠ)`; `qᵃ` is the humidity of the air the leaf exchanges with.
@inline function canopy_conductance_terms(q::CanopyConductanceHumidity, Tˡᵉᵃᶠ, qᵃ, Ψₛ, Ψₐ, Ψᵣ, ℙₐ,
                                          canopy_transmittance)
    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    pᵃᵗ = Ψₐ.p
    Tᵃᵗ = Ψₐ.T

    LAI  = Ψₛ.vegetation.leaf_area_index               # materialized per-cell (constant, Field, or FTS)
    qᵛ⁺  = saturation_specific_humidity(ℂᵃᵗ, Tˡᵉᵃᶠ, pᵃᵗ, q.phase)
    VPD  = vapor_pressure_deficit(ℂᵃᵗ, Tˡᵉᵃᶠ, Tᵃᵗ, pᵃᵗ, qᵃ, q.phase)
    β    = evaporation_efficiency(q.moisture_stress, Ψₛ.hydrology)
    APAR = absorbed_par_value(q.absorbed_par, Ψᵣ, LAI, canopy_transmittance)

    gₛ, _, _ = stomatal_conductance(q.conductance, q.photosynthesis,
                                    APAR, VPD, Tˡᵉᵃᶠ, q.atmospheric_co2, pᵃᵗ, β)

    # Molar leaf conductance → canopy mass conductance (kg m⁻² s⁻¹).
    gᶜ = LAI * gₛ * oftype(gₛ, default_dry_air_molar_mass)

    return gᶜ, qᵛ⁺
end

@inline function compute_interface_humidity(q::CanopyConductanceHumidity, Tₛ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ)
    FT = eltype(Ψₛ)
    gᶜ, qᵛ⁺ = canopy_conductance_terms(q, Tₛ, Ψₐ.q, Ψₛ, Ψₐ, Ψᵣ, ℙₐ, nothing)

    Gᵃ = aerodynamic_vapor_conductance(Ψₛ, Ψₐ, ℙₐ.thermodynamics_parameters)
    qˢ = conductance_weighted_node(Ψₛ.specific_humidity, (gᶜ, Gᵃ), (qᵛ⁺, Ψₐ.q))

    return convert(FT, qˢ)
end
