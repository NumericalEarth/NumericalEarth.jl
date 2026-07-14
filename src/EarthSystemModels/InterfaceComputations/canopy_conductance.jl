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
##### co-limitation (Eqs C1–C5), and the Medlyn conductance. The whole path is
##### Enzyme/Reactant-friendly.
#####
##### The stomatal conductance is pluggable: the photosynthesis-coupled
##### [`MedlynConductance`](@ref) (default) or the empirical, closed-form
##### [`JarvisConductance`](@ref), both behind `AbstractStomatalConductance`
##### (see `stomatal_conductance.jl`). The photosynthesis model lives in
##### `photosynthesis.jl` and absorbed PAR in `absorbed_par.jl`: either prescribed
##### ([`PrescribedAbsorbedPAR`](@ref), the offline default) or recomputed each step
##### from the downwelling shortwave ([`InteractiveAbsorbedPAR`](@ref)), so the
##### canopy can follow the diurnal light cycle. CO₂ is prescribed, and the leaf
##### temperature is the skin temperature `Tₛ` (single-source).
#####

# Leaf-to-air vapor pressure deficit (Pa), floored to a small positive value so
# the Medlyn `√VPD` stays finite and differentiable at saturation.
@inline function vapor_pressure_deficit(ℂᵃᵗ, Tₗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ, phase)
    eₛ = AtmosphericThermodynamics.saturation_vapor_pressure(ℂᵃᵗ, Tₗ, phase)
    ε  = 1 / AtmosphericThermodynamics.Parameters.Rv_over_Rd(ℂᵃᵗ)   # Rᵈ/Rᵥ ≈ 0.622
    eₐ = pᵃᵗ * qᵃᵗ / (ε + (1 - ε) * qᵃᵗ)                            # air vapor pressure
    return max(eₛ - eₐ, oftype(Tₗ, 1))                              # ≥ 1 Pa
end

# Convert only a scalar LAI; a `Field` (static map) or `FieldTimeSeries` (monthly
# data interpolated to the clock) passes through so the canopy conductance can
# vary in space and time. The per-cell value is materialized in the flux kernel.
@inline leaf_area_index_property(x::Number, FT) = convert(FT, x)
@inline leaf_area_index_property(x, FT) = x

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

    return CanopyConductanceHumidity(leaf_area_index_property(leaf_area_index, FT),
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

# The bulk LAI upscales the leaf conductance and shades the absorbed PAR. It is a
# prescribed vegetation input (constant, static `Field`, or `FieldTimeSeries`),
# materialized per-cell here so the fixed-point solve reads a plain scalar.
@inline canopy_leaf_area_index(q::CanopyConductanceHumidity) = q.leaf_area_index
# Convert to the grid float type: time-interpolating a `FieldTimeSeries` blends
# the data with the (often `Float64`) times, so the raw value may not be `FT`.
@inline interface_vegetation_state(i, j, grid, ::CanopyConductanceHumidity, vegetation, time_interpolator) =
    (leaf_area_index = convert(eltype(grid), surface_field_value(vegetation, i, j, time_interpolator)),)

# Canopy flux terms, split off so the standalone formulation and the composite
# (soil + canopy) share them. Returns the bulk canopy (stomatal) mass conductance
# `g_c = LAI · gₛ · Mᵈ` (kg m⁻² s⁻¹) and the leaf saturation source `qᵛ⁺(Tₗ)`.
# The canopy (leaf) reservoir is saturated at the leaf temperature (= skin
# temperature Tₛ, single-source). `Ψᵣ` is the interface radiation state (drives
# `InteractiveAbsorbedPAR`).
@inline function canopy_conductance_terms(q::CanopyConductanceHumidity, Tₗ, Ψₛ, Ψₐ, Ψᵣ, ℙₐ)
    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    pᵃᵗ = Ψₐ.p
    qᵃᵗ = Ψₐ.q
    Tᵃᵗ = Ψₐ.T

    LAI  = Ψₛ.vegetation.leaf_area_index               # materialized per-cell (constant, Field, or FTS)
    qᵛ⁺  = saturation_specific_humidity(ℂᵃᵗ, Tₗ, pᵃᵗ, q.phase)
    VPD  = vapor_pressure_deficit(ℂᵃᵗ, Tₗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ, q.phase)
    β    = evaporation_efficiency(q.moisture_stress, Ψₛ.hydrology)
    APAR = absorbed_par_value(q.absorbed_par, Ψᵣ, LAI)

    gs, _, _ = stomatal_conductance(q.conductance, q.photosynthesis,
                                    APAR, VPD, Tₗ, q.atmospheric_co2, pᵃᵗ, β)

    # Molar leaf conductance → canopy mass conductance (kg m⁻² s⁻¹).
    g_c = LAI * gs * oftype(gs, default_dry_air_molar_mass)

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
