using ClimaSeaIce: ClimaSeaIce
using Thermodynamics: Thermodynamics as AtmosphericThermodynamics

#####
##### Interface properties
#####

"""
$(TYPEDSIGNATURES)

The physics of one atmosphere interface (air–sea, air–ice, or air–land): the turbulent-flux
closure (e.g. `SimilarityTheoryFluxes`), the interface-state formulation, and the
atmosphere–surface velocity difference. `state` is a named tuple of `temperature` and
`specific_humidity` formulations, or a coupled formulation such as `CanopyInterface`.
"""
struct InterfaceFormulation{F, S, V}
    turbulent_fluxes     :: F
    state               :: S
    velocity_difference :: V
end

Adapt.@adapt_structure InterfaceFormulation

@inline temperature_formulation(state::NamedTuple) = state.temperature
@inline specific_humidity_formulation(state::NamedTuple) = state.specific_humidity

@inline local_interface_formulation(state::NamedTuple, i, j) =
    map(formulation -> local_interface_formulation(formulation, i, j), state)

# Collapse Field-valued roughness lengths, displacement, and surface optics to the values at cell `(i, j)`.
@inline local_interface_formulation(formulation::InterfaceFormulation, i, j) =
    InterfaceFormulation(local_flux_formulation(formulation.turbulent_fluxes, i, j),
                         local_interface_formulation(formulation.state, i, j),
                         formulation.velocity_difference)

#####
##### Interface specific humidity formulations
#####

# TODO: allow different saturation models
# struct ClasiusClapyeronSaturation end
struct ImpureSaturationSpecificHumidity{Φ, X, E}
    # saturation :: S
    phase :: Φ
    water_mole_fraction :: X
    saturation_enhancement :: E
end

function Base.summary(q★::ImpureSaturationSpecificHumidity)
    phase_str = if q★.phase == AtmosphericThermodynamics.Ice()
        "Ice"
    elseif q★.phase == AtmosphericThermodynamics.Liquid()
        "Liquid"
    end

    return string("ImpureSaturationSpecificHumidity{$phase_str}(water_mole_fraction=",
                  prettysummary(q★.water_mole_fraction),
                  ", saturation_enhancement=", prettysummary(q★.saturation_enhancement), ")")
end

Base.show(io::IO, q★::ImpureSaturationSpecificHumidity) = print(io, summary(q★))

"""
    ImpureSaturationSpecificHumidity(phase [, water_mole_fraction=1]; saturation_enhancement=nothing)

Return the formulation for computing specific humidity at an interface. `saturation_enhancement` multiplies the
saturation vapor pressure by the enhancement factor of moist air, for example `GillSaturationEnhancement()`;
with `nothing` the saturation vapor pressure is not enhanced.
"""
ImpureSaturationSpecificHumidity(phase, water_mole_fraction=nothing; saturation_enhancement=nothing) =
    ImpureSaturationSpecificHumidity(phase, water_mole_fraction, saturation_enhancement)

@inline compute_water_mole_fraction(::Nothing, salinity) = 1
@inline compute_water_mole_fraction(x_H₂O::Number, salinity) = x_H₂O

"""
    GillSaturationEnhancement{FT}

Enhancement factor of the saturation vapor pressure in moist air from [Gill (1982)](@cite gill1982atmosphere),

    fᵛ = 1 + p (a + b t²),

with `p` the air pressure (Pa), `t` the temperature (ᵒC), `a = pressure_coefficient` (Pa⁻¹) and
`b = temperature_coefficient` (Pa⁻¹ ᵒC⁻²).
"""
struct GillSaturationEnhancement{FT}
    pressure_coefficient :: FT
    temperature_coefficient :: FT
end

GillSaturationEnhancement(FT = Oceananigans.defaults.FloatType; pressure_coefficient = 4.5e-8, temperature_coefficient = 6e-12) =
    GillSaturationEnhancement(convert(FT, pressure_coefficient), convert(FT, temperature_coefficient))

Base.summary(::GillSaturationEnhancement{FT}) where FT = "GillSaturationEnhancement{$FT}"

@inline saturation_enhancement_factor(::Nothing, T, p) = 1

@inline function saturation_enhancement_factor(enhancement::GillSaturationEnhancement, T, p)
    t = convert(typeof(T), T - celsius_to_kelvin)
    return 1 + p * (enhancement.pressure_coefficient + enhancement.temperature_coefficient * t^2)
end

# COARE 3.6 / Edson (2013) pressure-based saturation specific humidity:
#   qₛ = εᵈᵛ⁻¹ pᵛ⁺ / (p − (1 − εᵈᵛ⁻¹) pᵛ⁺),   εᵈᵛ⁻¹ = Rᵈ / Rᵛ
# Direct evaluation at the atmospheric pressure p. The 6th positional
# argument `qᵃᵗ` is accepted (and ignored) so the same call site can
# dispatch on either `ImpureSaturationSpecificHumidity` or
# [`BulkHumidity`](@ref), which does need it.
@inline function surface_specific_humidity(formulation::ImpureSaturationSpecificHumidity, ℂᵃᵗ, pᵃᵗ, Tₛ, Sₛ=zero(Tₛ), qᵃᵗ=zero(Tₛ))
    FT = eltype(Tₛ)
    CT = eltype(ℂᵃᵗ)
    T  = convert(CT, Tₛ)
    p  = convert(CT, pᵃᵗ)

    # Raoult's law on the saturation vapor pressure, enhanced in moist air by fᵛ
    χ_H₂O = compute_water_mole_fraction(formulation.water_mole_fraction, Sₛ)
    fᵛ    = saturation_enhancement_factor(formulation.saturation_enhancement, T, p)
    pᵛ⁺   = fᵛ * χ_H₂O * AtmosphericThermodynamics.saturation_vapor_pressure(ℂᵃᵗ, T, formulation.phase)
    εᵈᵛ⁻¹ = 1 / AtmosphericThermodynamics.Parameters.Rv_over_Rd(ℂᵃᵗ)

    # Guard against unphysically warm interface temperatures: once pᵛ⁺ exceeds
    # p / (1 − εᵈᵛ⁻¹) the denominator below turns negative, producing a negative
    # qₛ that drives a runaway spurious-condensation instability. In the physical
    # regime pᵛ⁺ ≪ p the cap is inert; it keeps qₛ ∈ [0, 1).
    pᵛ⁺   = min(pᵛ⁺, convert(CT, 0.999) * p)
    qₛ    = εᵈᵛ⁻¹ * pᵛ⁺ / (p - (1 - εᵈᵛ⁻¹) * pᵛ⁺)

    return convert(FT, qₛ)
end

# Pressure-based saturation specific humidity qᵛ⁺ (COARE / Edson 2013):
#   qᵛ⁺ = εᵈᵛ⁻¹ pᵛ⁺ / (p − (1 − εᵈᵛ⁻¹) pᵛ⁺),   εᵈᵛ⁻¹ = Rᵈ / Rᵛ.
# Shared by `BulkHumidity` and `SkinHumidity`.
@inline function saturation_specific_humidity(ℂᵃᵗ, Tₛ, pᵃᵗ, phase)
    CT = eltype(ℂᵃᵗ)
    T  = convert(CT, Tₛ)
    p  = convert(CT, pᵃᵗ)
    pᵛ⁺   = AtmosphericThermodynamics.saturation_vapor_pressure(ℂᵃᵗ, T, phase)
    εᵈᵛ⁻¹ = 1 / AtmosphericThermodynamics.Parameters.Rv_over_Rd(ℂᵃᵗ)

    # Same negative-denominator guard as in `surface_specific_humidity` above;
    # inert in the physical regime pᵛ⁺ ≪ p.
    pᵛ⁺   = min(pᵛ⁺, convert(CT, 0.999) * p)
    return εᵈᵛ⁻¹ * pᵛ⁺ / (p - (1 - εᵈᵛ⁻¹) * pᵛ⁺)
end

# dqᵛ⁺/dT at fixed pressure — the Newton derivative of each balance's latent term.
# Clausius-Clapeyron gives d(ln pᵛ⁺)/dT = ℒ(T)/(Rᵛ T²), and differentiating
# qᵛ⁺ = εᵈᵛ⁻¹ pᵛ⁺ / (p - (1 - εᵈᵛ⁻¹) pᵛ⁺) at fixed p leaves
#
#     dqᵛ⁺/dT = qᵛ⁺ (1 + (1 - εᵈᵛ⁻¹) qᵛ⁺/εᵈᵛ⁻¹) ℒ(T) / (Rᵛ T²) .
@inline function saturation_humidity_slope(ℂᵃᵗ, Tₛ, pᵃᵗ, phase,
                                           qᵛ⁺ = saturation_specific_humidity(ℂᵃᵗ, Tₛ, pᵃᵗ, phase))
    CT    = eltype(ℂᵃᵗ)
    T     = convert(CT, Tₛ)
    εᵈᵛ⁻¹ = 1 / AtmosphericThermodynamics.Parameters.Rv_over_Rd(ℂᵃᵗ)
    Rᵛ    = AtmosphericThermodynamics.Parameters.R_v(ℂᵃᵗ)
    ℒ     = ifelse(phase isa AtmosphericThermodynamics.Liquid,
                   AtmosphericThermodynamics.latent_heat_vapor(ℂᵃᵗ, T),
                   AtmosphericThermodynamics.latent_heat_sublim(ℂᵃᵗ, T))
    return qᵛ⁺ * (1 + (1 - εᵈᵛ⁻¹) * qᵛ⁺ / εᵈᵛ⁻¹) * ℒ / (Rᵛ * T^2)
end

# Saturation phase of a humidity formulation, used only for the initial surface-
# humidity guess in the flux kernel. Most formulations store it directly;
# composite formulations forward to their soil branch.
@inline interface_phase(q_formulation) = q_formulation.phase

# Aerodynamic vapor `gᵃᵛ = ρᵃᵗ u★ χq` and heat `gᵃᵀ = ρᵃᵗ cᵖᵐ u★ χθ` conductances of the
# previous similarity iterate, the bulk linearizations of the similarity fluxes the surface
# balances close against.
@inline function aerodynamic_vapor_conductance(Ψₛ, Ψₐ, ℂᵃᵗ)
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Ψₐ.T, Ψₐ.p, Ψₐ.q)
    return ρᵃᵗ * Ψₛ.fluxes.u★ * Ψₛ.fluxes.χq
end

@inline function aerodynamic_heat_conductance(Ψₛ, Ψₐ, ℂᵃᵗ)
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Ψₐ.T, Ψₐ.p, Ψₐ.q)
    cᵖᵐ = AtmosphericThermodynamics.cp_m(ℂᵃᵗ, Ψₐ.q)
    return ρᵃᵗ * cᵖᵐ * Ψₛ.fluxes.u★ * Ψₛ.fluxes.χθ
end

# Conductance-weighted mean of the `sources` behind the `conductances`: the value of a
# node in flux continuity with all of them. Every conductance zero keeps `x⁻`.
@inline function conductance_weighted_node(x⁻, conductances, sources)
    D = sum(conductances)
    return ifelse(D > 0, sum(map(*, conductances, sources)) / D, x⁻)
end

# `BulkHumidity` — surface specific humidity for a bulk land surface with no
# skin-resistance parameterization. The surface is saturated at the bulk
# (skin) temperature wherever there is water, and dry otherwise:
#
#     qₛ = qᵛ⁺(Tₛ, p)   where the surface is wet,   0   where it is dry.
#
# "Wet" / "dry" is decided by the land's surface saturation (for `BucketHydrology`,
# > 0 where `water_storage > 0`; `SaturatedSurface` → 1, `DryLand` → 0). This is a
# pure surface property: a dry surface has qₛ = 0, so under humid air the vapor flux
# runs downward (dew/frost). The skin-resistance model [`SkinHumidity`](@ref)
# instead lets the surface be sub-saturated even where the bulk holds water.
#
# The saturation is the land state's `𝒮`. `BulkHumidity` has no moisture-availability
# parameter of its own — only the saturation `phase`.
struct BulkHumidity{Φ}
    phase :: Φ
end

BulkHumidity(; phase=AtmosphericThermodynamics.Liquid()) = BulkHumidity(phase)

Base.summary(::BulkHumidity{Φ}) where Φ =
    string("BulkHumidity{", Φ === AtmosphericThermodynamics.Liquid ? "Liquid" : "Ice", "}")
Base.show(io::IO, q::BulkHumidity) = print(io, summary(q))

# Pressure-based saturation specific humidity (same Raoult / pressure formula
# as `ImpureSaturationSpecificHumidity`) where the surface is wet, else 0. The
# 6th positional `qᵃᵗ` is accepted and ignored so the call site can dispatch on
# either formulation. The 5th positional is the land surface saturation `𝒮`.
@inline function surface_specific_humidity(formulation::BulkHumidity,
                                           ℂᵃᵗ, pᵃᵗ, Tₛ, 𝒮=one(Tₛ), qᵃᵗ=zero(Tₛ))
    FT  = eltype(Tₛ)
    qᵛ⁺ = saturation_specific_humidity(ℂᵃᵗ, Tₛ, pᵃᵗ, formulation.phase)
    return convert(FT, ifelse(𝒮 > 0, qᵛ⁺, zero(qᵛ⁺)))
end

#####
##### FractionalHumidity — saturation scaled by an evaporation efficiency
#####

"""
    struct FractionalHumidity

Surface specific humidity as a fraction of saturation at the surface temperature,

```math
qˢ = β · qᵛ⁺(Tₛ),
```

where the evaporation efficiency `β` is set by `efficiency` — a [`CriticalSaturation`](@ref)
([Manabe, 1969](@cite manabe1969climate)) or a constant `Number`. Unlike [`SkinHumidity`](@ref), the saturation is
taken at the *skin* temperature: `β` is a surface evaporation efficiency, not a deep
reservoir. `BulkHumidity` is the `𝒮ᶜ → 0` corner (saturated wherever `𝒮 > 0`).
"""
struct FractionalHumidity{E, Φ}
    efficiency :: E
    phase :: Φ
end

FractionalHumidity(phase=AtmosphericThermodynamics.Liquid(); efficiency) =
    FractionalHumidity(efficiency, phase)

Base.summary(::FractionalHumidity{E, Φ}) where {E, Φ} =
    string("FractionalHumidity{", E, ", ", Φ === AtmosphericThermodynamics.Liquid ? "Liquid" : "Ice", "}")
Base.show(io::IO, q::FractionalHumidity) = print(io, summary(q))

#####
##### SkinHumidity — surface specific humidity from a soil vapor flux balance
#####

"""
    struct SkinHumidity

Surface specific humidity `qˢ` solved from a vapor-flux balance at the land
surface, the humidity analog of [`SkinTemperature`](@ref).

Vapor reaches the surface by diffusing up from saturated soil at the saturation
depth `d` (the `surface_thickness`), where the soil air is saturated at `qᵛ⁺(Tᵢ)` —
evaluated at the interior (bulk land) temperature, since the reservoir sits at
depth below the surface. Fick's law across `d` gives the internal (soil) vapor flux

```math
J^q = - κ^q/d \\, (qˢ - qᵛ⁺)
```

with soil vapor diffusivity `κ^q` (`vapor_diffusivity`). The surface is massless,
so `qˢ` is the value for which this soil flux balances the atmospheric vapor flux
carried away by turbulence — `qˢ` is solved inside the interface fixed-point
iteration (see `compute_interface_humidity`), exactly as `SkinTemperature` solves
`Tₛ` from a surface energy balance.

`surface_thickness` is a `Number` (fixed `d`). A future
`WetnessDependentSurfaceThickness` will let `d` grow as the soil dries, making
evaporation self-limiting.
"""
struct SkinHumidity{D, K, Φ}
    surface_thickness :: D
    vapor_diffusivity :: K
    phase :: Φ
end

SkinHumidity(phase=AtmosphericThermodynamics.Liquid(); surface_thickness, vapor_diffusivity) =
    SkinHumidity(surface_thickness, vapor_diffusivity, phase)

Base.summary(::SkinHumidity{D, K, Φ}) where {D, K, Φ} =
    string("SkinHumidity{", Φ === AtmosphericThermodynamics.Liquid ? "Liquid" : "Ice", "}")
Base.show(io::IO, q::SkinHumidity) = print(io, summary(q))

# Saturation depth d. For a fixed `Number` thickness it is the number itself;
# a future `WetnessDependentSurfaceThickness` will dispatch here on the land state.
@inline surface_layer_thickness(d::Number, Ψˡᵃ) = d

struct SalinityConstituent{FT}
    molar_mass :: FT
    mass_fraction :: FT
end

struct WaterMoleFraction{FT, C}
    water_molar_mass :: FT
    salinity_constituents :: C
end

function WaterMoleFraction(FT=Oceananigans.defaults.FloatType)
    water_molar_mass = convert(FT, 18.02)

    # TODO: find reference for these
    salinity_constituents = (
        chloride  = SalinityConstituent{FT}(35.45, 0.56),
        sodium    = SalinityConstituent{FT}(22.99, 0.31),
        sulfate   = SalinityConstituent{FT}(96.06, 0.08),
        magnesium = SalinityConstituent{FT}(24.31, 0.05),
    )

    return WaterMoleFraction(water_molar_mass, salinity_constituents)
end

@inline function compute_water_mole_fraction(wmf::WaterMoleFraction, S)
    # TODO: express the concept of "ocean_salinity_units"?
    s = S / 1000 # convert g/kg to concentration

    # Molecular weights
    μ_H₂O = wmf.water_molar_mass

    # Salinity constituents: Cl⁻, Na, SO₄, Mg
    μ_Cl  = wmf.salinity_constituents.chloride.molar_mass
    μ_Na  = wmf.salinity_constituents.sodium.molar_mass
    μ_SO₄ = wmf.salinity_constituents.sulfate.molar_mass
    μ_Mg  = wmf.salinity_constituents.magnesium.molar_mass

    # Salinity constituent fractions
    ϵ_Cl  = wmf.salinity_constituents.chloride.mass_fraction
    ϵ_Na  = wmf.salinity_constituents.sodium.mass_fraction
    ϵ_SO₄ = wmf.salinity_constituents.sulfate.mass_fraction
    ϵ_Mg  = wmf.salinity_constituents.magnesium.mass_fraction

    α = μ_H₂O * (ϵ_Cl/μ_Cl + ϵ_Na/μ_Na  + ϵ_SO₄/μ_SO₄ + ϵ_Mg/μ_Mg)

    return (1 - s) / (1 - s + α * s)
end

####
#### Velocity difference formulations
####

""" The exchange fluxes depend on the atmosphere velocity but not the interface velocity """
struct WindVelocity end

""" The exchange fluxes depend on the relative velocity between the atmosphere and the interface """
struct RelativeVelocity end

# Velocity components from either an interface state (`velocities` component, see
# the methods defined with `AbstractInterfaceState` below) or a flat atmosphere /
# ocean-current state.
@inline x_velocity(𝒰) = 𝒰.u
@inline y_velocity(𝒰) = 𝒰.v

@inline function velocity_difference(::RelativeVelocity, 𝒰₁, 𝒰₀)
    Δu = x_velocity(𝒰₁) - x_velocity(𝒰₀)
    Δv = y_velocity(𝒰₁) - y_velocity(𝒰₀)
    return Δu, Δv
end

@inline velocity_difference(::WindVelocity, 𝒰₁, 𝒰₀) = x_velocity(𝒰₁), y_velocity(𝒰₁)

####
#### Atmospheric temperature
####

# Temperature increment including the ``lapse rate'' `α = g / cᵖᵐ`
function surface_atmosphere_temperature(Ψₐ, ℙₐ)
    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    g  = ℙₐ.gravitational_acceleration
    Tᵃᵗ = Ψₐ.T
    qᵃᵗ = Ψₐ.q
    zᵃᵗ = Ψₐ.z
    Δh = zᵃᵗ # Assumption! The surface is at z = 0 -> Δh = zᵃᵗ - 0
    cᵃᵗ = AtmosphericThermodynamics.cp_m(ℂᵃᵗ, qᵃᵗ)
    return Tᵃᵗ + g * Δh / cᵃᵗ
end

####
#### Interface temperature formulations
####

"""
    struct BulkTemperature

A type to represent the interface temperature used in fixed-point iteration for interface
fluxes following similarity theory. The interface temperature is not calculated but instead
provided by either the ocean or the sea ice model.
"""
struct BulkTemperature end

# Do nothing (just copy the temperature)
@inline compute_interface_temperature(::BulkTemperature, Ψₛ, args...) = Ψₛ.temperature

####
#### Skin interface temperature calculated as a flux balance
####

"""
    struct SkinTemperature

A type to represent the interface temperature used in the flux calculation.
The interface temperature is calculated from the flux balance at the interface.
In particular, the interface temperature ``Tₛ`` is the root of:

```math
F(Tₛ) - Jᵀ = 0
```

where ``Jᵀ`` are the fluxes at the top of the interface (turbulent + radiative), and
``F`` is the internal diffusive flux dependent on the interface temperature itself.

Note that all fluxes positive upwards.
"""
struct SkinTemperature{I, FT}
    internal_flux :: I
    max_ΔT :: FT
end

SkinTemperature(internal_flux; max_ΔT=5) = SkinTemperature(internal_flux, max_ΔT)

"""
    DiffusiveFlux(κ, δ)

Internal flux ``J = - κ (Tₛ - Tᵢ) / δ`` between the interior temperature ``Tᵢ``,
located a distance ``δ`` below the interface (typically half the spacing of the topmost
interior cell), and the interface temperature ``Tₛ``. The diffusivity `κ` (m² s⁻¹) is
either a prescribed constant or an [`InteriorDiffusivity`](@ref) assessed from the
interior model.
"""
struct DiffusiveFlux{K, Z}
    κ :: K # diffusivity in m² s⁻¹
    δ :: Z # Boundary layer thickness, as a first guess we will use half the grid spacing
end

"""
    InteriorDiffusivity(FT = Oceananigans.defaults.FloatType; minimum_diffusivity = 1.4e-7)

Diffusivity for a [`DiffusiveFlux`](@ref) that is assessed from the interior model (for example the near-surface
vertical diffusivity predicted by the ocean turbulence closure) instead of being prescribed. The assessed value is floored by
`minimum_diffusivity`, which defaults to the molecular thermal diffusivity of seawater, guarding stably-stratified conditions
in which modeled diffusivities vanish.
"""
struct InteriorDiffusivity{FT}
    minimum_diffusivity :: FT
end

InteriorDiffusivity(FT::DataType = Oceananigans.defaults.FloatType; minimum_diffusivity = 1.4e-7) =  InteriorDiffusivity(convert(FT, minimum_diffusivity))

@inline internal_diffusivity(κ::Number, Ψᵢ) = κ
@inline internal_diffusivity(d::InteriorDiffusivity, Ψᵢ) = max(Ψᵢ.κ, d.minimum_diffusivity)

# A skin temperature whose internal flux uses the interior model's diffusivity
const IDST = SkinTemperature{<:DiffusiveFlux{<:InteriorDiffusivity}}

# We try to keep the parameter space clean. If we do not need the diffusivity we remove it.
assemble_interior_fields(state, temperature_formulation) = Base.structdiff(state, NamedTuple{(:κ,)})
assemble_interior_fields(state, temperature_formulation::IDST) = state

# The flux balance is solved by computing
#
#            κ
# Jᵃ(Tₛⁿ) + --- (Tₛⁿ⁺¹ - Tˢⁱ) = 0
#            δ
#
# where Jᵃ is the external flux impinging on the surface from above and
# Jᵢ = - κ (Tₛ - Tˢⁱ) / δ is the "internal flux" coming up from below.
# We have indicated that Jᵃ may depend on the surface temperature from the previous
# iterate. We thus find that
#
# Tₛⁿ⁺¹ = Tˢⁱ - δ * Jᵃ(Tₛⁿ) / κ
#
# Note that we could also use the fact that Jᵃ(T) = σ * ϵ * T^4 + ⋯
# to expand Jᵃ around Tⁿ⁺¹,
#
# Jᵃ(Tⁿ⁺¹) ≈ Jᵃ(Tⁿ) + (Tⁿ⁺¹ - Tⁿ) * ∂T_Jᵃ(Tⁿ)
#          ≈ Jᵃ(Tⁿ) + 4 * (Tⁿ⁺¹ - Tⁿ) σ * ϵ * Tⁿ^3 / (ρ c)
#
# which produces the alternative, semi-implicit flux balance
#
#                                      κ
# Jᵃ(Tₛⁿ) - 4 α Tₛⁿ⁴ + 4 α Tₛⁿ Tₛⁿ³ + --- (Tₛⁿ⁺¹ - Tˢⁱ) = 0
#                                      δ
#
# with α = σ ϵ / (ρ c) such that
#
# Tₛⁿ⁺¹ (κ / δ + 4 α Tₛⁿ³) = κ * Tˢⁱ / δ - Jᵃ + 4 α Tₛⁿ⁴)
#
# or
#
# Tₛⁿ⁺¹ = = (Tˢⁱ - δ / κ * (Jᵃ - 4 α Tₛⁿ⁴)) / (1 + 4 δ σ ϵ Tₛⁿ³ / ρ c κ)
#
# corresponding to a linearization of the outgoing longwave radiation term.
@inline function flux_balance_temperature(st::SkinTemperature{<:DiffusiveFlux}, Ψₛ, interface_formulation, 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd, Ψᵢ, ℙᵢ, Ψₐ, ℙₐ)
    FT = typeof(Ψₛ.temperature)
    F  = st.internal_flux
    κ  = convert(FT, internal_diffusivity(F.κ, Ψᵢ))
    δ  = convert(FT, F.δ)
    ρ  = ℙᵢ.reference_density
    c  = ℙᵢ.heat_capacity
    Qa = 𝒬ᵛ + ℐꜛˡʷ + Qd # Net flux excluding sensible heat (positive out of the ocean)
    λ  = 1 / (ρ * c) # m³ K J⁻¹
    Jᵀ = Qa * λ

    # Calculating the atmospheric temperature
    Tᵃᵗ = surface_atmosphere_temperature(Ψₐ, ℙₐ)
    ΔT = Tᵃᵗ - Ψₛ.temperature

    # Flux balance: T★ = (Tᵢ κ - (Jᵀ + Ωc Tᵃᵗ) δ) / (κ - Ωc δ)
    # where Ωc = 𝒬ᵀ λ / ΔT. Multiply through by ΔT to avoid Inf when ΔT → 0.
    Ωᵀ = 𝒬ᵀ * λ  # unnormalized sensible heat coefficient (= Ωc * ΔT)
    D  = κ * ΔT - Ωᵀ * δ
    T★ = (Ψᵢ.T * κ * ΔT - (Jᵀ * ΔT + Ωᵀ * Tᵃᵗ) * δ) / D
    T★ = ifelse(D == 0, Ψₛ.temperature, T★)
    max_ΔT = convert(FT, st.max_ΔT)
    return Ψᵢ.T + clamp(T★ - Ψᵢ.T, -max_ΔT, max_ΔT)
end

"""
    SoilConductiveFlux(conductivity, thickness)

Conduction between a land surface temperature and the bulk slab temperature,
`𝒬ᵍʳᵒ = Λᵍʳᵒ (Tᵍʳᵒ − Tˡᵃ)` with `Λᵍʳᵒ = κᵀ/ℓᵀ` (W m⁻² K⁻¹). `ℓᵀ` (`thickness`, m) is the
distance from the surface to the bulk temperature node, half the slab depth for a
node at its center, and `κᵀ` (`conductivity`, W m⁻¹ K⁻¹) the thermal conductivity
over that distance: the soil's, or that of a thin surface layer (litter, a crust)
with its own conductivity. No layer of thickness `ℓᵀ` is resolved. Use it through
[`SoilSkinTemperature`](@ref).
"""
struct SoilConductiveFlux{K, Z}
    conductivity :: K   # κᵀ, W m⁻¹ K⁻¹
    thickness    :: Z   # ℓᵀ, m
end

@inline ground_conductance(F::SoilConductiveFlux) = F.conductivity / F.thickness

# Skin→bulk conductance of an interface temperature formulation that closes its own
# surface energy balance, or `nothing` for a bulk skin.
@inline ground_conductance(formulation) = nothing

# Solve the surface flux balance equation:
#   Qa(Tₛ) + Ωc (Tᵃᵗ - Tₛ) + (Tₛ - Tb) / R = 0
# where R is the total thermal resistance (h/k for bare ice, hₛ/kₛ + hᵢ/kᵢ with snow),
# Ωc = 𝒬ᵀ/(Tᵃᵗ-Tₛ) is the linearized sensible heat coefficient, and Qa = 𝒬ᵛ + ℐꜛˡʷ + Qd.
# The upward longwave ℐꜛˡʷ = σ ε Tₛ⁴ is strongly nonlinear in Tₛ; a pure Picard
# iteration (treating Qa constant) is unstable when 4σεTₛ³ ≳ 1/R (radiation
# dominated). We linearize: Qa(Tₛ) ≈ Qa(Tₛ⁻) + β (Tₛ − Tₛ⁻) with β = 4σεTₛ⁻³,
# yielding the Newton-like semi-implicit update:
#   Tₛ = [Tb + β R Tₛ⁻ - Ωc R Tᵃᵗ - Qa R] / [1 + β R - Ωc R]
@inline function conductive_flux_balance_temperature(st, R, Ψₛ, interface_formulation, 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd, Ψᵢ, ℙᵢ, Ψₐ, ℙₐ)
    hᵢ = Ψᵢ.hi
    hc = Ψᵢ.hc

    # Bottom temperature at the melting point
    Tb = ClimaSeaIce.SeaIceThermodynamics.melting_temperature(ℙᵢ.liquidus, Ψᵢ.S)
    Tb = convert_to_kelvin(ℙᵢ.temperature_units, Tb)
    Tₛ⁻ = Ψₛ.temperature

    Tᵃᵗ = surface_atmosphere_temperature(Ψₐ, ℙₐ)
    ΔT = Tᵃᵗ - Tₛ⁻
    Qa = 𝒬ᵛ + ℐꜛˡʷ + Qd

    # Sensible transfer coefficient Ωc = 𝒬ᵀ/ΔT, safely handling ΔT → 0.
    Ωc = ifelse(ΔT == zero(ΔT), zero(Tₛ⁻), 𝒬ᵀ / ΔT)

    # Newton linearization of upwelling longwave: ℐꜛˡʷ(Tₛ) ≈ ℐꜛˡʷ(Tₛ⁻) + β (Tₛ − Tₛ⁻).
    # Since ℐꜛˡʷ = σ ϵ Tₛ⁻⁴, we have β = 4 σ ϵ Tₛ⁻³ = 4 ℐꜛˡʷ / Tₛ⁻.
    β = 4 * ℐꜛˡʷ / Tₛ⁻

    # Flux balance solution with T⁴ linearization (stable even at ΔT = 0):
    D  = 1 + β * R - Ωc * R
    T★ = (Tb + β * R * Tₛ⁻ - Ωc * R * Tᵃᵗ - Qa * R) / D
    T★ = ifelse(D == 0, Tₛ⁻, T★)
    T★ = ifelse(isnan(T★), Tₛ⁻, T★)

    # Damped-Newton trust region: cap how far the skin moves in one iterate.
    ΔT★ = T★ - Tₛ⁻
    maximum_temperature_step = convert(typeof(T★), st.max_ΔT)
    Tₛ⁺ = Tₛ⁻ + clamp(ΔT★, -maximum_temperature_step, maximum_temperature_step)

    # Cap at melting temperature
    Tₘ = ℙᵢ.liquidus.freshwater_melting_temperature
    Tₘ = convert_to_kelvin(ℙᵢ.temperature_units, Tₘ)
    Tₛ⁺ = min(Tₛ⁺, Tₘ)

    # If ice is not consolidated, use the bottom temperature
    Tₛ⁺ = ifelse(hᵢ ≥ hc, Tₛ⁺, Tb)

    return Tₛ⁺
end

# Bare ice: R = hᵢ / kᵢ
@inline function flux_balance_temperature(st::SkinTemperature{<:ClimaSeaIce.ConductiveFlux},
                                          Ψₛ, interface_formulation, 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd, Ψᵢ, ℙᵢ, Ψₐ, ℙₐ)
    k  = st.internal_flux.conductivity
    R  = Ψᵢ.hi / k
    return conductive_flux_balance_temperature(st, R, Ψₛ, interface_formulation, 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd, Ψᵢ, ℙᵢ, Ψₐ, ℙₐ)
end

# Snow + ice: R = hₛ / kₛ + hᵢ / kᵢ
@inline function flux_balance_temperature(st::SkinTemperature{<:ClimaSeaIce.SeaIceThermodynamics.IceSnowConductiveFlux},
                                          Ψₛ, interface_formulation, 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd, Ψᵢ, ℙᵢ, Ψₐ, ℙₐ)
    F  = st.internal_flux
    R  = Ψᵢ.hs / F.snow_conductivity + Ψᵢ.hi / F.ice_conductivity
    return conductive_flux_balance_temperature(st, R, Ψₛ, interface_formulation, 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd, Ψᵢ, ℙᵢ, Ψₐ, ℙₐ)
end

# Turbulent (sensible + latent, positive up) and radiative (upwelling longwave
# `ℐꜛˡʷ`, and the net-downward non-turbulent term `Qd`) fluxes at the previous
# iterate. Shared by the `SkinTemperature` and `EnergyBalanceTemperature`
# diagnostic temperature solves. `radiation_state` carries zero-valued σ/α/ϵ/SW/LW
# when radiation is off.
@inline function skin_surface_fluxes(interface_state, atmosphere_state, radiation_state, atmosphere_properties)
    ℂᵃᵗ = atmosphere_properties.thermodynamics_parameters
    Tᵃᵗ = atmosphere_state.T
    pᵃᵗ = atmosphere_state.p
    qᵃᵗ = atmosphere_state.q
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)
    cᵃᵗ = AtmosphericThermodynamics.cp_m(ℂᵃᵗ, qᵃᵗ) # moist heat capacity
    ℒⁱ  = interface_latent_heat(ℂᵃᵗ, Tᵃᵗ, interface_state)

    Tₛ⁻ = interface_state.temperature
    σ = radiation_state.σ
    ϵ = radiation_state.ϵ
    α = radiation_state.α
    ℐꜜˢʷ = radiation_state.ℐꜜˢʷ
    ℐꜜˡʷ = radiation_state.ℐꜜˡʷ
    ℐꜛˡʷ = σ * ϵ * Tₛ⁻^4
    Qd = (1 - α) * ℐꜜˢʷ + ϵ * ℐꜜˡʷ

    u★ = interface_state.fluxes.u★
    θ★ = interface_state.fluxes.θ★
    q★ = interface_state.fluxes.q★

    # Turbulent heat fluxes, sensible + latent (positive out of the surface)
    𝒬ᵀ = - ρᵃᵗ * cᵃᵗ * u★ * θ★
    𝒬ᵛ = - ρᵃᵗ * ℒⁱ * u★ * q★

    return 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd
end

@inline function compute_interface_temperature(st::SkinTemperature,
                                               interface_state,
                                               atmosphere_state,
                                               interior_state,
                                               radiation_state,
                                               interface_formulation,
                                               atmosphere_properties,
                                               interior_properties)

    𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd = skin_surface_fluxes(interface_state, atmosphere_state,
                                            radiation_state, atmosphere_properties)

    return flux_balance_temperature(st,
                                    interface_state,
                                    interface_formulation,
                                    𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd,
                                    interior_state,
                                    interior_properties,
                                    atmosphere_state,
                                    atmosphere_properties)
end

"""
$(TYPEDSIGNATURES)

Interface temperature from a surface energy balance with conductive coupling to the
bulk land. Without land skin storage, solve the equilibrium balance. With land skin
storage, evaluate the implicit energy balance over the coupled time step.
"""
struct EnergyBalanceTemperature{C}
    coupling :: C
end

@inline ground_conductance(t::EnergyBalanceTemperature) = ground_conductance(t.coupling)
Adapt.@adapt_structure EnergyBalanceTemperature

"""
$(TYPEDSIGNATURES)

Ground energy balance with conductance `conductivity/thickness` (W m⁻² K⁻¹)
to the bulk land. Configure `PrognosticSkin` on the land to include ground heat storage.
"""
SoilSkinTemperature(conductivity, thickness) = EnergyBalanceTemperature(SoilConductiveFlux(conductivity, thickness))

Base.summary(t::EnergyBalanceTemperature) = string("EnergyBalanceTemperature(Λ=", prettysummary(ground_conductance(t.coupling)), ")")
Base.show(io::IO, t::EnergyBalanceTemperature) = print(io, summary(t))

# The surface energy imbalance F(T) = net radiation − 𝒬ᵍʳᵒ − 𝒬ᵀ − 𝒬ᵛ at skin temperature T
# and the conductance sum Σg = −dF/dT, with the similarity scales frozen at the current
# iterate.
@inline function skin_energy_imbalance(T, t, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₛ, ℙₐ)
    FT  = eltype(Ψₛ)
    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    ℒ   = interface_latent_heat(ℂᵃᵗ, Ψₐ.T, Ψₛ)
    gᵃᵀ = aerodynamic_heat_conductance(Ψₛ, Ψₐ, ℂᵃᵗ)
    gᵃᵛ = aerodynamic_vapor_conductance(Ψₛ, Ψₐ, ℂᵃᵗ)
    Λ   = convert(FT, ground_conductance(t))
    Tᵃᵗ = surface_atmosphere_temperature(Ψₐ, ℙₐ)

    q             = compute_interface_humidity(specific_humidity_formulation(ℙₛ.state), T, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ)
    net_radiation = - (1 - Ψᵣ.α) * Ψᵣ.ℐꜜˢʷ - Ψᵣ.ϵ * (Ψᵣ.ℐꜜˡʷ + Ψᵣ.σ * T^4)
    𝒬ᵍʳᵒ          = Λ * (T - Ψˡᵃ.T)
    𝒬ᵀ            = gᵃᵀ * (T - Tᵃᵗ)
    𝒬ᵛ            = ℒ * gᵃᵛ * (q - Ψₐ.q)
    F             = net_radiation - 𝒬ᵍʳᵒ - 𝒬ᵀ - 𝒬ᵛ

    # dq/dT of the humidity the formulation actually diagnoses, by a one-sided difference.
    δT = sqrt(eps(FT)) * T
    dq = (compute_interface_humidity(specific_humidity_formulation(ℙₛ.state), T + δT, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ) - q) / δT
    Σg = 4 * Ψᵣ.ϵ * Ψᵣ.σ * T^3 + Λ + gᵃᵀ + ℒ * gᵃᵛ * dq
    return F, Σg
end

# A massless skin takes one Newton step on F(Tₛ) = 0 per iterate of the Monin–Obukhov
# fixed point, so the skin and the similarity scales converge jointly.
@inline function compute_interface_temperature(t::EnergyBalanceTemperature,
                                               interface_state,
                                               atmosphere_state,
                                               interior_state,
                                               radiation_state,
                                               interface_formulation,
                                               atmosphere_properties,
                                               interior_properties,
                                               storage::Nothing = interior_state.surface_storage)

    T = interface_state.temperature
    F, Σg = skin_energy_imbalance(T, t, interface_state, atmosphere_state, interior_state,
                                  radiation_state, interface_formulation, atmosphere_properties)
    # Σg = 0 leaves F independent of T: no root to step toward.
    return ifelse(Σg > 0, T + F / Σg, T)
end

# A prognostic skin is model state: frozen through the fixed point (only the
# similarity scales iterate) and advanced once per step after the flux solve.
@inline compute_interface_temperature(::EnergyBalanceTemperature,
                                      interface_state,
                                      atmosphere_state,
                                      interior_state,
                                      radiation_state,
                                      interface_formulation,
                                      atmosphere_properties,
                                      interior_properties, storage) = interface_state.temperature

####
#### Interface specific humidity
####

# Diagnostic formulations (`ImpureSaturationSpecificHumidity`): qˢ is an explicit
# function of the interface temperature `Tₛ` and the surface scalar (salinity) from
# `humidity_surface_scalar`. The interior state `Ψᵢ` and radiation state `Ψᵣ` are ignored.
@inline compute_interface_humidity(q_formulation, Tₛ, Ψₛ, Ψₐ, Ψᵢ, Ψᵣ, ℙₐ) =
    surface_specific_humidity(q_formulation, ℙₐ.thermodynamics_parameters, Ψₐ.p, Tₛ, humidity_surface_scalar(Ψₛ), Ψₐ.q)

# `BulkHumidity`: saturated at `Tₛ` where the land saturation `𝒮` is positive.
@inline compute_interface_humidity(q::BulkHumidity, Tₛ, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ) =
    surface_specific_humidity(q, ℙₐ.thermodynamics_parameters, Ψₐ.p, Tₛ, Ψˡᵃ.𝒮)

# `FractionalHumidity`: qˢ = β · qᵛ⁺(Tₛ) at the skin temperature, with the
# evaporation efficiency β derived from the land state.
@inline function compute_interface_humidity(q::FractionalHumidity, Tₛ, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ)
    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    FT  = eltype(Ψₛ)
    β   = evaporation_efficiency(q.efficiency, Ψˡᵃ)
    qᵛ⁺ = saturation_specific_humidity(ℂᵃᵗ, Tₛ, Ψₐ.p, q.phase)
    return convert(FT, β * qᵛ⁺)
end

# `SkinHumidity`: solve the surface vapor-flux balance for qˢ. The soil delivers
# vapor by diffusion from the saturation depth `d`,
#
#     Jˢᵒⁱˡ = gˢ (qᵛ⁺ - qˢ),     gˢ = κ^q / d   (positive upward),
#
# which must equal the atmospheric vapor flux carried away by turbulence,
# linearized as `Jᵃ(q) = Gᵃ (q - qᵃᵗ)` with the previous iterate's aerodynamic
# conductance `Gᵃ = ρᵃᵗ u★ χq` (see `aerodynamic_vapor_conductance`). The
# balance gˢ(qᵛ⁺ - qˢ) = Gᵃ(qˢ - qᵃᵗ) gives the series solution
#
#     qˢ = (gˢ qᵛ⁺ + Gᵃ qᵃᵗ) / (gˢ + Gᵃ).
#
# The reservoir is saturated at the *bulk land* temperature `Tᵈ = Tˡᵃ`, not the skin
# temperature: the saturated soil sits at depth `d` below the surface, so its temperature is the deep soil
# temperature — the same deep endpoint the conductive heat flux uses. `Tₛ` is
# therefore unused here (`qˢ` is decoupled from the skin temperature, as a dry
# skin implies).
@inline function compute_interface_humidity(q::SkinHumidity, Tₛ, Ψₛ, Ψₐ, Ψˡᵃ, Ψᵣ, ℙₐ)
    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    FT  = eltype(Ψₛ)
    pᵃᵗ = Ψₐ.p
    qᵃᵗ = Ψₐ.q

    Tᵈ  = Ψˡᵃ.T # bulk land temperature at the saturation depth `d`
    qᵛ⁺ = saturation_specific_humidity(ℂᵃᵗ, Tᵈ, pᵃᵗ, q.phase)

    d  = surface_layer_thickness(q.surface_thickness, Ψˡᵃ)
    κ  = q.vapor_diffusivity
    gˢ = κ / d # soil vapor conductance

    Gᵃ = aerodynamic_vapor_conductance(Ψₛ, Ψₐ, ℂᵃᵗ)
    qˢ = conductance_weighted_node(Ψₛ.specific_humidity, (gˢ, Gᵃ), (qᵛ⁺, qᵃᵗ))

    return convert(FT, qˢ)
end

######
###### Interface state
######

"""
    InterfaceFluxScales{FT}

The solved similarity-theory characteristic scales at an interface: friction
velocity `u★`, temperature flux scale `θ★`, and specific-humidity flux scale
`q★`, together with the transfer coefficients `χθ` and `χq` of the iterate that
produced them (`θ★ = χθ Δθ`, `q★ = χq Δq`), from which the aerodynamic
conductances `ρ cᵖᵐ u★ χθ` (heat) and `ρ u★ χq` (vapor) follow. The
three-argument constructor zeroes the transfer coefficients (an initial guess).
"""
struct InterfaceFluxScales{FT}
    u★ :: FT
    θ★ :: FT
    q★ :: FT
    χθ :: FT
    χq :: FT
end

@inline InterfaceFluxScales(u★, θ★, q★) = InterfaceFluxScales(u★, θ★, q★, zero(u★), zero(u★))

Base.eltype(::InterfaceFluxScales{FT}) where FT = FT

"""
    InterfaceVelocities{FT}

The interface velocity `(u, v)` — the ocean surface current, or zero over land.
"""
struct InterfaceVelocities{FT}
    u :: FT
    v :: FT
end

"""
    abstract type AbstractInterfaceState{FT}

Interface state carried through the similarity-theory fixed-point solver
(`compute_interface_state`). Concrete subtypes share the iterated quantities —
`fluxes` (`u★, θ★, q★`), `velocities` (`u, v`), `temperature` (the skin
temperature), and `specific_humidity` (`qˢ`) — and differ only in the surface
property each interface needs: `salinity` for air–sea, the land `hydrology` /
`energy` state for air–land.
"""
abstract type AbstractInterfaceState{FT} end

Base.eltype(::AbstractInterfaceState{FT}) where FT = FT

# Interface velocity components (see `velocity_difference`).
@inline x_velocity(Ψ::AbstractInterfaceState) = Ψ.velocities.u
@inline y_velocity(Ψ::AbstractInterfaceState) = Ψ.velocities.v

"""
    AirSeaInterfaceState{FT}

Air–sea (ocean and sea-ice) interface state. Carries `salinity`, used by
`ImpureSaturationSpecificHumidity` for the Raoult reduction of saturation.
"""
struct AirSeaInterfaceState{FT} <: AbstractInterfaceState{FT}
    fluxes            :: InterfaceFluxScales{FT}
    velocities        :: InterfaceVelocities{FT}
    temperature       :: FT
    specific_humidity :: FT
    salinity          :: FT
end

@inline AirSeaInterfaceState(u★, θ★, q★, u, v, T, S, q) =
    AirSeaInterfaceState(InterfaceFluxScales(u★, θ★, q★), InterfaceVelocities(u, v), T, q, S)

@inline humidity_surface_scalar(Ψ::AirSeaInterfaceState) = Ψ.salinity

"""
    AirIceInterfaceState{FT}

Air–sea-ice interface state. Sublimation is over *fresh* ice, so it carries no
salinity (the Ice-phase saturation involves none, and the melting-point salinity
the skin-temperature solve needs comes from the interior state). The humidity
scalar is therefore zero.
"""
struct AirIceInterfaceState{FT} <: AbstractInterfaceState{FT}
    fluxes            :: InterfaceFluxScales{FT}
    velocities        :: InterfaceVelocities{FT}
    temperature       :: FT
    specific_humidity :: FT
end

@inline AirIceInterfaceState(u★, θ★, q★, u, v, T, q) =
    AirIceInterfaceState(InterfaceFluxScales(u★, θ★, q★), InterfaceVelocities(u, v), T, q)

@inline humidity_surface_scalar(Ψ::AirIceInterfaceState) = zero(eltype(Ψ))

"""
    AirLandInterfaceState{FT}

Air–land interface state: the iterated scales, velocities, temperature, and specific
humidity. The land state the surface models read (e.g. `T`, `𝒮`) is not carried here;
it enters the solver as the land state `Ψˡᵃ`.
"""
struct AirLandInterfaceState{FT} <: AbstractInterfaceState{FT}
    fluxes            :: InterfaceFluxScales{FT}
    velocities        :: InterfaceVelocities{FT}
    temperature       :: FT
    specific_humidity :: FT
end

@inline AirLandInterfaceState(u★, θ★, q★, u, v, T, q) =
    AirLandInterfaceState(InterfaceFluxScales(u★, θ★, q★), InterfaceVelocities(u, v), T, q)

# Latent heat of the surface vapor flux entering the `SkinTemperature` energy
# balance. Land evaporates liquid water (vaporization); ocean and sea ice keep
# the existing sublimation value so their skin solves are unchanged.
@inline interface_latent_heat(ℂ, T, ::AbstractInterfaceState) = AtmosphericThermodynamics.latent_heat_sublim(ℂ, T)
@inline interface_latent_heat(ℂ, T, ::AirLandInterfaceState)  = AtmosphericThermodynamics.latent_heat_vapor(ℂ, T)

# Rebuild the next iterate, carrying the fixed per-surface state forward.
@inline rebuild_interface_state(Ψ⁻::AirSeaInterfaceState, fluxes, T, q) =
    AirSeaInterfaceState(fluxes, Ψ⁻.velocities, T, q, Ψ⁻.salinity)

@inline rebuild_interface_state(Ψ⁻::AirIceInterfaceState, fluxes, T, q) =
    AirIceInterfaceState(fluxes, Ψ⁻.velocities, T, q)

@inline rebuild_interface_state(Ψ⁻::AirLandInterfaceState, fluxes, T, q) =
    AirLandInterfaceState(fluxes, Ψ⁻.velocities, T, q)

# The next iterate at the node `(T, q)`, its scales re-evaluated there from the previous
# iterate's transfer coefficients.
@inline function rebuild_interface_state(Ψ⁻, Ψₐ, ℙₐ, T, q)
    FT  = eltype(Ψ⁻)
    θᵃᵗ = convert(FT, surface_atmosphere_temperature(Ψₐ, ℙₐ))
    χθ  = Ψ⁻.fluxes.χθ
    χq  = Ψ⁻.fluxes.χq
    fluxes = InterfaceFluxScales(Ψ⁻.fluxes.u★, convert(FT, χθ * (θᵃᵗ - T)), convert(FT, χq * (Ψₐ.q - q)), χθ, χq)
    return rebuild_interface_state(Ψ⁻, fluxes, convert(FT, T), convert(FT, q))
end

function Base.show(io::IO, Ψ::AbstractInterfaceState)
    print(io, nameof(typeof(Ψ)), "(",
          "u★=", prettysummary(Ψ.fluxes.u★), " ",
          "θ★=", prettysummary(Ψ.fluxes.θ★), " ",
          "q★=", prettysummary(Ψ.fluxes.q★), " ",
          "u=", prettysummary(Ψ.velocities.u), " ",
          "v=", prettysummary(Ψ.velocities.v), " ",
          "T=", prettysummary(Ψ.temperature), " ",
          "q=", prettysummary(Ψ.specific_humidity), ")")
end

@inline zero_interface_state(FT) = AirSeaInterfaceState(zero(FT), zero(FT), zero(FT),
                                                        zero(FT), zero(FT),
                                                        convert(FT, 273.15),
                                                        zero(FT), zero(FT))

"""
    AirLandRadiationState{FT}

Air-land interface radiation state at one cell: Stefan–Boltzmann constant `σ`,
surface albedo `α`, emissivity `ϵ`, downwelling shortwave `ℐꜜˢʷ`, and
downwelling longwave `ℐꜜˡʷ`, in W m⁻² with positive-upward signs.
"""
struct AirLandRadiationState{FT}
    σ    :: FT
    α    :: FT
    ϵ    :: FT
    ℐꜜˢʷ :: FT
    ℐꜜˡʷ :: FT
end

Base.eltype(::AirLandRadiationState{FT}) where FT = FT
