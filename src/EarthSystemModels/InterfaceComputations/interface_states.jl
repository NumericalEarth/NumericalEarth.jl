using ClimaSeaIce: ClimaSeaIce
using Thermodynamics: Thermodynamics as AtmosphericThermodynamics

#####
##### Interface properties
#####

struct InterfaceProperties{Q, T, V}
    specific_humidity_formulation :: Q
    temperature_formulation :: T
    velocity_formulation :: V
end

#####
##### Interface specific humidity formulations
#####

# TODO: allow different saturation models
# struct ClasiusClapyeronSaturation end
struct ImpureSaturationSpecificHumidity{Φ, X}
    # saturation :: S
    phase :: Φ
    water_mole_fraction :: X
end

function Base.summary(q★::ImpureSaturationSpecificHumidity)
    phase_str = if q★.phase == AtmosphericThermodynamics.Ice()
        "Ice"
    elseif q★.phase == AtmosphericThermodynamics.Liquid()
        "Liquid"
    end

    return string("ImpureSaturationSpecificHumidity{$phase_str}(water_mole_fraction=",
                  prettysummary(q★.water_mole_fraction), ")")
end

Base.show(io::IO, q★::ImpureSaturationSpecificHumidity) = print(io, summary(q★))

"""
    ImpureSaturationSpecificHumidity(phase [, water_mole_fraction=1])

Return the formulation for computing specific humidity at an interface.
"""
ImpureSaturationSpecificHumidity(phase) = ImpureSaturationSpecificHumidity(phase, nothing)

@inline compute_water_mole_fraction(::Nothing, salinity) = 1
@inline compute_water_mole_fraction(x_H₂O::Number, salinity) = x_H₂O

# COARE 3.6 / Edson (2013) pressure-based saturation specific humidity:
#   qₛ = εᵈᵛ⁻¹ pᵛ⁺ / (p − (1 − εᵈᵛ⁻¹) pᵛ⁺),   εᵈᵛ⁻¹ = Rᵈ / Rᵥ
# Direct evaluation at the atmospheric pressure p. The 6th positional
# argument `qᵃᵗ` is accepted (and ignored) so the same call site can
# dispatch on either `ImpureSaturationSpecificHumidity` or
# [`BulkHumidity`](@ref), which does need it.
@inline function surface_specific_humidity(formulation::ImpureSaturationSpecificHumidity, ℂᵃᵗ, pᵃᵗ, Tₛ, Sₛ=zero(Tₛ), qᵃᵗ=zero(Tₛ))
    FT = eltype(Tₛ)
    CT = eltype(ℂᵃᵗ)
    T  = convert(CT, Tₛ)
    p  = convert(CT, pᵃᵗ)

    # Raoult's law on the saturation vapor pressure.
    χ_H₂O = compute_water_mole_fraction(formulation.water_mole_fraction, Sₛ)
    pᵛ⁺   = χ_H₂O * AtmosphericThermodynamics.saturation_vapor_pressure(ℂᵃᵗ, T, formulation.phase)
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
#   qᵛ⁺ = εᵈᵛ⁻¹ pᵛ⁺ / (p − (1 − εᵈᵛ⁻¹) pᵛ⁺),   εᵈᵛ⁻¹ = Rᵈ / Rᵥ.
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
# The saturation arrives via `humidity_surface_scalar(AirLandInterfaceState)`
# (`Ψ.hydrology.saturation`). `BulkHumidity` has no moisture-availability
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
    struct CriticalSaturation

Evaporation efficiency after [Manabe (1969)](@cite manabe1969climate): the surface is saturated (`β = 1`) above a
critical saturation `𝒮ᶜ`, and the efficiency falls off linearly below it,

```math
β(𝒮) = \\min(𝒮 / 𝒮ᶜ, 1),   𝒮 = Mˡᵃ / Mˡᵃ⁺.
```

Used as the `efficiency` of [`FractionalHumidity`](@ref). The type declares its
land-state dependency (the saturation `𝒮`); the interface materializes exactly
that into the land interface state.
"""
struct CriticalSaturation{FT}
    critical_saturation :: FT
end

@inline function evaporation_efficiency(𝒮ᶜ::CriticalSaturation, hydrology)
    𝒮 = hydrology.saturation
    return min(𝒮 / convert(typeof(𝒮), 𝒮ᶜ.critical_saturation), one(𝒮))
end

# Constant efficiency — a uniformly sub-saturated surface; reads no land state.
@inline evaporation_efficiency(β::Number, hydrology) = β

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
surface, the humidity analogue of [`SkinTemperature`](@ref).

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
# a future `WetnessDependentSurfaceThickness` will dispatch here on the land
# water state carried by the interface state.
@inline surface_layer_thickness(d::Number, Ψₛ) = d

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
@inline function flux_balance_temperature(st::SkinTemperature{<:DiffusiveFlux}, Ψₛ, ℙₛ, 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd, Ψᵢ, ℙᵢ, Ψₐ, ℙₐ)
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

# Solve the surface flux balance equation:
#   Qa(Tₛ) + Ωc (Tᵃᵗ - Tₛ) + (Tₛ - Tᵦ) / R = 0
# where R is the total thermal resistance (h/k for bare ice, hₛ/kₛ + hᵢ/kᵢ with snow),
# Ωc = 𝒬ᵀ/(Tᵃᵗ-Tₛ) is the linearized sensible heat coefficient, and Qa = 𝒬ᵛ + ℐꜛˡʷ + Qd.
# The upward longwave ℐꜛˡʷ = σ ε Tₛ⁴ is strongly nonlinear in Tₛ; a pure Picard
# iteration (treating Qa constant) is unstable when 4σεTₛ³ ≳ 1/R (radiation
# dominated). We linearize: Qa(Tₛ) ≈ Qa(Tₛ⁻) + β (Tₛ − Tₛ⁻) with β = 4σεTₛ⁻³,
# yielding the Newton-like semi-implicit update:
#   Tₛ = [Tᵦ + β R Tₛ⁻ - Ωc R Tᵃᵗ - Qa R] / [1 + β R - Ωc R]
@inline function conductive_flux_balance_temperature(st, R, Ψₛ, ℙₛ, 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd, Ψᵢ, ℙᵢ, Ψₐ, ℙₐ)
    hᵢ = Ψᵢ.hi
    hc = Ψᵢ.hc

    # Bottom temperature at the melting point
    Tᵦ = ClimaSeaIce.SeaIceThermodynamics.melting_temperature(ℙᵢ.liquidus, Ψᵢ.S)
    Tᵦ = convert_to_kelvin(ℙᵢ.temperature_units, Tᵦ)
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
    T★ = (Tᵦ + β * R * Tₛ⁻ - Ωc * R * Tᵃᵗ - Qa * R) / D
    T★ = ifelse(D == 0, Tₛ⁻, T★)
    T★ = ifelse(isnan(T★), Tₛ⁻, T★)

    # Cap the temperature step for iteration stability
    ΔT★ = T★ - Tₛ⁻
    max_ΔT = convert(typeof(T★), st.max_ΔT)
    Tₛ⁺ = Tₛ⁻ + clamp(ΔT★, -max_ΔT, max_ΔT)

    # Cap at melting temperature
    Tₘ = ℙᵢ.liquidus.freshwater_melting_temperature
    Tₘ = convert_to_kelvin(ℙᵢ.temperature_units, Tₘ)
    Tₛ⁺ = min(Tₛ⁺, Tₘ)

    # If ice is not consolidated, use the bottom temperature
    Tₛ⁺ = ifelse(hᵢ ≥ hc, Tₛ⁺, Tᵦ)

    return Tₛ⁺
end

# Bare ice: R = hᵢ / kᵢ
@inline function flux_balance_temperature(st::SkinTemperature{<:ClimaSeaIce.ConductiveFlux},
                                          Ψₛ, ℙₛ, 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd, Ψᵢ, ℙᵢ, Ψₐ, ℙₐ)
    k  = st.internal_flux.conductivity
    R  = Ψᵢ.hi / k
    return conductive_flux_balance_temperature(st, R, Ψₛ, ℙₛ, 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd, Ψᵢ, ℙᵢ, Ψₐ, ℙₐ)
end

# Snow + ice: R = hₛ / kₛ + hᵢ / kᵢ
@inline function flux_balance_temperature(st::SkinTemperature{<:ClimaSeaIce.SeaIceThermodynamics.IceSnowConductiveFlux},
                                          Ψₛ, ℙₛ, 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd, Ψᵢ, ℙᵢ, Ψₐ, ℙₐ)
    F  = st.internal_flux
    R  = Ψᵢ.hs / F.snow_conductivity + Ψᵢ.hi / F.ice_conductivity
    return conductive_flux_balance_temperature(st, R, Ψₛ, ℙₛ, 𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd, Ψᵢ, ℙᵢ, Ψₐ, ℙₐ)
end

@inline function compute_interface_temperature(st::SkinTemperature,
                                               interface_state,
                                               atmosphere_state,
                                               interior_state,
                                               radiation_state,
                                               interface_properties,
                                               atmosphere_properties,
                                               interior_properties)

    ℂᵃᵗ = atmosphere_properties.thermodynamics_parameters
    Tᵃᵗ = atmosphere_state.T
    pᵃᵗ = atmosphere_state.p
    qᵃᵗ = atmosphere_state.q
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)
    cᵃᵗ = AtmosphericThermodynamics.cp_m(ℂᵃᵗ, qᵃᵗ) # moist heat capacity

    # TODO: this depends on the phase of the interface
    #ℰv = 0 #AtmosphericThermodynamics.latent_heat_vapor(ℂᵃᵗ, Tᵃᵗ)
    ℒⁱ = AtmosphericThermodynamics.latent_heat_sublim(ℂᵃᵗ, Tᵃᵗ)

    # upwelling radiation is calculated explicitly. radiation_state is
    # produced by `air_sea_interface_radiation_state` (or its sea-ice
    # variant) and contains zero-valued σ/α/ϵ/SW/LW when radiation is off.
    Tₛ⁻ = interface_state.temperature # approximate interface temperature from previous iteration
    σ = radiation_state.σ
    ϵ = radiation_state.ϵ
    α = radiation_state.α

    ℐꜜˢʷ = radiation_state.ℐꜜˢʷ
    ℐꜜˡʷ = radiation_state.ℐꜜˡʷ
    ℐꜛˡʷ = σ * ϵ * Tₛ⁻^4
    Qd = - (1 - α) * ℐꜜˢʷ - ϵ * ℐꜜˡʷ

    u★ = interface_state.fluxes.u★
    θ★ = interface_state.fluxes.θ★
    q★ = interface_state.fluxes.q★

    # Turbulent heat fluxes, sensible + latent (positive out of the ocean)
    𝒬ᵀ = - ρᵃᵗ * cᵃᵗ * u★ * θ★ # = - ρᵃᵗ cᵃᵗ u★ Ch / sqrt(Cd) * (θᵃᵗ - Tₛ)
    𝒬ᵛ = - ρᵃᵗ * ℒⁱ * u★ * q★

    Tₛ = flux_balance_temperature(st,
                                  interface_state,
                                  interface_properties,
                                  𝒬ᵀ, 𝒬ᵛ, ℐꜛˡʷ, Qd,
                                  interior_state,
                                  interior_properties,
                                  atmosphere_state,
                                  atmosphere_properties)

    return Tₛ
end

####
#### Interface specific humidity
####

# Diagnostic formulations (`ImpureSaturationSpecificHumidity`, `BulkHumidity`):
# qˢ is an explicit function of the interface temperature `Tₛ` and the surface
# scalar (salinity / saturation `𝒮`) from `humidity_surface_scalar`. The interior
# state `Ψᵢ` is ignored.
@inline compute_interface_humidity(q_formulation, Tₛ, Ψₛ, Ψₐ, Ψᵢ, ℙₐ) =
    surface_specific_humidity(q_formulation, ℙₐ.thermodynamics_parameters, Ψₐ.p, Tₛ, humidity_surface_scalar(Ψₛ), Ψₐ.q)

# `FractionalHumidity`: qˢ = β · qᵛ⁺(Tₛ) at the skin temperature, with the
# evaporation efficiency β derived from the materialized hydrology state.
@inline function compute_interface_humidity(q::FractionalHumidity, Tₛ, Ψₛ, Ψₐ, Ψᵢ, ℙₐ)
    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    FT  = eltype(Ψₛ)
    β   = evaporation_efficiency(q.efficiency, Ψₛ.hydrology)
    qᵛ⁺ = saturation_specific_humidity(ℂᵃᵗ, Tₛ, Ψₐ.p, q.phase)
    return convert(FT, β * qᵛ⁺)
end

# `SkinHumidity`: solve the surface vapor-flux balance for qˢ. The soil delivers
# vapor by diffusion from the saturation depth `d`,
#
#     Jˢᵒⁱˡ = gˢ (qᵛ⁺ - qˢ),     gˢ = κ^q / d   (positive upward),
#
# which must equal the atmospheric vapor flux carried away by turbulence,
#
#     Jᵃ = - ρᵃᵗ u★ q★           (positive upward),
#
# evaluated at the previous iterate. Writing Jᵃ = Ωq (qˢ - qᵃᵗ) with the implicit
# coefficient Ωq = Jᵃ / (qˢ⁻ - qᵃᵗ) (the SkinTemperature trick — no prescribed
# conductance), the balance gˢ(qᵛ⁺ - qˢ) = Ωq(qˢ - qᵃᵗ) gives
#
#     qˢ = (gˢ qᵛ⁺ + Ωq qᵃᵗ) / (gˢ + Ωq).
#
# Multiplying through by Δq ≡ qˢ⁻ - qᵃᵗ (so Ωq Δq = Jᵃ) removes the division and
# stays finite as Δq → 0:
#
#     qˢ = (gˢ qᵛ⁺ Δq + Jᵃ qᵃᵗ) / (gˢ Δq + Jᵃ).
#
# The reservoir is saturated at the *bulk land* temperature `Tᵈ` (the energy
# component of the interface state), not the skin temperature: the saturated soil
# sits at depth `d` below the surface, so its temperature is the deep soil
# temperature — the same deep endpoint the conductive heat flux uses. `Tₛ` is
# therefore unused here (`qˢ` is decoupled from the skin temperature, as a dry
# skin implies).
@inline function compute_interface_humidity(q::SkinHumidity, Tₛ, Ψₛ, Ψₐ, Ψᵢ, ℙₐ)
    ℂᵃᵗ = ℙₐ.thermodynamics_parameters
    FT  = eltype(Ψₛ)
    pᵃᵗ = Ψₐ.p
    qᵃᵗ = Ψₐ.q
    Tᵃᵗ = Ψₐ.T
    ρᵃᵗ = AtmosphericThermodynamics.air_density(ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)

    Tᵈ  = Ψₛ.energy.temperature # bulk land temperature at the saturation depth `d`
    qᵛ⁺ = saturation_specific_humidity(ℂᵃᵗ, Tᵈ, pᵃᵗ, q.phase)

    d  = surface_layer_thickness(q.surface_thickness, Ψₛ)
    κ  = q.vapor_diffusivity
    gˢ = κ / d # soil vapor conductance

    u★  = Ψₛ.fluxes.u★
    q★  = Ψₛ.fluxes.q★
    qˢ⁻ = Ψₛ.specific_humidity

    Jᵃ = - ρᵃᵗ * u★ * q★ # atmospheric vapor flux (positive upward), previous iterate
    Δq = qˢ⁻ - qᵃᵗ
    D  = gˢ * Δq + Jᵃ
    qˢ = (gˢ * qᵛ⁺ * Δq + Jᵃ * qᵃᵗ) / D

    return convert(FT, ifelse(D == 0, qˢ⁻, qˢ))
end

######
###### Interface state
######

"""
    InterfaceFluxScales{FT}

The solved similarity-theory characteristic scales at an interface: friction
velocity `u★`, temperature flux scale `θ★`, and specific-humidity flux scale
`q★`. Shared by every interface-state type.
"""
struct InterfaceFluxScales{FT}
    u★ :: FT
    θ★ :: FT
    q★ :: FT
end

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
    AirLandInterfaceState{FT, H, E}

Air–land interface state. In place of salinity it carries the land's `hydrology`
and `energy` surface state (e.g. `(saturation = 𝒮,)` and `(temperature = Tᵢ,)`),
from which the surface humidity models derive what they need — the moisture
availability `β`, the reservoir temperature, etc. `β` is *not* stored: it is
`evaporation_efficiency(efficiency, saturation)`, computed by the formulation.
"""
struct AirLandInterfaceState{FT, H, E} <: AbstractInterfaceState{FT}
    fluxes            :: InterfaceFluxScales{FT}
    velocities        :: InterfaceVelocities{FT}
    temperature       :: FT
    specific_humidity :: FT
    hydrology         :: H
    energy            :: E
end

@inline AirLandInterfaceState(u★, θ★, q★, u, v, T, q, hydrology, energy) =
    AirLandInterfaceState(InterfaceFluxScales(u★, θ★, q★), InterfaceVelocities(u, v), T, q, hydrology, energy)

# (i, j, grid)-first convenience constructor — pulls the per-cell land
# energy/hydrology substate from `land_state` via the humidity formulation, so
# the kernel call site stays compact. `Tₛ` and `qₛ` are passed in because they
# typically share computation with the atmosphere thermodynamics at the call
# site (e.g. the saturation humidity needs `Tₛ`, `pᵃᵗ`, and `ℂᵃᵗ`).
@inline function AirLandInterfaceState(i, j, grid,
                                       fluxes::InterfaceFluxScales,
                                       velocities::InterfaceVelocities,
                                       q_formulation,
                                       land_state,
                                       Tₛ, qₛ)
    # `typeof(Tₛ)`, not `eltype(grid)`, for Reactant traced-grid compatibility.
    FT  = typeof(Tₛ)
    energy    = interface_energy_state(i, j, grid, q_formulation, land_state)
    hydrology = interface_hydrology_state(i, j, grid, q_formulation, land_state)
    return AirLandInterfaceState(fluxes, velocities, convert(FT, Tₛ), convert(FT, qₛ), hydrology, energy)
end

@inline humidity_surface_scalar(Ψ::AirLandInterfaceState) = Ψ.hydrology.saturation

# Rebuild the next iterate, carrying the fixed per-surface state forward.
@inline rebuild_interface_state(Ψ⁻::AirSeaInterfaceState, fluxes, T, q) =
    AirSeaInterfaceState(fluxes, Ψ⁻.velocities, T, q, Ψ⁻.salinity)

@inline rebuild_interface_state(Ψ⁻::AirIceInterfaceState, fluxes, T, q) =
    AirIceInterfaceState(fluxes, Ψ⁻.velocities, T, q)

@inline rebuild_interface_state(Ψ⁻::AirLandInterfaceState, fluxes, T, q) =
    AirLandInterfaceState(fluxes, Ψ⁻.velocities, T, q, Ψ⁻.hydrology, Ψ⁻.energy)

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
downwelling longwave `ℐꜜˡʷ`. Returned by `air_land_interface_radiation_state`
and consumed by the air-land flux kernel and `apply_air_land_radiative_fluxes!`.
"""
struct AirLandRadiationState{FT}
    σ    :: FT
    α    :: FT
    ϵ    :: FT
    ℐꜜˢʷ :: FT
    ℐꜜˡʷ :: FT
end

Base.eltype(::AirLandRadiationState{FT}) where FT = FT
