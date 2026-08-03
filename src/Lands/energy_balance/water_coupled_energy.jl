#####
##### `WaterCoupledEnergy` — force-restore energy closure with a
##### water-mass-dependent heat capacity and (optional) advective energy
##### carried across the slab boundaries by liquid water flux.
#####
##### Like `ForceRestoreEnergy`, this closure restores the bulk land temperature
##### toward a prescribed deep climatology, but with two refinements:
#####
##### 1. The areal heat capacity `C(Mˡᵃ) = C_dry + cˡ Mˡᵃ` is recomputed every
#####    step, so a wetting/drying slab has the correct thermal inertia.
##### 2. The energy budget is written conservatively as
#####
#####        dEˡᵃ/dt = Λᵈᵉᵉᵖ(Tᵈᵉᵉᵖ − Tˡᵃ) + eᵇ Jˡᵇ − eˢ Jˡˢ − Jᴱˢ − eˡ Jᵛ − eˡ Rˡᵃᵗ
#####
#####    with `eˡ = cˡ(Tˡᵃ − Tᵣ)` the internal energy of slab liquid, and `Tˡᵃ`
#####    is updated via
#####
#####        dTˡᵃ/dt = [dEˡᵃ/dt − eˡ dMˡᵃ/dt] / C(Mˡᵃ),
#####
#####    so adding/removing water *at the slab temperature* leaves `Tˡᵃ`
#####    unchanged. `dMˡᵃ/dt` is consumed from `land.diagnostics.water_storage_tendency`
#####    written by the hydrology step.
#####
##### The internal energies carried by the boundary liquid fluxes are upwinded:
##### with `advect_deep_liquid_energy = true`, capillary rise brings in
##### `eᵇ = cˡ(Tᵈᵉᵉᵖ − Tᵣ)` and drainage exports `eᵇ = eˡ`; with
##### `advect_surface_liquid_energy = true`, infiltrating precipitation carries
##### `eˢ = cˡ(Tˡᵖ − Tᵣ)` at the precipitation temperature `Tˡᵖ` and upward
##### surface liquid carries `eˢ = eˡ`. With a switch disabled the corresponding
##### flux exchanges mass but no heat — it enters and leaves at the slab
##### temperature (`eᵇ = eˢ = eˡ`) — so the arbitrary reference `Tᵣ` cancels
##### from the update exactly. Evaporating liquid always leaves at the slab
##### temperature; its *latent* heat is separate, already inside
##### `surface_energy_flux` via the interface balance.
#####
##### `Tᵈᵉᵉᵖ` may be a `Number`, `AbstractField`, or any state-indexable
##### property (e.g. a `FieldTimeSeries`); `Λᵈᵉᵉᵖ` may be a `Number` or
##### `AbstractField`.
#####

"""
    WaterCoupledEnergy(FT = Oceananigans.defaults.FloatType;
                       dry_heat_capacity = 1480 * 1500 * 0.10,
                       liquid_heat_capacity = 4186,
                       reference_temperature = 273.15,
                       deep_temperature = 280,
                       deep_conductance = nothing,
                       deep_time_scale = nothing,
                       advect_deep_liquid_energy = true,
                       advect_surface_liquid_energy = false)

Force-restore energy closure with `Mˡᵃ`-dependent heat capacity and conservative
treatment of water-energy advection across the slab boundaries.

Use either `deep_conductance` (`Λᵈᵉᵉᵖ`, W m⁻² K⁻¹) or `deep_time_scale` (`τᵈᵉᵉᵖ`,
s) to set the deep restoring strength; exactly one must be supplied. With
`deep_time_scale`, the effective conductance is `C(Mˡᵃ)/τᵈᵉᵉᵖ` (matches
[`ForceRestoreEnergy`](@ref)).

`reference_temperature` is the reference for internal energy `eˡ(T) = cˡ(T − Tᵣ)`.
The temperature update is invariant to `Tᵣ`: fluxes whose advective switch is
disabled enter and leave at the slab temperature, so `Tᵣ` cancels exactly (up
to the positivity floor on `Mˡᵃ`). A standard choice is the triple point 273.15 K.
"""
struct WaterCoupledEnergy{FT, TD, ΛD, Tau} <: AbstractEnergyBalance
    dry_heat_capacity            :: FT
    liquid_heat_capacity         :: FT
    reference_temperature        :: FT
    deep_temperature             :: TD
    deep_conductance             :: ΛD
    deep_time_scale              :: Tau
    advect_deep_liquid_energy    :: Bool
    advect_surface_liquid_energy :: Bool
end

function WaterCoupledEnergy(FT::Type = Oceananigans.defaults.FloatType;
                            dry_heat_capacity = 1480 * 1500 * 0.10,
                            liquid_heat_capacity = 4186,
                            reference_temperature = 273.15,
                            deep_temperature = 280,
                            deep_conductance = nothing,
                            deep_time_scale = nothing,
                            advect_deep_liquid_energy = true,
                            advect_surface_liquid_energy = false)
    if isnothing(deep_conductance) === isnothing(deep_time_scale)
        throw(ArgumentError(
            "WaterCoupledEnergy requires exactly one of " *
            "`deep_conductance` (Λᵈᵉᵉᵖ, W m⁻² K⁻¹) or `deep_time_scale` (τᵈᵉᵉᵖ, s)."))
    end
    # The surface-advective term needs the temperature of incoming liquid
    # precipitation, which the atmosphere–land flux assembly does not write.
    if advect_surface_liquid_energy
        throw(ArgumentError(
            "advect_surface_liquid_energy=true is not yet supported: the " *
            "atmosphere–land flux assembly does not write " *
            "`land.fluxes.liquid_precipitation_temperature`."))
    end
    Λ = isnothing(deep_conductance) ? nothing : normalize_property(FT, deep_conductance)
    τ = isnothing(deep_time_scale)  ? nothing : convert(FT, deep_time_scale)
    Td = deep_temperature isa Number ? convert(FT, deep_temperature) : deep_temperature
    return WaterCoupledEnergy(convert(FT, dry_heat_capacity),
                              convert(FT, liquid_heat_capacity),
                              convert(FT, reference_temperature),
                              Td, Λ, τ,
                              advect_deep_liquid_energy,
                              advect_surface_liquid_energy)
end

# Consumes the signed surface energy flux `Jᴱˢ` (positive upward, written by
# the interface), the signed vapor flux `Jᵛ` (for the advective sensible-energy
# term), and the optional precipitation temperature for the surface advective term.
flux_variables(::WaterCoupledEnergy) =
    (:surface_energy_flux, :vapor_flux, :liquid_precipitation_temperature)

# The closure rides into `_step_land_temperature!` whole, so Field-valued
# properties must adapt to their GPU forms.
Adapt.adapt_structure(to, energy::WaterCoupledEnergy) =
    WaterCoupledEnergy(Adapt.adapt(to, energy.dry_heat_capacity),
                       Adapt.adapt(to, energy.liquid_heat_capacity),
                       Adapt.adapt(to, energy.reference_temperature),
                       Adapt.adapt(to, energy.deep_temperature),
                       Adapt.adapt(to, energy.deep_conductance),
                       Adapt.adapt(to, energy.deep_time_scale),
                       energy.advect_deep_liquid_energy,
                       energy.advect_surface_liquid_energy)

#####
##### Helpers
#####

@inline function deep_conductance_value(energy::WaterCoupledEnergy, C, i, j, grid, time)
    FT = eltype(grid)
    Λ  = energy.deep_conductance
    τ  = energy.deep_time_scale
    if Λ === nothing
        return C / convert(FT, τ)
    else
        return property_value(Λ, i, j, 1)
    end
end

@inline upwind_deep_internal_energy(Jˡᵇ, Tᵈ, T, Tᵣ, cˡ) =
    ifelse(Jˡᵇ > 0, cˡ * (Tᵈ - Tᵣ), cˡ * (T - Tᵣ))

@inline upwind_surface_internal_energy(Jˡˢ, T, Tˡᵖ, Tᵣ, cˡ) =
    ifelse(Jˡˢ > 0, cˡ * (T - Tᵣ), cˡ * (Tˡᵖ - Tᵣ))

#####
##### Time-step kernel.
#####

@inline function temperature_tendency(i, j, grid, energy::WaterCoupledEnergy,
                                      prognostic, fluxes, diagnostics, time)
    @inbounds begin
        Tᵢⱼ  = prognostic.T[i, j, 1]
        Mᵢⱼ  = prognostic.M[i, j, 1]
        dMdt = diagnostics.water_storage_tendency[i, j, 1]
        Jˡᵇ  = diagnostics.deep_liquid_flux[i, j, 1]
        Jˡˢ  = diagnostics.surface_liquid_flux[i, j, 1]
        Rˡᵃᵗ = diagnostics.subsurface_runoff[i, j, 1]
        Jᴱs  = fluxes.surface_energy_flux[i, j, 1]
        Jᵛ   = fluxes.vapor_flux[i, j, 1]
        Tˡᵖ  = fluxes.liquid_precipitation_temperature[i, j, 1]
    end

    cˡ   = energy.liquid_heat_capacity
    Tᵣ   = energy.reference_temperature
    Cdry = property_value(energy.dry_heat_capacity, i, j, 1)
    C    = Cdry + cˡ * max(Mᵢⱼ, 0)

    Tᵈ = stateindex(energy.deep_temperature, i, j, 1, grid, time, (Center, Center, Center))
    Λᵈ = deep_conductance_value(energy, C, i, j, grid, time)

    # Internal energy of slab liquid; the boundary fluxes upwind to it. A flux
    # whose advective switch is off carries eˡ — it exchanges mass but no heat —
    # so the arbitrary reference Tᵣ cancels from the update.
    eˡ = cˡ * (Tᵢⱼ - Tᵣ)
    eᵇ = ifelse(energy.advect_deep_liquid_energy,
                upwind_deep_internal_energy(Jˡᵇ, Tᵈ, Tᵢⱼ, Tᵣ, cˡ), eˡ)
    eˢ = ifelse(energy.advect_surface_liquid_energy,
                upwind_surface_internal_energy(Jˡˢ, Tᵢⱼ, Tˡᵖ, Tᵣ, cˡ), eˡ)

    # Evaporating liquid (Jᵛ > 0) and lateral runoff leave at the slab
    # temperature; the latent heat of evaporation is already inside Jᴱs.
    dEdt = Λᵈ * (Tᵈ - Tᵢⱼ) + eᵇ * Jˡᵇ - eˢ * Jˡˢ - Jᴱs - eˡ * Jᵛ - eˡ * Rˡᵃᵗ

    return (dEdt - eˡ * dMdt) / C
end

time_step!(energy::WaterCoupledEnergy, land, Δt, time) = step_land_temperature!(energy, land, Δt, time)

EarthSystemModels.surface_temperature(::WaterCoupledEnergy, land) = land.temperature

Base.summary(energy::WaterCoupledEnergy) =
    string("WaterCoupledEnergy(dry_heat_capacity=", prettysummary(energy.dry_heat_capacity),
           ", liquid_heat_capacity=", prettysummary(energy.liquid_heat_capacity),
           ", reference_temperature=", prettysummary(energy.reference_temperature),
           ", deep_temperature=", prettysummary(energy.deep_temperature),
           energy.deep_conductance === nothing ?
               string(", deep_time_scale=", prettysummary(energy.deep_time_scale)) :
               string(", deep_conductance=", prettysummary(energy.deep_conductance)),
           ", advect_deep_liquid_energy=", energy.advect_deep_liquid_energy,
           ", advect_surface_liquid_energy=", energy.advect_surface_liquid_energy, ")")
