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
#####        dEˡᵃ/dt = Λᵈᵉᵉᵖ(Tᵈᵉᵉᵖ − Tˡᵃ) + advective deep − Jᴱ_s − energy carried by Rᴹ_lat
#####
#####    and `Tˡᵃ` is updated via
#####
#####        dTˡᵃ/dt = [dEˡᵃ/dt − cˡ(Tˡᵃ − Tᵣ) dMˡᵃ/dt] / C(Mˡᵃ),
#####
#####    so adding/removing water *at the slab temperature* leaves `Tˡᵃ`
#####    unchanged. `dMˡᵃ/dt` is consumed from `land.diagnostics.water_storage_tendency`
#####    written by the hydrology step.
#####
##### Deep liquid advective energy uses upwind values: capillary rise brings in
##### internal energy at `T = Tᵈᵉᵉᵖ`; drainage exports at `T = Tˡᵃ`. The surface
##### advective term (precipitation/evaporation) is optional and uses bulk
##### `Tˡᵃ` as the up-flux proxy and `liquid_precipitation_temperature` as the
##### down-flux value. (Vapor latent heat is *not* an advective liquid term —
##### it is already inside `surface_energy_flux` via the interface balance.)
#####
##### `Tᵈᵉᵉᵖ` and `Λᵈᵉᵉᵖ` may each be a `Number`, `AbstractField`, or any
##### state-indexable property (e.g. `FieldTimeSeries`).
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
The choice affects only the *advective* energy budget; the temperature update is
invariant to `Tᵣ`. A standard choice is the triple point 273.15 K.
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
    # The surface-advective term needs the temperature at which liquid
    # precipitation enters the slab, but `_assemble_slab_land_fluxes!` does not
    # yet write that quantity. Until the precip-temperature is plumbed from the
    # atmospheric state, this branch silently reads zeros and would treat rain
    # as T = 0 K. Refuse the configuration with a pointer to the followup.
    if advect_surface_liquid_energy
        throw(ArgumentError(
            "advect_surface_liquid_energy=true is not yet plumbed: the precip " *
            "temperature `land.fluxes.liquid_precipitation_temperature` is " *
            "declared but not written by the atmosphere–land flux assembly. " *
            "Track via the followup item in docs/src/land/evaporation_front_slab_land.md §10."))
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

# Consumes the signed surface energy flux `Jᴱ_s` (positive upward, written by
# the interface) plus the optional precipitation temperature for the surface
# advective term. The legacy `net_energy_flux` is kept as a deprecated fallback
# when the new interface isn't writing `surface_energy_flux`.
flux_variables(::WaterCoupledEnergy) =
    (:surface_energy_flux, :liquid_precipitation_temperature)

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

@inline upwind_deep_internal_energy(Jˡb, Tᵈ, T, Tᵣ, cˡ) =
    ifelse(Jˡb > 0, cˡ * (Tᵈ - Tᵣ), cˡ * (T - Tᵣ))

@inline upwind_surface_internal_energy(Jˡs, T, Tˡp, Tᵣ, cˡ) =
    ifelse(Jˡs > 0, cˡ * (T - Tᵣ), cˡ * (Tˡp - Tᵣ))

#####
##### Time-step kernel.
#####

@inline function temperature_tendency(i, j, grid, energy::WaterCoupledEnergy,
                                      prognostic, fluxes, diagnostics, time)
    FT = eltype(grid)

    @inbounds begin
        Tij   = prognostic.T[i, j, 1]
        Mij   = prognostic.M[i, j, 1]
        dMdt  = diagnostics.water_storage_tendency[i, j, 1]
        Jˡb   = diagnostics.deep_liquid_flux[i, j, 1]
        Jˡs   = diagnostics.surface_liquid_flux[i, j, 1]
        Rlat  = diagnostics.subsurface_runoff[i, j, 1]
        Jᴱs   = fluxes.surface_energy_flux[i, j, 1]
        Tˡpij = fluxes.liquid_precipitation_temperature[i, j, 1]
    end

    cˡ      = energy.liquid_heat_capacity
    Tᵣ      = energy.reference_temperature
    Cdry_ij = property_value(energy.dry_heat_capacity, i, j, 1)
    C       = Cdry_ij + cˡ * max(Mij, zero(Mij))

    Tᵈ_ij = stateindex(energy.deep_temperature, i, j, 1, grid, time, (Center, Center, Center))
    Λᵈ    = deep_conductance_value(energy, C, i, j, grid, time)

    Jᴱ_cond = Λᵈ * (Tᵈ_ij - Tij)

    eb_up    = upwind_deep_internal_energy(Jˡb, Tᵈ_ij, Tij, Tᵣ, cˡ)
    Jᴱ_adv_b = ifelse(energy.advect_deep_liquid_energy, eb_up * Jˡb, zero(FT))

    es_up    = upwind_surface_internal_energy(Jˡs, Tij, Tˡpij, Tᵣ, cˡ)
    Jᴱ_adv_s = ifelse(energy.advect_surface_liquid_energy, es_up * Jˡs, zero(FT))

    eR_lat = cˡ * (Tij - Tᵣ) * Rlat

    dEdt = Jᴱ_cond + Jᴱ_adv_b - (Jᴱs + Jᴱ_adv_s) - eR_lat
    return (dEdt - cˡ * (Tij - Tᵣ) * dMdt) / C
end

time_step!(energy::WaterCoupledEnergy, land, Δt, time) = step_land_temperature!(energy, land, Δt, time)

EarthSystemModels.surface_temperature(::WaterCoupledEnergy, land) = land.temperature

Base.summary(energy::WaterCoupledEnergy) =
    string("WaterCoupledEnergy(",
           "C_dry=", prettysummary(energy.dry_heat_capacity),
           ", cˡ=", prettysummary(energy.liquid_heat_capacity),
           ", Tᵣ=", prettysummary(energy.reference_temperature),
           ", Tᵈᵉᵉᵖ=", prettysummary(energy.deep_temperature),
           energy.deep_conductance === nothing ?
               string(", τᵈᵉᵉᵖ=", prettysummary(energy.deep_time_scale)) :
               string(", Λᵈᵉᵉᵖ=", prettysummary(energy.deep_conductance)),
           ", advect_deep=", energy.advect_deep_liquid_energy,
           ", advect_surf=", energy.advect_surface_liquid_energy, ")")
