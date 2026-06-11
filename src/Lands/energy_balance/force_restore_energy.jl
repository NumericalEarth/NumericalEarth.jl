#####
##### `ForceRestoreEnergy` — slab energy closure that relaxes the bulk land
##### temperature toward a prescribed deep climatology.
#####
##### The single prognostic variable is the bulk temperature `T`, evolving
##### under the net surface energy flux plus a restoring term toward a
##### prescribed deep temperature `Tᵈᵉᵉᵖ`:
#####
#####     ∂T/∂t = Q / C + (Tᵈᵉᵉᵖ − T) / τ
#####
##### where `C = Cdry + Cl · Mˡᵃ` is the effective areal heat capacity
##### (the liquid-water term `Cl · Mˡᵃ` is included when bucket hydrology is
##### present), and `τ` is the deep-restore time scale.
#####
##### `Tᵈᵉᵉᵖ` is prescribed, not prognostic: a `Number`, per-cell
##### `AbstractField`, or any state-indexable climatology (e.g. a
##### `FieldTimeSeries`). This is the single-layer simplification of the
##### two-layer force-restore method — the deep temperature is an external
##### target rather than a companion prognostic.
#####
"""
    ForceRestoreEnergy(FT = Oceananigans.defaults.FloatType;
                       dry_heat_capacity = 1480 * 1500 * 0.10,
                       liquid_heat_capacity = 4186,
                       deep_temperature = 280.0,
                       deep_time_scale = 12 * 3600)

Slab energy model that relaxes the bulk land temperature toward a prescribed
deep climatological temperature `deep_temperature` (math `Tᵈᵉᵉᵖ`) on the time
scale `deep_time_scale` (math `τ`), in addition to the surface energy flux.

`deep_temperature` may be a `Number`, per-cell `AbstractField`, or any
state-indexable climatology (e.g. `FieldTimeSeries`).
"""
struct ForceRestoreEnergy{C, L, Td, T} <: AbstractEnergyBalance
    dry_heat_capacity    :: C
    liquid_heat_capacity :: L
    deep_temperature     :: Td
    deep_time_scale      :: T
end

function ForceRestoreEnergy(FT::Type = Oceananigans.defaults.FloatType;
                            dry_heat_capacity = 1480 * 1500 * 0.10,
                            liquid_heat_capacity = 4186,
                            deep_temperature = 280,
                            deep_time_scale = 12 * 3600)
    dry_heat_capacity    = normalize_property(FT, dry_heat_capacity)
    liquid_heat_capacity = normalize_property(FT, liquid_heat_capacity)
    deep_temperature     = deep_temperature isa Number ? convert(FT, deep_temperature) : deep_temperature
    deep_time_scale      = convert(FT, deep_time_scale)
    return ForceRestoreEnergy(dry_heat_capacity, liquid_heat_capacity,
                              deep_temperature, deep_time_scale)
end

flux_variables(::ForceRestoreEnergy) = (:net_energy_flux,)

# `τᵈ` is the deep-restore time scale (math `τᵈᵉᵉᵖ` in notation.md); not the
# kinematic momentum flux `τ`. `Tᵈ` is the deep-target temperature.
# ∂T/∂t = Q/C + (Tᵈ − T)/τᵈ, with C = Cdry + Cl·max(Mˡᵃ, 0).
@inline function temperature_tendency(i, j, grid, energy::ForceRestoreEnergy,
                                      prognostic, fluxes, diagnostics, time)
    @inbounds begin
        Tᵢⱼ = prognostic.T[i, j, 1]
        Mᵢⱼ = prognostic.M[i, j, 1]
        Q   = fluxes.net_energy_flux[i, j, 1]
    end
    Cdry = property_value(energy.dry_heat_capacity, i, j, 1)
    Cl   = property_value(energy.liquid_heat_capacity, i, j, 1)
    C    = Cdry + Cl * max(Mᵢⱼ, 0)
    Tᵈ   = stateindex(energy.deep_temperature, i, j, 1, grid, time, (Center, Center, Center))
    return Q / C + (Tᵈ - Tᵢⱼ) / energy.deep_time_scale
end

time_step!(energy::ForceRestoreEnergy, land, Δt, time) = step_land_temperature!(energy, land, Δt, time)

EarthSystemModels.surface_temperature(::ForceRestoreEnergy, land) = land.temperature

Base.summary(energy::ForceRestoreEnergy) =
    string("ForceRestoreEnergy(dry_heat_capacity=", prettysummary(energy.dry_heat_capacity),
           ", liquid_heat_capacity=", prettysummary(energy.liquid_heat_capacity),
           ", deep_temperature=", prettysummary(energy.deep_temperature),
           ", deep_time_scale=", prettysummary(energy.deep_time_scale), ")")

"""
    SlabEnergy(FT = Oceananigans.defaults.FloatType;
               dry_heat_capacity = 1480 * 1500 * 0.10,
               liquid_heat_capacity = 4186)

Pure slab energy balance — the `τ → ∞` limit of [`ForceRestoreEnergy`](@ref),
with no deep restoring term. Returns a `ForceRestoreEnergy` configured so the
restoring contribution vanishes: `deep_time_scale = Inf` (so `(Tᵈ − T)/τ = 0`)
and `deep_temperature` is set to a finite sentinel that is never used.
"""
SlabEnergy(FT::Type = Oceananigans.defaults.FloatType;
           dry_heat_capacity = 1480 * 1500 * 0.10,
           liquid_heat_capacity = 4186) =
    ForceRestoreEnergy(FT;
                       dry_heat_capacity,
                       liquid_heat_capacity,
                       deep_temperature = 0,
                       deep_time_scale = FT(Inf))
