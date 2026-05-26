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
    ForceRestoreEnergy(FT = Float64;
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

function ForceRestoreEnergy(FT::Type = Float64;
                            dry_heat_capacity = 1480.0 * 1500.0 * 0.10,
                            liquid_heat_capacity = 4186.0,
                            deep_temperature = 280.0,
                            deep_time_scale = 12 * 3600.0)
    dry_heat_capacity    = normalize_property(FT, dry_heat_capacity)
    liquid_heat_capacity = normalize_property(FT, liquid_heat_capacity)
    deep_temperature     = deep_temperature isa Number ? convert(FT, deep_temperature) : deep_temperature
    deep_time_scale      = convert(FT, deep_time_scale)
    return ForceRestoreEnergy(dry_heat_capacity, liquid_heat_capacity,
                              deep_temperature, deep_time_scale)
end

prognostic_variables(::ForceRestoreEnergy) = (:T,)
flux_variables(::ForceRestoreEnergy)       = (:net_energy_flux,)

@kernel function _force_restore_step_no_water!(T, Q, Δt, Cdry, Cl, Tdeep, τ, grid, time)
    i, j = @index(Global, NTuple)
    @inbounds begin
        # No-water variant: hydrology contributes nothing to the slab
        # heat capacity, so C reduces to the dry value.
        C     = property_value(Cdry, i, j, 1)
        C_inv = ifelse(C <= 0, 0, inv(C))
        τ⁻    = ifelse(τ > 0, inv(τ), 0)

        Tᵈᵉᵉᵖ = stateindex(Tdeep, i, j, 1, grid, time, (Center, Center, Center))
        Tˢ    = T[i, j, 1]

        T[i, j, 1] = Tˢ + (Q[i, j, 1] * C_inv + (Tᵈᵉᵉᵖ - Tˢ) * τ⁻) * Δt
    end
end

@kernel function _force_restore_step_with_water!(T, Q, M, Δt, Cdry, Cl, Tdeep, τ, grid, time)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Mᵃ       = ifelse(M[i, j, 1] < 0, 0, M[i, j, 1])
        Cdry_ij1 = property_value(Cdry, i, j, 1)
        Cl_ij1   = property_value(Cl, i, j, 1)
        C        = Cdry_ij1 + Cl_ij1 * Mᵃ
        C_inv    = ifelse(C <= 0, 0, inv(C))
        τ⁻       = ifelse(τ > 0, inv(τ), 0)

        Tᵈᵉᵉᵖ = stateindex(Tdeep, i, j, 1, grid, time, (Center, Center, Center))
        Tˢ    = T[i, j, 1]

        T[i, j, 1] = Tˢ + (Q[i, j, 1] * C_inv + (Tᵈᵉᵉᵖ - Tˢ) * τ⁻) * Δt
    end
end

function step!(energy::ForceRestoreEnergy, state, fluxes, surface, grid, Δt, time)
    arch = architecture(grid)
    Q = fluxes.net_energy_flux

    if hasproperty(state, :water_storage)
        launch!(arch, grid, :xy, _force_restore_step_with_water!,
                state.T, Q, state.water_storage, Δt,
                energy.dry_heat_capacity, energy.liquid_heat_capacity,
                energy.deep_temperature, energy.deep_time_scale,
                grid, time)
    else
        launch!(arch, grid, :xy, _force_restore_step_no_water!,
                state.T, Q, Δt,
                energy.dry_heat_capacity, energy.liquid_heat_capacity,
                energy.deep_temperature, energy.deep_time_scale,
                grid, time)
    end

    return nothing
end

surface_temperature(::ForceRestoreEnergy, state) = state.T

Base.summary(energy::ForceRestoreEnergy) =
    string("ForceRestoreEnergy(dry_heat_capacity=", prettysummary(energy.dry_heat_capacity),
           ", liquid_heat_capacity=", prettysummary(energy.liquid_heat_capacity),
           ", deep_temperature=", prettysummary(energy.deep_temperature),
           ", deep_time_scale=", prettysummary(energy.deep_time_scale), ")")
