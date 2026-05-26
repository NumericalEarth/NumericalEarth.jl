#####
##### `ForceRestoreEnergy` — slab energy closure with companion deep-soil
##### state for force-restore behavior.
#####
##### Adds a second prognostic variable `Tᵈ` (deep soil temperature) and two
##### restoring pathways:
#####
##### - a surface-to-deep restoring term `((Tᵈ - T) / τˢ)` in the surface
#####   temperature tendency;
##### - a deep-to-climatology restoring term `((Tᶜ - Tᵈ) / τᵈ)` for
#####   the companion deep temperature.
#####
##### The effective heat capacity uses `W` when bucket hydrology is present:
##### `C = Cdry + Cl · W` (kg m⁻² K⁻¹).
#####
"""
    ForceRestoreEnergy(FT = Float64;
                      dry_heat_capacity = 1480 * 1500 * 0.10,
                      liquid_heat_capacity = 4186,
                      deep_temperature = 280.0,
                      surface_to_deep_time_scale = 12 * 3600,
                      deep_to_climate_time_scale = 30 * 24 * 3600)

Force-restore slab energy model.

`deep_temperature` may be a `Number`, per-cell `AbstractField`, or any
state-indexable climatology (e.g. `FieldTimeSeries`) and is interpreted as
the deep climatological target temperature `Tᶜ`.
"""
struct ForceRestoreEnergy{C, L, Td, STS, DTS} <: AbstractEnergyBalance
    dry_heat_capacity    :: C
    liquid_heat_capacity :: L
    deep_temperature     :: Td
    surface_to_deep_time_scale :: STS
    deep_to_climate_time_scale :: DTS
end

function ForceRestoreEnergy(FT::Type = Float64;
                           dry_heat_capacity = 1480.0 * 1500.0 * 0.10,
                           liquid_heat_capacity = 4186.0,
                           deep_temperature = 280.0,
                           surface_to_deep_time_scale = 12 * 3600.0,
                           deep_to_climate_time_scale = 30 * 24 * 3600.0)
    dry_heat_capacity      = normalize_property(FT, dry_heat_capacity)
    liquid_heat_capacity   = normalize_property(FT, liquid_heat_capacity)
    deep_temperature       = deep_temperature isa Number ? convert(FT, deep_temperature) : deep_temperature
    surface_to_deep_time_scale = convert(FT, surface_to_deep_time_scale)
    deep_to_climate_time_scale = convert(FT, deep_to_climate_time_scale)
    return ForceRestoreEnergy(dry_heat_capacity, liquid_heat_capacity, deep_temperature,
                             surface_to_deep_time_scale, deep_to_climate_time_scale)
end

function initial_state(energy::ForceRestoreEnergy, name::Symbol, grid)
    name === :Tᵈ || return CenterField(grid)

    Tᵈ = CenterField(grid)
    if energy.deep_temperature isa Number
        fill!(Tᵈ, energy.deep_temperature)
        return Tᵈ
    end

    arch = architecture(grid)
    launch!(arch, grid, :xy, _force_restore_initial_deep_temperature!,
            Tᵈ, energy.deep_temperature, grid, zero(eltype(Tᵈ)))
    return Tᵈ
end

prognostic_variables(::ForceRestoreEnergy) = (:T, :Tᵈ)
flux_variables(::ForceRestoreEnergy)       = (:net_energy_flux,)

@kernel function _force_restore_initial_deep_temperature!(Tᵈ, deep_temperature, grid, time)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Tᵈ[i, j, 1] = stateindex(deep_temperature, i, j, 1, grid, time, (Center, Center, Center))
    end
end

@kernel function _force_restore_step_no_water!(T, Tᵈ, Q, Δt, Cdry, Cl, Td, τˢ, τᵈ, grid, time)
    i, j = @index(Global, NTuple)
    @inbounds begin
        # No-water variant: hydrology contributes nothing to the slab
        # heat capacity, so C reduces to the dry value.
        Cdry_ij1 = property_value(Cdry, i, j, 1)
        C = Cdry_ij1
        C_inv = ifelse(C <= 0, 0, inv(C))

        τˢ⁻ = ifelse(τˢ > 0, inv(τˢ), 0)
        τᵈ⁻ = ifelse(τᵈ > 0, inv(τᵈ), 0)
        Tᶜ = stateindex(Td, i, j, 1, grid, time, (Center, Center, Center))
        T_surface = T[i, j, 1]
        T_deep = Tᵈ[i, j, 1]

        T_new = T_surface + (Q[i, j, 1] * C_inv + (T_deep - T_surface) * τˢ⁻) * Δt
        Tᵈ_new = T_deep + (Tᶜ - T_deep) * τᵈ⁻ * Δt

        T[i, j, 1] = T_new
        Tᵈ[i, j, 1] = Tᵈ_new
    end
end

@kernel function _force_restore_step_with_water!(T, Tᵈ, Q, W, Δt, Cdry, Cl, Td, τˢ, τᵈ, grid, time)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Wᵃ = ifelse(W[i, j, 1] < 0, 0, W[i, j, 1])
        Cdry_ij1 = property_value(Cdry, i, j, 1)
        Cl_ij1   = property_value(Cl, i, j, 1)
        C = Cdry_ij1 + Cl_ij1 * Wᵃ
        C_inv = ifelse(C <= 0, 0, inv(C))

        τˢ⁻ = ifelse(τˢ > 0, inv(τˢ), 0)
        τᵈ⁻ = ifelse(τᵈ > 0, inv(τᵈ), 0)
        Tᶜ = stateindex(Td, i, j, 1, grid, time, (Center, Center, Center))
        T_surface = T[i, j, 1]
        T_deep = Tᵈ[i, j, 1]

        T_new = T_surface + (Q[i, j, 1] * C_inv + (T_deep - T_surface) * τˢ⁻) * Δt
        Tᵈ_new = T_deep + (Tᶜ - T_deep) * τᵈ⁻ * Δt

        T[i, j, 1] = T_new
        Tᵈ[i, j, 1] = Tᵈ_new
    end
end

function step!(energy::ForceRestoreEnergy, state, fluxes, surface, grid, Δt, time)
    arch = architecture(grid)
    Q = fluxes.net_energy_flux

    if hasproperty(state, :water_storage)
        launch!(arch, grid, :xy, _force_restore_step_with_water!,
                state.T, state.Tᵈ, Q,
                state.water_storage, Δt,
                energy.dry_heat_capacity, energy.liquid_heat_capacity,
                energy.deep_temperature,
                energy.surface_to_deep_time_scale, energy.deep_to_climate_time_scale,
                grid, time)
    else
        launch!(arch, grid, :xy, _force_restore_step_no_water!,
                state.T, state.Tᵈ, Q, Δt,
                energy.dry_heat_capacity, energy.liquid_heat_capacity,
                energy.deep_temperature,
                energy.surface_to_deep_time_scale, energy.deep_to_climate_time_scale,
                grid, time)
    end

    return nothing
end

surface_temperature(::ForceRestoreEnergy, state) = state.T

Base.summary(energy::ForceRestoreEnergy) =
    string("ForceRestoreEnergy(dry_heat_capacity=", prettysummary(energy.dry_heat_capacity),
           ", liquid_heat_capacity=", prettysummary(energy.liquid_heat_capacity),
           ", deep_temperature=", prettysummary(energy.deep_temperature),
           ", surface_to_deep_time_scale=", prettysummary(energy.surface_to_deep_time_scale),
           ", deep_to_climate_time_scale=", prettysummary(energy.deep_to_climate_time_scale), ")")
