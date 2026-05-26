#####
##### `SlabEnergy` — single-temperature explicit-Euler energy balance.
#####
##### One prognostic variable `state.T`, one flux accumulator
##### `fluxes.net_energy_flux` (W m⁻², positive into the land slab).
##### The areal heat capacity is either a uniform value or includes a
##### liquid-water contribution proportional to `state.water_storage`:
#####
##### ``dry_heat_capacity + liquid_heat_capacity * W``
#####
##### where `W` is the land water storage field when bucket hydrology is used.
##### The atmosphere-facing `surface_temperature` returns `state.T`.
#####

"""
    SlabEnergy(FT = Float64;
               dry_heat_capacity = 1480 * 1500 * 0.10,
               liquid_heat_capacity = 4186)

Single-temperature slab energy balance. `dry_heat_capacity` and
`liquid_heat_capacity` may be scalars or per-cell `Field`s.

The slab heat capacity used to update `state.T` is

``dry_heat_capacity + liquid_heat_capacity * state.water_storage``

with default `liquid_heat_capacity = 4186 J m⁻² K⁻¹` per `kg m⁻²` of liquid
water.
"""
struct SlabEnergy{C, L} <: AbstractEnergyBalance
    dry_heat_capacity    :: C
    liquid_heat_capacity :: L
end

function SlabEnergy(FT::Type = Float64;
                    dry_heat_capacity = 1480.0 * 1500.0 * 0.10,
                    liquid_heat_capacity = 4186.0)
    dry_heat_capacity    = normalize_property(FT, dry_heat_capacity)
    liquid_heat_capacity = normalize_property(FT, liquid_heat_capacity)
    return SlabEnergy(dry_heat_capacity, liquid_heat_capacity)
end

prognostic_variables(::SlabEnergy) = (:T,)
flux_variables(::SlabEnergy)       = (:net_energy_flux,)

@kernel function _slab_energy_step!(T, Q, Δt, Cdry)
    i, j = @index(Global, NTuple)
    @inbounds begin
        heat_capacity = property_value(Cdry, i, j, 1)
        T[i, j, 1] += Q[i, j, 1] * Δt / heat_capacity
    end
end

@kernel function _slab_energy_step_with_water!(T, Q, W, Δt, Cdry, Cl)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Cdry_ij1 = property_value(Cdry, i, j, 1)
        Cl_ij1   = property_value(Cl, i, j, 1)
        heat_capacity = Cdry_ij1 + Cl_ij1 * W[i, j, 1]
        T[i, j, 1] += Q[i, j, 1] * Δt / heat_capacity
    end
end

function step!(energy::SlabEnergy, state, fluxes, surface, grid, Δt)
    arch = architecture(grid)
    Cdry = energy.dry_heat_capacity

    if hasproperty(state, :water_storage)
        launch!(arch, grid, :xy, _slab_energy_step_with_water!,
                state.T, fluxes.net_energy_flux, state.water_storage, Δt, Cdry, energy.liquid_heat_capacity)
    else
        launch!(arch, grid, :xy, _slab_energy_step!,
                state.T, fluxes.net_energy_flux, Δt, Cdry)
    end
    return nothing
end

surface_temperature(::SlabEnergy, state) = state.T

Base.summary(s::SlabEnergy) =
    string("SlabEnergy(dry_heat_capacity=", prettysummary(s.dry_heat_capacity),
           ", liquid_heat_capacity=", prettysummary(s.liquid_heat_capacity), ")")
