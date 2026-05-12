#####
##### `SlabEnergy` — single-temperature explicit-Euler energy balance.
#####
##### One prognostic variable `state.T`, one flux accumulator
##### `fluxes.net_energy_flux` (W m⁻²). Heat capacity is the areal
##### `(ρ c H)` of the slab, a scalar uniform over the grid. The
##### atmosphere-facing `surface_temperature` returns `state.T`.
#####

"""
    SlabEnergy(FT = Float64; heat_capacity = 1480 * 1500 * 0.10)

Single-temperature slab energy balance. `heat_capacity` is the areal
heat capacity `(ρ c H)` in `J m⁻² K⁻¹`.
"""
struct SlabEnergy{FT} <: AbstractEnergyBalance
    heat_capacity :: FT
end

function SlabEnergy(FT::Type = Float64; heat_capacity = 1480.0 * 1500.0 * 0.10)
    return SlabEnergy{FT}(convert(FT, heat_capacity))
end

prognostic_variables(::SlabEnergy) = (:T,)
flux_variables(::SlabEnergy)       = (:net_energy_flux,)

@kernel function _slab_energy_step!(T, Q, Δt, C)
    i, j = @index(Global, NTuple)
    @inbounds T[i, j, 1] += Q[i, j, 1] * Δt / C
end

function step!(energy::SlabEnergy, state, fluxes, surface, parameters, grid, Δt)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _slab_energy_step!,
            state.T, fluxes.net_energy_flux, Δt, energy.heat_capacity)
    return nothing
end

surface_temperature(::SlabEnergy, state) = state.T

Base.summary(s::SlabEnergy{FT}) where FT =
    string("SlabEnergy{$FT}(ρcH=", s.heat_capacity, ")")
