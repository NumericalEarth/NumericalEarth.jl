#####
##### `SlabEnergy` — single-temperature explicit-Euler energy balance.
#####
##### One prognostic variable `state.T`, one flux accumulator
##### `fluxes.net_energy_flux` (W m⁻²). Heat capacity is the areal
##### `(ρ c H)` of the slab; it may be a scalar (uniform slab) or a
##### `CenterField` for per-cell heterogeneity (e.g. urban vs. forest).
##### The atmosphere-facing `surface_temperature` returns `state.T`.
#####

"""
    SlabEnergy(FT = Float64; heat_capacity = 1480 * 1500 * 0.10)

Single-temperature slab energy balance. `heat_capacity` is the areal
heat capacity `(ρ c H)` in `J m⁻² K⁻¹`; pass a scalar or a 2D
`CenterField` for per-cell heterogeneity.
"""
struct SlabEnergy{C} <: AbstractEnergyBalance
    heat_capacity :: C
end

function SlabEnergy(FT::Type = Float64; heat_capacity = 1480.0 * 1500.0 * 0.10)
    if heat_capacity isa Number
        return SlabEnergy{FT}(convert(FT, heat_capacity))
    else
        return SlabEnergy{typeof(heat_capacity)}(heat_capacity)
    end
end

prognostic_variables(::SlabEnergy) = (:T,)
flux_variables(::SlabEnergy)       = (:net_energy_flux,)

@inline _heat_capacity_at(C::Number, i, j) = C
@inline _heat_capacity_at(C, i, j)         = @inbounds C[i, j, 1]

@kernel function _slab_energy_step!(T, Q, Δt, C)
    i, j = @index(Global, NTuple)
    Cij = _heat_capacity_at(C, i, j)
    @inbounds T[i, j, 1] += Q[i, j, 1] * Δt / Cij
end

function step!(energy::SlabEnergy, state, fluxes, surface, parameters, grid, Δt)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _slab_energy_step!,
            state.T, fluxes.net_energy_flux, Δt, energy.heat_capacity)
    return nothing
end

surface_temperature(::SlabEnergy, state) = state.T

Base.summary(::SlabEnergy{C}) where C = "SlabEnergy{$C}"
