# PrescribedRadiation-aware methods for the radiation getter functions
# declared (with `nothing` fallbacks) in InterfaceComputations.

@inline EarthSystemModels.InterfaceComputations.kernel_radiation_properties(r::PrescribedRadiation) =
    (σ = r.stefan_boltzmann_constant,
     surface_properties = r.surface_properties)

@inline function _zero_radiation_state(grid)
    z = zero(eltype(grid))
    return (σ = z, α = z, ϵ = z, ℐꜜˢʷ = z, ℐꜜˡʷ = z)
end

# Generic per-surface kernel: read σ from `rk`, downwelling SW/LW from
# the exchanger, and the surface-specific albedo/emissivity from `s`.
@inline function _surface_radiation_state(s, rk, exchanger_state, i, j, k, grid, time)
    σ = rk.σ
    @inbounds ℐꜜˢʷ = exchanger_state.ℐꜜˢʷ[i, j, 1]
    @inbounds ℐꜜˡʷ = exchanger_state.ℐꜜˡʷ[i, j, 1]
    α = stateindex(s.albedo,     i, j, k, grid, time, (Center, Center, Center), ℐꜜˢʷ)
    ϵ = stateindex(s.emissivity, i, j, k, grid, time, (Center, Center, Center))
    return (; σ, α, ϵ, ℐꜜˢʷ, ℐꜜˡʷ)
end

@inline EarthSystemModels.InterfaceComputations.air_sea_interface_radiation_state(
        rk, exchanger_state, i, j, k, grid, time) =
    _surface_radiation_state(rk.surface_properties.ocean,
                             rk, exchanger_state, i, j, k, grid, time)

@inline EarthSystemModels.InterfaceComputations.air_sea_ice_interface_radiation_state(
        rk, exchanger_state, i, j, k, grid, time) =
    _surface_radiation_state(rk.surface_properties.sea_ice,
                             rk, exchanger_state, i, j, k, grid, time)

# Land radiative properties are optional (no `:land` key ⇒ no land
# radiative forcing); fall back to a zero state in that case. The land variant
# returns an `AirLandRadiationState` struct rather than a NamedTuple.
@inline function _air_land_zero_radiation_state(grid)
    z = zero(eltype(grid))
    return AirLandRadiationState(z, z, z, z, z)
end

@inline function _air_land_surface_radiation_state(s, rk, exchanger_state, i, j, k, grid, time)
    FT = eltype(grid)
    σ = convert(FT, rk.σ)
    @inbounds ℐꜜˢʷ = convert(FT, exchanger_state.ℐꜜˢʷ[i, j, 1])
    @inbounds ℐꜜˡʷ = convert(FT, exchanger_state.ℐꜜˡʷ[i, j, 1])
    α = convert(FT, stateindex(s.albedo,     i, j, k, grid, time, (Center, Center, Center), ℐꜜˢʷ))
    ϵ = convert(FT, stateindex(s.emissivity, i, j, k, grid, time, (Center, Center, Center)))
    return AirLandRadiationState(σ, α, ϵ, ℐꜜˢʷ, ℐꜜˡʷ)
end

@inline function EarthSystemModels.InterfaceComputations.air_land_interface_radiation_state(
        rk, exchanger_state, i, j, k, grid, time)
    haskey(rk.surface_properties, :land) || return _air_land_zero_radiation_state(grid)
    return _air_land_surface_radiation_state(rk.surface_properties.land,
                                             rk, exchanger_state, i, j, k, grid, time)
end
