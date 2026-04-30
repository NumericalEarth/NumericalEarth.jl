# PrescribedRadiation-aware methods for the radiation getter functions
# declared (with `nothing` fallbacks) in InterfaceComputations.

@inline kernel_radiation_properties(r::PrescribedRadiation) =
    (σ = r.stefan_boltzmann_constant,
     surface_properties = r.surface_properties)

@inline function air_sea_interface_radiation_state(rk, exchanger_state, i, j, k, grid, time)
    σ = rk.σ
    @inbounds ℐꜜˢʷ = exchanger_state.ℐꜜˢʷ[i, j, 1]
    @inbounds ℐꜜˡʷ = exchanger_state.ℐꜜˡʷ[i, j, 1]
    s = rk.surface_properties.ocean
    α = stateindex(s.albedo,     i, j, k, grid, time, (Center, Center, Center), ℐꜜˢʷ)
    ϵ = stateindex(s.emissivity, i, j, k, grid, time, (Center, Center, Center))
    return (; σ, α, ϵ, ℐꜜˢʷ, ℐꜜˡʷ)
end

@inline function air_sea_ice_interface_radiation_state(rk, exchanger_state, i, j, k, grid, time)
    σ = rk.σ
    @inbounds ℐꜜˢʷ = exchanger_state.ℐꜜˢʷ[i, j, 1]
    @inbounds ℐꜜˡʷ = exchanger_state.ℐꜜˡʷ[i, j, 1]
    s = rk.surface_properties.sea_ice
    α = stateindex(s.albedo,     i, j, k, grid, time, (Center, Center, Center), ℐꜜˢʷ)
    ϵ = stateindex(s.emissivity, i, j, k, grid, time, (Center, Center, Center))
    return (; σ, α, ϵ, ℐꜜˢʷ, ℐꜜˡʷ)
end
