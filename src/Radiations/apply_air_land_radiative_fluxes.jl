using Oceananigans.Grids: inactive_node

using ..EarthSystemModels: EarthSystemModel
using ..EarthSystemModels.InterfaceComputations: kernel_radiation_properties

"""
    apply_air_land_radiative_fluxes!(coupled_model)

Add the radiative contribution to the land net energy flux `Q` and write
diagnostic radiative fluxes into `coupled_model.radiation.interface_fluxes.land`.

When `coupled_model.radiation === nothing`, this is a no-op.
"""
EarthSystemModels.apply_air_land_radiative_fluxes!(::EarthSystemModel{<:Nothing}) = nothing

function EarthSystemModels.apply_air_land_radiative_fluxes!(coupled_model::EarthSystemModel)
    land = coupled_model.land
    isnothing(land) && return nothing

    # No atmosphere--land interface (no atmosphere or no land): nothing to do.
    al_interface = coupled_model.interfaces.atmosphere_land_interface
    isnothing(al_interface) && return nothing

    radiation = coupled_model.radiation
    interface_fluxes = radiation.interface_fluxes
    isnothing(interface_fluxes) && return nothing
    haskey(interface_fluxes, :land) || return nothing

    grid = coupled_model.interfaces.exchanger.grid
    arch = architecture(grid)
    clock = coupled_model.clock

    net_land_fluxes = coupled_model.land.fluxes
    hasproperty(net_land_fluxes, :net_energy_flux) || return nothing

    radiation_state = coupled_model.interfaces.exchanger.radiation.state
    rk = kernel_radiation_properties(radiation)

    # Skip land radiative forcing unless land surface properties were provided.
    haskey(rk.surface_properties, :land) || return nothing

    interface_temperature = al_interface.temperature

    launch!(arch, grid, :xy,
            _apply_air_land_radiative_fluxes!,
            net_land_fluxes.net_energy_flux,
            interface_fluxes.land,
            grid,
            clock,
            rk,
            radiation_state,
            interface_temperature)

    return nothing
end

@kernel function _apply_air_land_radiative_fluxes!(land_net_energy_flux,
                                                  interface_radiative_flux,
                                                  grid,
                                                  clock,
                                                  rk,
                                                  radiation_state,
                                                  land_surface_temperature)

    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)
    time = Time(clock.time)

    @inbounds begin
        Tₛ = land_surface_temperature[i, j, 1]
    end

    rs = air_land_interface_radiation_state(rk, radiation_state, i, j, kᴺ, grid, time)

    ℐꜛˡʷ = emitted_longwave_radiation(Tₛ, rs.σ, rs.ϵ)
    ℐₐˡʷ = absorbed_longwave_radiation(rs.ϵ, rs.ℐꜜˡʷ)
    ℐₜˢʷ = transmitted_shortwave_radiation(rs.α, rs.ℐꜜˢʷ)

    # Total radiative contribution to surface energy balance, positive into the land.
    ΣQ_rad = -ℐꜛˡʷ - (ℐₐˡʷ + ℐₜˢʷ)

    inactive = inactive_node(i, j, 1, grid, Center(), Center(), Center())

    @inbounds begin
        land_net_energy_flux[i, j, 1] += ifelse(inactive, zero(grid), ΣQ_rad)
        interface_radiative_flux.upwelling_longwave[i, j, 1]    = ℐꜛˡʷ
        interface_radiative_flux.downwelling_longwave[i, j, 1]  = - ℐₐˡʷ
        interface_radiative_flux.downwelling_shortwave[i, j, 1] = - ℐₜˢʷ
    end
end
