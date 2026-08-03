using Oceananigans.Grids: inactive_node

using ..EarthSystemModels: EarthSystemModel
using ..EarthSystemModels.InterfaceComputations: kernel_radiation_properties

"""
    apply_air_land_radiative_fluxes!(coupled_model)

Add the radiative contribution to the land `surface_energy_flux` (positive
upward) and write diagnostic radiative fluxes into
`coupled_model.radiation.interface_fluxes.land`.

When `coupled_model.radiation === nothing`, this is a no-op.
"""
EarthSystemModels.apply_air_land_radiative_fluxes!(::EarthSystemModel{<:Nothing}) = nothing

EarthSystemModels.apply_air_land_radiative_fluxes!(coupled_model::EarthSystemModel) =
    apply_air_land_radiative_fluxes!(coupled_model, coupled_model.land)

# No land: nothing to do.
apply_air_land_radiative_fluxes!(coupled_model, ::Nothing) = nothing

function apply_air_land_radiative_fluxes!(coupled_model, land)
    # No atmosphere--land interface (no atmosphere): nothing to do.
    al_interface = coupled_model.interfaces.atmosphere_land_interface
    isnothing(al_interface) && return nothing

    radiation = coupled_model.radiation
    interface_fluxes = radiation.interface_fluxes
    isnothing(interface_fluxes) && return nothing

    # Skip when the radiation was not configured with a land surface — the
    # accumulator and the surface properties must both be present.
    land_radiative_flux = get(interface_fluxes, :land, nothing)
    isnothing(land_radiative_flux) && return nothing

    rk = kernel_radiation_properties(radiation)
    land_surface_props = get(rk.surface_properties, :land, nothing)
    isnothing(land_surface_props) && return nothing

    grid  = coupled_model.interfaces.exchanger.grid
    arch  = architecture(grid)
    clock = coupled_model.clock
    radiation_state = coupled_model.interfaces.exchanger.radiation.state

    fluxes = land.fluxes
    hasproperty(fluxes, :surface_energy_flux) || return nothing
    land_energy_flux = fluxes.surface_energy_flux

    launch!(arch, grid, :xy,
            _apply_air_land_radiative_fluxes!,
            land_energy_flux,
            land_radiative_flux,
            grid,
            clock,
            rk,
            radiation_state,
            al_interface.temperature)

    return nothing
end

@kernel function _apply_air_land_radiative_fluxes!(land_energy_flux,
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

    # `surface_energy_flux` is positive upward, so the into-land `ΣQ_rad` enters as `-ΣQ_rad`.
    @inbounds begin
        land_energy_flux[i, j, 1] += ifelse(inactive, zero(grid), -ΣQ_rad)
        interface_radiative_flux.upwelling_longwave[i, j, 1]    = ℐꜛˡʷ
        interface_radiative_flux.downwelling_longwave[i, j, 1]  = - ℐₐˡʷ
        interface_radiative_flux.downwelling_shortwave[i, j, 1] = - ℐₜˢʷ
    end
end
