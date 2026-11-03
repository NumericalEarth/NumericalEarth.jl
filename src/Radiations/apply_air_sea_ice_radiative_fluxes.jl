"""
    apply_air_sea_ice_radiative_fluxes!(coupled_model)

Add the radiative contribution to the net sea-ice top heat flux and write
the diagnostic radiative fluxes (upwelling LW, absorbed LW, transmitted SW)
into `coupled_model.radiation.interface_fluxes.sea_ice`.

When `coupled_model.radiation === nothing`, or sea ice receives no fluxes, this is a no-op.
"""
EarthSystemModels.apply_air_sea_ice_radiative_fluxes!(::EarthSystemModel{<:Nothing}) = nothing

function EarthSystemModels.apply_air_sea_ice_radiative_fluxes!(coupled_model::EarthSystemModel)
    isnothing(coupled_model.interfaces.net_fluxes.sea_ice) && return nothing

    sea_ice = coupled_model.sea_ice
    radiation = coupled_model.radiation
    interface_fluxes = radiation.interface_fluxes
    isnothing(interface_fluxes) && return nothing
    haskey(interface_fluxes, :sea_ice) || return nothing

    grid = Oceananigans.Grids.grid(sea_ice)
    arch = architecture(grid)
    clock = coupled_model.clock

    top_heat_flux = coupled_model.interfaces.net_fluxes.sea_ice.top.heat
    radiation_state = coupled_model.interfaces.exchanger.radiation.state
    rk = kernel_radiation_properties(radiation)

    ice_concentration = sea_ice_concentration(sea_ice)
    surface_temperature = coupled_model.interfaces.atmosphere_sea_ice_interface.temperature
    sea_ice_properties = coupled_model.interfaces.sea_ice_properties

    launch!(arch, grid, :xy,
            _apply_air_sea_ice_radiative_fluxes!,
            top_heat_flux,
            interface_fluxes.sea_ice,
            grid,
            clock,
            rk,
            radiation_state,
            ice_concentration,
            surface_temperature,
            sea_ice_properties)

    return nothing
end

@kernel function _apply_air_sea_ice_radiative_fluxes!(top_heat_flux,
                                                      interface_radiative_flux,
                                                      grid,
                                                      clock,
                                                      rk,
                                                      radiation_state,
                                                      ice_concentration,
                                                      surface_temperature,
                                                      sea_ice_properties)

    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)
    time = Time(clock.time)

    @inbounds begin
        ℵᵢ = ice_concentration[i, j, 1]
        Ts = surface_temperature[i, j, kᴺ]
    end
    Ts = convert_to_kelvin(sea_ice_properties.temperature_units, Ts)

    rs = air_sea_ice_interface_radiation_state(rk, radiation_state, i, j, kᴺ, grid, time)

    ℐꜛˡʷ = emitted_longwave_radiation(Ts, rs.σ, rs.ϵ)
    ℐₐˡʷ = absorbed_longwave_radiation(rs.ϵ, rs.ℐꜜˡʷ)
    ℐₜˢʷ = transmitted_shortwave_radiation(rs.α, rs.ℐꜜˢʷ)
    ΣQ  = (ℐꜛˡʷ + ℐₐˡʷ + ℐₜˢʷ) * ℵᵢ

    inactive = inactive_node(i, j, kᴺ, grid, Center(), Center(), Center())

    @inbounds begin
        top_heat_flux[i, j, 1] += ifelse(inactive, zero(grid), ΣQ)
        interface_radiative_flux.upwelling_longwave[i, j, 1]    = ℐꜛˡʷ
        interface_radiative_flux.downwelling_longwave[i, j, 1]  = ℐₐˡʷ
        interface_radiative_flux.downwelling_shortwave[i, j, 1] = ℐₜˢʷ
    end
end
