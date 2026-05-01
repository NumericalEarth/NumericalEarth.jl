using Oceananigans.Grids: inactive_node
using NumericalEarth.EarthSystemModels: EarthSystemModel
using NumericalEarth.EarthSystemModels.InterfaceComputations: convert_to_kelvin, sea_ice_concentration
using NumericalEarth.Oceans: shortwave_radiative_forcing, get_radiative_forcing

"""
    apply_air_sea_radiative_fluxes!(coupled_model)

Add the radiative contribution to the net ocean heat flux `Jᵀ` and write
the diagnostic radiative fluxes (upwelling LW, absorbed LW, transmitted SW)
into `coupled_model.radiation.interface_fluxes.ocean`.

When `coupled_model.radiation === nothing`, this is a no-op.
"""
apply_air_sea_radiative_fluxes!(::EarthSystemModel{<:Nothing}) = nothing

function apply_air_sea_radiative_fluxes!(coupled_model::EarthSystemModel)
    ocean = coupled_model.ocean
    isnothing(ocean) && return nothing

    # No atmosphere--ocean interface (no atmosphere or no ocean): nothing to do.
    ao_interface = coupled_model.interfaces.atmosphere_ocean_interface
    isnothing(ao_interface) && return nothing

    radiation = coupled_model.radiation
    interface_fluxes = radiation.interface_fluxes
    isnothing(interface_fluxes) && return nothing
    haskey(interface_fluxes, :ocean) || return nothing

    grid = ocean.model.grid
    arch = architecture(grid)
    clock = coupled_model.clock

    net_ocean_fluxes = coupled_model.interfaces.net_fluxes.ocean
    radiation_state = coupled_model.interfaces.exchanger.radiation.state
    rk = kernel_radiation_properties(radiation)

    sea_ice = coupled_model.sea_ice
    ice_concentration = sea_ice_concentration(sea_ice)

    interface_temperature = ao_interface.temperature
    ocean_properties = coupled_model.interfaces.ocean_properties
    penetrating_radiation = get_radiative_forcing(ocean.model)

    launch!(arch, grid, :xy,
            _apply_air_sea_radiative_fluxes!,
            net_ocean_fluxes,
            interface_fluxes.ocean,
            penetrating_radiation,
            grid,
            clock,
            rk,
            radiation_state,
            ice_concentration,
            interface_temperature,
            ocean_properties)

    return nothing
end

@kernel function _apply_air_sea_radiative_fluxes!(net_ocean_fluxes,
                                                  interface_radiative_flux,
                                                  penetrating_radiation,
                                                  grid,
                                                  clock,
                                                  rk,
                                                  radiation_state,
                                                  ice_concentration,
                                                  ocean_surface_temperature,
                                                  ocean_properties)

    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)
    time = Time(clock.time)

    @inbounds begin
        ℵᵢ = ice_concentration[i, j, 1]
        Tₛ = ocean_surface_temperature[i, j, 1]
    end
    Tₛ = convert_to_kelvin(ocean_properties.temperature_units, Tₛ)

    rs = air_sea_interface_radiation_state(rk, radiation_state, i, j, kᴺ, grid, time)

    ℐꜛˡʷ = rs.σ * rs.ϵ * Tₛ^4
    ℐₐˡʷ = - rs.ϵ * rs.ℐꜜˡʷ
    ℐₜˢʷ = - (1 - rs.α) * rs.ℐꜜˢʷ

    # Multiply by ocean fraction (only the parts not blocked by ice)
    ℐₐˡʷ *= (1 - ℵᵢ)
    ℐₜˢʷ *= (1 - ℵᵢ)
    ℐꜛˡʷ_ocean = ℐꜛˡʷ * (1 - ℵᵢ)

    Qss = shortwave_radiative_forcing(i, j, grid, penetrating_radiation, ℐₜˢʷ, ocean_properties)

    # Total radiative contribution to surface heat flux
    ΣQ_rad = ℐꜛˡʷ_ocean + ℐₐˡʷ + Qss

    ρᵒᶜ⁻¹ = 1 / ocean_properties.reference_density
    cᵒᶜ⁻¹ = 1 / ocean_properties.heat_capacity
    Jᵀ_rad = ΣQ_rad * ρᵒᶜ⁻¹ * cᵒᶜ⁻¹

    inactive = inactive_node(i, j, kᴺ, grid, Center(), Center(), Center())

    @inbounds begin
        net_ocean_fluxes.T[i, j, 1] += ifelse(inactive, zero(grid), Jᵀ_rad)
        interface_radiative_flux.upwelling_longwave[i, j, 1]    = ℐꜛˡʷ
        interface_radiative_flux.downwelling_longwave[i, j, 1]  = - ℐₐˡʷ
        interface_radiative_flux.downwelling_shortwave[i, j, 1] = - ℐₜˢʷ
    end
end
