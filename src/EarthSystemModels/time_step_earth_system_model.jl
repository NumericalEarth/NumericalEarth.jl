using ClimaSeaIce: SeaIceThermodynamics
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: maybe_prepare_first_time_step!

using .InterfaceComputations: compute_atmosphere_ocean_fluxes!,
                              compute_atmosphere_land_fluxes!,
                              compute_sea_ice_ocean_fluxes!

# Hooks called from `update_state!` to apply radiative contributions on top of
# turbulent fluxes. Concrete radiation types overload these (no-op when
# `coupled_model.radiation === nothing`).
apply_air_sea_radiative_fluxes!(::Any) = nothing
apply_air_sea_ice_radiative_fluxes!(::Any) = nothing

fill_interface_flux_halos!(::Nothing) = nothing

function fill_interface_flux_halos!(fluxes::InterfaceComputations.AtmosphereSurfaceFluxes)
    fill_halo_regions!((fluxes.latent_heat, fluxes.sensible_heat, fluxes.water_vapor,
                        fluxes.x_momentum, fluxes.y_momentum, fluxes.friction_velocity,
                        fluxes.temperature_scale, fluxes.water_vapor_scale))
end

function fill_interface_flux_halos!(fluxes::InterfaceComputations.AtmosphereSeaIceFluxes)
    fill_halo_regions!((fluxes.latent_heat, fluxes.sensible_heat, fluxes.water_vapor,
                        fluxes.x_momentum, fluxes.y_momentum))
end

function fill_interface_flux_halos!(fluxes::InterfaceComputations.SeaIceOceanFluxes)
    fill_halo_regions!((fluxes.interface_heat, fluxes.frazil_heat, fluxes.salt,
                        fluxes.x_momentum, fluxes.y_momentum))
end

function Oceananigans.TimeSteppers.time_step!(coupled_model::EarthSystemModel, Δt; callbacks=[])
    maybe_prepare_first_time_step!(coupled_model, Δt, callbacks)

    radiation  = coupled_model.radiation
    atmosphere = coupled_model.atmosphere
    land       = coupled_model.land
    sea_ice    = coupled_model.sea_ice
    ocean      = coupled_model.ocean

    !isnothing(radiation)  && time_step!(radiation, Δt)
    !isnothing(atmosphere) && time_step!(atmosphere, Δt)
    !isnothing(land)       && time_step!(land, Δt)
    !isnothing(sea_ice)    && time_step!(sea_ice, Δt)
    !isnothing(ocean)      && time_step!(ocean, Δt)

    # TODO:
    # - Store fractional ice-free / ice-covered _time_ for more
    #   accurate flux computation?
    tick!(coupled_model.clock, Δt)
    update_state!(coupled_model)

    return nothing
end

function Oceananigans.TimeSteppers.update_state!(coupled_model::EarthSystemModel, callbacks=[])

    radiation  = coupled_model.radiation
    atmosphere = coupled_model.atmosphere
    land       = coupled_model.land
    sea_ice    = coupled_model.sea_ice
    ocean      = coupled_model.ocean

    exchanger = coupled_model.interfaces.exchanger
    grid      = exchanger.grid

    # Phase 1: bring all component states onto the exchange grid
    interpolate_state!(exchanger.radiation,  grid, radiation,  coupled_model)
    interpolate_state!(exchanger.atmosphere, grid, atmosphere, coupled_model)
    interpolate_state!(exchanger.land,       grid, land,       coupled_model)
    interpolate_state!(exchanger.sea_ice,    grid, sea_ice,    coupled_model)
    interpolate_state!(exchanger.ocean,      grid, ocean,      coupled_model)

    # Phase 1.5: apply each component's optional post-regrid correction
    # (no-op when the component carries no correction).
    InterfaceComputations.correct_state!(exchanger.radiation,  grid)
    InterfaceComputations.correct_state!(exchanger.atmosphere, grid)
    InterfaceComputations.correct_state!(exchanger.land,       grid)
    InterfaceComputations.correct_state!(exchanger.sea_ice,    grid)
    InterfaceComputations.correct_state!(exchanger.ocean,      grid)

    !isnothing(exchanger.atmosphere) && fill_halo_regions!(exchanger.atmosphere.state)

    # Phase 2: compute interface turbulent fluxes
    compute_atmosphere_ocean_fluxes!(coupled_model)
    compute_atmosphere_sea_ice_fluxes!(coupled_model)
    compute_atmosphere_land_fluxes!(coupled_model)
    compute_sea_ice_ocean_fluxes!(coupled_model)

    fill_interface_flux_halos!(isnothing(coupled_model.interfaces.atmosphere_ocean_interface) ? nothing : coupled_model.interfaces.atmosphere_ocean_interface.fluxes)
    fill_interface_flux_halos!(isnothing(coupled_model.interfaces.atmosphere_sea_ice_interface) ? nothing : coupled_model.interfaces.atmosphere_sea_ice_interface.fluxes)
    fill_interface_flux_halos!(isnothing(coupled_model.interfaces.sea_ice_ocean_interface) ? nothing : coupled_model.interfaces.sea_ice_ocean_interface.fluxes)
    fill_interface_flux_halos!(isnothing(coupled_model.interfaces.atmosphere_land_interface) ? nothing : coupled_model.interfaces.atmosphere_land_interface.fluxes)

    # Phase 3: assemble net component fluxes (turbulent only)
    update_net_fluxes!(coupled_model, radiation)
    update_net_fluxes!(coupled_model, atmosphere)
    update_net_fluxes!(coupled_model, land)
    update_net_fluxes!(coupled_model, sea_ice)
    update_net_fluxes!(coupled_model, ocean)

    net_ocean_fluxes = coupled_model.interfaces.net_fluxes.ocean
    fill_halo_regions!((net_ocean_fluxes.u, net_ocean_fluxes.v, net_ocean_fluxes.T, net_ocean_fluxes.S))
    fill_halo_regions!(coupled_model.interfaces.net_fluxes.sea_ice)

    # Phase 4: add radiative contributions on top
    apply_air_sea_radiative_fluxes!(coupled_model)
    apply_air_land_radiative_fluxes!(coupled_model)
    apply_air_sea_ice_radiative_fluxes!(coupled_model)

    return nothing
end
