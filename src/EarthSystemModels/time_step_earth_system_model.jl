using ClimaSeaIce: SeaIceThermodynamics
using Oceananigans.Architectures: architecture
using Oceananigans.TimeSteppers: maybe_prepare_first_time_step!
using Oceananigans.Utils: sync_device!

using .InterfaceComputations: compute_atmosphere_ocean_fluxes!,
                              compute_atmosphere_land_fluxes!,
                              compute_sea_ice_ocean_fluxes!

# Hooks called from `update_state!` to apply radiative contributions on top of
# turbulent fluxes. Concrete radiation types overload these (no-op when
# `coupled_model.radiation === nothing`).
apply_air_sea_radiative_fluxes!(::Any) = nothing
apply_air_sea_ice_radiative_fluxes!(::Any) = nothing

debug_earth_system_sync_enabled() =
    lowercase(get(ENV, "RYF_DEBUG_CUDA_SYNC", "false")) in ("1", "true", "yes")

function debug_earth_system_sync!(label, coupled_model)
    debug_earth_system_sync_enabled() || return nothing

    try
        sync_device!(architecture(coupled_model))
        @info "CUDA sync passed during earth system time_step!" label
    catch err
        @error "CUDA sync failed during earth system time_step!" label exception=(err, catch_backtrace())
        rethrow()
    end

    return nothing
end

function Oceananigans.TimeSteppers.time_step!(coupled_model::EarthSystemModel, Δt; callbacks=[])
    debug_earth_system_sync!("before maybe_prepare_first_time_step!", coupled_model)
    maybe_prepare_first_time_step!(coupled_model, Δt, callbacks)
    debug_earth_system_sync!("after maybe_prepare_first_time_step!", coupled_model)

    radiation  = coupled_model.radiation
    atmosphere = coupled_model.atmosphere
    land       = coupled_model.land
    sea_ice    = coupled_model.sea_ice
    ocean      = coupled_model.ocean

    !isnothing(radiation)  && time_step!(radiation, Δt)
    debug_earth_system_sync!("after radiation time_step!", coupled_model)
    !isnothing(atmosphere) && time_step!(atmosphere, Δt)
    debug_earth_system_sync!("after atmosphere time_step!", coupled_model)
    !isnothing(land)       && time_step!(land, Δt)
    debug_earth_system_sync!("after land time_step!", coupled_model)
    # Ocean before sea ice: the ice-ocean drag is evaluated against the just-updated ocean velocity.
    !isnothing(ocean)      && time_step!(ocean, Δt)
    debug_earth_system_sync!("after ocean time_step!", coupled_model)
    !isnothing(sea_ice)    && time_step!(sea_ice, Δt)
    debug_earth_system_sync!("after sea ice time_step!", coupled_model)

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

    # Phase 2: compute interface turbulent fluxes
    compute_atmosphere_ocean_fluxes!(coupled_model)
    compute_atmosphere_sea_ice_fluxes!(coupled_model)
    compute_atmosphere_land_fluxes!(coupled_model)
    compute_sea_ice_ocean_fluxes!(coupled_model)

    # Phase 3: assemble net component fluxes (turbulent only)
    update_net_fluxes!(coupled_model, radiation)
    update_net_fluxes!(coupled_model, atmosphere)
    update_net_fluxes!(coupled_model, land)
    update_net_fluxes!(coupled_model, sea_ice)
    update_net_fluxes!(coupled_model, ocean)

    # Phase 4: add radiative contributions on top
    apply_air_sea_radiative_fluxes!(coupled_model)
    apply_air_land_radiative_fluxes!(coupled_model)
    apply_air_sea_ice_radiative_fluxes!(coupled_model)

    return nothing
end
