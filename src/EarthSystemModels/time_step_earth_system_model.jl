using .InterfaceComputations:
    compute_atmosphere_ocean_fluxes!,
    compute_atmosphere_land_fluxes!,
    compute_sea_ice_ocean_fluxes!

using Oceananigans.TimeSteppers: maybe_prepare_first_time_step!
using ClimaSeaIce: SeaIceModel, SeaIceThermodynamics
using Oceananigans.Grids: φnode
using Printf

# Hooks called from `update_state!` to apply radiative contributions on top of
# turbulent fluxes. Concrete radiation types overload these (no-op when
# `coupled_model.radiation === nothing`).
apply_air_sea_radiative_fluxes!(::Any) = nothing
apply_air_sea_ice_radiative_fluxes!(::Any) = nothing

sync_atmosphere_land_auxiliary_forcings!(::Nothing, ::Any) = nothing
sync_atmosphere_land_auxiliary_forcings!(::Any, ::Nothing) = nothing

function sync_atmosphere_land_auxiliary_forcings!(land, atmosphere_exchanger)
    # `SlabLand` exposes coupler-supplied fluxes at `land.fluxes`; the legacy
    # `RucSlabLand` (now a SlabLand-with-RUC-closures alias) carries
    # `air_temperature` and `air_humidity` under that NamedTuple. Older
    # land types may have used the now-deprecated `:forcings` slot.
    auxiliary = if hasproperty(land, :fluxes)
        land.fluxes
    elseif hasproperty(land, :forcings)
        land.forcings
    else
        return nothing
    end

    hasproperty(auxiliary, :air_temperature) || return nothing
    hasproperty(auxiliary, :air_humidity) || return nothing

    atmosphere_state = atmosphere_exchanger.state

    parent(auxiliary.air_temperature) .= parent(atmosphere_state.T)
    parent(auxiliary.air_humidity) .= parent(atmosphere_state.q)

    return nothing
end

function time_step!(coupled_model::EarthSystemModel, Δt; callbacks=[])
    maybe_prepare_first_time_step!(coupled_model, callbacks)

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

function update_state!(coupled_model::EarthSystemModel, callbacks=[])

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
    sync_atmosphere_land_auxiliary_forcings!(land, exchanger.atmosphere)
    interpolate_state!(exchanger.land,       grid, land,       coupled_model)
    interpolate_state!(exchanger.sea_ice,    grid, sea_ice,    coupled_model)
    interpolate_state!(exchanger.ocean,      grid, ocean,      coupled_model)

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
    apply_air_sea_ice_radiative_fluxes!(coupled_model)

    return nothing
end
