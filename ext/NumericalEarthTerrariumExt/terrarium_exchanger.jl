using Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: architecture
using Oceananigans.Fields: set!, interior

import NumericalEarth.EarthSystemModels: update_net_fluxes!, interpolate_state!
import NumericalEarth.EarthSystemModels.InterfaceComputations: net_fluxes, ComponentExchanger

net_fluxes(::LandSimulation) = nothing

# Land exchanger constructor.
# For now, no regridder is needed since the exchange grid is assumed to match the land grid.
# The state holds a single surface temperature field that is communicated back to the atmosphere.
# TODO: add regridder when exchange grid differs from land grid.
function ComponentExchanger(land::LandSimulation, exchange_grid)
    regridder = nothing
    state = (; Ts = Field{Center, Center, Nothing}(exchange_grid))
    return ComponentExchanger(state, regridder)
end

# Read the land surface state onto the exchange grid.
# Currently: exchange grid == land grid → direct copy, no regridding.
# TODO: regrid when exchange_grid differs from land grid.
function interpolate_state!(exchanger, exchange_grid, land::LandSimulation, coupled_model)
    Ts_land = land.state.skin_temperature  # °C
    Ts_exchange = exchanger.state.Ts

    # Convert skin temperature from °C to K and copy to exchange grid field
    set!(Ts_exchange, Ts_land + 273.15)
    fill_halo_regions!(Ts_exchange) # TODO: is this necessary?

    return nothing
end

# Update Terrarium land model inputs from the atmospheric state on the exchange grid.
# This is the primary atmosphere → land coupling step.
# TODO: regrid when land grid differs from the exchange grid.
function update_net_fluxes!(coupled_model, land::LandSimulation)
    atmos_state = coupled_model.interfaces.exchanger.atmosphere.state

    # Air temperature: atmosphere provides K, Terrarium expects °C
    set!(land.state.air_temperature, atmos_state.T - 273.15)

    # Wind speed: compute magnitude from (u, v) components
    # TODO: implement as a proper GPU-compatible kernel
    set!(land.state.windspeed, sqrt(atmos_state.u^2 + atmos_state.v^2))

    # Remaining atmospheric scalars: direct copy (same units)
    set!(land.state.specific_humidity, atmos_state.q)
    set!(land.state.air_pressure, atmos_state.p)
    set!(land.state.surface_shortwave_down, atmos_state.ℐꜜˢʷ)
    set!(land.state.surface_longwave_down, atmos_state.ℐꜜˡʷ)

    # Total precipitation → rainfall; snowfall set to zero.
    # TODO: partition rain/snow based on air temperature.
    set!(land.state.rainfall, atmos_state.Jᶜ)
    set!(land.state.snowfall, 0)

    return nothing
end
