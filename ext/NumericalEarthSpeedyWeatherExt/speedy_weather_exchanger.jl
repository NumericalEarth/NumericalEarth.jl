using Oceananigans.Architectures: architecture
using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger, exchange_over_water, regrid_over_water!

# The fluxes go straight to SpeedyWeather's prognostic ocean in `update_net_fluxes!`
NumericalEarth.EarthSystemModels.InterfaceComputations.net_fluxes(::SpeedySimulation) = nothing

Oceananigans.Grids.grid(atmosphere::SpeedySimulation) = atmosphere.model.spectral_grid.grid

Oceananigans.Architectures.on_architecture(::Oceananigans.CPU, grid::RingGrids.AbstractGrid) =
    SpeedyWeather.on_architecture(SpeedyWeather.CPU(), grid)

function NumericalEarth.EarthSystemModels.surface_state(atmosphere::SpeedySimulation)
    surface_values = atmosphere.variables.parameterizations.rain_rate.data
    return (; u    = zero(surface_values),
              v    = zero(surface_values),
              T    = zero(surface_values),
              p    = zero(surface_values),
              q    = zero(surface_values),
              Jʳⁿ  = zero(surface_values),
              Jˢⁿ  = zero(surface_values))
end

function NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(atmosphere::SpeedySimulation, exchange_grid;
                                                                                   correction = nothing)
    arch = architecture(exchange_grid)
    atmosphere_architecture = atmosphere.model.spectral_grid.architecture
    if (arch isa Oceananigans.CPU) != (atmosphere_architecture isa SpeedyWeather.CPU)
        error("The exchange grid is on $arch but the SpeedyWeather atmosphere is on \
              $atmosphere_architecture. The atmosphere and the exchange grid must run on the \
              same architecture (both CPU or both GPU) for the ConservativeRegridding regridders \
              to be transferred correctly.")
    end

    exchanger = invoke(ComponentExchanger, Tuple{Any, Any}, atmosphere, exchange_grid; correction)
    wet_fraction = similar(atmosphere.variables.prognostic.ocean.sea_surface_temperature.data)
    return exchange_over_water(exchanger, exchange_grid, wet_fraction)
end

function NumericalEarth.EarthSystemModels.update_surface_state!(state, atmos::SpeedySimulation)
    surface_layer   = atmos.model.spectral_grid.nlayers
    prognostic_step = SpeedyWeather.which_prognostic_step(atmos.variables.grid.u, atmos.model.time_stepping, SpeedyWeather.DummyParameterization())
    parameterizations = atmos.variables.parameterizations

    state.u .= RingGrids.field_view(atmos.variables.grid.u, :, surface_layer, prognostic_step).data
    state.v .= RingGrids.field_view(atmos.variables.grid.v, :, surface_layer, prognostic_step).data
    state.T .= RingGrids.field_view(atmos.variables.grid.temperature, :, surface_layer, prognostic_step).data
    state.q .= RingGrids.field_view(atmos.variables.grid.humidity, :, surface_layer, prognostic_step).data
    state.p .= exp.(RingGrids.field_view(atmos.variables.grid.pressure, :, prognostic_step).data)

    state.Jʳⁿ .= parameterizations.rain_rate.data

    # `snow_rate` is only registered when SpeedyWeather's large-scale
    # condensation parameterization is part of the model
    haskey(parameterizations, :snow_rate) && (state.Jˢⁿ .= parameterizations.snow_rate.data)

    return nothing
end

function NumericalEarth.EarthSystemModels.update_net_fluxes!(coupled_model, atmos::SpeedySimulation)
    regridder = coupled_model.interfaces.exchanger.atmosphere.regridder
    partition = coupled_model.interfaces.surface_partition
    ocean = atmos.variables.prognostic.ocean

    regrid_over_water!(ocean.sensible_heat_flux.data, regridder, partition.fluxes.sensible_heat)
    regrid_over_water!(ocean.surface_humidity_flux.data, regridder, partition.fluxes.water_vapor)

    return nothing
end
