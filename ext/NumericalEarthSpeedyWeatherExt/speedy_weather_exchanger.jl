using Oceananigans.Architectures: architecture
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Fields: compute!
using Oceananigans.Grids: inactive_node
using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger, regrid_surface_field!

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
              ℐꜜˢʷ = zero(surface_values),
              ℐꜜˡʷ = zero(surface_values),
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

    wet_mask = Field(KernelFunctionOperation{Center, Center, Nothing}(wet_cell, exchange_grid))
    compute!(wet_mask)
    wet_field = Field{Center, Center, Nothing}(exchange_grid)
    wet_fraction = similar(atmosphere.variables.prognostic.ocean.sea_surface_temperature.data)
    regrid_surface_field!(wet_fraction, exchanger.regridder.exchange.to_component, wet_mask)

    regridder = merge(exchanger.regridder, (; wet_mask, wet_field, wet_fraction))

    return ComponentExchanger(exchanger.state, regridder, exchanger.correction)
end

@inline wet_cell(i, j, k, grid) = !inactive_node(i, j, size(grid, 3), grid, Center(), Center(), Center())

function NumericalEarth.EarthSystemModels.update_surface_state!(state, atmos::SpeedySimulation)
    surface_layer   = atmos.model.spectral_grid.nlayers
    prognostic_step = SpeedyWeather.which_prognostic_step(atmos.variables.grid.u, atmos.model.time_stepping, SpeedyWeather.DummyParameterization())
    parameterizations = atmos.variables.parameterizations

    state.u .= RingGrids.field_view(atmos.variables.grid.u, :, surface_layer, prognostic_step).data
    state.v .= RingGrids.field_view(atmos.variables.grid.v, :, surface_layer, prognostic_step).data
    state.T .= RingGrids.field_view(atmos.variables.grid.temperature, :, surface_layer, prognostic_step).data
    state.q .= RingGrids.field_view(atmos.variables.grid.humidity, :, surface_layer, prognostic_step).data
    state.p .= exp.(RingGrids.field_view(atmos.variables.grid.pressure, :, prognostic_step).data)

    state.ℐꜜˢʷ .= .- parameterizations.surface_shortwave_down.data
    state.ℐꜜˡʷ .= .- parameterizations.surface_longwave_down.data
    state.Jʳⁿ  .= parameterizations.rain_rate.data

    # `snow_rate` is only registered when SpeedyWeather's large-scale
    # condensation parameterization is part of the model
    haskey(parameterizations, :snow_rate) && (state.Jˢⁿ .= parameterizations.snow_rate.data)

    return nothing
end

# The area average over the water below each atmospheric cell, the per-unit-ocean value that SpeedyWeather reads
function regrid_over_water!(destination, regridder, field)
    parent(regridder.wet_field) .= parent(regridder.wet_mask) .* parent(field)
    regrid_surface_field!(destination, regridder.exchange.to_component, regridder.wet_field)
    destination .= ifelse.(regridder.wet_fraction .> 0, destination ./ regridder.wet_fraction, destination)
    return destination
end

function NumericalEarth.EarthSystemModels.update_net_fluxes!(coupled_model, atmos::SpeedySimulation)
    regridder = coupled_model.interfaces.exchanger.atmosphere.regridder
    partition = coupled_model.interfaces.surface_partition
    ocean = atmos.variables.prognostic.ocean

    # TODO: Figure out how we are going to deal with upwelling radiation
    regrid_over_water!(ocean.sensible_heat_flux.data, regridder, partition.fluxes.sensible_heat)
    regrid_over_water!(ocean.surface_humidity_flux.data, regridder, partition.fluxes.water_vapor)
    regrid_over_water!(ocean.sea_surface_temperature.data, regridder, partition.surface_temperature)

    return nothing
end
