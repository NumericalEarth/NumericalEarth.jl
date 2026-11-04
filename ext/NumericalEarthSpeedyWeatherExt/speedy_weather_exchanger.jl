using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger, exchange_regridder, regrid_surface_field!,
                                                              rotate_to_exchange_frame!, vector_component_boundary_conditions

# We do not need this...
NumericalEarth.EarthSystemModels.InterfaceComputations.net_fluxes(::SpeedySimulation) = nothing

function NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(atmosphere::SpeedySimulation, exchange_grid;
                                                                                   correction = nothing)
    spectral_grid = atmosphere.model.spectral_grid.grid

    arch = architecture(exchange_grid)
    atmosphere_architecture = atmosphere.model.spectral_grid.architecture
    if (arch isa Oceananigans.CPU) != (atmosphere_architecture isa SpeedyWeather.CPU)
        error("The exchange grid is on $arch but the SpeedyWeather atmosphere is on \
              $atmosphere_architecture. The atmosphere and the exchange grid must run on the \
              same architecture (both CPU or both GPU) for the ConservativeRegridding regridders \
              to be transferred correctly.")
    end

    # TODO: distributed GPUs?
    cpu_spectral_grid = SpeedyWeather.on_architecture(SpeedyWeather.CPU(), spectral_grid)
    regridder = exchange_regridder(cpu_spectral_grid, exchange_grid)

    velocity_bcs = vector_component_boundary_conditions(exchange_grid, (Center(), Center(), nothing))

    state = (; u    = Field{Center, Center, Nothing}(exchange_grid; boundary_conditions = velocity_bcs),
               v    = Field{Center, Center, Nothing}(exchange_grid; boundary_conditions = velocity_bcs),
               T    = Field{Center, Center, Nothing}(exchange_grid),
               p    = Field{Center, Center, Nothing}(exchange_grid),
               q    = Field{Center, Center, Nothing}(exchange_grid),
               ℐꜜˢʷ = Field{Center, Center, Nothing}(exchange_grid),
               ℐꜜˡʷ = Field{Center, Center, Nothing}(exchange_grid),
               Jʳⁿ  = Field{Center, Center, Nothing}(exchange_grid),
               Jˢⁿ  = Field{Center, Center, Nothing}(exchange_grid))

    correction = NumericalEarth.EarthSystemModels.InterfaceComputations.materialize_correction(correction, exchange_grid, atmosphere)
    return ComponentExchanger(state, regridder, correction)
end

# Regrid the atmospheric state on the exchange grid
function NumericalEarth.EarthSystemModels.interpolate_state!(exchanger, exchange_grid, atmos::SpeedySimulation, coupled_model)
    from_atmosphere = exchanger.regridder.from_component
    exchange_state  = exchanger.state
    surface_layer   = atmos.model.spectral_grid.nlayers
    prognostic_step = SpeedyWeather.which_prognostic_step(atmos.variables.grid.u, atmos.model.time_stepping, SpeedyWeather.DummyParameterization())

    ua   = RingGrids.field_view(atmos.variables.grid.u, :, surface_layer, prognostic_step).data
    va   = RingGrids.field_view(atmos.variables.grid.v, :, surface_layer, prognostic_step).data
    Ta   = RingGrids.field_view(atmos.variables.grid.temperature, :, surface_layer, prognostic_step).data
    qa   = RingGrids.field_view(atmos.variables.grid.humidity, :, surface_layer, prognostic_step).data
    pa   = exp.(RingGrids.field_view(atmos.variables.grid.pressure, :, prognostic_step).data)
    ℐꜜˢʷ = atmos.variables.parameterizations.surface_shortwave_down.data
    ℐꜜˡʷ = atmos.variables.parameterizations.surface_longwave_down.data
    Jʳⁿ  = atmos.variables.parameterizations.rain_rate.data

    # `snow_rate` is only registered when SpeedyWeather's large-scale
    # condensation parameterization is part of the model
    Jˢⁿ = haskey(atmos.variables.parameterizations, :snow_rate) ?
          atmos.variables.parameterizations.snow_rate.data : nothing

    regrid_surface_field!(exchange_state.u,    from_atmosphere, ua)
    regrid_surface_field!(exchange_state.v,    from_atmosphere, va)
    regrid_surface_field!(exchange_state.T,    from_atmosphere, Ta)
    regrid_surface_field!(exchange_state.q,    from_atmosphere, qa)
    regrid_surface_field!(exchange_state.p,    from_atmosphere, pa)
    regrid_surface_field!(exchange_state.ℐꜜˢʷ, from_atmosphere, ℐꜜˢʷ)
    regrid_surface_field!(exchange_state.ℐꜜˡʷ, from_atmosphere, ℐꜜˡʷ)
    parent(exchange_state.ℐꜜˢʷ) .*= -1
    parent(exchange_state.ℐꜜˡʷ) .*= -1
    regrid_surface_field!(exchange_state.Jʳⁿ,  from_atmosphere, Jʳⁿ)
    isnothing(Jˢⁿ) || regrid_surface_field!(exchange_state.Jˢⁿ, from_atmosphere, Jˢⁿ)

    u = exchange_state.u
    v = exchange_state.v

    rotate_to_exchange_frame!(u, v, exchange_grid)

    fill_halo_regions!((u, v))
    fill_halo_regions!(exchange_state.T)
    fill_halo_regions!(exchange_state.q)
    fill_halo_regions!(exchange_state.p)
    fill_halo_regions!(exchange_state.ℐꜜˢʷ)
    fill_halo_regions!(exchange_state.ℐꜜˡʷ)
    fill_halo_regions!(exchange_state.Jʳⁿ)
    isnothing(Jˢⁿ) || fill_halo_regions!(exchange_state.Jˢⁿ)

    return nothing
end

function NumericalEarth.EarthSystemModels.update_net_fluxes!(coupled_model, atmos::SpeedySimulation)
    to_atmosphere = coupled_model.interfaces.exchanger.atmosphere.regridder.to_component
    partition = coupled_model.interfaces.surface_partition

    # TODO: Figure out how we are going to deal with upwelling radiation
    regrid_surface_field!(atmos.variables.parameterizations.ocean.sensible_heat_flux.data, to_atmosphere, partition.fluxes.sensible_heat)
    regrid_surface_field!(atmos.variables.parameterizations.ocean.surface_humidity_flux.data, to_atmosphere, partition.fluxes.water_vapor)
    regrid_surface_field!(atmos.variables.prognostic.ocean.sea_surface_temperature.data, to_atmosphere, partition.surface_temperature)

    return nothing
end
