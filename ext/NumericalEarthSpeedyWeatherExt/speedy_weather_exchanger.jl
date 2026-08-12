using ConservativeRegridding: ConservativeRegridding, Regridder, regrid!
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: BoundaryConditions, fill_halo_regions!
using Oceananigans.Operators: intrinsic_vector
using Oceananigans.Utils: launch!
using NumericalEarth.EarthSystemModels: sea_ice_concentration
using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger

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

    # build regridders on the CPU and move the result onto `arch` afterwards
    # TODO: distributed GPUs?
    cpu_exchange_grid = Oceananigans.on_architecture(Oceananigans.CPU(), exchange_grid)
    cpu_spectral_grid = SpeedyWeather.on_architecture(SpeedyWeather.CPU(), spectral_grid)

    # Use the exchange_grid's manifold for both grids to avoid radius mismatch between Oceananigans and SpeedyWeather
    manifold = ConservativeRegridding.GOCore.best_manifold(cpu_exchange_grid)
    from_atmosphere = Regridder(manifold, cpu_exchange_grid, cpu_spectral_grid)
    to_atmosphere   = Regridder(manifold, cpu_spectral_grid, cpu_exchange_grid)

    to_atmosphere   = Oceananigans.on_architecture(arch, to_atmosphere)
    from_atmosphere = Oceananigans.on_architecture(arch, from_atmosphere)
    regridder = (; to_atmosphere, from_atmosphere)

    state = (; u    = Field{Center, Center, Nothing}(exchange_grid),
               v    = Field{Center, Center, Nothing}(exchange_grid),
               T    = Field{Center, Center, Nothing}(exchange_grid),
               p    = Field{Center, Center, Nothing}(exchange_grid),
               q    = Field{Center, Center, Nothing}(exchange_grid),
               ℐꜜˢʷ = Field{Center, Center, Nothing}(exchange_grid),
               ℐꜜˡʷ = Field{Center, Center, Nothing}(exchange_grid),
               Jʳⁿ  = Field{Center, Center, Nothing}(exchange_grid),
               Jˢⁿ  = Field{Center, Center, Nothing}(exchange_grid),
               tmp  = Field{Center, Center, Nothing}(exchange_grid),  # allocate scratch space
            )


    correction = NumericalEarth.EarthSystemModels.InterfaceComputations.materialize_correction(correction, exchange_grid, atmosphere)
    return ComponentExchanger(state, regridder, correction)
end

function ConservativeRegridding.regrid!(field::Oceananigans.Field, regridder::Regridder, data::AbstractArray)
    regrid!(vec(interior(field)), regridder, vec(data))
end

function ConservativeRegridding.regrid!(data::AbstractArray, regridder::Regridder, field::Oceananigans.Field)
    regrid!(vec(data), regridder, vec(interior(field)))
end

# Regrid the atmospheric state on the exchange grid
function NumericalEarth.EarthSystemModels.interpolate_state!(exchanger, exchange_grid, atmos::SpeedySimulation, coupled_model)
    from_atmosphere = exchanger.regridder.from_atmosphere
    exchange_state  = exchanger.state
    surface_layer   = atmos.model.spectral_grid.nlayers

    ua   = RingGrids.field_view(atmos.variables.grid.u, :, surface_layer).data
    va   = RingGrids.field_view(atmos.variables.grid.v, :, surface_layer).data
    Ta   = RingGrids.field_view(atmos.variables.grid.temperature, :, surface_layer).data
    qa   = RingGrids.field_view(atmos.variables.grid.humidity, :, surface_layer).data
    pa   = exp.(atmos.variables.grid.pressure.data)
    ℐꜜˢʷ = atmos.variables.parameterizations.surface_shortwave_down.data
    ℐꜜˡʷ = atmos.variables.parameterizations.surface_longwave_down.data
    Jʳⁿ  = atmos.variables.parameterizations.rain_rate.data

    # `snow_rate` is only registered when SpeedyWeather's large-scale
    # condensation parameterization is part of the model
    Jˢⁿ = haskey(atmos.variables.parameterizations, :snow_rate) ?
          atmos.variables.parameterizations.snow_rate.data : nothing

    regrid!(exchange_state.u,     from_atmosphere, ua)
    regrid!(exchange_state.v,     from_atmosphere, va)
    regrid!(exchange_state.T,     from_atmosphere, Ta)
    regrid!(exchange_state.q,     from_atmosphere, qa)
    regrid!(exchange_state.p,     from_atmosphere, pa)
    regrid!(exchange_state.ℐꜜˢʷ,  from_atmosphere, ℐꜜˢʷ)
    regrid!(exchange_state.ℐꜜˡʷ,  from_atmosphere, ℐꜜˡʷ)
    regrid!(exchange_state.Jʳⁿ,   from_atmosphere, Jʳⁿ)
    isnothing(Jˢⁿ) || regrid!(exchange_state.Jˢⁿ, from_atmosphere, Jˢⁿ)

    arch = architecture(exchange_grid)

    u = exchange_state.u
    v = exchange_state.v

    launch!(arch, exchange_grid, :xy, _rotate_winds!, u, v, exchange_grid)

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

@kernel function _rotate_winds!(u, v, grid)
    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)
    uₑ, vₑ = intrinsic_vector(i, j, kᴺ, grid, u, v)
    @inbounds u[i, j, kᴺ] = uₑ
    @inbounds v[i, j, kᴺ] = vₑ
end

# TODO: Fix the coupling with the sea ice model and make sure that
# this function works also for sea_ice=nothing
function NumericalEarth.EarthSystemModels.update_net_fluxes!(coupled_model, atmos::SpeedySimulation)
    to_atmosphere = coupled_model.interfaces.exchanger.atmosphere.regridder.to_atmosphere
    tmp = coupled_model.interfaces.exchanger.atmosphere.state.tmp
    ao_fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes
    ai_fluxes = coupled_model.interfaces.atmosphere_sea_ice_interface.fluxes

    𝒬ᵀᵃᵒ = ao_fluxes.sensible_heat
    𝒬ᵀᵃⁱ = ai_fluxes.sensible_heat
    Jᵛᵃᵒ = ao_fluxes.water_vapor
    Jᵛᵃⁱ = ai_fluxes.water_vapor
    ℵ    = sea_ice_concentration(coupled_model.sea_ice)


    # All the location of these fluxes will change
    𝒬ᵀ_speedy = atmos.variables.parameterizations.ocean.sensible_heat_flux.data
    Jᵛ_speedy = atmos.variables.parameterizations.ocean.surface_humidity_flux.data
    sst = atmos.variables.prognostic.ocean.sea_surface_temperature.data
    To  = coupled_model.interfaces.atmosphere_ocean_interface.temperature
    Ti  = coupled_model.interfaces.atmosphere_sea_ice_interface.temperature

    # TODO: Figure out how we are going to deal with upwelling radiation
    # TODO: regrid longwave rather than a mixed surface temperature
    tmp .= 𝒬ᵀᵃᵒ * (1 - ℵ) + ℵ * 𝒬ᵀᵃⁱ
    regrid!(𝒬ᵀ_speedy, to_atmosphere, tmp)
    tmp .= Jᵛᵃᵒ * (1 - ℵ) + ℵ * Jᵛᵃⁱ
    regrid!(Jᵛ_speedy, to_atmosphere, tmp)
    tmp .= To * (1 - ℵ) + ℵ * Ti + 273.15
    regrid!(sst, to_atmosphere, tmp)

    return nothing
end

# Simple case -> there is no sea ice!
function NumericalEarth.EarthSystemModels.update_net_fluxes!(coupled_model::SpeedyNoSeaIceEarthSystemModel, atmos::SpeedySimulation)
    to_atmosphere = coupled_model.interfaces.exchanger.atmosphere.regridder.to_atmosphere
    tmp       = coupled_model.interfaces.exchanger.atmosphere.state.tmp
    ao_fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes
    𝒬ᵀᵃᵒ = ao_fluxes.sensible_heat
    Jᵛᵃᵒ = ao_fluxes.water_vapor

    # All the location of these fluxes will change
    𝒬ᵀ_speedy = atmos.variables.parameterizations.ocean.sensible_heat_flux.data
    Jᵛ_speedy = atmos.variables.parameterizations.ocean.surface_humidity_flux.data
    sst = atmos.variables.prognostic.ocean.sea_surface_temperature.data
    To  = coupled_model.interfaces.atmosphere_ocean_interface.temperature

    # TODO: Figure out how we are going to deal with upwelling radiation
    regrid!(𝒬ᵀ_speedy, to_atmosphere, 𝒬ᵀᵃᵒ)
    regrid!(Jᵛ_speedy, to_atmosphere, Jᵛᵃᵒ)
    tmp .= To .+ 273.15
    regrid!(sst, to_atmosphere, tmp)

    return nothing
end
