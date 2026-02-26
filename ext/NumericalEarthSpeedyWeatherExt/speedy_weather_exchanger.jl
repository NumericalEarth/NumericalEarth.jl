using Oceananigans
using Oceananigans.BoundaryConditions
using Oceananigans.Grids: architecture
using Oceananigans.Utils: launch!
using Oceananigans.Operators: intrinsic_vector
using XESMF

using NumericalEarth.EarthSystemModels: sea_ice_concentration

# TODO: Implement conservative regridding when ready
# using ConservativeRegridding
# using GeoInterface: Polygon, LinearRing
import NumericalEarth.EarthSystemModels: update_net_fluxes!, interpolate_state!
import NumericalEarth.Atmospheres: atmosphere_regridder
import NumericalEarth.EarthSystemModels.InterfaceComputations: net_fluxes, ComponentExchanger

# We do not need this...
net_fluxes(::SpeedySimulation) = nothing

# For the moment the workflow is:
# 1. Perform the regridding on the CPU
# 2. Eventually copy the regridded fields to the GPU
# If this work we can
# 1. Copy speedyweather gridarrays to the GPU
# 2. Perform the regridding on the GPU
function ComponentExchanger(atmosphere::SpeedySimulation, exchange_grid) 

    spectral_grid = atmosphere.model.spectral_grid
    # TODO: Implement a conservative regridder when ready
    from_atmosphere = XESMF.Regridder(spectral_grid, exchange_grid)
    to_atmosphere   = XESMF.Regridder(exchange_grid, spectral_grid)
    regridder = (; to_atmosphere, from_atmosphere)
    
    state = (; u  = Field{Center, Center, Nothing}(exchange_grid),
               v  = Field{Center, Center, Nothing}(exchange_grid),
               T  = Field{Center, Center, Nothing}(exchange_grid),
               p  = Field{Center, Center, Nothing}(exchange_grid),
               q  = Field{Center, Center, Nothing}(exchange_grid),
               ℐꜜˢʷ = Field{Center, Center, Nothing}(exchange_grid),
               ℐꜜˡʷ = Field{Center, Center, Nothing}(exchange_grid),
               Jᶜ = Field{Center, Center, Nothing}(exchange_grid))
    
    return ComponentExchanger(state, regridder)
end

@inline (regrid!::XESMF.Regridder)(field::Oceananigans.Field, data::AbstractArray) = regrid!(vec(interior(field)), data)
@inline (regrid!::XESMF.Regridder)(data::AbstractArray, field::Oceananigans.Field) = regrid!(data, vec(interior(field)))

# Regrid the atmospheric state on the exchange grid
function interpolate_state!(exchanger, exchange_grid, atmos::SpeedySimulation, coupled_model)
    regrid!        = exchanger.regridder.from_atmosphere
    exchange_state = exchanger.state
    surface_layer  = atmos.model.spectral_grid.nlayers

    ua  = RingGrids.field_view(atmos.diagnostic_variables.grid.u_grid,     :, surface_layer).data
    va  = RingGrids.field_view(atmos.diagnostic_variables.grid.v_grid,     :, surface_layer).data
    Ta  = RingGrids.field_view(atmos.diagnostic_variables.grid.temp_grid,  :, surface_layer).data
    qa  = RingGrids.field_view(atmos.diagnostic_variables.grid.humid_grid, :, surface_layer).data
    pa  = exp.(atmos.diagnostic_variables.grid.pres_grid.data)
    ℐꜜˢʷ = atmos.diagnostic_variables.physics.surface_shortwave_down.data
    ℐꜜˡʷ = atmos.diagnostic_variables.physics.surface_longwave_down.data
    Jᶜ  = atmos.diagnostic_variables.physics.total_precipitation_rate.data

    regrid!(exchange_state.u,    ua)
    regrid!(exchange_state.v,    va)
    regrid!(exchange_state.T,    Ta)
    regrid!(exchange_state.q,    qa)
    regrid!(exchange_state.p,    pa)
    regrid!(exchange_state.ℐꜜˢʷ, ℐꜜˢʷ)
    regrid!(exchange_state.ℐꜜˡʷ, ℐꜜˡʷ)
    regrid!(exchange_state.Jᶜ,   Jᶜ)

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
    fill_halo_regions!(exchange_state.Jᶜ)

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
# the this function works also for sea_ice=nothing and on GPUs without
# needing to allocate memory.
function update_net_fluxes!(coupled_model, atmos::SpeedySimulation)
    regrid!   = coupled_model.interfaces.exchanger.atmosphere.regridder.to_atmosphere
    ao_fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes
    ai_fluxes = coupled_model.interfaces.atmosphere_sea_ice_interface.fluxes

    𝒬ᵀᵃᵒ = ao_fluxes.sensible_heat
    𝒬ᵀᵃⁱ = ai_fluxes.sensible_heat
    Jᵛᵃᵒ = ao_fluxes.water_vapor
    Jᵛᵃⁱ = ai_fluxes.water_vapor
    ℵ    = interior(sea_ice_concentration(coupled_model.sea_ice))

    # All the location of these fluxes will change
    𝒬ᵀ_speedy = atmos.prognostic_variables.ocean.sensible_heat_flux.data
    Jᵛ_speedy = atmos.prognostic_variables.ocean.surface_humidity_flux.data
    sst = atmos.prognostic_variables.ocean.sea_surface_temperature.data
    To  = coupled_model.interfaces.atmosphere_ocean_interface.temperature
    Ti  = coupled_model.interfaces.atmosphere_sea_ice_interface.temperature

    # TODO: Figure out how we are going to deal with upwelling radiation
    # TODO: regrid longwave rather than a mixed surface temperature
    # TODO: This does not work on GPUs!!
    regrid!(𝒬ᵀ_speedy, vec(interior(𝒬ᵀᵃᵒ) .* (1 .- ℵ) .+ ℵ .* interior(𝒬ᵀᵃⁱ)))
    regrid!(Jᵛ_speedy, vec(interior(Jᵛᵃᵒ) .* (1 .- ℵ) .+ ℵ .* interior(Jᵛᵃⁱ)))
    regrid!(sst, vec(interior(To)  .* (1 .- ℵ) .+ ℵ .* interior(Ti) .+ 273.15))

    return nothing
end

# Simple case -> there is no sea ice!
function update_net_fluxes!(coupled_model::SpeedyNoSeaIceEarthSystemModel, atmos::SpeedySimulation)
    regrid!   = coupled_model.interfaces.exchanger.atmosphere.regridder.to_atmosphere
    ao_fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes
    𝒬ᵀᵃᵒ = ao_fluxes.sensible_heat
    Jᵛᵃᵒ = ao_fluxes.water_vapor

    # All the location of these fluxes will change
    𝒬ᵀ_speedy = atmos.prognostic_variables.ocean.sensible_heat_flux.data
    Jᵛ_speedy = atmos.prognostic_variables.ocean.surface_humidity_flux.data
    sst = atmos.prognostic_variables.ocean.sea_surface_temperature.data
    To  = coupled_model.interfaces.atmosphere_ocean_interface.temperature

    # TODO: Figure out how we are going to deal with upwelling radiation
    # TODO: This does not work on GPUs!!
    regrid!(𝒬ᵀ_speedy, vec(interior(𝒬ᵀᵃᵒ)))
    regrid!(Jᵛ_speedy, vec(interior(Jᵛᵃᵒ)))
    regrid!(sst, vec(interior(To) .+ 273.15))

    return nothing
end
