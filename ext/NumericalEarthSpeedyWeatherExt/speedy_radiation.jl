using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: @kernel, @index
using Oceananigans.Units: Time
using Oceananigans.Utils: launch!
using NumericalEarth.EarthSystemModels: sea_ice_concentration
using NumericalEarth.EarthSystemModels.InterfaceComputations: air_sea_interface_radiation_state,
                                                              air_sea_ice_interface_radiation_state
using NumericalEarth.Radiations: InterfaceRadiationFlux, SurfaceRadiationProperties, default_water_emissivity

mutable struct SpeedyRadiation{A, S, FT}
    atmosphere :: A
    surface_properties :: S
    stefan_boltzmann_constant :: FT
    # Allocated once the exchange grid is known
    interface_fluxes
    albedo
    upwelling_longwave
end

"""
$(TYPEDSIGNATURES)

Return the radiation component of a coupled model with a SpeedyWeather `atmosphere`. SpeedyWeather
computes the radiation. The ocean and sea ice absorb its downwelling and hand back their albedo and
upwelling longwave, weighted by the sea ice concentration.

SpeedyWeather emits from its `sea_surface_temperature`, so that variable holds the temperature at
which SpeedyWeather's ocean emissivity gives the upwelling longwave, not the temperature of the ocean.

Keyword arguments
=================

- `ocean_surface`: `SurfaceRadiationProperties` of the ocean. Default: albedo 0.05, `default_water_emissivity`.
- `sea_ice_surface`: `SurfaceRadiationProperties` of the sea ice. Default: albedo 0.7, emissivity 1.
"""
function EarthSystemModels.atmosphere_radiation(atmosphere::SpeedySimulation;
                                                ocean_surface = SurfaceRadiationProperties(0.05, default_water_emissivity),
                                                sea_ice_surface = SurfaceRadiationProperties(0.7, 1.0))

    surface_properties = (ocean = ocean_surface, sea_ice = sea_ice_surface)
    σ = atmosphere.model.atmosphere.stefan_boltzmann
    return SpeedyRadiation(atmosphere, surface_properties, σ, nothing, nothing, nothing)
end

function EarthSystemModels.materialize_earth_system_radiation!(atmosphere::SpeedySimulation, radiation)
    radiation isa SpeedyRadiation && atmosphere.model.albedo isa SpeedyWeather.OceanLandAlbedo{<:SpeedyWeather.PrescribedAlbedo} ||
        throw(ArgumentError("a coupled SpeedyWeather model needs `atmosphere_radiation(atmosphere)` and a `PrescribedAlbedo` over the ocean"))
    return atmosphere
end

Base.summary(::SpeedyRadiation) = "SpeedyRadiation"

Oceananigans.Grids.grid(radiation::SpeedyRadiation) = Oceananigans.Grids.grid(radiation.atmosphere)
Oceananigans.TimeSteppers.time_step!(::SpeedyRadiation, Δt) = nothing

function EarthSystemModels.surface_state(radiation::SpeedyRadiation)
    surface_values = radiation.atmosphere.variables.parameterizations.rain_rate.data
    return (; ℐꜜˢʷ = zero(surface_values), ℐꜜˡʷ = zero(surface_values))
end

function EarthSystemModels.update_surface_state!(state, radiation::SpeedyRadiation)
    parameterizations = radiation.atmosphere.variables.parameterizations
    state.ℐꜜˢʷ .= .- parameterizations.surface_shortwave_down.data
    state.ℐꜜˡʷ .= .- parameterizations.surface_longwave_down.data
    return nothing
end

@inline InterfaceComputations.kernel_radiation_properties(radiation::SpeedyRadiation, exchanger=nothing) =
    (σ = radiation.stefan_boltzmann_constant, surface_properties = radiation.surface_properties)

function EarthSystemModels.allocate_interface_fluxes!(radiation::SpeedyRadiation, exchange_grid, surfaces)
    radiation.interface_fluxes = (ocean = InterfaceRadiationFlux(exchange_grid),
                                  sea_ice = InterfaceRadiationFlux(exchange_grid))
    radiation.albedo = Field{Center, Center, Nothing}(exchange_grid)
    radiation.upwelling_longwave = Field{Center, Center, Nothing}(exchange_grid)
    return nothing
end

function EarthSystemModels.update_net_fluxes!(coupled_model, radiation::SpeedyRadiation)
    exchanger = coupled_model.interfaces.exchanger
    grid = exchanger.grid
    fluxes = radiation.interface_fluxes

    launch!(architecture(grid), grid, :xy, _blend_surface_radiation!,
            radiation.albedo,
            radiation.upwelling_longwave,
            grid,
            Time(coupled_model.clock.time),
            InterfaceComputations.kernel_radiation_properties(radiation),
            exchanger.radiation.state,
            sea_ice_concentration(coupled_model.sea_ice),
            fluxes.ocean.upwelling_longwave,
            fluxes.sea_ice.upwelling_longwave)

    atmosphere = radiation.atmosphere
    regridder = exchanger.atmosphere.regridder
    regrid_over_water!(atmosphere.variables.parameterizations.ocean.albedo.data, regridder, radiation.albedo)

    # Regridded before the inversion, so that the upwelling longwave is conserved and not the temperature
    T = atmosphere.variables.prognostic.ocean.sea_surface_temperature.data
    regrid_over_water!(T, regridder, radiation.upwelling_longwave)
    ϵ = atmosphere.model.radiation.longwave.radiative_transfer.emissivity_ocean
    T .= sqrt.(sqrt.(T ./ (ϵ * radiation.stefan_boltzmann_constant)))

    return nothing
end

@kernel function _blend_surface_radiation!(α, ℐꜛˡʷ, grid, time, rk, radiation_state, ℵ, ocean_emission, sea_ice_emission)
    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)

    ocean = air_sea_interface_radiation_state(rk, radiation_state, i, j, kᴺ, grid, time)
    sea_ice = air_sea_ice_interface_radiation_state(rk, radiation_state, i, j, kᴺ, grid, time)

    @inbounds begin
        ℵᵢ = ℵ[i, j, 1]
        α[i, j, 1] = (1 - ℵᵢ) * ocean.α + ℵᵢ * sea_ice.α
        ℐꜛˡʷ[i, j, 1] = (1 - ℵᵢ) * (ocean_emission[i, j, 1] - (1 - ocean.ϵ) * ocean.ℐꜜˡʷ) +
                             ℵᵢ  * (sea_ice_emission[i, j, 1] - (1 - sea_ice.ϵ) * sea_ice.ℐꜜˡʷ)
    end
end
