mutable struct AtmosphereRadiation{A, D, S, FT}
    atmosphere :: A
    downwelling :: D
    surface_properties :: S
    stefan_boltzmann_constant :: FT
    # Allocated once the exchange grid is known
    interface_fluxes
    albedo
    upwelling_longwave
end

function AtmosphereRadiation(atmosphere, downwelling;
                             ocean_surface = SurfaceRadiationProperties(0.05, default_water_emissivity),
                             sea_ice_surface = SurfaceRadiationProperties(0.7, 1.0),
                             stefan_boltzmann_constant = default_stefan_boltzmann_constant)

    surface_properties = (ocean = ocean_surface, sea_ice = sea_ice_surface)
    return AtmosphereRadiation(atmosphere, downwelling, surface_properties, stefan_boltzmann_constant, nothing, nothing, nothing)
end

Base.summary(radiation::AtmosphereRadiation) = string("AtmosphereRadiation of ", summary(radiation.atmosphere))

Oceananigans.Grids.grid(radiation::AtmosphereRadiation) = Oceananigans.Grids.grid(radiation.atmosphere)
Oceananigans.TimeSteppers.time_step!(::AtmosphereRadiation, Δt) = nothing
EarthSystemModels.surface_state(radiation::AtmosphereRadiation) = radiation.downwelling

@inline EarthSystemModels.InterfaceComputations.kernel_radiation_properties(radiation::AtmosphereRadiation, exchanger=nothing) =
    (σ = radiation.stefan_boltzmann_constant, surface_properties = radiation.surface_properties)

function EarthSystemModels.allocate_interface_fluxes!(radiation::AtmosphereRadiation, exchange_grid, surfaces)
    radiation.interface_fluxes = (ocean = InterfaceRadiationFlux(exchange_grid),
                                  sea_ice = InterfaceRadiationFlux(exchange_grid))
    radiation.albedo = Field{Center, Center, Nothing}(exchange_grid)
    radiation.upwelling_longwave = Field{Center, Center, Nothing}(exchange_grid)
    return nothing
end

function EarthSystemModels.update_net_fluxes!(coupled_model, radiation::AtmosphereRadiation)
    exchanger = coupled_model.interfaces.exchanger
    grid = exchanger.grid
    fluxes = radiation.interface_fluxes

    launch!(architecture(grid), grid, :xy, _blend_surface_radiation!,
            radiation.albedo,
            radiation.upwelling_longwave,
            grid,
            Time(coupled_model.clock.time),
            kernel_radiation_properties(radiation),
            exchanger.radiation.state,
            sea_ice_concentration(coupled_model.sea_ice),
            fluxes.ocean.upwelling_longwave,
            fluxes.sea_ice.upwelling_longwave)

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
