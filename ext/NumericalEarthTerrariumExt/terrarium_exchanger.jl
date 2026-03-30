#####
##### Atmosphere <-> Terrarium-land state exchange.
#####
##### Land -> exchange grid (`interpolate_state!`): publish the land surface
##### temperature (uppermost soil layer) and top-layer saturation so
##### NumericalEarth's `compute_atmosphere_land_fluxes!` can evaluate the
##### Monin-Obukhov turbulent fluxes.
#####
##### Exchange grid -> land (`update_net_fluxes!`): push the computed skin
##### temperature and turbulent fluxes, together with the near-surface
##### atmospheric forcing and precipitation, into the Terrarium input fields.
#####
##### No regridding: the exchange grid is the Terrarium field grid, so lateral
##### points map one-to-one onto columns (direct copy).
#####

using Oceananigans: Field, Center, architecture
using Oceananigans.Fields: ZeroField
using Oceananigans.Utils: launch!

import NumericalEarth.EarthSystemModels: interpolate_state!, update_net_fluxes!
import NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger

# Land exchanger: expose skin temperature `T` (K) and surface `saturation`, the two
# fields the atmosphere-land flux kernel reads. No regridder (exchange grid == land grid).
function ComponentExchanger(land::TerrariumLand, exchange_grid)
    state = (T          = Field{Center, Center, Nothing}(exchange_grid),
             saturation = Field{Center, Center, Nothing}(exchange_grid))
    return ComponentExchanger(state, nothing)
end

#####
##### Land -> exchange grid
#####

@kernel function _terrarium_surface_to_exchange!(T, saturation, soil_T, soil_saturation, Nz, T₀)
    i, j = @index(Global, NTuple)
    @inbounds begin
        # Uppermost soil layer temperature is the land surface temperature (°C -> K).
        T[i, j, 1] = soil_T[i, j, Nz] + T₀
        saturation[i, j, 1] = soil_saturation[i, j, Nz]
    end
end

function interpolate_state!(exchanger, exchange_grid, land::TerrariumLand, coupled_model)
    state = land.integrator.state
    arch = architecture(exchange_grid)
    Nz = size(exchange_grid, 3)
    T₀ = convert(eltype(exchange_grid), 273.15)
    launch!(arch, exchange_grid, :xy,
            _terrarium_surface_to_exchange!,
            exchanger.state.T,
            exchanger.state.saturation,
            state.temperature,
            state.saturation_water_ice,
            Nz, T₀)
    return nothing
end

#####
##### Exchange grid -> land
#####

@kernel function _terrarium_push_forcing!(skin_temperature, sensible_heat_flux, latent_heat_flux,
                                          air_temperature, specific_humidity, air_pressure,
                                          windspeed, rainfall, snowfall,
                                          interface_temperature, sensible_heat, latent_heat,
                                          atmos_u, atmos_v, atmos_T, atmos_q, atmos_p, Jʳⁿ, Jˢⁿ, T₀)
    i, j = @index(Global, NTuple)
    @inbounds begin
        # Prescribed skin temperature and turbulent fluxes from the atmosphere-land interface.
        skin_temperature[i, j, 1]   = interface_temperature[i, j, 1] - T₀   # K -> °C
        sensible_heat_flux[i, j, 1] = sensible_heat[i, j, 1]
        latent_heat_flux[i, j, 1]   = latent_heat[i, j, 1]
        # Near-surface atmospheric forcing for Terrarium's hydrology / evapotranspiration.
        air_temperature[i, j, 1]    = atmos_T[i, j, 1] - T₀                 # K -> °C
        specific_humidity[i, j, 1]  = atmos_q[i, j, 1]
        air_pressure[i, j, 1]       = atmos_p[i, j, 1]
        windspeed[i, j, 1]          = sqrt(atmos_u[i, j, 1]^2 + atmos_v[i, j, 1]^2)
        rainfall[i, j, 1]           = Jʳⁿ[i, j, 1]
        snowfall[i, j, 1]           = Jˢⁿ[i, j, 1]
    end
end

# Push the downwelling shortwave/longwave radiation from the radiation exchanger into
# Terrarium's radiation input fields, so its local `DiagnosedRadiativeFluxes` sees the
# coupled downwelling. No-op when no radiation is configured (Terrarium keeps its defaults).
@kernel function _terrarium_push_downwelling!(shortwave_down, longwave_down, ℐꜜˢʷ, ℐꜜˡʷ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        shortwave_down[i, j, 1] = ℐꜜˢʷ[i, j, 1]
        longwave_down[i, j, 1] = ℐꜜˡʷ[i, j, 1]
    end
end

function update_net_fluxes!(coupled_model, land::TerrariumLand)
    al_interface = coupled_model.interfaces.atmosphere_land_interface
    isnothing(al_interface) && return nothing

    state = land.integrator.state
    grid = land.grid
    arch = architecture(grid)
    T₀ = convert(eltype(grid), 273.15)

    interface_fluxes = al_interface.fluxes
    atmos_state = coupled_model.interfaces.exchanger.atmosphere.state
    Jʳⁿ = hasproperty(atmos_state, :Jʳⁿ) ? atmos_state.Jʳⁿ : ZeroField()
    Jˢⁿ = hasproperty(atmos_state, :Jˢⁿ) ? atmos_state.Jˢⁿ : ZeroField()

    launch!(arch, grid, :xy,
            _terrarium_push_forcing!,
            state.skin_temperature,
            state.sensible_heat_flux,
            state.latent_heat_flux,
            state.air_temperature,
            state.specific_humidity,
            state.air_pressure,
            state.windspeed,
            state.rainfall,
            state.snowfall,
            al_interface.temperature,
            interface_fluxes.sensible_heat,
            interface_fluxes.latent_heat,
            atmos_state.u,
            atmos_state.v,
            atmos_state.T,
            atmos_state.q,
            atmos_state.p,
            Jʳⁿ, Jˢⁿ, T₀)

    # Downwelling radiation (when a radiation component is configured).
    radiation_exchanger = coupled_model.interfaces.exchanger.radiation
    if !isnothing(radiation_exchanger)
        radiation_state = radiation_exchanger.state
        launch!(arch, grid, :xy,
                _terrarium_push_downwelling!,
                state.surface_shortwave_down,
                state.surface_longwave_down,
                radiation_state.ℐꜜˢʷ,
                radiation_state.ℐꜜˡʷ)
    end
    return nothing
end
