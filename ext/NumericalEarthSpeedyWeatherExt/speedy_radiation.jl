using NumericalEarth.Radiations: AtmosphereRadiation

const SpeedyRadiation = AtmosphereRadiation{<:SpeedySimulation}

function EarthSystemModels.atmosphere_radiation(atmosphere::SpeedySimulation; kw...)
    surface_values = atmosphere.variables.parameterizations.rain_rate.data
    downwelling = (; ℐꜜˢʷ = zero(surface_values), ℐꜜˡʷ = zero(surface_values))
    σ = atmosphere.model.atmosphere.stefan_boltzmann
    return AtmosphereRadiation(atmosphere, downwelling; stefan_boltzmann_constant = σ, kw...)
end

function EarthSystemModels.materialize_earth_system_radiation!(atmosphere::SpeedySimulation, radiation)
    radiation isa SpeedyRadiation && atmosphere.model.albedo isa SpeedyWeather.OceanLandAlbedo{<:SpeedyWeather.PrescribedAlbedo} ||
        throw(ArgumentError("a coupled SpeedyWeather model needs `atmosphere_radiation(atmosphere)` and a `PrescribedAlbedo` over the ocean"))
    return atmosphere
end

function EarthSystemModels.update_surface_state!(state, radiation::SpeedyRadiation)
    parameterizations = radiation.atmosphere.variables.parameterizations
    state.ℐꜜˢʷ .= .- parameterizations.surface_shortwave_down.data
    state.ℐꜜˡʷ .= .- parameterizations.surface_longwave_down.data
    return nothing
end

# SpeedyWeather emits from its `sea_surface_temperature`, the temperature at which its ocean emissivity gives the upwelling longwave
function EarthSystemModels.update_net_fluxes!(coupled_model, radiation::SpeedyRadiation)
    invoke(EarthSystemModels.update_net_fluxes!, Tuple{Any, AtmosphereRadiation}, coupled_model, radiation)

    atmosphere = radiation.atmosphere
    regridder = coupled_model.interfaces.exchanger.atmosphere.regridder
    regrid_over_water!(atmosphere.variables.parameterizations.ocean.albedo.data, regridder, radiation.albedo)

    # Regridded before the inversion, so that the upwelling longwave is conserved and not the temperature
    T = atmosphere.variables.prognostic.ocean.sea_surface_temperature.data
    regrid_over_water!(T, regridder, radiation.upwelling_longwave)
    ϵ = atmosphere.model.radiation.longwave.radiative_transfer.emissivity_ocean
    T .= sqrt.(sqrt.(T ./ (ϵ * radiation.stefan_boltzmann_constant)))

    return nothing
end
