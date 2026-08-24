using OceanBioME: GasExchange

import NumericalEarth.EarthSystemModels.InterfaceComputations: net_fluxes
import NumericalEarth.Radiations: apply_air_sea_biogeochemical_radiative_fluxes!
import NumericalEarth.Oceans: update_net_ocean_biogeochemical_fluxes!

function update_net_ocean_biogeochemical_fluxes!(coupled_model, biogeochemistry::CompleteBiogeochemistry, ocean, grid)
    update_net_ocean_biogeochemical_fluxes!(coupled_model, biogeochemistry.underlying_biogeochemistry, biogeochemistry, ocean, grid)
    update_net_ocean_biogeochemical_fluxes!(coupled_model, biogeochemistry.light_attenuation, biogeochemistry, ocean, grid)
    update_net_ocean_biogeochemical_fluxes!(coupled_model, biogeochemistry.sediment, ocean, biogeochemistry, grid)
    update_net_ocean_biogeochemical_fluxes!(coupled_model, biogeochemistry.particles, ocean, biogeochemistry, grid)
    update_net_ocean_biogeochemical_fluxes!(coupled_model, biogeochemistry.modifiers, ocean, biogeochemistry, grid)

    return nothing
end

update_net_ocean_biogeochemical_fluxes!(coupled_model, component, ocean, biogeochemistry, grid) = nothing

apply_air_sea_biogeochemical_radiative_fluxes!(coupled_model) = 
    apply_air_sea_biogeochemical_radiative_fluxes!(coupled_model, coupled_model.ocean.model.biogeochemistry)

apply_air_sea_biogeochemical_radiative_fluxes!(coupled_model, biogeochemistry) = nothing

apply_air_sea_biogeochemical_radiative_fluxes!(coupled_model, biogeochemistry::CompleteBiogeochemistry) = 
    apply_air_sea_biogeochemical_radiative_fluxes!(coupled_model, biogeochemistry.light_attenuation)

function apply_air_sea_biogeochemical_radiative_fluxes!(coupled_model, light::AbstractPhotosyntheticallyActiveRadiation{<:PARFromShortwave})
@info "here"
    penetrating_shortwave = coupled_model.radiation.interface_fluxes.ocean.downwelling_shortwave
    PAR = surface_PAR(light)

    set!(PAR.surface_shortwave, penetrating_shortwave)

    return nothing
end

function update_ocean_biogeochemical_boundary_conditions!(coupled_model, biogeochemistry, ocean, grid)
    fluxes = 0
end

@inline net_fluxes(condition::GasExchange) = condition