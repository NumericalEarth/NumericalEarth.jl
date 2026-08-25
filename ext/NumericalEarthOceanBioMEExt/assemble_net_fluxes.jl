#####
##### radiation from atmosphere blocked by sea ice
#####
apply_air_sea_biogeochemical_radiative_fluxes!(coupled_model, biogeochemistry::DiscreteBiogeochemistry{<:NutrientsPlanktonDetritus}) = 
    apply_air_sea_biogeochemical_radiative_fluxes!(coupled_model, biogeochemistry.light_attenuation)

function apply_air_sea_biogeochemical_radiative_fluxes!(coupled_model, light::AbstractPhotosyntheticallyActiveRadiation{<:PARFromShortwave})
    penetrating_shortwave = coupled_model.radiation.interface_fluxes.ocean.downwelling_shortwave
    PAR = surface_PAR(light)

    set!(PAR.surface_shortwave, penetrating_shortwave)

    return nothing
end
