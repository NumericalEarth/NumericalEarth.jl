using Oceananigans.Biogeochemistry: required_biogeochemical_tracers
using Oceananigans.Grids: AbstractGrid
using OceanBioME.Models.NutrientsPlanktonDetritusModels.InorganicCarbonModels: CarbonateSystem

import OceanBioME.Light: PARFromShortwave

function PARFromShortwave(grid::AbstractGrid;
                          photosynthetic_fraction_of_shortwave = 0.43)

    surface_PAR = Field{Center, Center, Nothing}(grid)

    return PARFromShortwave(surface_PAR; photosynthetic_fraction_of_shortwave)
end

apply_air_sea_biogeochemical_radiative_fluxes!(coupled_model, biogeochemistry::DiscreteBiogeochemistry{<:NutrientsPlanktonDetritus}) = 
    apply_air_sea_biogeochemical_radiative_fluxes!(coupled_model, biogeochemistry.light_attenuation)

function apply_air_sea_biogeochemical_radiative_fluxes!(coupled_model, light::AbstractPhotosyntheticallyActiveRadiation{<:PARFromShortwave})
    penetrating_shortwave = coupled_model.radiation.interface_fluxes.ocean.downwelling_shortwave
    PAR = surface_PAR(light)

    set!(PAR.surface_shortwave, penetrating_shortwave)

    return nothing
end
