using Oceananigans.Biogeochemistry: required_biogeochemical_tracers
using Oceananigans.Grids: AbstractGrid
using OceanBioME.Models.NutrientsPlanktonDetritusModels.InorganicCarbonModels: CarbonateSystem

import OceanBioME.Light: PARFromShortwave

function PARFromShortwave(grid::AbstractGrid;
                          photosynthetic_fraction_of_shortwave = 0.43)

    surface_PAR = Field{Center, Center, Nothing}(grid)

    return PARFromShortwave(surface_PAR; photosynthetic_fraction_of_shortwave)
end

exchanged_tracers(biogeochemistry) = 
    (required_biogeochemical_tracers(biogeochemistry.oxygen)...,
     dissolved_inorganic_carbon_names(biogeochemistry.inorganic_carbon))

dissolved_inorganic_carbon_names(::CarbonateSystem{1}) = (:DIC, )
dissolved_inorganic_carbon_names(::CarbonateSystem{N}) where N = map(n->Symbol(:DIC, n), 1:N)
