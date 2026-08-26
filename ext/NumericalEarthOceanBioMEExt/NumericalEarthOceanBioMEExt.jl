module NumericalEarthOceanBioMEExt

using Adapt
using Oceananigans
using OceanBioME

using KernelAbstractions: KernelAbstractions, @index, @kernel
using Oceananigans.Architectures: architecture
using Oceananigans.Utils: launch!

using OceanBioME: GasExchange, NutrientsPlanktonDetritus, CompleteBiogeochemistry, DiscreteBiogeochemistry
using OceanBioME.Models.NutrientsPlanktonDetritusModels.OxygenModels: Oxygen
using OceanBioME.Models.NutrientsPlanktonDetritusModels.InorganicCarbonModels: AbstractInorganicCarbon
using OceanBioME.Light: 
    AbstractPhotosyntheticallyActiveRadiation, 
    surface_PAR,
    PARFromShortwave

import Adapt: adapt_structure

import NumericalEarth.EarthSystemModels.InterfaceComputations: biogeochemical_interface
import NumericalEarth.Radiations: apply_air_sea_biogeochemical_radiative_fluxes!
import NumericalEarth.Oceans: update_net_ocean_biogeochemical_fluxes!, biogeochemistry_surface_exchanged_tracers

include("radiation.jl")
include("gas_exchange.jl")

end # module NumericalEarthOceanBioMEExt
