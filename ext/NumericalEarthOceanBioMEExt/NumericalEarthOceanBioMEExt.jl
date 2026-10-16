module NumericalEarthOceanBioMEExt

using Adapt
using Oceananigans
using OceanBioME

using KernelAbstractions: KernelAbstractions, @index, @kernel
using Oceananigans.Architectures: architecture
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Utils: launch!

using OceanBioME: GasExchange, NutrientsPlanktonDetritus, CompleteBiogeochemistry, DiscreteBiogeochemistry
using OceanBioME.Models.NutrientsPlanktonDetritusModels.OxygenModels: Oxygen
using OceanBioME.Models.NutrientsPlanktonDetritusModels.InorganicCarbonModels: AbstractInorganicCarbon
using OceanBioME.Light: 
    AbstractPhotosyntheticallyActiveRadiation, 
    surface_PAR,
    PARFromShortwave

import Adapt: adapt_structure

include("radiation.jl")
include("gas_exchange.jl")

end # module NumericalEarthOceanBioMEExt
