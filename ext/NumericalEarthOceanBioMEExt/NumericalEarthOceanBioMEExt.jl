module NumericalEarthOceanBioMEExt

using KernelAbstractions: KernelAbstractions, @index, @kernel
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: Field, set!
using Oceananigans.Grids: AbstractGrid, Center
using Oceananigans.Utils: launch!

using OceanBioME: NutrientsPlanktonDetritus, DiscreteBiogeochemistry
using OceanBioME.Models.NutrientsPlanktonDetritusModels.OxygenModels: Oxygen
using OceanBioME.Models.NutrientsPlanktonDetritusModels.InorganicCarbonModels: AbstractInorganicCarbon
using OceanBioME.Light:
    AbstractPhotosyntheticallyActiveRadiation,
    surface_PAR,
    PARFromShortwave

include("radiation.jl")
include("gas_exchange.jl")

end # module NumericalEarthOceanBioMEExt
