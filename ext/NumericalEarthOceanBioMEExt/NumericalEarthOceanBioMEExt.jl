module NumericalEarthOceanBioMEExt

using KernelAbstractions: KernelAbstractions, @index, @kernel
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: Field, set!
using Oceananigans.Grids: AbstractGrid, Center
using Oceananigans.Utils: launch!
using NumericalEarth: NumericalEarth

using OceanBioME: OceanBioME, NutrientsPlanktonDetritus, DiscreteBiogeochemistry
using OceanBioME.Light: AbstractPhotosyntheticallyActiveRadiation, surface_PAR, PARFromShortwave
using OceanBioME.Models.NutrientsPlanktonDetritusModels.OxygenModels: Oxygen
using OceanBioME.Models.NutrientsPlanktonDetritusModels.InorganicCarbonModels: AbstractInorganicCarbon

include("radiation.jl")
include("gas_exchange.jl")

end # module NumericalEarthOceanBioMEExt
