module NumericalEarthOceanBioMEExt

using Adapt
using Oceananigans
using OceanBioME

using KernelAbstractions: KernelAbstractions, @index, @kernel
using Oceananigans.Architectures: architecture
using Oceananigans.Utils: launch!
using OceanBioME: CompleteBiogeochemistry

using OceanBioME.Light: 
    AbstractPhotosyntheticallyActiveRadiation, 
    surface_PAR,
    PARFromShortwave

import Adapt: adapt_structure

include("state_exchange_interface.jl")
include("assemble_net_fluxes.jl")

end # module NumericalEarthOceanBioMEExt
