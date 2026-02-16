module NumericalEarthMITgcmExt

using NumericalEarth
using MITgcm
using Oceananigans

include("mitgcm_ocean_coupling.jl")
include("mitgcm_state_exchanger.jl")

end # module NumericalEarthMITgcmExt
