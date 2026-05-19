module JRA55

export JRA55PrescribedAtmosphere,
       JRA55PrescribedLand,
       JRA55PrescribedRadiation,
       RepeatYearJRA55,
       MultiYearJRA55

using Oceananigans
using Oceananigans.Units
using Oceananigans: location
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: child_architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: λnodes, φnodes, on_architecture
using Oceananigans.Fields: interpolate!
using Oceananigans.OutputReaders: Cyclical, TotallyInMemory, time_indices,
                                  AbstractInMemoryBackend, FlavorOfFTS

using ...NumericalEarth
using ...Atmospheres: PrescribedAtmosphere, PrescribedPrecipitationFlux
using ...Radiations: PrescribedRadiation, SurfaceRadiationProperties, default_stefan_boltzmann_constant

using NCDatasets
using JLD2
using Dates
using Scratch

using GPUArraysCore: @allowscalar

import Oceananigans.Fields: set!
import Oceananigans.OutputReaders: new_backend, update_field_time_series!

download_JRA55_cache::String = ""

function __init__()
    global download_JRA55_cache = @get_scratch!("JRA55")
end

include("JRA55_metadata.jl")
include("JRA55_field_time_series.jl")
include("JRA55_prescribed_atmosphere.jl")
include("JRA55_prescribed_land.jl")
include("JRA55_prescribed_radiation.jl")

end # module
