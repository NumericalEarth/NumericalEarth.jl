module JRA55

export JRA55FieldTimeSeries,
       JRA55PrescribedAtmosphere,
       JRA55PrescribedLand,
       JRA55PrescribedRadiation,
       RepeatYearJRA55,
       MultiYearJRA55

import Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.BoundaryConditions: fill_halo_regions!, FieldBoundaryConditions
using Oceananigans.DistributedComputations: DistributedComputations, Distributed, child_architecture, @root
using Oceananigans.Fields: interior
using Oceananigans.Grids: λnodes, φnodes, Bounded, Center, Flat, LatitudeLongitudeGrid, Periodic
using Oceananigans.OutputReaders: Cyclical, TotallyInMemory, AbstractInMemoryBackend, FlavorOfFTS, time_indices, FieldTimeSeries, InMemory, OnDisk
using Oceananigans.Units: Units

import NumericalEarth

using NumericalEarth.Atmospheres: PrescribedAtmosphere, PrescribedPrecipitationFlux
using NumericalEarth.Radiations: PrescribedRadiation, SurfaceRadiationProperties, default_stefan_boltzmann_constant

using GPUArraysCore: @allowscalar

using Adapt: Adapt
using NCDatasets: NCDatasets, Dataset
using JLD2: JLD2
using Dates: Dates, DateTime, Day, Hour, Second
using CFTime: CFTime
using Scratch: Scratch, @get_scratch!

using Oceananigans: location
import Oceananigans.Fields: set!
import Oceananigans.OutputReaders: new_backend
using Downloads: Downloads
using NumericalEarth.DataWrangling: Metadatum, first_date, last_date, DataWrangling

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
