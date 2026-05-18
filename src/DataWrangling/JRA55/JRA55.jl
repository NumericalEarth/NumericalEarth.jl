module JRA55

export JRA55FieldTimeSeries,
       JRA55PrescribedAtmosphere,
       JRA55PrescribedLand,
       JRA55PrescribedRadiation,
       RepeatYearJRA55,
       MultiYearJRA55

using CFTime: CFTime
using Dates: Dates, DateTime, Day, Hour, Second
using Downloads: Downloads
using GPUArraysCore: @allowscalar
using Oceananigans: Oceananigans, location
using Oceananigans.Architectures: CPU
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, fill_halo_regions!
using Oceananigans.DistributedComputations: DistributedComputations, @root, Distributed, child_architecture
using Oceananigans.Fields: interior, set!
using Oceananigans.Grids: Flat, Bounded, Periodic, Center, LatitudeLongitudeGrid, λnodes, φnodes
using Oceananigans.OutputReaders: Cyclical, TotallyInMemory, time_indices, FieldTimeSeries,
                                  AbstractInMemoryBackend, FlavorOfFTS, OnDisk, InMemory
using Oceananigans.Units: Units
using NCDatasets: NCDatasets, Dataset
using Scratch: Scratch, @get_scratch!

using ..DataWrangling: DataWrangling, Metadatum, first_date, last_date
using ...NumericalEarth: NumericalEarth
using ...Atmospheres: PrescribedAtmosphere, PrescribedPrecipitationFlux
using ...Radiations: PrescribedRadiation, SurfaceRadiationProperties, default_stefan_boltzmann_constant


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
