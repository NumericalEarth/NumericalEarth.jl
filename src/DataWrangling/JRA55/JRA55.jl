module JRA55

export JRA55FieldTimeSeries,
       JRA55PrescribedAtmosphere,
       JRA55PrescribedLand,
       JRA55PrescribedRadiation,
       RepeatYearJRA55,
       MultiYearJRA55

using Adapt: Adapt
using CFTime: CFTime
using Dates: Dates, DateTime, Day, Hour, Second
using Downloads: Downloads
using GPUArraysCore: @allowscalar
using JLD2: JLD2
using NCDatasets: NCDatasets, Dataset
using Scratch: Scratch, @get_scratch!

using Oceananigans: Oceananigans, location
using Oceananigans.Architectures: CPU
using Oceananigans.BoundaryConditions: fill_halo_regions!, FieldBoundaryConditions
using Oceananigans.DistributedComputations: DistributedComputations, Distributed, child_architecture, @root
using Oceananigans.Fields: interior
using Oceananigans.Grids: λnodes, φnodes, Bounded, Center, Flat, LatitudeLongitudeGrid, Periodic
using Oceananigans.OutputReaders: Cyclical, TotallyInMemory, AbstractInMemoryBackend, FlavorOfFTS, time_indices, FieldTimeSeries, InMemory, OnDisk
using Oceananigans.Units: Units

using NumericalEarth: NumericalEarth
using NumericalEarth.DataWrangling: DataWrangling, Metadata, Metadatum, first_date, last_date,
                                    metadata_path, download_progress, download_dataset,
                                    all_dates, native_times, compute_native_date_range
using NumericalEarth.Atmospheres: PrescribedAtmosphere, PrescribedPrecipitationFlux
using NumericalEarth.Radiations: PrescribedRadiation, SurfaceRadiationProperties, default_stefan_boltzmann_constant
using NumericalEarth.Lands: PrescribedLand

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
