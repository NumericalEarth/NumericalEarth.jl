module JRA55

export JRA55PrescribedAtmosphere,
       JRA55PrescribedLand,
       JRA55PrescribedRadiation,
       RepeatYearJRA55,
       MultiYearJRA55

using Adapt: Adapt
using CFTime: CFTime
using Dates: Dates, DateTime, Day, Hour
using Downloads: Downloads
using Oceananigans: Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.DistributedComputations: DistributedComputations, @root, Distributed, child_architecture
using Oceananigans.Grids: Center
using Oceananigans.OutputReaders: Cyclical, time_indices, FieldTimeSeries, FlavorOfFTS
using Oceananigans.Units: Units
using NCDatasets: NCDatasets, Dataset
using Scratch: Scratch, @get_scratch!

using ..DataWrangling: DataWrangling, Metadatum, first_date, last_date, set_region_data!
using ...NumericalEarth: NumericalEarth

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
