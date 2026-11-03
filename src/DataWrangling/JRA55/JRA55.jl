module JRA55

export RepeatYearJRA55,
       MultiYearJRA55

using Adapt: Adapt
using CFTime: CFTime
using Dates: Dates, DateTime, Day, Hour
using Downloads: Downloads
using Oceananigans: Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.DistributedComputations: DistributedComputations, @root
using Oceananigans.Grids: Center
using Oceananigans.OutputReaders: time_indices, FlavorOfFTS
using Oceananigans.Units: Units
using NCDatasets: NCDatasets, Dataset

using ..DataWrangling: DataWrangling, Metadatum, all_dates, set_region_data!

download_JRA55_cache::String = ""

function __init__()
    global download_JRA55_cache = DataWrangling.download_cache("JRA55")
    return nothing
end

include("JRA55_metadata.jl")
include("JRA55_field_time_series.jl")

end # module
