module MODISLand

export MCD15A2H, MCD12Q1, MODISLAIClimatology, build_lai_climatology!,
       retained_retrieval_metadatum,
       lai_screening_mask, recommended_lai_screening,
       mask_lai_landcover, modis_landcover_class_names,
       landcover_class_names, igbp_class_names, igbp_non_vegetated_classes,
       class_maximum_gap, zero_non_vegetated!,
       period_index, composite_window

using Dates: Dates, DateTime, Day, dayofyear
using Downloads: Downloads
using NCDatasets: NCDataset, defDim, defVar
using Oceananigans: Center
using Oceananigans.Architectures: child_architecture
using Oceananigans.DistributedComputations: @root
using Oceananigans.Fields: Field, interior, regrid!
using Statistics: mean

using ..DataWrangling: DataWrangling, Metadata, Metadatum, BoundingBox,
                       metadata_path, default_download_directory,
                       native_cell_range, native_convention_longitude,
                       cmr_granules, write_atomically, class_fractions, majority_class!

import Oceananigans

download_MODISLand_cache::String = ""
function __init__()
    global download_MODISLand_cache = DataWrangling.download_cache("MODISLand")
    return nothing
end

include("digital_numbers.jl")  # raw bytes → values: fill rejection, scale factors, quality bits
include("landcover.jl")        # class legends and the class-keyed gap-fill helpers
include("datasets.jl")         # the three dataset types and their DataWrangling traits
include("composite_dates.jl")  # the year-anchored composite calendar
include("granules.jl")         # CMR discovery, download, warp target, and reading the local files
include("climatology.jl")      # multi-year per-period compositing

end # module MODISLand
