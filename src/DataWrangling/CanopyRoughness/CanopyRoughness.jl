module CanopyRoughness

export DragPartitionParameters, canopy_roughness, canopy_wind_ratio,
       zero_plane_displacement, canopy_roughness_length,
       semiempirical_roughness, semiempirical_displacement,
       compute_canopy_roughness!, canopy_roughness_climatology,
       fill_temporal_gaps!

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF, TYPEDFIELDS
using Oceananigans: Oceananigans

include("closure.jl")
include("canopy_classes.jl")
include("canopy_roughness_field.jl")
include("gapfill.jl")

end # module CanopyRoughness
