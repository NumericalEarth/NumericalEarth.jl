module CanopyRoughness

export DragPartitionParameters, canopy_roughness, canopy_wind_ratio,
       zero_plane_displacement, canopy_roughness_length,
       semiempirical_roughness, semiempirical_displacement,
       canopy_drag_parameters, drag_partition_group, is_vegetated,
       class_canopy_height, nonvegetated_roughness, snow_adjusted,
       compute_canopy_roughness!, canopy_roughness_climatology

using DocStringExtensions: TYPEDSIGNATURES, TYPEDEF, TYPEDFIELDS
using Oceananigans: Oceananigans

include("closure.jl")
include("canopy_classes.jl")
include("canopy_roughness_field.jl")

end # module CanopyRoughness
