#####
##### IGBP land-cover class taxonomy for the drag-partition roughness closure. Classes follow
##### MODIS MCD12Q1 (`LC_Type1`) but are named here rather than coded. Each vegetated class maps
##### to one of the five Borak et al. (2025) drag-partition groups (`drag_partition_group`,
##### Table 2), carries a representative canopy height (`representative_canopy_height`, Table 4);
##### the four non-vegetated classes carry prescribed roughness constants (`nonvegetated_roughness`,
##### Table 3). The five group parameter sets live in `canopy_roughness_closure.jl`.
#####

"""
$(TYPEDSIGNATURES)

Borak et al. (2025) drag-partition group for an IGBP vegetation class — one of `:boreal`,
`:broadleaf`, `:grassland`, `:cropland`, `:shrubland`. Woody savanna is split by name into
`:woody_savanna` (broadleaf) and `:boreal_woody_savanna` (boreal). Non-vegetated classes have
no group; use [`nonvegetated_roughness`](@ref) for those.
"""
drag_partition_group(class::Symbol) = drag_partition_group(Val(class))
drag_partition_group(::Val{:evergreen_needleleaf_forest}) = :boreal
drag_partition_group(::Val{:evergreen_broadleaf_forest})  = :broadleaf
drag_partition_group(::Val{:deciduous_needleleaf_forest}) = :boreal
drag_partition_group(::Val{:deciduous_broadleaf_forest})  = :broadleaf
drag_partition_group(::Val{:mixed_forest})                = :boreal
drag_partition_group(::Val{:closed_shrubland})            = :shrubland
drag_partition_group(::Val{:open_shrubland})              = :shrubland
drag_partition_group(::Val{:woody_savanna})               = :broadleaf
drag_partition_group(::Val{:boreal_woody_savanna})        = :boreal
drag_partition_group(::Val{:savanna})                     = :shrubland
drag_partition_group(::Val{:grassland})                   = :grassland
drag_partition_group(::Val{:permanent_wetland})           = :boreal
drag_partition_group(::Val{:cropland})                    = :cropland
drag_partition_group(::Val{:cropland_vegetation_mosaic})  = :cropland

"""
$(TYPEDSIGNATURES)

Drag-partition parameters (Borak et al. 2025) for an IGBP vegetation `class` in `FloatType`
`FT`, resolved through the class's [`drag_partition_group`](@ref).

```jldoctest
julia> using NumericalEarth.Lands

julia> p = canopy_drag_parameters(Float64, :evergreen_broadleaf_forest);

julia> p.maximum_area_index
1.7
```
"""
canopy_drag_parameters(FT, class::Symbol) = drag_group_parameters(FT, drag_partition_group(class))

"""
$(TYPEDSIGNATURES)

Representative canopy height (m) for an IGBP vegetation class (Borak et al. 2025, Table 4) —
the fallback height where no measured canopy height is supplied to the closure.
"""
representative_canopy_height(FT, class::Symbol) = representative_canopy_height(FT, Val(class))
representative_canopy_height(FT, ::Val{:evergreen_needleleaf_forest}) = FT(16.62)
representative_canopy_height(FT, ::Val{:evergreen_broadleaf_forest})  = FT(24.72)
representative_canopy_height(FT, ::Val{:deciduous_needleleaf_forest}) = FT(14.49)
representative_canopy_height(FT, ::Val{:deciduous_broadleaf_forest})  = FT(17.43)
representative_canopy_height(FT, ::Val{:mixed_forest})                = FT(17.75)
representative_canopy_height(FT, ::Val{:closed_shrubland})            = FT(1.57)
representative_canopy_height(FT, ::Val{:open_shrubland})              = FT(1.42)
representative_canopy_height(FT, ::Val{:woody_savanna})               = FT(12.48)
representative_canopy_height(FT, ::Val{:boreal_woody_savanna})        = FT(12.48)
representative_canopy_height(FT, ::Val{:savanna})                     = FT(9.02)
representative_canopy_height(FT, ::Val{:grassland})                   = FT(1.39)
representative_canopy_height(FT, ::Val{:permanent_wetland})           = FT(6.69)
representative_canopy_height(FT, ::Val{:cropland})                    = FT(1.32)
representative_canopy_height(FT, ::Val{:cropland_vegetation_mosaic})  = FT(1.40)

"""
$(TYPEDSIGNATURES)

Prescribed momentum roughness length and zero-plane displacement `(z0, d0)` (meters) for a
non-vegetated IGBP class (Borak et al. 2025, Table 3): `:urban`, `:snow_and_ice`, `:water`,
`:barren`.
"""
nonvegetated_roughness(FT, class::Symbol) = nonvegetated_roughness(FT, Val(class))
nonvegetated_roughness(FT, ::Val{:urban})        = (FT(0.8000), FT(4.83))
nonvegetated_roughness(FT, ::Val{:snow_and_ice}) = (FT(0.0024), FT(0.012))
nonvegetated_roughness(FT, ::Val{:barren})       = (FT(0.0100), FT(0.05))
nonvegetated_roughness(FT, ::Val{:water})        = (FT(0.0010), FT(0.005))

"""
$(TYPEDSIGNATURES)

`true` if `class` is a vegetated IGBP class (has a drag-partition group and canopy height),
`false` for the non-vegetated classes `:urban`, `:snow_and_ice`, `:barren`, `:water`.
"""
is_vegetated(class::Symbol) = !(class in (:urban, :snow_and_ice, :barren, :water))
