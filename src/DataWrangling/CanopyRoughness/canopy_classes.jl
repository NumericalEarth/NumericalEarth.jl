#####
##### IGBP land-cover → drag-partition parameters, canopy heights, and non-vegetated
##### constants. IGBP class integers follow MODIS MCD12Q1 (`LC_Type1`):
##### 1 ENF, 2 EBF, 3 DNF, 4 DBF, 5 MXF, 6 CSL, 7 OSL, 8 WSV, 9 SAV, 10 GRS, 11 PWL,
##### 12 CRP, 13 URB, 14 MOS, 15 SNO, 16 BAR, 17 Water. Parameter groups, heights and
##### non-vegetated constants are Tables 2–4 of [Borak et al. (2025)](@cite Borak2025).
#####

"absolute latitude (°) above which needleleaf/mixed/wetland/woody-savanna are treated as boreal"
const BOREAL_LATITUDE = 50

"""
$(TYPEDSIGNATURES)

Drag-partition parameters for one of the five vegetation groups (`group ∈ 1:5`), in
`FloatType` `FT`. Group order: 1 boreal (needleleaf/mixed/wetland/boreal woody savanna),
2 broadleaf (broadleaf forest/non-boreal woody savanna), 3 grassland,
4 cropland/mosaic, 5 shrubs/savanna.
"""
@inline function canopy_drag_parameters(FT, group)
    groups = (DragPartitionParameters{FT}(0.21, 0.0030, 0.27, 0.28, 1.90, 1.90),
              DragPartitionParameters{FT}(0.31, 0.0030, 0.31, 0.36, 1.15, 1.70),
              DragPartitionParameters{FT}(0.43, 0.0030, 0.32, 0.49, 1.30, 1.30),
              DragPartitionParameters{FT}(0.31, 0.0030, 0.29, 0.39, 1.55, 1.50),
              DragPartitionParameters{FT}(0.50, 0.0030, 0.38, 0.48, 1.00, 1.60))
    return @inbounds groups[group]
end

"""
$(TYPEDSIGNATURES)

Vegetation drag-partition group `1:5` for an IGBP class at a given latitude, or `0` for
non-vegetated classes (urban, snow/ice, barren, water). Woody savanna splits on latitude
(boreal vs non-boreal).
"""
@inline function drag_partition_group(igbp, latitude)
    woody_savanna = ifelse(abs(latitude) ≥ BOREAL_LATITUDE, 1, 2)
    g = 0
    g = ifelse(igbp == 1,  1, g)
    g = ifelse(igbp == 2,  2, g)
    g = ifelse(igbp == 3,  1, g)
    g = ifelse(igbp == 4,  2, g)
    g = ifelse(igbp == 5,  1, g)
    g = ifelse(igbp == 6,  5, g)
    g = ifelse(igbp == 7,  5, g)
    g = ifelse(igbp == 8,  woody_savanna, g)
    g = ifelse(igbp == 9,  5, g)
    g = ifelse(igbp == 10, 3, g)
    g = ifelse(igbp == 11, 1, g)
    g = ifelse(igbp == 12, 4, g)
    g = ifelse(igbp == 14, 4, g)
    return g
end

"""
$(TYPEDSIGNATURES)

`true` if the IGBP class is vegetated (has a drag-partition group).
"""
@inline is_vegetated(igbp) = drag_partition_group(igbp, 0) != 0

"""
$(TYPEDSIGNATURES)

Representative canopy height (m) for an IGBP class (Table 4). Returns `0` for
non-vegetated classes.
"""
@inline function class_canopy_height(FT, igbp)
    h = zero(FT)
    h = ifelse(igbp == 1,  FT(16.62), h)
    h = ifelse(igbp == 2,  FT(24.72), h)
    h = ifelse(igbp == 3,  FT(14.49), h)
    h = ifelse(igbp == 4,  FT(17.43), h)
    h = ifelse(igbp == 5,  FT(17.75), h)
    h = ifelse(igbp == 6,  FT(1.57),  h)
    h = ifelse(igbp == 7,  FT(1.42),  h)
    h = ifelse(igbp == 8,  FT(12.48), h)
    h = ifelse(igbp == 9,  FT(9.02),  h)
    h = ifelse(igbp == 10, FT(1.39),  h)
    h = ifelse(igbp == 11, FT(6.69),  h)
    h = ifelse(igbp == 12, FT(1.32),  h)
    h = ifelse(igbp == 14, FT(1.40),  h)
    return h
end

"""
$(TYPEDSIGNATURES)

Prescribed `(z0, d0)` (metres) for non-vegetated IGBP classes (Table 3): urban (13),
snow/ice (15), water (17); all other non-vegetated classes fall back to barren.
"""
@inline function nonvegetated_roughness(FT, igbp)
    z0 = FT(0.0100); d0 = FT(0.05)                                                  # barren
    z0 = ifelse(igbp == 13, FT(0.8000), z0); d0 = ifelse(igbp == 13, FT(4.83),  d0) # urban
    z0 = ifelse(igbp == 15, FT(0.0024), z0); d0 = ifelse(igbp == 15, FT(0.012), d0) # snow/ice
    z0 = ifelse(igbp == 17, FT(0.0010), z0); d0 = ifelse(igbp == 17, FT(0.005), d0) # water
    return z0, d0
end
