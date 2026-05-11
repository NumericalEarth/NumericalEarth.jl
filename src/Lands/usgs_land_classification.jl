#####
##### Vegetation lookup tables for `RucSlabLand`.
#####
##### `LandClassification{FT}` is a per-vegetation-type parameter set; it is
##### the slab-equivalent of an entry in WRF's `VEGPARM.TBL` (RUC LSM). A
##### `Vector{LandClassification}` plays the role of the table itself.
#####
##### `usgs_land_classifications(FT)` returns the 24-class USGS registry
##### with values consistent with the RUC entries of WRF's VEGPARM.TBL
##### (Smirnova et al. 1997, 2016).
#####
##### `apply_land_classifications!(land, vegtype, registry)` populates the
##### per-cell vegetation Fields on a `SlabLand` whose surface closure is
##### `RucSurfaceProperties`, from an integer-valued `vegtype` map
##### (Nx × Ny) and a category registry.
#####
##### Reference table (peak summer LAI / shdfac, RUC defaults):
#####
##### | id | name                          | shdfac | LAI | z₀ (m) | α  | ε     | r_smin |
##### |----|-------------------------------|--------|-----|--------|----|-------|--------|
##### | 1  | Urban                         | 0.10   | 1.00 | 0.50   |0.18|0.88 | 200 |
##### | 2  | Dryland Cropland/Pasture      | 0.80   | 5.68 | 0.20   |0.17|0.92 | 40  |
##### | 3  | Irrigated Cropland/Pasture    | 0.80   | 5.68 | 0.20   |0.18|0.92 | 40  |
##### | 4  | Mixed Cropland/Pasture        | 0.80   | 4.50 | 0.20   |0.18|0.92 | 40  |
##### | 5  | Cropland/Grassland Mosaic     | 0.80   | 4.29 | 0.25   |0.18|0.92 | 40  |
##### | 6  | Cropland/Woodland Mosaic      | 0.80   | 4.00 | 0.30   |0.16|0.93 | 70  |
##### | 7  | Grassland                     | 0.80   | 2.90 | 0.075  |0.19|0.92 | 40  |
##### | 8  | Shrubland                     | 0.70   | 3.66 | 0.10   |0.22|0.88 | 300 |
##### | 9  | Mixed Shrubland/Grassland     | 0.70   | 2.60 | 0.11   |0.20|0.90 | 170 |
##### | 10 | Savanna                       | 0.50   | 3.66 | 0.15   |0.20|0.92 | 70  |
##### | 11 | Deciduous Broadleaf Forest    | 0.80   | 3.31 | 0.80   |0.16|0.93 | 100 |
##### | 12 | Deciduous Needleleaf Forest   | 0.70   | 5.16 | 0.80   |0.14|0.94 | 150 |
##### | 13 | Evergreen Broadleaf Forest    | 0.95   | 6.48 | 0.80   |0.12|0.95 | 150 |
##### | 14 | Evergreen Needleleaf Forest   | 0.70   | 6.40 | 0.80   |0.12|0.95 | 125 |
##### | 15 | Mixed Forest                  | 0.80   | 5.50 | 0.80   |0.13|0.94 | 125 |
##### | 16 | Water                         | 0.00   | 0.01 | 0.0001 |0.08|0.98 | 100 |
##### | 17 | Herbaceous Wetland            | 0.60   | 5.65 | 0.20   |0.14|0.95 | 40  |
##### | 18 | Wooded Wetland                | 0.60   | 5.80 | 0.40   |0.14|0.95 | 100 |
##### | 19 | Barren or Sparsely Vegetated  | 0.01   | 0.75 | 0.05   |0.25|0.85 | 999 |
##### | 20 | Herbaceous Tundra             | 0.60   | 3.35 | 0.10   |0.15|0.92 | 150 |
##### | 21 | Wooded Tundra                 | 0.60   | 3.35 | 0.15   |0.15|0.93 | 150 |
##### | 22 | Mixed Tundra                  | 0.60   | 3.35 | 0.10   |0.15|0.92 | 150 |
##### | 23 | Bare Ground Tundra            | 0.30   | 3.35 | 0.065  |0.25|0.85 | 200 |
##### | 24 | Snow or Ice                   | 0.00   | 0.01 | 0.011  |0.55|0.98 | 999 |

"""
    LandClassification{FT}

A single entry in a vegetation lookup table. Mirrors the per-type
columns of WRF's `VEGPARM.TBL` (RUC LSM):

- `id`             : 1-based integer index in the table.
- `name`           : Symbol (e.g., `:grassland`).
- `vegfrac`        : peak green-vegetation fraction `shdfac` ∈ [0, 1].
- `lai`            : peak leaf area index [m² m⁻²].
- `z0`             : roughness length [m].
- `albedo`         : snow-free shortwave albedo.
- `emissivity`     : snow-free longwave emissivity.
- `r_smin`         : minimum stomatal resistance [s m⁻¹].
- `root_depth`     : nominal root-zone depth [m] (informational).

Use `usgs_land_classifications(FT)` for the default 24-class registry.
"""
struct LandClassification{FT}
    id :: Int
    name :: Symbol
    vegfrac :: FT
    lai :: FT
    z0 :: FT
    albedo :: FT
    emissivity :: FT
    r_smin :: FT
    root_depth :: FT
end

"""
    usgs_land_classifications(FT::Type = Float64) -> Vector{LandClassification{FT}}

Default USGS 24-class vegetation registry, with values consistent with
the RUC entries of WRF's `VEGPARM.TBL` (Smirnova et al. 1997, 2016).
The categories are returned in id order so `registry[id]` looks up the
entry directly.
"""
function usgs_land_classifications(FT::Type = Float64)
    return LandClassification{FT}[
        LandClassification{FT}(1,  :urban,                       0.10, 1.00, 0.50,   0.18, 0.88, 200.0, 0.1),
        LandClassification{FT}(2,  :dryland_cropland_pasture,    0.80, 5.68, 0.20,   0.17, 0.92,  40.0, 1.0),
        LandClassification{FT}(3,  :irrigated_cropland_pasture,  0.80, 5.68, 0.20,   0.18, 0.92,  40.0, 1.0),
        LandClassification{FT}(4,  :mixed_cropland_pasture,      0.80, 4.50, 0.20,   0.18, 0.92,  40.0, 1.0),
        LandClassification{FT}(5,  :cropland_grassland_mosaic,   0.80, 4.29, 0.25,   0.18, 0.92,  40.0, 1.0),
        LandClassification{FT}(6,  :cropland_woodland_mosaic,    0.80, 4.00, 0.30,   0.16, 0.93,  70.0, 1.0),
        LandClassification{FT}(7,  :grassland,                   0.80, 2.90, 0.075,  0.19, 0.92,  40.0, 1.0),
        LandClassification{FT}(8,  :shrubland,                   0.70, 3.66, 0.10,   0.22, 0.88, 300.0, 1.0),
        LandClassification{FT}(9,  :mixed_shrubland_grassland,   0.70, 2.60, 0.11,   0.20, 0.90, 170.0, 1.0),
        LandClassification{FT}(10, :savanna,                     0.50, 3.66, 0.15,   0.20, 0.92,  70.0, 1.0),
        LandClassification{FT}(11, :deciduous_broadleaf_forest,  0.80, 3.31, 0.80,   0.16, 0.93, 100.0, 2.0),
        LandClassification{FT}(12, :deciduous_needleleaf_forest, 0.70, 5.16, 0.80,   0.14, 0.94, 150.0, 2.0),
        LandClassification{FT}(13, :evergreen_broadleaf_forest,  0.95, 6.48, 0.80,   0.12, 0.95, 150.0, 2.0),
        LandClassification{FT}(14, :evergreen_needleleaf_forest, 0.70, 6.40, 0.80,   0.12, 0.95, 125.0, 2.0),
        LandClassification{FT}(15, :mixed_forest,                0.80, 5.50, 0.80,   0.13, 0.94, 125.0, 2.0),
        LandClassification{FT}(16, :water,                       0.00, 0.01, 0.0001, 0.08, 0.98, 100.0, 0.0),
        LandClassification{FT}(17, :herbaceous_wetland,          0.60, 5.65, 0.20,   0.14, 0.95,  40.0, 0.5),
        LandClassification{FT}(18, :wooded_wetland,              0.60, 5.80, 0.40,   0.14, 0.95, 100.0, 0.5),
        LandClassification{FT}(19, :barren_sparsely_vegetated,   0.01, 0.75, 0.05,   0.25, 0.85, 999.0, 0.1),
        LandClassification{FT}(20, :herbaceous_tundra,           0.60, 3.35, 0.10,   0.15, 0.92, 150.0, 0.5),
        LandClassification{FT}(21, :wooded_tundra,               0.60, 3.35, 0.15,   0.15, 0.93, 150.0, 0.5),
        LandClassification{FT}(22, :mixed_tundra,                0.60, 3.35, 0.10,   0.15, 0.92, 150.0, 0.5),
        LandClassification{FT}(23, :bare_ground_tundra,          0.30, 3.35, 0.065,  0.25, 0.85, 200.0, 0.1),
        LandClassification{FT}(24, :snow_or_ice,                 0.00, 0.01, 0.011,  0.55, 0.98, 999.0, 0.0),
    ]
end

"""
    apply_land_classifications!(land::SlabLand, vegtype, registry)

Populate the per-cell vegetation Fields on the surface-property
closure (`vegfrac`, `lai`, `albedo_vegetation`, `emissivity_vegetation`,
`roughness_length_vegetation`, `stomatal_resistance_min`, `is_urban`) from a 2D
`vegtype` map of integer category ids
and a `registry` returned by e.g. `usgs_land_classifications(FT)`.

`vegtype` may be any 2D `AbstractArray{<:Real}` (typically Int) of size
`Nx × Ny`. Cells with out-of-range ids are left untouched. The lookup
runs on the host; if `land` lives on a GPU, the populated host arrays
are uploaded with `set!`. Call once at simulation setup.

Requires `land.surface :: RucSurfaceProperties`. A future
`PrescribedSurfaceProperties` or learned variant would provide its own
specialisation of this function.
"""
function apply_land_classifications!(land::SlabLand,
                                     vegtype::AbstractArray,
                                     registry::AbstractVector{<:LandClassification})
    return apply_land_classifications!(land.surface, eltype(land), vegtype, registry)
end

function apply_land_classifications!(s::RucSurfaceProperties,
                                     FT::Type,
                                     vegtype::AbstractArray,
                                     registry::AbstractVector{<:LandClassification})

    field_xy(field) = dropdims(Array(interior(field)); dims = 3)

    Nx, Ny = size(vegtype)

    vegfrac                     = field_xy(s.vegfrac)
    lai                         = field_xy(s.lai)
    albedo_vegetation           = field_xy(s.albedo_vegetation)
    emissivity_vegetation       = field_xy(s.emissivity_vegetation)
    roughness_length_vegetation = field_xy(s.roughness_length_vegetation)
    stomatal_resistance_min     = field_xy(s.stomatal_resistance_min)
    is_urban                    = field_xy(s.is_urban)

    size(vegfrac) == (Nx, Ny) ||
        throw(DimensionMismatch("vegtype has size $(size(vegtype)); expected $(size(vegfrac))"))

    for j in 1:Ny, i in 1:Nx
        idx = Int(vegtype[i, j])
        if 1 ≤ idx ≤ length(registry)
            c = registry[idx]
            vegfrac[i, j]                     = c.vegfrac
            lai[i, j]                         = c.lai
            albedo_vegetation[i, j]           = c.albedo
            emissivity_vegetation[i, j]       = c.emissivity
            roughness_length_vegetation[i, j] = c.z0
            stomatal_resistance_min[i, j]     = c.r_smin
            is_urban[i, j]                    = c.name === :urban ? one(FT) : zero(FT)
        end
    end

    set!(s.vegfrac,                     vegfrac)
    set!(s.lai,                         lai)
    set!(s.albedo_vegetation,           albedo_vegetation)
    set!(s.emissivity_vegetation,       emissivity_vegetation)
    set!(s.roughness_length_vegetation, roughness_length_vegetation)
    set!(s.stomatal_resistance_min,     stomatal_resistance_min)
    set!(s.is_urban,                    is_urban)

    return nothing
end
