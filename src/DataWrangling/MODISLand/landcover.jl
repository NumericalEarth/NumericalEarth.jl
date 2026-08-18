#####
##### Land-cover legends
#####

"""
    igbp_class_names

The 17 International Geosphere-Biosphere Programme classes of `LC_Type1`, the default
legend of [`MCD12Q1`](@ref).

```jldoctest
julia> using NumericalEarth

julia> igbp_class_names.deciduous_broadleaf_forest, igbp_class_names.water
(4, 17)
```
"""
const igbp_class_names = (evergreen_needleleaf_forest = 1,
                          evergreen_broadleaf_forest = 2,
                          deciduous_needleleaf_forest = 3,
                          deciduous_broadleaf_forest = 4,
                          mixed_forest = 5,
                          closed_shrubland = 6,
                          open_shrubland = 7,
                          woody_savanna = 8,
                          savanna = 9,
                          grassland = 10,
                          permanent_wetland = 11,
                          cropland = 12,
                          urban = 13,
                          cropland_natural_mosaic = 14,
                          permanent_snow_and_ice = 15,
                          barren = 16,
                          water = 17)

"""
    modis_lai_class_names

The 11 classes of `LC_Type3` — the biome stratification the MCD15 leaf-area retrieval
itself is keyed on.
"""
const modis_lai_class_names = (water = 0,
                               grassland = 1,
                               shrubland = 2,
                               broadleaf_cropland = 3,
                               savanna = 4,
                               evergreen_broadleaf_forest = 5,
                               deciduous_broadleaf_forest = 6,
                               evergreen_needleleaf_forest = 7,
                               deciduous_needleleaf_forest = 8,
                               unvegetated = 9,
                               urban = 10)

"""
    modis_plant_functional_type_names

The 12 classes of `LC_Type5`, the plant-functional-type legend a canopy conductance or
radiative-transfer closure keys its parameters on.
"""
const modis_plant_functional_type_names = (water = 0,
                                           evergreen_needleleaf_trees = 1,
                                           evergreen_broadleaf_trees = 2,
                                           deciduous_needleleaf_trees = 3,
                                           deciduous_broadleaf_trees = 4,
                                           shrub = 5,
                                           grass = 6,
                                           cereal_crop = 7,
                                           broadleaf_crop = 8,
                                           urban = 9,
                                           permanent_snow_and_ice = 10,
                                           barren = 11)

"""
    igbp_non_vegetated_classes

The IGBP classes a leaf-area gap fill must never write into. They carry no canopy, so any
value borrowed for them is an invention rather than an estimate — pass them as the
`unfilled_classes` of [`fill_seasonal_gaps!`](@ref).
"""
const igbp_non_vegetated_classes = (igbp_class_names.urban,
                                    igbp_class_names.permanent_snow_and_ice,
                                    igbp_class_names.barren,
                                    igbp_class_names.water)

const MODIS_LANDCOVER_LEGENDS = (IGBP = (layer = "LC_Type1",
                                         valid = 1:17,
                                         names = igbp_class_names),
                                 LAI  = (layer = "LC_Type3",
                                         valid = 0:10,
                                         names = modis_lai_class_names),
                                 PFT  = (layer = "LC_Type5",
                                         valid = 0:11,
                                         names = modis_plant_functional_type_names))

function validate_legend(legend::Symbol)
    haskey(MODIS_LANDCOVER_LEGENDS, legend) ||
        throw(ArgumentError("$legend is not an MCD12Q1 legend; valid legends are " *
                            "$(keys(MODIS_LANDCOVER_LEGENDS))."))
    return legend
end

"""
    mask_landcover_fill(code, valid)

Return the MCD12Q1 class `code` as a `Float32`, or `NaN` when it falls outside the legend's
`valid` range — the fill value 255, which the product also uses for "unclassified". The
codes are classes rather than measurements, so nothing is scaled.
"""
@inline mask_landcover_fill(code, valid) =
    ifelse((code ≥ first(valid)) & (code ≤ last(valid)), Float32(code), NaN32)

#####
##### What the class field contributes to the gap-fill chain
#####

# How many consecutive 8-day periods a linear bridge across a gap may span, by IGBP class,
# set by how fast the class's leaf area moves: nearly constant over evergreen canopies, a
# factor of ~5 across a three-week green-up for deciduous stands and crops, and zero for the
# non-vegetated classes, which must not be filled at all.
#
#  1 evergreen needleleaf  6 | 7  open shrubland         3 | 13 urban                    0
#  2 evergreen broadleaf   6 | 8  woody savanna          3 | 14 cropland/natural mosaic  1
#  3 deciduous needleleaf  1 | 9  savanna                3 | 15 permanent snow and ice   0
#  4 deciduous broadleaf   1 | 10 grassland              1 | 16 barren                   0
#  5 mixed forest          3 | 11 permanent wetland      2 | 17 water                    0
#  6 closed shrubland      3 | 12 cropland               1 |
const igbp_maximum_gap_periods = (6, 6, 1, 1, 3, 3, 3, 3, 3, 1, 2, 1, 0, 1, 0, 0, 0)

"""
    class_maximum_gap(land_cover; periods = igbp_maximum_gap_periods, default = 3)

The per-cell `max_gap` a class-aware temporal fill should use, as an array matching
`land_cover` — a `Field` or a plain array of IGBP class codes. Pass it straight to
[`fill_gaps!`](@ref) or [`fill_seasonal_gaps!`](@ref) in place of the scalar.

A uniform tolerance cannot be right everywhere: the same 32-day bridge that is nearly exact
over evergreen broadleaf invents a green-up ramp over cropland. `periods` is the table
mapping IGBP code to the number of 8-day periods a bridge may span, and `default` covers
cells the product left unclassified. A different legend needs its own table.

```jldoctest
julia> using NumericalEarth

julia> classes = Float32[igbp_class_names.evergreen_broadleaf_forest,
                         igbp_class_names.cropland,
                         igbp_class_names.water];

julia> class_maximum_gap(classes)
3-element Vector{Int64}:
 6
 1
 0
```
"""
function class_maximum_gap(land_cover; periods = igbp_maximum_gap_periods, default = 3)
    codes = DataWrangling.horizontal_array(land_cover)
    return map(codes) do code
        isfinite(code) || return default
        class = round(Int, code)
        return (1 ≤ class ≤ length(periods)) ? periods[class] : default
    end
end

"""
    zero_non_vegetated!(series, land_cover; classes = igbp_non_vegetated_classes)

Write a leaf area index of zero into every cell of `series` whose land-cover class is in
`classes` — water, urban, permanent snow and barren by default. Returns `series`.

These are the cells [`fill_seasonal_gaps!`](@ref) deliberately leaves missing; zero is not
a stand-in for the missing value but the value itself — leaf area per unit *ground* area
over open water is zero.

Run it **after** the fill and **before** landing the series on a model grid: a `NaN` at a
shoreline dilates into its neighbors under the regrid's stencil, while a zero blends into
the cell mean correctly. Score a fill ([`gap_fill_denial`](@ref)) before zeroing, never
after — zeroed water would enter the sample as truth the chain reproduces exactly.
"""
function zero_non_vegetated!(data::AbstractArray, land_cover;
                             classes = igbp_non_vegetated_classes)

    𝒜 = DataWrangling.seasonal_array(data)
    codes = DataWrangling.horizontal_array(land_cover)

    size(codes) == size(𝒜)[1:2] ||
        throw(ArgumentError("The land-cover array is $(size(codes)) but the series is " *
                            "$(size(𝒜)[1:2]) in space; both must be on the same lattice."))

    for t in axes(𝒜, 3), j in axes(𝒜, 2), i in axes(𝒜, 1)
        code = codes[i, j]
        isfinite(code) && round(Int, code) in classes && (𝒜[i, j, t] = 0)
    end

    return data
end

function zero_non_vegetated!(fts::DataWrangling.FieldTimeSeries, land_cover; kw...)
    DataWrangling.validate_whole_series(fts)
    data = Array(DataWrangling.interior(fts))
    zero_non_vegetated!(data, land_cover; kw...)
    copyto!(DataWrangling.interior(fts), data)
    return fts
end
