#####
##### Land-cover legends
#####

"""
    igbp_class_names

The 17 International Geosphere-Biosphere Programme classes of `LC_Type1`, the default
legend of [`MCD12Q1`](@ref).

```jldoctest
julia> using NumericalEarth.DataWrangling.MODISLand

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

The IGBP classes selected by the default zero-LAI and water mask: urban, permanent snow
and ice, barren, and water. Urban and barren classes can contain vegetation.
"""
const igbp_non_vegetated_classes = (igbp_class_names.urban,
                                    igbp_class_names.permanent_snow_and_ice,
                                    igbp_class_names.barren,
                                    igbp_class_names.water)

const MODIS_LANDCOVER_LEGENDS = (IGBP = (layer = "LC_Type1", names = igbp_class_names),
                                 LAI  = (layer = "LC_Type3", names = modis_lai_class_names),
                                 PFT  = (layer = "LC_Type5", names = modis_plant_functional_type_names))

function validate_legend(legend::Symbol)
    haskey(MODIS_LANDCOVER_LEGENDS, legend) ||
        throw(ArgumentError("$legend is not an MCD12Q1 legend; valid legends are " *
                            "$(keys(MODIS_LANDCOVER_LEGENDS))."))
    return legend
end

"""
    $(TYPEDSIGNATURES)

Return the MCD12Q1 class `code` as a `Float32`, or `NaN` when it falls outside the legend's
`valid` range — the fill value 255, which the product also uses for "unclassified". The
codes are classes rather than measurements, so nothing is scaled.
"""
@inline mask_landcover_fill(code, valid) =
    ifelse((code ≥ first(valid)) & (code ≤ last(valid)), Float32(code), NaN32)

#####
##### What the class field contributes to the gap-fill chain
#####

# Consecutive 8-day periods a linear bridge may span, by IGBP class: set by how fast the
# class's leaf area moves, and zero for the non-vegetated classes.
#
#  1 evergreen needleleaf  6 | 7  open shrubland         3 | 13 urban                    0
#  2 evergreen broadleaf   6 | 8  woody savanna          3 | 14 cropland/natural mosaic  1
#  3 deciduous needleleaf  1 | 9  savanna                3 | 15 permanent snow and ice   0
#  4 deciduous broadleaf   1 | 10 grassland              1 | 16 barren                   0
#  5 mixed forest          3 | 11 permanent wetland      2 | 17 water                    0
#  6 closed shrubland      3 | 12 cropland               1 |
const igbp_maximum_gap_periods = (6, 6, 1, 1, 3, 3, 3, 3, 3, 1, 2, 1, 0, 1, 0, 0, 0)

"""
    $(TYPEDSIGNATURES)

The per-cell `max_gap` for the temporal fill, as an array matching `land_cover`, a `Field`
or array of IGBP class codes: `periods` maps each code to the number of consecutive missing
periods a linear bridge may span, and `default` covers unclassified cells.

```jldoctest
julia> using NumericalEarth.DataWrangling.MODISLand

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
    $(SIGNATURES)

Set the array `series` to `NaN` in cells whose land-cover class is in `water_classes`, and
to zero in other cells whose class is in `classes`, overwriting existing values. Returns `series`.

This imposes a zero-LAI modeling assumption on the selected classes. The default urban
and barren classes can include vegetation; their class labels do not measure zero LAI.
Other cells retain their values, including missing retrievals.
"""
function zero_non_vegetated!(series::AbstractArray, land_cover;
                             classes = igbp_non_vegetated_classes,
                             water_classes = (igbp_class_names.water,))

    data = DataWrangling.seasonal_array(series)
    codes = DataWrangling.horizontal_array(land_cover)

    size(codes) == size(data)[1:2] ||
        throw(ArgumentError("The land-cover array is $(size(codes)) but the series is $(size(data)[1:2]) in space."))

    for t in axes(data, 3), j in axes(data, 2), i in axes(data, 1)
        code = codes[i, j]
        isfinite(code) || continue
        class = round(Int, code)
        if class in water_classes
            data[i, j, t] = NaN
        elseif class in classes
            data[i, j, t] = 0
        end
    end

    return series
end

"""
    $(TYPEDSIGNATURES)

Apply the array method to `fts` through a host copy of its interior.
"""
function zero_non_vegetated!(fts::DataWrangling.FieldTimeSeries, land_cover; kw...)
    DataWrangling.on_host!(data -> zero_non_vegetated!(data, land_cover; kw...), fts)
    return fts
end
