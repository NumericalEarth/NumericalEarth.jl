#####
##### Dataset types
#####

abstract type AbstractMODISLandDataset end

"""
    MODISLAIDataset

Supertype for the MCD15/MOD15 leaf-area-index and FPAR products, which share science-data-set
names, digital-number scales, and quality conventions, and differ only in cadence and sensor.
"""
abstract type MODISLAIDataset <: AbstractMODISLandDataset end

"""
    MCD15A2H(; screened_flags = recommended_lai_screening())

The MODIS MCD15A2H V061 combined Terra + Aqua leaf-area-index / FPAR product: 500 m, 8-day
composites on the sinusoidal grid, from 2002-07-04 onwards. Provides `:leaf_area_index`
(``𝒜``, one-sided green leaf area per unit *ground* area, m² m⁻²), `:fpar`,
`:leaf_area_index_uncertainty`, and `:landcover_code` — the class the product names in place of
a retrieval, listed under [`mask_lai_landcover`](@ref). The product reports *true* leaf
area index, with per-biome clumping applied inside the retrieval.

Granules are HDF-EOS2 tiles on the sinusoidal projection, discovered from NASA's Common
Metadata Repository and reprojected to a regional latitude-longitude window at download
time. Build the `Metadata` with a lon/lat [`BoundingBox`](@ref); a global read is not
supported. Requires `ArchGDAL` (for GDAL's HDF4 driver) and a NASA Earthdata login
(`EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`).

`screened_flags` is a mask of the quality criteria whose pixels are read as `NaN`; see
[`recommended_lai_screening`](@ref) and [`lai_screening_mask`](@ref). Pass `0x0000` to keep
every retrieval the product marks as valid.

Data source: MCD15A2H.061, `10.5067/MODIS/MCD15A2H.061`.

```jldoctest
julia> using NumericalEarth

julia> MCD15A2H()
MCD15A2H(screened_flags=0x0007)
```
"""
struct MCD15A2H <: MODISLAIDataset
    screened_flags :: UInt16
end

MCD15A2H(; screened_flags = recommended_lai_screening()) = MCD15A2H(screened_flags)

Base.show(io::IO, dataset::MCD15A2H) =
    print(io, "MCD15A2H(screened_flags=", repr(dataset.screened_flags), ")")

"""
    MODISLAIClimatology(; dataset = MCD15A2H(), years = 2003:2019, reducer = mean)

A seasonal climatology of a [`MODISLAIDataset`](@ref): one composite per period of the
year, each combining the retrievals `dataset`'s screen retains across `years` pixel by pixel
with `reducer`, a named function of the vector of retained values (`mean` for a seasonal
mean, `maximum` for a peak-season field). An 8-day source gives 46 periods, so
`FieldTimeSeries(Metadata(:leaf_area_index; dataset = MODISLAIClimatology(), region), grid)`
is a 46-slot cyclic seasonal series. Cells no year could observe stay `NaN`, and the
number of retained retrievals behind every cell is stored beside the reduction, see
[`retained_retrieval_metadatum`](@ref).

`years` defaults to 2003–2019, the span over which both Terra and Aqua held their
equatorial crossing times.

```jldoctest
julia> using NumericalEarth

julia> MODISLAIClimatology()
MODISLAIClimatology(MCD15A2H(screened_flags=0x0007), years=2003:2019, reducer=mean)
```
"""
struct MODISLAIClimatology{D <: MODISLAIDataset, Y, R} <: AbstractMODISLandDataset
    dataset :: D
    years :: Y
    reducer :: R
end

MODISLAIClimatology(; dataset = MCD15A2H(), years = 2003:2019, reducer = mean) =
    MODISLAIClimatology(dataset, years, reducer)

Base.show(io::IO, climatology::MODISLAIClimatology) =
    print(io, "MODISLAIClimatology(", climatology.dataset, ", years=", climatology.years,
          ", reducer=", climatology.reducer, ")")

"""
    MCD12Q1(; legend = :IGBP)

The MODIS MCD12Q1 V061 annual land-cover product: 500 m, one map per calendar year from
2001 onwards, on the same sinusoidal granules and the same reprojected 1/240° lattice as
[`MCD15A2H`](@ref) — so a class field and a leaf-area field read over the same region share
their cells one for one, with no aggregation in between.

Provides `:landcover_class` under one of three legends, `:quality_flag` (the product's
enumerated classification quality, 0 being good classified land), and `:land_water_mask`.

| `legend` | layer | classes | names |
|---|---|---|---|
| `:IGBP` | `LC_Type1` | 1–17 | [`igbp_class_names`](@ref) |
| `:LAI` | `LC_Type3` | 0–10 | `modis_lai_class_names` |
| `:PFT` | `LC_Type5` | 0–11 | `modis_plant_functional_type_names` |

On a model grid, `Field(metadatum, grid)` lands the class covering the largest area of
each cell, and [`class_fractions`](@ref) returns every class's area fraction beside it.

Granules are HDF-EOS2 tiles discovered through NASA's Common Metadata Repository, so a
lon/lat [`BoundingBox`](@ref) is required, `ArchGDAL` must be loaded, and a NASA Earthdata
login supplies the credentials — as for [`MCD15A2H`](@ref).

Data source: MCD12Q1.061, `10.5067/MODIS/MCD12Q1.061`.

```jldoctest
julia> using NumericalEarth

julia> MCD12Q1()
MCD12Q1(legend=:IGBP)
```
"""
struct MCD12Q1 <: AbstractMODISLandDataset
    legend :: Symbol
end

MCD12Q1(; legend = :IGBP) = MCD12Q1(validate_legend(legend))

Base.show(io::IO, dataset::MCD12Q1) = print(io, "MCD12Q1(legend=", repr(dataset.legend), ")")

"""
    landcover_class_names(dataset::MCD12Q1)

The `(class_name = code, …)` table of the dataset's legend.

```jldoctest
julia> using NumericalEarth.DataWrangling.MODISLand

julia> landcover_class_names(MCD12Q1()).cropland
12
```
"""
landcover_class_names(dataset::MCD12Q1) = MODIS_LANDCOVER_LEGENDS[dataset.legend].names

"""
    landcover_valid_range(dataset::MCD12Q1)

The legend's range of class codes. Codes outside it — the fill value 255, and 0 under IGBP,
which has no class 0 — are read as `NaN`.
"""
landcover_valid_range(dataset::MCD12Q1) = MODIS_LANDCOVER_LEGENDS[dataset.legend].valid

landcover_layer(dataset::MCD12Q1) = MODIS_LANDCOVER_LEGENDS[dataset.legend].layer

const MODISLandMetadata{D}  = Metadata{<:AbstractMODISLandDataset, D}
const MODISLandMetadatum    = Metadatum{<:AbstractMODISLandDataset}
const MODISLAIMetadatum     = Metadatum{<:MODISLAIDataset}
const MODISLAIClimatologyMetadata{D} = Metadata{<:MODISLAIClimatology, D}
const MODISLAIClimatologyMetadatum   = Metadatum{<:MODISLAIClimatology}
const MODISLandCoverMetadata{D}      = Metadata{<:MCD12Q1, D}
const MODISLandCoverMetadatum        = Metadatum{<:MCD12Q1}

#####
##### Product identity
#####

modis_short_name(::MCD15A2H) = "MCD15A2H"
modis_short_name(::MCD12Q1) = "MCD12Q1"
modis_short_name(climatology::MODISLAIClimatology) = modis_short_name(climatology.dataset)

modis_version(::AbstractMODISLandDataset) = "061"

composite_period_days(::MCD15A2H) = 8
composite_period_days(climatology::MODISLAIClimatology) = composite_period_days(climatology.dataset)

first_composite_date(::MCD15A2H) = DateTime(2002, 7, 4)

# Conservative end of the range `all_dates` advertises — the record is ongoing, and later
# composites can still be requested with explicit `dates`.
last_composite_date(::MCD15A2H) = DateTime(2025, 12, 31)

# The land-cover map is produced about a year and a half in arrears, so the advertised range
# lags the leaf-area record.
first_landcover_year(::MCD12Q1) = 2001
last_landcover_year(::MCD12Q1)  = 2024

source_dataset(dataset::MODISLAIDataset) = dataset
source_dataset(climatology::MODISLAIClimatology) = climatology.dataset

screened_flags(dataset::MODISLAIDataset) = dataset.screened_flags
screened_flags(climatology::MODISLAIClimatology) = screened_flags(climatology.dataset)

#####
##### Variables
#####

const MODISLAI_variable_names = Dict(:leaf_area_index             => "Lai_500m",
                                     :fpar                       => "Fpar_500m",
                                     :leaf_area_index_uncertainty => "LaiStdDev_500m")

# `:landcover_code` is the class the leaf-area layer writes where it has no retrieval; a
# code cannot be composited, so it is absent from `MODISLAI_variable_names`.
const MODISLAI_readable_variable_names =
    merge(MODISLAI_variable_names, Dict(:landcover_code => "Lai_500m"))

const lai_quality_variable       = "FparLai_QC"
const lai_extra_quality_variable = "FparExtra_QC"

const landcover_quality_variable   = "QC"
const landcover_water_mask_variable = "LW"

"""
    stored_granule_layers(dataset)

The granule layers copied into the local regional file. One warp per granule serves every
variable a read can ask for, so the set is a property of the product rather than of the
variable requested.
"""
stored_granule_layers(::MODISLAIDataset) = ("Lai_500m", "Fpar_500m", "LaiStdDev_500m",
                                            lai_quality_variable, lai_extra_quality_variable)

stored_granule_layers(dataset::MCD12Q1) = (landcover_layer(dataset),
                                           landcover_quality_variable,
                                           landcover_water_mask_variable)

const retained_count_variable = "retained_retrieval_count"

const MODISLAIClimatology_variable_names =
    merge(MODISLAI_variable_names, Dict(:retained_retrieval_count => retained_count_variable))

DataWrangling.available_variables(::MODISLAIDataset) = MODISLAI_readable_variable_names
DataWrangling.available_variables(::MODISLAIClimatology) = MODISLAIClimatology_variable_names

DataWrangling.available_variables(dataset::MCD12Q1) =
    Dict(:landcover_class  => landcover_layer(dataset),
         :quality_flag     => landcover_quality_variable,
         :land_water_mask  => landcover_water_mask_variable)

DataWrangling.dataset_variable_name(metadata::MODISLandMetadata) =
    DataWrangling.available_variables(metadata.dataset)[metadata.name]

#####
##### Grid traits: the granules are reprojected onto a global 1/240° latitude-longitude lattice
#####

const MODIS_LATTICE_SPACING = 1/240

Base.size(::AbstractMODISLandDataset, variable) =
    (round(Int, 360 / MODIS_LATTICE_SPACING), round(Int, 180 / MODIS_LATTICE_SPACING), 1)

DataWrangling.longitude_interfaces(::AbstractMODISLandDataset) = (-180, 180)
DataWrangling.latitude_interfaces(::AbstractMODISLandDataset)  = (-90, 90)

DataWrangling.is_three_dimensional(::MODISLandMetadata) = false
DataWrangling.reversed_latitude_axis(::AbstractMODISLandDataset) = false
DataWrangling.default_inpainting(::MODISLandMetadata) = nothing
DataWrangling.default_download_directory(::AbstractMODISLandDataset) = download_MODISLand_cache
DataWrangling.longitude_name(::MODISLandMetadata) = "lon"
DataWrangling.latitude_name(::MODISLandMetadata)  = "lat"

Oceananigans.Fields.location(::MODISLandMetadatum) = (Center, Center, Nothing)

struct MODISLAIScale end
struct MODISFPARScale end

DataWrangling.convert_units(x::FT, ::MODISLAIScale)  where FT = x * convert(FT, MODIS_LAI_SCALE)
DataWrangling.convert_units(x::FT, ::MODISFPARScale) where FT = x * convert(FT, MODIS_FPAR_SCALE)

# Files store digital numbers (and the climatology their reduction, which is linear in
# them), so the product's scale factor is applied on the way onto the grid.
function DataWrangling.conversion_units(metadatum::MODISLandMetadatum)
    metadatum.name === :fpar && return MODISFPARScale()
    metadatum.name in (:retained_retrieval_count, :landcover_code) && return nothing
    return MODISLAIScale()
end

# Class codes carry no scale factor.
DataWrangling.conversion_units(::MODISLandCoverMetadatum) = nothing

#####
##### Filenames
#####
##### Each granule read produces one regional file holding every layer, so the raw filename
##### is keyed by date and region but not by variable. The climatology reduces a single
##### variable, so its filename carries the variable, the years, the reducer, and the period.
#####

date_tag(date) = Dates.format(DateTime(date), "yyyymmdd")
years_tag(years) = string(first(years), "-", last(years))

bound_tag(bound) = replace(string(round(bound, digits = 3)), "-" => "m")

region_tag(::Nothing) = "global"

function region_tag(region::BoundingBox)
    (isnothing(region.longitude) || isnothing(region.latitude)) && return "global"
    west, east = region.longitude
    south, north = region.latitude
    return string("lon", bound_tag(west), "-", bound_tag(east),
                  "_lat", bound_tag(south), "-", bound_tag(north))
end

DataWrangling.metadata_filename(dataset::MODISLAIDataset, name, date, region) =
    string(modis_short_name(dataset), "_", modis_version(dataset), "_",
           date_tag(date), "_", region_tag(region), ".nc")

# A different legend reads a different layer, so the legend takes the variable's place.
DataWrangling.metadata_filename(dataset::MCD12Q1, name, date, region) =
    string(modis_short_name(dataset), "_", modis_version(dataset), "_", dataset.legend, "_",
           Dates.year(date), "_", region_tag(region), ".nc")

function DataWrangling.metadata_filename(dataset::MODISLAIClimatology, name, date, region)
    period = period_index(date, composite_period_days(dataset))
    return string(modis_short_name(dataset), "_", modis_version(dataset), "_", name,
                  "_climatology_", years_tag(dataset.years), "_", nameof(dataset.reducer),
                  "_p", lpad(period, 2, '0'), "_", region_tag(region), ".nc")
end

"""
    retained_retrieval_metadatum(metadatum)

The companion [`Metadatum`](@ref) for the retained-retrieval count stored beside a
climatology period's reduction — how many of the contributing years survived screening in
each cell. It reads the same file as `metadatum` and builds a `Field` through the same
path: `Field(retained_retrieval_metadatum(metadatum), grid)`.
"""
retained_retrieval_metadatum(metadatum::MODISLAIClimatologyMetadatum) =
    Metadatum(:retained_retrieval_count; dataset = metadatum.dataset, region = metadatum.region,
              date = metadatum.dates, dir = metadatum.dir, filename = metadatum.filename)

function DataWrangling.validate_dataset_coverage(grid, metadata::MODISLandMetadata)
    region = metadata.region
    if !(region isa BoundingBox) || isnothing(region.longitude) || isnothing(region.latitude)
        error("$(modis_short_name(metadata.dataset)) needs a longitude/latitude BoundingBox region.")
    end
    return nothing
end

#####
##### The class map on a model grid
#####

"""
    class_fractions(grid, dataset::MCD12Q1; date,
                    region = BoundingBox(grid),
                    dir = default_download_directory(dataset))

The MCD12Q1 class map of `date`'s year on `grid` as `(; fractions, majority_class)`: one
area-fraction `Field` per legend class name, regridded conservatively from the native
lattice over `region` with no-data counted as water, and the `Field` of the class code
covering the largest fraction of each cell, ties resolving to the lower code and `NaN`
outside `region`.
"""
function DataWrangling.class_fractions(grid, dataset::MCD12Q1;
                                       date,
                                       region = BoundingBox(grid),
                                       dir = default_download_directory(dataset))
    metadatum = Metadatum(:landcover_class; dataset, region, date, dir)
    fractions = class_fractions(grid, Field(metadatum, child_architecture(grid)), dataset)
    majority_class = majority_class!(Field{Center, Center, Nothing}(grid), fractions,
                                     landcover_class_names(dataset))
    return (; fractions, majority_class)
end

"""
    class_fractions(grid, classes::Field, dataset::MCD12Q1)

The area fraction of each legend class of `dataset` on `grid`, regridded conservatively
from the class map `classes` on its native lattice with no-data counted as water.
"""
function DataWrangling.class_fractions(grid, classes::Field, dataset::MCD12Q1)
    class_names = landcover_class_names(dataset)
    codes = interior(classes)
    water = convert(eltype(codes), class_names.water)
    indicator = Field{Center, Center, Nothing}(classes.grid)
    return map(class_names) do code
        interior(indicator) .= ifelse.(isfinite.(codes), codes, water) .== code
        regrid!(Field{Center, Center, Nothing}(grid), indicator)
    end
end

function DataWrangling.interpolate_physical!(target, classes, metadatum::Metadatum{<:MCD12Q1})
    metadatum.name === :landcover_class ||
        return DataWrangling.interpolate_physical!(target, classes)
    fractions = class_fractions(target.grid, classes, metadatum.dataset)
    return majority_class!(target, fractions, landcover_class_names(metadatum.dataset))
end
