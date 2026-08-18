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
a retrieval, listed under [`mask_lai_landcover`](@ref).

The product targets *true* leaf area index, applying its own per-biome clumping inside the
retrieval, so it is the quantity a canopy drag closure calibrated on MODIS expects — unlike
the *effective* leaf area index the two-stream inversion records report.

Granules are HDF-EOS2 tiles on the sinusoidal projection, discovered from NASA's Common
Metadata Repository and reprojected to a regional latitude-longitude window at download
time. Build the `Metadata` with a lon/lat [`BoundingBox`](@ref); a global read is not
supported. Requires `ArchGDAL` (for GDAL's HDF4 driver) and a NASA Earthdata login
(`EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` — see this module's README).

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
    MODISLAIClimatology(; dataset = MCD15A2H(), years = 2003:2019)

A seasonal climatology of a [`MODISLAIDataset`](@ref): one composite per period of the
year, each reducing that period's retrievals across `years` pixel by pixel with the
screen `dataset` carries. An 8-day source gives 46 periods, so
`FieldTimeSeries(Metadata(:leaf_area_index; dataset = MODISLAIClimatology(), region), grid)`
is a 46-slot cyclic seasonal series.

Multi-year compositing is what makes an 8-day leaf-area series usable: a period that is
cloudy in one year is usually clear in another, so the composite's residual gap fraction
falls far below any single retrieval's. Cells no year could observe stay `NaN` rather than
reading as zero, and the number of retained retrievals behind every cell is stored beside
the reduction — see [`retained_retrieval_metadatum`](@ref).

`years` defaults to 2003–2019, the span over which both Terra and Aqua held their
equatorial crossing times; Terra began drifting in 2020, which changes the composite's
sensor characteristics.

```jldoctest
julia> using NumericalEarth

julia> MODISLAIClimatology()
MODISLAIClimatology(MCD15A2H(screened_flags=0x0007), years=2003:2019)
```
"""
struct MODISLAIClimatology{D <: MODISLAIDataset, Y} <: AbstractMODISLandDataset
    dataset :: D
    years :: Y
end

MODISLAIClimatology(; dataset = MCD15A2H(), years = 2003:2019) =
    MODISLAIClimatology(dataset, years)

Base.show(io::IO, climatology::MODISLAIClimatology) =
    print(io, "MODISLAIClimatology(", climatology.dataset, ", years=", climatology.years, ")")

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

`:IGBP` is the default because the roughness literature's drag and stem-area tables are
keyed on it. `:LAI` is the stratification the MCD15 retrieval itself uses, which makes it
the closer match when the class field is there to pool leaf-area donors.

Class codes are **not** interpolable: read them on the product's own grid, where
`Field(metadatum)` lands them, and take [`class_fraction`](@ref) if a model grid is wanted.
A bilinear regrid averages urban (13) against water (17) into permanent snow (15).

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
julia> using NumericalEarth

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

# `:landcover_code` is read from the leaf-area layer itself, which substitutes a land-cover
# code where it has no retrieval. A class code cannot be averaged, so it is absent from
# `MODISLAI_variable_names`, the set the climatology composites.
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
##### Grid traits
#####
##### The sinusoidal granules are reprojected to a global 1/240° latitude-longitude lattice
##### (≈464 m, the 500 m product's actual pixel size) restricted to the requested region —
##### see [`regional_lattice`](@ref).
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

# Class codes carry no scale factor, and the leaf-area fallthrough above would silently
# turn IGBP class 12 into 1.2.
DataWrangling.conversion_units(::MODISLandCoverMetadatum) = nothing

#####
##### Filenames
#####
##### Each granule read produces one regional file holding every layer, so the raw filename
##### is keyed by date and region but not by variable. The climatology reduces a single
##### variable, so its filename carries the variable, the years, and the period.
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
                  "_climatology_", years_tag(dataset.years),
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
        error("$(modis_short_name(metadata.dataset)) must be used with a bounded region. " *
              "Build the metadata with a longitude/latitude BoundingBox, e.g.\n" *
              "    metadata = Metadata(:$(metadata.name); dataset = $(metadata.dataset),\n" *
              "                        region = BoundingBox(longitude = (λ₁, λ₂), latitude = (φ₁, φ₂)))")
    end
    return nothing
end
