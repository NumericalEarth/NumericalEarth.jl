module MODISLand

export MCD15A2H, MCD12Q1, MODISLAIClimatology, build_lai_climatology!,
       retained_retrieval_metadatum,
       lai_screening_mask, recommended_lai_screening, lai_rejection_flags,
       mask_lai_landcover, modis_landcover_class_names,
       landcover_class_names, igbp_class_names, igbp_non_vegetated_classes,
       class_fraction, class_fractions,
       class_maximum_gap, landcover_change_flag, zero_non_vegetated!,
       period_index, composite_window

using Dates: Dates, DateTime, Day, dayofyear
using Downloads: Downloads
using NCDatasets: NCDataset, defDim, defVar
using Oceananigans: Center
using Oceananigans.DistributedComputations: @root
using Statistics: mean

using ..DataWrangling: DataWrangling, Metadata, Metadatum, BoundingBox,
                       metadata_path, default_download_directory,
                       native_cell_range, native_convention_longitude,
                       cmr_granules_url, download_with_retries

import Oceananigans

download_MODISLand_cache::String = ""
function __init__()
    global download_MODISLand_cache = DataWrangling.download_cache("MODISLand")
    return nothing
end

#####
##### Digital-number decode
#####
##### MCD15 stores LAI, FPAR, and their standard deviations as `UInt8` digital numbers with
##### a valid range of 0–100. Codes above 100 are not measurements: 249 is unclassified, 250
##### urban, 251 wetland, 252 permanent snow/ice, 253 barren, 254 water, 255 fill, and 248
##### marks a missing standard deviation. They must be rejected *before* the product's scale
##### factor is applied, or a fill of 255 decodes to a leaf area index of 25.5. The scaling
##### itself is applied downstream by `conversion_units`.
#####

const MODIS_LAI_MAXIMUM_VALID = 100
const MODIS_LAI_SCALE  = 0.1
const MODIS_FPAR_SCALE = 0.01

# The land-cover codes the product substitutes for a retrieval, and the classes they name.
const MODIS_LAI_LANDCOVER_CODES = 249:254

const modis_landcover_class_names = (unclassified = 249,
                                     urban = 250,
                                     wetland = 251,
                                     snow_and_ice = 252,
                                     barren = 253,
                                     water = 254)

"""
    mask_lai_fill(DN)

Return the MCD15 digital number `DN` as a `Float32`, or `NaN` when `DN` exceeds the valid
range of `0:$(MODIS_LAI_MAXIMUM_VALID)` — the fill value and the land-cover special codes.
The product's scale factor is applied downstream, so the rejection has to happen here.
"""
@inline mask_lai_fill(DN) = ifelse(DN > MODIS_LAI_MAXIMUM_VALID, NaN32, Float32(DN))

"""
    mask_lai_landcover(DN)

The complement of [`mask_lai_fill`](@ref): return the digital number `DN` as a `Float32` when
it is one of the land-cover codes `$(MODIS_LAI_LANDCOVER_CODES)`, and `NaN` otherwise — for a
valid retrieval, and for the fill value 255, neither of which names a class.

The product substitutes these codes for a retrieval it does not attempt, so they carry the
non-vegetated classification of the same MODIS land cover the retrieval itself is keyed on:

| code | class |
|---|---|
| 249 | unclassified — not a usable class |
| 250 | urban / built-up |
| 251 | wetland / inundated marshland |
| 252 | perennial snow or ice |
| 253 | barren or sparsely vegetated |
| 254 | perennial salt or fresh water |

Reading them is how a surface-property closure learns which cells are not vegetated without a
separate land-cover product. They are **class codes**, so read them on the product's own grid:
interpolating them onto another grid averages 250 against 254 into 252, which is a different
class. A finer land-cover product (and a nearest-neighbor regrid) is the right tool once the
model grid differs.
"""
@inline mask_lai_landcover(DN) =
    ifelse((DN ≥ first(MODIS_LAI_LANDCOVER_CODES)) & (DN ≤ last(MODIS_LAI_LANDCOVER_CODES)),
           Float32(DN), NaN32)

#####
##### Land-cover legends
#####
##### MCD12Q1 stores every layer as a `UInt8` class code with fill 255, and each legend has
##### its own valid range: 0 is water in the leaf-area and plant-functional-type schemes but
##### is not a class at all under IGBP. The names and ranges below are the granules' own
##### attributes, not a secondary source.
#####

"""
    igbp_class_names

The 17 International Geosphere-Biosphere Programme classes of `LC_Type1`, the default
legend of [`MCD12Q1`](@ref) — the stratification the aerodynamic-roughness literature keys
its drag and minimum-stem-area tables on.

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
itself is keyed on, so it is the legend that matches the record this module composites
most closely.
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

They should agree with the codes MCD15A2H writes in place of a retrieval
([`mask_lai_landcover`](@ref)), which come from a related land-cover input. Where they do
not, the disagreement localizes a geolocation or a vintage difference between the two
products and is worth counting rather than reconciling silently.
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

"""
    class_fraction(codes, class)

The fraction of the valid entries of `codes` that carry `class`. Averaging class codes is
meaningless, but their per-class *fractions* are continuous fields that ride the shared
bilinear regrid onto a model grid safely, and over valid cells they sum to one.

```jldoctest
julia> using NumericalEarth

julia> codes = Float32[1 1 4; 4 NaN 1];

julia> class_fraction(codes, 1), class_fraction(codes, 4)
(0.6f0, 0.4f0)
```
"""
function class_fraction(codes, class)
    valid = count(isfinite, codes)
    valid == 0 && return NaN32
    return Float32(count(code -> isfinite(code) && code == class, codes) / valid)
end

"""
    class_fractions(codes, classes, factor)

Aggregate a class map onto a lattice `factor` times coarser, as one continuous area-fraction
field per class: `Dict(class => fraction)`, each `size(codes) .÷ factor`, summing to one over
cells with any valid code.

This is the safe way to carry a categorical field onto a model grid. The codes themselves
cannot be interpolated, but their fractions are ordinary continuous fields, and they say more
than a dominant class does — a cell that is 60% forest and 40% crop is not a forest cell.

`factor` must divide both dimensions, which it does when the coarse lattice is built by
grouping whole native cells.
"""
function class_fractions(codes, classes, factor)
    Nx, Ny = size(codes)
    (mod(Nx, factor) == 0 && mod(Ny, factor) == 0) ||
        throw(ArgumentError("An aggregation factor of $factor does not divide the class " *
                            "map's $((Nx, Ny)) cells."))

    fractions = Dict(class => fill(NaN32, Nx ÷ factor, Ny ÷ factor) for class in classes)

    for j in 1:(Ny ÷ factor), i in 1:(Nx ÷ factor)
        block = view(codes, ((i - 1) * factor + 1):(i * factor),
                            ((j - 1) * factor + 1):(j * factor))
        for class in classes
            fractions[class][i, j] = class_fraction(block, class)
        end
    end

    return fractions
end

#####
##### Quality screening
#####
##### Two bit-packed bytes accompany every retrieval. `FparLai_QC` carries the retrieval
##### provenance — bit 0 MODLAND_QC, bit 1 SENSOR, bit 2 DEADDETECTOR, bits 3–4 CLOUDSTATE,
##### bits 5–7 SCF_QC — and `FparExtra_QC` the scene state — bits 0–1 LANDSEA, bit 2
##### SNOW_ICE, bit 3 AEROSOL, bit 4 CIRRUS, bit 5 INTERNAL_CLOUDMASK, bit 6 CLOUD_SHADOW,
##### bit 7 SCF_BIOME_MASK.
#####

@inline modland_quality_control(qc) = qc & 0x01
@inline dead_detector(qc)           = (qc >> 0x02) & 0x01
@inline cloud_state(qc)             = (qc >> 0x03) & 0x03
@inline scf_quality_control(qc)     = (qc >> 0x05) & 0x07

@inline snow_or_ice(extra_qc)        = (extra_qc >> 0x02) & 0x01
@inline high_aerosol(extra_qc)       = (extra_qc >> 0x03) & 0x01
@inline cirrus(extra_qc)             = (extra_qc >> 0x04) & 0x01
@inline internal_cloud_mask(extra_qc) = (extra_qc >> 0x05) & 0x01
@inline cloud_shadow(extra_qc)       = (extra_qc >> 0x06) & 0x01

# Each named criterion a pixel can fail. The values are a private bit assignment used to
# combine criteria into one screening mask; they are unrelated to the product's own bit
# positions, which the accessors above decode.
const lai_screening_flags = (other_quality    = 0x0001,
                             backup_algorithm = 0x0002,
                             cloudy           = 0x0004,
                             dead_detector    = 0x0008,
                             snow_or_ice      = 0x0010,
                             high_aerosol     = 0x0020,
                             cirrus           = 0x0040,
                             internal_cloud   = 0x0080,
                             cloud_shadow     = 0x0100)

"""
    lai_rejection_flags(qc, extra_qc)

Return the [`lai_screening_mask`](@ref) bits of every quality criterion the pixel with
`FparLai_QC` byte `qc` and `FparExtra_QC` byte `extra_qc` fails:

| flag | meaning |
|---|---|
| `:other_quality` | `MODLAND_QC` reports other than good quality |
| `:backup_algorithm` | the radiative-transfer retrieval failed and the empirical back-up was used, or no value was produced |
| `:cloudy` | `CLOUDSTATE` reports significant or mixed cloud (the clear and assumed-clear states pass) |
| `:dead_detector` | dead detectors forced a retrieval from adjacent detectors |
| `:snow_or_ice` | snow or ice was detected |
| `:high_aerosol` | average or high aerosol was detected |
| `:cirrus` | cirrus was detected |
| `:internal_cloud` | the surface-reflectance internal cloud mask flagged cloud |
| `:cloud_shadow` | cloud shadow was detected |

A pixel that passes everything returns `0x0000`.
"""
@inline function lai_rejection_flags(qc, extra_qc)
    f = lai_screening_flags
    cloud = cloud_state(qc)

    rejected  = ifelse(modland_quality_control(qc) != 0, f.other_quality,    0x0000)
    rejected |= ifelse(scf_quality_control(qc) > 0x01,   f.backup_algorithm, 0x0000)
    rejected |= ifelse((cloud == 0x01) | (cloud == 0x02), f.cloudy,          0x0000)
    rejected |= ifelse(dead_detector(qc) != 0,           f.dead_detector,    0x0000)
    rejected |= ifelse(snow_or_ice(extra_qc) != 0,       f.snow_or_ice,      0x0000)
    rejected |= ifelse(high_aerosol(extra_qc) != 0,      f.high_aerosol,     0x0000)
    rejected |= ifelse(cirrus(extra_qc) != 0,            f.cirrus,           0x0000)
    rejected |= ifelse(internal_cloud_mask(extra_qc) != 0, f.internal_cloud, 0x0000)
    rejected |= ifelse(cloud_shadow(extra_qc) != 0,      f.cloud_shadow,     0x0000)
    return rejected
end

"""
    lai_screening_mask(names::Symbol...)

Combine the named quality criteria into one screening mask, for use as the
`screened_flags` of an [`MCD15A2H`](@ref) dataset. A pixel failing any criterion in the
mask is read as `NaN`. The criteria are listed under [`lai_rejection_flags`](@ref).

```jldoctest
julia> using NumericalEarth

julia> lai_screening_mask(:other_quality, :snow_or_ice)
0x0011
```
"""
function lai_screening_mask(names::Symbol...)
    mask = 0x0000
    for name in names
        haskey(lai_screening_flags, name) ||
            throw(ArgumentError("$name is not a quality criterion; valid names are " *
                                "$(keys(lai_screening_flags))."))
        mask |= lai_screening_flags[name]
    end
    return mask
end

"""
    recommended_lai_screening()

The screen the product's user guide recommends and the aerodynamic-roughness literature
applies: keep only good-quality pixels retrieved by the main radiative-transfer algorithm
under a clear (or assumed-clear) sky. This is the default `screened_flags` of
[`MCD15A2H`](@ref).

The scene-state detections — snow, aerosol, cirrus, internal cloud, cloud shadow — are
deliberately left in. Snow in particular is a physical state rather than a retrieval failure:
it depresses winter leaf area over evergreen needleleaf, which downstream closures may want
to see and treat explicitly, so screening it out is opt-in via
[`lai_screening_mask`](@ref).

```jldoctest
julia> using NumericalEarth

julia> recommended_lai_screening()
0x0007
```
"""
recommended_lai_screening() =
    lai_screening_mask(:other_quality, :backup_algorithm, :cloudy)

@inline lai_screened(qc, extra_qc, mask) = !iszero(lai_rejection_flags(qc, extra_qc) & mask)

#####
##### Granule names
#####

"""
    parse_granule_name(name)

Split a MODIS land-product granule name — `MCD15A2H.A2020185.h10v05.061.2020340132006` —
into `(; date, tile, production)`: the composite's first day, the sinusoidal tile it covers,
and the processing timestamp that distinguishes reprocessings of the same tile and date.
"""
function parse_granule_name(name::AbstractString)
    m = match(r"\.A(\d{4})(\d{3})\.(h\d{2}v\d{2})\.\d{3}\.(\d+)", name)
    isnothing(m) &&
        throw(ArgumentError("could not parse a MODIS granule name from \"$name\""))
    composite_year, composite_day, tile, production = m.captures
    return (date = DateTime(parse(Int, composite_year)) + Day(parse(Int, composite_day) - 1),
            tile = String(tile),
            production = parse(Int, production))
end

"""
    select_granules(urls, date)

Keep the granule `urls` whose composite begins on `date`, one per sinusoidal tile: the
most recently processed, so a reprocessed tile supersedes its predecessor. A bounding-box
granule search returns the neighboring composites too (their date ranges overlap the
requested day), which is why the date is matched rather than trusted.
"""
function select_granules(urls, date)
    latest = Dict{String, Tuple{Int, String}}()
    for url in urls
        granule = parse_granule_name(basename(url))
        granule.date == DateTime(date) || continue
        current = get(latest, granule.tile, nothing)
        if isnothing(current) || granule.production > first(current)
            latest[granule.tile] = (granule.production, url)
        end
    end
    return [last(latest[tile]) for tile in sort!(collect(keys(latest)))]
end

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
sensor characteristics. Extending the range past 2019 is allowed, and worth noting when
reporting a result.

Carrying `years` on the dataset is a deliberate exception to keeping dates out of dataset
objects: it is part of the climatology's identity — a different span is a different product
— and it lets the shared time-series machinery treat the climatology as an ordinary
time-varying dataset. The monthly albedo climatology sets the same precedent.

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

The `(class_name = code, …)` table of the dataset's legend, so a class is named rather than
written as a bare integer.

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

# The product's first composite. Later composites are year-anchored, not a rolling cadence.
first_composite_date(::MCD15A2H) = DateTime(2002, 7, 4)

# The record is ongoing; this is the conservative end of the range `all_dates` advertises,
# in the same spirit as the reanalysis datasets. Later composites can still be requested
# by passing explicit `dates`.
last_composite_date(::MCD15A2H) = DateTime(2025, 12, 31)

# The land-cover map is produced about a year and a half in arrears, so the advertised range
# lags the leaf-area record. A later year can still be requested with an explicit date.
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
# code where it has no retrieval. It is readable but not reducible: a class code cannot be
# averaged, so it is deliberately absent from `MODISLAI_variable_names`, which is the set the
# climatology composites.
const MODISLAI_readable_variable_names =
    merge(MODISLAI_variable_names, Dict(:landcover_code => "Lai_500m"))

const lai_quality_variable       = "FparLai_QC"
const lai_extra_quality_variable = "FparExtra_QC"

const landcover_quality_variable   = "QC"
const landcover_water_mask_variable = "LW"

"""
    stored_granule_layers(dataset)

The granule layers copied into the local regional file. One warp per layer serves every
variable a read can ask for, so the set is a property of the product rather than of the
variable requested: the leaf-area product stores its three physical layers plus the two
quality bytes its screen decodes, and the land-cover product stores the legend's own layer
plus the classification quality and the land/water mask.
"""
stored_granule_layers(::MODISLAIDataset) = ("Lai_500m", "Fpar_500m", "LaiStdDev_500m",
                                            lai_quality_variable, lai_extra_quality_variable)

stored_granule_layers(dataset::MCD12Q1) = (landcover_layer(dataset),
                                           landcover_quality_variable,
                                           landcover_water_mask_variable)

# The name a climatology file stores its retained-retrieval count under.
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
##### (≈464 m, the 500 m product's actual pixel size) restricted to the requested region,
##### so the stored file *is* the native grid the shared regrid path expects — see
##### [`regional_lattice`](@ref).
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

# Files store digital numbers (and the climatology stores their reduction, which is linear
# in them), so the product's scale factor is applied on the way onto the grid. The retained
# count is a count.
function DataWrangling.conversion_units(metadatum::MODISLandMetadatum)
    metadatum.name === :fpar && return MODISFPARScale()
    metadatum.name in (:retained_retrieval_count, :landcover_code) && return nothing
    return MODISLAIScale()
end

# Class codes, an enumerated quality flag, and a land/water mask carry no scale factor, and
# the leaf-area fallthrough above would turn IGBP class 12 into 1.2 without erroring.
DataWrangling.conversion_units(::MODISLandCoverMetadatum) = nothing

#####
##### Dates
#####

"""
    modis_composite_dates(start_date, end_date, period_days)

Return the year-anchored composite dates between `start_date` and `end_date` inclusive: in
each year, day-of-year `1, 1 + period_days, 1 + 2 * period_days, …` up to the last period
that begins within that year.

MODIS land products restart their compositing period at day-of-year 1 every January, so
the last period of a year is short (5 days, or 6 in a leap year, for an 8-day product) and
the sequence is *not* a uniform cadence across a year boundary. Stepping uniformly from the
first date instead would drift out of phase after one year and request composites that do
not exist.

```jldoctest
julia> using Dates, NumericalEarth.DataWrangling.MODISLand

julia> dates = MODISLand.modis_composite_dates(DateTime(2020), DateTime(2021, 12, 31), 8);

julia> length(dates), dates[46], dates[47]
(92, DateTime("2020-12-26T00:00:00"), DateTime("2021-01-01T00:00:00"))
```
"""
function modis_composite_dates(start_date, end_date, period_days)
    dates = DateTime[]
    for year in Dates.year(start_date):Dates.year(end_date)
        january_first = DateTime(year, 1, 1)
        for day in 1:period_days:Dates.daysinyear(year)
            date = january_first + Dates.Day(day - 1)
            start_date ≤ date ≤ end_date && push!(dates, date)
        end
    end
    return dates
end

DataWrangling.all_dates(dataset::MODISLAIDataset, variable) =
    modis_composite_dates(first_composite_date(dataset), last_composite_date(dataset),
                          composite_period_days(dataset))

"""
    periods_per_year(dataset)

The number of composites a year of `dataset` holds — 46 for an 8-day product. The
compositing period restarts at day-of-year 1 every January, so the last period of a year
is short and the count is the same in leap and common years.
"""
periods_per_year(dataset) = length(modis_composite_dates(climatology_year_start(),
                                                         climatology_year_end(),
                                                         composite_period_days(dataset)))

# A common (non-leap) placeholder year carries the climatological stamps, so the period
# count and the day-of-year of each stamp are the ones every year shares.
climatology_year_start() = DateTime(2018, 1, 1)
climatology_year_end()   = DateTime(2018, 12, 31)

DataWrangling.all_dates(climatology::MODISLAIClimatology, variable) =
    modis_composite_dates(climatology_year_start(), climatology_year_end(),
                          composite_period_days(climatology))

DataWrangling.is_seasonal_climatology(::MODISLAIClimatology) = true

# One map per calendar year, stamped on 1 January — not a day-stepped composite cadence.
DataWrangling.all_dates(dataset::MCD12Q1, variable) =
    [DateTime(year) for year in first_landcover_year(dataset):last_landcover_year(dataset)]

"""
    period_index(date, period_days)
    period_index(date, dataset)

The 1-based index of the year-anchored composite period containing `date` — which of a
seasonal climatology's periods a calendar date belongs to, and so the `anchor_periods` a
date-window series needs to map onto a climatology.

```jldoctest
julia> using NumericalEarth, Dates

julia> period_index(DateTime(2019, 7, 4), MCD15A2H())
24
```
"""
period_index(date, period_days::Integer) = (dayofyear(date) - 1) ÷ period_days + 1

period_index(date, dataset::AbstractMODISLandDataset) =
    period_index(date, composite_period_days(dataset))

"""
    composite_window(dataset, date)

The `(start, stop)` dates of the compositing window a file stamped `date` holds. The
cadence restarts on 1 January, so the last window of a year is short — five days, or six in
a leap year, for an 8-day product.
"""
function composite_window(dataset, date)
    period_days = composite_period_days(dataset)
    start = DateTime(date)
    year_start = DateTime(Dates.year(start), 1, 1)
    stop = year_start + Day(period_index(start, period_days) * period_days)
    return start, min(stop, year_start + Dates.Year(1))
end

DataWrangling.sample_window(metadatum::Union{MODISLAIMetadatum,
                                            MODISLAIClimatologyMetadatum}) =
    composite_window(metadatum.dataset, metadatum.dates)

#####
##### Filenames
#####
##### Each granule read produces one regional file holding every layer, so the raw filename
##### is keyed by date and region but not by variable — three variables and both quality
##### bytes come out of one download. The climatology reduces a single variable, so its
##### filename carries the variable, the contributing years, and the period.
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

# One warp per year and region serves every land-cover variable too, but a different legend
# reads a different layer, so the legend takes the place the variable would otherwise hold.
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
each cell, so a composite's coverage can be mapped next to its values. It reads the same
file as `metadatum`, because a count only means anything beside the variable it counts, and
builds a `Field` through the same path: `Field(retained_retrieval_metadatum(metadatum), grid)`.
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

#####
##### The regional lattice the granules are warped onto
#####

"""
    regional_lattice(metadata)

The regional latitude-longitude window the sinusoidal granules are reprojected onto:
`(; west, south, east, north, Nx, Ny)`, in degrees and cells of the product's 1/240°
lattice.

The window is exactly the set of native cells `native_grid` keeps for the metadata's
region, so the stored file and the native grid share their cells one for one. That pins the
region offset of the shared regrid to zero instead of leaving it to a floating-point
comparison between the grid's nodes and the file's coordinates — the difference between a
correct read and one shifted by a cell on a fine grid.
"""
function regional_lattice(metadata::MODISLandMetadata)
    region = metadata.region
    (region isa BoundingBox && !isnothing(region.longitude) && !isnothing(region.latitude)) ||
        throw(ArgumentError("regional_lattice requires a bounded (longitude, latitude) BoundingBox."))

    Nx, Ny, _ = size(metadata.dataset, metadata.name)
    native_longitude = DataWrangling.longitude_interfaces(metadata)
    native_latitude  = DataWrangling.latitude_interfaces(metadata)
    bbox_longitude = native_convention_longitude(region.longitude, native_longitude)

    last(bbox_longitude) > last(native_longitude) &&
        throw(ArgumentError("The requested longitude window $(region.longitude) wraps the ±180° " *
                            "seam of the MODIS sinusoidal grid's reprojection. Split it into two " *
                            "requests, one on each side of the seam."))

    icols = native_cell_range(bbox_longitude, native_longitude, Nx)
    jrows = native_cell_range(region.latitude, native_latitude, Ny)

    return (west  = first(native_longitude) + (first(icols) - 1) * MODIS_LATTICE_SPACING,
            east  = first(native_longitude) + last(icols) * MODIS_LATTICE_SPACING,
            south = first(native_latitude) + (first(jrows) - 1) * MODIS_LATTICE_SPACING,
            north = first(native_latitude) + last(jrows) * MODIS_LATTICE_SPACING,
            Nx = length(icols), Ny = length(jrows))
end

#####
##### Granule discovery
#####

"""
    MissingGranulesError

Raised when the Common Metadata Repository holds no granule for a requested region and
date. The record has occasional holes where an instrument outage prevented a composite —
2016-02-18 is one — so this is a distinguishable condition rather than a plain error:
[`build_lai_climatology!`](@ref) skips a date the archive does not carry and composites the
rest, while an explicit read of that date still fails, because there is nothing to read.
"""
struct MissingGranulesError <: Exception
    message :: String
end

Base.showerror(io::IO, err::MissingGranulesError) = print(io, err.message)

"""
    granule_urls(metadatum)

Query the Common Metadata Repository and return the download URLs of the granules covering
the metadatum's region and date, one per sinusoidal tile (see [`select_granules`](@ref)).
Requires network access, but no credentials.
"""
function granule_urls(metadatum::MODISLandMetadatum)
    dataset = metadatum.dataset

    # One day of one product is a handful of sinusoidal tiles, so a single page always covers it.
    url = cmr_granules_url(modis_short_name(dataset), modis_version(dataset), metadatum.region;
                           date = metadatum.dates)

    candidates = mktempdir() do tmp
        json = joinpath(tmp, "cmr_granules.json")
        download_with_retries(url, json; description = "CMR granule query")
        text = read(json, String)
        unique(m.match for m in eachmatch(r"https://[^\"]+\.hdf", text))
    end

    granules = select_granules(candidates, metadatum.dates)
    isempty(granules) &&
        throw(MissingGranulesError(
            "The Common Metadata Repository holds no $(modis_short_name(dataset)) granules " *
            "for the region $(metadatum.region) on $(metadatum.dates). The record has " *
            "occasional holes where an instrument outage prevented a composite; a " *
            "climatology skips them, but a read of that date alone cannot."))

    return granules
end

#####
##### Download
#####

function Downloads.download(metadata::MODISLandMetadata)
    @root for metadatum in metadata
        path = metadata_path(metadatum)
        isfile(path) || modis_granules_to_netcdf(metadatum, path)
    end
    return metadata_path(metadata)
end

function Downloads.download(metadata::MODISLAIClimatologyMetadata)
    @root for metadatum in metadata
        isfile(metadata_path(metadatum)) && continue
        period = period_index(metadatum.dates, composite_period_days(metadatum.dataset))
        build_lai_climatology!(metadatum.dataset; name = climatology_build_name(metadatum),
                               periods = period:period, region = metadatum.region,
                               dir = metadatum.dir)
    end
    return metadata_path(metadata)
end

# The retained count is written beside the variable its file composites, so a cold-cache read
# of the count builds that variable — recovered from the filename, which the companion
# metadatum of `retained_retrieval_metadatum` shares with the variable it counts.
function climatology_build_name(metadatum::MODISLAIClimatologyMetadatum)
    metadatum.name === :retained_retrieval_count || return metadatum.name

    for name in keys(MODISLAI_variable_names)
        DataWrangling.metadata_filename(metadatum.dataset, name, metadatum.dates,
                                        metadatum.region) == metadatum.filename && return name
    end

    throw(ArgumentError("A retained-retrieval count is stored beside the variable it counts, " *
                        "so it cannot say on its own which climatology to build. Read it with " *
                        "`retained_retrieval_metadatum` of the composited variable's metadatum."))
end

# Implemented in ext/NumericalEarthArchGDALExt.jl once `ArchGDAL` is loaded.
modis_granules_to_netcdf(metadatum, nc_path) =
    error("Reading MODIS HDF-EOS granules requires ArchGDAL.jl built with GDAL's HDF4 " *
          "driver, and NASA Earthdata credentials (EARTHDATA_USERNAME / " *
          "EARTHDATA_PASSWORD). Load ArchGDAL with `using ArchGDAL`.")

#####
##### Reading
#####
##### The stored file holds the whole regional lattice, so the read is a plain slurp: mask
##### the digital numbers that are not measurements, apply the quality screen, and let the
##### shared path scale and bracket the result onto the grid.
#####

function DataWrangling.retrieve_data(metadatum::MODISLAIMetadatum)
    variable = DataWrangling.dataset_variable_name(metadatum)
    mask = screened_flags(metadatum.dataset)

    # The land-cover code describes the surface, not a retrieval, so the retrieval screen does
    # not apply to it: a cloudy urban pixel is still urban.
    metadatum.name === :landcover_code &&
        return NCDataset(metadata_path(metadatum)) do ds
            mask_lai_landcover.(ds[variable][:, :])
        end

    return NCDataset(metadata_path(metadatum)) do ds
        𝒜 = mask_lai_fill.(ds[variable][:, :])

        if !iszero(mask)
            qc = ds[lai_quality_variable][:, :]
            extra_qc = ds[lai_extra_quality_variable][:, :]
            𝒜 = ifelse.(lai_screened.(qc, extra_qc, mask), NaN32, 𝒜)
        end

        𝒜
    end
end

function DataWrangling.retrieve_data(metadatum::MODISLAIClimatologyMetadatum)
    variable = DataWrangling.dataset_variable_name(metadatum)
    return NCDataset(metadata_path(metadatum)) do ds
        Float32.(ds[variable][:, :])
    end
end

# `QC` is an enumerated classification outcome (0 good classified land, 10 no data), not a
# packed bitfield like `FparLai_QC`, so there is nothing to screen against by default and
# bit-masking it would produce nonsense. Every layer decodes the same way: reject the codes
# outside the layer's own valid range, and leave the rest as the class they name.
function DataWrangling.retrieve_data(metadatum::MODISLandCoverMetadatum)
    variable = DataWrangling.dataset_variable_name(metadatum)
    valid = metadatum.name === :landcover_class ? landcover_valid_range(metadatum.dataset) :
            metadatum.name === :quality_flag    ? (0:10) : (1:2)

    return NCDataset(metadata_path(metadatum)) do ds
        mask_landcover_fill.(ds[variable][:, :], Ref(valid))
    end
end

#####
##### Seasonal climatology
#####

"""
    reduce_retained(reducer, samples)

Apply `reducer` to the finite entries of `samples`, returning `(value, count)` — the
reduction and how many retrievals survived screening. With no finite entry the value is
`NaN32` and the count `0`, so a period no year could observe stays visibly empty instead of
silently reading as zero.
"""
function reduce_retained(reducer, samples)
    retained = filter(isfinite, samples)
    isempty(retained) && return NaN32, 0
    return Float32(reducer(retained)), length(retained)
end

"""
    build_lai_climatology!(dataset::MODISLAIClimatology;
                           name = :leaf_area_index,
                           region,
                           periods = 1:periods_per_year(dataset),
                           reducer = mean,
                           dir = default_download_directory(dataset))

Build the cached per-period files behind a [`MODISLAIClimatology`](@ref) over `region`. For
each composite period in `periods`, every contributing date of `dataset.years` is
downloaded and screened, the retained retrievals are combined pixel by pixel with
`reducer`, and one file is written holding the reduction and the retained-retrieval count.
Periods already on disk are skipped, so an interrupted build resumes. Returns the paths of
the files.

`reducer` acts on the vector of retained values of a pixel: `mean` gives the seasonal mean
the roughness literature composites, while `maximum` (or a high quantile) gives a
peak-season field, which is the more useful boundary condition for a simulation spanning
days rather than a year.

Screening happens before the reduction, so the count records retained retrievals and the
reported retained fraction is the honest one.

A year the archive holds no composite for — the record has a few, where an instrument
outage interrupted it — is skipped with a warning and costs that period one sample rather
than aborting the build. Every other failure still raises.
"""
function build_lai_climatology!(dataset::MODISLAIClimatology;
                                name = :leaf_area_index,
                                region,
                                periods = 1:periods_per_year(dataset),
                                reducer = mean,
                                dir = default_download_directory(dataset))

    haskey(MODISLAI_variable_names, name) ||
        throw(ArgumentError("$name cannot be composited — a retained-retrieval count and a " *
                            "land-cover code are not quantities a reducer means anything on. " *
                            "Build the climatology of a variable that is, and read the count " *
                            "with `retained_retrieval_metadatum`. The variables that can be " *
                            "composited are $(keys(MODISLAI_variable_names))."))

    source = source_dataset(dataset)
    period_days = composite_period_days(dataset)
    source_dates = DataWrangling.all_dates(source, name)
    paths = String[]

    for period in periods
        stamp = climatology_year_start() + Day((period - 1) * period_days)
        filepath = joinpath(dir, DataWrangling.metadata_filename(dataset, name, stamp, region))
        push!(paths, filepath)
        isfile(filepath) && continue

        dates = [date for date in source_dates
                 if Dates.year(date) in dataset.years && period_index(date, period_days) == period]
        isempty(dates) &&
            error("No $(modis_short_name(source)) composites fall in period $period of " *
                  "the years $(dataset.years).")

        available = materialize_composites(name, source, dates, region, dir)
        isempty(available) &&
            error("The archive holds no $(modis_short_name(source)) composite of period " *
                  "$period in any of the years $(dataset.years).")

        metadata = Metadata(name; dataset = source, dates = available, region, dir)

        @info string("Compositing ", length(available), " retrievals into period ", period,
                     " of the ", modis_short_name(source), " ", name, " climatology...")
        write_lai_composite(filepath, metadata, reducer)
    end

    return paths
end

# The years contributing to one period, minus the ones the archive has no composite for. A
# multi-year mean is exactly the reduction that tolerates a missing year — it is what the
# retained-retrieval count records — so a hole in the record costs one sample rather than the
# whole period. Anything other than a missing granule still raises.
function materialize_composites(name, source, dates, region, dir)
    available = eltype(dates)[]

    for date in dates
        metadatum = Metadatum(name; dataset = source, region, date, dir)
        try
            Downloads.download(metadatum)
            push!(available, date)
        catch err
            err isa MissingGranulesError || rethrow(err)
            @warn string("The ", modis_short_name(source), " record has no composite on ",
                         date, "; compositing this period without it.")
        end
    end

    return available
end

function write_lai_composite(filepath, metadata, reducer)
    variable = DataWrangling.dataset_variable_name(first(metadata))
    λ, φ = DataWrangling.read_file_coords(first(metadata))
    samples = stack(DataWrangling.retrieve_data(metadatum) for metadatum in metadata)

    Nx, Ny = size(samples, 1), size(samples, 2)
    composite = Array{Float32}(undef, Nx, Ny)
    retained_count = Array{Int32}(undef, Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        composite[i, j], retained_count[i, j] = reduce_retained(reducer, view(samples, i, j, :))
    end

    retained_fraction = sum(retained_count) / (Nx * Ny * size(samples, 3))
    @info string(" ... retained ", round(100 * retained_fraction, digits = 1),
                 "% of the retrievals; ", round(100 * mean(isnan.(composite)), digits = 1),
                 "% of the cells have none")

    # Unique per writer, in the destination directory — see the note in the granule writer: a
    # shared staging name turns the atomic rename into a way to publish a truncated file.
    staging_path = tempname(dirname(filepath); cleanup = false) * ".nc"
    NCDataset(staging_path, "c") do ds
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)
        defVar(ds, "lon", collect(λ), ("lon",);
               attrib = ["units" => "degrees_east", "long_name" => "longitude"])
        defVar(ds, "lat", collect(φ), ("lat",);
               attrib = ["units" => "degrees_north", "long_name" => "latitude"])
        defVar(ds, variable, composite, ("lon", "lat"); deflatelevel = 2, shuffle = true)
        defVar(ds, retained_count_variable, retained_count, ("lon", "lat");
               attrib = ["long_name" => "number of retained retrievals"],
               deflatelevel = 2, shuffle = true)
    end

    mv(staging_path, filepath; force = true)
    return nothing
end

#####
##### What the class field contributes to the gap-fill chain
#####

# How many consecutive 8-day periods a linear bridge across a gap may span, by IGBP class.
# The number is set by how fast the class's leaf area moves: an evergreen canopy holds its
# leaf area to within a factor of about 1.4 over the year, so a month-long bridge is nearly
# exact, while a deciduous stand or a crop swings by close to a factor of five across a
# green-up that takes three weeks, and a bridge longer than one period fabricates a ramp
# with the wrong slope and the wrong timing. The non-vegetated classes are zero because
# nothing should be filled there at all.
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

These are the cells [`fill_seasonal_gaps!`](@ref) deliberately leaves missing, because the
product does not retrieve there and borrowing a neighbor's canopy for them would be an
invention. Zero is not a stand-in for the missing value: it is the value. Leaf area per unit
ground area over open water is zero.

Run it **after** the fill and **before** landing the series on a model grid, which is the
order that leaves nothing for the regrid's stencil to spread — a `NaN` at a shoreline
otherwise dilates into its neighbors, while a zero blends into the cell mean correctly,
since leaf area is already per unit *ground* area.

Keep the class field alongside the result. Zero says there is no canopy; it does not say
whether the surface is a lake or a car park, and those want roughness lengths four orders of
magnitude apart. A canopy closure fed this field alone reads a city as smoother than a wheat
field.

Score a fill before zeroing, never after: [`gap_fill_denial`](@ref) samples cells that carry
a value, so zeroed water would enter the sample as truth the chain reproduces exactly.
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

"""
    landcover_change_flag(class_series; window = 3)

Flag the cells whose land cover changed inside a multi-year span, given `class_series` —
the annual [`MCD12Q1`](@ref) maps of one region stacked along a third dimension, oldest
first.

A cell that was closed forest at the start of a climatology's span and pasture at its end
averages into one composite describing neither surface. The annual series is the only way
to see that, and this is a flag rather than a correction: which of the two surfaces a
simulation wants is the caller's decision.

A single year's label at 500 m is not reliable enough for a first-versus-last difference —
the product's user guide says so explicitly — so the test is persistence. A cell is flagged
only when the same class holds through each of the first `window` years, the same class
holds through each of the last `window` years, and the two differ. Interannual label noise
therefore reads as "not stable" rather than as change.
"""
function landcover_change_flag(class_series; window = 3)
    Nx, Ny, Nyears = size(class_series)
    2 * window ≤ Nyears ||
        throw(ArgumentError("A persistence test over $window years needs at least " *
                            "$(2 * window) years of land cover; got $Nyears."))

    persistent(years) = all(isfinite, years) && all(==(first(years)), years)

    changed = falses(Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        early = view(class_series, i, j, 1:window)
        late  = view(class_series, i, j, (Nyears - window + 1):Nyears)
        changed[i, j] = persistent(early) && persistent(late) && first(early) != first(late)
    end

    return changed
end

end # module MODISLand
