# MODIS 500 m leaf area index and land cover

This module ingests two MODIS V061 land products and builds seasonal climatologies from the
first:

- **MCD15A2H** — combined Terra + Aqua leaf area index / FPAR: 500 m, 8-day composites on
  the sinusoidal grid, from 2002-07-04 onwards.
- **MCD12Q1** — annual land cover: 500 m, one map per calendar year from 2001, on the same
  granules and the same reprojected lattice, so a class field and a leaf-area field over the
  same region share their cells one for one.

The product targets **true** leaf area index: per-biome foliage clumping is applied inside
the retrieval itself. Two-stream inversion records (the C3S 300 m record this repository also
ingests) report the **effective** quantity instead. A canopy closure calibrated against MODIS
therefore expects these values directly, with no clumping conversion in between — which is
the main reason to keep both records.

## Credentials

Downloads need a free NASA Earthdata login and GDAL's HDF4 driver:

1. Register at https://urs.earthdata.nasa.gov
2. Export `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` (the variables the `earthaccess`
   library also honours), or add a `~/.netrc` entry for `urs.earthdata.nasa.gov`
3. Load the backend before downloading: `using ArchGDAL`

Granule *discovery* goes through NASA's Common Metadata Repository and is anonymous; only
the granule fetch is authenticated.

## What gets stored locally

Granules are HDF-EOS2 tiles on the sinusoidal projection, so a lon/lat region generally
spans several of them and none of them is a lon/lat raster. At download time every granule
covering the region and date is fetched, and each layer is mosaicked and reprojected in one
`gdalwarp` call onto a regional latitude-longitude window. A region is required; a global
read is rejected rather than silently pulling 300 tiles.

The window is exactly the set of cells of a global 1/240° lattice (≈464 m, the product's
actual pixel size) that the shared native grid keeps for the region, so the stored file and
the native grid share their cells one for one. That pins the region offset of the regrid to
zero instead of leaving it to a floating-point comparison between the grid's nodes and the
file's coordinates.

Resampling is nearest-neighbour throughout. The source and target cell sizes agree to within
a percent, and every stored layer is either a bit-packed quality byte or a digital number
whose out-of-range codes carry meaning — averaging either would invent values. Cells the
granules do not cover take the fill code 255, which the read path rejects on both the digital
numbers and the quality bytes.

One file per date and region holds every layer — `Lai_500m`, `Fpar_500m`, `LaiStdDev_500m`,
`FparLai_QC`, `FparExtra_QC` — so all three variables and the screen come out of one download.
A land-cover file holds the chosen legend's layer plus `QC` and `LW`, keyed by legend and
year rather than by variable, for the same reason.

## Decode order

Digital numbers above 100 are not measurements: 249 is unclassified, 250 urban, 251 wetland,
252 permanent snow/ice, 253 barren, 254 water, and 255 fill. They are rejected **before** the
0.1 (LAI) or 0.01 (FPAR) scale factor is applied — decode the other way round and a fill of
255 becomes a leaf area index of 25.5.

## The land-cover codes are data, not just fill

Codes 249–254 are the classification the product substitutes where it does not attempt a
retrieval, and they come from the same MODIS land cover the retrieval itself is keyed on. They
are exposed as `:landcover_code`, read from the leaf-area layer and left unscaled, so a
surface-property closure can find the non-vegetated cells — and which kind — without a separate
land-cover product. Measured on two real granules:

| | central Amazon | Ozarks |
|---|---|---|
| urban / built-up (250) | 1.07% (Manaus) | 0.21% |
| barren / sparse (253) | 0.04% | — |
| perennial water (254) | 9.26% (Rio Negro–Solimões) | 0.03% |
| no class | 89.6% | 99.8% |

The quality screen does not apply to them: a cloudy urban pixel is still urban. `NaN` means
"no class" — either a retrieval is present, or the cell is fill.

These are **class codes**, so read them on the product's own grid. Interpolating them onto
another grid averages 250 against 254 into 252, which is a different class; a finer land-cover
product with a nearest-neighbour regrid is the right tool once the model grid differs.

## Quality screening

Two bit-packed bytes accompany every retrieval: `FparLai_QC` carries the retrieval
provenance and `FparExtra_QC` the scene state. The criteria they encode are exposed by name:

```julia
using NumericalEarth

recommended_lai_screening()                             # the default
lai_screening_mask(:other_quality, :cloudy, :snow_or_ice)  # build your own
lai_rejection_flags(qc, extra_qc)                       # what one pixel fails
```

Unlike the C3S adapter, screening is **on** by default:
`recommended_lai_screening()` keeps only good-quality pixels retrieved by the main
radiative-transfer algorithm under a clear or assumed-clear sky. Pass
`MCD15A2H(screened_flags = 0x0000)` to keep every retrieval the product marks as valid.

Snow, aerosol, cirrus, and cloud-shadow detections are deliberately *not* in the default.
Snow in particular is a physical state rather than a retrieval failure — it depresses winter
leaf area over evergreen needleleaf, which a downstream closure may want to see and handle
explicitly rather than receive as a gap.

## The seasonal climatology

A single 8-day composite loses a fifth of a good scene to cloud and most of a bad one, so it
is not usable as a boundary condition on its own. `MODISLAIClimatology` composites the same
period across years, which is where the coverage comes from: a period cloudy in one year is
usually clear in another.

```julia
climatology = MODISLAIClimatology(years = 2003:2019)

build_lai_climatology!(climatology; region)                     # mean
build_lai_climatology!(climatology; region, reducer = maximum)   # peak season
```

`years` defaults to 2003–2019, the span over which both Terra and Aqua held their equatorial
crossing times; Terra began drifting in 2020, which changes the composite's sensor
characteristics. `reducer` acts on the retained values of a pixel, so `maximum` (or a high
quantile) gives a peak-season field — often the more useful boundary condition for a
simulation spanning days rather than a year.

Each period's file also stores the number of retained retrievals behind every cell, read
through `retained_retrieval_metadatum`, so a composite's coverage can be mapped rather than
assumed. Cells no year could observe stay `NaN`.

The record has a few holes where an instrument outage prevented a composite altogether —
2016-02-18 is one. A climatology skips those years for the affected period and warns, since
a multi-year mean is exactly the reduction that tolerates a missing sample; reading that one
date on its own raises `MissingGranulesError`, because there is nothing to read.

## Annual land cover

`MCD12Q1` reads a per-cell class under one of three legends. All three are `uint8` with fill
255, and each has its own valid range — 0 is water in two of them and not a class at all in
IGBP, so the range is per legend rather than one shared constant.

| `legend` | layer | classes | table |
|---|---|---|---|
| `:IGBP` (default) | `LC_Type1` | 1–17 | `igbp_class_names` |
| `:LAI` | `LC_Type3` | 0–10 | `modis_lai_class_names` |
| `:PFT` | `LC_Type5` | 0–11 | `modis_plant_functional_type_names` |

IGBP is the default because the roughness literature's drag and stem-area tables are keyed on
it. `LC_Type3` is the stratification the MCD15 retrieval itself uses, which makes it the
closer match when the class field is there to pool leaf-area donors.

`QC` is an **enumerated** classification outcome (0 good classified land, 10 no data), not a
packed bitfield like `FparLai_QC` — bit-masking it produces nonsense, so there is no default
screen.

Class codes are not interpolable. Read them on the product's own grid, and take
`class_fractions` when a model grid is wanted:

```julia
classes = Field(Metadatum(:landcover_class; dataset = MCD12Q1(), region, date = DateTime(2019)))
codes = Array(interior(classes, :, :, 1))

fractions = class_fractions(codes, [4, 8, 9, 12], 4)   # four 500 m cells per model cell
```

`landcover_change_flag(annual_maps)` flags cells whose class is stable at each end of a span
and differs between them — a cell that was forest at the start of a climatology and pasture
at its end composites into one field describing neither. It is a flag, not a correction, and
the test is persistence rather than a first-versus-last difference, because a single year's
label at 500 m is not reliable enough to call a change on.

## Filling what compositing leaves behind

Compositing across years removes cloud that is random from one year to the next. Cloud that
recurs at the same calendar period every year — the ITCZ, a monsoon, a windward slope —
survives it, and survives interpolation along the seasonal axis too, because the neighbouring
periods are cloudy for the same reason. `fill_seasonal_gaps!` borrows the rest, using the
class field to decide how far and from whom:

```julia
classes = Field(Metadatum(:landcover_class; dataset = MCD12Q1(), region, date = DateTime(2019)))

Λ = FieldTimeSeries(Metadata(:leaf_area_index; dataset = climatology, region))

filled = fill_seasonal_gaps!(Λ, classes;
                             cyclic = true,
                             maximum_gap = class_maximum_gap(classes),
                             valid_range = (0, 10),
                             unfilled_classes = igbp_non_vegetated_classes)
```

Three stages run in order, and none rewrites an observation:

1. Interpolation along time, with `class_maximum_gap` setting the bridge length per cell —
   six periods over evergreen, where 48 days is nearly exact, and one over deciduous forest
   and cropland, where more than one period across green-up puts the ramp in the wrong place.
2. A same-class donor's seasonal *shape*, scaled to the cell's own level. This keeps the
   magnitude, which is what varies most in space, and borrows only the timing, which is what
   the class actually determines.
3. The donor curve itself, for cells with no valid period to scale by.

The donor pool is an expanding stencil over blocks of cells: a fixed few-kilometre
neighbourhood is empty exactly when it is needed, so the search grows its radius until it
finds enough same-class donors. `filled.reach` records how far each cell had to go, and
`filled.provenance` which stage produced each value.

With a climatology in hand, one specific year is the same chain with a better donor — that
cell's own climatological curve, which preserves its seasonal shape too and needs no class
field. There is no second dataset type:

```julia
Λ2019 = FieldTimeSeries(Metadata(:leaf_area_index; dataset = MCD15A2H(), region,
                                 dates = (DateTime(2019, 1, 1), DateTime(2019, 12, 31))))

fill_seasonal_gaps!(Λ2019, classes; anchor = Λ, cyclic = false)
```

`gap_fill_denial` scores any of this by withholding values the chain would otherwise have
kept and comparing the estimates against them, per class. It needs no new downloads.

What the chain leaves missing is the non-vegetated classes, and measured over both demo boxes
that is *all* of it — no vegetated cell runs out of donors. Those cells are not missing a
value: leaf area per unit ground area over water or tarmac is zero, and that is the value to
write.

```julia
zero_non_vegetated!(Λ, classes)     # after the fill, before the regrid
```

Do it in that order. A `NaN` at a shoreline dilates into its neighbours through the
interpolation stencil, while a zero blends into the cell mean correctly — leaf area is
already per unit ground area, so a model cell half lake and half forest genuinely carries
half the forest's. Score the fill before zeroing, never after: `gap_fill_denial` samples
cells that carry a value, and zeroed water would enter the sample as truth the chain
reproduces exactly.

Keep the class field alongside the result. Zero says there is no canopy; it does not say
whether the surface is a lake or a car park, and those want roughness lengths four orders of
magnitude apart.

An averaging cadence is a plain function on the filled series rather than another product.
The composites are window averages already and 8-day periods do not nest inside months, so
the samples straddling each edge are split by their days of overlap:

```julia
bounds = [metadata.dates; DateTime(2020, 1, 1)]
Λbimonthly, edges = time_average(Λ2019, bounds, Month(2))
```

## Usage

```julia
using NumericalEarth
using ArchGDAL
using Oceananigans
using Dates

region = BoundingBox(longitude = (-92.5, -91.5), latitude = (36.5, 37.5))

# One 8-day composite on its native grid
Λ = Field(Metadatum(:leaf_area_index; dataset = MCD15A2H(), region,
                    date = DateTime(2020, 7, 3)))

# The 46-period seasonal climatology on a model grid, cyclic in time
grid = LatitudeLongitudeGrid(size = (200, 200), longitude = (-92.5, -91.5),
                             latitude = (36.5, 37.5), topology = (Bounded, Bounded, Flat))

climatology = MODISLAIClimatology(years = 2003:2019)
build_lai_climatology!(climatology; region)

metadata = Metadata(:leaf_area_index; dataset = climatology, region)
Λ = FieldTimeSeries(metadata, grid)

# Fill what compositing could not, wrapping across the turn of the year
fill_gaps!(Λ; max_gap = 4, cyclic = true)
```

## Notes

- Composite dates are **year-anchored**: day-of-year 1, 9, 17, …, 361, restarting every
  January, so a year holds exactly 46 periods and the last one is short. Stepping uniformly
  by 8 days from the first date drifts out of phase after a year and requests composites that
  do not exist — `composite_dates` handles this.
- A composite is stamped at the **start** of its window, so the value it holds represents
  four days later. `native_times` applies that offset, which is also what makes the 46
  climatological stamps span exactly one year rather than 46 × 8 = 368 days — the period a
  cyclic series has to wrap on.
- Water and unretrieved pixels are `NaN`; there is no inpainting by default, as for every
  land dataset here.
- A longitude window straddling the ±180° seam is rejected rather than silently pulling the
  whole globe; split it into two requests.
- HDF4 defeats GDAL's `/vsicurl` windowed reads, so whole tiles are downloaded. At 500 m that
  is a few megabytes each, but a climatology multiplies quickly: 46 periods × `length(years)` ×
  the number of sinusoidal tiles the region spans. A 1.5° × 1.1° box straddling a tile boundary
  over three years is ~280 granules and takes tens of minutes. Budget for it, build the periods
  you need with the `periods` keyword while iterating, and note that the per-date files are
  cached and shared across periods and variables.
- The record is ongoing. `all_dates` advertises a conservative end; later composites can be
  requested by passing explicit `dates`.
