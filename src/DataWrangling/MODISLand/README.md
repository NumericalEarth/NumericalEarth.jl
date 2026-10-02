# MODIS 500 m leaf area index and land cover

This module ingests two MODIS V061 land products and builds seasonal climatologies from
the first:

- **MCD15A2H** — combined Terra + Aqua leaf area index / FPAR: 500 m, 8-day composites
  from 2002-07-04. Variables `:leaf_area_index`, `:fpar`, `:leaf_area_index_uncertainty`,
  and `:landcover_code`.
- **MCD12Q1** — annual land cover from 2001 under the `:IGBP` (default), `:LAI`, or `:PFT`
  legend, plus `:quality_flag` and `:land_water_mask`. Same granules, same reprojected
  lattice, so a class field and a leaf-area field pair cell for cell.

The retrieval targets *true* leaf area index (per-biome clumping applied inside the
product) — the quantity a canopy closure calibrated on MODIS expects.

## Credentials

Downloads need a free NASA Earthdata login and GDAL's HDF4 driver:

1. Register at https://urs.earthdata.nasa.gov
2. Export `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`, or add a `~/.netrc` entry for
   `urs.earthdata.nasa.gov`
3. Load the backend before downloading: `using ArchGDAL`

Granule discovery goes through NASA's Common Metadata Repository and is anonymous; only
the granule fetch is authenticated.

## What gets stored locally

Granules are HDF-EOS2 tiles on the sinusoidal projection. At download time every granule
covering the region and date is fetched, and all layers are mosaicked and reprojected
(nearest-neighbor) in one `gdalwarp` call onto the region's window of a global 1/240°
lattice — one NetCDF per date and region holding every layer as raw digital numbers.
Decoding (fill rejection, scale factors, quality screening) happens at read time. A
lon/lat `BoundingBox` region is required; a global read is rejected.

The granules themselves are kept under `granules/` beside the reprojected files, keyed by
granule name, so a new region or date over the same tiles warps from local files.

## Usage

```julia
using NumericalEarth, ArchGDAL, Oceananigans, Dates

region = BoundingBox(longitude = (-92.5, -91.5), latitude = (36.5, 37.5))

# One 8-day composite and the class map, on the product's grid
leaf_area_index = Field(Metadatum(:leaf_area_index; dataset = MCD15A2H(), region,
                                  date = DateTime(2020, 7, 3)))
classes = Field(Metadatum(:landcover_class; dataset = MCD12Q1(), region,
                          date = DateTime(2019)))

# The 46-period seasonal climatology, cyclic in time
climatology = MODISLAIClimatology(years = 2003:2019)
build_lai_climatology!(climatology; region)
metadata = Metadata(:leaf_area_index; dataset = climatology, region)
leaf_area_index = FieldTimeSeries(metadata; time_indices_in_memory = length(metadata.dates))
```

`fill_seasonal_gaps!` with `class_maximum_gap` and `zero_non_vegetated!` fills what
compositing leaves behind — see their docstrings and `examples/modis_landcover_gap_fill.jl`.

## Notes

- Composite dates are year-anchored (day-of-year 1, 9, …, 361): 46 periods per year, the
  last one short. `modis_composite_dates` generates them; a uniform 8-day step drifts out
  of phase after a year.
- Class codes are not interpolable: read them on the product's own grid, and take
  `class_fractions` when a model grid is wanted.
- Water and unretrieved pixels are `NaN`; no inpainting by default.
- A longitude window straddling the ±180° seam is rejected; split it into two requests.
- Whole tiles are downloaded (HDF4 defeats GDAL's windowed reads), and a climatology
  multiplies quickly: 46 periods × `length(years)` × tiles — a 1° box over five years is
  ~230 granules and tens of minutes. Per-date files and the granules behind them are cached
  and shared across periods and variables; build only the periods you need with the
  `periods` keyword while iterating.
- The record has outage holes (2016-02-18 is one): a climatology skips them with a warning,
  a single-date read raises `MissingGranulesError`.
- `years` defaults to 2003–2019, the span over which Terra and Aqua held their equatorial
  crossing times.
