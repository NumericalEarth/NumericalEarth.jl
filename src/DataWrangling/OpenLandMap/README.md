# OpenLandMap-soilDB

This module ingests the OpenLandMap-soilDB global soil properties at 30 m
resolution (Hengl et al., 2026, *ESSD* 18:989, CC-BY 4.0), predicted from
spatiotemporal machine learning over Landsat/MODIS/Sentinel covariates. It
delivers texture mass fractions (`:sand_fraction`, `:silt_fraction`,
`:clay_fraction`, in kg/kg) and fine-earth `:bulk_density` (kg/m³) over the three
native depth intervals 0–30, 30–60, and 60–100 cm, stored as a three-dimensional
field whose vertical axis carries the depths (deepest first). Data are plain
geographic (EPSG:4326), so no reprojection is needed. Coverage spans latitudes
−56° to 76° (Antarctica excluded).

Data source: https://stac.openlandmap.org/

## Loading

A few things are specific to this dataset:

1. **No credentials.** The cloud-optimized GeoTIFFs are read anonymously.

2. **`using ArchGDAL` is required.** The windowed GeoTIFF reader lives in the
   ArchGDAL extension, so load ArchGDAL before materializing a `Field`.

3. **Regional windows only.** The global grid is ~1.44M × 528k cells, so the
   dataset must be read through a longitude/latitude `BoundingBox` — constructing
   a `Metadatum` without a bounded `region` errors.

## Usage

```julia
using NumericalEarth
using ArchGDAL   # activates the windowed cloud-optimized-GeoTIFF reader

region = BoundingBox(longitude = (-112.3, -111.9), latitude = (36.0, 36.4))

# Native 30 m field: horizontal window × three depth intervals
clay = Field(Metadatum(:clay_fraction; dataset = OpenLandMapSoilDB(), region))

# Interpolate onto your own grid by materializing on it directly
grid = LatitudeLongitudeGrid(size = (400, 400, 3),
                             longitude = region.longitude, latitude = region.latitude,
                             z = [-1.0, -0.6, -0.3, 0.0], halo = (3, 3, 3))
clay_on_grid = Field(Metadatum(:clay_fraction; dataset = OpenLandMapSoilDB(), region), grid)
```

## Notes

- Masked cells — permanent ice, sand deserts, water — carry `NaN`. Pass
  `inpainting` to `Field` to fill them.
- Downloaded windows are cached like every other dataset (see
  `NUMERICALEARTH_DATA_DIRECTORY`); a per-`Metadatum` `dir` overrides the default.
