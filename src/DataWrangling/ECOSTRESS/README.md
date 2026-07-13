# ECOSTRESS land surface temperature

This module ingests the ECOSTRESS `ECO_L2G_LSTE` gridded Land Surface Temperature
product: ~70 m thermal-infrared skin temperature from the radiometer on the ISS,
distributed as HDF5 on a geographic (EPSG:4326) lon/lat grid. It supplies the
high-resolution, all-hours diurnal `Tˡᵃ` supervision target a ~100 m
atmosphere-coupled LES needs.

LST is a *training target*, not a boundary-condition grab, so cloud and
no-retrieval gaps are the observation operator's valid mask and are never
inpainted (`default_inpainting = nothing`, `missing_value = NaN`).

## Irregular overpasses

The ISS orbit precesses, so ECOSTRESS overpasses are opportunistic — there is no
regular date range. Discover the overpasses intersecting a region and time window
from NASA's Common Metadata Repository (no credentials needed), then pass them as
explicit dates:

```julia
using NumericalEarth
using Dates

region = BoundingBox(longitude = (-101, -100), latitude = (33.5, 34.5))
dates = ecostress_overpasses(region, DateTime(2021, 7, 1), DateTime(2021, 7, 3))
```

## Credentials

The granule download is authenticated against NASA Earthdata Login. Set the
`EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` environment variables (the same ones
the `earthaccess` library uses):

1. Register (free) at https://urs.earthdata.nasa.gov
2. Approve the *LP DAAC Data Pool* / *LP CLOUD* application in your Earthdata profile
3. Export the credentials before launching Julia:

```bash
export EARTHDATA_USERNAME=your_username
export EARTHDATA_PASSWORD=your_password
```

or within Julia:

```julia
ENV["EARTHDATA_USERNAME"] = "your_username"
ENV["EARTHDATA_PASSWORD"] = "your_password"
```

## Usage

The HDF5 read routes through GDAL's HDF5 driver, so `ArchGDAL` must be loaded.
Each overpass is fetched, decoded to Kelvin (with `NaN` cloud/no-retrieval gaps),
and clipped to a compact regional lon/lat NetCDF at download time; the generic
`Field` / `FieldTimeSeries` machinery then windows it onto your grid.

```julia
using NumericalEarth
using ArchGDAL
using Dates

region = BoundingBox(longitude = (-101, -100), latitude = (33.5, 34.5))

# Single overpass
metadatum = Metadatum(:land_surface_temperature; dataset = ECOSTRESSL2G(),
                      region, date = DateTime(2021, 7, 1, 8, 27, 49))
lst = Field(metadatum)

# Irregular-time series over all overpasses in a window
dates = ecostress_overpasses(region, DateTime(2021, 7, 1), DateTime(2021, 7, 8))
metadata = Metadata(:land_surface_temperature; dataset = ECOSTRESSL2G(), dates, region)
lst_series = FieldTimeSeries(metadata)
```

## Notes

- Must be used with a longitude/latitude `BoundingBox`; the swath footprint is
  windowed at download time.
- `:land_surface_temperature` (`LST`) and `:lst_uncertainty` (`LST_err`) are read
  from one downloaded file per overpass; both are in Kelvin with `NaN` gaps.
