# Copernicus Global Land Surface Albedo

This module ingests the Copernicus Global Land Service (CGLS) 1 km surface-albedo
collection (version 2): dekadal (10-daily) global NetCDF files on a regular 1/112°
latitude-longitude grid spanning 80°N–60°S, derived from SPOT/VGT (1998–2014) and
PROBA-V (2014–2020) observations.

The files are downloaded from the Copernicus Climate Data Store's
[`satellite-albedo`](https://cds.climate.copernicus.eu/datasets/satellite-albedo)
catalogue entry (the C3S redistribution of the CGLS product — the legacy VITO
distribution server was retired when CLMS data moved to the Copernicus Data Space
Ecosystem in September 2025, and the 1 km albedo archive is not served there).

## Credentials

Downloads go through the CDS API, the same setup as ERA5:

1. Create an account at https://cds.climate.copernicus.eu
2. Put your API key in `~/.cdsapirc` (see https://cds.climate.copernicus.eu/how-to-api)
3. Accept the *Copernicus Global Land product licence* on the dataset page
4. Load the backend before downloading: `using CDSAPI`

## What gets stored locally

For each dekad, the extension downloads the black-sky (`albb_dh`,
directional-hemispherical) and white-sky (`albb_bh`, bi-hemispherical) broadband
products — one CDS request per calendar month — and repacks the pair into a single
compact local NetCDF (variables `AL_DH_BB` and `AL_BH_BB`, packed integers with CF
attributes preserved). The blue-sky (actual) albedo is blended at read time with the
dataset's `diffuse_fraction` `f`:

    α = (1 − f) α_bs + f α_ws

## Usage

```julia
using NumericalEarth
using CDSAPI
using Dates

# Single dekad over a region
region = BoundingBox(longitude = (-114, -111), latitude = (35, 37))
metadatum = Metadatum(:albedo; dataset = CopernicusAlbedo(), region, date = DateTime(2019, 7, 10))
α = Field(metadatum)

# 12-month climatology FieldTimeSeries (builds monthly means on first use)
metadata = Metadata(:albedo; dataset = CopernicusAlbedoClimatology(years = 2019:2019), region)
albedo_climatology = FieldTimeSeries(metadata)
```

`build_monthly_climatology!(dataset)` precomputes the monthly-mean files explicitly
(useful to control when the dekadal downloads happen); months are cached and skipped
on rebuild.

## Notes

- Ocean and unretrieved pixels are `NaN` (no inpainting by default — this is a land dataset).
- Downloads are global and shared across regions; regional windowing happens at read
  time via the `BoundingBox` machinery.
