# C3S 300 m Leaf Area Index

This module ingests the Copernicus Climate Change Service (C3S) leaf-area-index climate
data record at 300 m: dekadal (10-daily) global NetCDF files on a regular 1/336°
latitude-longitude grid spanning 80°N–60°S.

Version 4.1 of the record — the 300 m collection — is retrieved from Sentinel-3 OLCI and
SLSTR broadband white-sky albedo with the Two-stream Inversion Package (TIP), so the
product is an **effective** leaf area index rather than a true one, and it is *not* the
Copernicus Global Land Service operational LAI (only version 0.0 of this catalogue entry
is brokered from CGLS). Coverage runs from July 2018 to December 2024.

Files are downloaded from the Copernicus Climate Data Store's
[`satellite-lai-fapar`](https://cds.climate.copernicus.eu/datasets/satellite-lai-fapar)
catalogue entry.

## Credentials

Downloads go through the CDS API, the same setup as ERA5:

1. Create an account at https://cds.climate.copernicus.eu
2. Put your API key in `~/.cdsapirc` (see https://cds.climate.copernicus.eu/how-to-api)
3. Accept the *Copernicus Global Land product licence* on the dataset page
4. Load the backend before downloading: `using CDSAPI`

## What gets stored locally

The global grid holds 120960 × 47040 = 5.7 billion cells per dekad, so a whole-globe file
is impractical to download. Instead the region is sent as the request's `area` and the
Climate Data Store subsets server-side; one file is stored per date *and* region, and the
filename is keyed by both. Requests are batched one per calendar month, and the area is
widened by a few native cells so the delivered file covers the whole native grid the data
is interpolated from.

A read hyperslabs exactly the cells of that native grid off disk. Two conventions of the
product matter and are handled here:

- The `lon`/`lat` coordinate variables label each cell's **west and south edge**, not its
  center — the half-cell shift implied by the file's `GeoTransform`. Centers are recovered
  on read.
- Latitude is stored **north→south** and flipped to ascending on read.

## Quality screening

Each file carries a `retrieval_flag` bitfield describing the TIP inversion, whose
single-bit diagnostics are exposed by name:

```julia
using NumericalEarth

retrieval_flag_mask(:obs_unusable, :tip_untrusted)   # build your own screen
unusable_retrieval_flags()                           # the recommended one
```

By default **no pixel is screened**, so what you read is what the product contains,
including its fill values (as `NaN`). Note that this record is not gap-free: it has gaps
over equatorial and high-latitude regions, and reports zero LAI over sparsely vegetated
targets in the leaf-off season. Pass `screened_flags` to drop the pixels whose retrieval
the product itself marks as unusable:

```julia
dataset = CopernicusVegetation(screened_flags = unusable_retrieval_flags())
```

Build a screen by measuring the bits, not by reading their names. Over a mid-latitude
July scene, `:obs_inconsistent` is set on ~95% of pixels and `:obs_nosnow_only` on
~100% — both describe how the inversion was done rather than disqualifying it, so
screening on either discards nearly the whole field. The three bits in
`unusable_retrieval_flags()` remove ~0.02% of that same scene; adding
`:obs_nosnow_hiunc` takes it to ~14%.

## Usage

```julia
using NumericalEarth
using CDSAPI
using Oceananigans
using Dates

region = BoundingBox(longitude = (-92, -91), latitude = (37, 38))

# A single dekad on its native grid
Λ = Field(Metadatum(:leaf_area_index; dataset = CopernicusVegetation(),
                    region, date = DateTime(2021, 7, 20)))

# A seasonal series on a model grid, cyclic in time
grid = LatitudeLongitudeGrid(size = (96, 96), longitude = (-92, -91), latitude = (37, 38),
                             topology = (Bounded, Bounded, Flat))

metadata = Metadata(:leaf_area_index; dataset = CopernicusVegetation(), region,
                    dates = (DateTime(2021, 1, 1), DateTime(2021, 12, 31)))

Λ = FieldTimeSeries(metadata, grid)
```

## Notes

- Ocean and unretrieved pixels are `NaN` (no inpainting by default — this is a land dataset).
- A longitude window straddling the ±180° seam is rejected rather than silently pulling the
  whole globe; split it into two requests.
- The same files also carry `LAI_ERR`, the retrieval's standard deviation, and the
  catalogue entry serves fAPAR from the same request — neither is exposed yet.
