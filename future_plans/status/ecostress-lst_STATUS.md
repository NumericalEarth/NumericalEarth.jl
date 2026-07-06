# Status — high-resolution LST supervision targets (GOES-R + ECOSTRESS)

Branch `xk/ecostress-lst`. Implements Plan 07 Half A (ingest) + an `H_LST`
observation-operator scaffold. LST is a **training target**, wired toward the
loss / observation operator — never the radiation/flux slots.

## Files & functions

- `src/DataWrangling/LandSurfaceTemperature/LandSurfaceTemperature.jl` (new module)
  - **Pure decode/parse core (no IO, no credentials — main deliverable):**
    - `goes_lst(DN)` / `goes_lst(DN, DQF)` — `K = 0.0025·DN + 190.0`; fill `65535`,
      out-of-range `[213,343]` K, and `DQF ≥ 3` → `NaN`.
    - `ecostress_lst(LST, cloud)` — float32 K passthrough, `NaN` passthrough,
      nonzero `cloud` → `NaN`.
    - `granule_timestamp(name)` — parses `sYYYYDDDThhmmss` (day-of-year) → `DateTime`.
    - `lst_masked_residual(T_model, LST_obs, LST_err, cloudy)` — cloud/QC mask and
      NaN/err≤0 guard applied **before** the variance-normalized residual
      `(T_model−LST_obs)/LST_err`; masked pixels contribute `0`.
  - **Datasets:** `GOES_LST(; satellite=:goes16) <: AbstractGOESDataset` (regular
    hourly `all_dates`); `ECOSTRESS_L2G(; version="002") <: AbstractECOSTRESSDataset`
    (**irregular** `all_dates` = actual overpass timestamps). Both zero-arg
    constructible → appear in `supported_datasets()` (verified).
  - **Interface (Part D.2):** `is_three_dimensional=false`,
    `default_inpainting=nothing` (cloud gaps are the operator's valid mask — NOT
    inpainted), `missing_value=NaN`, region-encoded `metadata_filename`,
    `validate_dataset_coverage` requiring a `BoundingBox`, `available_variables`
    (`:land_surface_temperature→"LST"`, ECOSTRESS also `:lst_uncertainty→"LST_err"`,
    `:cloud_mask→"cloud"`), `location=(Center,Center,Center)`, `eltype=Float32`.
  - **Operator scaffold:** `LSTObservationOperator` struct + `show` + docstring
    describing steps 1–4 (sample model `Tˡᵃ(t_obs)`, apply cloud mask, form
    variance-normalized residual, feed loss).
- `ext/NumericalEarthArchGDALExt.jl` (extended): the **real end-to-end fetch**,
  loaded under `using ArchGDAL`. Both products are downloaded, decoded (via the
  pure `goes_lst`/`ecostress_lst` core), and reprojected/clipped to a clean
  regional lat/lon NetCDF (`lon`, `lat`, `LST` in Kelvin with `NaN` gaps) at
  download time — mirroring the proven MODISLand ingest — so the generic
  `Field`/`set_region_data!` machinery brackets the raster onto the native grid:
  - `goes_granule_to_netcdf` — anonymous S3 `ListObjectsV2` of
    `noaa-goes16|18|19/ABI-L2-LSTC/YYYY/DDD/HH/`, picks the object nearest the
    requested hour, downloads it, warps the geostationary `+proj=geos` `LST`/`DQF`
    subdatasets → EPSG:4326 (nearest, 0.02°) clipped to the bbox, applies the pure
    `goes_lst(DN, DQF)` decode.
  - `ecostress_cmr_overpasses` / `ecostress_granule_to_netcdf` — CMR granule
    discovery for `ECO_L2G_LSTE` (irregular overpass timestamps parsed from the
    granule ids), Earthdata-authenticated download via `netrc_downloader`
    (`urs.earthdata.nasa.gov`), GDAL HDF5-driver read of `…/Data_Fields/LST` +
    `…/cloud`, 0-fill → `NaN`, pure `ecostress_lst` cloud-mask decode, clip to bbox
    (nearest, 0.0007°).
- `test/test_lst.jl` — auto-discovered by `find_tests`; no runner edit needed.
- Registration (Part D.3): `src/DataWrangling/DataWrangling.jl`
  (`include` + `using .LandSurfaceTemperature`) and `src/NumericalEarth.jl`
  (`using .DataWrangling.LandSurfaceTemperature`).

## What is gated (and why)

Real IO lives in the ArchGDAL extension; the module keeps clear fallback
`error(...)`s (mirroring `CopernicusDEM.zarr_to_netcdf`) so an environment without
`ArchGDAL` / network / credentials fails loudly rather than silently:

- **GOES fetch + reprojection** (`goes_granule_to_netcdf`) — the module stub errors
  unless `using ArchGDAL` (needs GDAL's netCDF driver + the `+proj=geos` warp and
  anonymous network access). No credentials required.
- **ECOSTRESS HDF5 read** (`ecostress_granule_to_netcdf`) — `HDF5.jl` is NOT a
  project dep and must not be added (AGENTS.md rule 10), so it is routed via GDAL's
  HDF5 driver in the ArchGDAL ext.
- **ECOSTRESS Earthdata download + CMR discovery** — need `EARTHDATA_USERNAME` /
  `EARTHDATA_PASSWORD` (used through `netrc_downloader`) and network. The
  region-less `all_dates(::ECOSTRESS_L2G)` still returns a documented, unevenly
  spaced stub (the irregular-time axis exists per region+window, discovered by
  `ecostress_cmr_overpasses`, which needs a `BoundingBox`).

## Operator-scaffold status

`lst_masked_residual` (the arithmetic of steps 2–3) is implemented and tested.
`LSTObservationOperator` bundles the ingredients (irregular-time obs FTS, cloud
mask, `LST_err`, model variable, rate) with a docstring. **Not yet wired** to a
live model loss: sampling the running model at `t_obs`, trajectory accumulation,
and extending `restoring.jl`'s masked-residual machinery to a land `Tˡᵃ`
variable are documented next steps (Plan 07 P3).

## Verbatim test results

1. `julia --project=. -e 'using Pkg; Pkg.instantiate()'` → `instantiated OK`;
   `Precompiling packages... 20431.4 ms ✓ NumericalEarth` (compiled clean with
   the new module registered).
2. Standalone pure-logic script (`scratchpad/verify_lst_pure.jl`, decode/parse/
   operator copied verbatim):
   `pure GOES 7/7`, `pure ECOSTRESS 3/3`, `pure timestamp 3/3`,
   `pure operator 4/4` — all pass.
3. Live `using NumericalEarth` + LST interface testset: **20 real assertions
   pass** (dataset `show`, `available_variables`, `dataset_variable_name`,
   `is_three_dimensional=false`, `missing_value=NaN`, `default_inpainting=nothing`,
   region-encoded filenames, GOES regular vs ECOSTRESS irregular `all_dates`,
   `validate_dataset_coverage` requires BoundingBox, gated `retrieve_data`
   errors clearly, operator kernel, `show`). Both datasets confirmed present in
   `supported_datasets()` (as types).
   - Note: the committed `test/test_lst.jl` follows the repo convention
     (`include("runtests_setup.jl")`, which pulls in CUDA/etc.); the assertions
     above were run directly against a loaded `NumericalEarth` to avoid the
     heavier setup. Parse-checked all three files with `Meta.parseall` (OK).
4. Whitespace/EOF: no trailing whitespace, single trailing newline in all files.
5. Full `test/test_lst.jl` (`julia --project=test test/test_lst.jl`): **52/52
   pass** across all 7 testsets (GOES decode 9, ECOSTRESS decode 5, timestamp 3,
   operator kernel 6, GOES interface 12, ECOSTRESS interface 11, bounded-region 6)
   after the download/read refactor.

## Live end-to-end verification (real data)

Both datasets now **load real data end-to-end** through `Field(Metadatum(...), grid)`
with `using ArchGDAL`. Environment: `julia --project=test`, macOS/arm64, GDAL with
netCDF + HDF5 + HDF4 drivers confirmed present. Commands prefixed with
`source ~/.zshrc` so the non-interactive shell picks up `EARTHDATA_*`.

### GOES-R (anonymous S3 — the priority, fully landed)

Command (verbatim script `scratchpad/test_goes_field.jl`):

```julia
using NumericalEarth, Oceananigans, ArchGDAL, Dates
using NumericalEarth.DataWrangling.LandSurfaceTemperature
grid = LatitudeLongitudeGrid(CPU(); size = (40, 40),
                             longitude = (-105, -100), latitude = (35, 40),
                             topology = (Bounded, Bounded, Flat))
region = BoundingBox(longitude = (-105, -100), latitude = (35, 40))
md = Metadatum(:land_surface_temperature; dataset = GOES_LST(satellite = :goes16),
               region, date = DateTime(2023, 1, 1, 0))
field = Field(md, grid)
```

Outcome (`julia --project=test scratchpad/test_goes_field.jl`):

```
GOES field size (40, 40, 1)
finite count 861 / 1600
has NaN (cloud/off-disk) true
K range 266.2389831542969 .. 286.35552978515625
in [213,343] true
GOES_OK
```

- **Real Field?** Yes — anonymous `ListObjectsV2` resolved the granule
  `OR_ABI-L2-LSTC-M6_G16_s20230010001173_…nc`, downloaded it (~2.4 MB), warped
  `geos → EPSG:4326`, decoded, and interpolated onto the CONUS grid.
- **K range?** 266.2–286.4 K, all within the physical `[213, 343]` K window.
- **Cloud NaN?** Yes — 739/1600 cells are `NaN` (off-disk / cloud / no-retrieval),
  not inpainted.

### ECOSTRESS (Earthdata-gated — also landed)

Command (verbatim script `scratchpad/test_ecostress_field.jl`):

```julia
using NumericalEarth, Oceananigans, ArchGDAL, Dates
using NumericalEarth.DataWrangling.LandSurfaceTemperature
using NumericalEarth.DataWrangling.LandSurfaceTemperature: ecostress_cmr_overpasses
region = BoundingBox(longitude = (-101.0, -100.0), latitude = (33.5, 34.5))
overpasses = ecostress_cmr_overpasses(region, DateTime(2021, 7, 1), DateTime(2021, 7, 3))
grid = LatitudeLongitudeGrid(CPU(); size = (30, 30),
                             longitude = (-101.0, -100.0), latitude = (33.5, 34.5),
                             topology = (Bounded, Bounded, Flat))
md = Metadatum(:land_surface_temperature; dataset = ECOSTRESS_L2G(),
               region, date = DateTime(2021, 7, 1, 8, 27, 49))
field = Field(md, grid)
```

Outcome (`source ~/.zshrc; julia --project=test scratchpad/test_ecostress_field.jl`):

```
CMR overpasses found: 5
first few: [DateTime("2021-07-01T08:26:57"), DateTime("2021-07-01T08:27:49"), DateTime("2021-07-01T08:28:41"), DateTime("2021-07-02T07:40:14")]
ECOSTRESS field size (30, 30, 1)
finite count 605 / 900
has NaN (cloud/off-swath) true
K range 286.8515625 .. 297.5313720703125
ECOSTRESS_OK
```

- **Real Field?** Yes — CMR returned 5 irregular overpasses; the nearest granule
  `ECOv002_L2G_LSTE_16928_005_20210701T082749_…h5` (~226 MB) was downloaded with
  Earthdata credentials via `netrc_downloader`, read through GDAL's HDF5 driver
  (`…/Data_Fields/LST` + `…/cloud`), and interpolated onto the grid.
- **K in Kelvin?** Yes — 286.9–297.5 K (nighttime summer, Texas panhandle).
- **Cloud NaN?** Yes — 295/900 cells are `NaN` (0-fill off-swath + cloud), not
  inpainted.

## Load / register status

`NumericalEarth` precompiles with the module; `GOES_LST()` and `ECOSTRESS_L2G()`
surface unqualified via `using NumericalEarth` and appear in
`supported_datasets()`.

## Next steps

- Wire `H_LST` into a live trajectory loss (Plan 07 P3): sample model `Tˡᵃ` at
  `t_obs`, accumulate over the rollout, extend `DatasetRestoring`'s pattern.
- Wire `all_dates(::ECOSTRESS_L2G)` (currently a region-less stub) to
  `ecostress_cmr_overpasses` when a region+window is available, so an
  irregular-time `FieldTimeSeries` builds its axis from real overpasses.
- P4 diurnal compositing (local-hour binning, GOES shape-fill, clear-sky bias),
  and optional MODIS/VIIRS/Landsat anchor ingests.

(Real GOES S3 object-key listing + ECOSTRESS CMR discovery and the ArchGDAL
warp/HDF5 read paths are now implemented and validated against real granules —
see "Live end-to-end verification" above.)
