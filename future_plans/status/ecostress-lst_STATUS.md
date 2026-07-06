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
- `ext/NumericalEarthArchGDALExt.jl` (extended): gated `read_goes_lst_lonlat`
  (geostationary `+proj=geos` → EPSG:4326 warp of `LST`/`DQF`, then pure
  `goes_lst` decode) and `read_ecostress_l2g` (GDAL HDF5 driver → pure
  `ecostress_lst` decode). Loaded only under `using ArchGDAL`.
- `test/test_lst.jl` — auto-discovered by `find_tests`; no runner edit needed.
- Registration (Part D.3): `src/DataWrangling/DataWrangling.jl`
  (`include` + `using .LandSurfaceTemperature`) and `src/NumericalEarth.jl`
  (`using .DataWrangling.LandSurfaceTemperature`).

## What is gated (and why)

- **GOES anonymous S3 fetch** (`download_goes_granule`) and
  **geostationary→lat/lon reprojection** (`read_goes_lst_lonlat`) — need
  `ArchGDAL.jl` + network (S3 ListBucket to resolve the scan-time object key,
  `+proj=geos` warp). Module-level fallback `error(...)`; real impl in the ext.
- **ECOSTRESS HDF5 read** (`read_ecostress_l2g`) — `HDF5.jl` is NOT a project dep
  and must not be added (AGENTS.md rule 10), so routed via GDAL's HDF5 driver in
  the ArchGDAL ext; fallback errors otherwise.
- **ECOSTRESS Earthdata download** (`download_ecostress_granule`) and
  **irregular `all_dates` CMR discovery** (`ecostress_cmr_overpasses`) — need
  Earthdata credentials + network. `all_dates` returns a documented, unevenly
  spaced stub so the irregular-time `FieldTimeSeries` path can be exercised.

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

## Load / register status

`NumericalEarth` precompiles with the module; `GOES_LST()` and `ECOSTRESS_L2G()`
surface unqualified via `using NumericalEarth` and appear in
`supported_datasets()`.

## Next steps

- Wire `H_LST` into a live trajectory loss (Plan 07 P3): sample model `Tˡᵃ` at
  `t_obs`, accumulate over the rollout, extend `DatasetRestoring`'s pattern.
- Implement real GOES S3 object-key listing + ECOSTRESS CMR overpass discovery
  (network) and validate the ArchGDAL warp/HDF5 read paths against real granules.
- P4 diurnal compositing (local-hour binning, GOES shape-fill, clear-sky bias),
  and optional MODIS/VIIRS/Landsat anchor ingests.
