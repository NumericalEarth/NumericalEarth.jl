# MODIS land datasets — implementation status

Branch `xk/modis`. Implements the shared `MODISLand` submodule (infra C.3) with
three dataset structs on top of a common SIN-grid / CMR / Earthdata / decode
scaffold.

## Implemented

- **`src/DataWrangling/MODISLand/MODISLand.jl`** — the submodule.
  - **Pure decode/mask/blend/aggregation (the core deliverable, no IO/creds):**
    `decode_albedo` (fill 32767→NaN *before* ×0.001), `bluesky_blend(α_bs, α_ws, f_diff)`,
    `albedo_quality_ok`, `decode_lai` / `decode_fpar` (mask DN>100 *before* scaling),
    `lai_quality_ok` (MODLAND_QC==0 & SCF_QC∈{0,1}), `mask_landcover`,
    `mode_aggregate` / `class_fraction` (categorical — never averages codes).
  - **SIN grid math (pure):** `sinusoidal_tile_bounds`, `sinusoidal_to_longitude_latitude`,
    `longitude_latitude_to_sinusoidal`, and the `cmr_granules_url` builder (W,S,E,N).
  - **Datasets:** `MCD43Albedo(; diffuse_fraction=0.2)`, `MCD15A3H`/`MCD15A2H`/`MOD15A2H`
    (`<: MODISLAIDataset`), `MCD12Q1(; legend=:PFT)`. All `<: AbstractMODISLandDataset
    <: AbstractStaticDataset`.
  - **Interface (per Part D.2):** `is_three_dimensional=false`, `default_inpainting=nothing`,
    `location=(Center,Center,Center)`, `longitude_name/latitude_name="lon"/"lat"`,
    global lat/lon hull + nominal 500 m `Base.size`, `validate_dataset_coverage` requiring a
    `BoundingBox` (CopernicusDEM error style), region+date-encoded `metadata_filename`.
  - **`retrieve_data`** reads the reprojected regional NetCDF of *raw DN* and applies the pure
    decode/blend/QA (albedo blends BSA+WSA; LAI masks via `FparLai_QC` if present; PFT masks fill).
  - **`Downloads.download`** guards `@root if !isfile` and calls `modis_granules_to_netcdf`, which
    **falls back to a clear `error(...)`** when the ArchGDAL extension is not loaded.
- **`ext/NumericalEarthArchGDALExt.jl`** — real read path (mirrors `reproject_ibcao_to_netcdf`):
  `earthdata_cmr_granules` (CMR query → `.hdf` URLs), `earthdata_download` (Earthdata `.netrc`
  via `EARTHDATA_USERNAME`/`EARTHDATA_PASSWORD`), `modis_subdataset` (find HDF4-EOS subdataset by
  layer suffix), `modis_granules_to_netcdf` (download tiles → `gdalwarp` SIN→EPSG:4326 clipped to
  bbox, `near` for categorical / `bilinear` otherwise → regional raw-DN NetCDF).
- **Registration:** `include`+`using .MODISLand` in `src/DataWrangling/DataWrangling.jl`;
  `using .DataWrangling.MODISLand` in `src/NumericalEarth.jl`.
- **Tests:** `test/test_modisland.jl` (auto-discovered by `find_tests`).
- **Docs:** `jldoctest` on the `MCD43Albedo`, `MCD15A3H`, `MCD12Q1` constructors (exercise `show`).

## Gated / stubbed (and why)

- **Real granule reads** (`modis_granules_to_netcdf`, `earthdata_cmr_granules`) live in the
  ArchGDAL extension and require: (1) `using ArchGDAL`, (2) NASA Earthdata credentials,
  (3) network, and (4) **`GDAL_jll` built with the HDF4 driver** (GDAL.jl #84) to open
  `HDF4_EOS:EOS_GRID:` subdatasets. When any are absent the module entry point errors with a
  copy-pasteable message. Extension code is written to spec but **untested end-to-end** (no creds
  or HDF4 driver here) — the Tier-1 AppEEARS-preprocessed-NetCDF path (infra B.3) is the
  documented fallback and reads through the same `retrieve_data`.
- **FTS / seasonal LAI** — deliberately static single-composite only (task scope); FTS is a later
  refinement. LAI datasets are `AbstractStaticDataset` for now (pass a `date`).
- **Categorical bilinear caveat** — the standard `Field(metadatum, grid)` interpolation is
  bilinear. For production PFT use, warp with `-r near` (done in the extension) and/or apply
  `mode_aggregate`/`class_fraction` when coarsening; helpers are shipped + tested.
- **No `[deps]` changes** — routed through the existing ArchGDAL weakdep + NCDatasets (no HDF5.jl).

## Verification (what was run)

- Pure helpers in an isolated module: all 12 checks `true`.
- `julia --project=.` (worktree): `Pkg.instantiate()` OK; `using NumericalEarth` **precompiled
  cleanly** with the new module (Project.toml/Manifest.toml unchanged).
- Constructors show as `MCD43Albedo(0.2)`, `MCD12Q1(:PFT)`, `MCD15A3H()` (match the jldoctests).
- `include("test/test_modisland.jl")` → **all pass**: 21 + 10 + 10 + 4 + 27 + 3 = **75 tests, 0 fail**.
- ArchGDAL not installable here (weakdep), so the extension was verified to **parse** cleanly
  (`Meta.parseall`, 210 lines) but not precompiled/executed.

## Next steps to make real downloads work

1. `using ArchGDAL` in a session with `GDAL_jll` that has the HDF4 driver (verify:
   `"HDF4" in ArchGDAL.listdrivers()`), else use the Tier-1 AppEEARS NetCDF path.
2. Set `EARTHDATA_USERNAME`/`EARTHDATA_PASSWORD` (or a `~/.netrc` entry for
   `urs.earthdata.nasa.gov`).
3. `Field(Metadatum(:albedo; dataset=MCD43Albedo(), region=BoundingBox(...)), grid)` and confirm a
   sane 0–1 field; likewise `:leaf_area_index` (0–~7) and `:plant_functional_type` (0–11).
4. Add a token/HDF4-gated CI job (like the CopernicusDEM downloading test) exercising one bbox.
5. Then: seasonal LAI `FieldTimeSeries`, MCD43A1 solar-angle BSA, and the PFT→prior crosswalk.
