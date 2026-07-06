# ESA WorldCover ingestion — status

Implements plan 05 (ESA WorldCover land cover / `f_veg`) on branch `xk/worldcover`.

## Files / functions

- `src/DataWrangling/ESAWorldCover/ESAWorldCover.jl` — new dataset module.
  - `struct ESAWorldCover <: AbstractStaticDataset` with `version::Symbol`;
    keyword constructor `ESAWorldCover(; version = :v200)` (`:v200`=2021 default,
    `:v100`=2020). Custom `Base.show` → `ESAWorldCover(version = :v200)`.
  - Class legend constants (enumerated, non-uniform steps): `ESA_WORLDCOVER_CLASS_CODES`
    `(10,20,30,40,50,60,70,80,90,95,100)`, `ESA_WORLDCOVER_CLASS_NAMES` (name→code
    NamedTuple), `ESA_WORLDCOVER_VEGETATED_CLASSES` `(10,20,30,40,90,95)` (exposed —
    modeling choice), `ESA_WORLDCOVER_MISSING_VALUE = 0`.
  - **Pure, IO-free aggregation helpers (main deliverable):** `mode_aggregate(codes)`
    (majority, ignores 0, ties→smaller code, all-0→0), `class_fraction(codes, c)`
    (`count(==c)/count(!=0)`), `class_fractions(codes)` (NamedTuple, sums to 1 over
    valid), `vegetation_fraction(codes; vegetated_classes=…)`, and
    `aggregate_blockwise(codes, factor, reduction)` (integer-factor block reduction,
    rejects non-divisible sizes).
  - Interface per Part D.2: `available_variables` (`:landcover_class`,
    `:landcover_fractions`, `:vegetation_fraction` → all `"Map"`),
    `default_download_directory`, `longitude_interfaces=(-180,180)`,
    `latitude_interfaces=(-60,84)`, `Base.size=(36000,14400,1)` (global at ~0.01°
    aggregated resolution, factor 120 over the 10 m native step),
    region-encoded `metadata_filename`, `validate_dataset_coverage` (requires a
    `BoundingBox`), `is_three_dimensional=false`, `dataset_variable_name="Map"`,
    `longitude_name="lon"`/`latitude_name="lat"`, `default_inpainting=nothing`,
    `missing_value=0`, `location=(Center,Center,Center)`.
  - `Downloads.download` guards with `@root` and calls the extension entry point
    `worldcover_cog_to_netcdf`, which module-level falls back to a clear
    `error(...)` when ArchGDAL is not loaded. Custom `retrieve_data` reads the
    materialized NetCDF band; `:landcover_fractions` errors with guidance (it is a
    per-class NamedTuple product, not a single Field).
- `ext/NumericalEarthArchGDALExt.jl` — extended with the anonymous COG read
  (`worldcover_cog_to_netcdf` + helpers `worldcover_tile_url`, `worldcover_tile_label`,
  `worldcover_tiles`). Reads `/vsis3/esa-worldcover/...` with
  `AWS_NO_SIGN_REQUEST=YES` (region `eu-central-1`), builds a VRT over the 3°×3°
  tiles intersecting the bbox, `gdalwarp`s the raw `Map` codes onto the snapped
  native window with **nearest** resampling (never averages codes), then aggregates
  by the integer factor via the pure helpers and writes `lon`/`lat` + bands
  `landcover_class`, `vegetation_fraction`, and per-class `fraction_<name>`.
- `test/test_esaworldcover.jl` — legend, mode aggregation, per-class fractions
  sum-to-1, no-data (0) masking, vegetation fraction (incl. overridable set),
  integer-factor alignment, dataset interface, bounded-region requirement,
  region-keyed filenames.
- Registered in `src/DataWrangling/DataWrangling.jl` (`include` + `using .ESAWorldCover`)
  and `src/NumericalEarth.jl` (`using .DataWrangling.ESAWorldCover`).

## What is gated

- The real COG read (`worldcover_cog_to_netcdf`) needs `using ArchGDAL`, network,
  and the anonymous S3 bucket — not exercised in CI. Written to spec; falls back to
  a clear error when ArchGDAL is absent (verified). GDAL VRT/warp call shapes are
  plausible but **not run end-to-end** against live tiles.
- `:landcover_fractions` as a NamedTuple **of Fields** is not materialized through
  the single-`Field` path (a Field is one array). The per-class fractions are fully
  available via the pure `class_fractions` helper and are written as per-class bands
  in the NetCDF; a NamedTuple-of-Fields accessor is a next step.

## Verbatim verification (what I ran)

- `julia --project=. -e 'using Pkg; Pkg.instantiate()'` → `instantiate exit=0`.
- Standalone pure-helper check (extracted helpers, no NumericalEarth precompile):
  `Test Summary: pure helpers | Pass 15 Total 15`.
- Parse check (`Meta.parseall`) of module, extension, and test files → all `OK`.
- Loaded NumericalEarth from the worktree; confirmed: `show`=`ESAWorldCover(version = :v200)`,
  `size(d,:landcover_class)=(36000,14400,1)`, interfaces `(-180,180)`/`(-60,84)`,
  vars `[:landcover_class,:landcover_fractions,:vegetation_fraction]`,
  `dataset_variable_name="Map"`, `is_three_dimensional=false`, `missing_value=0`,
  filename `ESA_WorldCover_v200_landcover_class_lon_4_7_lat_50_53.nc`,
  fallback error fires, `ESAWorldCover ∈ supported_datasets()`.
- Ran the full test body against the main project (runtests_setup's CUDA include
  swapped for the minimal `using`s it provides) — all pass:
  legend 7/7, mode 4/4, fractions-sum-to-1 7/7, no-data masking 4/4,
  vegetation 3/3, integer-factor alignment 4/4, interface 19/19,
  bounded-region 2/2, distinct-filenames 1/1 → **51/51**.
- Could NOT run `test/test_esaworldcover.jl` verbatim via the `test/` env
  (full CUDA/Reactant/SpeedyWeather instantiate is impractical here); the file
  itself parses and its assertions pass under the main project as above.

## Live end-to-end verification (2026-07-06)

**Result: YES — a real `Field` materializes from a real ESA WorldCover tile with
sane values.** Verified on branch `xk/worldcover` (worktree), `julia --project=test`,
Julia 1.12.6, macOS/aarch64, with `using ArchGDAL` (extension loaded).

What I ran (a scratch script, not committed) over the Belgium tile N51E003,
`BoundingBox(longitude = (3.0, 3.5), latitude = (51.0, 51.5))`, on a
`LatitudeLongitudeGrid` of `size = (20, 20)` over that box, after deleting any
cached NetCDF to force a clean re-materialization:

- `Field(Metadatum(:landcover_class; dataset = ESAWorldCover(), region), CPU())`
  (native aggregated grid) → size `(52, 52, 1)`; unique codes
  `[10, 30, 40, 50, 60, 80, 90]` — **all integer-valued and all in the legend**
  (tree cover, grassland, cropland, built-up, bare/sparse, water, herbaceous
  wetland — physically sensible for the Belgian coast/Flanders).
- `Field(meta_class, grid)` (interpolated onto the 20×20 target) → size `(20, 20, 1)`,
  range `10.0 … 80.0`. Values are **non-integer** here because the generic
  `Field(metadata, grid)` path bilinearly interpolates; categorical nearest-neighbour
  regridding onto model grids remains the documented next step. Integer codes are
  exact on the native categorical grid above.
- `Field(Metadatum(:vegetation_fraction; …), CPU())` → **no NaN**, range `0.0 … 1.0`,
  all in `[0, 1]`. On the target grid, range `0.0 … 0.998`, all in `[0, 1]`.
- `Field(Metadatum(:landcover_fractions; …), grid)` → raised the designed error
  (per-class NamedTuple product, not a single `Field`).
- Materialized NetCDF has the 11 `fraction_<class>` bands; **sum-of-fractions over
  valid (non-0) cells ∈ [0.99999996, 1.00000004]** (≈ 1, as required).
- `julia --project=test test/test_esaworldcover.jl` → all pass:
  legend 7/7, mode 4/4, fractions-sum-to-1 7/7, no-data masking 4/4,
  vegetation 3/3, integer-factor alignment 4/4, interface 19/19,
  bounded-region 2/2, distinct-filenames 1/1 → **51/51**.

Live checks that confirmed the read path shape:
- `curl -I https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_N51E003_Map.tif`
  → `HTTP/1.1 200 OK`, `Content-Length: 54556617`. The extension's
  `worldcover_tile_label`/`worldcover_tile_url` produce exactly this key for the
  bbox (3°-tile SW-corner math: `fld(51,3)*3 = 51`, `fld(3,3)*3 = 3` → `N51E003`).
- Anonymous `/vsis3/esa-worldcover/...` open via ArchGDAL returned a `36000 × 36000`
  raster (3° at the 10 m native step) once TLS could verify.

**Bugs fixed during verification:**
1. **`missing_value` masked legitimate zeros in the fraction products.**
   `missing_value(::ESAWorldCoverMetadatum)` returned `0` for *every* variable,
   so on load `nan_convert_missing` turned every genuine `0.0`
   (`vegetation_fraction`/`landcover_fractions` — e.g. water cells) into `NaN`.
   `vegetation_fraction` came back **all-NaN**. Fixed so `missing_value` returns `0`
   only for the categorical `:landcover_class` (its true no-data code) and `NaN`
   (which matches no real value, hence masks nothing) for the fraction products.
   Unit test updated to match; `vegetation_fraction` now returns `0.0 … 1.0`.
2. **GDAL_jll TLS to the anonymous HTTPS S3 endpoint failed** with
   `CURL error: SSL certificate problem: unable to get local issuer certificate`
   because GDAL_jll's bundled libcurl could not locate a CA bundle on this machine.
   Added `ensure_curl_ca_bundle!()` to the extension: if neither `CURL_CA_BUNDLE`
   nor `SSL_CERT_FILE` is already set, it points curl at Julia's own bundled
   `cert.pem` (`Sys.BINDIR/../share/julia/cert.pem`) — no new dependency, no root
   `Project.toml` change, and an explicit user configuration always wins. TLS
   verification was **not** disabled. The verification script sets no CA variable
   itself, so the successful read proves the baked-in fallback works.

**Remaining gaps:**
- Categorical `:landcover_class` still uses bilinear interpolation in the generic
  `Field(metadata, grid)` regrid, giving non-integer codes on the target grid.
  Nearest-neighbour categorical regridding is still a next step (codes are exact
  on the native grid).
- `:landcover_fractions` as a NamedTuple **of Fields** is still not materialized
  through the single-`Field` path (per-class bands are in the NetCDF).
- The CA-bundle fallback is a pragmatic workaround for GDAL_jll's curl not finding
  a system CA store; on machines/CI that already configure one it is a no-op.

## Next steps

- P4 biome-prior crosswalk (WorldCover class → `g_{s,max}`, extinction `k`,
  roughness) — where land cover feeds the canopy model.
- NamedTuple-of-Fields accessor for `:landcover_fractions`; nearest-neighbor
  (not bilinear) regridding for the categorical `:landcover_class` onto model grids.
- End-to-end validation of the ArchGDAL COG path against a small live bbox.
