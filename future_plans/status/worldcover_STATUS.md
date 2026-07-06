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

## Next steps

- P4 biome-prior crosswalk (WorldCover class → `g_{s,max}`, extinction `k`,
  roughness) — where land cover feeds the canopy model.
- NamedTuple-of-Fields accessor for `:landcover_fractions`; nearest-neighbor
  (not bilinear) regridding for the categorical `:landcover_class` onto model grids.
- End-to-end validation of the ArchGDAL COG path against a small live bbox.
