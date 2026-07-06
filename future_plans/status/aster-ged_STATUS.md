# ASTER GED v3 emissivity — implementation status

Branch `xk/aster-ged`. Implements Plan 02 (ASTER GED surface emissivity) as a
standalone static DataWrangling submodule.

## Implemented files

- `src/DataWrangling/ASTERGED/ASTERGED.jl` — new submodule.
  - `struct ASTERGEDv3{R} <: AbstractStaticDataset` (`resolution::Symbol`,
    `broadband_coefficients::R`) + keyword constructor
    (`resolution=:AG100`, `broadband_coefficients=OGAWA_2003_BROADBAND_COEFFICIENTS`),
    argument-validated, with `show`/`summary` and a `jldoctest`.
  - **Pure core (no credentials/IO, fully unit-tested):**
    `decode_mean(DN)=ifelse(DN==-9999, NaN, 0.001*DN)`,
    `decode_sdev(DN)=ifelse(DN==-9999, NaN, 1.0e-4*DN)` (0.0001 vs 0.001 10× split),
    `broadband_emissivity(ε, c)` (dot; convex-combination), `broadband_uncertainty`,
    `broadband_emissivity_map`/`broadband_uncertainty_map` (band index = dim 1),
    `mask_water(field, lwmap; water_code)`.
  - Interface per Part D.2: `is_three_dimensional=false`, `default_inpainting=nothing`,
    `location=(Center,Center,Center)`, `missing_value=-9999`, `reversed_latitude_axis=false`
    (the regional NetCDF is written south→north), `longitude/latitude_name = lon/lat`,
    global `longitude/latitude_interfaces` + resolution-dependent `Base.size`
    (CopernicusDEM windowing model), `validate_dataset_coverage` requiring a `BoundingBox`,
    region+resolution-encoded `metadata_filename`, `available_variables`/`dataset_variable_name`
    (`:emissivity→"/Emissivity/Mean"`, `:emissivity_uncertainty→"/Emissivity/SDev"`),
    plus `asterged_short_name`/`asterged_version`/`asterged_cmr_granules_url` for CMR.
  - `retrieve_data` reads the regional NetCDF of raw DN (written by the download
    step in the ArchGDAL extension) → decode → broadband combine → water mask;
    returns the regional `(Nx,Ny)` array.
- `ext/NumericalEarthArchGDALExt.jl` — the real HDF5 read + Earthdata/CMR download
  (`asterged_tiles_to_netcdf`, `earthdata_cmr_granules`, `earthdata_download`,
  `read_asterged_subdataset`, `nearest_index`). See the "Live end-to-end
  verification" section below.
- `test/test_asterged.jl` — synthetic-tile unit tests (auto-discovered by ParallelTestRunner).
- Registered in both `src/DataWrangling/DataWrangling.jl` (include + `using .ASTERGED`)
  and `src/NumericalEarth.jl` (`using .DataWrangling.ASTERGED`).

## Gated (not wired) and why

- **NOW WIRED (option a).** The HDF5 tile read + Earthdata/CMR download are
  implemented in `ext/NumericalEarthArchGDALExt.jl` via ArchGDAL/GDAL's HDF5
  driver (no `HDF5.jl` dependency; AGENTS.md rule 10 respected). Entry points:
  - `asterged_tiles_to_netcdf(metadatum, nc_path)` — CMR discovery →
    `netrc_downloader` (urs.earthdata.nasa.gov) tile download → GDAL subdataset
    read (`HDF5:"file.h5"://Emissivity/Mean` etc.) → clip + mosaic to the bbox →
    write a regional NetCDF of *raw digital numbers* (`emissivity_mean`,
    `emissivity_sdev` band-first `(5,Nx,Ny)`; `land_water_map` `(Nx,Ny)`), with
    lon/lat cell centers from the `/Geolocation/*` arrays (no reprojection —
    ASTER GED is already plain WGS84 lat/lon).
  - `earthdata_cmr_granules(short_name, version, bbox::BoundingBox)` — CMR
    `granules.json` query, de-duplicated by granule (keeps the protected `data#`
    endpoint).
  - The module keeps generic fallback methods (`asterged_tiles_to_netcdf(_, _)`,
    `earthdata_cmr_granules(_, _, _)`) that error clearly when ArchGDAL is not
    loaded; the ext methods are strictly more specific so they *add* rather than
    *overwrite* (required on Julia 1.12, where precompile-time method overwriting
    is now a hard error).
  - `retrieve_data(::ASTERGEDMetadatum)` reads that NetCDF and feeds the existing
    pure `decode_mean`/`broadband_emissivity_map`/`mask_water` core.
  Credentials come from `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`.

## Decisions

- **Broadband coefficients**: `OGAWA_2003_SLOPES = (0.025,0.057,0.237,0.333,0.146)`
  (Ogawa et al. 2003 regression slopes) normalized to sum to 1 → a convex
  combination, so broadband ε stays within band range (guarantees [0.7,1.0] over
  land). Intercept dropped (interface is a pure dot). Tunable via the struct field.
- **LWmap water code**: default `ASTERGED_WATER_CODE = 1` — **verified empirically
  on a real AG100 v003 tile**: `/Land_Water_Map/LWmap` holds only `{0, 1}` (the
  GEE-style 0=land / 1=water coding, NOT the LP DAAC 1/2 coding). On tile
  `AG100.v003.37.-112` (Grand Canyon; lat 36–37, lon −112 to −111) exactly 851 of
  10⁶ cells are `1`, tracing the Colorado River → water = 1. `mask_water` still
  exposes a `water_code` keyword in case a future tile differs.
- **Native size / interfaces** follow CopernicusDEM: global integer-degree hull
  (`(-180,180)`/`(-90,90)`) + global px counts (AG100 360000×180000,
  AG1km 36000×18000); `construct_native_grid` restricts to the bbox and
  `retrieve_data` returns the restricted window.
- **`supported_datasets()` omits `ASTERGEDv3`**: the discovery scan enumerates
  only concrete `DataType`s, and `ASTERGEDv3{R}` is a `UnionAll`. Consequence of
  the required parametric signature; the dataset is fully usable directly via
  `Metadatum(:emissivity; dataset=ASTERGEDv3(), region)`. Make it concrete
  (`broadband_coefficients::Vector{Float64}`) if listing is later desired.

## Verified (verbatim)

- `julia --project=. -e 'using Pkg; Pkg.instantiate(); using NumericalEarth; ...'`
  → `NumericalEarth` precompiled (18.6 s) and loaded; `ASTERGEDv3()` →
  `ASTERGEDv3(resolution = :AG100)`, `:AG1km` variant OK.
- Pure core (standalone `scratchpad/verify_asterged_pure.jl`): all pass —
  decode scaling 6/6, broadband 6/6, band-index 3/3, fill→NaN 2/2, water 5/5,
  uncertainty 2/2.
- Full `test/test_asterged.jl` body run against the real loaded module (CUDA-free
  harness, `runtests_setup.jl` include stripped): **all pass** — decode 8/8,
  broadband 6/6, band-index 3/3, fill→NaN 2/2, water 5/5, uncertainty 2/2,
  interfaces 27/27, filenames 1/1, bounded-region 2/2 (56 total).
- `retrieve_data(::ASTERGEDMetadatum)` errors with the gated HDF5/Earthdata
  message, as intended.
- **Not run**: `test/test_asterged.jl` via `julia --project=test` /
  `Pkg.test()` (the test env pulls CUDA and the full heavy stack); the body was
  instead exercised directly against the precompiled main-project module, which
  covers every assertion. The `jldoctest` was not executed via Documenter but its
  expected output matches the live `show`.

## Live end-to-end verification

Real data, real credentials, no stubs. Run from the `xk/aster-ged` worktree with
Earthdata credentials in the environment (`source ~/.zshrc` exports
`EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`):

```
source ~/.zshrc 2>/dev/null; julia --project=test -e 'using Pkg; Pkg.instantiate()'
source ~/.zshrc 2>/dev/null; julia --project=test test/test_asterged.jl   # 56/56 pass
source ~/.zshrc 2>/dev/null; julia --project=test scratch_e2e.jl
```

`scratch_e2e.jl` builds a real regional `Field` over a 1° tile:

```julia
region = BoundingBox(longitude = (-112.0, -111.5), latitude = (36.0, 36.5))
grid   = LatitudeLongitudeGrid(size = (50, 50), longitude = (-112.0, -111.5),
                               latitude = (36.0, 36.5), topology = (Bounded, Bounded, Flat))
md     = Metadatum(:emissivity; dataset = ASTERGEDv3(), region)
field  = Field(md, grid)   # using ArchGDAL activates the read/download extension
```

**Outcome (verbatim program output):**

```
metadata_path = .../ASTERGED/ASTERGED_AG100_emissivity_lon_-112.0_-111.5_lat_36.0_36.5.nc
=== emissivity Field ===
size(field) = (50, 50, 1)
n finite = 2499 / 2500
emissivity min = 0.9090495705604553 max = 0.9788420796394348 mean = 0.9493083550768788
in [0.7, 1.0]? true
=== raw NetCDF ===
emissivity_mean dims = (5, 539, 539) (band, lon, lat)
LWmap unique = Int16[0, 1]
  LWmap == 0 count = 290517
  LWmap == 1 count = 4
=== uncertainty Field ===
uncertainty min = 0.002790805185213685 max = 0.017697779461741447
```

- **Real `Field`?** Yes — CMR resolved the intersecting `AG100.v003` tiles, they
  were downloaded from Earthdata Cloud (`data.lpdaac.earthdatacloud.nasa.gov`,
  ~44 MB each), read through GDAL's HDF5 driver, mosaicked, decoded and combined.
- **Emissivity range?** `[0.909, 0.979]`, mean `0.949`, all in `[0.7, 1.0]`;
  2499/2500 target cells finite (one edge/water cell → `NaN`, as designed).
- **LWmap coding confirmed?** Yes — the raw NetCDF `land_water_map` holds only
  `{0, 1}` → `ASTERGED_WATER_CODE = 1` (land 0 / water 1).
- **Band order + scale confirmed?** `/Emissivity/Mean` is `[5×1000×1000]` int16;
  GDAL reads it band-last `(1000,1000,5)`, permuted to band-first `(5,Nx,Ny)`;
  the `0.001` scale gives per-band valid means `0.884–0.960` — physical.
- **`:emissivity_uncertainty`** also loads (broadband σ `[0.0028, 0.0177]`),
  confirming the `/Emissivity/SDev` path (0.0001 scale) works end-to-end.

Direct HDF5 inspection (`scratch_inspect.jl`) confirmed the subdataset list, the
`[5×1000×1000]` `/Emissivity/Mean` layout, the identity geotransform (hence coords
built from `/Geolocation/*`), and the `{0,1}` LWmap.

(The `scratch_*.jl` helper scripts are not committed; the reproducible entry
points are the module + extension source and `test/test_asterged.jl`.)

## Next steps

1. Wire `ε` into the radiation `stateindex` slot; add an `@example` regional map to docs.
2. Cache/robustness: CMR returns up to 4 edge-touching tiles for a sub-degree
   bbox (each ~44 MB); a transient download failure was observed once and
   succeeded on retry — consider a retry wrapper and/or restricting to tiles that
   actually contribute interior cells.
3. Exercise a larger bbox spanning several 1° tiles to stress the mosaic
   (single- and adjacent-tile cases are verified).
