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
    `location=(Center,Center,Center)`, `missing_value=-9999`, `reversed_latitude_axis=true`,
    global `longitude/latitude_interfaces` + resolution-dependent `Base.size`
    (CopernicusDEM windowing model), `validate_dataset_coverage` requiring a `BoundingBox`,
    region+resolution-encoded `metadata_filename`, `available_variables`/`dataset_variable_name`
    (`:emissivity→"/Emissivity/Mean"`, `:emissivity_uncertainty→"/Emissivity/SDev"`).
  - `retrieve_data` orchestrates: gated raw read → decode → broadband combine →
    water mask; returns the regional `(Nx,Ny)` array.
- `test/test_asterged.jl` — synthetic-tile unit tests (auto-discovered by ParallelTestRunner).
- Registered in both `src/DataWrangling/DataWrangling.jl` (include + `using .ASTERGED`)
  and `src/NumericalEarth.jl` (`using .DataWrangling.ASTERGED`).

## Gated (not wired) and why

- **HDF5 tile read** (`read_asterged_region`) and **Earthdata download**
  (`download_asterged`) are plain module functions with fallback `error(...)`
  pointing here — **option (b)** from the brief. ASTER GED tiles are HDF5, but
  `HDF5.jl` is not a NumericalEarth dependency and **must not be added to root
  `Project.toml`** (AGENTS.md rule 10); the download also needs NASA Earthdata
  credentials. No weak-dep extension was wired: ArchGDAL/GDAL's HDF5 driver
  reading of the `/Emissivity/Mean` subdataset (option (a)) is unverifiable
  without a real tile + credentials, and the geolocation lives in separate
  `/Geolocation/*` arrays (no embedded geotransform), so a GDAL warp is not a
  clean fit. The pure decode/broadband/mask core is fully implemented and tested
  independent of the read.

## Decisions

- **Broadband coefficients**: `OGAWA_2003_SLOPES = (0.025,0.057,0.237,0.333,0.146)`
  (Ogawa et al. 2003 regression slopes) normalized to sum to 1 → a convex
  combination, so broadband ε stays within band range (guarantees [0.7,1.0] over
  land). Intercept dropped (interface is a pure dot). Tunable via the struct field.
- **LWmap water code**: default `ASTERGED_WATER_CODE = 2` (LP DAAC User Guide V3:
  water=2, land=1); coding is documented-ambiguous (GEE uses 0/1), so
  `mask_water` exposes a `water_code` keyword. Verify empirically on a real tile.
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

## Next steps

1. Wire the HDF5 read + Earthdata/CMR download (extension gated on `HDF5` +
   `.netrc`/bearer token; shared `earthdata_download` per infra C.2).
2. Verify the LWmap coding on a real AG100 tile; fix `ASTERGED_WATER_CODE` if needed.
3. Multi-tile mosaic across a bbox spanning several 1° tiles (`read_asterged_region`).
4. Wire `ε` into the radiation `stateindex` slot; add an `@example` regional map to docs.
