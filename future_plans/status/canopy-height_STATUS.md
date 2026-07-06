# Canopy-height ingestion — implementation status

Plan: `future_plans/03_canopy_height_roughness.md` (+ shared infra Part A/D).
Branch: `xk/canopy-height`.

## Files

- `src/DataWrangling/CanopyHeight/CanopyHeight.jl` — new module. Types
  `ETHCanopyHeight` / `GLADCanopyHeight` (`<: AbstractStaticDataset`), the
  `RoughnessFromCanopyHeight` closure, and the pure decode/mask/coarsen/roughness
  core.
- `ext/NumericalEarthArchGDALExt.jl` — extended with the real `/vsicurl/`
  windowed COG read (`canopy_height_cog_to_netcdf`) for both products.
- `src/DataWrangling/DataWrangling.jl` — `include(...)` + `using .CanopyHeight`.
- `src/NumericalEarth.jl` — `using .DataWrangling.CanopyHeight`.
- `test/test_canopyheight.jl` — unit tests (auto-discovered by ParallelTestRunner).

## Interface (per Part D.2)

- `is_three_dimensional = false`, `default_inpainting = nothing`
  (**zeros are valid non-forest heights, never inpainted**),
  `location = (Center, Center, Center)`, `longitude/latitude_name = "lon"/"lat"`
  (the regional NetCDF we materialize), global `longitude/latitude_interfaces`
  with native `Base.size` (ETH 10 m → 4_320_000×2_160_000; GLAD 30 m →
  1_440_000×720_000), region-encoded `metadata_filename`,
  `validate_dataset_coverage` **requires a `BoundingBox`** (mirrors CopernicusDEM).
- Variables: ETH `:canopy_height → "Map"`, `:canopy_height_uncertainty → "SD"`;
  GLAD `:canopy_height → "Map"`. `missing_value(ETH) = 255`; GLAD fill codes
  masked at read time.
- Default `retrieve_data` (NetCDF reader) is reused unchanged — the extension
  writes a clean regional NetCDF with `Map`(/`SD`) already NaN-masked.

## Pure, unit-tested core

- `mask_glad(code) = ifelse(code >= 101, NaN, float(code))` — masks 101/102/103,
  **keeps 0**. `mask_eth(x, 255)` — masks the no-data byte, keeps 0.
- `coarsen_canopy_height(fine, factor)` — antialiased **NaN-masked** block mean
  (10 m → coarse cell); NaN only if all contributing cells are NaN.
- `roughness_length(h, a)` / `displacement_height(h, b)` and the
  `RoughnessFromCanopyHeight(; momentum_roughness_coefficient = 0.10,
  displacement_coefficient = 0.70)` closure — **coefficients exposed**, kept
  separate from the data layer (ClimaLand 0.13 documented as an alternative).
- `eth_tile_token` / `eth_tiles_in_bbox` / `eth_tile_urls` — 3° SW-corner tile
  addressing for the ETH COG mosaic.

## Verification (verbatim)

- `julia --project=. -e 'using Pkg; Pkg.instantiate()'` → exit 0.
- Syntax parse (`Meta.parseall`) of module, extension, test file → all OK.
- Standalone pure-helper suite (scratchpad, no precompile) →
  `Test Summary: pure helpers | 19 passed, 19 total`.
- `using NumericalEarth` precompiles and loads cleanly; `ETHCanopyHeight()`,
  `GLADCanopyHeight()`, `RoughnessFromCanopyHeight()` accessible via
  `using NumericalEarth.DataWrangling.CanopyHeight` (same access pattern as
  `SoilGrids2`; neither is surfaced by a bare `using NumericalEarth`, matching
  the existing SoilGrids/CopernicusDEM convention). `CanopyHeight` is present in
  `dataset_modules()`.
- Full `test/test_canopyheight.jl` under the test env (`julia --project=test`)
  → exit 0, all 67 assertions pass across 7 testsets: no-data masking (14),
  coarsening (7), roughness (11), ETH tile tokens (7), dataset interface (25),
  bounded-region requirement (2), COG read extension-gated (1).

## Gated / not exercised in CI

- The real COG read (`canopy_height_cog_to_netcdf`) needs `using ArchGDAL` +
  network `/vsicurl/` access; without the extension the module entry point
  errors clearly (mirrors `CopernicusDEM.zarr_to_netcdf`). Not run in CI (Part
  D.5). The ETH tile-URL host path and 3° tile-token convention are coded to the
  documented ETH layout but should be verified against the live share host on
  first real read; likewise the GLAD single-mosaic URL is a documented
  best-effort (GLAD ships continental / 10° tiles — the URL builder there is the
  spot to refine).

## Deferred follow-up

- **Displacement height in `SimilarityTheoryFluxes` (§3.5)** — left OUT of this
  task as instructed. `RoughnessFromCanopyHeight` already produces `d = 0.7·h_c`,
  but threading `z − d` through the MO stability functions in
  `SimilarityTheoryFluxes` / `LandRoughnessLength` is a separate, focused commit.
- Wiring the antialiased `coarsen_canopy_height` into the `Field(metadata, grid)`
  regrid path (currently single-pass `interpolate_physical!`); the pure helper is
  the reference, following `interpolate_bathymetry_in_passes`.
- GEDI / ICESat-2 sparse-lidar validation (plan §6), out-of-band.
