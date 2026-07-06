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

## Live end-to-end verification (2026-07-06)

Real network read tested on macOS (Julia 1.12, `--project=test`, GDAL 3.12.3,
ArchGDAL 0.10.11). **ETH loads real data end-to-end; GLAD does not (host
unreachable here).** Bugs found and fixed along the way are noted inline.

### 1. ETH share-host URL — the flagged suspect was correct

The documented full-resolution path **no longer serves the 10 m tiles
anonymously** — it now 301-redirects to the dataset DOI landing page:

```
$ curl -sIL ".../ETH_GlobalCanopyHeight_10m_2020_version1/3deg_cogs/ETH_GlobalCanopyHeight_10m_2020_N48E011_Map.tif"
HTTP/2 301   location: https://doi.org/10.3929/ethz-b-000609802
HTTP/2 302   location: http://hdl.handle.net/20.500.11850/609802
... → https://www.research-collection.ethz.ch/handle/20.500.11850/609802 (HTTP 500)
```

The `.../version1/` directory itself 301-redirects to the DOI, so the whole 3°
COG tree is DOI-gated (not `/vsicurl/`-readable). The `~pf/nlangdata/` index
still lists two entries; only the **downsampled** one is served (HTTP 200):

```
$ curl -s ".../~pf/nlangdata/ETH_GlobalCanopyHeight_10m_2020_version1_downsampled/" | grep href
ETH_GlobalCanopyHeight_10m_2020_mosaic_Map_0.001deg.tif   # 5.18 GB, accept-ranges: bytes
ETH_GlobalCanopyHeight_10m_2020_mosaic_Map_0.01deg.tif
ETH_GlobalCanopyHeight_10m_2020_mosaic_Map_0.1deg.tif
ETH_GlobalCanopyHeight_10m_2020_mosaic_Map_1deg.tif
```

**Fix:** repointed the ETH read to the anonymously served pre-downsampled global
mosaic; finest publicly served grid is **0.001° (~111 m)**, a single COG windowed
by `/vsicurl/` (`ETH_MOSAIC_RESOLUTION`, `eth_mosaic_url()` in the module;
`Base.size(::ETHCanopyHeight)` → `(360_000, 144_000, 1)` so `360/Nx = 0.001°`).
The 10 m `3deg_cogs` addressing (`eth_tile_token`/`eth_tiles_in_bbox`/
`eth_tile_urls`) is kept and documented as the DOI-archive layout; only `"Map"`
is served (SD/uncertainty needs the DOI record).

### 2. TLS bug — GDAL_jll libcurl had no CA store

First open failed with a transport-layer error (not a 404); shell `curl` to the
same URL returned 200:

```
open failed  msg = "HTTP response code on https://share.phys.ethz.ch/...0.01deg.tif: 0"
```

**Fix:** `configure_vsicurl!()` in the extension points libcurl at a CA bundle via
`CURL_CA_BUNDLE` (Julia's bundled `cert.pem`, then common system paths), with a
`GDAL_HTTP_UNSAFESSL` fallback if none is found. With the CA bundle set the mosaic
opens (`36000×14400` for 0.01°; `360000×144000` for 0.001°).

### 3. `warp_canopy_layer` bug — `ArchGDAL.read(::Vector)` has no method

The original mosaic read path (`ArchGDAL.read(sources) do datasets`) never
worked — `read` takes a single path. **Fix:** open each source explicitly
(`[ArchGDAL.read(s) for s in sources]`), pass the vector to `gdalwarp`, and
`destroy` them in a `finally`. Also switched ETH resampling to `near` so the
categorical `255` no-data byte stays exact for `mask_eth`.

### 4. Real `Field` — ETH over Bavaria (11.0–11.5°E, 47.5–48.0°N)

```julia
region = BoundingBox(longitude=(11.0,11.5), latitude=(47.5,48.0))
meta   = Metadatum(:canopy_height; dataset=ETHCanopyHeight(), region)
grid   = LatitudeLongitudeGrid(CPU(); size=(24,24), longitude=(11.0,11.5),
                               latitude=(47.5,48.0), topology=(Bounded,Bounded,Flat))
field  = Field(meta, grid)     # → Field{Center,Center,Center} size (24,24,1)
```

Result (real data): **min 0.43 m, max 36.9 m, mean 19.4 m**, 555 finite cells,
21 no-data → NaN. Windowed `/vsicurl` read ~5–11 s per 0.5°×0.5° window; Amazon
(−63…−62.5°E, −5…−4.5°N) and a Sahara box (all 0.0, non-forest zeros kept) also
verified at the raw-read level. Heights lie in `[0, ~50] m` as required.

### 5. GLAD — URL corrected, but unreachable from this environment

The module's original GLAD URL (`glad.umd.edu/users/Potapov/GLCLUC2020/
Forest_height_2019.tif`, single mosaic) is wrong on both host and structure. The
GLAD dataset page lists **continental** 30 m mosaics at
`https://glad.geog.umd.edu/Potapov/Forest_height_2019/Forest_height_2019_<CONT>.tif`
(`NAM`, `SAM`, `EURA`, `NAFR`, `SAFR`, `AUS`, `SASIA`, `NASIA`). **Fix:** corrected
`GLAD_COG_HOST` and added `glad_continent`/`glad_tile_urls` (bbox → intersecting
continental tile). **However every candidate URL returned HTTP 404 from this
environment** (host resolves, path rejected — GET, ranged GET, browser UA, and
http all 404), so GLAD is **documented best-effort / unverified**: the read fails
cleanly (`GDALError` open-failed), it does not hang. ETH is primary.

### 6. Regression check

`julia --project=test test/test_canopyheight.jl` → **67/67 pass** (no-data
masking 14, coarsening 7, roughness 11, ETH tile tokens 7, dataset interface 25,
bounded-region 2, extension-gated 1) after the `Base.size`/URL changes.

## Gated / not exercised in CI

- The real COG read (`canopy_height_cog_to_netcdf`) needs `using ArchGDAL` +
  network `/vsicurl/` access; without the extension the module entry point
  errors clearly (mirrors `CopernicusDEM.zarr_to_netcdf`). Not run in CI (Part
  D.5). ETH is now verified against the live host (§ above); GLAD remains
  unverified (host 404s here — revisit its URL/access on a network that can
  reach `glad.geog.umd.edu`).
- The 24×24 Bavaria Field showed 21/576 no-data → NaN cells: the single-pass
  `interpolate_physical!` regrid does not yet use the NaN-aware
  `coarsen_canopy_height` reference, so native `255`/edge NaNs can propagate to a
  coarse cell. Wiring the multi-pass coarsening into the regrid path (as in
  `interpolate_bathymetry_in_passes`) remains the follow-up noted below.

## Deferred follow-up

- **Displacement height in `SimilarityTheoryFluxes` (§3.5)** — left OUT of this
  task as instructed. `RoughnessFromCanopyHeight` already produces `d = 0.7·h_c`,
  but threading `z − d` through the MO stability functions in
  `SimilarityTheoryFluxes` / `LandRoughnessLength` is a separate, focused commit.
- Wiring the antialiased `coarsen_canopy_height` into the `Field(metadata, grid)`
  regrid path (currently single-pass `interpolate_physical!`); the pure helper is
  the reference, following `interpolate_bathymetry_in_passes`.
- GEDI / ICESat-2 sparse-lidar validation (plan §6), out-of-band.
