# SlabLand follow-up roadmap

This page sets out how follow-up work on `SlabLand` after the
"Conservative variably saturated SlabLand + evaporation-front humidity"
PR (#323) should be staged — what is metadata-backed external data,
what is implementation-only physics, and what is training provenance
for learned maps. It is a contributor-facing planning document, not user
documentation; the user-facing physics walk-through lives in the
[SlabLand tutorial](./evaporation_front_slab_land.md).

The companion tutorial documents what is implemented today. This page
documents what is *not yet* implemented and how to add it without
blurring three separate concerns:

1. **External data products** described by `Metadata` / `MetadataSet`.
2. **Implementation** — physical formulas, closures, observation
   operators — described by code, tests, and docstrings.
3. **Learned maps** described by training provenance.

## 1. The central decision rule

> Use `Metadata` / `MetadataSet` for **satellite remote-sensing data
> products**. Use plain implementation for physical formulas, closures,
> constants, and observation operators. Use **training provenance**, not
> product metadata, for learned maps.

This is sharper than treating `Metadata` as a generic provenance
mechanism. The schema satellite products share is sharp — instrument,
swath, overpass time, view geometry, compositing window, QA bitmask,
retrieval uncertainty, mission/version — and a focused
`Metadata`/`MetadataSet` type can enforce it. Other data sources
(reanalysis, hybrid products, point-scale tower observations) need
*different* schemas and should not be forced into the satellite-metadata
shape.

### 1.1 What fits `Metadata`

Satellite remote-sensing products only:

| Product | Why it fits |
|---|---|
| MCD43 BRDF / albedo | MODIS swath, 16-day composite, view-zenith QA |
| MOD21 / MYD21 / ASTER GED emissivity | TES algorithm, GED bands, overpass-time semantics |
| MOD15 LAI / FPAR, MOD13 NDVI / EVI | composite window, QA bitmask, scale factors |
| GEDI / ICESat-2 | swath, beam, footprint, retrieval uncertainty |
| CERES SW / LW | TOA-to-surface unfolding QA, mission/version |
| MODIS / VIIRS / GOES / ECOSTRESS LST | overpass time, view angle, band conversion, cloud QA |
| SMAP / SMOS / ASCAT | polarization, look angle, RFI flags |
| GRACE / GRACE-FO | mascon scale, GIA correction, gap months |
| MCD12 / ESA WorldCover | class definitions, year, accuracy tables |

### 1.2 What does not fit `Metadata`

These need different homes:

| Product | Why it differs | Home |
|---|---|---|
| ERA5, MERRA-2, NLDAS | Reanalysis — assimilated model output with forecast/analysis time semantics and accumulation conventions, not satellite acquisition geometry. | Existing `ERA5PrescribedAtmosphere` / `PrescribedRadiation` family. |
| SoilGrids | Geostatistical ML product fusing many inputs. No overpass, no QA bitmask. | A `PreprocessedLandCovariate` family (to be added) or a versioned static-field provider. |
| GLEAM / MOD16 / FLUXCOM ET | Model-data hybrids — partly satellite, partly reanalysis-forced. | Same as SoilGrids: hybrid-product provenance. |
| FLUXNET / AmeriFlux / ICOS / OzFlux towers | Point-scale observations. Site-by-site instrument heights, IGBP class, footprint, gap-filling QC. | A `TowerObservation` family (to be added). |
| ETOPO / SRTM / Copernicus DEM | Satellite-derived but static, blended products used as covariates not validated against. | Existing `regrid_topography` path. |
| Physical constants, closure formulas | Not data. | Code + tests. |
| Learned property maps | Provenance is the training run, not the product. | Sidecar training-provenance file. |

### 1.3 The clean pipeline

```text
Satellite product
  └── Metadata / MetadataSet
        └── preprocessing, QA, unit conversion, regridding
              └── runtime property provider (Field / FieldTimeSeries / pure formula)
                    └── SlabLand / Radiation / interface kernels
```

and separately:

```text
Physical formula or closure
  └── implementation + tests
        └── optional learned coefficients with training provenance
```

Kernel-time objects must be numerical and type-stable. No product
names, URLs, strings, QA-flag lookup tables, file handles, dictionaries,
or dynamic dispatch on product type inside kernels.

## 2. What PR #323 already implements

These are physical closures, not data products:

- `VariablySaturatedBucketHydrology` — conservative signed-flux water
  budget; augmented-liquid-fraction storage; supports `Mˡᵃ > Mˡᵃ⁺` via
  pressure head.
- `WaterCoupledForceRestoreEnergy` — `Mˡᵃ`-dependent heat capacity
  `C(Mˡᵃ) = C_dry + cˡ Mˡᵃ`; conservative `dTˡᵃ/dt` (adding water at
  slab temperature leaves T invariant).
- `VanGenuchtenRetention` + `VanGenuchtenConductivity` — retention curve
  `Π(𝒮)` and Mualem `K(𝒮)`.
- `NoDeepLiquidFlux` / `FreeDrainageFlux` / `DarcyDeepLiquidFlux` /
  `LinearReservoirDrainage` — bottom-boundary closures.
- `NoRunoff` / `InfiltrationCapacityRunoff` — surface/subsurface runoff
  diagnostics.
- `EvaporationFrontHumidity` — Fickian vapor-flux balance for `qⁱⁿ`
  through an unresolved evaporation front at saturation-dependent depth
  `δᵛ(𝒮)`. Sub-closures: `StorageBasedEvaporationFrontDepth`,
  `DryLayerVaporPistonVelocity` (with `ConstantTortuosity` /
  `MillingtonQuirk` dispatch), `UnitWaterActivity`.
- `SlabLand.diagnostics` — closure-extensible diagnostics slot.
- Signed flux assembly — `vapor_flux`, `surface_energy_flux`,
  `liquid_precipitation_flux` populated alongside the legacy
  positive-part fields.

PR #323 introduces **no `Metadata` consumers**. The closures accept
scalar parameters with default values; field-valued parameter support is
deferred to follow-up.

## 3. Refinements to the original scope

### 3.1 Canopy-height-driven roughness is a roughness concern, not vegetation

Aerodynamic roughness `ℓ_m` is a property of the atmosphere–land flux
closure (`SimilarityTheoryFluxes`), not of `SlabLand`. Canopy height
`h_c` (from GEDI / ICESat-2 / a fused product) feeds roughness through a
one-line formula:

```math
\ell_m \approx 0.1\, h_c, \qquad
d \approx 0.7\, h_c, \qquad
\ell_h/\ell_m = \exp(-b_h).
```

This is a satellite-`Metadata` consumer for `h_c` plus an implementation
formula — **not a vegetation closure**. It does not need LAI, stomatal
resistance, or transpiration. It belongs as a small Tier-2 follow-up
(roughness-input property provider on `SimilarityTheoryFluxes`), not as
part of the deferred vegetation stack.

### 3.2 Roughness, albedo, and emissivity all live outside `SlabLand`

The bare-ground `SlabLand` carries no roughness, no albedo, and no
emissivity. Those are properties of the atmosphere–land flux closure
(`SimilarityTheoryFluxes`) and the radiation module respectively.
Metadata-backed satellite products (MCD43, MOD21, ASTER GED, GEDI)
should produce `Field`-valued property providers that flow into those
sites, never into `SlabLand`.

## 4. Tiered roadmap

### Tier 0 — finish PR #323 (small commit, ~200 LOC)

Test-coverage gaps Codecov flagged:

1. `DarcyDeepLiquidFlux` capillary-rise vs drainage sign test.
2. `LinearReservoirDrainage` no-drain below equilibrium / drain above.
3. `InfiltrationCapacityRunoff` exactly-at-capacity edge case.
4. `VanGenuchtenRetention` endpoint + monotonicity.
5. `VanGenuchtenConductivity` `K(0) = 0`, `K(1) = K_saturated`.
6. `MillingtonQuirk` diffusivity → 0 at saturation.
7. `EvaporationFrontHumidity` `D == 0` fallback branch.
8. `WaterCoupledForceRestoreEnergy` both `deep_time_scale` and
   `deep_conductance` branches exercised.
9. Construction error for `advect_surface_liquid_energy = true` (loud
   fail).
10. Assertion that `surface_energy_flux` is additive with the radiative
    fluxes (not overwritten).

Plus process: CPU CI, GPU CI, docs build, both converted examples render.

### Tier 1 — property-provider extension (one follow-up PR, ~400 LOC)

Turn scalar `FT` closure parameters into
`Union{FT, AbstractField, FieldTimeSeries-like}` using the existing
`normalize_property` / `property_value` / `stateindex` machinery.

**Highest-priority parameters:**

| Closure | Parameters |
|---|---|
| `WaterCoupledForceRestoreEnergy` | `dry_heat_capacity`; verify non-scalar `deep_temperature` / `deep_conductance` on CPU + GPU |
| `VariablySaturatedBucketHydrology` | `porosity`, `residual_liquid_fraction`, `specific_storage`, `critical_saturation` |
| `VanGenuchtenRetention` | `α`, `n` |
| `VanGenuchtenConductivity` | `K_saturated`, `n`, `ℓ` |
| `StorageBasedEvaporationFrontDepth` | `maximum_front_depth`, `front_depth_exponent` |
| `DryLayerVaporPistonVelocity` | `minimum_front_depth`, `molecular_diffusivity` |

**Per parameter:** struct field → generic; constructor calls
`normalize_property(FT, value)`; kernel access via `property_value(...)`;
one test for scalar-vs-constant-`Field` equivalence; one GPU test.

Plus a small set of constant-or-harmonic radiation providers:
`ConstantLandAlbedo`, `ConstantLandEmissivity`, harmonic variants
(~50 LOC each, in the radiation module).

### Tier 2 — satellite-Metadata adapters

Each is an independent follow-up PR. Order by science value:

| Adapter | Priority | New code | Status of inputs |
|---|---|---|---|
| **MCD43 BRDF / albedo** | High | ~400 LOC | New `DataWrangling.MCD43` module: download, BRDF kernel handling, direct/diffuse split, SZA evaluation. Output is a `MaterializedAlbedo` consumed by the radiation module. |
| **MOD21 / ASTER GED emissivity** | Medium | ~300 LOC | Similar pattern; broadband conversion. |
| **SoilGrids → pedotransfer → VG params** | High | ~500 LOC | Not a satellite product — gets its own `PreprocessedLandCovariate` abstraction. Pedotransfer (Rosetta / Cosby) produces `Field`s for the Tier-1 property API. |
| **GEDI / ICESat-2 canopy height** | Medium | ~200 LOC adapter + ~30 LOC roughness formula + ~50 LOC `SimilarityTheoryFluxes` provider extension | Roughness-input only; not vegetation. |
| **MOD15 LAI / FPAR** | Low until vegetation | — | Defer until canopy closure exists. |
| **MOD13 NDVI / EVI** | Low until vegetation | — | Defer. |
| **MCD12 / ESA WorldCover** | Medium | ~200 LOC | Class definitions, biome priors for learnable parameters. |
| **ERA5 schema completeness** | Already mostly exists | ~100 LOC tests | Confirm accumulation→rate, dewpoint→qᵛ, valid-time semantics. *Not* a satellite-`Metadata` consumer — stays in `ERA5PrescribedAtmosphere`. |
| **Topography elevation correction** | Already exists | — | `ElevationCorrection` ships in main. |

### Tier 3 — observation operators (new module, ~200–500 LOC each)

These live in a new `LandObservations` (or similar) module, not in
`SlabLand`. Each operator maps model state to a satellite observable
through metadata-backed inputs (view geometry, retrieval uncertainty,
QA mask).

| Operator | Maps | Effort |
|---|---|---|
| `H_LST(Tˡᵃ, ε, view)` | model → MODIS / VIIRS / ECOSTRESS LST | ~300 LOC |
| `H_mw(Mˡᵃ, 𝒮, texture, T)` | model → SMAP / SMOS brightness temperature | ~500 LOC (tau-omega or Mironov) |
| `H_ET(Jᵛ)` | model → GLEAM / MOD16 / FLUXCOM ET | ~150 LOC |
| `H_TWS(Mˡᵃ)` | model → GRACE TWS anomaly | ~150 LOC |
| `H_tower(𝒬ᵀ, 𝒬ᵛ, u★)` | model → FLUXNET / AmeriFlux / ICOS | ~300 LOC (footprint weighting) |

Plus loss-function composition (~200 LOC).

### Tier 4 — learning infrastructure (1–2 PRs, ~500–800 LOC)

- **Training provenance sidecar format** (~150 LOC) — TOML / YAML / JLD2
  schema for: model commit, configuration, training domain & period,
  observations used, priors used, optimizer, hyperparameters, loss
  terms, regularization, posterior uncertainty, artifact checksum.
- **Loss composition** (~200 LOC) — weighted sum of operators from Tier
  3 with regularization terms.
- **Differentiability checks** (~300 LOC) — Enzyme / Zygote compatibility
  for each Tier-1 parameter as a learnable target; finite-difference
  gradient verification.
- **Gradient-safe property providers** — confirm `normalize_property` /
  `property_value` round-trip through autodiff.

## 5. Property-by-property classification

### 5.1 Prognostic states and diagnostics

None of these need `Metadata`. They are runtime state.

`Tˡᵃ`, `Mˡᵃ`, `𝒮`, `Π`, `θˡ`, `ϑˡ`, all `land.diagnostics`, `qⁱⁿ`, `δᵛ`,
`Tᵉ` — diagnostic relations from the closures.

Satellite observations of LST or soil moisture are *compared* to these
through Tier-3 observation operators; they do not *set* the states.

### 5.2 Energy closure parameters

| Parameter | Satellite `Metadata`? | Implementation? | Notes |
|---|---:|---:|---|
| Conservative `dT/dt` formula | No | Yes | Implemented |
| `C(Mˡᵃ) = C_dry + cˡ Mˡᵃ` formula | No | Yes | Implemented |
| `dry_heat_capacity` scalar | No | Yes | Tier 1 extends to `Field` |
| `dry_heat_capacity` learned map | Training provenance | Yes | Tier 4 |
| `liquid_heat_capacity` | No | Yes | Physical constant |
| `deep_temperature` scalar | No | Yes | Already supports non-scalar |
| `deep_temperature` from climatology | Hybrid (input might be satellite-blended) | Yes | Loader → `FieldTimeSeries` |
| `deep_conductance` scalar | No | Yes | Already supports non-scalar |
| `deep_time_scale` | No | Yes | Closure parameter |

### 5.3 Hydrology closure parameters

| Parameter | Satellite `Metadata`? | Implementation? | Notes |
|---|---:|---:|---|
| Signed mass budget | No | Yes | Implemented |
| Augmented liquid fraction `ϑˡ` | No | Yes | Implemented conceptually |
| `slab_depth` | No | Yes | Closure parameter |
| `porosity` scalar | No | Yes | Tier 1 → `Field` |
| `porosity` field from SoilGrids | Not satellite — `PreprocessedLandCovariate` | Yes (pedotransfer) | Tier 2 |
| `residual_liquid_fraction` | Same as porosity | Yes | Same |
| `specific_storage` | No | Yes | Closure parameter |
| `critical_saturation` | No | Yes | Closure parameter, possibly learned |
| Van Genuchten / Mualem formulas | No | Yes | Implemented |
| Van Genuchten parameters | Not satellite — SoilGrids | Yes (pedotransfer) | Tier 2 |
| All deep-flux and runoff closures | No | Yes | Implemented formulas |
| `deep_pressure_head` | Possibly (if from a groundwater product) | Yes | Already normalized as a property |

### 5.4 Interface humidity parameters

| Parameter | Satellite `Metadata`? | Implementation? | Notes |
|---|---:|---:|---|
| Evaporation-front humidity solve | No | Yes | Implemented |
| `qⁱⁿ` | No | Yes | Solved diagnostic; never prescribed from satellite |
| `δᵛ_max`, `η` | No | Yes | Tier 1 → `Field`; Tier 4 learnable |
| `δᵛ_min` | No | Yes | Numerical floor |
| `ℓᵀ` | No | Yes | Aligns with future land-side skin-temperature solve |
| `Dᵛ₀` | No | Yes | Physical constant |
| `ConstantTortuosity`, `MillingtonQuirk` | No | Yes | Implemented |
| `UnitWaterActivity` | No | Yes | Implemented |
| `MatricPotentialActivity` (Kelvin) | No | Yes | Deferred implementation, not metadata |

### 5.5 Atmosphere–land flux closure (lives outside `SlabLand`)

| Parameter | Satellite `Metadata`? | Implementation? | Notes |
|---|---:|---:|---|
| Constant `ℓ_m, ℓ_T, ℓ_v` | No | Yes | Closure parameters |
| Field-valued roughness from canopy height | Yes for `h_c` (GEDI / ICESat-2) | Yes for `ℓ_m ≈ a h_c` formula | Tier 2 follow-up |
| Scalar-transfer barrier `b_h = log(ℓ_m/ℓ_h)` | No (lookup or learnable) | Yes | Tier 1/4 |
| Stability-dependent scalar barrier | No | Yes | Future closure |

### 5.6 Radiation properties (lives outside `SlabLand`)

| Property | Satellite `Metadata`? | Implementation? | Notes |
|---|---:|---:|---|
| Constant albedo / emissivity | No | Yes | Tier 1 fallback |
| Harmonic albedo / emissivity | No | Yes | Tier 1 fallback |
| MCD43 BRDF / albedo | Yes | Yes (BRDF / SZA formula) | Tier 2 |
| MOD21 / ASTER GED emissivity | Yes | Yes (broadband conversion) | Tier 2 |
| Correction factors `s_α`, `s_ε` | Training provenance if learned | Yes | Tier 4 |
| Wetness correction to albedo | No unless derived from data | Yes | Future closure |

## 6. What `SlabLand` (and friends) must never do

Three failure modes the architecture must keep ruling out:

1. **Product names in kernels.** No `if soil_product isa SoilGrids` or
   string lookups inside `@kernel` bodies. Preprocess to numerical
   `Field`s and pass property providers.
2. **Hard upper clamps that destroy mass.** `Mˡᵃ > Mˡᵃ⁺` must remain
   admissible as positive pressure head. The positivity floor on `Mˡᵃ`
   (`max(M, 0)`) is the only allowed clamp.
3. **Observations as state.** Do not set `land.temperature` from MODIS
   LST or `land.water_storage` from SMAP. Those are observations
   compared via Tier-3 operators, not initial-condition shortcuts.

## 7. Out of scope and why

Each of these is its own PR cluster, distinct from the data / learning
follow-ups above:

- **Snow and sea ice on land.** New prognostic state, phase change,
  energy partitioning. Needs `Eˡᵃ` as a first-class state variable.
- **Vegetation closure.** LAI, NDVI, stomatal resistance, transpiration,
  canopy interception. *Distinct from* canopy-height-driven roughness,
  which is roughness-input only (see §3.1).
- **Multi-layer soil columns.** Replaces the single-slab depth
  integration with a column scheme.
- **River routing.** Consumes the existing runoff diagnostics; needs
  river-network topology.
- **Two-node evaporation-front energy solve** — `Tᵉ` as a second
  residual in the interface fixed point.
- **Land-side `SkinTemperature(DiffusiveFlux)` solve** so `Tⁱⁿ ≠ Tˡᵃ`.
  The χ-interpolation machinery in `EvaporationFrontHumidity` is already
  in place for this.
- **Brooks–Corey retention** alongside Van Genuchten.
- **Implicit / semi-implicit deep Darcy** treatment for small `Sₛ`.
- **Subgrid tile blending** with ocean / sea ice (mixed grid cells).
- **`MatricPotentialActivity`** — Kelvin-equation suction-driven vapor
  reduction. Interface slot exists in `EvaporationFrontHumidity`;
  implementation deferred (matters only at extreme dryness).

## 8. Acceptance criteria per tier

### Tier 0
- All 10 §4 tests pass on CPU.
- No allocation regression in any new kernel.
- CPU + GPU CI green.
- Docs build; both converted examples render.

### Tier 1
- For each newly field-supporting parameter: scalar-vs-constant-`Field`
  equivalence test (bit-exact on CPU; within `eps(FT)` on GPU).
- ExplicitImports test still passes.
- No new `Any` field types or type parameters.
- Kernels remain allocation-free.

### Tier 2
- Each adapter has its own download / cache / regridding tests.
- QA / fill / scale-factor logic is tested in isolation.
- Outputs are `Field`s consumed by Tier-1 property providers, not
  passed directly to kernels.
- Satellite-`Metadata` schema is enforced by construction.
- Reanalysis / hybrid / tower data uses its own non-satellite
  abstraction.

### Tier 3
- Each observation operator has analytic-limit tests (e.g.,
  unit-emissivity LST recovers `Tˡᵃ`; zero-runoff TWS aggregates to
  total storage).
- Operators are pure (no model-state mutation).
- Loss function composition is associative and commutative for
  weighting.

### Tier 4
- Training provenance round-trips: write provenance, reload, reproduce
  the artifact checksum.
- Finite-difference gradient checks within `1e-4` relative error for
  every Tier-1 learnable parameter.
- No new ExplicitImports or allocation regressions.

## 9. Cross-references

- [SlabLand tutorial](./evaporation_front_slab_land.md) — physics
  walk-through of the model as it stands.
- [Notation](../appendix/notation.md) — symbol table including the new
  variably-saturated-slab additions.
- [PR #323 on GitHub](https://github.com/NumericalEarth/NumericalEarth.jl/pull/323)
  for the conversation that produced this roadmap.
