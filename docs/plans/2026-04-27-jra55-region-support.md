# JRA55 Region Support (BoundingBox + Column) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `JRA55PrescribedAtmosphere(...; region=BoundingBox(...))` and `JRA55PrescribedAtmosphere(...; region=Column(...))` produce a correctly-sliced atmosphere, and harden the underlying `restrict()` so it also works for stretched native interfaces.

**Architecture:** Three loosely-coupled tracks.

1. **Track A — `restrict()` for stretched grids.** Replace the extent-ratio formula with `searchsortedlast` / `searchsortedfirst` to snap the bounding box to the nearest native cell interfaces and return the actual sliced interface vector. This generalises the helper to arbitrary 1D interface arrays (uniform or stretched) and is a drop-in for `construct_native_grid(::BoundingBox, …)`.

2. **Track B — `BoundingBox` for the JRA55 time series.** The slicing infrastructure (`compute_bounding_indices`) already exists in `src/DataWrangling/JRA55/JRA55_field_time_series.jl` but the two `set!` methods hard-code `(nothing, nothing)` for longitude/latitude bounds. Pull bounds from `metadata.region` and pass them through.

3. **Track C — `Column` for the JRA55 time series.** Mirror the snapshot path (`column_field_from_file` → `extract_column!`): load a tiny intermediate FTS over the 2×2 cell window around the column point, then bilinearly interpolate to a `(Flat, Flat, Flat)` 1×1 column FTS. Add a JRA55-specific `set!` dispatch for `Column` regions.

A is independent of B/C. B is the small, mostly-wiring change. C is the biggest piece.

**Tech Stack:** Julia, Oceananigans (`LatitudeLongitudeGrid`, `RectilinearGrid`, `FieldTimeSeries`), NCDatasets, KernelAbstractions kernels.

---

## Pre-flight

### Task 0: Reproduce the bug (failing tests)

**Why:** Lock in the current broken behaviour as red tests so we have something to drive the fix and detect regressions.

**Files:**
- Create: `test/test_jra55_region.jl`
- Modify: `test/runtests.jl` to include the new file

**Step 0.1: Sketch the test file**

Create `test/test_jra55_region.jl` with skeletons for the three behaviours we want. Mark them broken / `@test_skip` where the implementation does not yet exist, so the suite stays green while we plan but flips on automatically as we implement.

```julia
include("runtests_setup.jl")

using NumericalEarth.DataWrangling: BoundingBox, Column, Linear

@testset "JRA55 region support" begin
    arch = CPU()

    @testset "BoundingBox slices the right window" begin
        bbox = BoundingBox(longitude=(120, 240), latitude=(-30, 30))
        atm = JRA55PrescribedAtmosphere(arch;
                                        time_indices_in_memory=2,
                                        include_rivers_and_icebergs=false,
                                        region=bbox)
        Ta = atm.tracers.T
        # Coordinates of the field grid should fall inside the requested bbox.
        λnodes_T = λnodes(Ta.grid, Center())
        φnodes_T = φnodes(Ta.grid, Center())
        @test minimum(λnodes_T) ≥ 120 - 1.5  # 1.5° JRA55 spacing tolerance
        @test maximum(λnodes_T) ≤ 240 + 1.5
        @test minimum(φnodes_T) ≥ -30 - 1.5
        @test maximum(φnodes_T) ≤  30 + 1.5
        # The fts data should not be all-NaN / all-zero (i.e. it was actually filled).
        @test any(!iszero, interior(Ta))
        @test !any(isnan, interior(Ta))
    end

    @testset "Column extracts a single point" begin
        col = Column(150.0, 0.0)  # equator, central Pacific
        atm = JRA55PrescribedAtmosphere(arch;
                                        time_indices_in_memory=2,
                                        include_rivers_and_icebergs=false,
                                        region=col)
        Ta = atm.tracers.T
        @test size(Ta.grid, 1) == 1
        @test size(Ta.grid, 2) == 1
        @test any(!iszero, interior(Ta))
        @test !any(isnan, interior(Ta))
    end
end
```

**Step 0.2: Wire into the suite**

Add `include("test_jra55_region.jl")` at the JRA55 block in `test/runtests.jl` (look for the existing `test_jra55.jl` include and put the new line right after).

**Step 0.3: Run and confirm failure**

```
julia --project=test test/runtests.jl  # or your usual test driver
```

Expected: the bbox test fails with the SW-corner mismatch (coords ≈ (0, 0)–(120, 60)), and the Column test fails because no `Column` dispatch exists for the JRA55 FTS path.

**Step 0.4: Commit**

```bash
git add test/test_jra55_region.jl test/runtests.jl
git commit -m "test: add failing JRA55 BoundingBox and Column region tests"
```

---

## Track A — Stretched-grid `restrict()`

### Task A1: Unit-test the new `restrict` semantics on a stretched 1D array

**Files:**
- Modify: `test/test_metadata.jl` (append a new `@testset` near the existing `BoundingBox` block at lines 30–41)

**Step A1.1: Write the failing test**

```julia
@testset "restrict() snaps to native interfaces" begin
    using NumericalEarth.DataWrangling: restrict
    # Uniform: behaviour should be (almost) unchanged.
    interfaces = collect(0.0:1.0:10.0)
    sliced, rN = restrict((2.5, 6.5), interfaces, 10)
    @test sliced[1]   ≤ 2.5
    @test sliced[end] ≥ 6.5
    @test rN == length(sliced) - 1

    # Stretched: cells get wider with index. searchsorted-based snapping
    # must return the actual native interfaces, not (lo, hi).
    stretched = [0.0, 0.5, 1.5, 3.0, 5.5, 9.5, 15.0]
    sliced, rN = restrict((1.0, 6.0), stretched, length(stretched) - 1)
    @test sliced == [0.5, 1.5, 3.0, 5.5, 9.5]
    @test rN == 4

    # Out-of-range bbox is clamped, not crashed.
    sliced, rN = restrict((-100.0, 100.0), stretched, length(stretched) - 1)
    @test sliced == stretched
    @test rN == length(stretched) - 1
end
```

**Step A1.2: Run, expect failure**

The current `restrict` returns `(bbox_interfaces, rN)` — a 2-tuple of the user's bbox endpoints, not a sliced vector. Tests fail on the equality with the stretched interfaces.

```
julia --project=test -e 'using Pkg; Pkg.test("NumericalEarth"; test_args=["test_metadata.jl"])'
```

Expected: `restrict() snaps to native interfaces` testset fails.

### Task A2: Reimplement `restrict`

**Files:**
- Modify: `src/DataWrangling/metadata_field.jl:27-34`

**Step A2.1: Replace the function**

```julia
# Snap a requested bounding interval to the nearest native cell interfaces.
# Works for both uniform and stretched `interfaces`. Returns the sliced
# interface vector (length rN+1) and the cell count rN.
function restrict(bbox_interfaces, interfaces, N)
    i_lo = max(searchsortedlast(interfaces,  bbox_interfaces[1]), 1)
    i_hi = min(searchsortedfirst(interfaces, bbox_interfaces[2]), length(interfaces))
    rN = max(i_hi - i_lo, 1)
    return interfaces[i_lo:i_hi], rN
end
```

Delete the `# TODO support stretched native grids` line above it.

**Step A2.2: Run the unit test, expect green**

```
julia --project=test -e 'using Pkg; Pkg.test("NumericalEarth"; test_args=["test_metadata.jl"])'
```

Expected: `restrict() snaps to native interfaces` passes; the existing `BoundingBox` snapshot tests at `test_metadata.jl:30–41` should still pass too — verify by looking at the failure output if any. Note the small behaviour change: snapped (≤ one native cell) endpoints instead of exact user endpoints.

**Step A2.3: Spot-check a known dataset**

Run a quick REPL sanity check with an ECCO-shaped vertical interface array to confirm the sliced output looks right. (Optional, no commit.)

**Step A2.4: Commit**

```bash
git add src/DataWrangling/metadata_field.jl test/test_metadata.jl
git commit -m "feat(restrict): snap bounding box to native interfaces (works for stretched grids)"
```

### Task A3: Update bbox snapshot tests if endpoints changed

**Files:**
- Modify: `test/test_metadata.jl:30-41` (only if existing assertions hard-code the bbox endpoints exactly)

**Step A3.1:** Read the existing testset. If it asserts `grid.Lx == bbox_extent` exactly, relax to a `≈` within one native cell. If it just checks `Nx, Ny`, no change needed.

**Step A3.2:** Re-run `test_metadata.jl`, ensure all existing tests still pass.

**Step A3.3: Commit (if changed)**

```bash
git add test/test_metadata.jl
git commit -m "test: relax bbox-endpoint assertions to allow native-interface snapping"
```

---

## Track B — `BoundingBox` for JRA55 FTS

### Task B1: Inspect `compute_bounding_indices` plumbing

**Read-only step.** Confirm:
- `src/DataWrangling/JRA55/JRA55_field_time_series.jl:64-73` — `compute_bounding_indices(longitude, latitude, grid, LX, LY, λc, φc)` already accepts the longitude/latitude bounds and does the `searchsortedfirst` dance.
- Lines 185 and 254 hard-code `(nothing, nothing)` for those two arguments.
- `compute_bounding_nodes(bounds, ::Nothing, LH, hnodes) = bounds` at line 10 means passing a `(lo, hi)` tuple with `grid=nothing` is the right call shape — the bounds flow through unchanged.

No edit, no commit. This is a verification step before B2.

### Task B2: Pull region bounds into JRA55 `set!` (RepeatYear)

**Files:**
- Modify: `src/DataWrangling/JRA55/JRA55_field_time_series.jl:173-213`

**Step B2.1: Add a helper for region → (λbounds, φbounds)**

Add near the top of the file, alongside the other `compute_bounding_*` helpers:

```julia
# Extract horizontal bounds from a Metadata region.
region_horizontal_bounds(::Nothing) = (nothing, nothing)
region_horizontal_bounds(bbox::BoundingBox) = (bbox.longitude, bbox.latitude)
# Column is handled by a separate set! dispatch (Track C); fall through.
region_horizontal_bounds(::Column) = (nothing, nothing)
```

You'll need `using NumericalEarth.DataWrangling: BoundingBox, Column` near the top, or qualify the names — match whatever style the file already uses.

**Step B2.2: Use the helper in the RepeatYear `set!`**

Change line 185 from:

```julia
i₁, i₂, j₁, j₂, TX = compute_bounding_indices(nothing, nothing, fts.grid, LX, LY, λc, φc)
```

to:

```julia
λbounds, φbounds = region_horizontal_bounds(metadata.region)
i₁, i₂, j₁, j₂, TX = compute_bounding_indices(λbounds, φbounds, fts.grid, LX, LY, λc, φc)
```

**Step B2.3: Run the bbox test from Task 0, expect partial improvement**

```
julia --project=test -e 'using Pkg; Pkg.test("NumericalEarth"; test_args=["test_jra55_region.jl"])'
```

Expected: the `BoundingBox` test passes for `RepeatYearJRA55` (the default dataset).

**Step B2.4: Commit**

```bash
git add src/DataWrangling/JRA55/JRA55_field_time_series.jl
git commit -m "feat(JRA55): respect metadata.region BoundingBox in RepeatYear set!"
```

### Task B3: Same change for the MultiYear `set!`

**Files:**
- Modify: `src/DataWrangling/JRA55/JRA55_field_time_series.jl:223-285` (mirror change at line 254)

**Step B3.1:** Apply the same `region_horizontal_bounds` lookup at line 254. Identical pattern to B2.

**Step B3.2: Add a MultiYear bbox test** to `test/test_jra55_region.jl`:

```julia
@testset "MultiYear BoundingBox" begin
    using NumericalEarth.DataWrangling.JRA55: MultiYearJRA55
    bbox = BoundingBox(longitude=(120, 240), latitude=(-30, 30))
    atm = JRA55PrescribedAtmosphere(arch;
                                    dataset = MultiYearJRA55(),
                                    start_date = DateTime(1991, 1, 1),
                                    end_date   = DateTime(1991, 1, 2),
                                    time_indices_in_memory=2,
                                    include_rivers_and_icebergs=false,
                                    region=bbox)
    Ta = atm.tracers.T
    @test any(!iszero, interior(Ta))
end
```

**Step B3.3:** Run the new test, expect green.

**Step B3.4: Commit**

```bash
git add src/DataWrangling/JRA55/JRA55_field_time_series.jl test/test_jra55_region.jl
git commit -m "feat(JRA55): respect metadata.region BoundingBox in MultiYear set!"
```

### Task B4: Verify topology inference is sane for bbox

**Files:**
- Read: `src/DataWrangling/JRA55/JRA55_field_time_series.jl:56-62`

**Step B4.1:** `infer_longitudinal_topology` returns `Periodic` if the bbox spans 360° and `Bounded` otherwise. Confirm that the FTS grid built by `native_grid` for a sub-360° bbox uses a non-periodic x topology — otherwise `fill_halo_regions!` will wrap data across the bbox boundary (wrong).

**Step B4.2:** If `construct_native_grid(metadata, bbox::BoundingBox, …)` at `metadata_field.jl:60–77` always builds a `LatitudeLongitudeGrid` with the default (periodic) x, change it to pass `topology = (TX, Bounded, Bounded)` where `TX = bbox spans 360 ? Periodic : Bounded`. Add a small assertion test in `test_metadata.jl` that a non-360 bbox produces a `Bounded` x topology.

**Step B4.3:** Run all metadata + jra55 region tests.

**Step B4.4: Commit**

```bash
git add src/DataWrangling/metadata_field.jl test/test_metadata.jl
git commit -m "fix(metadata): bbox grid uses Bounded x-topology when not spanning 360°"
```

---

## Track C — `Column` for JRA55 FTS

### Task C1: Identify the `Column` JRA55 FTS shape

**Read-only.** Confirm:
- `restrict_location((LX, LY, LZ), ::Column) = (Nothing, Nothing, LZ)` at `metadata_field.jl:19` — for JRA55 (surface-only data, LZ ≈ Nothing or Bounded(1)) the column field becomes `Field{Nothing, Nothing, Nothing}` on a 1×1 grid.
- `construct_native_grid(::Column, …)` at `metadata_field.jl:80–93` builds a `RectilinearGrid` with `topology = (Flat, Flat, Bounded)`.
- For a 2D surface variable like JRA55 temperature, `Nz` from `size(metadata)` may be 1; ensure the Column `RectilinearGrid` builder still works in that case (or branch to a `(Flat, Flat, Flat)` grid for surface-only data).

If the Column native grid for JRA55 is wrong (e.g. expects `z`), patch `construct_native_grid` to handle 2D-only metadata. Otherwise no change.

### Task C2: JRA55 `set!` dispatch for `Column`

**Files:**
- Modify: `src/DataWrangling/JRA55/JRA55_field_time_series.jl`

**Step C2.1: Define a typed alias for column-region JRA55 FTS**

Near the existing `JRA55NetCDFFTS*` consts (line 95+):

```julia
const JRA55NetCDFFTSColumn = FlavorOfFTS{
    <:Any, <:Any, <:Any, <:Any,
    <:DatasetBackend{<:Any, <:Any, <:Any, <:Metadata{<:Any, <:Any, <:Column}},
}
```

Adjust the `Metadata` type-parameter slot to match the actual struct layout — check `src/DataWrangling/metadata.jl:1–60` for the order of `Metadata`'s parameters.

**Step C2.2: Write a column-aware `set!`**

```julia
# Column case: for each requested time index, read the 2x2 cell window
# around the column's (longitude, latitude) and bilinearly interpolate
# into the 1x1 column FTS.
function set!(fts::JRA55NetCDFFTSColumn, backend=fts.backend)
    metadata = backend.metadata
    col      = metadata.region

    path = joinpath(metadata.dir, metadata.filename)
    ds   = Dataset(path)

    λc = ds["lon"][:]
    φc = ds["lat"][:]

    # Indices of the lower-left cell containing (col.longitude, col.latitude).
    i_ll, i_ur, wx = bracket_with_weight(λc, col.longitude)
    j_ll, j_ur, wy = bracket_with_weight(φc, col.latitude)

    name = dataset_variable_name(metadata)
    nn   = collect(time_indices(fts))

    data = if issorted(nn)
        ds[name][i_ll:i_ur, j_ll:j_ur, nn]
    else
        m  = findfirst(==(1), nn)
        d1 = ds[name][i_ll:i_ur, j_ll:j_ur, nn[1:m-1]]
        d2 = ds[name][i_ll:i_ur, j_ll:j_ur, nn[m:end]]
        cat(d1, d2; dims=3)
    end

    close(ds)

    # Bilinear blend into the 1x1 column FTS.
    interp = column_blend(data, wx, wy, col.interpolation)
    copyto!(interior(fts, :, :, 1, :), reshape(interp, 1, 1, :))

    fill_halo_regions!(fts)
    return nothing
end

# Bracket a point in a 1D coordinate array; returns (i_lower, i_upper, weight ∈ [0,1]).
function bracket_with_weight(coords, x)
    i_upper = searchsortedfirst(coords, x)
    i_upper = clamp(i_upper, 2, length(coords))
    i_lower = i_upper - 1
    w = (x - coords[i_lower]) / (coords[i_upper] - coords[i_lower])
    return i_lower, i_upper, clamp(w, 0, 1)
end

column_blend(data, wx, wy, ::Linear) =
    @views @. (1 - wx) * (1 - wy) * data[1, 1, :] +
                   wx  * (1 - wy) * data[2, 1, :] +
              (1 - wx) *      wy  * data[1, 2, :] +
                   wx  *      wy  * data[2, 2, :]

column_blend(data, wx, wy, ::Nearest) = begin
    i = wx ≥ 0.5 ? 2 : 1
    j = wy ≥ 0.5 ? 2 : 1
    @views data[i, j, :]
end
```

**Step C2.3: Add a MultiYear column `set!`** along the same lines (mirror of `set!(::JRA55NetCDFFTSMultipleYears, ...)`), since the column case is independent of how many files we read from. If the body would be a near-duplicate, factor a private `_load_column_data(ds, name, i_ll, i_ur, j_ll, j_ur, nn)` and have both `set!` methods call it.

**Step C2.4: Run the Column test from Task 0, expect green**

```
julia --project=test -e 'using Pkg; Pkg.test("NumericalEarth"; test_args=["test_jra55_region.jl"])'
```

Expected: column test passes, bbox tests still pass.

**Step C2.5: Commit**

```bash
git add src/DataWrangling/JRA55/JRA55_field_time_series.jl
git commit -m "feat(JRA55): support Column region in FieldTimeSeries set!"
```

### Task C3: Validate column values against bbox-extracted reference

**Files:**
- Modify: `test/test_jra55_region.jl`

**Step C3.1:** Add an end-to-end correctness test that the column value at `(150, 0)` matches the bilinearly-interpolated value from the same JRA55 snapshot loaded as a small bbox. This protects against off-by-one and lon/lat-swap regressions.

```julia
@testset "Column matches bbox-extracted bilinear value" begin
    col_atm  = JRA55PrescribedAtmosphere(arch; time_indices_in_memory=2,
                                         include_rivers_and_icebergs=false,
                                         region=Column(150.0, 0.0))
    bbox_atm = JRA55PrescribedAtmosphere(arch; time_indices_in_memory=2,
                                         include_rivers_and_icebergs=false,
                                         region=BoundingBox(longitude=(148, 152),
                                                            latitude=(-2, 2)))
    # Bilinear-interpolate bbox temperature to (150, 0) at t=1.
    T_bbox = interior(bbox_atm.tracers.T, :, :, 1, 1)
    grid_b = bbox_atm.tracers.T.grid
    interp_T = Oceananigans.Fields.interpolate((150.0, 0.0, 0.0),
                                               bbox_atm.tracers.T, location(bbox_atm.tracers.T), grid_b)
    T_col = interior(col_atm.tracers.T, 1, 1, 1, 1)
    @test T_col ≈ interp_T  rtol = 1e-3
end
```

(Adjust to use a fixed time index rather than relying on time alignment if the two atmospheres have different `time_indices_in_memory` semantics.)

**Step C3.2:** Run the test.

**Step C3.3: Commit**

```bash
git add test/test_jra55_region.jl
git commit -m "test(JRA55): verify Column extraction agrees with bbox bilinear interp"
```

---

## Integration

### Task I1: Document the `region` kwarg

**Files:**
- Modify: `src/DataWrangling/JRA55/JRA55_prescribed_atmosphere.jl:6-22` (docstring)

**Step I1.1:** Update the docstring to describe `region`, with one-line examples for both `BoundingBox` and `Column`. Note the snap-to-native-interfaces behaviour.

**Step I1.2: Commit**

```bash
git add src/DataWrangling/JRA55/JRA55_prescribed_atmosphere.jl
git commit -m "docs(JRA55): document region kwarg on JRA55PrescribedAtmosphere"
```

### Task I2: Final smoke test

**Step I2.1:** Run the full JRA55 test file:

```
julia --project=test -e 'using Pkg; Pkg.test("NumericalEarth"; test_args=["test_jra55.jl", "test_jra55_region.jl"])'
```

Expected: all green. No new warnings about NaN data, halo size, or topology mismatch.

**Step I2.2:** Spot-check via REPL that an `OceanSeaIceModel` driven by a `JRA55PrescribedAtmosphere(region=BoundingBox(...))` runs one timestep without error. (Use whatever existing example or model setup exercises the atmosphere fluxes.)

**Step I2.3: Commit (if anything changed)**

No commit if no edits — this is just a verification step.

---

## Risks & open questions

- **JRA55 longitude convention.** JRA55 is on `[0, 360)`. If a user passes `BoundingBox(longitude=(-10, 10))`, `searchsortedfirst` will return wrong indices because the data is not in that range. Decide: normalise input bbox longitudes to `[0, 360)` (fail loudly if the bbox crosses the wrap), or document the constraint. Not in scope of this plan unless tests reveal it as a problem.
- **MultiYear `Column` performance.** Each year-file open + close, even when slicing only a 2×2 window, may dominate the per-call cost. If profiling shows this, consider keeping the file handles open across the loop in `set!(::JRA55NetCDFFTSMultipleYears, ...)`. Out of scope for the initial landing.
- **Column with `Nearest` over land.** If the four bracketing cells are all on land for a coastal point, the result will be NaN. Inpainting the source FTS first solves this, but `inpainting=nothing` is the JRA55 default. Document and move on.
- **`PrescribedAtmosphere` consumers.** Confirm that downstream code (sea-ice fluxes, ocean models) does not assume the atmosphere grid spans the globe. A bbox or column atmosphere should be valid input but interpolation onto an ocean grid that extends outside the bbox will need NaN-handling.
