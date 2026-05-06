# Region-Aware FTS Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.
>
> **User-specific rule for this plan:** Do **not** run `git commit` / `git push` / `git tag`. The user creates all commits themselves. Skip every commit step in the template; report "ready to commit" at the end of each task instead.

**Goal:** Pull region-handling (BoundingBox + Column) out of the JRA55 module into a generic core that all dataset backends share, and apply the same core to the snapshot (`Field(::Metadata)`) path so ECCO/GLORYS/ERA5 inherit working bbox + column support automatically.

**Architecture (three tracks):**

1. **Track R — Relocate dataset-agnostic helpers.** Move `bracket_with_weight`, `column_blend`, `compute_bounding_indices`, and topology inference out of the JRA55 module into the shared layer (`metadata_field_time_series.jl` for FTS-specific; `metadata_field.jl` for grid-construction helpers). Nothing in these helpers actually depends on JRA55.

2. **Track G — Generalize `construct_native_grid` for 2D-only datasets.** Branch the bbox / column / nothing dispatchers on the existing `is_three_dimensional(metadata)` predicate so that 2D datasets (JRA55 today; tomorrow possibly any 2D forcing dataset) fall through the generic path. Once that lands, JRA55's `native_grid` override at `JRA55_metadata.jl:54-99` can go away entirely.

3. **Track F — Add a generic `set_region_data!` core** that takes a `read_window(i_range, j_range, t_indices)` closure and dispatches on `metadata.region`. JRA55's two `set!` methods collapse to a thin shell that opens the file and supplies the closure. The column-specific `set!` dispatches (`JRA55NetCDFFTSColumnRepeatYear`, `JRA55NetCDFFTSColumnMultipleYears`) become redundant and get deleted.

4. **Track S — Apply the generic core to the snapshot path** so `Field(::Metadatum with BoundingBox)` actually loads the right window for ECCO/GLORYS/ERA5/etc. (Today the snapshot path reads the full dataset and indexes positionally — a latent bug for any non-SW-corner bbox; see the task list of `2026-04-27-jra55-region-support.md` for the analysis.)

5. **Track V — Cross-dataset verification.** A bbox + column smoke test for at least one ocean dataset (ECCO4Monthly, since the data is small and already downloaded in CI).

Tracks R → G → F → S → V are roughly sequential. Within each track, tasks follow TDD red-green discipline: failing test, minimal code change, run-to-green, leave for the user to commit.

**Tech Stack:** Julia, Oceananigans (`LatitudeLongitudeGrid`, `RectilinearGrid`, `FieldTimeSeries`), NCDatasets, KernelAbstractions kernels.

---

## Pre-flight

### Task 0: Lock in the cross-dataset gap as failing tests

**Files:**
- Create: `test/test_dataset_region.jl`
- (`test/runtests.jl` uses `find_tests(@__DIR__)` autodiscovery, so no manual include is needed.)

**Step 0.1: Write tests for ECCO bbox + column snapshots**

```julia
include("runtests_setup.jl")

using NumericalEarth.DataWrangling: BoundingBox, Column
using Oceananigans: λnodes, φnodes, Center, interior

@testset "Cross-dataset region support (snapshot path)" begin
    arch = CPU()

    @testset "ECCO4 BoundingBox loads the right window" begin
        bbox = BoundingBox(longitude=(120, 240), latitude=(-30, 30))
        md = Metadatum(:temperature; dataset=ECCO4Monthly(),
                       date=DateTime(1992, 1, 1), region=bbox)
        f = Field(md)
        # Field grid coordinates fall inside the bbox.
        λc, φc = λnodes(f.grid, Center()), φnodes(f.grid, Center())
        @test minimum(λc) ≥ 120 - 1.0  # 0.5° ECCO4 spacing tolerance
        @test maximum(λc) ≤ 240 + 1.0
        @test minimum(φc) ≥ -30 - 1.0
        @test maximum(φc) ≤  30 + 1.0
        # The field is non-zero/non-NaN-only somewhere in the bbox interior
        # (allowing for ocean masks at the edges).
        @test any(!iszero,  interior(f))
    end

    @testset "ECCO4 Column extracts a single point" begin
        col = Column(150.0, 0.0)
        md = Metadatum(:temperature; dataset=ECCO4Monthly(),
                       date=DateTime(1992, 1, 1), region=col)
        f = Field(md)
        @test size(f.grid, 1) == 1
        @test size(f.grid, 2) == 1
        @test any(!iszero, interior(f))
    end
end
```

**Step 0.2: Run the test, confirm failure**

```
julia --project=test -e 'using Pkg; Pkg.test("NumericalEarth"; test_args=["test_dataset_region.jl"])'
```

Expected failure mode for the bbox case: `Field` is built on the correctly-restricted grid (the snapshot grid path *does* respect bbox — see `metadata_field.jl:60-77`), but `set_metadata_field!` reads the full dataset via `retrieve_data` and indexes positionally, so the field's *data* corresponds to the SW corner of the dataset, not the bbox. The lat/lon-extrema assertions pass (grid is correct) but `any(!iszero, interior(f))` may pass too — the SW corner is mostly Antarctic ocean, which is non-zero. The real failure is silent, so the test should also assert *correctness* against a hand-extracted reference.

Strengthen the bbox test by adding a reference comparison (Step 0.3 below).

**Step 0.3: Add a hand-extracted reference assertion**

Append inside the bbox testset:

```julia
        # Reference: open the NetCDF directly and bilinear-interpolate at
        # the bbox interior point (180, 0).
        using NCDatasets
        path = NumericalEarth.DataWrangling.metadata_path(md)
        ds = Dataset(path)
        T_full = ds["theta"][:, :, 1, 1]  # adjust var name + indexing for ECCO4
        λfile = ds["longitude"][:]
        φfile = ds["latitude"][:]
        close(ds)
        # Find the file index closest to (180, 0).
        i★ = argmin(abs.(λfile .- 180))
        j★ = argmin(abs.(φfile .- 0))
        T_ref = T_full[i★, j★]
        # Find the same point in the bbox-restricted field.
        i_grid = argmin(abs.(λc .- 180))
        j_grid = argmin(abs.(φc .- 0))
        T_field = interior(f)[i_grid, j_grid, end]  # surface
        @test T_field ≈ T_ref  rtol=1e-2
```

Adjust variable names (`theta` → whichever ECCO uses) and dimensionality (3D ECCO has a Z axis) once you read the actual file structure. The point is: a *correctness* assertion that catches silent SW-corner indexing.

**Step 0.4: Re-run, confirm it actually fails**

The reference assertion should fail because `T_field` is the SW-corner data, not the (180, 0) data.

**Step 0.5: Skip commit. Leave changes uncommitted.**

---

## Track R — Relocate dataset-agnostic helpers

### Task R1: Move `bracket_with_weight` and `column_blend` to the shared module

**Files:**
- Modify: `src/DataWrangling/JRA55/JRA55_field_time_series.jl` (delete the helpers)
- Modify: `src/DataWrangling/metadata_field_time_series.jl` (add the helpers)

**Step R1.1: Cut the helpers from JRA55**

Delete `bracket_with_weight`, `column_blend`, and `read_column_window` from the JRA55 file. (Lines visible after grep — they're at the bottom of the file in the column-set! block.)

**Step R1.2: Paste them into `metadata_field_time_series.jl`**

At the top of the file, after the existing `using` lines, add:

```julia
# Bracket a point in a 1D coordinate array; returns (i_lower, i_upper, w ∈ [0,1])
# such that coords[i_lower] ≤ x ≤ coords[i_upper] (clamped to the array's interior).
function bracket_with_weight(coords, x)
    i_upper = searchsortedfirst(coords, x)
    i_upper = clamp(i_upper, 2, length(coords))
    i_lower = i_upper - 1
    Δ = coords[i_upper] - coords[i_lower]
    w = Δ == 0 ? zero(x) : (x - coords[i_lower]) / Δ
    return i_lower, i_upper, clamp(w, 0, 1)
end

column_blend(data, wx, wy, ::Linear) =
    @views @. (1 - wx) * (1 - wy) * data[1, 1, :] +
                   wx  * (1 - wy) * data[2, 1, :] +
              (1 - wx) *      wy  * data[1, 2, :] +
                   wx  *      wy  * data[2, 2, :]

function column_blend(data, wx, wy, ::Nearest)
    i = wx ≥ 0.5 ? 2 : 1
    j = wy ≥ 0.5 ? 2 : 1
    return @views data[i, j, :]
end
```

(`Linear`/`Nearest` are already imported via `using NumericalEarth.DataWrangling`. `read_column_window` was JRA55-specific; we'll re-introduce it as a closure below in Track F. Don't keep it.)

**Step R1.3: Run JRA55 region tests, confirm green**

```
julia --project=test -e 'using Pkg; Pkg.test("NumericalEarth"; test_args=["test_jra55_region.jl"])'
```

Expected: 12/12 still pass. (The current JRA55 column `set!` dispatches still call these helpers — they're now imported from a different scope, but Julia's module system resolves them.)

**Step R1.4: No commit.**

### Task R2: Move `compute_bounding_indices` and `compute_bounding_nodes` to the shared module

**Files:**
- Modify: `src/DataWrangling/JRA55/JRA55_field_time_series.jl` (delete)
- Modify: `src/DataWrangling/metadata_field_time_series.jl` (add)

**Step R2.1: Cut the four `compute_bounding_*` methods** (lines ~9-73 of the JRA55 file) and paste them into `metadata_field_time_series.jl`. They depend only on `λnodes`/`φnodes` and `@allowscalar`, both already imported in the shared module (or easily added).

**Step R2.2: Verify imports**

`infer_longitudinal_topology` and `compute_bounding_indices` need access to `Periodic`, `Bounded`, `λnodes`, `φnodes`. Add to the shared module's `using` block if missing.

**Step R2.3: Run all FTS tests, confirm green**

```
julia --project=test -e 'using Pkg; Pkg.test("NumericalEarth"; test_args=["test_jra55_region.jl", "test_jra55.jl"])'
```

Expected: still all green. JRA55 `set!` calls `compute_bounding_indices(nothing, nothing, fts.grid, ...)` — the function still exists, just lives elsewhere now.

**Step R2.4: No commit.**

### Task R3: Extract a single topology-inference helper

**Files:**
- Modify: `src/DataWrangling/metadata_field.jl`
- Modify: `src/DataWrangling/JRA55/JRA55_metadata.jl`

**Step R3.1: Add the helper**

In `metadata_field.jl`, alongside the `restrict` definitions, add:

```julia
# Periodic in the restricted longitude only when the restricted span equals
# the full native span. Used by bbox-restricted grid construction.
function infer_lon_topology(full_longitude, restricted_longitude)
    full_span = full_longitude[end] - full_longitude[1]
    restricted_span = restricted_longitude[end] - restricted_longitude[1]
    return restricted_span ≈ full_span ? Periodic : Bounded
end
```

**Step R3.2: Use it in both call sites**

In `metadata_field.jl:60-77`, replace the inline computation:

```julia
full_lon_span = full_longitude[end] - full_longitude[1]
restricted_lon_span = longitude[end] - longitude[1]
TX = restricted_lon_span ≈ full_lon_span ? Periodic : Bounded
```

with `TX = infer_lon_topology(full_longitude, longitude)`.

In `JRA55_metadata.jl:54-99`, do the same.

**Step R3.3: Run tests; confirm green.**

**Step R3.4: No commit.**

---

## Track G — Generalize `construct_native_grid` for 2D-only datasets

### Task G1: 2D-aware Column dispatch

**Files:**
- Modify: `src/DataWrangling/metadata_field.jl:80-93` (the existing `construct_native_grid(::Column)`)

**Step G1.1: Failing test (move the JRA55-Column assertion to the generic side)**

Add to `test/test_metadata.jl`:

```julia
@testset "Column native_grid for 2D-only datasets" begin
    # JRA55 metadata routes through the generic Column path (after Track G).
    md = Metadatum(:temperature; dataset=RepeatYearJRA55(),
                   region=Column(150.0, 0.0))
    g = native_grid(md)
    # 2D dataset → no z dimension; grid is a 1×1 (Bounded, Bounded, Flat) LatLon.
    @test size(g) == (1, 1, 1)
    @test topology(g)[3] == Flat
end
```

This test passes today only because of the JRA55 override — but Track G3 will delete that override. Run it now to confirm it passes; we'll re-run at G3 to confirm it still passes after deletion.

**Step G1.2: Generalize the Column branch**

In `metadata_field.jl:80-93`, replace the body with a 2D/3D branch:

```julia
function construct_native_grid(metadata, col::Column, arch; halo)
    FT = eltype(metadata)
    if is_three_dimensional(metadata)
        _, _, Nz, _ = size(metadata)
        z = z_interfaces(metadata)
        return RectilinearGrid(arch, FT;
                               size = Nz,
                               x = FT(col.longitude),
                               y = FT(col.latitude),
                               z,
                               halo = halo[3],
                               topology = (Flat, Flat, Bounded))
    else
        ε = convert(FT, 0.5)
        h = min.(halo[1:2], (1, 1))
        return LatitudeLongitudeGrid(arch, FT; size = (1, 1),
                                     halo = h,
                                     longitude = (FT(col.longitude) - ε, FT(col.longitude) + ε),
                                     latitude  = (FT(col.latitude)  - ε, FT(col.latitude)  + ε),
                                     topology = (Bounded, Bounded, Flat))
    end
end
```

(The 2D branch is what `jra55_column_grid` does today, lifted here.)

**Step G1.3: Run `test_metadata.jl`, confirm green.**

**Step G1.4: No commit.**

### Task G2: 2D-aware BoundingBox dispatch

**Files:**
- Modify: `src/DataWrangling/metadata_field.jl:67-87` (current `construct_native_grid(::BoundingBox)`)

**Step G2.1: Branch on 2D / 3D**

```julia
function construct_native_grid(metadata, bbox::BoundingBox, arch; halo)
    FT = eltype(metadata)
    full_longitude = longitude_interfaces(metadata)
    full_latitude  = latitude_interfaces(metadata)

    Nx, Ny = size(metadata)[1:2]
    longitude, Nx = restrict(bbox.longitude, full_longitude, Nx)
    latitude,  Ny = restrict(bbox.latitude,  full_latitude,  Ny)

    TX = infer_lon_topology(full_longitude, longitude)

    if is_three_dimensional(metadata)
        _, _, Nz, _ = size(metadata)
        z = z_interfaces(metadata)
        halo3 = min.(halo, (Nx, Ny, Nz))
        return LatitudeLongitudeGrid(arch, FT; size = (Nx, Ny, Nz),
                                     halo = halo3, longitude, latitude, z,
                                     topology = (TX, Bounded, Bounded))
    else
        halo2 = min.(halo[1:2], (Nx, Ny))
        return LatitudeLongitudeGrid(arch, FT; size = (Nx, Ny),
                                     halo = halo2, longitude, latitude,
                                     topology = (TX, Bounded, Flat))
    end
end
```

**Step G2.2: Run all metadata + JRA55 region tests; confirm green.**

**Step G2.3: No commit.**

### Task G3: Delete the JRA55 `native_grid` override

**Files:**
- Modify: `src/DataWrangling/JRA55/JRA55_metadata.jl` (delete lines ~54-99)

**Step G3.1: Special-case JRA55 native interfaces**

The generic path uses `longitude_interfaces(metadata)` / `latitude_interfaces(metadata)`. JRA55 currently returns the trivial `(0, 360)` / `(-90, 90)` tuples but actually uses Gaussian latitudes read from the file via `jra55_native_interfaces`. To make the generic path work, change JRA55's `latitude_interfaces` to read from disk lazily:

```julia
longitude_interfaces(md::JRA55Metadata) = jra55_native_interfaces(metadata_path(first(md)))[1]
latitude_interfaces(md::JRA55Metadata)  = jra55_native_interfaces(metadata_path(first(md)))[2]
```

(The `first(md)` pattern is what JRA55 already uses to get a single Metadatum from a Metadata range.)

**Step G3.2: Delete the override**

Remove `native_grid(metadata::JRA55Metadata, ...)` and `jra55_column_grid(...)` from `JRA55_metadata.jl`. Also remove `restrict` from JRA55's imports (it's no longer used here).

**Step G3.3: Run tests, confirm green**

```
julia --project=test -e 'using Pkg; Pkg.test("NumericalEarth"; test_args=["test_metadata.jl", "test_jra55_region.jl", "test_jra55.jl"])'
```

If the `latitude_interfaces` change opens the file every grid construction (it does — twice, once for lon and once for lat), profile and consider memoizing if measurable. Likely fine because grid construction is rare.

**Step G3.4: No commit.**

---

## Track F — Generic `set_region_data!` core

### Task F1: Add the core dispatcher

**Files:**
- Modify: `src/DataWrangling/metadata_field_time_series.jl`

**Step F1.1: Add the function**

Below the imports:

```julia
"""
    set_region_data!(fts, read_window, λc, φc, metadata)

Generic region-aware data filling for a `FieldTimeSeries` whose `metadata.region`
is `nothing`, a `BoundingBox`, or a `Column`. `read_window(i_range, j_range, t_idx)`
is a closure provided by the dataset backend that returns a 3D array of shape
`(length(i_range), length(j_range), length(t_idx))` from the underlying file(s).

`λc` / `φc` are the file's native cell centres (1D vectors). `t_idx` is the
caller-supplied time-axis selector (`:` for "all in-memory slots", or a vector of
integer indices when chunk-glueing is required).
"""
set_region_data!(fts, read_window, λc, φc, metadata, t_idx=:) =
    set_region_data!(fts, read_window, λc, φc, metadata, t_idx, metadata.region)

# No region: full extent.
function set_region_data!(fts, read_window, λc, φc, _, t_idx, ::Nothing)
    data = read_window(1:length(λc), 1:length(φc), t_idx)
    copyto!(interior(fts, :, :, 1, :), data)
end

# BoundingBox: slice via grid centres.
function set_region_data!(fts, read_window, λc, φc, _, t_idx, ::BoundingBox)
    LX, LY, _ = location(fts)
    i₁, i₂, j₁, j₂ = compute_bounding_indices(nothing, nothing, fts.grid, LX, LY, λc, φc)[1:4]
    data = read_window(i₁:i₂, j₁:j₂, t_idx)
    copyto!(interior(fts, :, :, 1, :), data)
end

# Column: 2×2 read + bilinear (or nearest) blend.
function set_region_data!(fts, read_window, λc, φc, _, t_idx, col::Column)
    i_ll, i_ur, wx = bracket_with_weight(λc, col.longitude)
    j_ll, j_ur, wy = bracket_with_weight(φc, col.latitude)
    data = read_window(i_ll:i_ur, j_ll:j_ur, t_idx)
    blended = column_blend(data, wx, wy, col.interpolation)
    copyto!(interior(fts, :, :, 1, :), reshape(blended, 1, 1, :))
end
```

**Step F1.2: Unit-test against an in-memory fake closure**

Add to `test/test_dataset_region.jl`:

```julia
@testset "set_region_data! on a synthetic FTS" begin
    using NumericalEarth.DataWrangling: set_region_data!, BoundingBox, Column
    # Build a 1×640×320×2-shaped synthetic raw array; reuse JRA55's grid.
    md = Metadatum(:temperature; dataset=RepeatYearJRA55(),
                   region=BoundingBox(longitude=(120, 240), latitude=(-30, 30)))
    fts = FieldTimeSeries(md, CPU(); time_indices_in_memory=2)
    raw = randn(Float32, 640, 320, 8760)
    λc = collect(0.28125:0.5625:359.71875)  # JRA55 file lon centres
    φc = ... # likewise
    read_window(i, j, t) = raw[i, j, t]
    set_region_data!(fts, read_window, λc, φc, md, [1, 2])
    # Assert interior(fts) equals the corresponding window of `raw`.
    @test ...
end
```

(Refine indices once you have the actual JRA55 lat array. The point of this test is to verify the dispatcher slices correctly without involving NetCDF.)

**Step F1.3: Run, confirm green.**

**Step F1.4: No commit.**

### Task F2: Refactor JRA55 RepeatYear `set!` to use the core

**Files:**
- Modify: `src/DataWrangling/JRA55/JRA55_field_time_series.jl:185-227`

**Step F2.1: Collapse the body**

```julia
function set!(fts::JRA55NetCDFFTSRepeatYear, backend=fts.backend)
    metadata = backend.metadata
    ds = Dataset(joinpath(metadata.dir, metadata.filename))
    λc = ds["lon"][:]
    φc = ds["lat"][:]
    nn = collect(time_indices(fts))
    name = dataset_variable_name(metadata)
    read_window(i, j, t) = if issorted(nn)
        ds[name][i, j, nn]
    else
        m = findfirst(==(1), nn)
        cat(ds[name][i, j, nn[1:m-1]], ds[name][i, j, nn[m:end]]; dims=3)
    end
    set_region_data!(fts, read_window, λc, φc, metadata)
    close(ds)
    fill_halo_regions!(fts)
    return nothing
end
```

(Note: the `t` argument to `read_window` is currently ignored because RepeatYear hard-codes `nn`. That's fine — `set_region_data!` passes `:` by default. We could thread `t` through more carefully later, but the closure already captures `nn`.)

**Step F2.2: Run JRA55 region + full-globe tests, confirm green.**

**Step F2.3: No commit.**

### Task F3: Refactor JRA55 MultiYear `set!` to use the core

**Files:**
- Modify: `src/DataWrangling/JRA55/JRA55_field_time_series.jl:229-300`

**Step F3.1: Collapse — same pattern, but the per-year file iteration stays JRA55-specific**

```julia
function set!(fts::JRA55NetCDFFTSMultipleYears, backend=fts.backend)
    metadata = backend.metadata
    name     = dataset_variable_name(metadata)
    ftsn       = collect(time_indices(fts))
    slot_dates = metadata.dates[ftsn]
    needed_files = unique(getfilename(metadata.filename, n) for n in ftsn)

    for file in needed_files
        ds   = Dataset(joinpath(metadata.dir, file))
        file_dates = ds["time"][:]

        nn       = Int[]
        ftsn_loc = Int[]
        for (loc, slot_date) in enumerate(slot_dates)
            file_idx = jra55_no_leap_file_index(file_dates, slot_date)
            if !isnothing(file_idx)
                push!(nn, file_idx)
                push!(ftsn_loc, loc)
            end
        end

        if !isempty(nn)
            λc, φc = ds["lon"][:], ds["lat"][:]
            read_window(i, j, _) = if issorted(nn)
                ds[name][i, j, nn]
            else
                m = findfirst(==(1), nn)
                cat(ds[name][i, j, nn[1:m-1]], ds[name][i, j, nn[m:end]]; dims=3)
            end
            # set_region_data! into a temporary FTS view, then copy slot-by-slot.
            # OR: thread ftsn_loc through the closure. Simplest: build the data,
            # apply the region transform once, then place per-slot.
            ...
        end
        close(ds)
    end

    fill_halo_regions!(fts)
    return nothing
end
```

The MultiYear path is awkward because the slots in `interior(fts, :, :, 1, :)` aren't contiguous — we copy per slot via `ftsn_loc`. The cleanest refactor: have `set_region_data!` accept a `slot_indices` kwarg (default = `:`). Or: handle MultiYear via a thin wrapper that calls `set_region_data!` once per file with a sliced FTS view.

Pick whichever feels cleaner once you read the full existing code. The constraint: the column / bbox / nothing dispatch logic must not be duplicated.

**Step F3.2: Run MultiYear test (`test_jra55.jl`'s MultiYear block), confirm green.**

**Step F3.3: No commit.**

### Task F4: Delete the JRA55 column-specific `set!` dispatches

**Files:**
- Modify: `src/DataWrangling/JRA55/JRA55_field_time_series.jl`

**Step F4.1: Delete `JRA55NetCDFFTSColumnRepeatYear`, `JRA55NetCDFFTSColumnMultipleYears`, and their `set!` methods.** They're subsumed by `set_region_data!`'s Column dispatch.

**Step F4.2: Run all JRA55 + region tests, confirm green.**

**Step F4.3: No commit.**

---

## Track S — Apply the generic core to the snapshot path

### Task S1: Refactor `set_metadata_field!` to use `set_region_data!`

**Files:**
- Modify: `src/DataWrangling/metadata_field.jl:387-433`

**Step S1.1: Read the current implementation**

`set_metadata_field!(field, data, metadatum)` launches a kernel over `field.grid` and indexes `data[i,j,k]` positionally. We need to replace that with:

1. Read the file's `λc` / `φc` (or use the metadata's natural axis if not stored on disk — for some datasets the centres are inferable from `longitude_interfaces` / `latitude_interfaces`).
2. Build a trivial `read_window(i, j, _) = data[i, j, ...]` closure (data is already in memory; no time axis for a single Metadatum).
3. Call `set_region_data!(field_as_pseudo_fts, read_window, λc, φc, metadatum)`.

The complication: `field` is a `Field`, not a `FieldTimeSeries`. The slicing logic in `set_region_data!` takes `interior(fts, :, :, 1, :)` — the trailing `:` is the time axis. For a snapshot, there is no time axis. Two paths:

- **Path A:** Generalize `set_region_data!` to accept either a Field or an FTS, and parameterise the trailing-dim handling.
- **Path B:** Wrap the snapshot field in a fake-FTS-of-length-1 just for filling, then unwrap.

Path A is cleaner. Generalize the helpers to take `interior(target, :, :, 1)` for a Field and `interior(target, :, :, 1, :)` for an FTS via a `_target_view(target)` dispatcher.

**Step S1.2: Implement Path A**

In `metadata_field_time_series.jl`:

```julia
_target_view(fts::FieldTimeSeries) = interior(fts, :, :, 1, :)
_target_view(field::Field)         = interior(field, :, :, 1)

function set_region_data!(target, read_window, λc, φc, metadata, t_idx, region)
    # ... slice + copyto into _target_view(target)
end
```

Replace the `copyto!(interior(fts, :, :, 1, :), data)` calls with `copyto!(_target_view(target), data)` (and `reshape(blended, 1, 1)` for the snapshot column case instead of `reshape(blended, 1, 1, :)`).

**Step S1.3: In `set_metadata_field!`, replace the kernel launch with a call to `set_region_data!`**

```julia
function set_metadata_field!(field, data, metadatum)
    # Read file centres (or compute from metadata).
    λc, φc = file_centers(metadatum)  # add as a small dispatcher per dataset
    read_window(i, j, _) = data[i, j, :]  # data is already (Nx, Ny, Nz?) in memory
    set_region_data!(field, read_window, λc, φc, metadatum)
end
```

(The mangling logic — `ShiftSouth`, `AverageNorthSouth`, conversion — needs to stay; either fold it into the closure, or apply it after the slice. Cleanest: fold into the closure.)

**Step S1.4: Add `file_centers(::Metadatum)` per dataset that needs it**

For datasets where the file centres are stored on disk (most NetCDF datasets), add a small helper. For datasets where they're computable from `longitude_interfaces`/`latitude_interfaces`, fall back to that.

**Step S1.5: Run `test_dataset_region.jl`, confirm the ECCO bbox/column tests now pass.**

**Step S1.6: Run the full test suite, confirm no regressions.**

**Step S1.7: No commit.**

### Task S2: Verify ECCO column extraction goes through the generic path

**Files:**
- Read-only inspection of `column_field_from_file` at `metadata_field.jl:252-305`.

**Step S2.1:** `column_field_from_file` builds an intermediate full-extent grid then `extract_column!`s. Since the snapshot bbox path now works, the intermediate-grid step is unnecessary — the generic `Field(::Metadatum with Column region)` path can build a 1×1 grid directly via Track G's generalised Column construction and fill it via `set_region_data!`'s Column dispatch.

**Step S2.2:** If `column_field_from_file` still exists at this point, decide: delete it (and route Column metadata through the regular `Field(::Metadatum)` path) or leave it as a fallback. The cleaner choice is to delete it; the bilinear interpolation downstream is the same logic as `column_blend(_, _, _, ::Linear)`.

**Step S2.3:** Run all column-related tests (both the JRA55 FTS and the snapshot ECCO column).

**Step S2.4: No commit.**

---

## Track V — Cross-dataset verification

### Task V1: Run the full suite

**Step V1.1:**

```
julia --project=test -e 'using Pkg; Pkg.test("NumericalEarth"; test_args=["test_metadata.jl", "test_jra55.jl", "test_jra55_region.jl", "test_dataset_region.jl", "test_ecco4_en4.jl", "test_ecco2_monthly.jl", "test_woa.jl"])'
```

Expected: all green.

### Task V2: Smoke test — bbox JRA55 atmosphere into an OceanSeaIceModel

**Step V2.1:** REPL spot-check: `OceanSeaIceModel(...)` driven by a `JRA55PrescribedAtmosphere(region=BoundingBox(...))` runs one timestep. (Use whatever existing model setup exercises the atmosphere fluxes.)

**Step V2.2: No commit.** Report ready.

---

## Risks & open questions

- **JRA55 longitude convention.** Still on `[0, 360)`; user-supplied bbox longitudes that wrap (`(-10, 10)`) will silently produce a wrong slice. Add a normalisation in `BoundingBox` or document. Out of scope for this refactor.

- **`compute_bounding_indices` Float32 tolerance.** The 1-ULP slack (`eps(Float32) * max(...)`) was tuned for JRA55's 0.5625° spacing. For a finer dataset (ECCO at 0.5° or finer) the tolerance is still adequate (it's 4×10⁻⁶ on a degree, much smaller than any cell). For something coarser like an LMR proxy at 5°, also fine. No change expected, but worth a unit test to lock it in.

- **`column_field_from_file` deletion in Task S2.** Anything in CI or examples that calls it directly will need to migrate. Quick `git grep column_field_from_file` to find downstream callers before deleting.

- **MultiYear column performance.** Each year-file open + close, even when slicing only a 2×2 window, may dominate the per-call cost. If profiling shows this, keep file handles open across the loop. Out of scope.

- **The `latitude_interfaces(::JRA55Metadata)` change opens the file** every grid construction — twice, once for lon and once for lat. Fix: read both at once via `jra55_native_interfaces` and cache, or memoize on `metadata_path`. Defer to a follow-up unless profiling flags it.

- **Track S regression risk.** The snapshot path is much more heavily tested (every dataset's tests exercise it). Run the full suite at every step of Track S, not just the new bbox/column tests, before declaring done.
