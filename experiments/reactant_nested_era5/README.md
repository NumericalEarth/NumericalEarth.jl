# ERA5 → Breeze `NestedModel` under Reactant: compile smoke test

`debug_compile.jl` builds the smallest possible ERA5-forced nest on `ReactantState` and compiles
one time step. It is a straight-line script: the first error is the first blocker. This README
records which pieces of the full workflow (`examples/breeze_downscaling_era5.jl`) are expected to
fail under Reactant, with the reason and where the fix goes.

```sh
# from the NumericalEarth.jl root
julia --project=experiments/reactant_nested_era5 -e 'using Pkg; Pkg.instantiate()'
julia --project=experiments/reactant_nested_era5 experiments/reactant_nested_era5/debug_compile.jl
```

ERA5 comes through the Copernicus Climate Data Store (credentials in `~/.cdsapirc` or
`CDSAPI_URL`/`CDSAPI_KEY`); the first run downloads into `./era5`. `BACKEND=gpu` selects CUDA.
Resolved versions: Oceananigans 0.113.5, Breeze 0.11.3, Reactant 0.2.290, CUDA 6.2.2 (the same
pairing Breeze's own Reactant test manifest uses).

## State of the data loading

The good news: with the default `parent_time_indices_in_memory = nothing`, `nested_atmosphere_model`
already loads **every** ERA5 snapshot into memory at construction. Each variable is a
`DatasetBackend` `FieldTimeSeries` on the native pressure-level grid with a full window, and the
state exchanger's derived prognostics (`PrognosticStateBackend`) also hold the full window, so
nothing reads disk during a step. Downloads (`Downloads.download(MetadataSet)`) happen inside the
constructor, outside any compiled region. So "load all the data, then compile around it" is
structurally already how the nest works.

What is **not** in place is making that loaded state usable on a `ReactantState` architecture:

1. The parent's native grid (a `LatitudeLongitudeGrid` with `PressureLevelVerticalDiscretization`)
   cannot be built on or moved to `ReactantState` (A).
2. The parent's clock is a plain host `Clock{Float32}` (B).
3. Even with a full window, the per-step `update_field_time_series!` on `DatasetBackend` series
   does a host branch on traced time indices (C).

The precedent for the fix is `examples/era5_forced_slab_land.jl` (functions `inmemory_forcing`
and `transfer_forcing`): load ERA5 on the CPU, interpolate/copy every slice into plain
`InMemory()` `FieldTimeSeries` on the target grid, then copy buffers into a `ReactantState` twin
(`parent(dst) .= Array(parent(src))`). For the nest the equivalent is:

- build the parent on CPU: `ERA5PrescribedAtmosphere(bbox, dates; architecture = CPU())`;
- build a `ReactantState` `PrescribedAtmosphere(reactant_parent_grid, times; velocities = …,
  temperature = …, …)` whose fields are `InMemory()` series filled by buffer copy, with a Reactant
  clock (`Clock(reactant_parent_grid)` from the Oceananigans Reactant extension);
- use the parent-first method `nested_atmosphere_model(parent, child_grid; …)` so the exchanger
  and child are built against the transferred parent.

This needs (A) solved for `PressureLevelVerticalDiscretization` (its `TimeSeriesInterpolation`
geopotential and surface geopotential must move with it, and the interpolation must be bound to
the Reactant clock), and either `InMemory()` series or a Reactant no-op for (C).

## Pieces of the full workflow expected to fail under Reactant

Ordered by where they hit. "High" means the failure follows from reading the code; "Medium"
means it depends on what Reactant's kernel raising accepts and needs a run to tell.

**A. Grid construction and transfer on `ReactantState` (High).** The only
`LatitudeLongitudeGrid(::ReactantState, …)` constructor
(`Oceananigans.jl/ext/OceananigansReactantExt/Architectures.jl:116`) builds the grid on the CPU
and moves each field through `_to_reactant`, which has methods for numbers, arrays,
`OffsetArray` and `StaticVerticalDiscretization` only. Both verticals this workflow uses,
NumericalEarth's `PressureLevelVerticalDiscretization`
(`src/Grids/pressure_level_vertical_discretization.jl`) for the ERA5 parent and Breeze's
`TerrainFollowingVerticalDiscretization` for the child, hit a `MethodError`. Fix: extend
`_to_reactant` (or route it through `on_architecture`, which NumericalEarth already defines for
the pressure-level vertical) in `NumericalEarthReactantExt` and `BreezeReactantExt`
respectively. A `RectilinearGrid` would sidestep the lat-lon constructor, but the nest is lat-lon.

**B. Host clock on the parent and the parent tick (High).** `ERA5PrescribedAtmosphere` creates
`Clock{FT}(time = 0)` on the host (`src/DataWrangling/ERA5/ERA5_prescribed_atmosphere.jl`).
`NestedModel.time_step!` (`src/NestedModels/nested_model.jl`) computes
`Δt_parent = child.clock.time - parent.clock.time`, traced minus host, then branches
`Δt_parent > 0 && time_step!(parent, Δt_parent)`: a `TracedRNumber{Bool}` in `&&` is an error
inside `@compile`. Removing the guard still fails, because `tick!` assigns a traced value into the
host clock's `Float32` field. Fix: give the parent a Reactant clock when the architecture is
`ReactantState` (or share the child's clock object outright, since the parent only uses it to
drive the geopotential `TimeSeriesInterpolation` and FTS windows), and tick it unconditionally.

**C. `update_field_time_series!` on `DatasetBackend` series inside the step (High).**
`time_step!(::PrescribedAtmosphere)` calls `update_state!`, which calls
`update_field_time_series!(fts, Time(t))` on every series. For a `DatasetBackend` (a
`PartlyInMemory` backend even when the window is full) that goes through
`cpu_interpolating_time_indices(::ReactantState, …)` → `in_time_range` → `n₁ ∈ idxs` on traced
indices → `if !in_range`, a traced branch. Fix: after loading, convert the parent series to
`InMemory()` (`TotallyInMemory`, whose update is the no-op fallback), or define a Reactant method
that no-ops when the window covers the whole series. The exchanger's `PrognosticStateFTS` already
no-ops its update, so the child side is fine.

**D. `exchange_state!` host logic (High for streaming, OK for full window).**
`ext/NumericalEarthBreezeExt/breeze_state_exchanger.jl` computes `interpolating_time_indices`
on the traced time, then `moved = backend.start != start` and `if moved … compute …`. With a
full window `start` is a host `1`, so `moved` is `false` and the branch never traces; with a
streaming window (`parent_time_indices_in_memory ≥ 3`) `start` is traced and this fails, and the
conditional recompute cannot live inside a program anyway. Under Reactant the parent must be
fully resident.

**E. In-kernel interpolation of the parent (Medium).** The lateral BCs (`Interpolated` in
`src/NestedModels/interpolated_fts_boundary.jl`) and the Davies `Relaxation`
(`FieldTimeSeriesTarget`) interpolate the exchanger series in space and time inside kernels.
Time: `clock_time` converts the traced clock time to the grid float type inside the kernel; the
CUDA tracing extension defines that `convert`, so this probably works but is untested here.
Space: on a `PressureLevelGrid`, `column_fractional_z_index` runs `index_binary_search` and
`first_above_surface_level` runs a data-dependent `while` loop per column. Kernel raising may
reject or not vectorize those loops. If the compile fails inside a halo-fill or forcing kernel, this is
where to look.

**F. Eager construction on `ReactantState` (Medium).** `initialize_nested_child!` uses
`interpolate!(field, fts[Time(t₀)])` from the parent grid to the child grid; Oceananigans'
Reactant extension notes that `interpolate!`'s kernel does not trace (Reactant.jl#2364) and
falls back to CPU only for field-to-field `set!`. `mean_sea_level_pressure` does
`set!(p₀, Metadatum)` onto a Reactant field, a host regrid onto an XLA buffer. The adiabatic
balancer (`set!(nest; balancer = true)`) runs tens of eager steps, each a separate XLA program;
the script passes `balancer = false`. Terrain (`regrid_topography`, `smooth_topography!`,
`blend_parent_terrain!`, `materialize_terrain!`) is eager host+kernel work on a grid that cannot
exist yet (A); the script runs with no terrain.

**G. The `Simulation` layer (High, by design).** A `NestedModel` on `ReactantState` dispatches to
the Reactant `Simulation`: fixed `Δt`, `stop_iteration` or a divisible `stop_time`, no
`output_writers`, no `TimeStepWizard`, `TimeInterval` schedules only when `Δt` divides them,
callbacks limited to device work. The example's `conjure_time_step_wizard!`, three `JLD2Writer`s,
`TimeInterval(20minutes)` output, and the `@info`/`maximum` progress callback all have to move
out of the compiled loop (run N steps per program, do IO between programs).

**H. Layers not covered by any Reactant extension (High).** `NumericalEarthReactantExt` only
defines `reconcile_state!` for `EarthSystemModel` and `same_time_type`. `AtmosphereLandModel`,
`SlabLand`, RRTMGP `RadiativeTransferModel` with `CopernicusAlbedo`, and the Monin–Obukhov
interface fluxes have no Reactant tests here (the only existing coupled Reactant test is an
ocean + idealized atmosphere `OceanOnlyModel`). Get the bare nest compiling first.

**I. Lower-risk items to keep in mind.** `Cyclical` time indexing uses `mod` and
`unsafe_trunc` on traced values (supported). Parent series times are `Float32`, the Reactant
clock is `Float64`; Oceananigans promotes. The 1-moment `CloudMicrophysics` scheme is not loaded
in this environment (warm-phase `SaturationAdjustment` is used instead); Breeze has Reactant
compile tests for the 1-moment scheme, so adding it later is low risk.
