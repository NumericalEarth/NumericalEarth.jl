# ERA5 → Breeze nested downscaling — long-run plan

Convection-permitting 12 h hindcasts that downscale ERA5 to a Breeze compressible child across a
resolution sweep (12 → 0.5 km), run simultaneously across the A100 fleet via Slurm, each
producing an animation of the result. Lamont, OK; 2011-05-20 00–12 UTC.

## Model configuration (the validated stable configuration)

All the pieces below were established while getting `max|w|` bounded (the initial cold-start blow-up
went from a 68 m/s runaway to a bounded ~5–6 m/s wall residual):

- **Consistent initial condition (option A):** `initialize_nested_child!` derives the child interior
  from the *same* exchanger prognostics that drive the boundary (`ρᵈ, ρθ, ρqᵛ, ρu, ρv` interpolated to
  the child), so interior ≡ boundary at `t = 0` — no standing IC↔BC pressure/density jump.
- **Lateral boundaries:** per-side momentum BCs — `NormalFlowBoundaryCondition` on each velocity's
  wall-normal side (`ρu` E/W, `ρv` N/S), `ValueBoundaryCondition` (parent value in the halo) on the
  tangential side; `ρᵈ`/`ρe`/moisture are `ValueBoundaryCondition`. Interior **Davies relaxation** of
  `ρᵈ, ρθ, ρu, ρv, <moisture>` toward the parent over a cosine ramp across the outermost 5 cells
  (matches the WRF/MPAS mass-nudging precedent; `ρw` is *not* relaxed, also per precedent).
- **Lid sponge:** `damping_depth = default_lid_depth(grid) = Lz/4` (~5 km for a ~19.5 km column, base
  ~15 km) — thin enough to leave the 12–16 km convective layer undamped, deep enough to absorb genuine
  top reflection now that the IC fix removes the wall-mode source. Lifts both the split-explicit
  `UpperSponge` and the `ρw` Rayleigh lid.
- **Advection (implicit LES, no explicit closure):** `momentum_advection = WENO(order=5)`; per-tracer
  `scalar_advection = (ρθ = WENO(5), ρqᵉ = WENO(5, bounds=(0,1)), ρqʳ = …, ρqˢ = …)` — moisture/precip
  are positivity-bounded, `ρθ` must **not** be bounded to `[0,1]`; `closure = nothing`.
- **Time stepping:** adaptive `conjure_time_step_wizard!(cfl = 0.7, max_Δt = 20 s)` on the (slower)
  advective CFL; acoustics are substepped by the split-explicit dynamics.
- **Davies rate & the over-relaxation constraint:** the relaxation is an *explicit* nudge
  `dφ/dt = −r(φ − φₚₐᵣₑₙₜ)`, so its per-step error factor is `(1 − rΔt)` — stable only for `rΔt ≲ 2`,
  and it overshoots the target for `rΔt > 1`. A rate frozen at `1/(10·Δt₀)` climbs to `rΔt ≈ 3` once the
  wizard grows Δt to 30 s → blow-up. These runs use a fixed `relaxation_rate = 1/300 s⁻¹` (so
  `rΔt ≈ 0.07` even at Δt = 20 s). **TODO (proper fix):** WRF-style `r = 1/(N·Δt_current)` via
  `Clock.last_Δt` (N ≈ 3–5) so `rΔt` is pinned regardless of the wizard.

## Domain and resolution sweep

Fixed physical domain — `Lλ, Lφ = 24/9°, 22/9°` (~290 km × 270 km) centered on (36.605°N, −97.485°E) —
refined by `Δ = res_km / 108°`. Because the domain is fixed, **every resolution reuses one cached ERA5
region** (no re-download per case). The parent is `ERA5HourlyPressureLevels` on its native pressure-level
grid; terrain is `ETOPO2022` blended to the parent orography over the outer 5 cells.

| Δx | grid (Nx × Ny × Nz) | cells | run on |
|------|---------------------|-------|--------|
| 12 km | 24 × 22 × 50 | 26 k | `gpuprod` (H100) |
| 6 km | 48 × 44 × 50 | 106 k | `gpua100x4` |
| 3 km | 96 × 88 × 50 | 422 k | `gpua100x4` |
| 1 km | 288 × 264 × 50 | 3.8 M | `gpua100x4` |
| 0.5 km | 576 × 528 × 50 | 15 M | `gpua100x4` |

**Prior stability check (30-step cold start, WENO-5 bounded, no closure):** stable at every resolution,
`max|w| @ 30` = 6.13 / 6.34 / 6.23 / 6.20 / 5.70 m/s for 12 / 9 / 6 / 3 / 1 km — essentially
dx-independent, so the wall residual is a bounded boundary artifact, not a CFL/gradient instability.

## Compute allocation (Slurm — everything runs through `sbatch`)

- **`gpua100x4`** (1 node, 4 × A100-40GB, 24 CPUs, 330 GB): 0.5, 1, 3, 6 km — one A100 per case
  (`--gres=gpu:1`, 4 CPUs each), all four concurrent.
- **`gpuprod`** (2 nodes, 1 × H100-80GB, 12 CPUs each): the 12 km case on one node, CPU-side
  rendering of the animations/panels on whichever is free. If a case ever outgrows an A100-40GB,
  an H100-80GB here is also the memory fallback.
- **`gpua100`** (1 node, 1 × A100, 6 CPUs): unused — the node failed two consecutive boot probes
  (16+ min silent vs gpuprod's 1m35s spin-up, 2026-07-03); revisit if it recovers.
- The `cpu` partition nodes (1 core, 1.5 GB) are too small to even load the Julia stack — unused.
- The head node (1 core, 3 GB) is orchestration only; the Julia depot lives on the NFS-shared home,
  so environment instantiation is itself a Slurm job on `gpua100x4` (`instantiate_long.batch`).

## Outputs (per run)

- 10-minute 2-D slices via `JLD2Writer` — never 3-D — into `scratch_runenv/runs/long_<tag>/`:
  a central x–z cross-section (`w, U, T`), a 2 km-AGL map (`w, U`), and a surface map (`U, θᵥ`).
  `T` is `model.temperature` (recomputed each `update_state!`), not the Boussinesq
  `TemperatureField` constructor.

## Visualization

Rendering is decoupled from the runs: each case gets its own CPU-only CairoMakie Slurm job
(`visualize_long.batch`, `gpuprod` partition) submitted with `--dependency=afterok:<run job>`, so
every animation starts rendering the moment its run finishes — no manual babysitting, and a crashed
run never blocks the other renders.

Per run (rendered by `visualize_long.jl` from the saved slices):

- **`long_<tag>_w.mp4`** — the 12 h squall-line evolution of `w`: 2 km-AGL map (left) + central x–z
  cross-section (right), 10-minute frames at 8 fps, color range fixed across frames (symmetric,
  from the late-time state) so intensification is visible rather than renormalized away.
- **`long_<tag>_panel.png`** — final-time 2 × 3 panel: cross-sections of `w, U, T` (top) over
  maps of `w₂ₖₘ, Uₛ, θᵥₛ` (bottom).

Across runs: a **gallery artifact** collecting all five panels (and animation stills) side by side,
ordered 12 → 0.5 km, so the convergence of the squall line's structure with resolution is the
headline comparison. The mp4s stay on the cluster (`scratch_runenv/runs/`); the gallery links their
paths.

## Breeze pin (2026-07-03) and stability tripwires

The branch tip of `glw/balance-twin-vapor-moisture-fractions` (rebased onto Breeze 0.7.1 +
the #821 acoustic-substep reland) carries a **nondeterministic adiabatic-balance-twin blowup**:
identical nested-init runs produce θ^γ DomainErrors spanning 24 orders of magnitude, silent NaNs,
or an occasional clean pass (uninitialized-read suspect; full evidence table in
`scratch_runenv/BREEZE_827_EVIDENCE.md`, for upstream Breeze.jl#827). The sweep therefore runs
Breeze from `~/breeze-prereland` — the branch with only that reland `git revert`ed (both
balance-twin fixes kept) — which passes the failing cell repeatedly where the tip fails ~7 in 8.
Two further layers guard the runs regardless: the smoke job errors (blocking `afterok`
dependents) if `max|w| > 15 m/s` at 30 iterations, and every run's progress callback kills the
job if `max|w|` exceeds 150 m/s or goes non-finite.

Two same-day fixes this campaign surfaced (committed on this branch): GPU region loading died at
kernel compilation on reshaped host NetCDF data (`architecture_ready` in
`src/DataWrangling/set_region_data.jl`), and `atmosphere_model` gained an `initialize` keyword so
the nested child skips the construction-time resting state (semantic correction — not the twin
cure).

## Files (scratch dev environment)

The runs live in `scratch_runenv/` (gitignored) — a dev env pinning Breeze as described above and
NumericalEarth to the working tree:

- `sweep_common.jl` — shared domain + model construction (ENV `RES_KM`, `ARCH ∈ {GPU, CPU}`, `SMOKE`).
- `sweep_long.jl` — the run harness: simulation, wizard, writers, progress.
- `predownload_long.jl` — pulls ERA5 00–12 UTC + ETOPO2022 for the sweep region once, shared by all
  runs (builds the 12 km model on CPU, which triggers every download). Needs `~/.cdsapirc`.
- `visualize_long.jl` — renders one case's mp4 + panel from the saved slices.
- `instantiate_long.batch`, `predownload_long.batch`, `sweep_long_gpu.batch`, `visualize_long.batch`
  — the Slurm batches.
- `launch_long.sh` — `predownload` | `smoke [jobid]` | `runs [jobid]`; `runs` submits the five cases
  with their per-case partition/memory and chains each case's render via `afterok`.

## Reproduce a single case

```julia
# ERA5 00–12 UTC must be cached for the region first (predownload_long.jl)
RES_KM=1 ARCH=GPU julia --project=scratch_runenv scratch_runenv/sweep_long.jl
```

## Provenance

This configuration is the endpoint of the nested-atmosphere work on `glw/nested-atmosphere-bc`: the
ERA5/era5cli download path, the Breeze adiabatic-balancer twin fix, the option-A exchanger-derived IC,
the domain-scaled lid sponge, the per-side momentum BCs, and the `ρᵈ` lateral relaxation. AIVA (adaptive
implicit vertical advection) is validated as an efficiency lever for large Δt but is not needed for
stability here.
