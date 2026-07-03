# ERA5 → Breeze nested downscaling — long-run plan

Convection-permitting 12 h hindcasts that downscale ERA5 to a Breeze compressible child across a
resolution sweep (12 → 0.5 km), run simultaneously across both H100s and the CPU partition, each
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
| 12 km | 24 × 22 × 50 | 26 k | `cpu` partition |
| 6 km | 48 × 44 × 50 | 106 k | `cpu` partition |
| 3 km | 96 × 88 × 50 | 422 k | H100 (queues) |
| 1 km | 288 × 264 × 50 | 3.8 M | H100 |
| 0.5 km | 576 × 528 × 50 | 15 M | H100 |

**Prior stability check (30-step cold start, WENO-5 bounded, no closure):** stable at every resolution,
`max|w| @ 30` = 6.13 / 6.34 / 6.23 / 6.20 / 5.70 m/s for 12 / 9 / 6 / 3 / 1 km — essentially
dx-independent, so the wall residual is a bounded boundary artifact, not a CFL/gradient instability.

## Compute allocation

- **gpu-prod** (2 × H100, 1 GPU/node): 0.5 km, 1 km, 3 km — the third queues until an H100 frees.
- **cpu partition** (2-core nodes): 12 km, 6 km.
- Head node has only 4 cores, so it is used for orchestration/pre-download, not the long runs.

## Outputs (per run)

- 10-minute 2-D slices via `JLD2Writer` — an x–z cross-section (`w, speed, T`) and a surface slice
  (`w, speed`) — never 3-D.
- An **mp4 animation** (`long_<tag>_w.mp4`): `w` x–z cross-section + surface `w` over the 12 h, rendered
  with `CairoMakie.record`.
- A final multi-field panel (`long_<tag>_panel.png`).
- Animations + panels are collected into the run gallery:
  https://claude.ai/code/artifact/2ac43382-134c-49e3-b4a3-a3c76246162b

## Files (scratch dev environment)

The runs live in `scratch_runenv/` — a dev env pinning Breeze + Oceananigans to their development
branches (the registry lacks Oceananigans 0.110.x):

- `sweep_long.jl` — the harness (ENV `RES_KM`, `ARCH ∈ {GPU, CPU}`).
- `sweep_long_gpu.batch`, `sweep_long_cpu.batch` — Slurm batches (`sbatch --export=ALL,RES_KM=…`).
- `launch_long.sh` — waits for the ERA5 pre-download, then submits all cases.
- `predownload_long.jl` — pulls ERA5 00–12 UTC for the sweep region once, shared by all runs.

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
