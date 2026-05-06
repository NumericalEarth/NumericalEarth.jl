# Coupled energy + freshwater conservation audit

**Date.** 2026-04-22
**Author.** Simone Silvestri (audit led with Claude assistance)
**Test driver.** `test/test_conservation.jl`
**Scope.** Coupled `OceanSeaIceModel` with `SeaIceModel` thermodynamics (no
dynamics, no advection) and a prescribed atmosphere, exercised across a
freeze-then-melt cycle. Purpose: establish under what conditions the
coupled energy budget closes to machine precision, and identify every
source of residual otherwise.

---

## Summary of what was found

Three real code fixes were made in `src/` while building the test. Two
further issues were isolated and characterised; one is an accounting
timing artefact handled on the diagnostic side, the other is a real
physics bug in `ClimaSeaIce._layered_thermodynamic_time_step!` staged as
a monkey-patch in the test pending upstream PR.

With all fixes + accounting corrections in place the budget closes to
machine precision per phase (~5 × 10⁻¹³ freeze, ~1 × 10⁻¹¹ melt) over a
60-day freeze-then-melt cycle. Without any fix we started at 8.8 × 10⁻³
(0.88% closure).

---

## Section A — Fixes applied in `src/`

### A.1 Newton linearisation of the surface skin-temperature solve

**File.** `src/EarthSystemModels/InterfaceComputations/interface_states.jl`
**Symptom before fix.** `SkinTemperature.conductive_flux_balance_temperature`
used a Picard iteration on the surface temperature. When the outgoing
longwave `σεTₛ⁴` dominates, the Picard stiffness `4σεTₛ³·R` can exceed 1
and the iteration diverges. Symptom in the test: spurious drift in `Tu`
and large per-step residuals during cold-atmosphere phases.

**Fix.** Replace the Picard step with a Newton-linearised update that
treats the upwelling longwave as

    ℐꜛˡʷ(Tₛ) ≈ ℐꜛˡʷ(Tₛ⁻) + β (Tₛ − Tₛ⁻),  β = 4σεTₛ⁻³.

Folded into the flux-balance equation this gives the semi-implicit update

    Tₛ = [Tᵦ + β R Tₛ⁻ − Ωc R Tᵃᵗ − Qa R] / [1 + β R − Ωc R]

which is stable for arbitrary `R`, `β` and small-`ΔT`. The wrappers
`flux_balance_temperature(::SkinTemperature{<:ConductiveFlux}, ...)` and
`flux_balance_temperature(::SkinTemperature{<:IceSnowConductiveFlux}, ...)`
were updated to pass `ℙₛ` (interface properties) through so the new
formulation has access to `σ` and `ε`.

### A.2 Sea-ice top heat flux convention (`× ℵ`, not `× (ℵ > 0)`)

**File.** `src/SeaIces/assemble_net_sea_ice_fluxes.jl:100`
**Context.** Audit `docs/audit/sea_ice_flux_conventions.md`.
**Symptom before fix.** `ΣQt = (ℐₜˢʷ + ℐₐˡʷ + ℐꜛˡʷ + 𝒬ᵀ + 𝒬ᵛ) * (ℵ > 0)`
treated the atmospheric-ice flux as per-ice-area (masking to zero only
over open water) while ClimaSeaIce's slab mass balance expects a
per-cell-area flux. For partial ice cover the ice was over-driven by a
factor `1/ℵ`, producing a systematic over-melt in summer.

**Fix.** Replace the boolean mask with an explicit multiplication by ℵ:

    ΣQt = (ℐₜˢʷ + ℐₐˡʷ + ℐꜛˡʷ + 𝒬ᵀ + 𝒬ᵛ) * ℵ

The comment references the audit document where the convention is
derived from dimensional analysis of the slab volume balance
`∂t_V = (Q_top − Q_bot) / ℰ` with `V = h·ℵ` per cell area.

### A.3 Ocean-side freshwater routing `(1 − ℵ)`

**File.** `src/Oceans/assemble_net_ocean_fluxes.jl`
**Symptom before fix.** The ocean-side freshwater flux
`ΣFao = − Jᶜ · ρ⁻¹` was accepting all atmospheric condensate at every
grid cell, ignoring sea-ice cover. Because snow is ALSO routed to the
sea-ice model as snowfall (via `top_fluxes.snowfall = Jˢⁿ`), snow on
ice-covered cells was being double-counted: it accumulated on the ice
AND was added to the ocean as liquid freshwater, silently freshening
the ocean during freezing. Observed in the test as a drop in ocean
salinity during the freeze phase when physics would predict a rise
(brine rejection from ice formation).

**Fix.** Split rain from snow and weight each appropriately:

```julia
Jʳⁿ  = Jᶜ - Jˢⁿ                                    # rain mass flux
ΣFao = - (Jʳⁿ + (1 - ℵᵢ) * Jˢⁿ) * ρᵒᶜ⁻¹ + (1 - ℵᵢ) * Jᵛ * ρᵒᶜ⁻¹
```

Rain reaches the ocean through the full cell. Snow only through the
open-water fraction `(1-ℵ)`. Evaporation `Jᵛ` also only through the
open-water fraction (over ice, the latent-heat flux is already in
`𝒬ᵛ` on the ice-side path).

As a corollary, the downstream `(1-ℵ)` factor on `Jˢao` was removed
(now redundant — `ΣFao` already carries the weighting):

```julia
Jˢ[i, j, 1] = Jˢao + Jˢio   # was: (1 - ℵᵢ) * Jˢao + Jˢio
```

---

## Section B — Diagnostic corrections on the test side

### B.1 Frazil lag (one-step bookkeeping artefact; NOT a compounding bug)

**Where.** `compute_sea_ice_ocean_fluxes!` in
`src/EarthSystemModels/InterfaceComputations/sea_ice_ocean_fluxes.jl`
is called inside `update_state!`. It mutates the ocean temperature
`Tᵒᶜ → Tₘ` in-place wherever the ocean has supercooled, and writes
`𝒬ᶠʳᶻ = −δE·Δz/Δt` into `interfaces.sea_ice_ocean_interface.fluxes.frazil_heat`.

The assembler then adds `𝒬ᶠʳᶻ` into `net_fluxes.sea_ice.bottom.heat`,
which the ClimaSeaIce slab reads on its **next** call as a bottom flux
driving ice growth via `wb = (Qii − Qbi)/ℰb`.

**Observation.** At a single end-of-step snapshot the ocean has been
warmed by the current step's frazil mutation, but the ice has not yet
grown from that frazil — it grew in this step from the PREVIOUS update's
`𝒬ᶠʳᶻ`. So `(H_o + E_is)` carries a one-step pending quantity of
`𝒬ᶠʳᶻ(n) · Δt · A`. Over a run with continuous frazil, this telescopes
to zero as the slab always catches up on the following step. So it is a
one-step lag, not a compounding bug.

**Where it becomes a leak.** At a phase boundary, `run_phase!` calls
`update_state!` after changing the atmosphere. The refresh detects the
ocean at `Tₘ` (post previous mutation) and zeroes `𝒬ᶠʳᶻ` — stranding
the latent energy that was already added to the ocean. In our cycle this
produced a ~4 × 10¹² J leak at the freeze→melt boundary.

**Test-side resolution.**

1. In `run_phase!`, before the atmosphere-refresh `update_state!`, save
   `𝒬ᶠʳᶻ(idx_fend)`. After `update_state!`, restore it AND add it back
   to the assembler's combined `bottom_heat_flux = 𝒬ᶠʳᶻ + 𝒬ⁱⁿᵗ`. The
   next slab call then consumes the pending frazil correctly.
2. In the budget analysis, the "corrected" `E_is` includes
   `𝒬ᶠʳᶻ(n) · Δt · A` — anticipating the pending ice growth.

Because the underlying coupled model is structurally correct (the slab
always catches up over an infinite run), no upstream change is required
for frazil. The phase-transition preservation is a test-harness detail
that would be absent from long production runs where atmospheres don't
switch discontinuously.

### B.2 Pending-frazil correction to `E_is`

Mirroring B.1 on the analysis side: define

    E_is_corr(n) = E_is(n) + 𝒬ᶠʳᶻ(n) · Δt_{n,n+1} · A

so that the phase-by-phase residual attributes the in-flight frazil
energy to the phase that will consume it. Budget totals are unaffected.

---

## Section C — Real physics bug in ClimaSeaIce

### C.1 Snow-melt layered-kernel ℵ inconsistency (fix staged as monkey-patch)

**File (upstream).** `ClimaSeaIce.jl/src/SeaIceThermodynamics/thermodynamic_time_step.jl`
**Kernel.** `_layered_thermodynamic_time_step!`

The kernel solves the snow-surface energy balance via

```julia
δQ          = Qui - Qis                        # (*)
melt_energy = max(0, -δQ)
Qs          = min(melt_energy, ρs·ℒs·hsⁿ/Δt)
Qui_eff     = Qui + Qs                         # (**)
Gs⁻         = Qs / (ρs·ℒs)                     # (***)
```

where
- `Qui = top_external_heat_flux[i,j,1]` is per-CELL (after the × ℵ
  assembler fix), i.e. `Qui = Qui_per_ice · ℵⁿ`.
- `Qis = getflux(Qi_column, Tus)` evaluates the column conductive flux
  `(Tb - Tus) / R` with `R = hs/ks + hi/ki`. This is per-ICE — the
  conductance applies to the ice-covered fraction only.

Line `(*)` compares a per-cell flux with a per-ice flux. Physically the
snow surface is only present on the ice-covered fraction, so the correct
balance is per-ice:

    δQ_per_ice = Qui_per_ice − Qis = Qui/ℵⁿ − Qis.

Because `Qui/ℵⁿ > Qui` when ℵ < 1, the current code under-estimates the
snow-melt driving term and therefore the snow-melt rate. The snow
lingers longer than physics dictates whenever ice cover is partial.

**Energy-budget leak.** The same inconsistency shows up in the budget
as a per-step residual of `Qs · (1 − ℵⁿ⁺¹) · Δt · A`. In a 60-day cycle
with the configuration of this test, the cumulative leak over the
~3-day snow-depletion window is ~3 × 10¹³ J, i.e. ~3 × 10⁻⁴ relative.

**Why it's not caught by ClimaSeaIce's own energy-conservation test.**
That test runs with `ℵ = 1` throughout, so the inconsistency is exactly
masked.

**A secondary, subtler inconsistency.** The snow mass balance applies
`Δ(hs·ℵ) = ℵⁿ⁺¹ · Δt · (Gs⁺ − Gs⁻)` because of the `hsⁿ ← hsⁿ·ℵⁿ/ℵⁿ⁺¹`
rescale (which preserves `hs·ℵ` under area change) followed by the
unscaled `Δt·(Gs⁺-Gs⁻)` increment. Equivalently, the snow absorbs
`Qs · ℵⁿ⁺¹ · Δt · A` of latent energy, not `Qs · ℵⁿ · Δt · A`. If the
primary fix uses `Qs · ℵⁿ` in `Qui_eff`, the two still disagree by
`Qs · (ℵⁿ⁺¹ − ℵⁿ) · Δt · A` per step — an O(Δℵ) truncation error.

### C.2 Fix (per-ice balance + one Picard iteration)

Pseudocode for the patched kernel section:

```julia
# Per-ice atmospheric flux (primary fix).
Qui_per_ice = ifelse(ℵⁿ > 0, Qui / ℵⁿ, zero(Qui))

δQ          = Qui_per_ice - Qis               # per-ice
melt_energy = max(0, -δQ)                     # per-ice
Qs          = min(melt_energy, ρs·ℒs·hsⁿ/Δt)  # per-ice
Gs⁻         = Qs / (ρs·ℒs)                    # per-ice, drives Δhs

# Self-consistent ℵⁿ⁺¹ via one Picard iteration (secondary fix).
Qui_eff_0 = Qui + Qs · ℵⁿ                     # first guess
∂t_V_0    = ice_melt_freeze_tendency(..., Qui_eff_0, ...)
_, ℵ_tent = ice_volume_update(..., ∂t_V_0, hiⁿ, ℵⁿ, hᶜ, Δt)

Qui_eff = Qui + Qs · ℵ_tent                   # converged
∂t_V    = ice_melt_freeze_tendency(..., Qui_eff, ...)
hiⁿ⁺¹, ℵⁿ⁺¹ = ice_volume_update(..., ∂t_V, hiⁿ, ℵⁿ, hᶜ, Δt)
```

With `Qs · Δt / (2hρℒ) ≪ 1` at ocean scales, the fixed-point is a
contraction with ratio ~10⁻⁷ and a single iteration converges the
per-step closure error from O(Δℵ) to O(Δℵ²).

The patched kernel is installed in `test/test_conservation.jl` via
`@eval ClimaSeaIce.SeaIceThermodynamics begin @kernel function
_layered_thermodynamic_time_step!(...) ... end end`, gated on
`PATCH_SNOW_MELT`.

### C.3 Status

This audit proposes the fix be pushed upstream to ClimaSeaIce.jl once
verified. The accompanying ClimaSeaIce test update should exercise a
snow+ice column with ℵ < 1 to catch regressions.

---

## Section D — Local override: constant ℒ in the slab

ClimaSeaIce's slab mass balance uses a T-dependent latent heat

    ℒ(T) = ℒ₀ + (ρℓ cℓ / ρᵢ − cᵢ) (T − T₀)

with `ℰu = ρᵢ · ℒ(Tu)` at the top interface and `ℰb = ρᵢ · ℒ(Tb)` at the
bottom. A single state-based `E_is = −ℵ · ρᵢ · ℒ · h · A` cannot close
both freeze and melt phases simultaneously because freeze accumulates
mass at `T_b` while top melt happens at 0 °C, with a 4.7 kJ/kg gap.

This is a diagnostic gap, not a physics bug: the slab's own internal
accounting uses the correct per-interface ℒ values. But a consistent
STATE-BASED `E_is` requires a constant ℒ. The test therefore locally
overrides `latent_heat(pt, T)` to return `pt.reference_latent_heat`
under the `PATCH_LATENT_HEAT` toggle. With the override active the
state-based `E_is` matches the slab's mass balance for any interface
temperature, and the residual measures purely the coupler / slab /
frazil bookkeeping.

For production runs where only the volume `h·ℵ` matters, the T-dependent
ℒ is physically correct and should be kept.

---

## Section E — Test methodology

### E.1 Configuration

Minimal-physics 1×1 ocean column at 70 °N, 100 m deep × 10 levels, no
momentum/tracer advection, no closure, no coriolis. Fresh ice
(`ice_salinity = 0`) atop warm ocean (`T = -1.5 °C, S = 34`). Optional
initial snow layer of 0.10 m (`WITH_SNOW` toggle).

Two-phase forcing, 30 days each at `Δt = 10 min`:

- **freeze**: `T_air = −20 °C`, `SW = 50, LW = 180 W/m²`, snowfall
  `1.0 × 10⁻⁵ kg/m²/s` (~0.9 mm/day SWE).
- **melt**: `T_air = +5 °C`, `SW = 250, LW = 320 W/m²`, rain
  `5.0 × 10⁻⁶ kg/m²/s`.

### E.2 Integration rule

Rectangle-at-START integration: during step `n` the coupled model
applies the flux computed at the end of step `n-1` (the coupler freezes
fluxes between flux solves). In `run_phase!` we explicitly call
`update_state!` after `set_atmosphere!` and overwrite the last history
entry with the new flux, so the first-step flux driving each phase is
recorded correctly.

### E.3 Residuals

- Energy: `ΔE_tot − ∫ Q_atm · dt` where `E_tot = E_is + H_o`,
  `E_is = −ℵ · (ρᵢ·ℒ·h + ρₛ·ℒ·hs) · A`, `H_o = ρᵒ cᵒ Σ T·V_k`.
- Freshwater: `ΔM_tot − ∫ FW·dt` with virtual-salt conversion for the
  ocean part: `M_fw = −ρᵒ V (S − S_ref)/S_ref`.

### E.4 Residual progression

Measured over a 60-day freeze-then-melt cycle with `WITH_SNOW=true`,
`PATCH_LATENT_HEAT=true`:

| Stage | Relative residual |
|---|---|
| Starting point (no fixes) | 8.8 × 10⁻³ |
| + A.2 (× ℵ assembler fix) | 8.5 × 10⁻³ |
| + D (constant-ℒ diagnostic) | 3.8 × 10⁻⁴ |
| + A.3 (1-ℵ snow routing on ocean side) | 3.8 × 10⁻⁴ |
| + rectangle-at-start integration | 2.6 × 10⁻⁵ |
| + B.1+B.2 (frazil preservation + correction) | 3.5 × 10⁻⁹ |
| + C.1 (snow ×ℵ kernel patch, one pass) | 1.5 × 10⁻⁷ |
| + C.2 (one Picard iteration) | **1 × 10⁻¹¹** |

The final value is machine precision for Float64 arithmetic over
~8 000 coupled time steps with O(10¹⁸ J) peak energy content.

### E.5 No-snow verification

With `WITH_SNOW = false` the snow-kernel path is not exercised. In that
configuration the coupled energy budget closes to **4 × 10⁻¹³ relative**
(full cycle) with only the A.1–A.3 fixes + frazil preservation. This
verifies that everything except snow is exactly conserving.

---

## Files touched / related

- `src/EarthSystemModels/InterfaceComputations/interface_states.jl`
  — Newton linearisation (A.1)
- `src/SeaIces/assemble_net_sea_ice_fluxes.jl`
  — `× ℵ` on top heat flux (A.2)
- `src/Oceans/assemble_net_ocean_fluxes.jl`
  — `(1-ℵ)` on snow + `Jᵛ`, removed downstream `(1-ℵ)` on `Jˢao` (A.3)
- `test/test_conservation.jl`
  — driver, diagnostic corrections, snow-kernel monkey-patch (C.2)
- `docs/audit/sea_ice_flux_conventions.md`
  — background for A.2
- `docs/plans/2026-04-21-sea-ice-freshwater-closure.md`
  — background for A.3
- `ClimaSeaIce.jl/src/SeaIceThermodynamics/thermodynamic_time_step.jl`
  — upstream target for C.2 (to be PR'd)
