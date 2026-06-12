# SlabLand tutorial: conservative variably saturated land

`SlabLand` is NumericalEarth's prognostic single-layer land-surface
component. This tutorial walks through the modern variably-saturated
configuration: a conservative slab water balance, a force-restore
internal-energy budget with a water-mass-dependent heat capacity, and an
atmosphere-facing humidity formulation that solves the surface specific
humidity from a vapor-flux balance across an unresolved dry surface
layer.

The headline observable behavior — wet patches evaporate strongly while
dry patches partition all surface energy into sensible heat — emerges
self-consistently from the dry-layer Fickian physics rather than from a
prescribed evaporation-efficiency function `β(𝒮)`.

All math symbols below follow
[`docs/src/appendix/notation.md`](../appendix/notation.md). The model is
built around three closures
([`VariablySaturatedHydrology`](@ref),
[`WaterCoupledEnergy`](@ref),
[`DryLayerHumidity`](@ref)) plus their sub-closures (deep liquid
flux, runoff, retention curve, dry-layer depth, dry-layer vapor
exchange). It targets bare ground without snow, ice, or vegetation; see
§10 ("Out of scope") below. §7 surveys the full closure menu, including
the simpler configurations (`SlabEnergy`, `BucketHydrology`,
`FractionalHumidity`) that predate this one.

## 1. What `SlabLand` stores

The composable container keeps a two-prognostic shape:

| Symbol | Code | Meaning |
|---|---|---|
| `Tˡᵃ` | `land.temperature` | bulk land temperature, K |
| `Mˡᵃ` | `land.water_storage` | slab water mass per land area, kg m⁻² |
| `𝒮`  | `land.saturation` | diagnostic surface saturation, recomputed from `Mˡᵃ` every step |

A closure-extensible `land.diagnostics` slot carries fields each closure
publishes for downstream consumers — for the variably-saturated hydrology:
`deep_liquid_flux`, `surface_liquid_flux`, `surface_runoff`,
`subsurface_runoff`, and `water_storage_tendency`. The hydrology step
publishes `water_storage_tendency`; the energy step consumes it for the
`Mˡᵃ`-dependent `Tˡᵃ` correction (§2.3 below).

`SlabLand` exposes a single atmosphere-facing convention through the
flux accumulator `land.fluxes`:

| Convention | Field | Sign |
|---|---|---|
| signed (new) | `vapor_flux`, `surface_energy_flux` | positive upward ⇒ leaves slab |
| signed (new) | `liquid_precipitation_flux` (`Pˡ`) | positive downward ⇒ enters slab |
| legacy | `net_energy_flux`, `evaporation`, `precipitation` | positive ⇒ into slab (pre-2026 convention) |

Both conventions are populated by the coupler's `update_net_fluxes!`
every step, so closures can read whichever set they declared in
`flux_variables`.

## 2. Continuous balances and the slab reduction

The slab is a depth-integrated reduction of the local Darcy/Richards
water balance and Fourier/conductive energy balance, with the
augmented-liquid-fraction trick that lets one prognostic carry both
unsaturated and saturated overflow storage.

### 2.1 Local PDEs

With `z` increasing upward, hydraulic head `h = z + Π`, and matric
pressure head `Π` (m), the upward-positive Darcy flux is

```math
J^l = -\rho^l K\,\partial_z h,
```

so the local water-mass balance is

```math
\frac{\partial}{\partial t}(\rho^l\vartheta^l) = -\partial_z J^l + S^M.
```

`ϑˡ` is the **augmented liquid fraction** — the actual pore liquid
fraction `θˡ` below saturation (`Π ≤ 0`), with `Π/hˢˢ` added once
saturated (`Π > 0`):

```math
\vartheta^l =
\begin{cases}
\theta^l(\Pi), & \Pi \le 0, \\
\nu + \Pi/h^{\mathrm{ss}}, & \Pi > 0.
\end{cases}
```

Below saturation this is the standard storage variable; above
saturation it carries the *elastic* storage of the saturated soil
skeleton (storage height `hˢˢ`, the reciprocal of the specific storage
`1/Sₛ`). This single-variable framing lets the
prognostic exceed `ν` without a hard clamp.

Internal energy (ignoring ice — out of scope here):

```math
e = \bigl[(1-\nu)\rho^{ds}c^{ds} + \rho^l\theta^l c^l\bigr](T - T_r),
\qquad J^E = -\kappa^T\,\partial_z T + e^l(T)\,J^l,
```

with `eˡ(T) = cˡ(T − Tᵣ)` the specific internal energy of liquid water
and `∂e/∂t = −∂_z Jᴱ + Sᴱ`.

### 2.2 Depth-integrated state and diagnostics

Integrating from the slab bottom `z_b = z_s − hˡᵃ` to the surface `z_s`:

```math
M^{la} = \int_{z_b}^{z_s} \rho^l\vartheta^l\,dz,
\qquad
E^{la} = \int_{z_b}^{z_s} e\,dz.
```

The saturation storage is `Mˡᵃ⁺ = ρˡ ν hˡᵃ`. The model carries `Mˡᵃ` as
the prognostic and reconstructs the rest as diagnostics:

```math
\bar\vartheta^l = \frac{M^{la}}{\rho^l h^{\mathrm{la}}},
\qquad
\bar\theta^l = \min(\bar\vartheta^l, \nu),
```

```math
\bar\Pi =
\begin{cases}
\Pi_m(\bar\theta^l), & M^{la} < M^{la+}\quad (\text{unsaturated}), \\
(M^{la} - M^{la+})\,h^{\mathrm{ss}}/(\rho^l h^{\mathrm{la}}), & M^{la} \ge M^{la+}\quad (\text{saturated overflow}),
\end{cases}
```

```math
\mathcal S = \operatorname{clip}\!\left(\frac{\bar\theta^l - \theta^r}{\nu - \theta^r},\ 0,\ 1\right),
\qquad
\beta = \operatorname{clip}(\mathcal S/\mathcal S^c,\ 0,\ 1).
```

Three things to keep separate:

* `ϑˡ` (and `Mˡᵃ`) — the **conservative** storage variable; can exceed `ν`.
* `θˡ` — the **physical** pore liquid fraction; what surface physics
  reads (albedo, soil-air vapor diffusivity).
* `Π` — the **head** variable; what Darcy fluxes and pressure-storage
  saturated overflow read. Negative unsaturated, positive once
  saturated.

`Π_m(θ̄ˡ)` is the unsaturated retention curve (Van Genuchten in this PR;
Brooks–Corey is a follow-up).

### 2.3 Slab water and energy budgets

The depth-integrated balances (positive upward):

```math
\boxed{\quad
\frac{dM^{la}}{dt} = J^l_b - J^l_s - J^v - R^M_{lat}
\quad}
```

```math
\boxed{\quad
\frac{dE^{la}}{dt} = \Lambda^{deep}(T^{deep} - T^{la})
                    + e^l_{b,\text{up}} J^l_b
                    - J^E_s
                    - c^l(T^{la}-T_r) R^M_{lat}
\quad}
```

with `Jˡ_s = −Pˡ + Rᴹ_sfc` (surface liquid flux is precipitation minus
rejected surface runoff) and upwind energy in the deep liquid flux,

```math
e^l_{b,\text{up}} =
\begin{cases}
c^l(T^{deep} - T_r), & J^l_b > 0\ (\text{capillary rise}), \\
c^l(T^{la}  - T_r), & J^l_b < 0\ (\text{drainage}).
\end{cases}
```

The `Mˡᵃ`-dependent areal heat capacity is

```math
C(M^{la}) = C_{dry} + c^l M^{la},
\qquad
E^{la} = C(M^{la})(T^{la} - T_r),
```

and `Tˡᵃ` updates conservatively:

```math
\frac{dT^{la}}{dt} = \frac{1}{C(M^{la})}\!\left[
                       \frac{dE^{la}}{dt} - c^l(T^{la} - T_r)\frac{dM^{la}}{dt}
                     \right].
```

The conservative `dTˡᵃ/dt` form guarantees that adding or removing
water *at the slab temperature* leaves `Tˡᵃ` invariant — verified
numerically to floating-point error in
[`test_water_coupled_energy.jl`](https://github.com/NumericalEarth/NumericalEarth.jl/blob/main/test/test_water_coupled_energy.jl).

## 3. Runoff: rejected input versus storage export

There are two distinct runoff categories that bookkeep differently:

* **Surface runoff** `Rᴹ_sfc ≥ 0` is a *rejected liquid input* — never
  enters storage. [`InfiltrationCapacityRunoff`](@ref) caps the
  downward `Jˡ_s` at the soil capacity; the excess is rejected to
  `land.diagnostics.surface_runoff`.
* **Subsurface runoff** `Rᴹ_lat ≥ 0` is a *storage export* — appears
  both as a sink in `dMˡᵃ/dt` and as `cˡ(Tˡᵃ − Tᵣ) Rᴹ_lat` in
  `dEˡᵃ/dt`. Stored in `land.diagnostics.subsurface_runoff`. No
  subsurface-runoff closure ships in this PR (`NoRunoff` and
  `InfiltrationCapacityRunoff` both return 0); future closures like
  `TopographicRunoff` plug into the same hook.

Both categories live in `land.diagnostics` (`surface_runoff`,
`subsurface_runoff`); they become a flux only when a river / ocean
coupler consumes them.

## 4. Solving the surface humidity

A traditional bucket-land closure prescribes `qⁱⁿ = β(𝒮) qᵛ⁺(Tⁱⁿ)`. Two
known weaknesses:

1. The evaporation efficiency `β` is empirical and tuned per soil type.
2. The closure does not respond to *atmospheric* conditions — the same
   `β` is used for calm/windy days, dry/humid air.

[`DryLayerHumidity`](@ref) replaces this with a *vapor-flux
balance* between the soil and the atmosphere. Vapor diffuses up to the
surface from saturated pore air at the base of the dry layer — the
*front*, at depth `δᵛ` below the surface — by Fickian diffusion through
the dry layer:

```math
J^e = G^e\,(q^e - q^{in}),
\qquad
G^e = \rho^{at}\,\frac{D^v_{eff}}{\max(\delta^v, \delta^v_{min})}.
```

The atmosphere carries vapor away at rate `Jᵃ = −ρᵃᵗ u★ q★`. The
balance `Jᵉ = Jᵃ` closes for `qⁱⁿ`:

```math
q^{in} = \frac{G^e q^e + G^a q^{at}}{G^e + G^a},
```

with `Gᵃ = Jᵃ/(qⁱⁿ⁻ − qᵃᵗ)` the implicit atmospheric conductance from
the previous fixed-point iterate (the same trick `SkinHumidity` uses —
no prescribed exchange coefficient). When the surface is wet enough
(`𝒮 ≥ 𝒮ᶜ`, so `δᵛ ≤ δᵛ_min`), the formulation collapses to a saturated
skin `qⁱⁿ = qᵛ⁺(Tⁱⁿ)`, reproducing the saturated-surface limit exactly.

`δᵛ` is itself diagnostic of saturation via
[`StorageBasedDryLayerDepth`](@ref):

```math
\delta^v(\mathcal S) = \delta^v_{max}\,\bigl[1 - \min(\mathcal S/\mathcal S^c, 1)\bigr]^\eta.
```

A wet surface has `δᵛ = 0` (front sits at the skin). A dry surface has
`δᵛ → δᵛ_max` (long Fickian path ⇒ small `wᵈ = Dᵛ_eff/δᵛ` ⇒ small `Jᵛ`).
The result is a self-consistent moisture-availability *response*, not a
prescribed `β(𝒮)`.

### 4.1 Source temperature `Tᵉ` and the χ interpolation

The vapor source temperature is the dry-layer temperature `Tᵉ`,
which we diagnose between the atmosphere-facing skin `Tⁱⁿ` and the
bulk-soil `Tˡᵃ` according to how deep the front sits relative to the
thermal exchange scale `ℓᵀ`:

```math
T^e = T^{in} + \chi\,(T^{la} - T^{in}),
\qquad
\chi = \operatorname{clip}(\delta^v/\ell^T,\ 0,\ 1).
```

When the front is at the surface (`δᵛ = 0`) the source temperature is
the skin; when the front has retreated to `δᵛ ≥ ℓᵀ` the source
temperature is the bulk soil. In this PR `Tⁱⁿ = Tˡᵃ` (the existing
`BulkTemperature()` is reused — land-side `SkinTemperature` integration
is deferred), so `Tᵉ` collapses to `Tˡᵃ` and the wet/dry contrast comes
entirely from `wᵈ`, not from temperature interpolation. The χ machinery
is already in place for the follow-up.

### 4.2 Picard fixed-point algorithm

The atmosphere–land flux solver iterates the interface state `(Tⁱⁿ,
qⁱⁿ)` to convergence. Each iteration `k`:

1. `δᵛ ← δᵛ(𝒮)` from [`StorageBasedDryLayerDepth`](@ref).
2. `χ ← clip(δᵛ/ℓᵀ, 0, 1)`.
3. `Tᵉ ← Tⁱⁿ + χ (Tˡᵃ − Tⁱⁿ)`.
4. `qᵉ ← qᵛ⁺(Tᵉ, pᵃᵗ)`.
5. `wᵈ ← Dᵛ_eff / max(δᵛ, δᵛ_min)`.
6. Surface-layer similarity scheme provides `Jᵃ = −ρᵃᵗ u★ q★` (and
   hence the implicit conductance `Gᵃ = Jᵃ/Δq`) from the current
   `(Tⁱⁿ, qⁱⁿ)`.
7. **Wet branch** (`δᵛ ≤ δᵛ_min`): `qⁱⁿ ← qᵛ⁺(Tⁱⁿ, pᵃᵗ)`.
   **Dry branch:**                `qⁱⁿ ← (Gᵉ qᵉ Δq + Jᵃ qᵃᵗ) / (Gᵉ Δq + Jᵃ)`
   (Δq-multiplied form, finite as `Δq → 0`).
8. The similarity solver recomputes `u★, q★` from the updated interface
   state; at convergence the vapor flux is `Jᵛ = −ρᵃᵗ u★ q★`.

The Δq-multiplied form avoids the `0/0` indeterminacy when the
near-iterate humidity matches `qᵃᵗ` exactly, and is the same trick the
existing `SkinHumidity` closure uses. Convergence is robust across the
dry / wet / windy / calm coverage matrix — see
[`test_dry_layer_humidity.jl`](https://github.com/NumericalEarth/NumericalEarth.jl/blob/main/test/test_dry_layer_humidity.jl).

### 4.3 Where this sits in the literature

Bare-soil evaporation proceeds in two stages
[Or et al. (2013)](@cite or2013advances). In *stage 1* the surface stays
hydraulically connected to moist soil below by capillary flow, the skin
is effectively saturated, and evaporation runs at the energy- and
demand-limited rate. Once the soil dries past a soil-specific
characteristic depth, the liquid network detaches, the evaporating front
retreats below the surface, and a *dry surface layer* (DSL) grows; in
*stage 2* evaporation is limited by Fickian vapor diffusion through that
layer.

Within the soil, vapor diffusion contributes meaningfully to the total
water flux only inside this thin, very dry layer — coupled liquid–vapor
theory goes back to [Philip and de Vries (1957)](@cite philip1957moisture),
and [Tang and Riley (2013)](@cite tang2013new) put the crossover where
the water-filled pore space falls below roughly a quarter. Everywhere
wetter, capillary liquid transport dominates. This is why land models —
including multi-layer Richards-equation models like GEOtop
[Endrizzi et al. (2014)](@cite endrizzi2014geotop) and
[ClimaLand](https://doi.org/10.1029/2025MS005118) — carry
*only liquid water* in the soil column and represent the dry-layer vapor
limitation as a surface condition
[Vanderborght et al. (2017)](@cite vanderborght2017heat). The neglected
process does not disappear; it reappears at the surface as a soil
resistance, an evaporation efficiency `β`, or a reduced surface
humidity `α qᵛ⁺`.

[`DryLayerHumidity`](@ref) is that surface condition, assembled from
three established pieces:

1. **The dry-layer resistance.** `Gᵉ = ρᵃᵗ Dᵛ_eff/δᵛ` is the reciprocal
   of the DSL soil resistance `r_soil = δᵛ/Dᵛ_eff` introduced by
   [Yamanaka et al. (1997)](@cite yamanaka1997surface) (resistance from
   the depth of the evaporating front, not from surface moisture) and
   adopted by CLM5 via
   [Swenson and Lawrence (2014)](@cite swenson2014dry), with the
   DSL thickness diagnosed from near-surface moisture exactly as our
   `δᵛ(𝒮)`. ClimaLand's bare-soil evaporation uses the same construction
   (`g_soil = Dᵛ/δ_DSL` with a saturation-dependent `δ_DSL`, maximum
   ≈ 15 mm). The [Millington and Quirk (1961)](@cite millington1961permeability)
   factor reduces `Dᵛ_eff` for partially wet pores; CLM5 uses a
   texture-dependent variant, ClimaLand omits it (`τ = 1`), and we make
   it optional.
2. **The flux balance.** Solving `Jᵉ = Jᵃ` for `qⁱⁿ` instead of
   prescribing `β(𝒮)` or `α(Π)` follows
   [Ye and Pielke (1993)](@cite yepielke1993), who derived the surface
   humidity from exactly this balance between in-pore vapor diffusion
   and the aerodynamic flux, and showed the two shortcut methods fail in
   complementary ways: `α qᵛ⁺` always overestimates evaporation from
   unsaturated soil, while a prescribed `β` is adequate by day but
   breaks at night because it cannot respond to the atmospheric state
   (see also the formulation comparison of
   [Mahfouf and Noilhan (1991)](@cite mahfouf1991comparative)). GEOtop
   adopts the Ye–Pielke parameterization directly as its surface
   evaporation scheme. In series-conductance form our solution
   `qⁱⁿ = (Gᵉ qᵉ + Gᵃ qᵃᵗ)/(Gᵉ + Gᵃ)` is algebraically the same
   expression ClimaLand evaluates with its `g_soil/g_h` ratio — we
   merely obtain the atmospheric conductance `Gᵃ` from the
   similarity-theory iterate instead of a fixed exchange coefficient.
3. **The front temperature.** Evaluating the source humidity at the
   interpolated front temperature `Tᵉ` rather than at the skin follows
   the same logic as Ye and Pielke's distinction between the
   surface-layer and in-soil source temperatures. (CLM5 and ClimaLand
   evaluate `qᵛ⁺` at the top-soil-layer temperature — the `χ → 1` limit.)

What we deliberately leave out of this first step: the Kelvin-equation
pore relative humidity `hₛ = exp(g Π/(Rᵛ Tᵉ))` (Ye and Pielke's `hₛ`,
present in CLM5 and ClimaLand as the `α` factor; only departs
appreciably from 1 at extreme dryness), texture-dependent tortuosity,
and any in-column vapor transport. Each slots in without structural
change.

## 5. Numerical robustness

A few notes for users running into convergence or stability issues:

* **Wet/dry branch is hard.** `δᵛ ≤ δᵛ_min` switches to `qⁱⁿ =
  qᵛ⁺(Tⁱⁿ)`. There is no smoothed blend in this PR; revisit only if
  convergence regressions appear.
* **Bounded humidity update.** The iteration update is followed by
  `qⁱⁿ ← clip(qⁱⁿ, 0, qᵛ⁺(Tⁱⁿ, pᵃᵗ))` to stay physical.
* **Positivity floor on `Mˡᵃ` only.** The augmented-storage framing
  intentionally has no upper clamp at `Mˡᵃ⁺` — overflow corresponds to
  `Π > 0`, not a state error.
* **Saturated-storage stiffness.** The Darcy deep-flux closure is
  acceptable as-is for `hˢˢ ≤ 10⁴ m`. Larger `hˢˢ` makes saturated
  overflow stiff; either document the constraint or fall back to
  `NoDeepLiquidFlux`. An implicit Darcy treatment is a follow-up.
* **Convergence tolerances.** Default `ε_q = 10⁻¹⁰ kg kg⁻¹`, `ε_T =
  10⁻⁴ K`, max 30 Picard iterations. The flux solver under-relaxes at
  `ω = 0.7`.

## 6. Parameter choices

A reasonable starting set for bare loamy soil:

| Symbol | Role | Suggested value | Notes |
|---|---|---:|---|
| `hˡᵃ` | depth of prognostic land | 1 m | classic Manabe-bucket depth |
| `ν` | porosity | 0.4 | loamy soil |
| `θʳ` | residual liquid fraction | 0.05 | finite for clay |
| `hˢˢ` | storage height | 10³–10⁴ m | larger ⇒ stiffer overflow |
| `𝒮ᶜ` | critical saturation | 0.5–0.75 | Manabe (1969) used 0.75 |
| `δᵛ_max` | maximum dry-layer depth | 0.015–0.05 m | CLM5 and ClimaLand default to 15 mm |
| `δᵛ_min` | minimum / wet-branch cutoff | 10⁻⁴ m | numerical, keeps `wᵈ` finite |
| `η` | dry-layer-depth exponent | 2 | steeper ⇒ later dry-down onset |
| `ℓᵀ` | thermal exchange depth | 0.10 m | sets χ = δᵛ/ℓᵀ |
| `κᵀ` | ground thermal conductivity | 0.5 W m⁻¹ K⁻¹ | loamy soil |
| `Dᵛ₀` | vapor diffusivity in air | 2.5 × 10⁻⁵ m² s⁻¹ | standard kinetic-theory value |

The [`MillingtonQuirk`](@ref) tortuosity model multiplies `Dᵛ_eff` by
`θᵍ^(10/3)/ν²` with `θᵍ = ν − θˡ` the gas-filled pore fraction
[Millington and Quirk (1961)](@cite millington1961permeability);
[`ConstantTortuosity`](@ref) skips this and uses `Dᵛ_eff = Dᵛ₀`.

## 7. The closure menu

`SlabLand` is a container; each physics slot accepts any closure from
the menus below, and the simpler entries remain first-class citizens —
the variably-saturated configuration of this tutorial is the most
elaborate column, not the only one.

**Energy** (`energy =`) — governs `Tˡᵃ`:

| Closure | `∂Tˡᵃ/∂t` | Use when |
|---|---|---|
| [`SlabEnergy`](@ref) | `Q/C` | pure slab; no deep ground coupling |
| [`ForceRestoreEnergy`](@ref) | `Q/C + (Tᵈᵉᵉᵖ − Tˡᵃ)/τ` | restoring toward a deep climatology (`SlabEnergy` is its `τ → ∞` limit) |
| [`WaterCoupledEnergy`](@ref) | conservative `dEˡᵃ/dt` form, §2.3 | `C(Mˡᵃ)`, advective water energy, exact `T`-invariance under water exchange |

(`Q` is the net energy flux into the slab, `C` the areal heat capacity,
`τ` the deep-restore time scale.)

**Hydrology** (`hydrology =`) — governs `Mˡᵃ` and the diagnostic `𝒮`:

| Closure | Behavior | Use when |
|---|---|---|
| [`DryLand`](@ref) | `𝒮 = 0`, no water | desert/idealized dry runs |
| [`SaturatedSurface`](@ref) | `𝒮 = 1`, no storage | swamp/idealized wet runs |
| [`BucketHydrology`](@ref) | Manabe bucket: fill to `Mˡᵃ_max`, spill | classic bucket experiments [Manabe (1969)](@cite manabe1969climate) |
| [`VariablySaturatedHydrology`](@ref) | conservative augmented-storage budget, §2 | retention physics, drainage, runoff |

`VariablySaturatedHydrology` composes four sub-closures: a retention
curve ([`VanGenuchtenRetention`](@ref)), a hydraulic conductivity
([`VanGenuchtenConductivity`](@ref)), a deep liquid flux
([`NoDeepLiquidFlux`](@ref), [`FreeDrainageFlux`](@ref),
[`DarcyDeepLiquidFlux`](@ref), [`LinearReservoirDrainage`](@ref)), and a
runoff model ([`NoRunoff`](@ref), [`InfiltrationCapacityRunoff`](@ref)).

**Interface humidity** (passed to `atmosphere_land_interface` via
`specific_humidity =`) — the atmosphere-facing `qⁱⁿ`. The three
formulations recapitulate the literature taxonomy of §4.3:

| Formulation | Scheme | Literature analogue |
|---|---|---|
| [`FractionalHumidity`](@ref) | `qⁱⁿ = β(𝒮) qᵛ⁺(Tⁱⁿ)` | the "β method" |
| [`SkinHumidity`](@ref) | flux balance, fixed soil conductance `κ^q/d` | constant soil resistance, after [Camillo and Gurney (1986)](@cite camillo1986resistance) |
| [`DryLayerHumidity`](@ref) | flux balance, dry-layer conductance `Dᵛ_eff/δᵛ(𝒮)` | DSL resistance [Swenson and Lawrence (2014)](@cite swenson2014dry); balance closure [Ye and Pielke (1993)](@cite yepielke1993) |

Finally, [`PrescribedLand`](@ref) replaces the prognostic component
entirely with dataset-driven surface fields (e.g. ERA5 skin temperature)
when no land physics should run at all.

## 8. Putting it together

```@example slabland
using NumericalEarth
using Oceananigans

grid = RectilinearGrid(size = (4, 4), x = (0, 1), y = (0, 1),
                       topology = (Periodic, Periodic, Flat))

land = SlabLand(grid;
    hydrology = VariablySaturatedHydrology(
        slab_depth = 1.0,
        porosity = 0.4,
        residual_liquid_fraction = 0.05,
        storage_height = 1000,
        critical_saturation = 0.5,
        retention_curve = VanGenuchtenRetention(α = 1.0, n = 2.0),
        hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-6, n = 2.0),
        deep_liquid_flux = FreeDrainageFlux(),
        runoff = InfiltrationCapacityRunoff(infiltration_capacity = 1e-3),
    ),
    energy = WaterCoupledEnergy(
        dry_heat_capacity = 1.5e6,
        liquid_heat_capacity = 4186,
        reference_temperature = 273.15,
        deep_temperature = 290.0,
        deep_time_scale = 12 * 3600,
        advect_deep_liquid_energy = true,
    ),
)
```

The humidity formulation belongs to the atmosphere–land interface, not
the land itself:

```@example slabland
interface_humidity = DryLayerHumidity(;
    dry_layer_depth = StorageBasedDryLayerDepth(
        maximum_dry_layer_depth = 0.05,
        critical_saturation = 0.5,
        dry_layer_exponent = 2),
    vapor_exchange = DryLayerVaporPistonVelocity(
        minimum_dry_layer_depth = 1e-4,
        molecular_diffusivity = 2.5e-5,
        tortuosity_model = MillingtonQuirk()),
    thermal_exchange_depth = 0.10,
    porosity = 0.4)
```

It is handed to the coupler via
`atmosphere_land_interface(grid, atmosphere, land; specific_humidity =
interface_humidity)` — see the Breeze example below for the full
coupled assembly.

## 9. Examples

Two coupled examples exercise the full atmosphere ↔ interface ↔
hydrology ↔ energy stack end-to-end:

* [Breeze over slab land](../literated/breeze_over_slab_land.md) — a 2D
  Breeze LES with a wet-center / dry-edge `Mˡᵃ` configuration. The
  wet/dry contrast in latent and sensible heat fluxes is driven by the
  saturation-dependent dry-layer depth `δᵛ(𝒮)`. Verified ratio: the
  wet patch evaporates approximately two orders of magnitude faster
  than the dry edges in the steady-state inversion above the heated
  surface.
* [ERA5-forced slab land](../literated/era5_forced_slab_land.md) — a
  ~1 km Greater Yellowstone simulation forced by ERA5 reanalysis,
  preserving the existing example's elevation-correction setup with the
  new closures swapped in.

## 10. Out of scope

The following are deferred to follow-up PRs:

* Snow, sea ice, vegetation, multi-layer soil columns, river routing.
* `Eˡᵃ` as a first-class state variable (needed for phase change).
* Kelvin-equation pore humidity — matric-suction-driven reduction of the
  dry-layer source humidity, `qᵉ = hₛ qᵛ⁺(Tᵉ)` with
  `hₛ = exp(g Π / (Rᵛ Tᵉ))` (the pore relative humidity `hₛ` of
  [Ye and Pielke (1993)](@cite yepielke1993), after Philip 1957). Only
  matters at extreme dryness; deferred.
* Land-side `SkinTemperature(DiffusiveFlux)` solve so that `Tⁱⁿ ≠ Tˡᵃ`
  and the χ interpolation has bite.
* `HydraulicInfiltrationRunoff`, `StorageOverflowRunoff`,
  `TopographicRunoff` (more elaborate runoff closures).
* Brooks–Corey retention curve.
* Two-node dry-layer energy solve (`Tᵉ` as a second residual).
* Implicit / semi-implicit deep Darcy treatment for large `hˢˢ`.
* Subgrid tile blending with ocean / sea ice.
