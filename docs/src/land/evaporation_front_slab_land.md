# SlabLand tutorial: conservative variably saturated land

`SlabLand` is NumericalEarth's prognostic single-layer land-surface
component. This tutorial walks through the variably saturated
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
§11 ("Limitations") below. §7 surveys the full closure menu, including
the simpler configurations (`SlabEnergy`, `BucketHydrology`,
`FractionalHumidity`).

## 1. What `SlabLand` stores

The composable container keeps a two-prognostic shape:

| Symbol | Code | Meaning |
|---|---|---|
| `Tˡᵃ` | `land.temperature` | bulk land temperature, K |
| `Mˡᵃ` | `land.water_storage` | slab water mass per land area, kg m⁻² |
| `𝒮`  | `land.saturation` | diagnostic surface saturation, recomputed from `Mˡᵃ` every step |

Land variables wear the component superscript `ˡᵃ` throughout this page
because they appear alongside atmosphere (`ᵃᵗ`) and interface (`ⁱⁿ`)
variables. Within the land model itself the bare symbols are used —
`prognostic_fields(land)` returns `(; T, M)` — per the
component-superscript rule in the
[notation appendix](../appendix/notation.md).

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
| signed | `vapor_flux`, `surface_energy_flux` | positive upward ⇒ leaves slab |
| signed | `liquid_precipitation_flux` (`Pˡ`) | positive downward ⇒ enters slab |
| positive-part | `net_energy_flux`, `evaporation`, `precipitation` | positive ⇒ into slab |

Both sets are populated by the coupler's `update_net_fluxes!` every
step; each closure declares which fields it reads via `flux_variables`
(`WaterCoupledEnergy` reads `surface_energy_flux`, `ForceRestoreEnergy`
reads `net_energy_flux`).

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
\theta^l(\Pi), & \Pi \le 0, \\[2pt]
\nu + \dfrac{\Pi}{h^{\mathrm{ss}}}, & \Pi > 0.
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
\Pi_m(\bar\theta^l), & M^{la} < M^{la+}\quad (\text{unsaturated}), \\[2pt]
\dfrac{(M^{la} - M^{la+})\,h^{\mathrm{ss}}}{\rho^l h^{\mathrm{la}}}, & M^{la} \ge M^{la+}\quad (\text{saturated overflow}),
\end{cases}
```

```math
\mathcal S = \operatorname{clip}\!\left(\frac{\bar\theta^l - \theta^r}{\nu - \theta^r},\ 0,\ 1\right),
\qquad
\beta = \operatorname{clip}\!\left(\frac{\mathcal S}{\mathcal S^c},\ 0,\ 1\right).
```

Three things to keep separate:

* `ϑˡ` (and `Mˡᵃ`) — the **conservative** storage variable; can exceed `ν`.
* `θˡ` — the **physical** pore liquid fraction; what surface physics
  reads (albedo, soil-air vapor diffusivity).
* `Π` — the **head** variable; what Darcy fluxes and pressure-storage
  saturated overflow read. Negative unsaturated, positive once
  saturated.

`Π_m(θ̄ˡ)` is the unsaturated retention curve
([`VanGenuchtenRetention`](@ref)).

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
  `dEˡᵃ/dt`. Stored in `land.diagnostics.subsurface_runoff`. Both
  available runoff closures (`NoRunoff`,
  `InfiltrationCapacityRunoff`) produce zero subsurface runoff.

Both categories live in `land.diagnostics` (`surface_runoff`,
`subsurface_runoff`); they become a flux only when a river / ocean
coupler consumes them.

## 4. The dry-layer humidity model

A traditional bucket-land closure prescribes `qⁱⁿ = β(𝒮) qᵛ⁺(Tⁱⁿ)`. Two
known weaknesses:

1. The evaporation efficiency `β` is empirical and tuned per soil type.
2. The closure does not respond to *atmospheric* conditions — the same
   `β` is used for calm/windy days, dry/humid air.

[`DryLayerHumidity`](@ref) instead *poses an equation* for `qⁱⁿ`: a
vapor-flux balance between the soil and the atmosphere. §4.1 describes
the physical model; §4.2 describes how the nonlinear problem it poses is
solved.

### 4.1 The model

**Dry-layer depth.** The depth `δᵛ` of the evaporating front below the
surface is diagnosed from saturation via
[`StorageBasedDryLayerDepth`](@ref):

```math
\delta^v(\mathcal S) = \delta^v_{max}
\left[1 - \min\!\left(\frac{\mathcal S}{\mathcal S^c},\ 1\right)\right]^\eta.
```

A wet surface has `δᵛ = 0` (the front sits at the skin); as the slab
dries, `δᵛ → δᵛ_max`.

**Front temperature and source humidity.** The pore air at the front is
saturated, so the vapor source is the *saturation specific humidity at
the front temperature*,

```math
q^e = q^{v+}(T^e, p^{at}),
```

where the front temperature `Tᵉ` is interpolated between the
atmosphere-facing skin `Tⁱⁿ` and the bulk soil `Tˡᵃ` according to how
deep the front sits relative to the thermal exchange scale `ℓᵀ`:

```math
T^e = T^{in} + \chi\,(T^{la} - T^{in}),
\qquad
\chi = \operatorname{clip}\!\left(\frac{\delta^v}{\ell^T},\ 0,\ 1\right).
```

When the front is at the surface (`δᵛ = 0`) the source temperature is
the skin; when the front has retreated to `δᵛ ≥ ℓᵀ` it is the bulk
soil. With the `BulkTemperature()` interface formulation the land uses,
`Tⁱⁿ = Tˡᵃ`, so `Tᵉ` collapses to `Tˡᵃ` and the wet/dry contrast comes
entirely from the dry-layer conductance; the χ interpolation acquires
bite once a skin-temperature formulation makes `Tⁱⁿ ≠ Tˡᵃ` (§11).

**The flux balance.** Vapor Fick-diffuses from the front up to the
interface,

```math
J^e = G^e \left[q^{v+}(T^e) - q^{in}\right],
\qquad
G^e = \rho^{at}\,\frac{D^v_{eff}}{\max(\delta^v, \delta^v_{min})},
```

while above the interface the atmosphere carries vapor away at the
Monin–Obukhov similarity rate

```math
J^a = -\rho^{at} u_\star q_\star,
```

where the scales `u★` and `q★` are themselves nonlinear functions of
the interface state `(Tⁱⁿ, qⁱⁿ)`. The interface stores no vapor, so the
two fluxes balance:

```math
G^e \left[q^{v+}(T^e) - q^{in}\right] = J^a(T^{in}, q^{in}).
```

This is the model: a nonlinear equation for `qⁱⁿ`, coupled to the
skin-temperature balance through `Tⁱⁿ`. When the surface is wet enough
(`𝒮 ≥ 𝒮ᶜ`, so `δᵛ ≤ δᵛ_min`) the front co-locates with the skin and the
model reduces to the saturated-skin condition `qⁱⁿ = qᵛ⁺(Tⁱⁿ)`,
reproducing the saturated-surface limit exactly. A dry surface has a
long Fickian path (small `Gᵉ`), throttling evaporation — a
self-consistent moisture-availability *response*, not a prescribed
`β(𝒮)`.

### 4.2 Numerical solution: Picard iteration

The atmosphere–land flux solver iterates the interface state
`(Tⁱⁿ, qⁱⁿ)` to convergence. Within one iteration the similarity scales
are evaluated at the *previous* iterate `qⁱⁿ⁻`, which linearizes the
atmospheric flux into a conductance law anchored there:

```math
J^a(q) \approx G^a\,(q - q^{at}),
\qquad
G^a = \frac{J^a}{\Delta q},
\qquad
\Delta q = q^{in-} - q^{at},
```

chosen so the linearization reproduces the flux `Jᵃ = −ρᵃᵗ u★ q★` the
similarity solver actually returned. The balance of §4.1 then has the
two-conductances-in-series solution

```math
q^{in} = \frac{G^e q^e + G^a q^{at}}{G^e + G^a}
       = \frac{G^e q^e \Delta q + J^a q^{at}}{G^e \Delta q + J^a},
```

where the second, `Δq`-multiplied form is the one implemented: it stays
finite as `Δq → 0` (the same trick `SkinHumidity` uses — no prescribed
exchange coefficient). Each iteration therefore:

1. Diagnoses `δᵛ(𝒮)`, `Tᵉ`, and `qᵉ = qᵛ⁺(Tᵉ, pᵃᵗ)`.
2. Obtains `u★, q★` (hence `Jᵃ`) from the similarity scheme at the
   previous iterate.
3. Updates the humidity — **wet branch** (`δᵛ ≤ δᵛ_min`):
   `qⁱⁿ ← qᵛ⁺(Tⁱⁿ, pᵃᵗ)`; **dry branch**: the `Δq`-multiplied series
   solution above.
4. Updates `Tⁱⁿ` from the skin-temperature balance.

At the fixed point `qⁱⁿ = qⁱⁿ⁻` the linearization is exact, so the
converged humidity satisfies the *nonlinear* model of §4.1, and the
vapor flux is `Jᵛ = −ρᵃᵗ u★ q★`. Convergence is robust across the
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

The formulation omits the Kelvin-equation pore relative humidity
`hₛ = exp(g Π/(Rᵛ Tᵉ))` (Ye and Pielke's `hₛ`, present in CLM5 and
ClimaLand as the `α` factor; it departs appreciably from 1 only at
extreme dryness), texture-dependent tortuosity, and in-column vapor
transport.

## 5. Numerical robustness

A few notes for users running into convergence or stability issues:

* **Wet/dry branch is hard.** `δᵛ ≤ δᵛ_min` switches to `qⁱⁿ =
  qᵛ⁺(Tⁱⁿ)`; there is no smoothed blend.
* **Degenerate-denominator guard.** When the series denominator
  `Gᵉ Δq + Jᵃ` vanishes the update returns the previous iterate
  unchanged. There is no clamp on `qⁱⁿ` between iterations (matching
  `SkinHumidity`); transient out-of-range iterates are corrected by the
  iteration itself.
* **Positivity floor on `Mˡᵃ` only.** The augmented-storage framing
  intentionally has no upper clamp at `Mˡᵃ⁺` — overflow corresponds to
  `Π > 0`, not a state error.
* **Saturated-storage stiffness.** With `DarcyDeepLiquidFlux`, storage
  heights `hˢˢ ≤ 10⁴ m` are well behaved; larger values make saturated
  overflow stiff — reduce `hˢˢ` or use `NoDeepLiquidFlux`.
* **Convergence criterion.** The flux solver stops when the similarity
  scales settle, `|Δu★| + |Δθ★| + |Δq★| < 10⁻⁸` (default), or after 100
  iterations — configurable via `solver_tolerance` and `solver_maxiter`
  of the flux formulation.

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
interface_humidity)` — the case study below and the Breeze example in
§10 show the full coupled assembly.

## 9. Case study: a slab drying under a warm atmosphere

The two-stage behavior of §4.3 can be reproduced in seconds with a
single-column `SlabLand` under a constant warm, dry atmosphere and
prescribed radiation. With no deep-drainage or runoff closures,
evaporation is the only water sink, so the dry-down is driven entirely
by the vapor-flux balance.

A thin (20 cm) slab starts at 90% of its saturated storage:

```@example slabland
using Oceananigans.Units

column = LatitudeLongitudeGrid(size = 1, latitude = 35, longitude = 0,
                               z = (-1, 0), topology = (Flat, Flat, Bounded))

slab = SlabLand(column;
    hydrology = VariablySaturatedHydrology(
        slab_depth = 0.2,
        porosity = 0.4,
        residual_liquid_fraction = 0.05,
        storage_height = 1000,
        critical_saturation = 0.5,
        retention_curve = VanGenuchtenRetention(α = 1.0, n = 2.0),
        hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-6, n = 2.0)),
    energy = WaterCoupledEnergy(
        dry_heat_capacity = 3e5,
        liquid_heat_capacity = 4186,
        deep_temperature = 295,
        deep_time_scale = 2days))

set!(slab; T = 300, M = 0.9 * 1000 * 0.4 * 0.2)   # M = 0.9 Mˡᵃ⁺, Mˡᵃ⁺ = ρˡ ν hˡᵃ
```

Force it with a constant warm, dry atmosphere and prescribed downwelling
radiation, then couple through the same [`DryLayerHumidity`](@ref)
interface built in §8. A `Number` passed to `set!` is a constant in
space and time — a steady forcing:

```@example slabland
atmosphere = PrescribedAtmosphere(column; surface_layer_height = 10,
                                          boundary_layer_height = 512)
set!(atmosphere; u = 3, T = 305, q = 0.004, p = 101_325)   # warm (305 K), dry (4 g kg⁻¹)

radiation = PrescribedRadiation(column; land_surface = SurfaceRadiationProperties(0.3, 0.95))
set!(radiation; downwelling_shortwave = 250, downwelling_longwave = 350)   # W m⁻²

interface = atmosphere_land_interface(column, atmosphere, slab;
                                      specific_humidity = interface_humidity)
model = AtmosphereLandModel(atmosphere, slab; radiation,
                            atmosphere_land_interface = interface)
```

Step forward twelve days, recording the slab state and the vapor flux:

```@example slabland
Δt = 10minutes
N  = Int(12days ÷ Δt)

days_elapsed = zeros(N)
saturation   = zeros(N)
evaporation  = zeros(N)
temperature  = zeros(N)

for n in 1:N
    time_step!(model, Δt)
    days_elapsed[n] = model.clock.time / day
    saturation[n]   = slab.saturation[1, 1, 1]
    evaporation[n]  = slab.fluxes.vapor_flux[1, 1, 1] * day   # kg m⁻² s⁻¹ → mm day⁻¹
    temperature[n]  = slab.temperature[1, 1, 1]
end
```

```@example slabland
using CairoMakie

fig = Figure(size = (700, 700))

axS = Axis(fig[1, 1], ylabel = "saturation 𝒮")
lines!(axS, days_elapsed, saturation)
hlines!(axS, [0.5], linestyle = :dash, color = :gray)
text!(axS, 0.4, 0.52, text = "𝒮ᶜ", color = :gray)

axJ = Axis(fig[2, 1], ylabel = "evaporation (mm day⁻¹)")
lines!(axJ, days_elapsed, evaporation)

axT = Axis(fig[3, 1], ylabel = "Tˡᵃ (K)", xlabel = "time (days)")
lines!(axT, days_elapsed, temperature)

linkxaxes!(axS, axJ, axT)
hidexdecorations!(axS, grid = false)
hidexdecorations!(axJ, grid = false)

fig
```

Both stages of §4.3 appear. After a few hours of adjustment from the
initial condition, the front sits at the skin while `𝒮 > 𝒮ᶜ` (the first
week): evaporation holds a demand-limited plateau near 3.4 mm day⁻¹ and
the slab temperature is steady near 298 K, evaporative cooling removing
most of the net surface heating. Once `𝒮` crosses `𝒮ᶜ` the dry layer
grows, the Fickian path lengthens, and evaporation collapses (to about a
third of the plateau by day twelve) while the slab *warms* by several
kelvin: the surface energy that latent heat no longer carries away is
repartitioned into sensible heat and ground warming. The small notch
right at the crossing is the hard wet/dry branch switch of §5. No
`β(𝒮)` function was prescribed anywhere — the transition emerges from
the flux balance.

## 10. Examples

Two coupled examples exercise the full atmosphere ↔ interface ↔
hydrology ↔ energy stack end-to-end:

* [Breeze over slab land](../literated/breeze_over_slab_land.md) — a 2D
  Breeze LES with a wet-center / dry-edge `Mˡᵃ` configuration. The
  wet/dry contrast in latent and sensible heat fluxes is driven by the
  saturation-dependent dry-layer depth `δᵛ(𝒮)`: the wet patch
  evaporates approximately two orders of magnitude faster than the dry
  edges in the steady-state inversion above the heated surface.
* [ERA5-forced slab land](../literated/era5_forced_slab_land.md) — a
  ~1 km Greater Yellowstone simulation forced by ERA5 reanalysis with
  elevation-corrected forcing.

## 11. Limitations

What `SlabLand` does not represent, for judging whether it fits a
problem:

* **Bare ground only** — no snow, vegetation, or multi-layer soil
  columns; every land grid cell is wholly land (no subgrid tiles shared
  with ocean or sea ice).
* **No soil-water phase change** — slab water does not freeze or thaw.
  (Sublimation over frozen ground is still representable on the
  interface side: [`DryLayerHumidity`](@ref) accepts an ice `phase` for
  the saturation humidity.)
* **No river routing** — runoff accumulates in `land.diagnostics` but
  is not transported anywhere.
* **The skin temperature equals the bulk temperature** (`Tⁱⁿ = Tˡᵃ`),
  so the front-temperature interpolation of §4.1 is inert and the
  diurnal skin cycle is damped by the full slab heat capacity.
* **The dry-layer source humidity is exactly saturated** — there is no
  matric-suction (Kelvin) reduction `hₛ = exp(g Π/(Rᵛ Tᵉ))` of the pore
  relative humidity, which overestimates the vapor source at extreme
  dryness (§4.3).
