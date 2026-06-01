# Evaporation-front slab land

A conservative, variably-saturated bucket-land model with a new
atmosphere-facing humidity formulation that solves the surface specific
humidity `qⁱⁿ` from a dry-layer vapor balance against an unresolved
*evaporation front* at diagnostic depth `δᵛ`.

The model replaces the clamped `M + (P − E) Δt` bucket update with a
signed-flux conservative water budget, the constant-`C` force-restore
energy update with one that uses an `Mˡᵃ`-dependent heat capacity
`C(Mˡᵃ) = C_dry + cˡ Mˡᵃ`, and the `β·qᵛ⁺(Tˡᵃ)` moisture-availability
closure with a vapor-flux balance through a soil dry layer whose depth
varies with surface saturation.

The model is built around three new closures
([`VariablySaturatedBucketHydrology`](@ref),
[`WaterCoupledForceRestoreEnergy`](@ref),
[`EvaporationFrontHumidity`](@ref)) plus their sub-closures (deep liquid
flux, runoff, retention curve, evaporation-front depth, dry-layer vapor
exchange). Together they target bare ground without snow, ice, or
vegetation; see §10 ("Out of scope") below.

All math symbols below follow
[`docs/src/appendix/notation.md`](../appendix/notation.md).

## 1. What `SlabLand` stores

The composable container keeps its two-prognostic shape:

| Symbol | Code | Meaning |
|---|---|---|
| `Tˡᵃ` | `land.temperature` | bulk land temperature, K |
| `Mˡᵃ` | `land.water_storage` | slab water mass per land area, kg m⁻² |
| `𝒮`  | `land.saturation` | diagnostic surface saturation, recomputed from `Mˡᵃ` every step |

A new diagnostics slot (`land.diagnostics`) carries closure-published
fields: `deep_liquid_flux`, `surface_liquid_flux`, `surface_runoff`,
`subsurface_runoff`, and `water_storage_tendency`. The hydrology
publishes `water_storage_tendency`; the energy step consumes it for the
`Mˡᵃ`-dependent `Tˡᵃ` update (§4).

## 2. `Tˡᵃ` versus `Tⁱⁿ`

The plan introduces an atmosphere-facing skin/interface temperature
`Tⁱⁿ` distinct from the bulk `Tˡᵃ`. In this PR `Tⁱⁿ = Tˡᵃ` (the default
`BulkTemperature()` interface temperature is reused — land-side
`SkinTemperature` integration is deferred). The evaporation-front
temperature `Tᵉ = Tⁱⁿ + χ(Tˡᵃ − Tⁱⁿ)` therefore equals `Tˡᵃ` here, and
the wet/dry contrast comes entirely from the dry-layer piston velocity
`wᵈ = Dᵛ_eff/max(δᵛ, δᵛ_min)`, not from temperature interpolation.

## 3. Actual liquid fraction `θˡ` versus augmented `ϑˡ`

The conservative storage variable is the **augmented liquid fraction**

```math
\vartheta^l = \theta^l + S_s \max(\Pi, 0),
\qquad
\bar\vartheta^l = M^{la}/(\rho^l D),
```

which equals the physical pore liquid fraction `θˡ` when the soil is
unsaturated (`Π ≤ 0`) and carries the saturated *pressure* storage above
saturation. Crucially, `Mˡᵃ > Mˡᵃ⁺ = ρˡ ν D` is admitted and corresponds
to positive `Π` rather than a hard clamp. Surface physics (saturation,
moisture availability, surface humidity) reads the physical
`θˡ = min(ϑˡ, ν)`; deep Darcy fluxes read the head `Π` (negative
unsaturated, positive saturated overflow).

## 4. Conservative budgets

Water (positive upward at the surface and bottom):

```math
\frac{dM^{la}}{dt} = J^l_b - J^l_s - J^v - R^M_{lat}.
```

Energy (with `M`-dependent areal heat capacity
`C(Mˡᵃ) = C_dry + cˡ Mˡᵃ`):

```math
\frac{dE^{la}}{dt} = \Lambda^{deep}(T^{deep} - T^{la})
                    + e^l_{b,\text{up}} J^l_b
                    - J^E_s
                    - c^l(T^{la}-T_r) R^M_{lat},
```

```math
\frac{dT^{la}}{dt} = \frac{1}{C(M^{la})}\!\left[
                       \frac{dE^{la}}{dt} - c^l(T^{la} - T_r)\frac{dM^{la}}{dt}
                       \right].
```

The conservative `dTˡᵃ/dt` formula guarantees that adding or removing
water *at the slab temperature* leaves `Tˡᵃ` invariant — verified
numerically bit-exact in
[`test_water_coupled_force_restore_energy.jl`](https://github.com/NumericalEarth/NumericalEarth.jl/blob/main/test/test_water_coupled_force_restore_energy.jl).

## 5. Positive-upward flux convention; legacy exceptions

Every new flux field is positive upward: `Jᵛ` (vapor flux),
`Jˡ_b` / `Jˡ_s` (deep / surface liquid mass flux),
`Jᴱ_s = surface_energy_flux` (surface energy flux). The legacy
`land.fluxes.net_energy_flux` field keeps its existing convention
(positive *into* slab) for back-compatibility with [`SlabEnergy`](@ref)
and [`ForceRestoreEnergy`](@ref); both signs are written by the
interface, so closures pick the one they declared.

## 6. Runoff: rejected input versus storage export

* **Surface runoff** `Rᴹ_sfc` is a *rejected liquid input* (never
  enters storage). [`InfiltrationCapacityRunoff`](@ref) caps the
  downward `Jˡ_s` at the soil capacity; everything above the cap is
  surface runoff. Stored in `land.diagnostics.surface_runoff`.
* **Subsurface runoff** `Rᴹ_lat` is a *storage export* — appears in
  both the water and the energy budget. No subsurface-runoff closure
  ships in this PR (`NoRunoff` and `InfiltrationCapacityRunoff` both
  return 0); future closures like `TopographicRunoff` plug into the same
  hook.

Runoff is exposed via the public accessor `runoff(land)`. It becomes a
flux only when a river / ocean coupler consumes it.

## 7. Why solve `qⁱⁿ` instead of prescribing it

The traditional bucket-land closure prescribes
`qˢ = β(𝒮) qᵛ⁺(Tˡᵃ)`. Two known weaknesses:

1. The evaporation efficiency `β` is empirical and tuned per soil type.
2. The closure does not respond to *atmospheric* conditions — the same
   `β` is used for calm and windy days, for dry and humid air.

[`EvaporationFrontHumidity`](@ref) replaces this with a vapor-flux
balance between the soil and the atmosphere. The soil delivers vapor
across a dry-layer thickness `δᵛ` by Fickian diffusion:

```math
J^{soil} = G^e (q^e - q^{in}),
\qquad
G^e = \rho^{at}\,\frac{D^v_{eff}}{\max(\delta^v, \delta^v_{min})}.
```

The atmosphere carries vapor away at rate `Jᵃ = -ρᵃᵗ u★ q★`. The
balance `J^soil = Jᵃ` closes for `qⁱⁿ`:

```math
q^{in} = \frac{G^e q^e + G^a q^{at}}{G^e + G^a},
```

where `Gᵃ = Jᵃ/(qⁱⁿ⁻ − qᵃᵗ)` is the implicit atmospheric conductance
from the previous fixed-point iterate (the SkinHumidity trick — no
prescribed exchange coefficient). The wet branch `δᵛ ≤ δᵛ_min`
collapses to `qⁱⁿ = qᵛ⁺(Tⁱⁿ)`, reproducing the saturated-surface limit
exactly.

`δᵛ` is itself diagnostic of saturation via
[`StorageBasedEvaporationFrontDepth`](@ref):

```math
\delta^v(\mathcal S) = \delta^v_{max}\,\bigl[1 - \min(\mathcal S/\mathcal S^c, 1)\bigr]^\eta.
```

A wet surface has `δᵛ = 0` (skin saturated). A dry surface has
`δᵛ → δᵛ_max` (long Fickian path ⇒ small `wᵈ` ⇒ small `Jᵛ`). The result
is a self-consistent moisture-availability *response* rather than a
prescribed `β(𝒮)`.

## 8. Choosing parameters

A reasonable starting set for bare loamy soil:

| Symbol | Role | Suggested value | Notes |
|---|---|---:|---|
| `D` | slab depth | 1 m | classic Manabe-bucket depth |
| `ν` | porosity | 0.4 | loamy soil |
| `θʳ` | residual liquid fraction | 0.05 | finite for clay |
| `Sₛ` | specific storage | 10⁻³–10⁻⁴ m⁻¹ | smaller ⇒ stiffer overflow |
| `𝒮ᶜ` | critical saturation | 0.5–0.75 | Manabe (1969) used 0.75 |
| `δᵛ_max` | maximum evap-front depth | 0.05 m | empirical; 1–10 cm typical |
| `δᵛ_min` | minimum / wet-branch cutoff | 10⁻⁴ m | numerical, keeps `wᵈ` finite |
| `η` | front-depth exponent | 2 | steeper ⇒ later dry-down |
| `ℓᵀ` | thermal exchange depth | 0.10 m | depth over which `Λⁱⁿ = κᵀ/ℓᵀ` |
| `κᵀ` | ground thermal conductivity | 0.5 W m⁻¹ K⁻¹ | loamy soil |
| `Dᵛ₀` | vapor diffusivity in air | 2.5 × 10⁻⁵ m² s⁻¹ | standard kinetic theory value |

The tortuosity model `:millington_quirk` multiplies `Dᵛ_eff` by
`θᵍ^(10/3)/ν²` where `θᵍ = ν − θˡ` is the gas-filled pore fraction;
`:constant` skips this and uses `Dᵛ_eff = Dᵛ₀`.

## 9. Minimal target configuration

```julia
using NumericalEarth

land = SlabLand(land_grid;
    hydrology = VariablySaturatedBucketHydrology(
        slab_depth = 1.0,
        porosity = 0.4,
        residual_liquid_fraction = 0.05,
        specific_storage = 1e-3,
        critical_saturation = 0.5,
        retention_curve = VanGenuchtenRetention(α = 1.0, n = 2.0),
        hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-6, n = 2.0),
        deep_liquid_flux = FreeDrainageFlux(),
        runoff = InfiltrationCapacityRunoff(infiltration_capacity = 1e-3),
    ),
    energy = WaterCoupledForceRestoreEnergy(
        dry_heat_capacity = 1.5e6,
        liquid_heat_capacity = 4186,
        reference_temperature = 273.15,
        deep_temperature = 290.0,
        deep_time_scale = 12 * 3600,
        advect_deep_liquid_energy = true,
    ),
)

interface = EvaporationFrontHumidity(;
    evaporation_front_depth = StorageBasedEvaporationFrontDepth(
        maximum_front_depth = 0.05,
        critical_saturation = 0.5,
        front_depth_exponent = 2),
    vapor_exchange = DryLayerVaporPistonVelocity(
        minimum_front_depth = 1e-4,
        molecular_diffusivity = 2.5e-5,
        tortuosity_model = :millington_quirk),
    thermal_exchange_depth = 0.10,
    porosity = 0.4)
```

See the two coupled examples for runnable demonstrations:

* [Breeze over slab land](../literated/breeze_over_slab_land.md) — a 2D
  Breeze LES with the wet-center / dry-edge `Mˡᵃ` configuration; the
  wet/dry contrast in latent and sensible heat fluxes is driven by the
  saturation-dependent dry-layer depth `δᵛ(𝒮)`.
* [ERA5-forced slab land](../literated/era5_forced_slab_land.md) — a
  ~1 km Greater Yellowstone simulation forced by ERA5 reanalysis; the
  new conservative water budget and `M`-dependent heat capacity inherit
  the elevation correction from the original example.

## 10. Out of scope for this PR

The following are deferred to follow-up PRs:

* Snow, sea-ice, vegetation, multi-layer soil columns, river routing.
* `Eˡᵃ` as a first-class state variable (needed for phase change).
* `MatricPotentialActivity` — matric-suction-driven vapor-pressure reduction
  via the Kelvin equation (interface slot exists; implementation deferred,
  only relevant at extreme dryness).
* Land-side `SkinTemperature(DiffusiveFlux)` solve so that
  `Tⁱⁿ ≠ Tˡᵃ` and the `Tᵉ` interpolation has bite.
* `HydraulicInfiltrationRunoff`, `StorageOverflowRunoff`,
  `TopographicRunoff`.
* Brooks–Corey retention curve.
* Two-node evaporation-front energy solve (`Tᵉ` as a second residual).
* Implicit/semi-implicit deep Darcy treatment for small `Sₛ`.
* Subgrid tile blending with ocean / sea ice.
