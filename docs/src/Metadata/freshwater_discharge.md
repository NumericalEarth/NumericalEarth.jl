# Freshwater discharge

River runoff and land-ice discharge reach the ocean through prescribed land components. Which ones
to use depends on the atmospheric forcing:

- **JRA55-do**: use [`JRA55PrescribedLand`](@ref) alone. It already includes Greenland's liquid
  runoff and solid ice discharge.
- **ERA5**: combine [`ERA5PrescribedAtmosphere`](@ref) with [`GloFASPrescribedLand`](@ref) for
  rivers and [`GEUSPrescribedLand`](@ref) for Greenland's solid ice discharge. ERA5 itself
  includes no Greenland discharge.

## Freshwater sources around Greenland and the Arctic

| | Typical magnitude | JRA55-do | ERA5 | GloFAS v4 | GEUS solid discharge |
|:---|:---|:---|:---|:---|:---|
| **Method** | | CaMa-Flood river routing of JRA-55 land runoff, plus [Bamber et al. (2018)](@cite Bamber2018) for land ice [Tsujino et al. (2018)](@cite tsujino2018jra) | Atmospheric reanalysis; no river or land-ice discharge | LISFLOOD hydrology model forced by ERA5 [Harrigan et al. (2020)](@cite harrigan2020glofas) | Ice velocity and thickness through flux gates [Mankoff et al. (2020)](@cite Mankoff2020solid) |
| **Major Arctic rivers** | ~4000 km³/yr | ✅ Rescaled to observed basin totals | ❌ | ✅ Calibrated against gauges | ❌ |
| **Other Arctic glaciers** | ~100–200 km³/yr | ✅ Liquid runoff from Bamber et al. (2018) | ❌ | ❌ Treated as deep snow | ❌ |
| **Greenland liquid runoff** | ~400–550 km³/yr | ✅ Bamber et al. (2018) | ❌ | ✅ But ~250 km³/yr, too low | ❌ |
| **Greenland ice discharge** | ~500 km³/yr | ✅ `licalvf` (`:iceberg_freshwater_flux`) | ❌ | ❌ | ✅ Per gate |
| **Basal melt** | ~20 km³/yr | ❌ | ❌ | ❌ | ❌ |
| **In NumericalEarth** | | [`JRA55PrescribedLand`](@ref) | [`ERA5PrescribedAtmosphere`](@ref) | [`GloFASPrescribedLand`](@ref) | [`GEUSPrescribedLand`](@ref) |

1 km³/yr of water is about 1 Gt/yr, or 32 m³ s⁻¹. Typical magnitudes are rough literature values.

Solid ice discharge enters the ocean as the `icebergs` freshwater component. Unlike liquid runoff,
it also takes from the ocean the latent heat needed to melt it (`latent_heat_of_fusion` in
[`ComponentInterfaces`](@ref), 334 kJ kg⁻¹ by default). It melts where it reaches the coast;
icebergs do not drift.
