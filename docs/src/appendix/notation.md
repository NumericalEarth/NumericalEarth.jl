# Notation

This page summarizes the mathematical and code notation used in NumericalEarth.jl,
following the conventions established in [Breeze.jl](https://github.com/CliMA/Breeze.jl).

## How the notation works

Variable names are built by combining a **base symbol** with **superscripts**
and, occasionally, a short plain-text **tag**.

**Base symbols** are single characters (often script letters) that identify the
physical category of a quantity — for example, `𝒬` for heat flux, `ℐ` for
radiative intensity, `J` for mass flux, and `τ` for kinematic momentum flux.

**Superscripts** refine the meaning in several ways:

- _Phase or species_: `ᵛ` (vapor), `ˡ` (liquid), `ⁱ` (ice), `ᶜ` (condensate)
- _Component_: `ᵃᵗ` (atmosphere), `ᵒᶜ` (ocean), `ˢⁱ` (sea ice), `ˡᵈ` (land)
- _Direction_: `ˣ` / `ʸ` (spatial), `ˢʷ` / `ˡʷ` (shortwave / longwave)
- _Process_: `ⁱⁿᵗ` (interface), `ᶠʳᶻ` (frazil)

**Modifier arrows** `ꜜ` (`\^downarrow`) and `ꜛ` (`\^uparrow`) denote
downwelling and upwelling directions in radiative fluxes.

**Subscripts** encode a small set of additional labels, such as
`★` (similarity-theory scale).

For example, `𝒬ᵛ` is the latent (vapor) heat flux, `ℐꜜˢʷ` is the downwelling
shortwave radiative intensity, and `τˣ` is the zonal kinematic momentum flux.

In Julia code, superscripts are entered with Unicode (e.g. `\scrQ<tab>` → `𝒬`,
then `\^v<tab>` → `ᵛ`). The modifier arrows `ꜜ` and `ꜛ` are entered with
`\^downarrow<tab>` and `\^uparrow<tab>`.

## Base flux symbols

| Math | Code | Tab completion | Meaning |
|:----:|:----:|:---------------|:--------|
| ``\mathcal{Q}`` | `𝒬` | `\scrQ` | Heat flux (W m⁻²) |
| ``\mathscr{I}`` | `ℐ` | `\scrI` | Radiative intensity (W m⁻²) |
| ``J`` | `J` | | Mass flux (kg m⁻² s⁻¹) |
| ``\tau`` | `τ` | `\tau` | Kinematic momentum flux (m² s⁻²) |
| ``\mathcal{L}`` | `ℒ` | `\scrL` | Latent heat (J kg⁻¹) |

Note: ``\tau^x`` (`τˣ`) is the _kinematic_ momentum flux (stress divided
by density). The mass-weighted stress is ``\rho \tau^x`` (`ρτˣ`, in N m⁻²).

These base symbols are combined with superscript and subscript labels
(documented below) to form specific variable names.

## Superscript and subscript labels

Superscripts and subscripts are used systematically to label physical quantities.
Superscripts generally denote the _type_ or _phase_ of a quantity, while subscripts
denote the _component_ or _location_.

### Superscript labels

| Label | Code | Meaning | Example |
|:-----:|:----:|:--------|:--------|
| ``v`` | `ᵛ` | water vapor | ``\mathcal{Q}^v`` (latent heat flux) |
| ``T`` | `ᵀ` | temperature / sensible | ``\mathcal{Q}^T`` (sensible heat flux) |
| ``c`` | `ᶜ` | condensate | ``J^c`` (precipitation mass flux) |
| ``S`` | `ˢ` | salinity | ``J^S`` (salinity flux) |
| ``i`` | `ⁱ` | ice | ``\mathcal{L}^i`` (latent heat of sublimation) |
| ``\ell`` | `ˡ` | liquid | ``\mathcal{L}^\ell`` (latent heat of vaporization) |
| ``p`` | `ᵖ` | constant pressure | ``c^{pm}`` (moist isobaric heat capacity) |
| ``m`` | `ᵐ` | mixture (moist air) | ``c^{pm}`` (moist isobaric heat capacity) |
| ``d`` | `ᵈ` | dry (air) | ``c^{pd}`` (dry air heat capacity) |
| ``D`` | `ᴰ` | drag | ``C^D`` (drag coefficient) |
| ``\mathrm{int}`` | `ⁱⁿᵗ` | interface | ``T^{\mathrm{int}}`` (interface temperature) |
| ``\mathrm{frz}`` | `ᶠʳᶻ` | frazil | ``\mathcal{Q}^{\mathrm{frz}}`` (frazil heat flux) |
| ``x`` | `ˣ` | zonal / x-direction | ``\tau^x`` (zonal kinematic stress) |
| ``y`` | `ʸ` | meridional / y-direction | ``\tau^y`` (meridional kinematic stress) |
| ``\mathrm{at}`` | `ᵃᵗ` | atmosphere | ``\rho^{\mathrm{at}}`` (air density) |
| ``\mathrm{oc}`` | `ᵒᶜ` | ocean | ``\rho^{\mathrm{oc}}`` (ocean reference density) |
| ``\mathrm{si}`` | `ˢⁱ` | sea ice | ``h^{\mathrm{si}}`` (sea ice thickness) |
| ``\mathrm{ld}`` | `ˡᵈ` | land | |
| ``\mathrm{sw}`` | `ˢʷ` | shortwave | ``\mathscr{I}`` ꜜ ``{}^{\mathrm{sw}}`` (downwelling shortwave) |
| ``\mathrm{lw}`` | `ˡʷ` | longwave | ``\mathscr{I}`` ꜜ ``{}^{\mathrm{lw}}`` (downwelling longwave) |

### Modifier arrows

| Symbol | Code | Tab completion | Meaning |
|:------:|:----:|:---------------|:--------|
| ꜜ | `ꜜ` | `\^downarrow` | downwelling |
| ꜛ | `ꜛ` | `\^uparrow` | upwelling |

### Subscript labels

| Label | Code | Meaning | Example |
|:-----:|:----:|:--------|:--------|
| ``\star`` | `★` | similarity theory scale | ``u_\star`` (friction velocity) |

## Atmosphere state variables

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``T`` | `T` | temperature | Air temperature (K) |
| ``p`` | `p` | pressure | Air pressure (Pa) |
| ``q`` | `q` | specific humidity | Mass mixing ratio of water vapor (kg kg⁻¹) |
| ``u`` | `u` | zonal velocity | Eastward wind component (m s⁻¹) |
| ``v`` | `v` | meridional velocity | Northward wind component (m s⁻¹) |
| ``\mathscr{I}_\downarrow^{\mathrm{sw}}`` | `ℐꜜˢʷ` | downwelling shortwave | Downwelling shortwave radiation (W m⁻²) |
| ``\mathscr{I}_\downarrow^{\mathrm{lw}}`` | `ℐꜜˡʷ` | downwelling longwave | Downwelling longwave radiation (W m⁻²) |
| ``J^c`` | `Jᶜ` | condensate flux | Precipitation (condensate) mass flux (kg m⁻² s⁻¹) |
| ``h_{b\ell}`` | `h_bℓ` | boundary layer height | Atmospheric boundary layer height (m) |

## Ocean state variables

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``T`` | `T` | temperature | Ocean potential temperature (ᵒC or K) |
| ``S`` | `S` | salinity | Practical salinity (g kg⁻¹) |
| ``u`` | `u` | zonal velocity | Eastward ocean velocity (m s⁻¹) |
| ``v`` | `v` | meridional velocity | Northward ocean velocity (m s⁻¹) |
| ``\rho^{\mathrm{oc}}`` | `ρᵒᶜ` | reference density | Ocean reference density (kg m⁻³) |
| ``c^{\mathrm{oc}}`` | `cᵒᶜ` | heat capacity | Ocean heat capacity (J kg⁻¹ K⁻¹) |

## Sea ice state variables

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``h^{\mathrm{si}}`` | `hˢⁱ` | ice thickness | Sea ice thickness (m) |
| ``\aleph`` | `ℵ` | ice concentration | Areal fraction of ice cover (–) |
| ``S^{\mathrm{si}}`` | `Sˢⁱ` | ice salinity | Sea ice bulk salinity (g kg⁻¹) |

## Radiation properties

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``\sigma`` | `σ` | Stefan–Boltzmann constant | (W m⁻² K⁻⁴) |
| ``\alpha`` | `α` | albedo | Surface reflectivity (–) |
| ``\epsilon`` | `ϵ` | emissivity | Surface emissivity (–) |

## Similarity theory / surface layer

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``u_\star`` | `u★` | friction velocity | Surface friction velocity (m s⁻¹) |
| ``\theta_\star`` | `θ★` | temperature scale | Flux characteristic temperature (K) |
| ``q_\star`` | `q★` | humidity scale | Flux characteristic specific humidity (kg kg⁻¹) |
| ``b_\star`` | `b★` | buoyancy scale | Flux characteristic buoyancy (m s⁻²) |
| ``L_\star`` | `L★` | Obukhov length | Monin–Obukhov length scale (m) |
| ``C^D`` | `Cᴰ` | drag coefficient | Bulk transfer coefficient for momentum (–) |
| ``\psi`` | `ψ` | stability function | Integrated stability correction (–) |
| ``\zeta`` | `ζ` | stability parameter | ``z / L_\star`` (–) |
| ``\ell`` | `ℓ` | roughness length | Aerodynamic roughness length (m) |
| ``\varkappa`` | `ϰ` | von Kármán constant | ``\approx 0.4`` (–) |

## Radiative fluxes

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``\mathscr{I}_\downarrow^{\mathrm{sw}}`` | `ℐꜜˢʷ` | downwelling shortwave | Downwelling shortwave radiation (W m⁻²) |
| ``\mathscr{I}_\downarrow^{\mathrm{lw}}`` | `ℐꜜˡʷ` | downwelling longwave | Downwelling longwave radiation (W m⁻²) |
| ``\mathscr{I}_\uparrow^{\mathrm{lw}}`` | `ℐꜛˡʷ` | upwelling longwave | Emitted longwave radiation (W m⁻²) |

Radiative fluxes use ``\mathscr{I}`` (`ℐ`, for "intensity") with a modifier
arrow (`ꜜ`/`ꜛ` for downwelling/upwelling) and superscript band (`ˢʷ`/`ˡʷ`).

## Turbulent interface fluxes

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``\mathcal{Q}^v`` | `𝒬ᵛ` | latent heat flux | Turbulent latent heat flux (W m⁻²) |
| ``\mathcal{Q}^T`` | `𝒬ᵀ` | sensible heat flux | Turbulent sensible heat flux (W m⁻²) |
| ``J^v`` | `Jᵛ` | water vapor flux | Turbulent mass flux of water vapor (kg m⁻² s⁻¹) |
| ``\tau^x`` | `τˣ` | zonal kinematic stress | Kinematic zonal momentum flux (m² s⁻²) |
| ``\tau^y`` | `τʸ` | meridional kinematic stress | Kinematic meridional momentum flux (m² s⁻²) |
| ``\rho \tau^x`` | `ρτˣ` | zonal wind stress | Mass-weighted zonal stress (N m⁻²) |
| ``\rho \tau^y`` | `ρτʸ` | meridional wind stress | Mass-weighted meridional stress (N m⁻²) |

## Net ocean fluxes

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``J^T`` | `Jᵀ` | temperature flux | Net ocean temperature flux (K m s⁻¹) |
| ``J^S`` | `Jˢ` | salinity flux | Net ocean salinity flux (g kg⁻¹ m s⁻¹) |
| ``\mathcal{Q}^{\mathrm{frz}}`` | `𝒬ᶠʳᶻ` | frazil heat flux | Heat released by frazil ice formation (W m⁻²) |

## Thermodynamic properties

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``\mathcal{L}^\ell`` | `ℒˡ` | latent heat of vaporization | Liquid-phase latent heat (J kg⁻¹) |
| ``\mathcal{L}^i`` | `ℒⁱ` | latent heat of sublimation | Ice-phase latent heat (J kg⁻¹) |
| ``c^{pm}`` | `cᵖᵐ` | moist air heat capacity | Moist isobaric specific heat (J kg⁻¹ K⁻¹) |
| ``c^{pd}`` | `cᵖᵈ` | dry air heat capacity | Dry-air isobaric specific heat (J kg⁻¹ K⁻¹) |
| ``\rho^{\mathrm{at}}`` | `ρᵃᵗ` | air density | Atmospheric air density (kg m⁻³) |

## CF standard name mapping

The following table maps code variable names to their
[CF standard names](http://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html)
where applicable.

| Code | CF standard name |
|:----:|:-----------------|
| `T` (atm) | `air_temperature` |
| `T` (ocn) | `sea_water_potential_temperature` |
| `S` | `sea_water_practical_salinity` |
| `u` (atm) | `eastward_wind` |
| `v` (atm) | `northward_wind` |
| `q` | `specific_humidity` |
| `p` | `air_pressure` |
| `ℐꜜˢʷ` | `surface_downwelling_shortwave_flux_in_air` |
| `ℐꜜˡʷ` | `surface_downwelling_longwave_flux_in_air` |
| `𝒬ᵛ` | `surface_upward_latent_heat_flux` |
| `𝒬ᵀ` | `surface_upward_sensible_heat_flux` |
| `Jᵛ` | `water_evapotranspiration_flux` |
| `ρτˣ` | `surface_downward_eastward_stress` |
| `ρτʸ` | `surface_downward_northward_stress` |
| `hˢⁱ` | `sea_ice_thickness` |
| `ℵ` | `sea_ice_area_fraction` |
