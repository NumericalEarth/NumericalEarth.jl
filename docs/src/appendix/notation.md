# Notation

This page summarizes the mathematical and code notation used in NumericalEarth.jl,
following the conventions established in [Breeze.jl](https://github.com/CliMA/Breeze.jl).

## Atmosphere state variables

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``T`` | `T` | temperature | Air temperature (K) |
| ``p`` | `p` | pressure | Air pressure (Pa) |
| ``q`` | `q` | specific humidity | Mass mixing ratio of water vapor (kg kg⁻¹) |
| ``u`` | `u` | zonal velocity | Eastward wind component (m s⁻¹) |
| ``v`` | `v` | meridional velocity | Northward wind component (m s⁻¹) |
| ``Q_s`` | `Qs` | downwelling shortwave | Downwelling shortwave radiation (W m⁻²) |
| ``Q_\ell`` | `Qℓ` | downwelling longwave | Downwelling longwave radiation (W m⁻²) |
| ``J^c`` | `Jᶜ` | condensate flux | Precipitation (condensate) mass flux (kg m⁻² s⁻¹) |
| ``h_{b\ell}`` | `h_bℓ` | boundary layer height | Atmospheric boundary layer height (m) |

## Ocean state variables

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``T`` | `T` | temperature | Ocean potential temperature (ᵒC or K) |
| ``S`` | `S` | salinity | Practical salinity (g kg⁻¹) |
| ``u`` | `u` | zonal velocity | Eastward ocean velocity (m s⁻¹) |
| ``v`` | `v` | meridional velocity | Northward ocean velocity (m s⁻¹) |
| ``\rho_o`` | `ρₒ` | reference density | Ocean reference density (kg m⁻³) |
| ``c_o`` | `cₒ` | heat capacity | Ocean heat capacity (J kg⁻¹ K⁻¹) |

## Sea ice state variables

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``h_i`` | `hᵢ` | ice thickness | Sea ice thickness (m) |
| ``\aleph`` | `ℵ` | ice concentration | Areal fraction of ice cover (–) |
| ``S^i`` | `Sⁱ` | ice salinity | Sea ice bulk salinity (g kg⁻¹) |

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

## Turbulent interface fluxes

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``\mathcal{Q}^v`` | `𝒬ᵛ` | latent heat flux | Turbulent latent heat flux (W m⁻²) |
| ``\mathcal{Q}^T`` | `𝒬ᵀ` | sensible heat flux | Turbulent sensible heat flux (W m⁻²) |
| ``J^v`` | `Jᵛ` | water vapor flux | Turbulent mass flux of water vapor (kg m⁻² s⁻¹) |
| ``\rho \tau_x`` | `ρτx` | zonal momentum flux | Zonal wind stress (N m⁻²) |
| ``\rho \tau_y`` | `ρτy` | meridional momentum flux | Meridional wind stress (N m⁻²) |

## Net ocean fluxes

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``J^T`` | `Jᵀ` | temperature flux | Net ocean temperature flux (K m s⁻¹) |
| ``J^S`` | `Jˢ` | salinity flux | Net ocean salinity flux (g kg⁻¹ m s⁻¹) |

## Thermodynamic properties

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``\mathcal{L}^\ell`` | `ℒˡ` | latent heat of vaporization | Liquid-phase latent heat (J kg⁻¹) |
| ``\mathcal{L}^i`` | `ℒⁱ` | latent heat of sublimation | Ice-phase latent heat (J kg⁻¹) |
| ``c_p`` | `cₚ` | heat capacity of air | Moist isobaric heat capacity (J kg⁻¹ K⁻¹) |
| ``\rho_a`` | `ρₐ` | air density | Atmospheric air density (kg m⁻³) |

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
| `Qs` | `surface_downwelling_shortwave_flux_in_air` |
| `Qℓ` | `surface_downwelling_longwave_flux_in_air` |
| `𝒬ᵛ` | `surface_upward_latent_heat_flux` |
| `𝒬ᵀ` | `surface_upward_sensible_heat_flux` |
| `Jᵛ` | `water_evapotranspiration_flux` |
| `ρτx` | `surface_downward_eastward_stress` |
| `ρτy` | `surface_downward_northward_stress` |
| `hᵢ` | `sea_ice_thickness` |
| `ℵ` | `sea_ice_area_fraction` |
