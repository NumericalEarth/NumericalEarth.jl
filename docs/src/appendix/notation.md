# Notation

This page summarizes the mathematical and code notation used in NumericalEarth.jl,
following the conventions established in [Breeze.jl](https://github.com/NumericalEarth/Breeze.jl).

## How the notation works

Variable names are built by combining a **base symbol** with **superscripts** and, occasionally, a short plain-text **tag**.

**Base symbols** are single characters (often script letters) that identify the physical category of a quantity — for example, `𝒬` for heat flux, `ℐ` for radiative intensity, `J` for mass flux, and `τ` for kinematic momentum flux.

**Superscripts** refine the meaning in several ways:

- _Phase or species_: `ᵛ` (vapor), `ˡ` (liquid), `ⁱ` (ice), `ᶜ` (condensate)
- _Component_: `ᵃᵗ` (atmosphere), `ᵒᶜ` (ocean), `ˢⁱ` (sea ice), `ˡᵃ` (land)
- _Interface pair_: `ᵃᵒ` (atm–ocean), `ᵃⁱ` (atm–ice), `ⁱᵒ` (ice–ocean)
- _Direction_: `ˣ` / `ʸ` (spatial), `ˢʷ` / `ˡʷ` (shortwave / longwave)
- _Process_: `ⁱⁿ` (interface), `ᶠʳᶻ` (frazil)

Component superscripts are used only in *cross-component* context — wherever
a variable appears alongside variables from other components, as in interface
computations or coupled-model discussions (`Tˡᵃ` next to `Tᵃᵗ` and `Tⁱⁿ`).
Within a single component's own namespace the bare symbol is used: the land
model's prognostic state is `(; T, M)` (as returned by
`prognostic_fields(land)`), just as the ocean's is `(; u, v, w, T, S)`.

**Modifier arrows** `ꜜ` (`\^downarrow`) and `ꜛ` (`\^uparrow`) denote
downwelling and upwelling directions in radiative fluxes.

**Subscripts** encode radiative process (`ₜ` transmitted, `ₐ` absorbed)
and the similarity-theory scale `★`.

For example, `𝒬ᵛ` is the latent (vapor) heat flux, `ℐꜜˢʷ` is the downwelling shortwave radiative intensity, and `τˣ` is the zonal kinematic momentum flux.

In Julia code, superscripts are entered with Unicode (e.g. `\scrQ<tab>` → `𝒬`, then `\^v<tab>` → `ᵛ`). The modifier arrows `ꜜ` and `ꜛ` are entered with `\^downarrow<tab>` and `\^uparrow<tab>`.

## Base symbols

| Math | Code | Tab completion | Meaning |
|:----:|:----:|:---------------|:--------|
| ``\mathcal{Q}`` | `𝒬` | `\scrQ` | Heat flux (W m⁻²) |
| ``\mathscr{I}`` | `ℐ` | `\scrI` | Radiative intensity (W m⁻²) |
| ``J`` | `J` | | Mass flux (kg m⁻² s⁻¹) |
| ``\tau`` | `τ` | `\tau` | Kinematic momentum flux (m² s⁻²) |
| ``\mathcal{L}`` | `ℒ` | `\scrL` | Latent heat (J kg⁻¹) |
| ``M`` | `M` | | Layer-integrated mass per area (kg m⁻²) |
| ``\aleph`` | `ℵ` | `\aleph` | Area fraction: the fraction of an enclosing surface covered by the labeled component (–) |
| ``P`` | `P` | | Precipitation (liquid water) mass flux reaching a surface (kg m⁻² s⁻¹) |

Note: ``\tau^x`` (`τˣ`) is the _kinematic_ momentum flux (stress divided
by density). The mass-weighted stress is ``\rho \tau^x`` (`ρτˣ`, in N m⁻²).

These base symbols are combined with superscript and subscript labels
(documented below) to form specific variable names.

## Superscript and subscript labels

Superscripts and subscripts are used systematically to label physical quantities.
Superscripts generally denote the _type_ or _phase_ of a quantity, while subscripts denote the _component_ or _location_.

### Superscript labels

| Label | Code | Meaning | Example |
|:-----:|:----:|:--------|:--------|
| ``v`` | `ᵛ` | water vapor | ``\mathcal{Q}^v`` (latent heat flux) |
| ``T`` | `ᵀ` | temperature / sensible | ``\mathcal{Q}^T`` (sensible heat flux) |
| ``\mathrm{rn}`` | `ʳⁿ` | rain | ``J^{\mathrm{rn}}`` (rainfall) |
| ``\mathrm{sn}`` | `ˢⁿ` | snow | ``J^{\mathrm{sn}}`` (snowfall) |
| ``S`` | `ˢ` | salinity | ``J^S`` (salinity flux) |
| ``w`` | `ʷ` | freshwater | ``J^w`` (freshwater volume flux per unit area) |
| ``h`` | `ʰ` | height (roughness element) | ``\sigma^h`` (height standard deviation) |
| ``i`` | `ⁱ` | ice | ``\mathcal{L}^i`` (latent heat of sublimation) |
| ``\ell`` | `ˡ` | liquid | ``\mathcal{L}^\ell`` (latent heat of vaporization) |
| ``p`` | `ᵖ` | constant pressure | ``c^{pm}`` (moist isobaric heat capacity) |
| ``m`` | `ᵐ` | mixture (moist air) | ``c^{pm}`` (moist isobaric heat capacity) |
| ``d`` | `ᵈ` | dry (air) | ``c^{pd}`` (dry air heat capacity) |
| ``D`` | `ᴰ` | drag | ``C^D`` (drag coefficient) |
| ``\mathrm{in}`` | `ⁱⁿ` | interface | ``T^{\mathrm{in}}`` (interface temperature) |
| ``\mathrm{frz}`` | `ᶠʳᶻ` | frazil | ``\mathcal{Q}^{\mathrm{frz}}`` (frazil heat flux) |
| ``x`` | `ˣ` | zonal / x-direction | ``\tau^x`` (zonal kinematic stress) |
| ``y`` | `ʸ` | meridional / y-direction | ``\tau^y`` (meridional kinematic stress) |
| ``\mathrm{at}`` | `ᵃᵗ` | atmosphere | ``\rho^{\mathrm{at}}`` (air density) |
| ``\mathrm{ae}`` | `ᵃᵉ` | air entry | ``\alpha^{\mathrm{ae}}`` (inverse air-entry head) |
| ``\mathrm{oc}`` | `ᵒᶜ` | ocean | ``\rho^{\mathrm{oc}}`` (ocean reference density) |
| ``\mathrm{si}`` | `ˢⁱ` | sea ice | ``h^{\mathrm{si}}`` (sea ice thickness) |
| ``\mathrm{la}`` | `ˡᵃ` | land | ``M^{\mathrm{la}}`` (land water mass per area) |
| ``\mathrm{leaf}`` | `ˡᵉᵃᶠ` | leaf | ``T^{\mathrm{leaf}}`` (leaf temperature), ``\lambda^{\mathrm{leaf}}`` (leaf area index) |
| ``\mathrm{lbl}`` | `ˡᵇˡ` | leaf boundary layer | ``g^{\mathrm{lbl}}`` (leaf boundary-layer conductance) |
| ``\mathrm{cnpy}`` | `ᶜⁿᵖʸ` | canopy | ``M^{\mathrm{cnpy}}`` (canopy water store), ``g^{\mathrm{cnpy}}`` (canopy conductance) |
| ``\mathrm{acan}`` | `ᵃᶜᵃⁿ` | canopy air | ``T^{\mathrm{acan}}`` (canopy-air temperature) |
| ``\mathrm{ucan}`` | `ᵘᶜᵃⁿ` | undercanopy | ``w^{\mathrm{ucan}}`` (undercanopy transfer velocity) |
| ``\mathrm{grnd}`` | `ᵍʳⁿᵈ` | ground surface (beneath the canopy, or bare soil) | ``T^{\mathrm{grnd}}`` (ground temperature), ``r^{\mathrm{grnd}}`` (ground-surface resistance) |
| ``\mathrm{soi}`` | `ˢᵒⁱ` | soil | ``r^{\mathrm{soi}}`` (moist-soil surface resistance) |
| ``\mathrm{dbr}`` | `ᵈᵇʳ` | dry-layer branch of the ground vapor exchange | ``f^{\mathrm{dbr}}`` (dry-branch weight), ``g^{\mathrm{dbr}}`` (dry-branch conductance) |
| ``\mathrm{wet}`` | `ʷᵉᵗ` | wet (intercepted) canopy water | ``J^{\mathrm{wet}}`` (wet-canopy evaporation) |
| ``\mathrm{int}`` | `ⁱⁿᵗ` | interception | ``f^{\mathrm{int}}`` (caught fraction of rain) |
| ``\mathrm{thru}`` | `ᵗʰʳᵘ` | throughfall | ``P^{\mathrm{thru}}`` (rain reaching the soil beneath the canopy) |
| ``\mathrm{drip}`` | `ᵈʳⁱᵖ` | drip | ``J^{\mathrm{drip}}`` (drainage from a full canopy store) |
| ``\mathrm{dfct}`` | `ᵈᶠᶜᵗ` | deficit | ``J^{\mathrm{dfct}}`` (evaporation the canopy store could not supply) |
| ``\mathrm{pond}`` | `ᵖᵒⁿᵈ` | surface pond | ``M^{\mathrm{pond}}`` (ponded surface water), ``\tau^{\mathrm{pond}}`` (pond drainage time scale) |
| ``\mathrm{over}`` | `ᵒᵛᵉʳ` | overland flow | ``R^{\mathrm{over}}`` (surface runoff) |
| ``\mathrm{tra}`` | `ᵗʳᵃ` | transpiration | ``J^{\mathrm{tra}}`` (transpiration flux) |
| ``\mathrm{rad}`` | `ʳᵃᵈ` | radiating (effective) | ``T^{\mathrm{rad}}`` (radiating temperature), ``\alpha^{\mathrm{rad}}`` (effective albedo) |
| ``\mathrm{fc}``, ``\mathrm{wp}`` | `ᶠᶜ`, `ʷᵖ` | field capacity, wilting point | ``𝒮^{\mathrm{fc}}``, ``𝒮^{\mathrm{wp}}`` (plant-stress endpoint saturations) |
| ``\mathrm{ao}`` | `ᵃᵒ` | atmosphere–ocean interface | ``\mathcal{Q}^{\mathrm{ao}}`` (atm–ocean heat flux) |
| ``\mathrm{ai}`` | `ᵃⁱ` | atmosphere–ice interface | ``\mathcal{Q}^{\mathrm{ai}}`` (atm–ice heat flux) |
| ``\mathrm{io}`` | `ⁱᵒ` | ice–ocean interface | ``\mathcal{Q}^{\mathrm{io}}`` (ice–ocean heat flux) |
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
| ``t`` | `ₜ` | transmitted | ``\mathscr{I}_{t}^{\mathrm{sw}}`` (transmitted shortwave) |
| ``a`` | `ₐ` | absorbed | ``\mathscr{I}_{a}^{\mathrm{lw}}`` (absorbed longwave) |
| ``\star`` | `★` | similarity theory scale | ``u_\star`` (friction velocity) |
| ``b`` | `b` | boundary value, or background | ``T_b`` (interface temperature), ``u_b`` (background bottom velocity) |

### Time levels

| Notation | Code | Meaning | Example |
|:--------:|:----:|:--------|:--------|
| ``\Delta t`` | `Δt` | time step (s) | |
| ``\cdot^n``, ``\cdot^{n+1}`` | `ⁿ`, `ⁿ⁺¹` | value at the start and end of a time step | ``M^{\mathrm{cnpy}\,n+1}`` (`Mᶜⁿᵖʸⁿ⁺¹`) |
| ``\tilde{\cdot}`` | `M̃` (`\tilde`) | provisional value before a constraint (e.g. clamping) is applied | ``\tilde{M}^{\mathrm{cnpy}}`` (`M̃ᶜⁿᵖʸ`) |

## Atmosphere state variables

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``T`` | `T` | temperature | Air temperature (K) |
| ``p`` | `p` | pressure | Air pressure (Pa) |
| ``q`` | `q` | specific humidity | Mass mixing ratio of water vapor (kg kg⁻¹) |
| ``u`` | `u` | zonal velocity | Eastward wind component (m s⁻¹) |
| ``v`` | `v` | meridional velocity | Northward wind component (m s⁻¹) |
| ``\mathscr{I}_\downarrow^{\mathrm{sw}}`` | `ℐꜜˢʷ` | downwelling shortwave | Downwelling shortwave radiation (W m⁻², negative) |
| ``\mathscr{I}_\downarrow^{\mathrm{lw}}`` | `ℐꜜˡʷ` | downwelling longwave | Downwelling longwave radiation (W m⁻², negative) |
| ``h_{b\ell}`` | `h_bℓ` | boundary layer height | Atmospheric boundary layer height (m) |
| ``pᵛ⁺`` | ``pᵛ⁺`` | saturation vapor pressure | Vapor pressure at saturation (Pa) |
| ``fᵛ`` | `fᵛ` | vapor pressure enhancement factor | Ratio of the saturation vapor pressure in moist air to that of pure water vapor (Gill 1982) (–) |
| ``qᵛ⁺`` | `qᵛ⁺` | saturation specific humidity | Specific humidity at saturation, ``q^{v+}(T)`` (kg kg⁻¹) |

## Land state variables and parameters

Bare symbols below are the land model's internal names; in cross-component
context they take the `ˡᵃ` superscript (`Tˡᵃ`, `Mˡᵃ`, `Mˡᵃ⁺`) per the
component-superscript rule above.

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``T`` | `temperature` | ground temperature | Prognostic land-column temperature (K) |
| ``M`` | `water_storage` | land water | Prognostic land water mass per area (kg m⁻²) |
| ``M^{+}`` | `maximum_water_storage` | maximum land water | Bucket capacity; soil-science "field capacity" (kg m⁻²) |
| ``𝒮`` | `saturation` | surface saturation | Continuous land surface saturation ``\mathrm{clamp}(M/M⁺, 0, 1)``; the interface humidity models derive their availability ``β`` from it (–) |
| ``𝒮ᶜ`` | `critical_saturation` | critical saturation | Saturation above which the surface evaporates at full efficiency, for `CriticalSaturation` (–) |
| ``𝒮ᶜ`` | `dry_layer_onset_saturation` | dry-layer onset saturation | Saturation below which a dry surface layer forms, for `StorageBasedDryLayerDepth`; shares the symbol ``𝒮ᶜ`` with `critical_saturation` above (–) |
| ``T^{\mathrm{grnd}}`` | `ground` | ground temperature | Temperature of the ground surface that exchanges heat with the bulk slab, over bare soil or beneath a canopy (K) |
| ``Λ^{\mathrm{grnd}}`` | `ground_heat_flux` | ground conductance | Ground↔bulk conduction coefficient ``Λ^{\mathrm{grnd}} = κ^T/ℓ^T`` of a `SoilConductiveFlux` (W m⁻² K⁻¹) |
| ``-\Pi^{fc}`` | `field_capacity_head` | field-capacity suction head | Suction (negative pressure head) at which transpiration becomes unstressed (``β = 1``), for `PlantAvailableWaterStress` (m) |
| ``-\Pi^{wp}`` | `wilting_point_head` | wilting-point suction head | Suction (negative pressure head) at which transpiration ceases (``β = 0``), for `PlantAvailableWaterStress` (m) |
| ``f`` | `cover_fraction` | vegetated cover fraction | Vegetated fraction of a cell: the area weight of the vegetated tile of a `TiledLandInterface`, and the cover an `InterceptingHydrology` store occupies (–) |
| ``w^{\mathrm{ucan}}`` | `undercanopy_transfer_velocity` | undercanopy transfer velocity | Ground↔canopy-air transfer velocity of a `CanopyInterface`: the flux per unit difference across the path, ``J = \rho^{\mathrm{at}} w^{\mathrm{ucan}} (q^{\mathrm{grnd}} - q^{\mathrm{acan}})``, as ``u_\star \Xi_q`` is for the canopy-air↔atmosphere exchange. A bulk transfer coefficient times a wind speed, not a diffusivity (m s⁻¹) |
| ``T^{\mathrm{deep}}`` | `deep_temperature` | deep climatological temperature | Prescribed deep/climatological target temperature for force-restore (K) |
| ``τ^{\mathrm{deep}}`` | `deep_time_scale` | deep-restore time scale | Time scale of surface relaxation toward ``T^{\mathrm{deep}}`` (s) |
| ``c^{\mathrm{dry}}`` | `dry_heat_capacity` | dry areal heat capacity | Areal heat capacity of the water-free slab; a `Number` or an `AbstractField` (J m⁻² K⁻¹) |
| ``c^l`` | `liquid_heat_capacity` | liquid heat capacity | Specific heat capacity of slab liquid water (J kg⁻¹ K⁻¹) |
| ``c^{\mathrm{la}}`` | – | land areal heat capacity | Water-dependent areal heat capacity ``c^{\mathrm{la}} = c^{\mathrm{dry}} + c^l M`` (J m⁻² K⁻¹) |
| ``d`` | `surface_thickness` | surface thickness | Thickness of the dry surface layer through which soil vapor diffuses, for `SkinHumidity` (m) |
| ``κ^q`` | `vapor_diffusivity` | soil vapor diffusivity | Vapor mass diffusivity in the surface soil layer, for `SkinHumidity` (kg m⁻¹ s⁻¹) |
| ``\chi^{\mathrm{sand}}`` | `sand` | soil sand fraction | Mass fraction of sand grains in the mineral (non-organic) solid matrix (kg kg⁻¹)
| ``\chi^{\mathrm{silt}}`` | `silt` | soil silt fraction | Mass fraction of silt grains in the mineral (non-organic) solid matrix (kg kg⁻¹)
| ``\chi^{\mathrm{clay}}`` | `clay` | soil clay fraction | Mass fraction of clay grains in the mineral (non-organic) solid matrix (kg kg⁻¹)
| ``\chi^{\mathrm{soc}}`` | `SOC` | soil organic carbon concentration | Mass fraction of organic carbon in the solid matrix (kg kg⁻¹)
| ``\rho^{\mathrm{soil}}`` | `ρ_soil` | soil bulk dry density | Bulk dry density of the soil within each vertical layer (kg m⁻³)
| ``\rho^{\mathrm{soc}}`` | `ρ_soc` | soil organic carbon density | Bulk density of organic material within each vertical layer (kg m⁻³)

### Variably-saturated slab land

Symbols introduced by [`VariablySaturatedHydrology`](@ref),
[`WaterCoupledEnergy`](@ref), and [`DryLayerHumidity`](@ref).

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``h^{\mathrm{la}}`` | `slab_depth` | depth of prognostic land | Vertical thickness of the integrated land slab, from ``z_b`` to ``z_s`` (m) |
| ``\nu`` | `porosity` | soil porosity | Total pore fraction (–) |
| ``\theta^l`` | – | pore liquid fraction | Physical liquid-filled pore fraction; surface physics consumes this (–) |
| ``\vartheta^l`` | – | augmented liquid fraction | Conservative storage variable ``= \theta^l + \max(\Pi, 0)/h^{\mathrm{ss}}``; allows ``M > M⁺`` saturated overflow (–) |
| ``\theta^r`` | `residual_liquid_fraction` | residual liquid fraction | Minimum liquid-filled pore fraction (–) |
| ``𝒮`` | `saturation` | effective saturation | Effective (relative) saturation ``𝒮 = \mathrm{clamp}\!\left((\theta^l - \theta^r)/(\nu - \theta^r),\, 0,\, 1\right)``; the humidity availability and the front depth ``\delta^v`` derive from it (–) |
| ``h^{\mathrm{ss}}`` | `storage_height` | storage height | Saturated storage height — the head built per unit fractional over-saturation; reciprocal of the specific storage (``1/S_s``) (m) |
| ``\Pi`` | – | soil pressure head | Matric/pressure head; ``\Pi \le 0`` unsaturated, ``\Pi > 0`` saturated overflow (m) |
| ``\Pi^d`` | `deep_pressure_head` | deep pressure head | Pressure head of the deep reservoir below the slab, passed to the deep-flux closure (m) |
| ``h`` | – | hydraulic head | ``h = z + \Pi`` (m) |
| ``K`` | – | hydraulic conductivity | Darcy conductivity (m s⁻¹) |
| ``K_0`` | `matching_point_conductivity` | matching-point conductivity | Conductivity the Mualem–van Genuchten curve reaches at ``𝒮 = 1``; a *matrix* value when it comes from a pedotransfer function, excluding macropore flow (m s⁻¹) |
| ``K^{+}`` | `saturated_conductivity` | saturated hydraulic conductivity | Macropore-inclusive Darcy conductivity of saturated soil, for an infiltration cap (m s⁻¹) |
| ``\alpha^{\mathrm{ae}}`` | `inverse_air_entry_head` | inverse air-entry head | Reciprocal of the air-entry pressure head, the suction at which the largest pores begin to empty (m⁻¹) |
| ``\mathscr{n}`` | `pore_size_uniformity` | pore-size uniformity | van Genuchten retention exponent; large ``\mathscr{n}`` drains over a narrow band of suctions, ``\mathscr{n} \to 1`` over orders of magnitude. Sets ``\mathscr{m} = 1 - 1/\mathscr{n}`` (–) |
| ``\eta^K`` | `ηᴷ`, `pore_connectivity_exponent` | pore-connectivity exponent | Mualem exponent on saturation in ``K(\mathcal S)`` (–) |
| ``T_0`` | `reference_temperature` | viscosity reference temperature | Temperature at which ``K_0`` holds and the viscosity factor ``\Theta(T)`` on ``K`` is unity (K) |
| ``J^{Es}`` | `surface_energy_flux` | surface energy flux | Signed surface energy flux, positive upward (out of the slab) (W m⁻²) |
| ``J^{lb}`` | `deep_liquid_water_flux` | deep-boundary liquid water flux | Liquid water mass flux across the slab bottom, positive upward (into the slab, capillary rise / groundwater return); drainage is ``J^{lb} < 0`` (kg m⁻² s⁻¹) |
| ``P^l`` | `Pˡ`, `liquid_water_precipitation_flux` | liquid precipitation | Rain reaching the land surface, positive into the land (kg m⁻² s⁻¹) |
| ``J^{ls}`` | `surface_liquid_water_flux` | surface liquid water flux | Liquid water mass flux at the surface, ``J^{ls} = -P^l + R^{\mathrm{over}}`` without a surface water store, positive upward (out of the slab); infiltration is ``J^{ls} < 0`` (kg m⁻² s⁻¹) |
| ``R^{\mathrm{over}}`` | `surface_runoff` | surface runoff | Liquid leaving the column at the surface, ``\ge 0``: the infiltration excess, or the share of the surface water store drained over the step (kg m⁻² s⁻¹) |
| ``M^{\mathrm{pond}}`` | `surface_water_storage` | surface water store | Infiltration excess ponded at the surface, draining on `drainage_time_scale` (kg m⁻²) |
| ``\tau^{\mathrm{pond}}`` | `τᵖᵒⁿᵈ`, `drainage_time_scale` | pond drainage time scale | Linear-reservoir time scale of the surface water store, ``dM^{\mathrm{pond}}/dt = P^l - I - M^{\mathrm{pond}}/\tau^{\mathrm{pond}}`` (s) |
| ``I``, ``\bar{I}`` | `I`, `Ī` | infiltration | Liquid water entering the slab at its surface, and its step mean, positive into the slab (kg m⁻² s⁻¹) |
| ``I^{\mathrm{cap}}``, ``I^{\mathrm{dry}}`` | `Iᶜᵃᵖ`, `Iᵈʳʸ` | infiltration capacity, unponded infiltration | Capacity limited by the remaining pore volume, ``\min(I_0, (M^+ - M)/\Delta t)``, and the infiltration of unponded ground, ``\min(P^l, I^{\mathrm{cap}})`` (kg m⁻² s⁻¹) |
| ``J^{\mathrm{pond}}`` | `Jᵖᵒⁿᵈ` | net pond inflow | ``P^l - I`` (kg m⁻² s⁻¹) |
| ``M^e`` | `Mᵉ` | equilibrium pond | Pond the linear reservoir relaxes toward, ``\tau^{\mathrm{pond}} J^{\mathrm{pond}}`` (kg m⁻²) |
| ``t^0`` | `t⁰` | pond emptying time | Time within a step at which the surface water store reaches zero (s) |
| ``R^{\mathrm{lat}}`` | `subsurface_runoff` | subsurface runoff | Lateral storage export, ``\ge 0`` (kg m⁻² s⁻¹) |
| ``\kappa^T`` | `thermal_conductivity` | thermal conductivity | Effective ground thermal conductivity (W m⁻¹ K⁻¹) |
| ``\Lambda^{\mathrm{deep}}`` | `deep_conductance` | deep energy conductance | Force-restore deep energy conductance (W m⁻² K⁻¹); see also ``τ^{\mathrm{deep}}`` |
| ``T_r`` | `reference_temperature` | reference temperature | Reference temperature for internal energy ``e^l(T) = c^l (T - T_r)`` (K) |
| ``T^{\mathrm{in}}`` | `Tⁱⁿ` | interface temperature | Atmosphere-facing skin temperature, ``T^{\mathrm{in}}`` (K) |
| ``q^{\mathrm{in}}`` | `qⁱⁿ` | interface specific humidity | Atmosphere-facing skin humidity, set by the humidity model (`β·qᵛ⁺` for `FractionalHumidity`, a vapor-flux balance for `SkinHumidity`) (kg kg⁻¹) |
| ``T^e`` | – | dry-layer temperature | Diagnostic temperature at the dry layer (K) |
| ``q^e`` | – | dry-layer specific humidity | Vapor source humidity at the dry layer (kg kg⁻¹) |
| ``\delta^v`` | `dry_layer_depth` | dry-layer depth | Dry-layer thickness through which vapor diffuses, diagnostic of ``𝒮`` (m) |
| ``\chi`` | – | blend coefficient | ``\chi = \mathrm{clamp}(\delta^v/\ell^T, 0, 1)``; weights ``T^e`` between ``T^{\mathrm{in}}`` and ``T^{\mathrm{la}}`` (–) |
| ``\eta`` | `dry_layer_exponent` | front-depth exponent | Exponent in ``\delta^v = \delta^v_{max}[1 - \min(𝒮/𝒮^c, 1)]^\eta`` (–) |
| ``\ell^T`` | `thermal_exchange_depth`, `exchange_depth` | thermal exchange depth | Depth over which ``\Lambda^{\mathrm{in}} = \kappa^T/\ell^T`` couples ``T^{\mathrm{la}}`` to ``T^{\mathrm{in}}`` (m) |
| ``D^v`` | `molecular_diffusivity` | vapor diffusivity in air | Molecular vapor diffusivity in air (m² s⁻¹) |
| ``w^d`` | – | dry-layer piston velocity | ``w^d = D^v_{eff}/\max(\delta^v, \delta^v_{min})`` (m s⁻¹) |

### Canopy and canopy air space

Symbols introduced by [`CanopyConductanceHumidity`](@ref), [`CanopyInterface`](@ref),
[`InterceptingHydrology`](@ref), and [`TiledLandInterface`](@ref).

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``\lambda^{\mathrm{stem}}`` | `stem_area_index`, `λˢᵗᵉᵐ` | stem area index | Leafless woody area per unit ground area, added to ``\lambda^{\mathrm{leaf}}`` in the undercanopy shielding (m² m⁻²) |
| ``K`` | `extinction` | extinction coefficient | Beer-Lambert extinction of the canopy shortwave split; not the hydraulic conductivity ``K`` (–) |
| ``\Omega`` | `clumping` | clumping index | Foliage clumping index, the ratio of the effective to the true leaf area index in the Beer-Lambert transmittance ``e^{-K \Omega \lambda^{\mathrm{leaf}}}``: ``\Omega = 1`` for randomly distributed foliage, ``\Omega < 1`` for foliage gathered into crowns and shoots, which transmits more light through gaps than a random canopy of the same ``\lambda^{\mathrm{leaf}}`` (–) |
| ``T^{\mathrm{leaf}}`` | – | leaf temperature | Diagnostic temperature of the massless leaf, whose net radiation balances ``\mathcal{Q}^{T\,\mathrm{leaf}} + \mathcal{Q}^{v\,\mathrm{leaf}}`` (K) |
| ``T^{\mathrm{acan}}``, ``q^{\mathrm{acan}}`` | – | canopy-air node | Temperature (K) and specific humidity (kg kg⁻¹) of the canopy air the atmosphere exchanges with |
| ``T^{\mathrm{rad}}`` | `effective_temperature` | effective radiating temperature | Blackbody-equivalent temperature defined by the total upwelling longwave flux above the canopy, ``\sigma (T^{\mathrm{rad}})^4 = \mathscr{I}_\uparrow^{\mathrm{lw}}``. This flux includes thermal emission and reflected downwelling longwave, so ``T^{\mathrm{rad}}`` need not equal ground temperature even over bare soil (K) |
| ``\mathcal{Q}^{\mathrm{grnd}}`` | `ground_heat_flux` | ground heat flux | Ground→bulk conduction ``\mathcal{Q}^{\mathrm{grnd}} = \Lambda^{\mathrm{grnd}}(T^{\mathrm{grnd}} - T^{\mathrm{la}})``, positive down (W m⁻²) |
| ``\mathcal{Q}^{T\,\mathrm{leaf}}``, ``\mathcal{Q}^{T\,\mathrm{grnd}}`` | `𝒬ᵀˡᵉᵃᶠ`, `𝒬ᵀᵍʳⁿᵈ` | per-source sensible heat | Leaf and ground sensible heat into the canopy air, positive up (W m⁻²) |
| ``\mathcal{Q}^{v\,\mathrm{leaf}}``, ``\mathcal{Q}^{v\,\mathrm{grnd}}`` | `𝒬ᵛˡᵉᵃᶠ`, `𝒬ᵛᵍʳⁿᵈ` | per-source latent heat | Leaf and ground latent heat into the canopy air, positive up (W m⁻²) |
| ``g_s`` | – | stomatal conductance | Leaf stomatal conductance from a [`JarvisConductance`](@ref) (mol m⁻² s⁻¹) |
| ``g^{\mathrm{cnpy}}`` | – | canopy conductance | Bulk stomatal conductance ``g^{\mathrm{cnpy}} = \lambda^{\mathrm{leaf}} M^d g_s``, mass units (kg m⁻² s⁻¹). The molar vapor flux ``g_s Δx`` across a mole-fraction difference ``Δx ≈ (M^d/M^w) Δq`` carries mass ``M^d g_s Δq`` |
| ``M^d`` | `molmass_dryair` | dry-air molar mass | From the atmosphere's thermodynamic parameters (kg mol⁻¹) |
| ``w^{\mathrm{lbl}}`` | `leaf_boundary_layer_velocity` | leaf boundary-layer velocity | Per-leaf-area transfer velocity through the leaf boundary layer (m s⁻¹) |
| ``g^{\mathrm{lbl}}`` | – | leaf boundary-layer mass conductance | ``g^{\mathrm{lbl}} = \rho^{\mathrm{at}} \lambda^{\mathrm{leaf}} w^{\mathrm{lbl}}``; the stomata-free wet-leaf vapor path (kg m⁻² s⁻¹) |
| ``g^{\mathrm{leaf}\,T}``, ``g^{\mathrm{leaf}\,v}`` | – | leaf conductances | Leaf sensible-heat and vapor conductance to the canopy air (W m⁻² K⁻¹, kg m⁻² s⁻¹) |
| ``g^{\mathrm{grnd}\,T}``, ``g^{\mathrm{grnd}\,v}`` | – | ground conductances | Ground sensible-heat and vapor conductance to the canopy air; ``g^{\mathrm{grnd}\,v}`` blends the dry-layer branch with the saturated-skin branch ``g^{\mathrm{grnd}\,v+}`` (W m⁻² K⁻¹, kg m⁻² s⁻¹) |
| ``g^{\mathrm{dbr}}`` | – | ground-surface vapor conductance | Undercanopy vapor path through the ground-surface resistance alone, in series with ``G^e`` on the dry-layer branch (kg m⁻² s⁻¹) |
| ``g^{aT}``, ``g^{av}`` | – | aerodynamic conductances | Canopy-air↔atmosphere sensible-heat and vapor conductance of the similarity solution, ``g^{av} = \rho^{\mathrm{at}} u_\star \Xi_q``; [`DryLayerHumidity`](@ref) writes ``g^{av}`` as ``G^a``, the divider every humidity formulation closes against (W m⁻² K⁻¹, kg m⁻² s⁻¹) |
| ``G^e`` | – | dry-layer vapor conductance | Soil vapor conductance of [`DryLayerHumidity`](@ref) (kg m⁻² s⁻¹) |
| ``f^{\mathrm{dbr}}`` | – | dry-branch weight | Weight of the dry-layer branch in the blended soil vapor exchange (–) |
| ``\Sigma g^T``, ``\Sigma g^v`` | – | node conductance sums | Sums over the three branches feeding the canopy-air node, per side |
| ``h^{\mathrm{acan}}`` | `layer_depth` | canopy air layer depth | Depth of the canopy air layer of a [`PrognosticCanopyAir`](@ref) storage (m) |
| ``C^{T}``, ``C^{v}`` | – | canopy-air heat and vapor capacities | ``\rho c^{pm} h^{\mathrm{acan}}`` (J m⁻² K⁻¹) and ``\rho h^{\mathrm{acan}}`` (kg m⁻²) |
| ``\epsilon^{\mathrm{leaf}}``, ``\epsilon^{\mathrm{grnd}}`` | `maximum_canopy_emissivity`, `ground_emissivity` | emissivities | Longwave emissivity of the canopy (``\epsilon^{\mathrm{leaf}} = \epsilon^{max}(1 - e^{-\lambda^{\mathrm{leaf}}})``) and of the ground (–) |
| ``\alpha^{\mathrm{leaf}}``, ``\alpha^{\mathrm{grnd}}`` | `leaf_albedo`, `ground_albedo` | albedos | Shortwave albedo of the canopy and of the ground below it (–) |
| ``r^{\mathrm{soi}}`` | – | ground surface resistance | Moist-soil evaporation resistance of an [`ExponentialSoilResistance`](@ref) (s m⁻¹) |
| ``r^{\mathrm{grnd}}``, ``C^{\mathrm{grnd}}`` | `transfer_coefficient` | ground-surface resistance, transfer coefficient | Bulk ground evaporation resistance of a [`GroundSurfaceResistance`](@ref) (s m⁻¹) and its transfer coefficient (–) |
| ``M^{\mathrm{cnpy}}``, ``M^{\mathrm{cnpy}+}`` | `canopy_water_storage` | canopy water store | Intercepted water on the foliage and its capacity ``M^{\mathrm{cnpy}+} = c \lambda^{\mathrm{leaf}}``, both per unit vegetated area, with ``c`` the capacity per unit leaf area (`capacity_per_leaf_area`) (kg m⁻²) |
| ``f^{\mathrm{int}}`` | – | interception fraction | Caught fraction of rain, ``1 - e^{-K \Omega \lambda^{\mathrm{leaf}}}`` (–) |
| ``P^{\mathrm{int}}``, ``P^{\mathrm{thru}}`` | `Pⁱⁿᵗ`, `throughfall` | intercepted rain, throughfall | Rain caught by the canopy, ``f^{\mathrm{int}} P^l``, and the liquid water reaching the soil beneath it (kg m⁻² s⁻¹) |
| ``J^{\mathrm{drip}}``, ``J^{\mathrm{dfct}}`` | `Jᵈʳⁱᵖ`, `Jᵈᶠᶜᵗ` | canopy drip, evaporation deficit | Water draining from a full canopy store, and wet-canopy evaporation the store could not supply, over one step (kg m⁻² s⁻¹) |
| ``f^{\mathrm{wet}}`` | – | wet-canopy fraction | Deardorff wetted-leaf fraction ``(M^{\mathrm{cnpy}}/M^{\mathrm{cnpy}+})^{2/3}`` (–) |
| ``g^{\mathrm{wet}}`` | – | wet-canopy vapor conductance | Wetted fraction of the leaf boundary-layer conductance, capped so that ``J^{\mathrm{wet}} \le M^{\mathrm{cnpy}}/\Delta t`` (kg m⁻² s⁻¹) |
| ``J^{\mathrm{wet}}``, ``\mathcal{Q}^{v\,\mathrm{wet}}`` | – | wet-canopy evaporation | Evaporation of intercepted water (kg m⁻² s⁻¹) and its latent heat (W m⁻²) |
| ``\Delta U`` | – | surface wind speed | Wind speed relative to the surface at the surface-layer reference height (m s⁻¹) |
| ``R^v`` | – | vapor gas constant | Specific gas constant of water vapor (J kg⁻¹ K⁻¹) |

## Area fractions

An area fraction ``\aleph`` carries the superscript of the component that covers the area,
and is always a fraction of the *enclosing* surface, not of the whole cell. The ocean
fraction partitions a cell into ocean and land, and the sea ice concentration further
partitions the ocean, so the cell is split into open ocean ``\aleph^{\mathrm{oc}} (1 - \aleph^{\mathrm{si}})``,
sea ice ``\aleph^{\mathrm{oc}} \aleph^{\mathrm{si}}``, and land ``1 - \aleph^{\mathrm{oc}}``.
Following the component-superscript rule, a component's own code uses the bare ``\aleph``:
ClimaSeaIce's `ice_concentration` is set with `set!(sea_ice.model, ℵ = ...)`.

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``\aleph^{\mathrm{oc}}`` | `ℵᵒᶜ` | ocean fraction | Fraction of a cell covered by ocean (`ocean_fraction`); the land fraction is ``1 - \aleph^{\mathrm{oc}}`` (–) |
| ``\aleph^{\mathrm{si}}`` | `ℵˢⁱ` | sea ice concentration | Fraction of the ocean covered by sea ice (–) |

## Ocean state variables

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``T`` | `T` | temperature | Ocean potential temperature (ᵒC or K) |
| ``S`` | `S` | salinity | Practical salinity (g kg⁻¹) |
| ``u`` | `u` | zonal velocity | Eastward ocean velocity (m s⁻¹) |
| ``v`` | `v` | meridional velocity | Northward ocean velocity (m s⁻¹) |
| ``u_b`` | `ub` | background bottom velocity | Unresolved velocity, mostly tidal, added in quadrature to the resolved speed in the bottom drag (m s⁻¹) |
| ``\rho^{\mathrm{oc}}`` | `ρᵒᶜ` | reference density | Ocean reference density (kg m⁻³) |
| ``c^{\mathrm{oc}}`` | `cᵒᶜ` | heat capacity | Ocean heat capacity (J kg⁻¹ K⁻¹) |

## Sea ice state variables

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``h^{\mathrm{si}}`` | `hˢⁱ` | ice thickness | Sea ice thickness (m) |
| ``S^{\mathrm{si}}`` | `Sˢⁱ` | ice salinity | Sea ice bulk salinity (g kg⁻¹) |

## Radiation properties

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``\sigma`` | `σ` | Stefan–Boltzmann constant | (W m⁻² K⁻⁴) |
| ``\alpha`` | `α` | albedo | Surface reflectivity (–) |
| ``\alpha^{\mathrm{rad}}`` | `effective_albedo` | column albedo | Broadband shortwave albedo of the canopy + ground column (–) |
| ``\epsilon`` | `ϵ` | emissivity | Surface emissivity (–) |

## Vegetation properties

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``\lambda^{\mathrm{leaf}}`` | `leaf_area_index`, `λˡᵉᵃᶠ` | leaf area index | One-sided leaf area per unit ground area (m² m⁻²) |

## Similarity theory / surface layer

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``u_\star`` | `u★` | friction velocity | Surface friction velocity (m s⁻¹) |
| ``\theta_\star`` | `θ★` | temperature scale | Flux characteristic temperature (K) |
| ``q_\star`` | `q★` | humidity scale | Flux characteristic specific humidity (kg kg⁻¹) |
| ``\Xi_u``, ``\Xi_\theta``, ``\Xi_q`` | `Ξu`, `Ξθ`, `Ξq` | flux-scale transfer coefficients | Relate the flux scales to the air–surface differences, ``u_\star = \Xi_u ΔU``, ``\theta_\star = \Xi_\theta Δθ``, ``q_\star = \Xi_q Δq``. Similarity theory gives ``\Xi = ϰ / (\log(Δh^d/ℓ) - ψ)``; bulk coefficients give ``\Xi_u = \sqrt{C^D}``, ``\Xi_\theta = C^H/\sqrt{C^D}``, ``\Xi_q = C^q/\sqrt{C^D}`` (–) |
| ``b_\star`` | `b★` | buoyancy scale | Flux characteristic buoyancy (m s⁻²) |
| ``L_\star`` | `L★` | Obukhov length | Monin–Obukhov length scale (m) |
| ``C^D`` | `Cᴰ` | drag coefficient | Bulk transfer coefficient for momentum (–) |
| ``\psi`` | `ψ` | stability function | Integrated stability correction (–) |
| ``\Psi`` | `Ψ` | interface state | Aggregate interface state (an `AbstractInterfaceState`) carried through the similarity-theory fixed-point solver `compute_interface_state` |
| ``\Psi^{\mathrm{in}}`` | `Ψⁱⁿ` | iterated interface state | The `AbstractInterfaceState` updated at each iteration of `compute_interface_state` |
| ``\Psi^{\mathrm{at}}``, ``\Psi^{\mathrm{la}}`` | `Ψᵃᵗ`, `Ψˡᵃ` | atmosphere, land state | Pointwise `NamedTuple`s of atmosphere and land quantities read by the interface solver; ``\Psi^{\mathrm{la}}`` holds the keys of `land_state_accessors`, e.g. ``(T, 𝒮)`` |
| ``\zeta`` | `ζ` | stability parameter | ``z / L_\star`` (–) |
| ``\Delta h^d`` | `Δhᵈ` | displaced profile height | Height ``\Delta h - d`` above the zero-plane displacement at which the similarity profiles are evaluated (m) |
| ``\ell`` | `ℓ` | roughness length | Aerodynamic roughness length (m) |
| ``\ell^\mathrm{m}`` | `ℓᵐ` | momentum roughness length | Aerodynamic momentum roughness length (m) |
| ``\ell^\mathrm{s}`` | `ℓˢ` | scalar roughness length | Aerodynamic scalar roughness length (m) |
| ``\varkappa`` | `ϰ` | von Kármán constant | ``\approx 0.4`` (–) |

Note the case distinction: lowercase ``\psi`` (`ψ`) is the stability
function, while capital ``\Psi`` (`Ψ`) is the aggregate interface-state object.

## Surface morphometry

Roughness elements — buildings or plant canopy — that set the aerodynamic
parameters ``\ell^\mathrm{m}`` and ``d`` of a land surface.

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``h`` | `h` | element height | Mean height of the roughness elements (m) |
| ``h^\mathrm{max}`` | `hᵐᵃˣ` | maximum element height | Tallest roughness element in the cell (m) |
| ``\sigma^h`` | `σʰ` | height standard deviation | Spread of the roughness-element heights (m) |
| ``\lambda^p h`` | `gross_building_height` | gross building height | Mean element height over *all* cells, built or not, ``\lambda^p h`` — the digital-surface lift (m) |
| ``d`` | `d` | zero-plane displacement | Displacement height of the roughness sublayer (m) |
| ``d^h`` | `dʰ` | displacement ratio | Packing displacement ``d/h`` of the obstacle-array fit, before the height-spread correction (–) |
| ``\lambda^p`` | `λᵖ` | plan-area index | Ground-plan area covered by the elements, per unit ground area (–) |
| ``\lambda^f`` | `λᶠ` | frontal-area index | Windward face area of the elements, per unit ground area (–) |
| ``\lambda^{p\,\mathrm{min}}`` | `λᵖᵐⁱⁿ` | plan-area index floor | Below this ``\lambda^p`` a cell reduces to bare soil (–) |
| ``\ell^\mathrm{soil}`` | `ℓˢᵒⁱˡ` | bare-soil roughness length | Momentum roughness length where the plan-area index vanishes (m) |
| ``A`` | `A` | array constant | Packing constant of the Macdonald displacement fit (–) |
| ``\beta`` | `β` | drag correction factor | Lumped drag correction of the Macdonald drag partition (–) |
| ``X`` | `X` | displacement parameter | ``(\sigma^h + h)/h^\mathrm{max} \in [0, 1]``, argument of the Kanda displacement fit (–) |
| ``Y`` | `Y` | height-spread parameter | ``\lambda^p \sigma^h / h``, argument of the Kanda roughness rescaling (–) |

The superscripts on ``\lambda^p`` and ``\lambda^f`` distinguish them from the
longitude ``\lambda``.

Two symbols here are shared with other land parameters and mean something
different. ``\beta`` is the drag correction of the Macdonald drag partition,
not the moisture availability ``\beta`` of the hydrology and interface-humidity
models; it keeps the symbol of its source parametrization. Likewise ``d`` is the
zero-plane displacement, not the `surface_thickness` ``d`` of
[`SkinHumidity`](@ref).

## Radiative fluxes

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``\mathscr{I}_\downarrow^{\mathrm{sw}}`` | `ℐꜜˢʷ` | downwelling shortwave | Downwelling shortwave radiation (W m⁻², negative) |
| ``\mathscr{I}_\downarrow^{\mathrm{lw}}`` | `ℐꜜˡʷ` | downwelling longwave | Downwelling longwave radiation (W m⁻², negative) |
| ``\mathscr{I}_\uparrow^{\mathrm{lw}}`` | `ℐꜛˡʷ` | upwelling longwave | Emitted longwave radiation (W m⁻²) |
| ``\mathscr{I}_\uparrow^{\mathrm{sw}}`` | `ℐꜛˢʷ` | upwelling shortwave | Reflected shortwave radiation (W m⁻²) |

| ``\mathscr{I}_{t}^{\mathrm{sw}}`` | `ℐₜˢʷ` | transmitted shortwave | Shortwave passing through the surface, ``(1-\alpha) \mathscr{I}_\downarrow^{\mathrm{sw}}`` (W m⁻²) |
| ``\mathscr{I}_{a}^{\mathrm{lw}}`` | `ℐₐˡʷ` | absorbed longwave | Longwave absorbed at the surface, ``\epsilon \mathscr{I}_\downarrow^{\mathrm{lw}}`` (W m⁻²) |

Radiative fluxes use ``\mathscr{I}`` (`ℐ`, for "intensity") with a modifier
arrow (`ꜜ`/`ꜛ` for downwelling/upwelling) and superscript band (`ˢʷ`/`ˡʷ`).
Derived radiative quantities use a subscript process label (`ₜ`, `ₐ`) with a
superscript band.

## Turbulent interface fluxes

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``\mathcal{Q}^v`` | `𝒬ᵛ` | latent heat flux | Turbulent latent heat flux (W m⁻²) |
| ``\mathcal{Q}^T`` | `𝒬ᵀ` | sensible heat flux | Turbulent sensible heat flux (W m⁻²) |
| ``J^v`` | `Jᵛ` | water vapor flux | Turbulent mass flux of water vapor (kg m⁻² s⁻¹) |
| ``J^{\mathrm{tra}}`` | `Jᵗʳᵃ` | transpiration flux | Mass flux of water vapor transpired by the canopy, part of ``J^v`` (kg m⁻² s⁻¹) |
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

## Net surface freshwater fluxes

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``J^{\mathrm{rn}}`` | `Jʳⁿ` | rain freshwater flux | Rain mass flux at the surface (kg m⁻² s⁻¹) |
| ``J^{\mathrm{sn}}`` | `Jˢⁿ` | snow freshwater flux | Snow mass flux at the surface (kg m⁻² s⁻¹) |
| ``J^{\mathrm{ib}}`` | `Jⁱᵇ` | iceberg freshwater flux | Iceberg calving mass flux into the ocean (kg m⁻² s⁻¹) |

## Thermodynamic properties

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``\mathcal{L}^\ell`` | `ℒˡ` | latent heat of vaporization | Liquid-phase latent heat (J kg⁻¹) |
| ``\mathcal{L}^i`` | `ℒⁱ` | latent heat of sublimation | Ice-phase latent heat (J kg⁻¹) |
| ``\mathcal{L}^f`` | `ℒᶠ` | latent heat of fusion | Heat the ocean supplies to melt snowfall and icebergs (J kg⁻¹) |
| ``c^{pm}`` | `cᵖᵐ` | moist air heat capacity | Moist isobaric specific heat (J kg⁻¹ K⁻¹) |
| ``c^{pd}`` | `cᵖᵈ` | dry air heat capacity | Dry-air isobaric specific heat (J kg⁻¹ K⁻¹) |
| ``\rho^{\mathrm{at}}`` | `ρᵃᵗ` | air density | Atmospheric air density (kg m⁻³) |
| ``\varepsilon^{\mathrm{dv}}`` | ``εᵈᵛ`` | vapor / dry-air gas-constant ratio | ``εᵈᵛ = R_v / R_d`` (so ``(εᵈᵛ)^{-1} = R_d / R_v ≈ 0.622`` is the conventional ε in ``q = ε e / p``) (–) |

## CF standard name mapping

The following table maps code variable names to their
[CF standard names](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html)
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
| `ℵˢⁱ` | `sea_ice_area_fraction` |

## Typing Unicode symbols in Julia

Most symbols can be entered in the Julia REPL and in editors with Julia support by typing a LaTeX-like abbreviation followed by `<tab>`. The table below collects the less obvious completions used in this notation.

| Symbol | Tab completion | Description |
|:------:|:---------------|:------------|
| `𝒬` | `\scrQ` | Script Q (heat flux) |
| `ℐ` | `\scrI` | Script I (radiative intensity) |
| `ℒ` | `\scrL` | Script L (latent heat) |
| `𝓃` | `\scrn` | Script n (pore-size uniformity) |
| `𝓂` | `\scrm` | Script m (van Genuchten `1 - 1/𝓃`) |
| `τ` | `\tau` | Tau (kinematic stress) |
| `ρ` | `\rho` | Rho (density) |
| `σ` | `\sigma` | Sigma (Stefan–Boltzmann constant) |
| `α` | `\alpha` | Alpha (albedo) |
| `ϵ` | `\epsilon` | Epsilon (emissivity) |
| `ℵ` | `\aleph` | Aleph (area fraction) |
| `ϰ` | `\varkappa` | Varkappa (von Kármán constant) |
| `Ψ` | `\Psi` | Capital Psi (interface state) |
| `★` | `\bigstar` | Star (similarity-theory scale) |
| `ꜜ` | `\^downarrow` | Modifier down arrow (downwelling) |
| `ꜛ` | `\^uparrow` | Modifier up arrow (upwelling) |
| `ᵛ` | `\^v` | Superscript v |
| `ᵀ` | `\^T` | Superscript T |
| `ˢ` | `\^s` | Superscript s |
| `ʷ` | `\^w` | Superscript w |
| `ʰ` | `\^h` | Superscript h |
| `ⁱ` | `\^i` | Superscript i |
| `ˡ` | `\^l` | Superscript l |
| `ᵖ` | `\^p` | Superscript p |
| `ᵐ` | `\^m` | Superscript m |
| `ᵈ` | `\^d` | Superscript d |
| `ᴰ` | `\^D` | Superscript D |
| `ˣ` | `\^x` | Superscript x |
| `ʸ` | `\^y` | Superscript y |
| `ᵃ` | `\^a` | Superscript a |
| `ᵗ` | `\^t` | Superscript t |
| `ᵒ` | `\^o` | Superscript o |
| `ᶜ` | `\^c` | Superscript c |
| `ⁿ` | `\^n` | Superscript n |
| `ᶠ` | `\^f` | Superscript f |
| `ʳ` | `\^r` | Superscript r |
| `ᶻ` | `\^z` | Superscript z |
| `ₜ` | `\_t` | Subscript t (transmitted) |
| `ₐ` | `\_a` | Subscript a (absorbed) |
| `ₚ` | `\_p` | Subscript p (penetrating) |
