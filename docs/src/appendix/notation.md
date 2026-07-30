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
| ``\mathrm{oc}`` | `ᵒᶜ` | ocean | ``\rho^{\mathrm{oc}}`` (ocean reference density) |
| ``\mathrm{si}`` | `ˢⁱ` | sea ice | ``h^{\mathrm{si}}`` (sea ice thickness) |
| ``\mathrm{la}`` | `ˡᵃ` | land | ``M^{\mathrm{la}}`` (land water mass per area) |
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
| ``h_{b\ell}`` | `h_bℓ` | boundary layer height | Atmospheric boundary layer height (m) |
| ``pᵛ⁺`` | ``pᵛ⁺`` | saturation vapor pressure | Vapor pressure at saturation (Pa) |
| ``qᵛ⁺`` | `qᵛ⁺` | saturation specific humidity | Specific humidity at saturation, ``q^{v+}(T)`` (kg kg⁻¹) |
| ``qˢ`` | `q` | surface specific humidity | Specific humidity at the interface; set by the humidity model (`β·qᵛ⁺` for `FractionalHumidity`, a vapor-flux balance for `SkinHumidity`) (kg kg⁻¹) |

## Land state variables and parameters

Bare symbols below are the land model's internal names; in cross-component
context they take the `ˡᵃ` superscript (`Tˡᵃ`, `Mˡᵃ`, `Mˡᵃ⁺`) per the
component-superscript rule above.

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``T`` | `temperature` | ground temperature | Prognostic land-column temperature (K) |
| ``M`` | `water_storage` | land water | Prognostic land water mass per area (kg m⁻²) |
| ``M^{+}`` | `maximum_water_storage` | maximum land water | Bucket capacity after [Manabe (1969)](@cite manabe1969climate); soil-science "field capacity" (kg m⁻²) |
| ``𝒮`` | `saturation` | surface saturation | Continuous land surface saturation ``\mathrm{clamp}(M/M⁺, 0, 1)``; the interface humidity models derive their availability ``β`` from it (–) |
| ``𝒮ᶜ`` | `critical_saturation` | critical saturation | Saturation above which the surface evaporates at full efficiency, for `CriticalSaturation` — the critical wetness of [Manabe (1969)](@cite manabe1969climate) (–) |
| ``𝒮ᶜ`` | `dry_layer_onset_saturation` | dry-layer onset saturation | Saturation below which a dry surface layer forms, for `StorageBasedDryLayerDepth`; shares the symbol ``𝒮ᶜ`` with `critical_saturation` above (–) |
| ``T^{\mathrm{deep}}`` | `deep_temperature` | deep climatological temperature | Prescribed deep/climatological target temperature for force-restore (K) |
| ``τ^{\mathrm{deep}}`` | `deep_time_scale` | deep-restore time scale | Time scale of surface relaxation toward ``T^{\mathrm{deep}}`` (s) |
| ``\delta^s`` | `surface_thickness` | surface thickness | Thickness of the dry surface layer through which soil vapor diffuses, for `SkinHumidity` (m); the prescribed sibling of the diagnostic ``\delta^v`` below |
| ``κ^q`` | `vapor_diffusivity` | soil vapor diffusivity | Vapor mass diffusivity in the surface soil layer, for `SkinHumidity` (kg m⁻¹ s⁻¹) |
| ``\chi^{\mathrm{sand}}`` | `sand` | soil sand fraction | Mass fraction of sand grains in the mineral (non-organic) solid matrix (kg kg⁻¹)
| ``\chi^{\mathrm{silt}}`` | `silt` | soil silt fraction | Mass fraction of silt grains in the mineral (non-organic) solid matrix (kg kg⁻¹)
| ``\chi^{\mathrm{clay}}`` | `clay` | soil clay fraction | Mass fraction of clay grains in the mineral (non-organic) solid matrix (kg kg⁻¹)
| ``\chi^{\mathrm{soc}}`` | `SOC` | soil organic carbon concentration | Mass fraction of organic carbon in the solid matrix (kg kg⁻¹)
| ``\rho^{\mathrm{soil}}`` | `ρ_soil` | soil bulk dry density | Bulk dry density of the soil within each vertical layer (kg m⁻³)
| ``\rho^{\mathrm{soc}}`` | `ρ_soc` | soil organic carbon density | Bulk density of organic material within each vertical layer (kg m⁻³)

### Variably-saturated slab land

Symbols introduced by [`VariablySaturatedHydrology`](@ref),
[`WaterCoupledEnergy`](@ref), and [`DryLayerHumidity`](@ref). The retention curve ``\Pi(𝒮)`` and
conductivity ``K(𝒮)`` follow [van Genuchten (1980)](@cite vangenuchten1980) with the
[Mualem (1976)](@cite mualem1976new) pore-bundle model; the dry-layer symbols (``\delta^v``,
``T^e``, ``q^e``, ``w^d``) follow the dry surface layer of
[Ye and Pielke (1993)](@cite yepielke1993) and [Swenson and Lawrence (2014)](@cite swenson2014dry).

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
| ``J^{Es}`` | `surface_energy_flux` | surface energy flux | Signed surface energy flux, positive upward (out of the slab) (W m⁻²) |
| ``J^{lb}`` | `deep_liquid_flux` | deep-boundary liquid flux | Liquid mass flux across the slab bottom, positive upward (into the slab, capillary rise / groundwater return); drainage is ``J^{lb} < 0`` (kg m⁻² s⁻¹) |
| ``J^{ls}`` | `surface_liquid_flux` | surface liquid flux | Liquid mass flux at the surface ``J^{ls} = -P^l + R^{\mathrm{sfc}}``, positive upward (out of the slab); infiltration is ``J^{ls} < 0`` (kg m⁻² s⁻¹) |
| ``R^{\mathrm{sfc}}`` | `surface_runoff` | surface runoff | Liquid input rejected at the surface, ``\ge 0`` (kg m⁻² s⁻¹) |
| ``R^{\mathrm{lat}}`` | `subsurface_runoff` | subsurface runoff | Lateral storage export, ``\ge 0`` (kg m⁻² s⁻¹) |
| ``\kappa^T`` | `thermal_conductivity` | thermal conductivity | Effective ground thermal conductivity (W m⁻¹ K⁻¹) |
| ``\Lambda^{\mathrm{deep}}`` | `deep_conductance` | deep energy conductance | Force-restore deep energy conductance (W m⁻² K⁻¹); see also ``τ^{\mathrm{deep}}`` |
| ``T_r`` | `reference_temperature` | reference temperature | Reference temperature for internal energy ``e^l(T) = c^l (T - T_r)`` (K) |
| ``T^{\mathrm{in}}`` | – | interface temperature | Atmosphere-facing skin temperature, ``T^{\mathrm{in}}`` (K) |
| ``q^{\mathrm{in}}`` | – | interface specific humidity | Atmosphere-facing skin humidity, ``q^{\mathrm{in}}`` (kg kg⁻¹) |
| ``T^e`` | – | dry-layer temperature | Diagnostic temperature at the dry layer (K) |
| ``q^e`` | – | dry-layer specific humidity | Vapor source humidity at the dry layer (kg kg⁻¹) |
| ``\delta^v`` | `dry_layer_depth` | dry-layer depth | Dry-layer thickness through which vapor diffuses, diagnostic of ``𝒮`` (m) |
| ``\chi`` | – | blend coefficient | ``\chi = \mathrm{clamp}(\delta^v/\ell^T, 0, 1)``; weights ``T^e`` between ``T^{\mathrm{in}}`` and ``T^{\mathrm{la}}`` (–) |
| ``\eta`` | `dry_layer_exponent` | front-depth exponent | Exponent in ``\delta^v = \delta^v_{max}[1 - \min(𝒮/𝒮^c, 1)]^\eta`` (–) |
| ``\ell^T`` | `thermal_exchange_depth`, `exchange_depth` | thermal exchange depth | Depth over which ``\Lambda^{\mathrm{in}} = \kappa^T/\ell^T`` couples ``T^{\mathrm{la}}`` to ``T^{\mathrm{in}}`` (m) |
| ``D^v`` | `molecular_diffusivity` | vapor diffusivity in air | Molecular vapor diffusivity in air (m² s⁻¹) |
| ``w^d`` | – | dry-layer piston velocity | ``w^d = D^v_{eff}/\max(\delta^v, \delta^v_{min})`` (m s⁻¹) |

### Canopy aerodynamic roughness

Symbols introduced by [`DragPartitionRoughness`](@ref), the drag-partition roughness sublayer of
[Raupach (1994)](@cite raupach1994simplified) as parameterized for land-cover classes by
[Jasinski et al. (2005)](@cite jasinski2005bulk) and recalibrated against satellite retrievals by
[Borak et al. (2025)](@cite borak2025global), whose equation numbers the closure's docstrings
quote. The momentum roughness length keeps
the surface-layer symbol ``\ell^\mathrm{m}`` (see [Similarity theory / surface layer](@ref) and
`interior_properties.momentum_roughness_length`, which consumes it). The displacement height is
plain ``d``, the boundary-layer convention, with no element-type superscript: whether the
roughness elements are leaves or roofs, ``d`` is the same quantity and only the closure that
computes it changes.

Area indices take the script ``𝒜``, superscripted when more than one is in play
(``𝒜^{\mathrm{stem}}``, ``𝒜^{\mathrm{plant}}``); bare ``𝒜`` is the leaf area index while it is the
only one. [Raupach's (1994)](@cite raupach1994simplified) own symbol ``\Lambda`` is unavailable — it is `deep_conductance`
(``\Lambda^{\mathrm{deep}}``) above — as are ``\lambda`` (longitude) and ``L`` (one letter from the
Obukhov length ``L_\star``). The critical index ``𝒜^c`` follows ``𝒮^c``: the value beyond which the
behavior changes.

Two of the source's symbols are avoided: `displacement_coefficient` keeps its verbose field name
because ``\alpha`` is albedo here, and the drag ratio is written out as ``C^R/C^S`` because
``\beta`` is moisture availability.

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``\ell^\mathrm{m}`` | `momentum_roughness_length` | momentum roughness length | Canopy aerodynamic roughness length from the drag partition (m) |
| ``d`` | `displacement_height` | zero-plane displacement | Height of the logarithmic-profile origin above ground; the profile is ``\log[(z - d)/\ell^\mathrm{m}]`` (m) |
| ``h`` | `canopy_height` | canopy height | Measured or class-representative canopy top height (m) |
| ``𝒜`` | `leaf_area_index` | leaf area index | One-sided leaf area per unit ground area (m² m⁻²); the closure's input in place of Raupach's canopy area index (–) |
| ``𝒜^c`` | `critical_leaf_area_index` | critical (skimming) index | Index above which the wind ratio saturates ([Borak et al. 2025](@cite borak2025global), Table 2); caps ``\gamma`` only, not ``d`` or ``\ell^\mathrm{m}`` (–) |
| – | `maximum_valid_leaf_area_index` | data-quality ceiling | A larger index is treated as fill/artifact and gapped; not physics (–) |
| ``\gamma`` | – | wind ratio | ``\gamma \equiv U_h/u_\star``, the drag partition between vegetation form drag and substrate friction (–) |
| ``C^R`` | `form_drag_coefficient` | form drag coefficient | Vegetation element form drag coefficient (–) |
| ``C^S`` | `substrate_drag_coefficient` | substrate drag coefficient | Ground-surface friction drag coefficient (–) |
| ``(u_\star/U_h)_{\mathrm{max}}`` | `maximum_friction_ratio` | friction-ratio cap | Ceiling on the inverse wind ratio, flooring ``\gamma`` (–) |
| ``c`` | `sublayer_decay_coefficient` | sublayer decay coefficient | Wind-profile decay in the roughness sublayer; closure-local, distinct from the heat capacities ``c^{pm}`` (–) |
| – | `displacement_coefficient` | displacement coefficient | Coefficient of the ``1/(\gamma\sqrt{𝒜})`` correction in ``d/h`` (–) |
| – | `sublayer_influence` | roughness-sublayer influence | Constant 0.193 offsetting the log profile within the roughness sublayer ([Raupach 1995](@cite raupach1995corrigenda)); distinct from the stability function ``\psi`` (–) |

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

Monin–Obukhov surface-layer symbols. The default roughness lengths and stability functions
follow [Edson et al. (2013)](@cite edson2013exchange).

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``u_\star`` | `u★` | friction velocity | Surface friction velocity (m s⁻¹) |
| ``\theta_\star`` | `θ★` | temperature scale | Flux characteristic temperature (K) |
| ``q_\star`` | `q★` | humidity scale | Flux characteristic specific humidity (kg kg⁻¹) |
| ``b_\star`` | `b★` | buoyancy scale | Flux characteristic buoyancy (m s⁻²) |
| ``L_\star`` | `L★` | Obukhov length | Monin–Obukhov length scale (m) |
| ``C^D`` | `Cᴰ` | drag coefficient | Bulk transfer coefficient for momentum (–) |
| ``\psi`` | `ψ` | stability function | Integrated stability correction (–) |
| ``\Psi`` | `Ψ` | interface state | Aggregate interface state (an `AbstractInterfaceState`) carried through the similarity-theory fixed-point solver `compute_interface_state` |
| ``\zeta`` | `ζ` | stability parameter | ``z / L_\star`` (–) |
| ``\ell`` | `ℓ` | roughness length | Aerodynamic roughness length (m) |
| ``\ell^\mathrm{m}`` | `ℓᵐ` | momentum roughness length | Aerodynamic momentum roughness length (m) |
| ``\ell^\mathrm{s}`` | `ℓˢ` | scalar roughness length | Aerodynamic scalar roughness length (m) |
| ``\varkappa`` | `ϰ` | von Kármán constant | ``\approx 0.4`` (–) |

Note the case distinction: lowercase ``\psi`` (`ψ`) is the stability
function, while capital ``\Psi`` (`Ψ`) is the aggregate interface-state object.

## Radiative fluxes

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``\mathscr{I}_\downarrow^{\mathrm{sw}}`` | `ℐꜜˢʷ` | downwelling shortwave | Downwelling shortwave radiation (W m⁻²) |
| ``\mathscr{I}_\downarrow^{\mathrm{lw}}`` | `ℐꜜˡʷ` | downwelling longwave | Downwelling longwave radiation (W m⁻²) |
| ``\mathscr{I}_\uparrow^{\mathrm{lw}}`` | `ℐꜛˡʷ` | upwelling longwave | Emitted longwave radiation (W m⁻²) |

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

## Thermodynamic properties

| Math | Code | Property | Description |
|:----:|:----:|:---------|:------------|
| ``\mathcal{L}^\ell`` | `ℒˡ` | latent heat of vaporization | Liquid-phase latent heat (J kg⁻¹) |
| ``\mathcal{L}^i`` | `ℒⁱ` | latent heat of sublimation | Ice-phase latent heat (J kg⁻¹) |
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
| `ℵ` | `sea_ice_area_fraction` |

## Typing Unicode symbols in Julia

Most symbols can be entered in the Julia REPL and in editors with Julia support by typing a LaTeX-like abbreviation followed by `<tab>`. The table below collects the less obvious completions used in this notation.

| Symbol | Tab completion | Description |
|:------:|:---------------|:------------|
| `𝒬` | `\scrQ` | Script Q (heat flux) |
| `ℐ` | `\scrI` | Script I (radiative intensity) |
| `ℒ` | `\scrL` | Script L (latent heat) |
| `𝒜` | `\scrA` | Script A (area index) |
| `ℓ` | `\ell` | Script ell (roughness length) |
| `τ` | `\tau` | Tau (kinematic stress) |
| `ρ` | `\rho` | Rho (density) |
| `σ` | `\sigma` | Sigma (Stefan–Boltzmann constant) |
| `α` | `\alpha` | Alpha (albedo) |
| `ϵ` | `\epsilon` | Epsilon (emissivity) |
| `ℵ` | `\aleph` | Aleph (ice concentration) |
| `ϰ` | `\varkappa` | Varkappa (von Kármán constant) |
| `Ψ` | `\Psi` | Capital Psi (interface state) |
| `★` | `\bigstar` | Star (similarity-theory scale) |
| `ꜜ` | `\^downarrow` | Modifier down arrow (downwelling) |
| `ꜛ` | `\^uparrow` | Modifier up arrow (upwelling) |
| `ᵛ` | `\^v` | Superscript v |
| `ᵀ` | `\^T` | Superscript T |
| `ˢ` | `\^s` | Superscript s |
| `ʷ` | `\^w` | Superscript w |
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
