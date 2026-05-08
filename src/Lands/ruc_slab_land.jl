#####
##### A 2D slab land-surface component for `EarthSystemModel`.
#####
##### Implements the slab-compatible subset of the RUC LSM
##### (Smirnova et al. 1997, 2016) plus complementary parameterizations
##### from ISBA, Noilhan-Planton (1989), and Mahfouf-Noilhan (1991):
#####
##### Snow:
#####   - density compaction (Anderson 1976; Smirnova 1997)
#####   - new-snow accumulation + density (Smirnova 1997, 2016)
#####   - snow-cover fraction (Koren 1999; Niu-Yang 2007)
#####   - albedo / emissivity / roughness blends (Robinson-Kukla 1985)
#####   - sublimation sink on SWE
#####   - snow melt with `T_g > 273.15 K` energy partition
#####   - liquid water retention in the pack (overflow → soil infiltration)
#####   - continuous fresh-snow-albedo aging with `T_g`-dependent timescale
##### Canopy:
#####   - canopy water store with Beer-Lambert rain/snow interception
#####   - throughfall partition (drip, intersn) into soil/snowpack
#####   - wet-canopy direct evaporation
#####   - two-source canopy temperature `T_c` (Deardorff 1978)
##### Vegetation/optics:
#####   - vegetation-modified end-members for `α, ε, z₀`
##### Soil:
#####   - single-bucket liquid soil moisture `θ_liq` (Manabe 1969)
#####   - prognostic frozen soil moisture `θ_ice` with phase-change at 273.15 K
#####     (latent heat of fusion absorbed/released into the slab heat budget)
##### Resistances:
#####   - RUC top-layer moisture availability `mavail`
#####   - RUC bare-soil evaporation limiter `soilres`
#####   - bare-soil resistance r_g (Sellers et al. 1992 form)
#####   - Jarvis-Stewart canopy resistance r_s (Jarvis 1976; Stewart 1988)
##### Surface energy balance solver:
#####   - implicit `vilka` Newton iteration for the skin temperature
#####     `T_s + d₁·q_sat(T_s) − d₂ = 0` (Smirnova 1997 §3),
#####     using a Magnus closed-form q_sat in place of the f90's lookup table
#####
##### Journal references:
#####
##### Anderson, E. A., 1976: A point energy and mass balance model of a snow
#####   cover. NOAA Tech. Rep. NWS 19.
##### Manabe, S., 1969: Climate and the ocean circulation: I. The atmospheric
#####   circulation and the hydrology of the Earth's surface. Mon. Wea. Rev.,
#####   97, 739–774.
##### Jarvis, P. G., 1976: The interpretation of the variations in leaf water
#####   potential and stomatal conductance found in canopies in the field.
#####   Phil. Trans. R. Soc. London B, 273, 593–610.
##### Deardorff, J. W., 1978: Efficient prediction of ground surface
#####   temperature and moisture, with inclusion of a layer of vegetation.
#####   J. Geophys. Res., 83, 1889–1903.
##### Buck, A. L., 1981: New equations for computing vapor pressure and
#####   enhancement factor. J. Appl. Meteor., 20, 1527–1532.
##### Robinson, D. A., and G. Kukla, 1985: Maximum surface albedo of seasonally
#####   snow-covered lands in the Northern Hemisphere. J. Climate Appl. Meteor.,
#####   24, 402–411.
##### Stewart, J. B., 1988: Modelling surface conductance of pine forest.
#####   Agric. For. Meteor., 43, 19–35.
##### Noilhan, J., and S. Planton, 1989: A simple parameterization of land
#####   surface processes for meteorological models. Mon. Wea. Rev., 117, 536–549.
##### Mahfouf, J.-F., and J. Noilhan, 1991: Comparative study of various
#####   formulations of evaporation from bare soil using in situ data.
#####   J. Appl. Meteor., 30, 1354–1365.
##### Sellers, P. J., M. D. Heiser, and F. G. Hall, 1992: Relations between
#####   surface conductance and spectral vegetation indices at intermediate
#####   (100 m² to 15 km²) length scales. J. Geophys. Res., 97, 19033–19059.
##### Smirnova, T. G., J. M. Brown, and S. G. Benjamin, 1997: Performance of
#####   different soil model configurations in simulating ground surface
#####   temperature and surface fluxes. Mon. Wea. Rev., 125, 1870–1884.
##### Koren, V., J. Schaake, K. Mitchell, Q.-Y. Duan, F. Chen, and J. M. Baker,
#####   1999: A parameterization of snowpack and frozen ground intended for
#####   NCEP weather and climate models. J. Geophys. Res., 104, 19569–19585.
##### Niu, G.-Y., and Z.-L. Yang, 2007: An observation-based formulation of
#####   snow cover fraction and its evaluation over large North American river
#####   basins. J. Geophys. Res., 112, D21101, doi:10.1029/2007JD008674.
##### Smirnova, T. G., J. M. Brown, S. G. Benjamin, and J. S. Kenyon, 2016:
#####   Modifications to the Rapid Update Cycle Land Surface Model (RUC LSM)
#####   available in the WRF model. Mon. Wea. Rev., 144, 1851–1865,
#####   doi:10.1175/MWR-D-15-0198.1.

using Oceananigans.Fields: ConstantField, ZeroField
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Utils: prettytime, prettysummary

const ρ_water_const = 1000.0   # kg m⁻³, RUC LSM `rhowater` parameter
const L_fusion_const = 3.337e5 # J kg⁻¹, latent heat of fusion of water

"""
    RucSlabLandParameters(FT = Float64; kwargs...)

Tunable scalar parameters of `RucSlabLand`. Defaults reproduce the
slab-compatible RUC LSM configuration of Smirnova et al. (1997, 2016) for
snow, canopy interception, and top-layer moisture availability, while retaining
the existing single-bucket soil-moisture state.

# Heat reservoir (ground)
- `depth = 0.10`           : ground slab thickness `H_g` [m]
- `density = 1500`         : soil bulk density `ρ_g` [kg m⁻³]
- `heat_capacity = 1480`   : soil specific heat `c_g` [J kg⁻¹ K⁻¹]

# Canopy reservoir
- `canopy_heat_capacity = 1.0e4`   : effective canopy areal heat capacity
   `(ρ c H)_c` [J m⁻² K⁻¹]
- `canopy_water_capacity = 5.0e-4` : RUC fixed maximum intercepted canopy
   water `sat` [m]

# Soil-moisture bucket (Manabe 1969 / ISBA Noilhan-Planton 1989)
- `soil_depth = 1.0`             : root-zone bucket depth `H_s` [m]
- `theta_sat = 0.45`             : saturated vol. water content [m³ m⁻³]
- `theta_fc = 0.30`              : field capacity [m³ m⁻³]
- `theta_wilt = 0.10`            : wilting point [m³ m⁻³]
- `theta_air_dry = 0.05`         : air-dry water content [m³ m⁻³]

# Surface optical / aerodynamic constants — end-members blended by `vegfrac`
# and then by `snowfrac`.
- `alb_snow_fresh = 0.85`   : fresh-snow shortwave albedo
- `alb_snow_aged = 0.50`    : fully-aged snow albedo
- `alb_bare = 0.25`         : bare-soil albedo
- `alb_veg = 0.18`          : dense-canopy albedo
- `emiss_snow = 0.98`       : snow emissivity
- `emiss_bare = 0.95`       : bare-soil emissivity
- `emiss_veg = 0.98`        : canopy emissivity
- `z0_snow = 0.011`         : snow roughness length [m]
- `z0_bare = 0.05`          : bare-soil roughness length [m]
- `z0_veg = 0.20`           : vegetated roughness length [m]

# Snow-cover fraction (Koren 1999; Niu-Yang 2007)
- `sncovfac = 0.04`           : Niu-Yang scale factor [m]
- `snowcovr_opt = 2`          : 1 = linear; 2 = blend; 3 = Niu-Yang `tanh`

The Koren critical depths `snhei_crit = 0.01601·ρ_w/ρ_sn` and
`snhei_crit_newsn = 0.0005·ρ_w/ρ_sn`
are computed dynamically from the local snow density inside the kernels.

# Snow density compaction (Anderson 1976; Smirnova 1997)
- `c1_compaction = 0.026`     : compaction parameter `c1sn`
- `c2_compaction = 21.0`      : compaction parameter `c2sn`
- `rhosn_min = 58.8`          : minimum bulk density [kg m⁻³]
- `rhosn_max = 500.0`         : maximum bulk density [kg m⁻³]

# Snow albedo aging (continuous decay toward `alb_snow_aged`)
- `snow_aging_tau_cold = 2.5e6`   : aging timescale at cold T (`< 273.15 K`)
   [s] (≈ 30 days; Robinson-Kukla 1985 magnitude)
- `snow_aging_tau_warm = 8.6e4`   : aging timescale at warm T [s] (≈ 1 day)

# Snow melt and liquid water retention
- `snow_liquid_capacity_frac = 0.04` : cap used when draining carried-over
   slab liquid water `swl` before new melt is applied
- `meltfactor = 2.0` : RUC open-area Egglston melt limiter multiplier
- `snow_retention_min_frac = 0.08` : RUC lower bound for retained melt fraction
- `snow_retention_max_frac = 0.18` : RUC upper bound for retained melt fraction
- `snow_retention_depth_scale = 0.10` : RUC depth scale for retained melt [m]
- `snow_retention_depth_factor = 0.13` : RUC Koren retained-melt factor

# Phase change
- `latent_heat_fusion = 3.337e5`  : J kg⁻¹

# Stomatal / bare-soil resistance
- `r_smin = 100.0`            : minimum (well-watered, optimal-light) `r_s` [s m⁻¹]
- `r_smax = 5000.0`           : maximum (closed) `r_s` [s m⁻¹]
- `r_gmin = 50.0`             : minimum bare-soil resistance [s m⁻¹]
- `r_gmax = 5000.0`           : maximum bare-soil resistance [s m⁻¹]
- `rg_lim = 100.0`            : SW radiation scaling for `f1` [W m⁻²]
- `vpd_lim = 4.0e-3`          : VPD scaling for `f2` [kg kg⁻¹]
- `T_opt = 298.0`             : optimal stomatal temperature [K]
"""
struct RucSlabLandParameters{FT}
    # Ground slab
    depth :: FT
    density :: FT
    heat_capacity :: FT
    # Canopy slab
    canopy_heat_capacity :: FT
    canopy_water_capacity :: FT
    # Soil-moisture bucket
    soil_depth :: FT
    theta_sat :: FT
    theta_fc :: FT
    theta_wilt :: FT
    theta_air_dry :: FT
    # Optics / roughness end-members
    alb_snow_fresh :: FT
    alb_snow_aged :: FT
    alb_bare :: FT
    alb_veg :: FT
    emiss_snow :: FT
    emiss_bare :: FT
    emiss_veg :: FT
    z0_snow :: FT
    z0_bare :: FT
    z0_veg :: FT
    # Snow-cover fraction
    sncovfac :: FT
    snowcovr_opt :: Int
    # Snow density compaction
    c1_compaction :: FT
    c2_compaction :: FT
    rhosn_min :: FT
    rhosn_max :: FT
    # Snow albedo aging + liquid retention + phase change
    snow_aging_tau_cold :: FT
    snow_aging_tau_warm :: FT
    snow_liquid_capacity_frac :: FT
    meltfactor :: FT
    snow_retention_min_frac :: FT
    snow_retention_max_frac :: FT
    snow_retention_depth_scale :: FT
    snow_retention_depth_factor :: FT
    latent_heat_fusion :: FT
    # Resistances
    r_smin :: FT
    r_smax :: FT
    r_gmin :: FT
    r_gmax :: FT
    rg_lim :: FT
    vpd_lim :: FT
    T_opt :: FT
end

function RucSlabLandParameters(FT::Type = Float64;
                            depth = 0.10,
                            density = 1500,
                            heat_capacity = 1480,
                            canopy_heat_capacity = 1.0e4,
                            canopy_water_capacity = 5.0e-4,
                            soil_depth = 1.0,
                            theta_sat = 0.45,
                            theta_fc = 0.30,
                            theta_wilt = 0.10,
                            theta_air_dry = 0.05,
                            alb_snow_fresh = 0.85,
                            alb_snow_aged = 0.50,
                            alb_bare = 0.25,
                            alb_veg = 0.18,
                            emiss_snow = 0.98,
                            emiss_bare = 0.95,
                            emiss_veg = 0.98,
                            z0_snow = 0.011,
                            z0_bare = 0.05,
                            z0_veg = 0.20,
                            sncovfac = 0.04,
                            snowcovr_opt = 2,
                            c1_compaction = 0.026,
                            c2_compaction = 21.0,
                            rhosn_min = 58.8,
                            rhosn_max = 500.0,
                            snow_aging_tau_cold = 2.5e6,
                            snow_aging_tau_warm = 8.6e4,
                            snow_liquid_capacity_frac = 0.04,
                            meltfactor = 2.0,
                            snow_retention_min_frac = 0.08,
                            snow_retention_max_frac = 0.18,
                            snow_retention_depth_scale = 0.10,
                            snow_retention_depth_factor = 0.13,
                            latent_heat_fusion = 3.337e5,
                            r_smin = 100.0,
                            r_smax = 5000.0,
                            r_gmin = 50.0,
                            r_gmax = 5000.0,
                            rg_lim = 100.0,
                            vpd_lim = 4.0e-3,
                            T_opt = 298.0)

    return RucSlabLandParameters{FT}(
        convert(FT, depth),
        convert(FT, density),
        convert(FT, heat_capacity),
        convert(FT, canopy_heat_capacity),
        convert(FT, canopy_water_capacity),
        convert(FT, soil_depth),
        convert(FT, theta_sat),
        convert(FT, theta_fc),
        convert(FT, theta_wilt),
        convert(FT, theta_air_dry),
        convert(FT, alb_snow_fresh),
        convert(FT, alb_snow_aged),
        convert(FT, alb_bare),
        convert(FT, alb_veg),
        convert(FT, emiss_snow),
        convert(FT, emiss_bare),
        convert(FT, emiss_veg),
        convert(FT, z0_snow),
        convert(FT, z0_bare),
        convert(FT, z0_veg),
        convert(FT, sncovfac),
        Int(snowcovr_opt),
        convert(FT, c1_compaction),
        convert(FT, c2_compaction),
        convert(FT, rhosn_min),
        convert(FT, rhosn_max),
        convert(FT, snow_aging_tau_cold),
        convert(FT, snow_aging_tau_warm),
        convert(FT, snow_liquid_capacity_frac),
        convert(FT, meltfactor),
        convert(FT, snow_retention_min_frac),
        convert(FT, snow_retention_max_frac),
        convert(FT, snow_retention_depth_scale),
        convert(FT, snow_retention_depth_factor),
        convert(FT, latent_heat_fusion),
        convert(FT, r_smin),
        convert(FT, r_smax),
        convert(FT, r_gmin),
        convert(FT, r_gmax),
        convert(FT, rg_lim),
        convert(FT, vpd_lim),
        convert(FT, T_opt),
    )
end

"""
    RucSlabLand(grid; FT = eltype(grid),
                   parameters = RucSlabLandParameters(FT),
                   clock = Clock{FT}(time = 0))

A slab land-surface component for `EarthSystemModel`. See the file header
for the parameterizations and references included.

Coupler-supplied forcings (kept as `Field`s on the grid):

- `temperature_flux`           ≡ `Jᵀ_g` [K m s⁻¹] kinematic flux into the
  ground slab; `T_g -= Jᵀ_g · Δt / depth`.
- `canopy_temperature_flux`    ≡ `Jᵀ_c` [W m⁻²] heat flux into the canopy
  reservoir; `T_c -= Jᵀ_c · Δt / canopy_heat_capacity`. Different
  convention from `Jᵀ_g` because the canopy is parameterised by an areal
  heat capacity `(ρcH)_c` rather than a depth.
- `forcings.snowfall_rate`     [m s⁻¹] solid precipitation, liquid-water equivalent
- `forcings.rainfall_rate`     [m s⁻¹] liquid precipitation
- `forcings.moisture_flux`     [kg m⁻² s⁻¹] vapor mass flux, F_v (positive up).
  Routed to snow sublimation on the snow-covered fraction and to bare-soil
  evaporation (β-weighted) on the snow-free fraction.
- `forcings.canopy_evaporation` [kg m⁻² s⁻¹] direct evaporation drain on
  the canopy water store `cst`.
- `forcings.transpiration`     [kg m⁻² s⁻¹] canopy transpiration (positive up)
- `forcings.solar_irradiance`  [W m⁻²]   for Jarvis `f1`
- `forcings.air_temperature`   [K]       for Jarvis `f4`
- `forcings.air_humidity`      [kg kg⁻¹] for Jarvis `f2`

Optional implicit-skin-T inputs (used only by `surface_balance_vilka!`):

- `forcings.vilka_d1, vilka_d2` [K, K] linearised energy-balance coefficients
- `forcings.surface_pressure`  [hPa]   for q_sat(T_s)

Initialise with `set!(land; T=..., θ=..., snwe=..., …)` after construction.
"""
struct RucSlabLand{FT, G, Clk, T, S, C, V, P, F, PAR}
    grid :: G
    clock :: Clk
    temperature :: T
    temperature_flux :: T
    canopy_temperature :: T
    canopy_temperature_flux :: T
    soil_moisture :: T          # liquid θ_liq
    soil_moisture_ice :: T      # frozen θ_ice
    snow :: S
    canopy :: C
    vegetation :: V
    properties :: P
    forcings :: F
    parameters :: PAR
end

# Inner constructor capturing FT
RucSlabLand{FT}(grid, clock, T, Jᵀ, Tc, JᵀT, θ, θi, snow, canopy, veg, props, forc, par) where FT =
    RucSlabLand{FT, typeof(grid), typeof(clock), typeof(T),
             typeof(snow), typeof(canopy), typeof(veg),
             typeof(props), typeof(forc), typeof(par)}(
        grid, clock, T, Jᵀ, Tc, JᵀT, θ, θi, snow, canopy, veg, props, forc, par)

function RucSlabLand(grid;
                  FT = eltype(grid),
                  parameters = RucSlabLandParameters(FT),
                  clock = Clock{FT}(time = 0))

    # Prognostic temperatures
    temperature              = CenterField(grid)
    temperature_flux         = CenterField(grid)
    canopy_temperature       = CenterField(grid)
    canopy_temperature_flux  = CenterField(grid)

    # Liquid and frozen soil moisture
    soil_moisture     = CenterField(grid); fill!(soil_moisture, parameters.theta_fc)
    soil_moisture_ice = CenterField(grid)   # default 0

    # Snow column
    snwe             = CenterField(grid)
    snhei            = CenterField(grid)
    rhosn            = CenterField(grid); fill!(rhosn,    250)
    rhonewsn         = CenterField(grid); fill!(rhonewsn, 100)
    rhosnfall        = CenterField(grid); fill!(rhosnfall, 100)
    snowfrac         = CenterField(grid)
    newsn            = CenterField(grid)
    snowfallac       = CenterField(grid)
    snowfracnewsn    = CenterField(grid)
    keep_snow_albedo = CenterField(grid)
    swl              = CenterField(grid)              # snow liquid water [m LWE]
    alb_snow_local   = CenterField(grid)              # per-cell aging snow albedo
    fill!(alb_snow_local, parameters.alb_snow_fresh)
    swe_inflow       = CenterField(grid)              # scratch: post-canopy throughfall to pack
    swl_overflow     = CenterField(grid)              # scratch: retention overflow → infiltration

    snow = (; snwe, snhei, rhosn, rhonewsn, rhosnfall,
              snowfrac, newsn, snowfallac, snowfracnewsn, keep_snow_albedo,
              swl, alb_snow_local, swe_inflow, swl_overflow)

    # Canopy water store
    canopy = (cst       = CenterField(grid),
              drip      = CenterField(grid),
              interw    = CenterField(grid),
              intersn   = CenterField(grid),
              infwater  = CenterField(grid),
              intwratio = CenterField(grid))

    # Vegetation inputs and derived diagnostics. Per-cell `albedo_veg`,
    # `emissivity_veg`, `z0_veg`, `r_smin` are filled from the scalar
    # parameters by default and can be overwritten by
    # `apply_land_classifications!` to install a USGS / MODIS lookup table.
    vegetation = (vegfrac         = CenterField(grid),
                  lai             = CenterField(grid),
                  canopy_capacity = CenterField(grid),
                  mavail          = CenterField(grid),
                  soilres         = CenterField(grid),
                  r_s             = CenterField(grid),
                  r_g             = CenterField(grid),
                  albedo_veg      = CenterField(grid),
                  emissivity_veg  = CenterField(grid),
                  z0_veg          = CenterField(grid),
                  r_smin          = CenterField(grid),
                  is_urban        = CenterField(grid))
    fill!(vegetation.r_s,            parameters.r_smin)
    fill!(vegetation.r_g,            parameters.r_gmin)
    fill!(vegetation.soilres,        1)
    fill!(vegetation.albedo_veg,     parameters.alb_veg)
    fill!(vegetation.emissivity_veg, parameters.emiss_veg)
    fill!(vegetation.z0_veg,         parameters.z0_veg)
    fill!(vegetation.r_smin,         parameters.r_smin)

    # Effective surface properties (initialised to bare-soil values)
    alb   = CenterField(grid); fill!(alb,   parameters.alb_bare)
    emiss = CenterField(grid); fill!(emiss, parameters.emiss_bare)
    znt   = CenterField(grid); fill!(znt,   parameters.z0_bare)
    properties = (; alb, emiss, znt)

    forcings = (snowfall_rate      = CenterField(grid),
                rainfall_rate      = CenterField(grid),
                moisture_flux      = CenterField(grid),
                canopy_evaporation = CenterField(grid),
                transpiration      = CenterField(grid),
                solar_irradiance   = CenterField(grid),
                air_temperature    = CenterField(grid),
                air_humidity       = CenterField(grid),
                vilka_d1           = CenterField(grid),
                vilka_d2           = CenterField(grid),
                surface_pressure   = CenterField(grid))

    return RucSlabLand{FT}(grid, clock,
                        temperature, temperature_flux,
                        canopy_temperature, canopy_temperature_flux,
                        soil_moisture, soil_moisture_ice,
                        snow, canopy, vegetation, properties, forcings,
                        parameters)
end

Base.eltype(::RucSlabLand{FT}) where FT = FT

function Oceananigans.set!(land::RucSlabLand;
                           T = nothing,
                           Tc = nothing,
                           θ = nothing,
                           θ_ice = nothing,
                           snwe = nothing,
                           snhei = nothing,
                           rhosn = nothing,
                           swl = nothing,
                           alb_snow_local = nothing,
                           vegfrac = nothing,
                           lai = nothing)
    !isnothing(T)              && set!(land.temperature, T)
    !isnothing(Tc)             && set!(land.canopy_temperature, Tc)
    isnothing(Tc) && !isnothing(T) && set!(land.canopy_temperature, T)
    !isnothing(θ)              && set!(land.soil_moisture, θ)
    !isnothing(θ_ice)          && set!(land.soil_moisture_ice, θ_ice)
    !isnothing(snwe)           && set!(land.snow.snwe,           snwe)
    !isnothing(snhei)          && set!(land.snow.snhei,          snhei)
    !isnothing(rhosn)          && set!(land.snow.rhosn,          rhosn)
    !isnothing(swl)            && set!(land.snow.swl,            swl)
    !isnothing(alb_snow_local) && set!(land.snow.alb_snow_local, alb_snow_local)
    !isnothing(vegfrac)        && set!(land.vegetation.vegfrac,  vegfrac)
    !isnothing(lai)            && set!(land.vegetation.lai,      lai)
    return nothing
end

function Base.summary(land::RucSlabLand{FT}) where FT
    A = nameof(typeof(architecture(land.grid)))
    G = nameof(typeof(land.grid))
    return string("RucSlabLand{$FT, $A, $G}",
                  "(time = ", prettytime(land.clock.time),
                  ", iteration = ", land.clock.iteration, ")")
end

function Base.show(io::IO, land::RucSlabLand)
    p = land.parameters
    print(io, summary(land), '\n',
              "├── grid: ",                summary(land.grid), '\n',
              "├── ground depth: ",        prettysummary(p.depth), " m, ρcH = ",
                  prettysummary(p.density * p.heat_capacity * p.depth), " J m⁻² K⁻¹\n",
              "├── canopy heat cap: ",     prettysummary(p.canopy_heat_capacity), " J m⁻² K⁻¹\n",
              "├── soil bucket depth: ",   prettysummary(p.soil_depth), " m, θ ∈ [",
                  prettysummary(p.theta_wilt), ", ", prettysummary(p.theta_sat), "]\n",
              "├── snow albedo: ",
                  prettysummary(p.alb_snow_fresh), " → ",
                  prettysummary(p.alb_snow_aged), " (fresh→aged)\n",
              "├── albedo end-members: ",
                  prettysummary(p.alb_bare), " / ",
                  prettysummary(p.alb_veg), " (bare / veg)\n",
              "├── roughness end-members: ",
                  prettysummary(p.z0_snow), " / ",
                  prettysummary(p.z0_bare), " / ",
                  prettysummary(p.z0_veg), " m\n",
              "├── snowcovr_opt: ",        p.snowcovr_opt, '\n',
              "├── snow: ",                keys(land.snow), '\n',
              "├── canopy: ",              keys(land.canopy), '\n',
              "└── vegetation: ",          keys(land.vegetation))
end

#####
##### Snow physics — kernels and inline utilities
#####

# Snow-cover fraction (Koren et al. 1999, Niu and Yang 2007). Mirrors
# `compute_snow_fraction` (lines 400–420 of module_sf_ruclsm_surface.f90).
@inline function compute_snow_fraction(opt::Int, snhei, snhei_crit, znt,
                                       rhosn, rhonewsn, sncovfac)
    FT = typeof(snhei)
    if opt == 1
        return min(one(FT), snhei / (FT(2) * snhei_crit))
    elseif opt == 2
        f1 = min(one(FT), snhei / (FT(2) * snhei_crit))
        z₀ = min(FT(0.2), znt)
        f2 = tanh(snhei / (FT(2.5) * z₀ * (rhosn / rhonewsn)))
        return FT(0.5) * (f1 + f2)
    else
        return tanh(snhei / (FT(10) * sncovfac * (rhosn / rhonewsn)))
    end
end

# Snow density compaction following Anderson (1976).
# RUC's `snwe` includes liquid retained in
# the pack, so the Julia split representation uses `snwe + swl` here.
@kernel function _compact_snow!(rhosn, snwe, snhei, swl, T,
                                Δt, c1, c2, ρ_min, ρ_max)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(rhosn)
        ρ  = max(rhosn[i, j, 1], ρ_min)
        h  = snhei[i, j, 1]
        total_swe = snwe[i, j, 1] + swl[i, j, 1]
        if h > FT(0.0081) * FT(1000) / ρ
            T_C = min(zero(FT), T[i, j, 1] - FT(273.15))
            bsn = (FT(Δt) / FT(3600)) * c1 *
                  exp(FT(0.08) * T_C - c2 * ρ * FT(1e-3))
            arg = bsn * total_swe * FT(100)
            if arg ≥ FT(1e-4)
                xsn = ρ * (exp(arg) - one(FT)) / arg
                rhosn[i, j, 1] = clamp(xsn, ρ_min, ρ_max)
            end
        end
        snhei[i, j, 1] = total_swe > zero(FT) ?
                         total_swe * FT(1000) / max(rhosn[i, j, 1], ρ_min) :
                         zero(FT)
    end
end

# Sublimation drain on SWE — only on snow-covered patches and only when
# T_g ≤ 273.15 K (above freezing the same flux is treated as melt). Mirrors
# the snow-mass closure in Smirnova et al. (1997).
@kernel function _apply_sublimation!(snwe, snhei, swl, F_v, snowfrac, rhosn, T,
                                     Δt, ρ_w, ρ_min)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(snwe)
        if snwe[i, j, 1] > zero(FT) && snowfrac[i, j, 1] > zero(FT) &&
           T[i, j, 1] ≤ FT(273.15)
            Δswe = F_v[i, j, 1] * snowfrac[i, j, 1] * FT(Δt) / ρ_w
            snwe[i, j, 1]  = max(zero(FT), snwe[i, j, 1] - Δswe)
            snhei[i, j, 1] = (snwe[i, j, 1] + swl[i, j, 1]) * FT(1000) /
                             max(rhosn[i, j, 1], ρ_min)
        end
    end
end

# Snow melt when `T_g > 273.15 K`. The slab heat surplus is converted to
# potential melt, limited by the RUC Egglston cap for low-density/cold packs,
# then split into retained liquid `swl` and overflow following the RUC
# Koren-style retained-melt fraction. The slab cools by exactly the latent
# heat consumed, so the energy balance is closed locally.
@kernel function _melt_snow!(snwe, snhei, swl, swl_overflow, T,
                             rhosn, newsn, rhonewsn,
                             ρcH_ground, L_f, ρ_w, ρ_min, ρ_max,
                             Δt, meltfactor,
                             rsm_min_frac, rsm_max_frac,
                             rsm_depth_scale, rsm_depth_factor)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(snwe)
        if snwe[i, j, 1] > zero(FT) && T[i, j, 1] > FT(273.15)
            snwepr = snwe[i, j, 1]
            snwepr_total = snwepr + swl[i, j, 1]
            ρ_sn = max(rhosn[i, j, 1], ρ_min)
            Δenergy = ρcH_ground * (T[i, j, 1] - FT(273.15))   # J m⁻²
            Δmass_max = Δenergy / L_f                          # kg m⁻²
            Δswe_max  = Δmass_max / ρ_w                        # m LWE

            smelt = Δswe_max / FT(Δt)
            if (ρ_sn < FT(350) ||
                (newsn[i, j, 1] > zero(FT) && rhonewsn[i, j, 1] < FT(450))) &&
               T[i, j, 1] < FT(283)
                smelt_cap = FT(Δt) / FT(60) * FT(5.6e-8) *
                            meltfactor * max(one(FT), T[i, j, 1] - FT(273.15))
                smelt = min(smelt, smelt_cap)
            end

            Δswe = min(smelt * FT(Δt), snwepr)
            rsmfrac = zero(FT)
            if snhei[i, j, 1] > FT(0.01) && ρ_sn < FT(350)
                rsmfrac = min(rsm_max_frac,
                              max(rsm_min_frac,
                                   snwepr_total / rsm_depth_scale * rsm_depth_factor))
            end

            retained = rsmfrac * Δswe
            overflow = Δswe - retained

            snwe[i, j, 1] = max(zero(FT), snwepr - Δswe)
            swl[i, j, 1] += retained
            swl_overflow[i, j, 1] += overflow
            T[i, j, 1]     -= Δswe * ρ_w * L_f / ρcH_ground

            # RUC folds retained liquid into `snwe`; the split Julia state
            # must still mass-average the full pack and use that total for
            # snow height.
            total_pack_swe = snwe[i, j, 1] + swl[i, j, 1]
            if total_pack_swe > zero(FT)
                xsn = (ρ_sn * snwe[i, j, 1] + ρ_w * swl[i, j, 1]) /
                      total_pack_swe
                rhosn[i, j, 1] = clamp(xsn, ρ_min, ρ_max)
            end

            # A single-T slab has no separate sub-snow soil temperature.
            snhei[i, j, 1] = total_pack_swe * FT(1000) /
                             max(rhosn[i, j, 1], ρ_min)
        end
    end
end

# Drain carried-over liquid water in the slab pack to the soil bucket once it
# exceeds the configured capacity. Current-step RUC melt retention/overflow is
# computed in `_melt_snow!`; this pass only handles liquid already in `swl`.
@kernel function _drain_swl!(swl, snwe, swl_overflow, retention_frac)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(swl)
        cap = retention_frac * snwe[i, j, 1]
        if swl[i, j, 1] > cap
            swl_overflow[i, j, 1] = swl[i, j, 1] - cap
            swl[i, j, 1]          = cap
        else
            swl_overflow[i, j, 1] = zero(FT)
        end
    end
end

# Add new snowfall (post-canopy throughfall in `swe_inflow`) to the pack.
# Fresh-snow density follows the Smirnova et al. (1997, 2016) `tanh`
# formulation, using air temperature `T_air`.
# Bulk density = mass-weighted mean of old and fresh snow. Fresh layers reset
# the local aging albedo toward the fresh-snow value.
@kernel function _accumulate_new_snow!(snwe, snhei, rhosn, swl, rhonewsn, rhosnfall,
                                       newsn, snowfracnewsn, keep_snow_albedo,
                                       snowfallac, alb_snow_local,
                                       swe_inflow, T_air,
                                       Δt,
                                       alb_fresh,
                                       ρ_min, ρ_max)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(snwe)
        Δswe = max(zero(FT), swe_inflow[i, j, 1])      # already × Δt

        if Δswe > zero(FT)
            T_K = T_air[i, j, 1]
            ρ_new = clamp(FT(1000) /
                           max(FT(8), FT(17) * tanh((FT(276.65) - T_K) * FT(0.15))),
                           ρ_min, FT(125))
            rhonewsn[i, j, 1]  = ρ_new
            rhosnfall[i, j, 1] = ρ_new

            old_swe = snwe[i, j, 1] + swl[i, j, 1]
            old_ρ   = max(rhosn[i, j, 1], ρ_min)
            total   = old_swe + Δswe

            xsn = (old_ρ * old_swe + ρ_new * Δswe) / total
            rhosn[i, j, 1] = clamp(xsn, ρ_min, ρ_max)

            snwe[i, j, 1] += Δswe
            snhei[i, j, 1] = (snwe[i, j, 1] + swl[i, j, 1]) * FT(1000) /
                             max(rhosn[i, j, 1], ρ_min)

            new_depth = Δswe * FT(1000) / ρ_new
            newsn[i, j, 1] = new_depth

            # `snhei_crit_newsn` follows `module_sf_ruclsm.F:1419`,
            # `snhei_crit_newsn = 0.0005·ρ_w/ρ_sn`, where `ρ_sn` is the bulk
            # pack density (Fortran captures it pre-step at line 1419 before
            # compaction and the new-snow blend; `old_ρ` is the closest Julia
            # analogue since compaction has already run this step but the
            # blend has not).
            snhei_crit_newsn_dyn = FT(0.0005) * FT(1000) / old_ρ
            snowfallac[i, j, 1]   += new_depth
            snowfracnewsn[i, j, 1] = min(one(FT),
                                         snowfallac[i, j, 1] / snhei_crit_newsn_dyn)

            keep_snow_albedo[i, j, 1] = (snowfracnewsn[i, j, 1] > FT(0.99) &&
                                         ρ_new < FT(450)) ? one(FT) : zero(FT)

            # Reset local aging albedo toward fresh-snow value, mass-weighted.
            old_alb = alb_snow_local[i, j, 1]
            alb_snow_local[i, j, 1] =
                (old_alb * old_swe + alb_fresh * Δswe) / total
        end
    end
end

# Continuous snow-albedo aging — exponential decay toward `alb_aged`, with
# `T_g`-dependent timescale (faster at warm T, slower at cold T). The form
# is consistent with Robinson and Kukla (1985) magnitudes and Verseghy
# (1991) decay-rate ranges.
@kernel function _age_snow_albedo!(alb_snow_local, T, snwe,
                                   alb_aged, τ_cold, τ_warm, Δt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(alb_snow_local)
        if snwe[i, j, 1] > zero(FT)
            τ = T[i, j, 1] > FT(273.15) ? τ_warm : τ_cold
            r = exp(-FT(Δt) / τ)
            alb_snow_local[i, j, 1] =
                alb_aged + (alb_snow_local[i, j, 1] - alb_aged) * r
        end
    end
end

# Recompute snow fraction. Mirrors lines 167 / 259–267 of the f90.
# `snhei_crit` is computed dynamically from `ρ_sn` per cell, following
# `module_sf_ruclsm.F:1418`: `snhei_crit = 0.01601·ρ_w/ρ_sn`. The urban
# clamp `snowfrac ≤ 0.75` mirrors `module_sf_ruclsm.F:1645`.
@kernel function _finalize_snow_cover!(snowfrac, snhei, znt, rhosn, rhonewsn,
                                       is_urban, sncovfac, snowcovr_opt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(snowfrac)
        if snhei[i, j, 1] == zero(FT)
            snowfrac[i, j, 1] = zero(FT)
        else
            ρ_sn  = max(rhosn[i, j, 1],    FT(58.8))
            ρ_new = max(rhonewsn[i, j, 1], FT(58.8))
            snhei_crit_dyn = FT(0.01601) * FT(1000) / ρ_sn
            f = compute_snow_fraction(snowcovr_opt,
                                      snhei[i, j, 1], snhei_crit_dyn,
                                      znt[i, j, 1],
                                      ρ_sn, ρ_new, sncovfac)
            if is_urban[i, j, 1] > FT(0.5)
                f = min(f, FT(0.75))
            end
            snowfrac[i, j, 1] = f
        end
    end
end

#####
##### Canopy water balance — RUC subset D, lines 105–127.
#####

# RUC LSM passes a fixed canopy-water saturation `sat = 5.e-4 m` into
# the interception routine.
@inline canopy_capacity(::FT, ::FT, capacity::FT) where FT = capacity

@kernel function _intercept_precip!(cst, drip, interw, intersn,
                                    infwater, intwratio,
                                    canopy_cap, vegfrac, lai,
                                    rainfall_rate, snowfall_rate,
                                    swe_inflow, Δt, canopy_water_capacity)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(cst)
        vf = vegfrac[i, j, 1]
        L  = lai[i, j, 1]
        sat = canopy_capacity(L, vf, FT(canopy_water_capacity))
        canopy_cap[i, j, 1] = sat

        Δrain = max(zero(FT), rainfall_rate[i, j, 1] * FT(Δt))
        Δsnow = max(zero(FT), snowfall_rate[i, j, 1] * FT(Δt))

        if vf > FT(0.01)
            transmission = one(FT) - exp(-FT(0.5) * L)
            iw = FT(0.25) * Δrain * transmission * vf
            is = FT(0.25) * Δsnow * transmission * vf
            interw[i, j, 1]  = iw
            intersn[i, j, 1] = is

            ratio = (iw + is) > zero(FT) ? iw / (iw + is) : zero(FT)
            intwratio[i, j, 1] = ratio

            cst_new = cst[i, j, 1] + iw + is
            if cst_new > sat
                drip[i, j, 1] = cst_new - sat
                cst[i, j, 1]  = sat
            else
                drip[i, j, 1] = zero(FT)
                cst[i, j, 1]  = cst_new
            end

            d = drip[i, j, 1]
            infwater[i, j, 1]   = max(zero(FT), Δrain - iw) + d * ratio
            swe_inflow[i, j, 1] = max(zero(FT), Δsnow - is) +
                                  d * (one(FT) - ratio)
        else
            cst[i, j, 1]        = zero(FT)
            drip[i, j, 1]       = zero(FT)
            interw[i, j, 1]     = zero(FT)
            intersn[i, j, 1]    = zero(FT)
            intwratio[i, j, 1]  = zero(FT)
            infwater[i, j, 1]   = Δrain
            swe_inflow[i, j, 1] = Δsnow
        end
    end
end

@kernel function _evaporate_canopy!(cst, F_v_canopy, Δt, ρ_w)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(cst)
        if cst[i, j, 1] > zero(FT)
            Δ = max(zero(FT), F_v_canopy[i, j, 1] * FT(Δt) / ρ_w)
            cst[i, j, 1] = max(zero(FT), cst[i, j, 1] - Δ)
        end
    end
end

#####
##### Surface optical / aerodynamic properties
#####

@kernel function _update_surface_properties!(alb, emiss, znt,
                                             snowfrac, keep_snow_albedo,
                                             T, newsn, snhei,
                                             vegfrac, alb_snow_local,
                                             albedo_veg, emissivity_veg, z0_veg_field,
                                             alb_bare,
                                             emiss_snow, emiss_bare,
                                             z0_snow, z0_bare)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT   = eltype(alb)
        f_sn = snowfrac[i, j, 1]
        keep = keep_snow_albedo[i, j, 1]
        h    = snhei[i, j, 1]
        vf   = vegfrac[i, j, 1]
        α_sn = alb_snow_local[i, j, 1]   # locally aged snow albedo

        # Composite snow-free end-members weighted by vegfrac.
        # The `_veg` end-members are per-cell Fields populated either
        # from scalar parameters or from a vegetation lookup table.
        α_veg_ij  = albedo_veg[i, j, 1]
        ε_veg_ij  = emissivity_veg[i, j, 1]
        z₀_veg_ij = z0_veg_field[i, j, 1]
        α_free  = (one(FT) - vf) * alb_bare   + vf * α_veg_ij
        ε_free  = (one(FT) - vf) * emiss_bare + vf * ε_veg_ij
        z₀_free = (one(FT) - vf) * z0_bare    + vf * z₀_veg_ij

        # Roughness: default to the snow-free composite. The RUC f90
        # snow-blending branch (lines 186–194) only applies when the
        # surface is short (z₀_free ≤ 0.2 m, i.e. grass / shrub /
        # cropland) and there is no fresh snow this step; tall canopies
        # (forests with z₀_free > 0.2 m) keep the canopy roughness.
        znt_new = z₀_free
        if h > zero(FT) && newsn[i, j, 1] == zero(FT) && z₀_free ≤ FT(0.2)
            if h ≤ FT(2) * z₀_free
                znt_new = FT(0.55) * z₀_free + FT(0.45) * z0_snow
            elseif h ≤ FT(4) * z₀_free
                znt_new = FT(0.2) * z₀_free + FT(0.8) * z0_snow
            else
                znt_new = z0_snow
            end
        end
        znt[i, j, 1] = znt_new

        # Shortwave-albedo blend (Robinson and Kukla 1985), with the local
        # aged snow albedo replacing the fixed end-member.
        α_blend = max(keep * α_sn,
                      min(α_free + (α_sn - α_free) * f_sn, α_sn))
        if α_blend < FT(0.4) || keep == one(FT)
            alb[i, j, 1] = α_blend
        else
            T_C = T[i, j, 1] - FT(273.15)
            alb[i, j, 1] = min(α_blend,
                               max(α_blend - FT(0.1) * (T_C + FT(10)) /
                                             FT(10) * α_blend,
                                   α_blend - FT(0.05)))
        end

        emiss[i, j, 1] = max(keep * emiss_snow,
                             min(ε_free + (emiss_snow - ε_free) * f_sn,
                                 emiss_snow))
    end
end

#####
##### Soil moisture, freeze/thaw, RUC `mavail`/`soilres`, Jarvis r_s, bare-soil r_g
#####

@inline function ruc_mavail(θ, snowfrac, θ_air_dry, θ_fc)
    FT = typeof(θ)
    ref = max(θ_fc - θ_air_dry, eps(FT))
    residual = max(zero(FT), θ - θ_air_dry)
    return clamp(residual / ref * (one(FT) - snowfrac) + snowfrac,
                 FT(1e-5), one(FT))
end

@inline function ruc_soilres(θ, qa, qg, θ_air_dry, θ_fc)
    FT = typeof(θ)
    fc = max(θ_air_dry, FT(0.5) * θ_fc)
    if θ > fc || qa > qg
        return one(FT)
    else
        fex_fc = clamp(θ / fc, FT(0.01), one(FT))
        return FT(0.25) * (one(FT) - cos(FT(π) * fex_fc))^2
    end
end

# Saturation specific humidity from Magnus / Buck (1981) over water/ice.
@inline function esat_buck(T)
    FT = typeof(T)
    if T > FT(273.15)
        return FT(6.1121) * exp(FT(17.502) * (T - FT(273.15)) / (T - FT(32.18)))
    else
        return FT(6.1115) * exp(FT(22.452) * (T - FT(273.15)) / (T - FT(0.61)))
    end
end

@inline function qsat_buck(T, p_hPa)
    FT = typeof(T)
    e = esat_buck(T)
    ε = FT(0.622)
    return ε * e / (p_hPa - (one(FT) - ε) * e)
end

# Jarvis-Stewart canopy resistance (Jarvis 1976; Stewart 1988).
@inline function jarvis_resistance(rg, qa, Ta, θ, lai,
                                   r_smin, r_smax, rg_lim, vpd_lim, T_opt,
                                   θ_wilt, θ_fc)
    FT = typeof(rg)
    if lai ≤ zero(FT)
        return r_smax
    end
    F1 = (rg / rg_lim) / (one(FT) + rg / rg_lim);  F1 = max(F1, FT(1e-3))
    qsat_air = qsat_buck(Ta, FT(1013.25))
    F2 = one(FT) / (one(FT) + max(zero(FT), (qsat_air - qa)) / vpd_lim)
    F2 = clamp(F2, FT(1e-3), one(FT))
    if θ ≥ θ_fc
        F3 = one(FT)
    elseif θ ≤ θ_wilt
        F3 = FT(1e-3)
    else
        F3 = clamp((θ - θ_wilt) / (θ_fc - θ_wilt), FT(1e-3), one(FT))
    end
    F4 = one(FT) - FT(0.0016) * (T_opt - Ta)^2;  F4 = clamp(F4, FT(1e-3), one(FT))
    return clamp(r_smin / (lai * F1 * F2 * F3 * F4), r_smin, r_smax)
end

# Bare-soil resistance (Sellers et al. 1992 form):
#
#   r_g = exp(8.206 − 4.255 · θ_liq / θ_sat)
#
# Clamped to [r_gmin, r_gmax]. Decreases with wetter soils and saturates
# at the dry end.
@inline function bare_soil_resistance(θ, θ_sat, r_gmin, r_gmax)
    FT = typeof(θ)
    rg = exp(FT(8.206) - FT(4.255) * (θ / θ_sat))
    return clamp(rg, r_gmin, r_gmax)
end

@kernel function _update_mavail_rs_rg!(mavail, soilres, r_s, r_g,
                                       soil_moisture, snowfrac,
                                       T, p_surf, vegfrac, lai,
                                       rg_irr, qa, Ta,
                                       r_smin_field,
                                       θ_wilt, θ_fc, θ_air_dry, θ_sat,
                                       r_smax, r_gmin, r_gmax,
                                       rg_lim, vpd_lim, T_opt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(mavail)
        θ  = soil_moisture[i, j, 1]
        f_sn = snowfrac[i, j, 1]
        m  = ruc_mavail(θ, f_sn, θ_air_dry, θ_fc)
        ps = p_surf[i, j, 1] > one(FT) ? p_surf[i, j, 1] : FT(1013.25)
        qg = qsat_buck(T[i, j, 1], ps) * m
        sr = ruc_soilres(θ, qa[i, j, 1], qg, θ_air_dry, θ_fc)
        vf = vegfrac[i, j, 1]
        L  = lai[i, j, 1]
        mavail[i, j, 1]  = m
        soilres[i, j, 1] = sr
        r_s[i, j, 1]    = jarvis_resistance(rg_irr[i, j, 1],
                                            qa[i, j, 1],
                                            Ta[i, j, 1],
                                            θ, L,
                                            r_smin_field[i, j, 1], r_smax,
                                            rg_lim, vpd_lim,
                                            T_opt, θ_wilt, θ_fc)
        r_g[i, j, 1]    = bare_soil_resistance(θ, θ_sat, r_gmin, r_gmax)
    end
end

# Single-bucket soil moisture update with snow-overflow infiltration.
# `infwater` is m of LWE applied this step (already × Δt). `swl_overflow`
# is m of LWE released by snow retention overflow.
@kernel function _step_soil_moisture!(soil_moisture, infwater, swl_overflow,
                                      F_v_total, transpiration,
                                      vegfrac, soilres, snowfrac,
                                      Δt, ρ_w, soil_depth,
                                      θ_sat, θ_air_dry)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(soil_moisture)
        vf = vegfrac[i, j, 1]
        snow_free = one(FT) - snowfrac[i, j, 1]

        Δθ_in = (infwater[i, j, 1] + swl_overflow[i, j, 1]) / soil_depth

        bare_share = snow_free * (one(FT) - vf) * soilres[i, j, 1]
        E_g  = max(zero(FT), F_v_total[i, j, 1]) * bare_share
        E_t  = max(zero(FT), transpiration[i, j, 1])
        Δθ_out = (E_g + E_t) * FT(Δt) / (ρ_w * soil_depth)

        θ_new = soil_moisture[i, j, 1] + Δθ_in - Δθ_out
        soil_moisture[i, j, 1] = clamp(θ_new, θ_air_dry, θ_sat)
    end
end

# Soil freeze/thaw: split between liquid (`θ_liq`) and frozen (`θ_ice`)
# moisture by the ground-slab temperature, with latent heat of fusion
# absorbed/released by the slab heat budget. Conserves total water
# (`θ_liq + θ_ice`).
@kernel function _freeze_thaw_soil!(θ_liq, θ_ice, T,
                                    ρcH_ground, L_f, ρ_w, soil_depth)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(θ_liq)
        Tg = T[i, j, 1]

        if Tg > FT(273.15) && θ_ice[i, j, 1] > zero(FT)
            # Thaw — convert ice to liquid, cool slab toward 273.15 K
            Δenergy = ρcH_ground * (Tg - FT(273.15))         # J m⁻²
            Δmax    = Δenergy / (L_f * ρ_w * soil_depth)      # vol-fraction
            Δθ      = min(Δmax, θ_ice[i, j, 1])
            θ_ice[i, j, 1] -= Δθ
            θ_liq[i, j, 1] = min(θ_liq[i, j, 1] + Δθ, one(FT))
            T[i, j, 1] -= Δθ * L_f * ρ_w * soil_depth / ρcH_ground

        elseif Tg < FT(273.15) && θ_liq[i, j, 1] > zero(FT)
            # Freeze — convert liquid to ice, warm slab toward 273.15 K
            Δenergy = ρcH_ground * (FT(273.15) - Tg)
            Δmax    = Δenergy / (L_f * ρ_w * soil_depth)
            Δθ      = min(Δmax, θ_liq[i, j, 1])
            θ_liq[i, j, 1] -= Δθ
            θ_ice[i, j, 1] = min(θ_ice[i, j, 1] + Δθ, one(FT))
            T[i, j, 1] += Δθ * L_f * ρ_w * soil_depth / ρcH_ground
        end
    end
end

#####
##### Slab and canopy temperature integrators (flux-driven mode)
#####

@kernel function _step_temperature!(T, Jᵀ, Δt, H)
    i, j = @index(Global, NTuple)
    @inbounds T[i, j, 1] -= Jᵀ[i, j, 1] * Δt / H
end

@kernel function _step_canopy_temperature!(Tc, Jᵀ_c, Δt, H_canopy_eff)
    i, j = @index(Global, NTuple)
    @inbounds Tc[i, j, 1] -= Jᵀ_c[i, j, 1] * Δt / H_canopy_eff
end

#####
##### Implicit skin-T solver (`vilka`) — Smirnova et al. (1997) §3,
##### f90 lines 366–398. Newton iteration on
#####
#####     f(T_s) = T_s + d₁ · q_sat(T_s) − d₂ = 0
#####
##### with `d₁` and `d₂` linearised energy-balance coefficients supplied by
##### the coupler. Uses the closed-form Buck (1981) `q_sat(T, p)` rather
##### than the f90's 5001-entry lookup table.
#####

@inline function _vilka_iteration(tn::FT, d1::FT, d2::FT, p_hPa::FT;
                                  max_iter = 20, tol = FT(1e-3)) where FT
    ts = tn
    for _ in 1:max_iter
        qs  = qsat_buck(ts, p_hPa)
        f   = ts + d1 * qs - d2
        # df/dts ≈ 1 + d1 · dqsat/dts (centred difference)
        h   = FT(0.05)
        dqs = (qsat_buck(ts + h, p_hPa) - qsat_buck(ts - h, p_hPa)) /
              (FT(2) * h)
        df  = one(FT) + d1 * dqs
        Δ   = f / df
        ts -= Δ
        if abs(Δ) < tol
            break
        end
    end
    return ts, qsat_buck(ts, p_hPa)
end

@kernel function _solve_vilka!(T, q_s, d1, d2, p_surf)
    i, j = @index(Global, NTuple)
    @inbounds begin
        FT = eltype(T)
        ts, qs = _vilka_iteration(T[i, j, 1],
                                  d1[i, j, 1], d2[i, j, 1],
                                  max(p_surf[i, j, 1], FT(1)))
        T[i, j, 1]   = ts
        q_s[i, j, 1] = qs
    end
end

"""
    surface_balance_vilka!(land::RucSlabLand, q_s)

Update `land.temperature` (skin temperature) by Newton iteration on the
linearised surface energy balance, using the coupler-supplied
`forcings.{vilka_d1, vilka_d2, surface_pressure}` fields. Writes the
saturation specific humidity at the new skin temperature into the user-
supplied `q_s` field.

This is an alternative to driving `land.temperature` via
`land.temperature_flux` and is intended for couplers that prefer to feed
linearised energy-balance coefficients (Smirnova et al. 1997 §3) instead
of an explicit temperature flux.
"""
function surface_balance_vilka!(land::RucSlabLand, q_s)
    grid = land.grid
    arch = architecture(grid)
    forc = land.forcings
    launch!(arch, grid, :xy, _solve_vilka!,
            land.temperature, q_s,
            forc.vilka_d1, forc.vilka_d2, forc.surface_pressure)
    fill_halo_regions!(land.temperature)
    return nothing
end

#####
##### Time stepping and state update
#####

"""
    update_state!(land::RucSlabLand)

Refresh diagnostic fields (snow fraction, effective α/ε/z₀, RUC `mavail`,
RUC `soilres`, canopy resistance, bare-soil resistance) from the current
prognostic state and forcings.
"""
function Oceananigans.TimeSteppers.update_state!(land::RucSlabLand)
    grid  = land.grid
    arch  = architecture(grid)
    p     = land.parameters
    snow  = land.snow
    veg   = land.vegetation
    props = land.properties
    forc  = land.forcings

    launch!(arch, grid, :xy, _finalize_snow_cover!,
            snow.snowfrac, snow.snhei, props.znt,
            snow.rhosn, snow.rhonewsn,
            veg.is_urban, p.sncovfac, p.snowcovr_opt)

    launch!(arch, grid, :xy, _update_surface_properties!,
            props.alb, props.emiss, props.znt,
            snow.snowfrac, snow.keep_snow_albedo,
            land.temperature, snow.newsn, snow.snhei,
            veg.vegfrac, snow.alb_snow_local,
            veg.albedo_veg, veg.emissivity_veg, veg.z0_veg,
            p.alb_bare,
            p.emiss_snow, p.emiss_bare,
            p.z0_snow, p.z0_bare)

    launch!(arch, grid, :xy, _update_mavail_rs_rg!,
            veg.mavail, veg.soilres, veg.r_s, veg.r_g,
            land.soil_moisture, snow.snowfrac,
            land.temperature, forc.surface_pressure,
            veg.vegfrac, veg.lai,
            forc.solar_irradiance, forc.air_humidity, forc.air_temperature,
            veg.r_smin,
            p.theta_wilt, p.theta_fc, p.theta_air_dry, p.theta_sat,
            p.r_smax, p.r_gmin, p.r_gmax,
            p.rg_lim, p.vpd_lim, p.T_opt)

    return nothing
end

"""
    time_step!(land::RucSlabLand, Δt)

Advance the slab by `Δt` in flux-driven mode (use `surface_balance_vilka!`
for the implicit alternative). Order:

  1. Ground-slab temperature: `T_g -= Jᵀ_g Δt / H_g`.
  2. Canopy-slab temperature: `T_c -= Jᵀ_c Δt / (ρcH)_c`.
  3. Drain carried-over liquid water above the slab `swl` capacity.
  4. Compaction of the existing pack.
  5. Canopy interception → throughfall to soil (`infwater`) and
     snowpack (`swe_inflow`).
  6. Wet-canopy direct evaporation drains `cst`.
  7. New-snow accumulation; resets the local aging snow albedo toward
     the fresh-snow value.
  8. Sublimation drain on SWE (cold pack, `T_g ≤ 273.15 K`).
  9. Snow melt (warm pack, `T_g > 273.15 K`) with the RUC melt cap and
     retained-melt split → `swl` plus `swl_overflow`; `T_g` cools by exactly
     the latent heat consumed.
 10. Continuous snow-albedo aging (decay toward `alb_snow_aged` with
     `T_g`-dependent timescale).
 11. Single-bucket soil-moisture update (gains: `infwater + swl_overflow`;
     losses: bare-soil evap × `(1-snowfrac)` × RUC `soilres` + transpiration).
 12. Soil freeze/thaw: split between `θ_liq` and `θ_ice` with phase-change
     latent heat absorbed/released by the ground slab.
 13. Refresh diagnostics (`snowfrac`, `α/ε/z₀`, `mavail`, `soilres`,
     `r_s`, `r_g`).
"""
function Oceananigans.TimeSteppers.time_step!(land::RucSlabLand, Δt)
    tick!(land.clock, Δt)
    grid   = land.grid
    arch   = architecture(grid)
    p      = land.parameters
    snow   = land.snow
    canopy = land.canopy
    veg    = land.vegetation
    forc   = land.forcings
    T      = land.temperature
    Jᵀ     = land.temperature_flux
    Tc     = land.canopy_temperature
    Jᵀc    = land.canopy_temperature_flux
    θ      = land.soil_moisture
    θ_ice  = land.soil_moisture_ice
    FT     = eltype(grid)
    ρ_w    = convert(FT, ρ_water_const)
    L_f    = convert(FT, p.latent_heat_fusion)
    ρcH_g  = p.density * p.heat_capacity * p.depth   # J m⁻² K⁻¹

    # 1, 2 — Slab temperatures
    launch!(arch, grid, :xy, _step_temperature!, T, Jᵀ, Δt, p.depth)
    launch!(arch, grid, :xy, _step_canopy_temperature!,
            Tc, Jᵀc, Δt, p.canopy_heat_capacity)

    # 3 — Drain retained liquid from previous steps above capacity.
    launch!(arch, grid, :xy, _drain_swl!,
            snow.swl, snow.snwe, snow.swl_overflow,
            p.snow_liquid_capacity_frac)

    # 4 — Compaction of the existing pack, before current snowfall is added.
    launch!(arch, grid, :xy, _compact_snow!,
            snow.rhosn, snow.snwe, snow.snhei, snow.swl, T, Δt,
            p.c1_compaction, p.c2_compaction, p.rhosn_min, p.rhosn_max)

    # 5 — Canopy interception
    launch!(arch, grid, :xy, _intercept_precip!,
            canopy.cst, canopy.drip,
            canopy.interw, canopy.intersn,
            canopy.infwater, canopy.intwratio,
            veg.canopy_capacity, veg.vegfrac, veg.lai,
            forc.rainfall_rate, forc.snowfall_rate,
            snow.swe_inflow, Δt, p.canopy_water_capacity)

    # 6 — Wet-canopy direct evaporation
    launch!(arch, grid, :xy, _evaporate_canopy!,
            canopy.cst, forc.canopy_evaporation, Δt, ρ_w)

    # 7 — New-snow accumulation
    fill!(snow.newsn, 0)
    fill!(snow.snowfracnewsn, 0)
    fill!(snow.keep_snow_albedo, 0)
    fill!(snow.rhonewsn, 100)
    launch!(arch, grid, :xy, _accumulate_new_snow!,
            snow.snwe, snow.snhei, snow.rhosn, snow.swl,
            snow.rhonewsn, snow.rhosnfall,
            snow.newsn, snow.snowfracnewsn, snow.keep_snow_albedo,
            snow.snowfallac, snow.alb_snow_local,
            snow.swe_inflow, forc.air_temperature, Δt,
            p.alb_snow_fresh,
            p.rhosn_min, p.rhosn_max)

    # 8 — Sublimation (cold pack)
    launch!(arch, grid, :xy, _apply_sublimation!,
            snow.snwe, snow.snhei, snow.swl, forc.moisture_flux,
            snow.snowfrac, snow.rhosn, T, Δt, ρ_w, p.rhosn_min)

    # 9 — Melt (warm pack), with RUC melt cap and retained-water split.
    launch!(arch, grid, :xy, _melt_snow!,
            snow.snwe, snow.snhei, snow.swl, snow.swl_overflow, T,
            snow.rhosn, snow.newsn, snow.rhonewsn,
            ρcH_g, L_f, ρ_w, p.rhosn_min, p.rhosn_max,
            Δt, p.meltfactor,
            p.snow_retention_min_frac,
            p.snow_retention_max_frac,
            p.snow_retention_depth_scale,
            p.snow_retention_depth_factor)

    # 10 — Continuous snow-albedo aging
    launch!(arch, grid, :xy, _age_snow_albedo!,
            snow.alb_snow_local, T, snow.snwe,
            p.alb_snow_aged, p.snow_aging_tau_cold, p.snow_aging_tau_warm, Δt)

    # 11 — Soil moisture
    launch!(arch, grid, :xy, _step_soil_moisture!,
            θ, canopy.infwater, snow.swl_overflow,
            forc.moisture_flux, forc.transpiration,
            veg.vegfrac, veg.soilres, snow.snowfrac,
            Δt, ρ_w, p.soil_depth,
            p.theta_sat, p.theta_air_dry)

    # 12 — Soil freeze/thaw
    launch!(arch, grid, :xy, _freeze_thaw_soil!,
            θ, θ_ice, T, ρcH_g, L_f, ρ_w, p.soil_depth)

    # 13 — Refresh diagnostics
    update_state!(land)

    fill_halo_regions!(T)
    fill_halo_regions!(Tc)
    fill_halo_regions!(θ)
    fill_halo_regions!(θ_ice)
    return nothing
end

#####
##### EarthSystemModel interface
#####

"""
    update_net_fluxes!(coupled_model, land::RucSlabLand)

Consume the atmosphere--land turbulent fluxes computed by
`compute_atmosphere_land_fluxes!` and write them into the slab's
forcing fields. The slab's own `time_step!` then advances the
prognostic state from these forcings.

The interface fluxes follow the atmosphere-side sign convention used
elsewhere in `EarthSystemModels`:

  𝒬ᵀ, 𝒬ᵛ : atmospheric net energy gain (W m⁻²); negative when the
            surface heats the atmosphere is cooled-by-surface, positive
            when atmosphere loses energy to surface.
  Jᵛ      : atmospheric vapor flux; negative when vapor flows from
            atmosphere to surface (condensation).

The slab uses surface-positive-upward conventions, so we sign-flip:

  Q_net (into ground) = -(𝒬ᵀ + 𝒬ᵛ)
  Jᵀ (slab boundary)   = -Q_net / (ρ · c)
  F_v (slab evap, upward) = -Jᵛ
"""
function update_net_fluxes!(coupled_model, land::RucSlabLand)
    al_interface = coupled_model.interfaces.atmosphere_land_interface
    isnothing(al_interface) && return nothing

    fluxes = al_interface.fluxes
    grid   = land.grid
    arch   = architecture(grid)
    p      = land.parameters
    ρcₛ    = p.density * p.heat_capacity

    launch!(arch, grid, :xy, _assemble_slab_land_fluxes!,
            land.temperature_flux,
            land.forcings.moisture_flux,
            fluxes, ρcₛ)
    return nothing
end

@kernel function _assemble_slab_land_fluxes!(Jᵀ, M, fluxes, ρcₛ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        𝒬ᵀ = fluxes.sensible_heat[i, j, 1]
        𝒬ᵛ = fluxes.latent_heat[i, j, 1]
        Jᵛ = fluxes.water_vapor[i, j, 1]

        Q_net_into_ground = -(𝒬ᵀ + 𝒬ᵛ)      # W m⁻²
        Jᵀ[i, j, 1] = -Q_net_into_ground / ρcₛ
        M[i, j, 1]  = -Jᵛ                    # kg m⁻² s⁻¹ upward
    end
end

interpolate_state!(exchanger, grid, ::RucSlabLand, coupled_model) = nothing

function ComponentExchanger(land::RucSlabLand, grid)
    state = (T      = land.temperature,
             Tc     = land.canopy_temperature,
             θ      = land.soil_moisture,
             θ_ice  = land.soil_moisture_ice,
             alb    = land.properties.alb,
             emiss  = land.properties.emiss,
             znt    = land.properties.znt,
             mavail = land.vegetation.mavail,
             soilres = land.vegetation.soilres,
             r_s    = land.vegetation.r_s,
             r_g    = land.vegetation.r_g)
    return ComponentExchanger(state, nothing)
end

initialize!(::ComponentExchanger, grid, ::RucSlabLand) = nothing

#####
##### Checkpointing
#####

import Oceananigans: prognostic_state, restore_prognostic_state!

function prognostic_state(land::RucSlabLand)
    snow_state   = map(f -> Array(interior(f)), land.snow)
    canopy_state = map(f -> Array(interior(f)), land.canopy)
    veg_state    = map(f -> Array(interior(f)), land.vegetation)
    props_state  = map(f -> Array(interior(f)), land.properties)
    return (; clock      = prognostic_state(land.clock),
              T          = Array(interior(land.temperature)),
              Tc         = Array(interior(land.canopy_temperature)),
              θ          = Array(interior(land.soil_moisture)),
              θ_ice      = Array(interior(land.soil_moisture_ice)),
              snow       = snow_state,
              canopy     = canopy_state,
              vegetation = veg_state,
              properties = props_state)
end

function restore_prognostic_state!(land::RucSlabLand, state)
    restore_prognostic_state!(land.clock, state.clock)
    interior(land.temperature)         .= state.T
    interior(land.canopy_temperature)  .= state.Tc
    interior(land.soil_moisture)       .= state.θ
    interior(land.soil_moisture_ice)   .= state.θ_ice
    for k in keys(land.snow)
        interior(getproperty(land.snow, k)) .= getproperty(state.snow, k)
    end
    for k in keys(land.canopy)
        interior(getproperty(land.canopy, k)) .= getproperty(state.canopy, k)
    end
    for k in keys(land.vegetation)
        interior(getproperty(land.vegetation, k)) .= getproperty(state.vegetation, k)
    end
    for k in keys(land.properties)
        interior(getproperty(land.properties, k)) .= getproperty(state.properties, k)
    end
    update_state!(land)
    return land
end

restore_prognostic_state!(land::RucSlabLand, ::Nothing) = land
