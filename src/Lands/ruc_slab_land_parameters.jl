#####
##### `RucSlabLandParameters` — scalar parameter bag for a SlabLand
##### composed of the RUC closure trio.
#####
##### Loaded before `RucEnergy`, `RucHydrology`, and
##### `RucSurfaceProperties` because each RUC closure dispatches on
##### `::RucSlabLandParameters` in its constructor.
#####

"""
    RucSlabLandParameters(FT = Float64; kwargs...)

Tunable scalar parameters for a SlabLand composed of the RUC closure
trio. The bag is the single source of truth for the convenience
constructor: `RucEnergy`, `RucHydrology`, and `RucSurfaceProperties`
each pluck the fields they need from it.

Defaults reproduce the RUC LSM configuration of Smirnova et al. (1997,
2016) for snow, canopy interception, and top-layer moisture
availability, plus the Manabe (1969) single-bucket soil moisture.

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
- `theta_air_dry = 0.05`         : air-dry / residual water content [m³ m⁻³]

# Soil hydraulics (Clapp-Hornberger; sub-zero unfrozen-water curve)
- `psi_sat = 0.355`              : magnitude of matric head at saturation [m]
- `bclh = 5.25`                  : Clapp-Hornberger pore-size index `b`

# Surface optical / aerodynamic constants
- `alb_snow = 0.85`, `alb_bare = 0.25`, `alb_veg = 0.18`
- `emiss_snow = 0.98`, `emiss_bare = 0.95`, `emiss_veg = 0.98`
- `z0_snow = 0.011`, `z0_bare = 0.05`, `z0_veg = 0.20` [m]

# Snow-cover fraction (Koren 1999; Niu-Yang 2007)
- `sncovfac = 0.04`           : Niu-Yang scale factor [m]
- `snowcovr_opt = 2`          : 1 = linear; 2 = blend; 3 = Niu-Yang `tanh`

# Snow density compaction (Anderson 1976; Smirnova 1997)
- `c1_compaction = 0.026`, `c2_compaction = 21.0`
- `rhosn_min = 58.8`, `rhosn_max = 500.0`         [kg m⁻³]

# Snow melt and liquid water retention
- `snow_liquid_capacity_frac = 0.04`, `meltfactor = 2.0`
- `snow_retention_min_frac = 0.08`, `snow_retention_max_frac = 0.18`
- `snow_retention_depth_scale = 0.10`, `snow_retention_depth_factor = 0.13`

# Phase change
- `latent_heat_fusion = 3.337e5`  : J kg⁻¹

# Stomatal resistance (Jarvis-Stewart)
- `r_smin = 100.0`, `r_smax = 5000.0`, `rg_lim = 100.0`,
  `vpd_lim = 4.0e-3`, `T_opt = 298.0`
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
    # Soil hydraulics
    psi_sat :: FT
    bclh :: FT
    # Optics / roughness end-members
    alb_snow :: FT
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
    # Snow liquid retention + phase change
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
                            psi_sat = 0.355,
                            bclh = 5.25,
                            alb_snow = 0.85,
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
                            snow_liquid_capacity_frac = 0.04,
                            meltfactor = 2.0,
                            snow_retention_min_frac = 0.08,
                            snow_retention_max_frac = 0.18,
                            snow_retention_depth_scale = 0.10,
                            snow_retention_depth_factor = 0.13,
                            latent_heat_fusion = 3.337e5,
                            r_smin = 100.0,
                            r_smax = 5000.0,
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
        convert(FT, psi_sat),
        convert(FT, bclh),
        convert(FT, alb_snow),
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
        convert(FT, snow_liquid_capacity_frac),
        convert(FT, meltfactor),
        convert(FT, snow_retention_min_frac),
        convert(FT, snow_retention_max_frac),
        convert(FT, snow_retention_depth_scale),
        convert(FT, snow_retention_depth_factor),
        convert(FT, latent_heat_fusion),
        convert(FT, r_smin),
        convert(FT, r_smax),
        convert(FT, rg_lim),
        convert(FT, vpd_lim),
        convert(FT, T_opt),
    )
end
