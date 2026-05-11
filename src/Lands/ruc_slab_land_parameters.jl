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
- `soil_depth = 1.0`                  : root-zone bucket depth `H_s` [m]
- `theta_saturation = 0.45`           : saturated vol. water content [m³ m⁻³]
- `theta_field_capacity = 0.30`       : field capacity [m³ m⁻³]
- `theta_wilting_point = 0.10`        : wilting point [m³ m⁻³]
- `theta_air_dry = 0.05`              : air-dry / residual water content [m³ m⁻³]

# Soil hydraulics (Clapp-Hornberger; sub-zero unfrozen-water curve)
- `psi_saturation = 0.355`            : magnitude of matric head at saturation [m]
- `clapp_hornberger_exponent = 5.25`  : Clapp-Hornberger pore-size index `b`

# Surface optical / aerodynamic constants
- `albedo_snow = 0.85`, `albedo_bare = 0.25`, `albedo_vegetation = 0.18`
- `emissivity_snow = 0.98`, `emissivity_bare = 0.95`, `emissivity_vegetation = 0.98`
- `roughness_length_snow = 0.011`, `roughness_length_bare = 0.05`,
  `roughness_length_vegetation = 0.20` [m]

# Snow-cover fraction (Koren 1999; Niu-Yang 2007)
- `snow_cover_scale_factor = 0.04`    : Niu-Yang scale factor [m]
- `snow_cover_option = 2`             : 1 = linear; 2 = blend; 3 = Niu-Yang `tanh`

# Snow density compaction (Anderson 1976; Smirnova 1997)
- `c1_compaction = 0.026`, `c2_compaction = 21.0`
- `snow_density_min = 58.8`, `snow_density_max = 500.0`         [kg m⁻³]

# Snow melt and liquid water retention
- `snow_liquid_capacity_fraction = 0.04`, `melt_factor = 2.0`
- `snow_retention_min_fraction = 0.08`, `snow_retention_max_fraction = 0.18`
- `snow_retention_depth_scale = 0.10`, `snow_retention_depth_factor = 0.13`

# Phase change
- `latent_heat_fusion = 3.337e5`  : J kg⁻¹

# Stomatal resistance (Jarvis-Stewart)
- `stomatal_resistance_min = 100.0`, `stomatal_resistance_max = 5000.0`,
  `solar_radiation_limit = 100.0`, `vapor_pressure_deficit_limit = 4.0e-3`,
  `temperature_optimum = 298.0`
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
    theta_saturation :: FT
    theta_field_capacity :: FT
    theta_wilting_point :: FT
    theta_air_dry :: FT
    # Soil hydraulics
    psi_saturation :: FT
    clapp_hornberger_exponent :: FT
    # Optics / roughness end-members
    albedo_snow :: FT
    albedo_bare :: FT
    albedo_vegetation :: FT
    emissivity_snow :: FT
    emissivity_bare :: FT
    emissivity_vegetation :: FT
    roughness_length_snow :: FT
    roughness_length_bare :: FT
    roughness_length_vegetation :: FT
    # Snow-cover fraction
    snow_cover_scale_factor :: FT
    snow_cover_option :: Int
    # Snow density compaction
    c1_compaction :: FT
    c2_compaction :: FT
    snow_density_min :: FT
    snow_density_max :: FT
    # Snow liquid retention + phase change
    snow_liquid_capacity_fraction :: FT
    melt_factor :: FT
    snow_retention_min_fraction :: FT
    snow_retention_max_fraction :: FT
    snow_retention_depth_scale :: FT
    snow_retention_depth_factor :: FT
    latent_heat_fusion :: FT
    # Resistances
    stomatal_resistance_min :: FT
    stomatal_resistance_max :: FT
    solar_radiation_limit :: FT
    vapor_pressure_deficit_limit :: FT
    temperature_optimum :: FT
end

function RucSlabLandParameters(FT::Type = Float64;
                            depth = 0.10,
                            density = 1500,
                            heat_capacity = 1480,
                            canopy_heat_capacity = 1.0e4,
                            canopy_water_capacity = 5.0e-4,
                            soil_depth = 1.0,
                            theta_saturation = 0.45,
                            theta_field_capacity = 0.30,
                            theta_wilting_point = 0.10,
                            theta_air_dry = 0.05,
                            psi_saturation = 0.355,
                            clapp_hornberger_exponent = 5.25,
                            albedo_snow = 0.85,
                            albedo_bare = 0.25,
                            albedo_vegetation = 0.18,
                            emissivity_snow = 0.98,
                            emissivity_bare = 0.95,
                            emissivity_vegetation = 0.98,
                            roughness_length_snow = 0.011,
                            roughness_length_bare = 0.05,
                            roughness_length_vegetation = 0.20,
                            snow_cover_scale_factor = 0.04,
                            snow_cover_option = 2,
                            c1_compaction = 0.026,
                            c2_compaction = 21.0,
                            snow_density_min = 58.8,
                            snow_density_max = 500.0,
                            snow_liquid_capacity_fraction = 0.04,
                            melt_factor = 2.0,
                            snow_retention_min_fraction = 0.08,
                            snow_retention_max_fraction = 0.18,
                            snow_retention_depth_scale = 0.10,
                            snow_retention_depth_factor = 0.13,
                            latent_heat_fusion = 3.337e5,
                            stomatal_resistance_min = 100.0,
                            stomatal_resistance_max = 5000.0,
                            solar_radiation_limit = 100.0,
                            vapor_pressure_deficit_limit = 4.0e-3,
                            temperature_optimum = 298.0)

    return RucSlabLandParameters{FT}(
        convert(FT, depth),
        convert(FT, density),
        convert(FT, heat_capacity),
        convert(FT, canopy_heat_capacity),
        convert(FT, canopy_water_capacity),
        convert(FT, soil_depth),
        convert(FT, theta_saturation),
        convert(FT, theta_field_capacity),
        convert(FT, theta_wilting_point),
        convert(FT, theta_air_dry),
        convert(FT, psi_saturation),
        convert(FT, clapp_hornberger_exponent),
        convert(FT, albedo_snow),
        convert(FT, albedo_bare),
        convert(FT, albedo_vegetation),
        convert(FT, emissivity_snow),
        convert(FT, emissivity_bare),
        convert(FT, emissivity_vegetation),
        convert(FT, roughness_length_snow),
        convert(FT, roughness_length_bare),
        convert(FT, roughness_length_vegetation),
        convert(FT, snow_cover_scale_factor),
        Int(snow_cover_option),
        convert(FT, c1_compaction),
        convert(FT, c2_compaction),
        convert(FT, snow_density_min),
        convert(FT, snow_density_max),
        convert(FT, snow_liquid_capacity_fraction),
        convert(FT, melt_factor),
        convert(FT, snow_retention_min_fraction),
        convert(FT, snow_retention_max_fraction),
        convert(FT, snow_retention_depth_scale),
        convert(FT, snow_retention_depth_factor),
        convert(FT, latent_heat_fusion),
        convert(FT, stomatal_resistance_min),
        convert(FT, stomatal_resistance_max),
        convert(FT, solar_radiation_limit),
        convert(FT, vapor_pressure_deficit_limit),
        convert(FT, temperature_optimum),
    )
end
