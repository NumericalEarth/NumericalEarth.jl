# # HI-SCALE at the ARM Southern Great Plains: a data-driven canopy land surface
#
# This example rebuilds the land surface of the 2016 HI-SCALE campaign case of
# [Fast et al. (2019)](@cite fast2019hiscale) — 30 August 2016 over the ARM Southern
# Great Plains (SGP) megasite in north-central Oklahoma — with the tiled
# canopy/bare land model, every surface parameter ingested from a measured dataset,
# and six days of ERA5-forced spin-up ahead of the case day.
#
# Fast et al. showed that the day's complex shallow-cumulus population was organized
# by the land: realistic soil-moisture variability and 30 m land use produced the
# observed cloud heterogeneity where smooth analysis fields produced uniform cloud
# streets. Dry-soil regions ran high sensible heat and deep boundary layers; wet
# regions and cool lakes suppressed convection. This example reproduces the land
# side of that story on the paper's own nested domains — an outer 297 km box at
# 300 m and an inner 120 km box at 100 m, both centered on the SGP Central Facility
# (36.607°N, 97.488°W) — and verifies that the modeled fluxes land on the observed
# magnitudes (midafternoon latent heat of 250–450 W m⁻² around the Central Facility,
# sensible heat anticorrelated with soil moisture).
#
# Each model component reads a real dataset:
#
# | surface property | dataset | resolution |
# |---|---|---|
# | terrain (DSM → bare earth) | Copernicus GLO-30 | 30 m |
# | canopy height | ETH Sentinel-2 canopy height | 10 m |
# | building morphometry | 3D-GloBFP footprints | building scale |
# | leaf area index | MODIS MCD15A2H + seasonal climatology | 500 m, 8-day |
# | land cover | MODIS MCD12Q1 (IGBP) + ESA WorldCover fractions | 500 m / 10 m |
# | soil hydraulics | OpenLandMap texture → pedotransfer functions | 30 m |
# | broadband emissivity | ASTER GED v3 | 100 m |
# | blue-sky albedo | Copernicus Global Land | 1 km |
# | forcing | ERA5 single levels | 31 km, hourly |
# | initial / deep soil state | ERA5-Land | 9 km |
#
# The vegetation and buildings feed the Monin–Obukhov solver as per-cell momentum
# and scalar roughness lengths and zero-plane displacements (the Raupach
# drag-partition closure over the vegetated tile, building morphometry over the
# built fabric), and the albedo/emissivity fields enter the canopy-air-space
# radiation balance per cell.
#
# !!! note "Access"
#     GLO-30 needs a (free) DestinE token in `DESTINE_ACCESS_TOKEN` and `Zarr`;
#     MODIS and ASTER GED need NASA Earthdata credentials (`EARTHDATA_USERNAME` /
#     `EARTHDATA_PASSWORD`) and `ArchGDAL`; ERA5, ERA5-Land, and the Copernicus
#     albedo need CDS credentials at `~/.cdsapirc` and `CDSAPI`. The ETH canopy
#     height, ESA WorldCover, 3D-GloBFP, and OpenLandMap tiles are open.

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: on_architecture
using Oceananigans.Fields: interpolate!
using NumericalEarth.EarthSystemModels.InterfaceComputations: atmosphere_land_stability_functions
using CUDA          ## GPU backend
using Zarr          ## GLO-30 Zarr store
using ArchGDAL      ## WorldCover / ETH / MODIS / ASTER / OpenLandMap / 3D-GloBFP rasters
using CDSAPI        ## ERA5 single levels, Copernicus albedo
using CopernicusClimateDataStore   ## ERA5-Land
using CairoMakie
using JLD2
using Printf
using Statistics: mean, median
import Dates
import Dates: DateTime

# ## The nested HI-SCALE domains
#
# The paper's two one-way-nested WRF-LES domains, centered on the SGP Central
# Facility. A land column has no lateral neighbors, so the "nesting" here is the
# same ERA5 forcing sampled at the two paper resolutions; the inner 100 m domain
# resolves the 10–30 m surface datasets that the 300 m outer domain smooths.

arch = GPU()

centre_latitude  = 36.607
centre_longitude = -97.488
kilometers_per_degree = 111.32

half_extents(kilometers) = (kilometers / 2 / (kilometers_per_degree * cosd(centre_latitude)),
                            kilometers / 2 / kilometers_per_degree)

## Float32: the flux solve dominates the cost and Tesla-class cards run Float64
## at a fraction of their Float32 rate.
function hiscale_grid(architecture, extent_kilometers, size)
    Δλ, Δφ = half_extents(extent_kilometers)
    return LatitudeLongitudeGrid(architecture, Float32; size,
                                 longitude = (centre_longitude - Δλ, centre_longitude + Δλ),
                                 latitude  = (centre_latitude  - Δφ, centre_latitude  + Δφ),
                                 topology  = (Bounded, Bounded, Flat))
end

# `footprint_resolution` (m) is the rasterization step for the 3D-GloBFP building
# polygons — about fifteen samples per grid cell keeps the morphometry statistics
# converged without materializing a continent-scale meter raster.
domains = (outer_300m = (extent = 297, size = ( 990,  990), footprint_resolution = 20),
           inner_100m = (extent = 120, size = (1200, 1200), footprint_resolution = 7))

# One padded region covers both domains, so every regional dataset file is shared.

region = BoundingBox(longitude = (-99.35, -95.65), latitude = (35.1, 38.1))

# ## The case window
#
# Six days of spin-up bring the slab water and temperature into diurnal
# equilibrium before the case day (1200 UTC 30 August = 0600 CST).

start_date = DateTime(2016, 8, 24, 12)
end_date   = DateTime(2016, 8, 31)
case_start = DateTime(2016, 8, 30, 12)

# ## Leaf area index with a class-aware gap fill
#
# MODIS MCD15A2H leaf area index for the composite containing the case day, with
# its cloud/quality gaps filled by the class-aware seasonal machinery: a 46-period
# climatology (2014–2018) anchors each cell's own seasonal cycle, MCD12Q1 classes
# decide how far a gap may be carried and from whom it may borrow, and
# non-vegetated classes are zeroed. Everything happens on the MODIS native lattice
# (≈464 m); the model grids sample it afterwards.

modis_classes = Field(Metadatum(:landcover_class; dataset = MCD12Q1(), region, date = DateTime(2016)))

climatology = MODISLAIClimatology(years = 2014:2018)
build_lai_climatology!(climatology; region)   ## resumable; skips periods already on disk

climatological_lai = FieldTimeSeries(Metadata(:leaf_area_index; dataset = climatology, region);
                                     time_indices_in_memory = 46)
fill_seasonal_gaps!(climatological_lai, modis_classes; cyclic = true,
                    max_gap = class_maximum_gap(modis_classes),
                    valid_range = (0, 10),
                    unfilled_classes = igbp_non_vegetated_classes)

# The 2016 composites bracketing the case day, anchored to the filled climatology.

case_dates = [DateTime(2016, 7, 27), DateTime(2016, 8, 4), DateTime(2016, 8, 12),
              DateTime(2016, 8, 20), DateTime(2016, 8, 28), DateTime(2016, 9, 5),
              DateTime(2016, 9, 13)]

case_lai_series = FieldTimeSeries{Center, Center, Nothing}(modis_classes.grid,
                                                           [Dates.datetime2unix(d) for d in case_dates])
for (n, date) in enumerate(case_dates)
    composite = Field(Metadatum(:leaf_area_index; dataset = MCD15A2H(), region, date))
    parent(case_lai_series[n]) .= parent(composite)
end

anchor_periods = [period_index(date, MCD15A2H()) for date in case_dates]
fill_seasonal_gaps!(case_lai_series, modis_classes;
                    anchor = climatological_lai, anchor_periods,
                    max_gap = class_maximum_gap(modis_classes),
                    cyclic = false, valid_range = (0, 10),
                    unfilled_classes = igbp_non_vegetated_classes)
zero_non_vegetated!(case_lai_series, modis_classes)

case_lai = case_lai_series[5]   ## the 28 Aug – 4 Sep composite covering the case day

# ## Soil texture onto both domains' three-layer lattices
#
# Each OpenLandMap 30 m texture window is read once (the heavy part is the native
# materialization — a few GB per variable) and immediately interpolated onto a
# three-layer lattice per domain, so no more than one native field is resident.

function soil_lattice(extent_kilometers, size)
    Δλ, Δφ = half_extents(extent_kilometers)
    return LatitudeLongitudeGrid(CPU(), Float64; size = (size[1], size[2], 3),
                                 longitude = (centre_longitude - Δλ, centre_longitude + Δλ),
                                 latitude  = (centre_latitude  - Δφ, centre_latitude  + Δφ),
                                 z = [-1.0, -0.6, -0.3, 0.0],
                                 topology = (Bounded, Bounded, Bounded))
end

soil_texture_names = (:sand_fraction, :silt_fraction, :clay_fraction, :bulk_density)

domain_texture = map(domains) do spec
    lattice = soil_lattice(spec.extent, spec.size)
    map(name -> Field{Center, Center, Center}(lattice),
        NamedTuple{soil_texture_names}(soil_texture_names))
end

for name in soil_texture_names
    native = Field(Metadatum(name; dataset = OpenLandMapSoilDB(), region), CPU())
    interior(native) .= ifelse.(isfinite.(interior(native)), interior(native), 0)
    for texture in domain_texture
        interpolate!(getproperty(texture, name), native)
    end
    native = nothing
    GC.gc()
end

# ## Per-domain land surface fields
#
# `land_surface_fields` regrids every dataset onto one model grid. Categorical
# fields (the IGBP class) are sampled nearest-neighbor from the MODIS lattice;
# continuous fields regrid through the shared bilinear path.

new_surface_field(grid) = Field{Center, Center, Nothing}(grid)

function nearest_native_classes(grid, native_classes)
    classes = zeros(Int, size(grid, 1), size(grid, 2))
    native  = interior(on_architecture(CPU(), native_classes), :, :, 1)
    native_grid = native_classes.grid
    λⁿ, φⁿ, _ = nodes(native_grid, Center(), Center(), Center())
    λ, φ, _ = nodes(grid, Center(), Center(), Center())
    Δλ = λⁿ[2] - λⁿ[1]
    Δφ = φⁿ[2] - φⁿ[1]
    for j in axes(classes, 2), i in axes(classes, 1)
        iⁿ = clamp(round(Int, 1 + (λ[i] - λⁿ[1]) / Δλ), 1, length(λⁿ))
        jⁿ = clamp(round(Int, 1 + (φ[j] - φⁿ[1]) / Δφ), 1, length(φⁿ))
        value = native[iⁿ, jⁿ]
        classes[i, j] = isfinite(value) ? round(Int, value) : igbp_class_names.water
    end
    return classes
end

## Fill NaN cells (water, no-retrieval) of a regridded field with a constant.
function fill_invalid!(field, value)
    parent(field) .= ifelse.(isfinite.(parent(field)), parent(field), value)
    return field
end

function transfer_to(grid, cpu_field)
    field = new_surface_field(grid)
    set!(field, Array(interior(cpu_field, :, :, 1)))
    return field
end

function land_surface_fields(grid, footprint_resolution, texture)
    cpu_grid = on_architecture(CPU(), grid)

    ## --- Terrain: GLO-30 DSM decomposed into bare earth + canopy + buildings.
    dsm           = regrid_topography(grid; dataset = GLO30())
    canopy_height = canopy_height_field(grid, ETHSentinel2CanopyHeight())
    fill_invalid!(canopy_height, 0)

    ## The building raster covers this grid only (a region-wide meter raster
    ## does not fit in memory), at the domain's footprint resolution.
    footprints  = GlobalBuildingFootprints3D(resolution = footprint_resolution)
    morphometry = building_morphometry(grid; dataset = footprints,
                                       region = BoundingBox(cpu_grid; padding = 0.05))
    building_lift = morphometry.gross_building_height
    ground        = bare_earth_elevation(dsm, (canopy_height, building_lift))

    ## --- Land cover: WorldCover fractions + the dominant IGBP class.
    worldcover = ESAWorldCover(aggregation_factor = 6)
    vegetation_fraction = Field(Metadatum(:vegetation_fraction; dataset = worldcover, region), grid)
    water_fraction      = Field(Metadatum(:permanent_water_bodies_fraction; dataset = worldcover, region), grid)
    built_fraction      = Field(Metadatum(:built_up_fraction; dataset = worldcover, region), grid)
    fill_invalid!(vegetation_fraction, 0); fill_invalid!(water_fraction, 0); fill_invalid!(built_fraction, 0)

    igbp = nearest_native_classes(cpu_grid, modis_classes)

    ## --- Leaf area index on the model grid (cell mean).
    leaf_area_index_cpu = new_surface_field(cpu_grid)
    interpolate!(leaf_area_index_cpu, case_lai)
    fill_invalid!(leaf_area_index_cpu, 0)
    parent(leaf_area_index_cpu) .= clamp.(parent(leaf_area_index_cpu), 0, 10)
    leaf_area_index = transfer_to(grid, leaf_area_index_cpu)

    ## --- Soil hydraulics: OpenLandMap texture → pedotransfer → van Genuchten fields.
    slab_depth = 0.5
    hydraulics = soil_hydraulic_properties(texture.sand_fraction, texture.silt_fraction,
                                           texture.clay_fraction, texture.bulk_density; slab_depth)

    ## Median scalar parameters for the closures that take scalars.
    finite_median(f) = median(filter(isfinite, Array(interior(f))))
    positive_median(f) = median(filter(x -> isfinite(x) && x > 0, Array(interior(f))))
    scalar_porosity = positive_median(hydraulics.porosity)
    scalar_α        = positive_median(hydraulics.inverse_air_entry_head)
    scalar_n        = positive_median(hydraulics.pore_size_uniformity)

    ## Macropore-inclusive Cosby conductivity caps infiltration (kg m⁻² s⁻¹ = m s⁻¹ × ρˡ).
    cosby_conductivity = compute!(Field(saturated_conductivity(CosbyConductivity(), texture.sand_fraction)))
    infiltration_capacity = 1000 * positive_median(cosby_conductivity)

    hydraulic_fields = map(hydraulics) do f
        g = transfer_to(grid, f)
        fill_invalid!(g, finite_median(f))
        return g
    end

    ## Dry areal heat capacity from the 0–30 cm bulk density over a 0.15 m diurnal skin.
    bulk_density_top = new_surface_field(cpu_grid)
    interior(bulk_density_top) .= interior(texture.bulk_density, :, :, 3)
    dry_heat_capacity = transfer_to(grid, bulk_density_top)
    fill_invalid!(dry_heat_capacity, 1350)
    parent(dry_heat_capacity) .= 840 .* 0.15 .* parent(dry_heat_capacity)

    ## --- Radiation: ASTER GED broadband emissivity + Copernicus blue-sky albedo.
    emissivity = Field(Metadatum(:emissivity; dataset = ASTERGEDv3(resolution = ASTERGEDHigh100m), region), grid)
    fill_invalid!(emissivity, 0.97)

    albedo_dataset = CopernicusAlbedo()
    albedo_date = maximum(d for d in all_dates(albedo_dataset, :albedo) if d <= case_start)
    albedo = Field(Metadatum(:albedo; dataset = albedo_dataset, region, date = albedo_date), grid)
    fill_invalid!(albedo, 0.18)

    ## --- Initial and deep soil state: ERA5 skin temperature at the spin-up start,
    ## ERA5-Land August-mean volumetric soil water and layer-3 soil temperature
    ## (0.1°, NaN over the ERA5-Land ocean/lake mask).
    initial_temperature = Field(Metadatum(:skin_temperature;
                                          dataset = ERA5HourlySingleLevel(), region, date = start_date), grid)
    fill_invalid!(initial_temperature, 300)

    era5_land_august = DateTime(2016, 8, 1)
    soil_water = new_surface_field(grid)
    layers = (:volumetric_soil_water_layer_1, :volumetric_soil_water_layer_2, :volumetric_soil_water_layer_3)
    weights = (0.14, 0.42, 0.44)   ## ERA5-Land layers 0–7, 7–28, 28–100 cm sampled over 0–50 cm
    for (name, w) in zip(layers, weights)
        θ = Field(Metadatum(name; dataset = ERA5MonthlyLand(), region, date = era5_land_august), grid)
        fill_invalid!(θ, 0.25)
        parent(soil_water) .+= w .* parent(θ)
    end

    deep_temperature = Field(Metadatum(:soil_temperature_level_3;
                                       dataset = ERA5MonthlyLand(), region, date = era5_land_august), grid)
    fill_invalid!(deep_temperature, 299)

    ## --- ERA5's own model topography, for the elevation correction.
    era5_topography = Field(Metadatum(:topography; dataset = ERA5HourlySingleLevel(), region, date = start_date), grid)

    return (; dsm, canopy_height, building_lift, ground, morphometry,
              vegetation_fraction, water_fraction, built_fraction, igbp,
              leaf_area_index, slab_depth, hydraulic_fields,
              scalar_porosity, scalar_α, scalar_n, infiltration_capacity,
              dry_heat_capacity, emissivity, albedo,
              initial_temperature, soil_water, deep_temperature, era5_topography)
end

# ## Aerodynamic roughness from vegetation and buildings
#
# The Raupach drag-partition closure turns leaf area index and canopy height into a
# momentum roughness length and zero-plane displacement, with drag parameters and a
# fallback height per IGBP class; where ETH sees no trees (grass, crops) the class's
# representative height carries the closure. Building morphometry supplies the
# urban `(ℓᵐ, d)` through the Kanda-corrected Macdonald formulation, and water and
# barren cells use the prescribed constants. The vegetated tile carries the canopy
# values; the bare tile carries bare-soil roughness under the urban/water overlay.

vegetated_igbp_classes = (:evergreen_needleleaf_forest, :evergreen_broadleaf_forest,
                          :deciduous_needleleaf_forest, :deciduous_broadleaf_forest,
                          :mixed_forest, :closed_shrubland, :open_shrubland,
                          :woody_savanna, :savanna, :grassland, :permanent_wetland,
                          :cropland)

igbp_symbol(code) = code == igbp_class_names.cropland_natural_mosaic  ? :cropland_vegetation_mosaic :
                    code == igbp_class_names.permanent_snow_and_ice   ? :snow_and_ice :
                    begin
                        key = findfirst(==(code), NamedTuple{keys(igbp_class_names)}(igbp_class_names))
                        isnothing(key) ? :unclassified : key
                    end

function aerodynamic_fields(grid, static)
    cpu_grid = on_architecture(CPU(), grid)
    FT = eltype(grid)
    igbp = static.igbp
    symbols = map(igbp_symbol, igbp)

    ## Vegetated-area leaf area index: the MODIS cell mean divided by the cover fraction.
    fveg = Array(interior(static.vegetation_fraction, :, :, 1))
    cell_lai = Array(interior(static.leaf_area_index, :, :, 1))
    tile_lai_cpu = new_surface_field(cpu_grid)
    interior(tile_lai_cpu, :, :, 1) .= clamp.(cell_lai ./ clamp.(fveg, 0.25, 1), 0, 8)
    tile_lai = transfer_to(grid, tile_lai_cpu)

    ## Canopy height: measured where the 10 m product sees trees, class fallback elsewhere.
    eth = Array(interior(static.canopy_height, :, :, 1))
    class_height = [class in vegetated_igbp_classes ? representative_canopy_height(FT, class) : FT(0.1)
                    for class in symbols]
    height_cpu = new_surface_field(cpu_grid)
    interior(height_cpu, :, :, 1) .= max.(eth, class_height)
    effective_height = transfer_to(grid, height_cpu)

    ## Per-class Raupach roughness on the full grid, then a per-cell class selection.
    ℓᵐ_veg = zeros(FT, size(igbp)); d_veg = zeros(FT, size(igbp))
    for class in unique(symbols)
        class in vegetated_igbp_classes || class == :cropland_vegetation_mosaic || continue
        closure = DragPartitionRoughness(FT; vegetation_type = class)
        ℓᵐ_class, d_class = canopy_roughness(closure, tile_lai, effective_height)
        cells = symbols .== class
        ℓᵐ_veg[cells] .= Array(interior(ℓᵐ_class, :, :, 1))[cells]
        d_veg[cells]  .= Array(interior(d_class,  :, :, 1))[cells]
    end

    ## Non-vegetated classes: prescribed constants.
    for (class, target) in ((:urban, :urban), (:water, :water), (:barren, :barren),
                            (:snow_and_ice, :snow_and_ice))
        cells = symbols .== class
        any(cells) || continue
        ℓᵐ_c, d_c = nonvegetated_roughness(FT, target)
        ℓᵐ_veg[cells] .= ℓᵐ_c
        d_veg[cells]  .= d_c
    end

    replace!(ℓᵐ_veg, NaN => 0.03); replace!(d_veg, NaN => 0.2)
    clamp!(ℓᵐ_veg, 1e-4, 3); clamp!(d_veg, 0, 20)

    ## Bare tile: bare-soil roughness with the measured building morphometry overlay.
    ℓᵐ_urban, d_urban = urban_roughness(static.morphometry.mean_building_height,
                                        static.morphometry.plan_area_index,
                                        static.morphometry.building_height_deviation,
                                        static.morphometry.maximum_building_height,
                                        static.morphometry.frontal_area_index)
    ℓᵐ_u = Array(interior(ℓᵐ_urban, :, :, 1)); d_u = Array(interior(d_urban, :, :, 1))

    λᵖ    = Array(interior(static.morphometry.plan_area_index, :, :, 1))
    water = Array(interior(static.water_fraction, :, :, 1)) .> 0.5
    built = λᵖ .> 0.05

    ℓᵐ_bare = fill(FT(0.01), size(igbp)); d_bare = zeros(FT, size(igbp))
    ℓᵐ_bare[built] .= max.(ℓᵐ_u[built], 0.01); d_bare[built] .= max.(d_u[built], 0)
    ℓᵐ_bare[water] .= 1e-4; d_bare[water] .= 0
    replace!(ℓᵐ_bare, NaN => 0.01); replace!(d_bare, NaN => 0)
    clamp!(ℓᵐ_bare, 1e-4, 3); clamp!(d_bare, 0, 20)

    to_field(a) = (f = new_surface_field(grid); set!(f, a); f)

    return (vegetated = (momentum_roughness_length = to_field(ℓᵐ_veg),
                         scalar_roughness_length   = to_field(ℓᵐ_veg ./ 10),
                         zero_plane_displacement   = to_field(d_veg)),
            bare      = (momentum_roughness_length = to_field(ℓᵐ_bare),
                         scalar_roughness_length   = to_field(ℓᵐ_bare ./ 10),
                         zero_plane_displacement   = to_field(d_bare)),
            tile_leaf_area_index = tile_lai,
            effective_height     = effective_height,
            water                = water)
end

# ## The coupled model
#
# The soil column is a `VariablySaturatedHydrology` whose van Genuchten parameters
# come from the pedotransfer fields, wrapped in a `SurfaceWaterStore` (rain rejected
# by the infiltration cap ponds and drains rather than vanishing) and an
# `InterceptingHydrology` (the canopy holds and re-evaporates intercepted rain).
# The energy budget restores toward the ERA5-Land deep soil temperature. The
# atmosphere-facing surface is a two-tile mosaic: a `CanopyAirSpace` over the
# vegetated fraction — big-leaf Jarvis conductance under plant-available water
# stress, interactive absorbed PAR, interception — and its bare counterpart over
# the rest, each tile carrying its own per-cell roughness, displacement, albedo,
# and emissivity.

surface_layer_height  = 30    ## above the tallest displacement heights in the domain
boundary_layer_height = 1500

function hiscale_land_model(grid, static, aero, atmosphere, radiation)
    FT = eltype(grid)

    soil = VariablySaturatedHydrology(FT;
        slab_depth = static.slab_depth,
        storage_height = 1000,
        porosity = static.hydraulic_fields.porosity,
        residual_liquid_fraction = static.hydraulic_fields.residual_liquid_fraction,
        retention_curve = VanGenuchtenRetention(FT;
            inverse_air_entry_head = static.hydraulic_fields.inverse_air_entry_head,
            pore_size_uniformity   = static.hydraulic_fields.pore_size_uniformity),
        hydraulic_conductivity = VanGenuchtenConductivity(FT;
            matching_point_conductivity = static.hydraulic_fields.matching_point_conductivity,
            pore_size_uniformity        = static.hydraulic_fields.pore_size_uniformity,
            pore_connectivity_exponent  = static.hydraulic_fields.pore_connectivity_exponent),
        deep_liquid_flux = FreeDrainageFlux(),
        runoff = InfiltrationCapacityRunoff(FT; infiltration_capacity = static.infiltration_capacity))

    hydrology = InterceptingHydrology(FT;
        soil = SurfaceWaterStore(FT; soil, drainage_timescale = 1hour),
        leaf_area_index = static.leaf_area_index,
        capacity_per_leaf_area = 0.2)

    energy = WaterCoupledEnergy(FT;
        dry_heat_capacity = static.dry_heat_capacity,
        liquid_heat_capacity = 4186,
        deep_temperature = static.deep_temperature,
        deep_time_scale = 1day)

    land = SlabLand(grid; energy, hydrology)

    dry_layer_soil = DryLayerHumidity(FT;
        dry_layer_depth = StorageBasedDryLayerDepth(FT; maximum_dry_layer_depth = 0.03,
                                                    dry_layer_onset_saturation = 0.4,
                                                    dry_layer_exponent = 2),
        vapor_exchange = DryLayerVaporPistonVelocity(FT; minimum_dry_layer_depth = 1e-3,
                                                     molecular_diffusivity = 2.4e-5,
                                                     tortuosity = ConstantTortuosity()),
        thermal_exchange_depth = 0.05,
        porosity = static.scalar_porosity)

    canopy = CanopyConductanceHumidity(FT;
        leaf_area_index = aero.tile_leaf_area_index,
        conductance = JarvisConductance(FT),
        moisture_stress = PlantAvailableWaterStress(FT;
            inverse_air_entry_head = static.scalar_α,
            pore_size_uniformity   = static.scalar_n),
        absorbed_par = InteractiveAbsorbedPAR(FT))

    vegetated = CanopyAirSpace(FT;
        soil = dry_layer_soil,
        canopy,
        soil_skin_flux = SoilConductiveFlux(1.5, 0.05),
        undercanopy_conductance = FrictionVelocityUndercanopyConductance(FT),
        inner_iterations = 16,
        interception = CanopyInterception())

    ## Fixed iteration counts keep every GPU thread on the same path; the
    ## tolerance-based solver leaves whole warps spinning on the slowest cell.
    land_roughness = SimilarityTheoryFluxes(FT;
        momentum_roughness_length    = LandRoughnessLength(FT),
        temperature_roughness_length = LandRoughnessLength(FT),
        water_vapor_roughness_length = LandRoughnessLength(FT),
        zero_plane_displacement      = LandZeroPlaneDisplacement(),
        stability_functions          = atmosphere_land_stability_functions(FT),
        solver_stop_criteria         = FixedIterations(8))

    interface = TiledLandInterface(grid, atmosphere, land;
        vegetated,
        fraction = static.vegetation_fraction,
        vegetated_fluxes = land_roughness,
        bare_fluxes      = land_roughness,
        vegetated_surface_properties = merge(aero.vegetated,
                                             (leaf_albedo = static.albedo,
                                              ground_emissivity = static.emissivity)),
        bare_surface_properties      = merge(aero.bare,
                                             (ground_albedo = static.albedo,
                                              ground_emissivity = static.emissivity)))

    correction = ElevationCorrection(static.ground, static.era5_topography; lapse_rate = 6.5e-3)

    return AtmosphereLandModel(atmosphere, land; radiation,
                               atmosphere_land_interface = interface,
                               exchanger_correction = correction)
end

# ## Forcing, initialization, and the runs
#
# Hourly ERA5 over the padded region forces both domains; the initial skin
# temperature and 0–50 cm soil water come from ERA5-Land at the spin-up start.

function run_domain!(name, spec, texture)
    grid   = hiscale_grid(arch, spec.extent, spec.size)
    static = land_surface_fields(grid, spec.footprint_resolution, texture)
    aero   = aerodynamic_fields(grid, static)

    atmosphere = ERA5PrescribedAtmosphere(arch; dataset = ERA5HourlySingleLevel(),
                                          start_date, end_date, region,
                                          surface_layer_height, boundary_layer_height)
    radiation = ERA5PrescribedRadiation(arch; dataset = ERA5HourlySingleLevel(),
                                        start_date, end_date, region,
                                        land_surface = SurfaceRadiationProperties(static.albedo, static.emissivity))

    model = hiscale_land_model(grid, static, aero, atmosphere, radiation)
    land  = model.land

    ## Initial state: ERA5-Land skin temperature lifted to the bare-earth surface,
    ## water storage from the ERA5-Land volumetric soil water over the slab depth.
    Δz = compute!(Field(static.ground - static.era5_topography))
    set!(land.temperature, compute!(Field(static.initial_temperature - 6.5e-3 * Δz)))

    ν  = static.hydraulic_fields.porosity
    θʳ = static.hydraulic_fields.residual_liquid_fraction
    initial_storage = new_surface_field(grid)
    parent(initial_storage) .= clamp.(parent(static.soil_water),
                                      1.05 .* parent(θʳ), 0.95 .* parent(ν)) .*
                               (1000 * static.slab_depth)
    set!(land; M = initial_storage, canopy_water_storage = 0, surface_water_storage = 0)

    simulation = Simulation(model; Δt = 5minutes, stop_time = Dates.value(end_date - start_date) / 1000)

    wall_time = Ref(time_ns())
    function progress(sim)
        interface = sim.model.interfaces.atmosphere_land_interface
        Tmin, Tmax = minimum(land.temperature), maximum(land.temperature)
        𝒮mean = mean(land.saturation)
        LEmax = maximum(interface.fluxes.latent_heat)
        elapsed = 1e-9 * (time_ns() - wall_time[]); wall_time[] = time_ns()
        @info @sprintf("[%s] iter %d t = %s  T %.1f–%.1f K  ⟨𝒮⟩ %.2f  max LE %.0f W m⁻²  wall Δ %.1f s",
                       name, iteration(sim), prettytime(sim), Tmin, Tmax, 𝒮mean, LEmax, elapsed)
        return nothing
    end
    add_callback!(simulation, progress, IterationInterval(72))   ## six-hourly

    interface = model.interfaces.atmosphere_land_interface
    outputs = (T   = land.temperature,
               𝒮   = land.saturation,
               W   = land.water_storage,
               Wᶜ  = land.prognostic.canopy_water_storage,
               Wᵖ  = land.prognostic.surface_water_storage,
               LST = interface.temperature.effective,
               Tᵛ  = interface.vegetated.temperature.canopy,
               Tᵍ  = interface.vegetated.temperature.soil_skin,
               LE  = interface.fluxes.latent_heat,
               H   = interface.fluxes.sensible_heat,
               LEᵛ = interface.vegetated.fluxes.latent_heat,
               LEᵇ = interface.bare.fluxes.latent_heat,
               LEᶜ = interface.vegetated.temperature.canopy_latent_heat,
               LEᵍ = interface.vegetated.temperature.soil_latent_heat,
               Gᶜ  = interface.vegetated.temperature.ground_heat_flux,
               u★  = interface.fluxes.friction_velocity,
               P   = land.fluxes.liquid_precipitation_flux)

    simulation.output_writers[:land] = JLD2Writer(model, outputs;
                                                  filename = "hiscale_$(name)",
                                                  schedule = TimeInterval(1hour),
                                                  array_type = Array{Float32},
                                                  overwrite_existing = true)

    @info "Running the $name domain..."
    run!(simulation)
    close(simulation.output_writers[:land])

    jldsave("hiscale_$(name)_static.jld2";
            ground = on_architecture(CPU(), static.ground),
            dsm = on_architecture(CPU(), static.dsm),
            canopy_height = on_architecture(CPU(), static.canopy_height),
            leaf_area_index = on_architecture(CPU(), static.leaf_area_index),
            vegetation_fraction = on_architecture(CPU(), static.vegetation_fraction),
            water_fraction = on_architecture(CPU(), static.water_fraction),
            built_fraction = on_architecture(CPU(), static.built_fraction),
            porosity = on_architecture(CPU(), static.hydraulic_fields.porosity),
            conductivity = on_architecture(CPU(), static.hydraulic_fields.matching_point_conductivity),
            albedo = on_architecture(CPU(), static.albedo),
            emissivity = on_architecture(CPU(), static.emissivity),
            momentum_roughness = on_architecture(CPU(), aero.vegetated.momentum_roughness_length),
            displacement = on_architecture(CPU(), aero.vegetated.zero_plane_displacement),
            bare_roughness = on_architecture(CPU(), aero.bare.momentum_roughness_length),
            initial_soil_water = on_architecture(CPU(), static.soil_water),
            igbp = static.igbp)
    return nothing
end

for (name, spec) in pairs(domains)
    run_domain!(String(name), spec, domain_texture[name])
    GC.gc(true); CUDA.reclaim()
end

@info "Both domains complete."

# ## What the land model was given
#
# The ingestion maps on the inner 100 m domain. The Cross Timbers oak belt runs up
# the eastern third (canopy heights of 5–15 m, leaf area index 2–4, momentum
# roughness of order 1 m with displacements near two-thirds of the canopy height);
# the winter-wheat belt to the west reads as low late-August leaf area over
# cropland; Ponca City, Blackwell, and Enid surface in the bare-tile roughness
# through the building morphometry; the Cimarron and Arkansas valleys carry the
# sandier, more conductive soils; and Kaw and Sooner lakes are masked out.

static = jldopen("hiscale_inner_100m_static.jld2")
water  = interior(static["water_fraction"], :, :, 1) .> 0.5
mask_water(a) = ifelse.(water, NaN, a)
λ, φ, _ = nodes(static["ground"].grid, Center(), Center(), Center())

panels = (("ground elevation (m)",       interior(static["ground"], :, :, 1),                        :terrain,  nothing),
          ("canopy height (m)",          interior(static["canopy_height"], :, :, 1),                 :speed,    (0, 15)),
          ("leaf area index",            interior(static["leaf_area_index"], :, :, 1),               :algae,    (0, 4)),
          ("vegetation fraction",        interior(static["vegetation_fraction"], :, :, 1),           :speed,    (0, 1)),
          ("canopy roughness ℓᵐ (m)",    interior(static["momentum_roughness"], :, :, 1),            :turbid,   (0, 1.5)),
          ("displacement d (m)",         interior(static["displacement"], :, :, 1),                  :turbid,   (0, 12)),
          ("bare/urban ℓᵐ (log₁₀ m)",    log10.(interior(static["bare_roughness"], :, :, 1)),        :thermal,  (-4, 0)),
          ("porosity ν",                 interior(static["porosity"], :, :, 1),                      :viridis,  (0.3, 0.55)),
          ("matching K₀ (log₁₀ m s⁻¹)",  log10.(interior(static["conductivity"], :, :, 1)),          :turbo,    (-7, -4)),
          ("blue-sky albedo",            interior(static["albedo"], :, :, 1),                        :grays,    (0.1, 0.25)),
          ("broadband emissivity",       interior(static["emissivity"], :, :, 1),                    :balance,  (0.94, 0.99)),
          ("initial soil water θ (ERA5-Land)", interior(static["initial_soil_water"], :, :, 1),      :dense,    (0.05, 0.45)))

fig = Figure(size = (1800, 1500), fontsize = 15)
for (k, (title, data, colormap, colorrange)) in enumerate(panels)
    row, column = fldmod1(k, 4)
    ax = Axis(fig[row, 2column - 1]; title, aspect = DataAspect())
    hidedecorations!(ax)
    hm = isnothing(colorrange) ?
         heatmap!(ax, λ, φ, mask_water(data); colormap, nan_color = :lightsteelblue1) :
         heatmap!(ax, λ, φ, mask_water(data); colormap, colorrange, nan_color = :lightsteelblue1)
    Colorbar(fig[row, 2column], hm)
end
Label(fig[0, 1:8], "HI-SCALE inner domain (120 km at 100 m) — the ingested land surface", fontsize = 20)
save("hiscale_sgp_ingestion.png", fig)
close(static)
nothing #hide

# ![](hiscale_sgp_ingestion.png)

# ## Spin-up and the case-day diurnal cycle
#
# Six days of spin-up settle the slab onto a repeating diurnal envelope; the case
# day is read against the Fast et al. (2019) observations — midafternoon latent
# heat between 250 and 450 W m⁻² around the Central Facility and a skin
# temperature peaking near 305–310 K.

date_of(t) = start_date + Dates.Second(round(Int, t))
hours_since_start(t) = t / 3600

series(name, file) = FieldTimeSeries(file, name; backend = OnDisk())

inner_file = "hiscale_inner_100m.jld2"
T_ts  = series("T", inner_file);  𝒮_ts = series("𝒮", inner_file)
LE_ts = series("LE", inner_file); H_ts = series("H", inner_file)
LST_ts = series("LST", inner_file)
Tᵛ_ts = series("Tᵛ", inner_file); Tᵍ_ts = series("Tᵍ", inner_file)
LEᶜ_ts = series("LEᶜ", inner_file); LEᵍ_ts = series("LEᵍ", inner_file)
Gᶜ_ts = series("Gᶜ", inner_file)
Wᶜ_ts = series("Wᶜ", inner_file); P_ts = series("P", inner_file)
times = T_ts.times

## The grid cell containing the SGP Central Facility.
grid_inner = T_ts.grid
λs, φs, _ = nodes(grid_inner, Center(), Center(), Center())
iᶜᶠ = argmin(abs.(λs .- centre_longitude))
jᶜᶠ = argmin(abs.(φs .- centre_latitude))

at_cf(fts, n) = interior(fts[n], iᶜᶠ, jᶜᶠ, 1)[]
land_mean(fts, n) = mean(mask_water(interior(fts[n], :, :, 1))[.!water])

fig = Figure(size = (1600, 900), fontsize = 15)

ax_spin = Axis(fig[1, 1:2]; title = "spin-up: land mean skin temperature and saturation",
               xlabel = "hours since 24 Aug 12 UTC", ylabel = "LST (K)")
lines!(ax_spin, hours_since_start.(times), [land_mean(LST_ts, n) for n in eachindex(times)]; color = :firebrick)
ax_spin2 = Axis(fig[1, 1:2]; ylabel = "⟨𝒮⟩", yaxisposition = :right)
hidespines!(ax_spin2); hidexdecorations!(ax_spin2)
lines!(ax_spin2, hours_since_start.(times), [land_mean(𝒮_ts, n) for n in eachindex(times)]; color = :navy)
vspan!(ax_spin, 144, 156; color = (:gold, 0.2))   ## the case window

case = findall(t -> case_start <= date_of(t) <= end_date, times)
case_hours = [Dates.value(date_of(times[n]) - case_start) / 3.6e6 + 6 for n in case]   ## clock hours CST

ax_flux = Axis(fig[2, 1]; title = "case day at the Central Facility",
               xlabel = "local time (CST)", ylabel = "flux (W m⁻²)")
lines!(ax_flux, case_hours, [at_cf(LE_ts, n) for n in case]; color = :navy,      label = "latent")
lines!(ax_flux, case_hours, [at_cf(H_ts,  n) for n in case]; color = :orangered, label = "sensible")
lines!(ax_flux, case_hours, [at_cf(Gᶜ_ts, n) for n in case]; color = :seagreen,  label = "ground")
band!(ax_flux, case_hours, 250, 450; color = (:navy, 0.08))   ## observed midafternoon LE range
axislegend(ax_flux; position = :lt)

ax_T = Axis(fig[2, 2]; title = "case-day temperatures at the Central Facility",
            xlabel = "local time (CST)", ylabel = "T (K)")
lines!(ax_T, case_hours, [at_cf(Tᵛ_ts, n) for n in case];  color = :seagreen,  label = "canopy leaf")
lines!(ax_T, case_hours, [at_cf(Tᵍ_ts, n) for n in case];  color = :chocolate, label = "soil skin")
lines!(ax_T, case_hours, [at_cf(LST_ts, n) for n in case]; color = :firebrick, label = "radiative LST")
lines!(ax_T, case_hours, [at_cf(T_ts, n) for n in case];   color = :gray,      label = "bulk slab")
axislegend(ax_T; position = :lt)

save("hiscale_sgp_case_day.png", fig)
nothing #hide

# ![](hiscale_sgp_case_day.png)

# ## The paper's mechanism: soil moisture organizes the fluxes
#
# At 1300 CST on the case day — when Fast et al. found cold pools taking over from
# soil moisture as the main organizer — the Bowen ratio map should mirror the soil
# moisture map: dry columns run sensible-heat-dominated, moist and leafy columns
# latent-dominated.

n1300 = case[argmin(abs.(case_hours .- 13))]

W_map  = mask_water(interior(𝒮_ts[n1300],  :, :, 1))
H_map  = mask_water(interior(H_ts[n1300],  :, :, 1))
LE_map = mask_water(interior(LE_ts[n1300], :, :, 1))
bowen  = H_map ./ max.(LE_map, 10)

fig = Figure(size = (1700, 620), fontsize = 15)
ax1 = Axis(fig[1, 1]; title = "saturation 𝒮 (1300 CST)", aspect = DataAspect())
hm1 = heatmap!(ax1, λs, φs, W_map; colormap = :dense, colorrange = (0, 1), nan_color = :lightsteelblue1)
Colorbar(fig[1, 2], hm1)
ax2 = Axis(fig[1, 3]; title = "Bowen ratio H/LE", aspect = DataAspect())
hm2 = heatmap!(ax2, λs, φs, bowen; colormap = :balance, colorrange = (0, 4), nan_color = :lightsteelblue1)
Colorbar(fig[1, 4], hm2)
ax3 = Axis(fig[1, 5]; title = "latent heat (W m⁻²)", aspect = DataAspect())
hm3 = heatmap!(ax3, λs, φs, LE_map; colormap = :solar, colorrange = (0, 500), nan_color = :lightsteelblue1)
Colorbar(fig[1, 6], hm3)
hidedecorations!.((ax1, ax2, ax3))
Label(fig[0, 1:6], "Soil moisture organizes the surface fluxes — 30 Aug 2016, 1300 CST", fontsize = 18)
save("hiscale_sgp_mechanism.png", fig)
nothing #hide

# ![](hiscale_sgp_mechanism.png)

# ## Case-day animation

fig = Figure(size = (1700, 950), fontsize = 14)
n = Observable(first(case))
LSTn = @lift mask_water(interior(LST_ts[$n], :, :, 1))
𝒮n   = @lift mask_water(interior(𝒮_ts[$n], :, :, 1))
LEn  = @lift mask_water(interior(LE_ts[$n], :, :, 1))
Hn   = @lift mask_water(interior(H_ts[$n], :, :, 1))

for (k, (title, obs, colormap, colorrange)) in enumerate(
        (("radiative LST (K)", LSTn, :thermal, (285, 320)),
         ("saturation 𝒮", 𝒮n, :dense, (0, 1)),
         ("latent heat (W m⁻²)", LEn, :solar, (0, 500)),
         ("sensible heat (W m⁻²)", Hn, :lajolla, (0, 400))))
    ax = Axis(fig[1, 2k - 1]; title, aspect = DataAspect())
    hidedecorations!(ax)
    hm = heatmap!(ax, λs, φs, obs; colormap, colorrange, nan_color = :lightsteelblue1)
    Colorbar(fig[1, 2k], hm)
end
label = Label(fig[0, 1:8], ""; fontsize = 18)

CairoMakie.record(fig, "hiscale_sgp_case_day.mp4", case; framerate = 4) do nn
    n[] = nn
    label.text = "HI-SCALE inner domain — " * Dates.format(date_of(times[nn]) - Dates.Hour(6), "dd u yyyy HH:MM") * " CST"
end
nothing #hide

# ![](hiscale_sgp_case_day.mp4)

# ## 100 m against 300 m
#
# The same land model at the two paper resolutions, over the inner box at 1300 CST:
# the 10–30 m datasets (canopy, buildings, soil texture) survive at 100 m as flux
# structure the 300 m domain smooths.

outer_file = "hiscale_outer_300m.jld2"
LE_outer = FieldTimeSeries(outer_file, "LE"; backend = OnDisk())
λo, φo, _ = nodes(LE_outer.grid, Center(), Center(), Center())
inner_window_i = findall(λ -> minimum(λs) <= λ <= maximum(λs), λo)
inner_window_j = findall(φ -> minimum(φs) <= φ <= maximum(φs), φo)

static_outer = jldopen("hiscale_outer_300m_static.jld2")
water_outer = interior(static_outer["water_fraction"], inner_window_i, inner_window_j, 1) .> 0.5
close(static_outer)

fig = Figure(size = (1400, 700), fontsize = 15)
axi = Axis(fig[1, 1]; title = "latent heat at 100 m", aspect = DataAspect())
hmi = heatmap!(axi, λs, φs, mask_water(interior(LE_ts[n1300], :, :, 1));
               colormap = :solar, colorrange = (0, 500), nan_color = :lightsteelblue1)
axo = Axis(fig[1, 2]; title = "latent heat at 300 m", aspect = DataAspect())
heatmap!(axo, λo[inner_window_i], φo[inner_window_j],
         ifelse.(water_outer, NaN, interior(LE_outer[n1300], inner_window_i, inner_window_j, 1));
         colormap = :solar, colorrange = (0, 500), nan_color = :lightsteelblue1)
Colorbar(fig[1, 3], hmi)
hidedecorations!.((axi, axo))
save("hiscale_sgp_resolutions.png", fig)
nothing #hide

# ![](hiscale_sgp_resolutions.png)
