# # ERA5 → CONUS convection-permitting hindcast with a tiled canopy land surface
#
# The 20 May 2011 MC3E squall-line case of `run12km_land.jl` — an ERA5-driven Breeze
# limited-area atmosphere on the HRRR CONUS footprint — over the vegetated land stack,
# with every surface parameter ingested from a high-resolution product read at the
# resolution the grid resolves (`ingest_surface.jl`, cached per product):
#
#   soil        OpenLandMap-soilDB 30 m texture, read from the COG overview matching the
#               grid and regridded in tiles → Weynants van Genuchten parameters per cell
#   land cover  MODIS MCD12Q1 class fractions (conservative) and the majority vegetated
#               class; optionally ESA WorldCover 10 m fractions (`LANDCOVER=worldcover`)
#   leaf area   MODIS MCD15A2H composites around the case day, gap-filled
#   urban       GHSL building height + built fraction → morphometric roughness at ~1 km,
#               log-averaged over the built pixels of each cell
#   radiation   Copernicus Global Land blue-sky albedo, ASTER GED broadband emissivity
#   soil state  ERA5-Land monthly soil water and deep soil temperature
#
# Roughness comes from the canopy (Raupach drag partition on the tile leaf area) and from
# the urban morphometry; the zero-plane displacement is zero on both tiles, since the
# atmosphere's terrain is a surface model that already stands on the canopy and roofs.
#
# The land surface radiation budget is prescribed from ERA5 downwelling fluxes and the
# atmosphere itself runs without radiative transfer, as in `run12km_land.jl`. There is no
# land-sea mask (NumericalEarth#450): the slab covers the ocean as a saturated,
# SST-restoring, water-roughness surface.
#
# `REFINEMENT` scales resolution at fixed extent: 1 → ~12 km, 2 → ~6 km.
# `CANOPY_HEIGHT=eth` replaces the IGBP class canopy heights by the ETH Sentinel-2 10 m
# product (area-averaged per cell, class height as the floor where it sees no trees);
# `CANOPY_HEIGHT=eth_trees` does so only where the MODIS class is a tree type.
# `LAND=bucket` runs the pre-vegetation bucket `SlabLand` (Manabe evaporation efficiency,
# constant roughness) in place of the tiled canopy; `RADIATION=rtm` runs all-sky RRTMGP in the
# nest in place of the ERA5 prescribed fluxes; `STOP_DATE` extends the window (default 21 May 00Z).
# `INGEST_ONLY=1` builds the surface caches on a CPU node and exits.
# `SMOKE=1` stops after 20 simulated minutes to exercise the full pipeline first.

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: on_architecture
using Breeze
using CopernicusClimateDataStore   # ERA5 + ERA5-Land downloads
using CloudMicrophysics            # nested default microphysics → 1-moment mixed-phase
using ArchGDAL                     # COG / GeoTIFF / HDF readers for the surface products
using CairoMakie                   # NumericalEarthMakieExt (visualize_nested_domain)
using NaturalEarth                 # NumericalEarthNaturalEarthExt (state/country lines)
using CUDA
using RRTMGP                       # Breeze RadiativeTransferModel (RADIATION=rtm)
using JLD2
using Printf
using Statistics: mean, median
import Dates
import Dates: DateTime

using NumericalEarth.EarthSystemModels.InterfaceComputations:
    atmosphere_land_stability_functions,
    FixedIterations   # also exported by Breeze.Solvers — qualify to break the ambiguity

Oceananigans.defaults.FloatType = Float32
FT = Float32

include(joinpath(@__DIR__, "ingest_surface.jl"))

# ## Configuration

SMOKE = get(ENV, "SMOKE", "0") == "1"
INGEST_ONLY = get(ENV, "INGEST_ONLY", "0") == "1"
refinement = parse(Int, get(ENV, "REFINEMENT", "1"))
landcover_source = Symbol(get(ENV, "LANDCOVER", "modis"))
canopy_height_source = Symbol(get(ENV, "CANOPY_HEIGHT", "class"))
land_source = Symbol(get(ENV, "LAND", "canopy"))
radiation_source = Symbol(get(ENV, "RADIATION", "prescribed"))

arch = INGEST_ONLY ? CPU() : GPU(CUDA.CUDABackend(always_inline = true))

kilometers_per_degree = 111.32
resolution_km = round(Int, kilometers_per_degree / (9 * refinement))
resolution_tag = get(ENV, "TAG", "conus$(resolution_km)km_v2")

φ₀, λ₀ = 37.0, -97.5                        # HRRR CONUS center
start_date = DateTime(2011, 5, 17, 0)       # three diurnal cycles of spin-up...
stop_date  = DateTime(get(ENV, "STOP_DATE", "2011-05-21T00:00:00"))   # ...then the MC3E squall-line day
dates = (start_date, stop_date)

Δλ = Δφ = 1 / (9 * refinement)
Nx, Ny = 657 * refinement, 288 * refinement # HRRR footprint: λ ∈ [-134°, -61°], φ ∈ [21°, 53°]

λ_west,  λ_east  = λ₀ - Nx * Δλ / 2, λ₀ + Nx * Δλ / 2
φ_south, φ_north = φ₀ - Ny * Δφ / 2, φ₀ + Ny * Δφ / 2

# The datasets are windowed on the domain extent, which is refinement-invariant, so
# every resolution shares one set of downloaded files.
ingest_region = BoundingBox(longitude = (λ_west - 0.2, λ_east + 0.2),
                            latitude  = (φ_south - 0.2, φ_north + 0.2))

# ERA5 single-level forcing (radiation, initial skin temperature) is requested on the
# same padded box as the nest's pressure-level files, so one prefetched cache serves all.
era5_forcing_region = BoundingBox(longitude = (λ_west - 0.5, λ_east + 0.5),
                                  latitude  = (φ_south - 0.5, φ_north + 0.5))
era5_datadir = "era5"

# Davies relaxation and terrain smoothing are physical widths, held fixed under refinement.
relax_width = 5 * refinement
terrain_smoothing_passes = 2 * refinement^2

land_grid = LatitudeLongitudeGrid(arch;
                                  longitude = (λ_west,  λ_east),
                                  latitude  = (φ_south, φ_north),
                                  size = (Nx, Ny),
                                  halo = (5, 5),
                                  topology = (Bounded, Bounded, Flat))

cpu_land_grid = on_architecture(CPU(), land_grid)

# ## Surface-field ingestion (cached per product and resolution)

cache_directory = "surface_cache"
cache_file(name) = joinpath(cache_directory, "$(name)_r$(refinement).jld2")
slab_depth = 0.5

soil = cached(cache_file("soil")) do
    ingest_soil(cpu_land_grid, ingest_region; slab_depth)
end
modis = cached(cache_file("modis")) do
    ingest_modis(cpu_land_grid, ingest_region, joinpath(cache_directory, "modis_lai_fill.jld2"))
end
urban = cached(cache_file("urban")) do
    ingest_urban(cpu_land_grid)
end
optics = cached(cache_file("optics")) do
    merge(ingest_albedo(cpu_land_grid, ingest_region), ingest_emissivity(cpu_land_grid, ingest_region))
end
era5_land = cached(cache_file("era5_land")) do
    ingest_era5_land(cpu_land_grid, ingest_region, era5_datadir)
end
worldcover = landcover_source == :worldcover ? cached(cache_file("worldcover")) do
    ingest_worldcover(cpu_land_grid, ingest_region)
end : nothing
eth = canopy_height_source in (:eth, :eth_trees) ? cached(cache_file("eth_canopy")) do
    ingest_canopy_height(cpu_land_grid, ingest_region)
end : nothing

if INGEST_ONLY
    @info "Surface caches written to $(abspath(cache_directory)); exiting."
    exit(0)
end

# ## Derived surface fields
#
# Land-cover fractions (vegetated, water, built-up) come from WorldCover when requested and
# from MODIS otherwise; the canopy type is always the majority vegetated MODIS class.

array(field) = Array(interior(field, :, :, 1))

vegetated_igbp_classes = (:evergreen_needleleaf_forest, :evergreen_broadleaf_forest,
                          :deciduous_needleleaf_forest, :deciduous_broadleaf_forest,
                          :mixed_forest, :closed_shrubland, :open_shrubland,
                          :woody_savanna, :savanna, :grassland, :permanent_wetland,
                          :cropland, :cropland_natural_mosaic)

# The roughness tables name two classes differently from the IGBP legend.
roughness_class(class) = class == :cropland_natural_mosaic ? :cropland_vegetation_mosaic :
                         class == :permanent_snow_and_ice  ? :snow_and_ice : class

leaf_area_index = array(modis.leaf_area_index)

if isnothing(worldcover)
    vegetated_cover = sum(array(modis.fractions[class]) for class in vegetated_igbp_classes)
    water_cover = array(modis.fractions.water)
    urban_cover = array(modis.fractions.urban)
else
    ## WorldCover maps land only: the unmapped share of a cell (no-data pixels inside a
    ## tile, or a 3° cell with no published tile) is sea.
    mapped = sum(array(worldcover[class]) for class in keys(worldcover) if class != :vegetation_fraction)
    vegetated_cover = array(worldcover.vegetation_fraction)
    water_cover = array(worldcover.permanent_water_bodies) .+ (1 .- mapped)
    urban_cover = array(worldcover.built_up)
    unmapped = .!isfinite.(mapped)
    vegetated_cover[unmapped] .= 0; water_cover[unmapped] .= 1; urban_cover[unmapped] .= 0
end
water = water_cover .> 0.5

vegetation_fraction = FT.(ifelse.(leaf_area_index .> 0.1, vegetated_cover, 0))
vegetation_fraction[water] .= 0
tile_lai = FT.(clamp.(leaf_area_index ./ max.(vegetation_fraction, 0.05), 0.1, 8))

vegetated_stack = cat((array(modis.fractions[class]) for class in vegetated_igbp_classes)...; dims = 3)
canopy_class = [vegetated_igbp_classes[argmax(view(vegetated_stack, i, j, :))]
                for i in axes(vegetated_stack, 1), j in axes(vegetated_stack, 2)]
class_canopy_height = [representative_canopy_height(FT, roughness_class(class)) for class in canopy_class]

# With the measured ETH height, the class height stays as the floor where the product sees
# no trees (crops, grass, shrubs), so herbaceous cover keeps its roughness. `eth_trees` trusts
# the product only where the MODIS class is a tree type, since it reads several meters over
# crops and grass.
tree_classes = (:evergreen_needleleaf_forest, :evergreen_broadleaf_forest, :deciduous_needleleaf_forest,
                :deciduous_broadleaf_forest, :mixed_forest, :woody_savanna, :savanna, :permanent_wetland)
if isnothing(eth)
    canopy_height = class_canopy_height
else
    eth_canopy_height = array(eth.eth_canopy_height)
    measured = FT.(max.(ifelse.(isfinite.(eth_canopy_height), eth_canopy_height, 0), min.(class_canopy_height, 1.6)))
    trees = map(class -> class in tree_classes, canopy_class)
    canopy_height = canopy_height_source == :eth_trees ? ifelse.(trees, measured, class_canopy_height) : measured
end

# Vegetated tile: Raupach drag-partition roughness of each cell's canopy type at the tile
# leaf area and the class's representative height (displacement discarded: d = 0).
cpu_tile_lai = surface_field(cpu_land_grid); set!(cpu_tile_lai, tile_lai)
cpu_canopy_height = surface_field(cpu_land_grid); set!(cpu_canopy_height, canopy_height)
vegetated_roughness_length = fill(FT(0.03), Nx, Ny)
for class in unique(canopy_class)
    class_roughness_length, _ = canopy_roughness(DragPartitionRoughness(FT; vegetation_type = roughness_class(class)),
                                   cpu_tile_lai, cpu_canopy_height)
    cells = canopy_class .== class
    vegetated_roughness_length[cells] .= array(class_roughness_length)[cells]
end
replace!(vegetated_roughness_length, NaN => 0.03)
clamp!(vegetated_roughness_length, 1e-4, 3)

# Bare tile: log-mean of the urban (GHSL morphometric), open-water and bare-soil roughness,
# weighted by each surface's share of the non-vegetated area.
default_urban_roughness_length, _ = nonvegetated_roughness(FT, :urban)
water_roughness_length, _ = nonvegetated_roughness(FT, :water)
soil_roughness_length, _ = nonvegetated_roughness(FT, :barren)
urban_roughness_length = array(urban.urban_roughness)
urban_roughness_length .= ifelse.(isfinite.(urban_roughness_length), urban_roughness_length, default_urban_roughness_length)
nonvegetated = max.(1 .- vegetation_fraction, 0.01)
urban_weight = min.(urban_cover, nonvegetated)
water_weight = min.(water_cover, nonvegetated .- urban_weight)
soil_weight  = nonvegetated .- urban_weight .- water_weight
bare_roughness_length = exp.((urban_weight .* log.(urban_roughness_length) .+ water_weight .* log(water_roughness_length) .+ soil_weight .* log(soil_roughness_length)) ./ nonvegetated)

# Radiative properties: satellite values over land, open-water constants where the cell is
# water, land medians in the remaining gaps.
albedo = array(optics.albedo)
albedo[water] .= 0.07
replace!(albedo, NaN => median(filter(isfinite, albedo[.!water])))
clamp!(albedo, 0.03, 0.6)
emissivity = array(optics.emissivity)
emissivity[water] .= 0.98
replace!(emissivity, NaN => median(filter(isfinite, emissivity[.!water])))
clamp!(emissivity, 0.9, 1)

# Every per-cell parameter the model reads lives on the (GPU) land grid.
to_field(a) = (f = surface_field(land_grid); set!(f, a); f)
hydraulics = map(f -> to_field(array(f)), soil.hydraulic_fields)
static = (; leaf_area_index = to_field(leaf_area_index),
            tile_lai = to_field(tile_lai),
            vegetation_fraction = to_field(vegetation_fraction),
            canopy_height = to_field(canopy_height),
            vegetated_roughness_length = to_field(vegetated_roughness_length),
            vegetated_scalar_roughness_length = to_field(vegetated_roughness_length ./ 10),
            bare_roughness_length = to_field(bare_roughness_length),
            bare_scalar_roughness_length = to_field(bare_roughness_length ./ 10),
            albedo = to_field(albedo),
            emissivity = to_field(emissivity),
            dry_heat_capacity = to_field(array(soil.dry_heat_capacity)),
            deep_temperature = to_field(array(era5_land.deep_temperature)))

# ## The land model
#
# `LAND=bucket` is the pre-vegetation configuration of the first CONUS radiation runs: the
# default bucket `SlabLand` with Manabe evaporation efficiency and constant roughness.

if land_source == :canopy

soil_hydrology = VariablySaturatedHydrology(FT;
    slab_depth,
    storage_height = 1000,
    porosity = hydraulics.porosity,
    residual_liquid_fraction = hydraulics.residual_liquid_fraction,
    retention_curve = VanGenuchtenRetention(FT;
        inverse_air_entry_head = hydraulics.inverse_air_entry_head,
        pore_size_uniformity   = hydraulics.pore_size_uniformity),
    hydraulic_conductivity = VanGenuchtenConductivity(FT;
        matching_point_conductivity = hydraulics.matching_point_conductivity,
        pore_size_uniformity        = hydraulics.pore_size_uniformity,
        pore_connectivity_exponent  = hydraulics.pore_connectivity_exponent),
    deep_liquid_flux = FreeDrainageFlux(),
    runoff = InfiltrationCapacityRunoff(FT; infiltration_capacity = soil.infiltration_capacity))

hydrology = InterceptingHydrology(FT;
    soil = SurfaceWaterStore(FT; soil = soil_hydrology, drainage_timescale = 1hour),
    leaf_area_index = static.leaf_area_index,
    capacity_per_leaf_area = 0.2)

energy = WaterCoupledEnergy(FT;
    dry_heat_capacity = static.dry_heat_capacity,
    liquid_heat_capacity = 4186,
    deep_temperature = static.deep_temperature,
    deep_time_scale = 1day)

land = SlabLand(land_grid; energy, hydrology)

# The dry surface layer grows from saturation down (Swenson & Lawrence 2014): with an
# onset below one, moist soil evaporates as an unresisted free-water surface, cools
# below the air, and its downward sensible heat cancels the canopy's upward flux.
dry_layer_soil = DryLayerHumidity(FT;
    dry_layer_depth = StorageBasedDryLayerDepth(FT; maximum_dry_layer_depth = 0.05,
                                                dry_layer_onset_saturation = 1.0,
                                                dry_layer_exponent = 2),
    vapor_exchange = DryLayerVaporPistonVelocity(FT; minimum_dry_layer_depth = 1e-3,
                                                 molecular_diffusivity = 2.4e-5,
                                                 tortuosity = PowerLawTortuosity()),
    thermal_exchange_depth = 0.05,
    porosity = soil.scalar_porosity)

# The stress endpoints are evaluated per cell on the land's own retention curve, so the
# stress and the soil water share one saturation frame.
canopy = CanopyConductanceHumidity(FT;
    leaf_area_index = static.tile_lai,
    conductance = JarvisConductance(FT),
    moisture_stress = PlantAvailableWaterStress(FT),
    absorbed_par = InteractiveAbsorbedPAR(FT))

vegetated = CanopyAirSpace(FT;
    soil = dry_layer_soil,
    canopy,
    soil_skin_flux = SoilConductiveFlux(FT(1.5), FT(0.05)),
    undercanopy_conductance = FrictionVelocityUndercanopyConductance(FT),
    inner_iterations = 16,
    interception = CanopyInterception(),
    storage = PrognosticCanopyAir(layer_depth = static.canopy_height),
    leaf_albedo = static.albedo,
    ground_albedo = static.albedo,
    ground_emissivity = static.emissivity)

# Fixed iteration counts keep every GPU thread on the same path; a tolerance-based
# solver leaves whole warps spinning on the slowest cell.
land_fluxes(ℓᵐ, ℓˢ) = SimilarityTheoryFluxes(FT;
    momentum_roughness_length    = ℓᵐ,
    temperature_roughness_length = ℓˢ,
    water_vapor_roughness_length = ℓˢ,
    stability_functions          = atmosphere_land_stability_functions(FT),
    solver_stop_criteria         = FixedIterations(8))

else
    land = SlabLand(land_grid)
end

# ## Atmosphere grid and nest

z = ReferenceToStretchedDiscretization(extent = 19525.0,
                                       bias = :left,
                                       bias_edge = 0.0,
                                       constant_spacing = 60.0,
                                       constant_spacing_extent = 60.0,
                                       maximum_spacing = 490.0,
                                       stretching = LinearStretching(0.15))
Nz = length(z)

@info @sprintf("grid: %d × %d × %d = %.1fM cells at Δλ = 1/%d° (≈ %d km)",
               Nx, Ny, Nz, Nx * Ny * Nz / 1e6, 9 * refinement, resolution_km)

grid = LatitudeLongitudeGrid(arch;
                             longitude = (λ_west,  λ_east),
                             latitude  = (φ_south, φ_north),
                             z = TerrainFollowingVerticalDiscretization(z),
                             size = (Nx, Ny, Nz),
                             halo = (5, 5, 5),
                             topology = (Bounded, Bounded, Bounded))

# WENO(5) with explicit vertical advection for momentum and every scalar — the one
# combination validated at both 12 and 3 km in this configuration.
reconstruction(; kw...) = WENO(; order = 5, kw...)
momentum_advection = reconstruction()
scalar_advection = (ρθ  = reconstruction(),
                    ρqᵉ = reconstruction(bounds = (0, 1)),
                    ρqʳ = reconstruction(bounds = (0, 1)),
                    ρqˢ = reconstruction(bounds = (0, 1)))

Δt = 1.0 / refinement

nest = nested_atmosphere_model(grid, ERA5HourlyPressureLevels();
                               dates,
                               dir = era5_datadir,
                               parent_time_indices_in_memory = 4,
                               terrain = ETOPO2022(),
                               terrain_smoothing_passes,
                               relaxation_rate = 1/300,
                               relaxation_width = relax_width,
                               momentum_advection,
                               scalar_advection)

era5_region = BoundingBox(nest.parent.grid)

fig = visualize_nested_domain(grid;
                              parent       = era5_region,
                              padding      = 2.5,
                              resolution   = 1/10,
                              title        = "ERA5 → $(resolution_km) km CONUS LAM + tiled canopy land (MC3E, 17–21 May 2011)",
                              label        = "$(resolution_km) km CONUS LAM (child)",
                              parent_label = "ERA5 parent",
                              landmarks    = tuple("ARM SGP" => (-97.485, 36.605)))
save("$(resolution_tag)_domains.png", fig)

# ## Radiation and coupling

# Prescribed: ERA5 downwelling fluxes drive the land, the atmosphere has no radiative transfer.
# RTM: all-sky RRTMGP on the child grid, hourly solves, interior heating on the nest's own clouds;
# the land interface binds its skin temperature and (for the canopy) its effective albedo.
radiation = if radiation_source == :rtm
    RadiativeTransferModel(grid, AllSkyOptics(), nest.child.thermodynamic_constants;
                           solar_position = ApparentSolarPosition(epoch = start_date),
                           surface_albedo = CopernicusAlbedo(),
                           schedule = TimeInterval(1hour))
else
    ERA5PrescribedRadiation(arch;
                            dataset = ERA5HourlySingleLevel(),
                            start_date, end_date = stop_date,
                            region = era5_forcing_region,
                            dir = era5_datadir,
                            land_surface = SurfaceRadiationProperties(static.albedo, static.emissivity),
                            ocean_surface = nothing, sea_ice_surface = nothing)
end

atmosphere = Simulation(nest; Δt)   ## the coupled model manages Δt; this sets the initial value

interface_kw = if land_source == :canopy
    (; atmosphere_land_interface = TiledLandInterface(land_grid, atmosphere, land;
           vegetated,
           fraction = static.vegetation_fraction,
           vegetated_fluxes = land_fluxes(static.vegetated_roughness_length, static.vegetated_scalar_roughness_length),
           bare_fluxes      = land_fluxes(static.bare_roughness_length, static.bare_scalar_roughness_length)))
else
    (; atmosphere_land_interface_specific_humidity = FractionalHumidity(efficiency = CriticalSaturation(0.75)))
end

# ## Initial state
#
# Skin temperature from ERA5 (SST over the ocean); soil water from ERA5-Land clamped
# inside the pedotransfer pore space; open water saturated.

skin_temperature = Metadatum(:skin_temperature; dataset = ERA5HourlySingleLevel(),
                             date = start_date, region = era5_forcing_region, dir = era5_datadir)
set!(land.temperature, skin_temperature)

porosity = array(soil.hydraulic_fields.porosity)
residual = array(soil.hydraulic_fields.residual_liquid_fraction)
θ₀ = clamp.(array(era5_land.soil_water), 1.05 .* residual, 0.95 .* porosity)
θ₀[water] .= porosity[water]
if land_source == :canopy
    set!(land; M = to_field(θ₀ .* (1000 * slab_depth)), canopy_water_storage = 0, surface_water_storage = 0)
else
    ## The bucket starts from the same ERA5-Land soil water as the canopy runs, relative to a
    ## 0.45 m³ m⁻³ saturation; water cells start saturated.
    capacity = land.hydrology.maximum_water_storage
    set!(land; M = to_field(FT.(ifelse.(water, capacity, clamp.(array(era5_land.soil_water) ./ 0.45, 0, 1) .* capacity))))
end

@info @sprintf("initial soil wetness 𝒮 ∈ [%.3f, %.3f], mean %.3f",
               minimum(land.saturation), maximum(land.saturation), mean(land.saturation))

model = AtmosphereLandModel(atmosphere, land; radiation, interface_kw...)

# Materializing the radiation rebuilds the nest around the child; diagnose against the objects
# the simulation steps.
child = model.atmosphere.model.child
interface = model.interfaces.atmosphere_land_interface

stop_time = parse(Float64, get(ENV, "STOP_TIME", SMOKE ? "1200" : string(Dates.value(stop_date - start_date) / 1000)))
simulation = Simulation(model; Δt, stop_time)

conjure_time_step_wizard!(simulation, IterationInterval(1); cfl = 0.7,
                          max_Δt = NumericalEarth.Atmospheres.estimate_maximum_Δt(grid))

# ## Output
#
# 2-D slices only. On the terrain-following grid a constant reference level ≈ constant
# height above ground near the surface, so `k_aloft` is the level nearest 2 km.

k_aloft = searchsortedfirst(Array(znodes(grid, Center())), 2000)
schedule = TimeInterval(15minutes)

include(joinpath(@__DIR__, "pressure_diagnostics.jl"))

surface_pressure = surface_pressure_field(child)
qᵛ = specific_humidity(child)

surface_fields = (θᵥ = VirtualPotentialTemperature(child),
                  U  = sqrt(child.velocities.u^2 + child.velocities.v^2),
                  qᵛ = qᵛ,
                  pₛ = surface_pressure)
if radiation_source == :rtm   ## ZFaceFields: the (:, :, 1) slice is the surface face, positive up
    surface_fields = merge(surface_fields, (ℐꜛˡʷ = model.radiation.upwelling_longwave_flux,
                                            ℐꜛˢʷ = model.radiation.upwelling_shortwave_flux))
end

aloft_fields = (w  = child.velocities.w,
                qᵛ = qᵛ,
                qʳ = child.microphysical_fields.qʳ)

simulation.output_writers[:surface] = JLD2Writer(model, surface_fields; schedule,
                                                 filename = "$(resolution_tag)_surface.jld2",
                                                 indices = (:, :, 1),
                                                 array_type = Array{Float32},
                                                 overwrite_existing = true)

simulation.output_writers[:aloft] = JLD2Writer(model, aloft_fields; schedule,
                                               filename = "$(resolution_tag)_aloft.jld2",
                                               indices = (:, :, k_aloft),
                                               array_type = Array{Float32},
                                               overwrite_existing = true)

# Land + canopy diagnostics: the blended fluxes the atmosphere sees, the two-source
# temperatures, the per-branch latent heat, and every water reservoir.
land_fields = if land_source == :bucket
    (Tˡᵃ = land.temperature,
     𝒮   = land.saturation,
     W   = land.water_storage,
     LE  = interface.fluxes.latent_heat,
     H   = interface.fluxes.sensible_heat,
     u★  = interface.fluxes.friction_velocity,
     Jʳⁿ = model.interfaces.exchanger.atmosphere.state.Jʳⁿ,
     ℐꜜˢʷ = model.interfaces.exchanger.radiation.state.ℐꜜˢʷ,
     ℐꜜˡʷ = model.interfaces.exchanger.radiation.state.ℐꜜˡʷ)
else
    (Tˡᵃ = land.temperature,
               𝒮   = land.saturation,
               W   = land.water_storage,
               Wᶜ  = land.prognostic.canopy_water_storage,
               Wᵖ  = land.prognostic.surface_water_storage,
               LST = interface.temperature.effective,
               αᵉᶠᶠ = interface.temperature.effective_albedo,
               Tᵃᶜ = interface.temperature.interface,
               Tᵛ  = interface.vegetated.temperature.canopy,
               Tᵍ  = interface.vegetated.temperature.soil_skin,
               LE  = interface.fluxes.latent_heat,
               H   = interface.fluxes.sensible_heat,
               u★  = interface.fluxes.friction_velocity,
               LEᵛ = interface.vegetated.fluxes.latent_heat,
               LEᵇ = interface.bare.fluxes.latent_heat,
               LEᶜ = interface.vegetated.temperature.canopy_latent_heat,
               LEᵍ = interface.vegetated.temperature.soil_latent_heat,
               Eʷ  = interface.vegetated.temperature.canopy_evaporation,
               Gᶜ  = interface.vegetated.temperature.ground_heat_flux,
               P   = land.fluxes.liquid_precipitation_flux,
               E   = land.fluxes.vapor_flux,
               R   = land.diagnostics.surface_water_runoff,
               D   = land.diagnostics.deep_liquid_flux,
               Jʳⁿ = model.interfaces.exchanger.atmosphere.state.Jʳⁿ,
               ℐꜜˢʷ = model.interfaces.exchanger.radiation.state.ℐꜜˢʷ,
               ℐꜜˡʷ = model.interfaces.exchanger.radiation.state.ℐꜜˡʷ)
end

simulation.output_writers[:land] = JLD2Writer(model, land_fields; schedule,
                                              filename = "$(resolution_tag)_land.jld2",
                                              array_type = Array{Float32},
                                              overwrite_existing = true)

# ## Isobaric diagnostics (u, v, w, T, mixing ratio, geopotential height)

geopotential_height = geopotential_height_field(grid)

isobaric_variables = (u = @at((Center, Center, Center), child.velocities.u),
                      v = @at((Center, Center, Center), child.velocities.v),
                      w = @at((Center, Center, Center), child.velocities.w),
                      T = child.temperature,
                      r = qᵛ / (1 - qᵛ),
                      Z = geopotential_height)

pressure_levels_hPa = (1000, 850, 700, 500, 300, 200)
isobaric = PressureLevelDiagnostics(child, pressure_levels_hPa, isobaric_variables)

function fill_diagnostics!(sim)
    fill_surface_pressure!(surface_pressure, child)
    fill_pressure_levels!(isobaric, child)
    return nothing
end
add_callback!(simulation, fill_diagnostics!, schedule)

simulation.output_writers[:isobaric] = JLD2Writer(model, isobaric.slices; schedule,
                                                  filename = "$(resolution_tag)_isobaric.jld2",
                                                  array_type = Array{Float32},
                                                  overwrite_existing = true)

# Static surface fields, saved once for the figure scripts.
jldsave("$(resolution_tag)_static.jld2";
        leaf_area_index, tile_lai, vegetation_fraction, canopy_height, canopy_class, class_canopy_height,
        eth_canopy_height = isnothing(eth) ? nothing : array(eth.eth_canopy_height),
        tall_canopy_fraction = isnothing(eth) ? nothing : array(eth.tall_canopy_fraction),
        vegetated_cover, water_cover, urban_cover, water,
        momentum_roughness_vegetated = vegetated_roughness_length, momentum_roughness_bare = bare_roughness_length,
        urban_roughness = array(urban.urban_roughness),
        urban_fraction = array(urban.urban_fraction),
        building_height = array(urban.building_height),
        plan_area_index = array(urban.plan_area_index),
        albedo, emissivity,
        porosity, residual_liquid_fraction = residual,
        matching_point_conductivity = array(soil.hydraulic_fields.matching_point_conductivity),
        inverse_air_entry_head = array(soil.hydraulic_fields.inverse_air_entry_head),
        pore_size_uniformity = array(soil.hydraulic_fields.pore_size_uniformity),
        sand = array(soil.texture_fields.sand), clay = array(soil.texture_fields.clay),
        bulk_density = array(soil.texture_fields.bulk_density),
        dry_heat_capacity = array(soil.dry_heat_capacity),
        initial_soil_water = θ₀,
        deep_temperature = array(era5_land.deep_temperature),
        landcover_source = String(landcover_source),
        longitude = Array(λnodes(cpu_land_grid, Center())),
        latitude = Array(φnodes(cpu_land_grid, Center())))

# ## Progress

wall_time = Ref(time_ns())
function progress(sim)
    child = sim.model.atmosphere.model.child
    u, v, w = child.velocities
    ρ  = child.dynamics.total_density
    qʳ = child.microphysical_fields.qʳ
    land = sim.model.land
    fluxes = sim.model.interfaces.atmosphere_land_interface.fluxes
    elapsed = 1e-9 * (time_ns() - wall_time[]); wall_time[] = time_ns()
    @info @sprintf("iter=%5d t=%8.1fs Δt=%5.2f  max|u|=%6.1f max|w|=%5.1f  ρ∈[%.3f,%.3f] qʳ≤%.2g  Tˡᵃ∈[%.1f,%.1f] ⟨𝒮⟩=%.3f  LE∈[%.0f,%.0f] H∈[%.0f,%.0f]  wall Δ %.1fs",
                   sim.model.clock.iteration, sim.model.clock.time, sim.Δt,
                   maximum(abs, u), maximum(abs, w), minimum(ρ), maximum(ρ), maximum(qʳ),
                   minimum(land.temperature), maximum(land.temperature), mean(land.saturation),
                   minimum(fluxes.latent_heat), maximum(fluxes.latent_heat),
                   minimum(fluxes.sensible_heat), maximum(fluxes.sensible_heat), elapsed)
    return nothing
end
add_callback!(simulation, progress, IterationInterval(50))

# ## Run

run!(simulation)

@info "Run complete: $(prettytime(simulation.model.clock.time)) simulated."
