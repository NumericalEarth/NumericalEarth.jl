# # ERA5 → CONUS convection-permitting hindcast with a tiled canopy land surface
#
# The 20 May 2011 MC3E squall-line case of `run12km_land.jl` — an ERA5-driven Breeze
# limited-area atmosphere on the HRRR CONUS footprint — with the bucket `SlabLand`
# replaced by the vegetated land stack of the slab-canopy branch, every surface
# parameter ingested from a measured dataset (the HI-SCALE SGP recipe at CONUS scale):
#
#   hydrology  VariablySaturatedHydrology (per-cell pedotransfer van Genuchten fields
#              from SoilGrids 2.0 texture) inside SurfaceWaterStore (ponds rejected
#              infiltration) inside InterceptingHydrology (prognostic canopy water)
#   energy     WaterCoupledEnergy, per-cell dry heat capacity from bulk density,
#              restoring to the ERA5-Land deep soil temperature
#   interface  TiledLandInterface: a CanopyAirSpace over the vegetated fraction
#              (Jarvis conductance under plant-available-water stress, interactive
#              absorbed PAR, canopy interception, prognostic canopy-air storage and
#              prognostic soil skin) and its bare counterpart, each tile carrying
#              per-cell roughness, displacement, and albedo
#   datasets   MODIS MCD12Q1 land cover + MCD15A2H leaf area (May 2011, gap-filled),
#              SoilGrids 2.0 texture, Copernicus land albedo, ERA5-Land initial and
#              deep soil state
#
# The land surface radiation budget is prescribed from ERA5 downwelling fluxes
# (`ERA5PrescribedRadiation`); the atmosphere itself runs without radiative transfer,
# as in `run12km_land.jl` (the Breeze RRTMGP path cannot yet feed a CanopyAirSpace:
# its interface radiation state is zeroed and its surface flux would double-count).
#
# There is no land-sea mask yet (NumericalEarth#450): the slab covers the ocean too,
# initialized saturated at the ERA5 skin temperature with water-surface roughness, so
# open water evaporates at the potential rate while restoring to its initial SST.
#
# Three diurnal cycles of spin-up (17–19 May) precede the squall-line day, so the soil,
# canopy, and boundary layer are on their diurnal attractor when convection organizes.
#
# `REFINEMENT` scales resolution at fixed extent: 1 → ~12 km, 2 → ~6 km.
# `SMOKE=1` stops after 20 simulated minutes to exercise the full pipeline first.

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: on_architecture
using Oceananigans.Fields: interpolate!
using Breeze
using CopernicusClimateDataStore   # ERA5 + ERA5-Land + Copernicus albedo downloads
using CloudMicrophysics            # nested default microphysics → 1-moment mixed-phase
using ArchGDAL                     # MODIS granule warping
using CairoMakie                   # NumericalEarthMakieExt (visualize_nested_domain)
using NaturalEarth                 # NumericalEarthNaturalEarthExt (state/country lines)
using CUDA
using JLD2
using Printf
using Statistics: mean, median
import Dates
import Dates: DateTime

using NumericalEarth.EarthSystemModels.InterfaceComputations:
    atmosphere_land_stability_functions, LandRoughnessLength, LandZeroPlaneDisplacement,
    FixedIterations   # also exported by Breeze.Solvers — qualify to break the ambiguity
using NumericalEarth.DataWrangling.SoilGrids: SoilGrids2

arch = GPU(CUDA.CUDABackend(always_inline = true))
Oceananigans.defaults.FloatType = Float32
FT = Float32

# ## Configuration

SMOKE = get(ENV, "SMOKE", "0") == "1"
refinement = parse(Int, get(ENV, "REFINEMENT", "1"))

kilometers_per_degree = 111.32
resolution_km = round(Int, kilometers_per_degree / (9 * refinement))
resolution_tag = get(ENV, "TAG", SMOKE ? "smoke$(resolution_km)km" : "conus$(resolution_km)km_veg")

φ₀, λ₀ = 37.0, -97.5                        # HRRR CONUS center
start_date = DateTime(2011, 5, 17, 0)       # three diurnal cycles of spin-up...
stop_date  = DateTime(2011, 5, 21, 0)       # ...then the MC3E squall-line day
dates = (start_date, stop_date)

Δλ = Δφ = 1 / (9 * refinement)
Nx, Ny = 657 * refinement, 288 * refinement # HRRR footprint: λ ∈ [-134°, -61°], φ ∈ [21°, 53°]

λ_west,  λ_east  = λ₀ - Nx * Δλ / 2, λ₀ + Nx * Δλ / 2
φ_south, φ_north = φ₀ - Ny * Δφ / 2, φ₀ + Ny * Δφ / 2

# The datasets are windowed on the domain extent, which is refinement-invariant, so
# every resolution shares one set of downloaded files.
ingest_region = BoundingBox(longitude = (λ_west - 0.2, λ_east + 0.2),
                            latitude  = (φ_south - 0.2, φ_north + 0.2))

# Davies relaxation and terrain smoothing are physical widths, held fixed under refinement.
relax_width = 5 * refinement
terrain_smoothing_passes = 2 * refinement^2

# Vertical grid matched to Fan et al. (2017)'s WRF nest: 50 cells, 60 m surface cell,
# 490 m maximum spacing, top at ~20 km.
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

# ## The nest
#
# WENO(5) with explicit vertical advection for momentum and every scalar — the one
# combination validated at both 12 and 3 km in this configuration.

reconstruction(; kw...) = WENO(; order = 5, kw...)
momentum_advection = reconstruction()
scalar_advection = (ρθ  = reconstruction(),
                    ρqᵉ = reconstruction(bounds = (0, 1)),
                    ρqʳ = reconstruction(bounds = (0, 1)),
                    ρqˢ = reconstruction(bounds = (0, 1)))

Δt = 1.0 / refinement
era5_datadir = "era5"
dataset = ERA5HourlyPressureLevels()

nest = nested_atmosphere_model(grid, dataset;
                               dates,
                               dir = era5_datadir,
                               parent_time_indices_in_memory = 4,
                               terrain = ETOPO2022(),
                               terrain_smoothing_passes,
                               relaxation_rate = 1/300,
                               relaxation_width = relax_width,
                               momentum_advection,
                               scalar_advection)

child = nest.child
parent_atmosphere = nest.parent
era5_region = BoundingBox(parent_atmosphere.grid)

# ERA5 single-level forcing (radiation, initial skin temperature) is requested on the
# same padded box as the nest's pressure-level files, so one prefetched cache serves all.
era5_forcing_region = BoundingBox(longitude = (λ_west - 0.5, λ_east + 0.5),
                                  latitude  = (φ_south - 0.5, φ_north + 0.5))

fig = visualize_nested_domain(grid;
                              parent       = era5_region,
                              padding      = 2.5,
                              resolution   = 1/10,
                              title        = "ERA5 → $(resolution_km) km CONUS LAM + tiled canopy land (MC3E, 17–21 May 2011)",
                              label        = "$(resolution_km) km CONUS LAM (child)",
                              parent_label = "ERA5 parent",
                              landmarks    = tuple("ARM SGP" => (-97.485, 36.605)))
save("$(resolution_tag)_domains.png", fig)

# ## Land grid (index-identical horizontally to the child, as the coupler requires)

land_grid = LatitudeLongitudeGrid(arch;
                                  longitude = (λ_west,  λ_east),
                                  latitude  = (φ_south, φ_north),
                                  size = (Nx, Ny),
                                  halo = (5, 5),
                                  topology = (Bounded, Bounded, Flat))

cpu_land_grid = on_architecture(CPU(), land_grid)

# ## Surface-field ingestion
#
# Raw rasters → per-cell model fields, cached per resolution so only the first run pays.

surface_cache = "conus_surface_cache_r$(refinement).jld2"

new_surface_field(g) = Field{Center, Center, Nothing}(g)

fill_invalid!(field, value) = (parent(field) .= ifelse.(isfinite.(parent(field)), parent(field), value); field)

function transfer_to(g, cpu_field)
    field = new_surface_field(g)
    set!(field, Array(interior(cpu_field, :, :, 1)))
    return field
end

# Categorical fields (the IGBP class) sample nearest-neighbor from the MODIS lattice;
# bilinear interpolation would average urban against water.
function nearest_native_classes(g, native_classes)
    classes = zeros(Int, size(g, 1), size(g, 2))
    native  = interior(on_architecture(CPU(), native_classes), :, :, 1)
    λⁿ, φⁿ, _ = nodes(native_classes.grid, Center(), Center(), Center())
    λ, φ, _ = nodes(g, Center(), Center(), Center())
    Δλⁿ = λⁿ[2] - λⁿ[1]
    Δφⁿ = φⁿ[2] - φⁿ[1]
    for j in axes(classes, 2), i in axes(classes, 1)
        iⁿ = clamp(round(Int, 1 + (λ[i] - λⁿ[1]) / Δλⁿ), 1, length(λⁿ))
        jⁿ = clamp(round(Int, 1 + (φ[j] - φⁿ[1]) / Δφⁿ), 1, length(φⁿ))
        value = native[iⁿ, jⁿ]
        classes[i, j] = isfinite(value) ? round(Int, value) : igbp_class_names.water
    end
    return classes
end

# The roughness tables use `:cropland_vegetation_mosaic` / `:snow_and_ice` where the
# IGBP legend says `cropland_natural_mosaic` / `permanent_snow_and_ice`.
igbp_symbol(code) = code == igbp_class_names.cropland_natural_mosaic ? :cropland_vegetation_mosaic :
                    code == igbp_class_names.permanent_snow_and_ice  ? :snow_and_ice :
                    begin
                        key = findfirst(==(code), NamedTuple{keys(igbp_class_names)}(igbp_class_names))
                        isnothing(key) ? :unclassified : key
                    end

vegetated_igbp_classes = (:evergreen_needleleaf_forest, :evergreen_broadleaf_forest,
                          :deciduous_needleleaf_forest, :deciduous_broadleaf_forest,
                          :mixed_forest, :closed_shrubland, :open_shrubland,
                          :woody_savanna, :savanna, :grassland, :permanent_wetland,
                          :cropland, :cropland_vegetation_mosaic)

function ingest_surface_fields()
    ## --- MODIS land cover (2011) and leaf area index: five 8-day composites around the
    ## case day on the native 464 m lattice, cloud gaps filled from each cell's own
    ## composites and same-class neighbors, non-vegetated classes zeroed.
    modis_classes = Field(Metadatum(:landcover_class; dataset = MCD12Q1(),
                                    region = ingest_region, date = DateTime(2011)))

    composite_stamps = [DateTime(2011, 5, 1), DateTime(2011, 5, 9), DateTime(2011, 5, 17),
                        DateTime(2011, 5, 25), DateTime(2011, 6, 2)]
    lai_series = FieldTimeSeries{Center, Center, Nothing}(modis_classes.grid,
                                                          [Dates.datetime2unix(d) for d in composite_stamps])
    for (n, date) in enumerate(composite_stamps)
        composite = Field(Metadatum(:leaf_area_index; dataset = MCD15A2H(),
                                    region = ingest_region, date))
        parent(lai_series[n]) .= parent(composite)
        composite = nothing
        GC.gc()
    end

    fill_seasonal_gaps!(lai_series, modis_classes;
                        cyclic = false,
                        max_gap = class_maximum_gap(modis_classes),
                        valid_range = (0, 10),
                        unfilled_classes = igbp_non_vegetated_classes)
    zero_non_vegetated!(lai_series, modis_classes)
    case_lai = lai_series[3]   ## the 17–25 May composite containing the case day

    igbp = nearest_native_classes(cpu_land_grid, modis_classes)
    water = [igbp_symbol(code) == :water for code in igbp]

    leaf_area_index = new_surface_field(cpu_land_grid)
    interpolate!(leaf_area_index, case_lai)
    fill_invalid!(leaf_area_index, 0)
    parent(leaf_area_index) .= clamp.(parent(leaf_area_index), 0, 10)
    modis_classes = lai_series = case_lai = nothing
    GC.gc()

    ## --- Soil hydraulics: SoilGrids 2.0 (10 km) texture combined onto a three-layer
    ## lattice, then pedotransfer functions → per-cell van Genuchten fields.
    slab_depth = 0.5
    soil_lattice = LatitudeLongitudeGrid(CPU(), Float64;
                                         size = (Nx, Ny, 3),
                                         longitude = (λ_west, λ_east),
                                         latitude  = (φ_south, φ_north),
                                         z = [-1.0, -0.6, -0.3, 0.0],
                                         topology = (Bounded, Bounded, Bounded))

    ## SoilGrids layers (deepest first): 100–200, 60–100, 30–60, 15–30, 5–15, 0–5 cm.
    ## Thickness-weighted onto the lattice's 0–30 / 30–60 / 60–100 cm layers.
    function lattice_texture(name)
        native = Field(Metadatum(name; dataset = SoilGrids2(), region = ingest_region), CPU())
        L = interior(native)
        layer(k) = view(L, :, :, k)
        λfaces = λnodes(native.grid, Face())
        φfaces = φnodes(native.grid, Face())
        combined = Field{Center, Center, Center}(
            LatitudeLongitudeGrid(CPU(), Float64;
                                  size = (size(L, 1), size(L, 2), 3),
                                  longitude = (first(λfaces), last(λfaces)),
                                  latitude  = (first(φfaces), last(φfaces)),
                                  z = [-1.0, -0.6, -0.3, 0.0],
                                  topology = (Bounded, Bounded, Bounded)))
        interior(combined, :, :, 3) .= (5 .* layer(6) .+ 10 .* layer(5) .+ 15 .* layer(4)) ./ 30
        interior(combined, :, :, 2) .= layer(3)
        interior(combined, :, :, 1) .= layer(2)
        ## Ocean and rock arrive as NaN; fill with the land median before regridding so
        ## the non-NaN-aware bilinear pass cannot dilate NaN inland along the coast.
        for k in 1:3
            data = interior(combined, :, :, k)
            data .= ifelse.(isfinite.(data), data, median(filter(isfinite, data)))
        end
        target = Field{Center, Center, Center}(soil_lattice)
        interpolate!(target, combined)
        return target
    end

    sand = lattice_texture(:sand_fraction)
    silt = lattice_texture(:silt_fraction)
    clay = lattice_texture(:clay_fraction)
    bulk_density = lattice_texture(:bulk_density)

    hydraulics = soil_hydraulic_properties(sand, silt, clay, bulk_density; slab_depth)

    finite_median(f) = median(filter(isfinite, Array(interior(f))))
    positive_median(f) = median(filter(x -> isfinite(x) && x > 0, Array(interior(f))))
    scalar_porosity = positive_median(hydraulics.porosity)

    ## Macropore-inclusive Cosby conductivity caps infiltration (m/s × ρˡ → kg m⁻² s⁻¹).
    cosby_conductivity = compute!(Field(saturated_conductivity(CosbyConductivity(), sand)))
    infiltration_capacity = 1000 * positive_median(cosby_conductivity)

    hydraulic_fields = map(hydraulics) do f
        g = new_surface_field(cpu_land_grid)
        interior(g, :, :, 1) .= interior(f, :, :, 1)
        fill_invalid!(g, finite_median(f))
        return g
    end

    ## Dry areal heat capacity from the 0–30 cm bulk density over a 0.15 m diurnal skin.
    dry_heat_capacity = new_surface_field(cpu_land_grid)
    interior(dry_heat_capacity, :, :, 1) .= interior(bulk_density, :, :, 3)
    fill_invalid!(dry_heat_capacity, 1350)
    parent(dry_heat_capacity) .= 840 .* 0.15 .* parent(dry_heat_capacity)

    sand = silt = clay = bulk_density = hydraulics = cosby_conductivity = nothing
    GC.gc()

    ## --- Blue-sky albedo: the Copernicus Global Land dekad stamped on the case day.
    albedo = new_surface_field(cpu_land_grid)
    albedo_native = Field(Metadatum(:albedo; dataset = CopernicusAlbedo(),
                                    region = ingest_region, date = DateTime(2011, 5, 20)), CPU())
    fill_invalid!(albedo_native, 0.18)
    interpolate!(albedo, albedo_native)
    fill_invalid!(albedo, 0.18)
    parent(albedo) .= clamp.(parent(albedo), 0.03, 0.6)
    albedo_native = nothing
    GC.gc()

    ## --- Initial and deep soil state: ERA5-Land, NaN-filled on the native 0.1° grid
    ## (ocean is masked there) before the bilinear regrid.
    function era5_land_field(name, era5_land_dataset, date, fallback)
        native = Field(Metadatum(name; dataset = era5_land_dataset, region = ingest_region,
                                 date, dir = era5_datadir), CPU())
        fill_invalid!(native, fallback)
        target = new_surface_field(cpu_land_grid)
        interpolate!(target, native)
        fill_invalid!(target, fallback)
        return target
    end

    soil_water = new_surface_field(cpu_land_grid)
    layers = (:volumetric_soil_water_layer_1, :volumetric_soil_water_layer_2, :volumetric_soil_water_layer_3)
    weights = (0.14, 0.42, 0.44)   ## ERA5-Land 0–7, 7–28, 28–100 cm sampled over the 0–50 cm slab
    for (name, w) in zip(layers, weights)
        θ = era5_land_field(name, ERA5MonthlyLand(), DateTime(2011, 5, 1), 0.25)
        parent(soil_water) .+= w .* parent(θ)
    end

    deep_temperature = era5_land_field(:soil_temperature_level_3, ERA5MonthlyLand(),
                                       DateTime(2011, 5, 1), 288)

    ## --- Vegetation fraction (Beer–Lambert cover of the cell-mean LAI) and the
    ## vegetated-area LAI the canopy tile sees.
    vegetation_fraction = compute!(Field(leaf_area_index_cover_fraction(leaf_area_index)))
    fveg = Array(interior(vegetation_fraction, :, :, 1))
    fveg[water] .= 0
    set!(vegetation_fraction, fveg)

    tile_lai = new_surface_field(cpu_land_grid)
    interior(tile_lai, :, :, 1) .= clamp.(Array(interior(leaf_area_index, :, :, 1)) ./ clamp.(fveg, 0.25, 1), 0, 8)

    ## --- Aerodynamics: Raupach drag-partition roughness per IGBP class from the tile
    ## LAI and the class's representative canopy height; prescribed constants for the
    ## non-vegetated classes; bare tile at bare-soil / open-water roughness.
    symbols = map(igbp_symbol, igbp)

    class_height = [class in vegetated_igbp_classes ? representative_canopy_height(FT, class) : FT(0.1)
                    for class in symbols]
    canopy_height = new_surface_field(cpu_land_grid)
    interior(canopy_height, :, :, 1) .= class_height

    ℓᵐ_veg = zeros(FT, size(igbp)); d_veg = zeros(FT, size(igbp))
    for class in unique(symbols)
        class in vegetated_igbp_classes || continue
        closure = DragPartitionRoughness(FT; vegetation_type = class)
        ℓᵐ_class, d_class = canopy_roughness(closure, tile_lai, canopy_height)
        cells = symbols .== class
        ℓᵐ_veg[cells] .= Array(interior(ℓᵐ_class, :, :, 1))[cells]
        d_veg[cells]  .= Array(interior(d_class,  :, :, 1))[cells]
    end
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

    ℓᵐ_bare = fill(FT(0.01), size(igbp)); d_bare = fill(FT(0.05), size(igbp))
    ℓᵐ_water, d_water = nonvegetated_roughness(FT, :water)
    ℓᵐ_bare[water] .= ℓᵐ_water; d_bare[water] .= d_water

    to_field(a) = (f = new_surface_field(cpu_land_grid); set!(f, a); f)

    return (; leaf_area_index, tile_lai, vegetation_fraction, igbp, water,
              canopy_height, slab_depth, hydraulic_fields, scalar_porosity, infiltration_capacity,
              dry_heat_capacity, albedo, soil_water, deep_temperature,
              vegetated_roughness = (momentum_roughness_length = to_field(ℓᵐ_veg),
                                     scalar_roughness_length   = to_field(ℓᵐ_veg ./ 10),
                                     zero_plane_displacement   = to_field(d_veg)),
              bare_roughness      = (momentum_roughness_length = to_field(ℓᵐ_bare),
                                     scalar_roughness_length   = to_field(ℓᵐ_bare ./ 10),
                                     zero_plane_displacement   = to_field(d_bare)))
end

on_cpu(x) = x
on_cpu(f::Field) = on_architecture(CPU(), f)
on_cpu(t::NamedTuple) = map(on_cpu, t)

on_grid(g, x) = x
on_grid(g, t::NamedTuple) = map(x -> on_grid(g, x), t)
on_grid(g, f::Field) = transfer_to(g, f)

static = if isfile(surface_cache)
    @info "Loading cached surface fields from $surface_cache"
    file = jldopen(surface_cache)
    loaded = file["static"]
    close(file)
    loaded
else
    @info "Ingesting surface fields (first run at this resolution)..."
    fresh = ingest_surface_fields()
    jldsave(surface_cache; static = on_cpu(fresh))
    fresh
end

## Move every cached CPU field onto the run architecture (non-fields pass through).
static = on_grid(land_grid, static)

# ## The land model

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
    porosity = static.scalar_porosity)

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
    storage = PrognosticCanopyAir(layer_depth = static.canopy_height))

# Fixed iteration counts keep every GPU thread on the same path; a tolerance-based
# solver leaves whole warps spinning on the slowest cell.
land_fluxes() = SimilarityTheoryFluxes(FT;
    momentum_roughness_length    = LandRoughnessLength(FT),
    temperature_roughness_length = LandRoughnessLength(FT),
    water_vapor_roughness_length = LandRoughnessLength(FT),
    zero_plane_displacement      = LandZeroPlaneDisplacement(),
    stability_functions          = atmosphere_land_stability_functions(FT),
    solver_stop_criteria         = FixedIterations(8))

# ## Radiation and coupling

radiation = ERA5PrescribedRadiation(arch;
                                    dataset = ERA5HourlySingleLevel(),
                                    start_date, end_date = stop_date,
                                    region = era5_forcing_region,
                                    dir = era5_datadir,
                                    land_surface = SurfaceRadiationProperties(static.albedo, 0.96),
                                    ocean_surface = nothing, sea_ice_surface = nothing)

atmosphere = Simulation(nest; Δt)   ## the coupled model manages Δt; this sets the initial value

interface = TiledLandInterface(land_grid, atmosphere, land;
    vegetated,
    fraction = static.vegetation_fraction,
    vegetated_fluxes = land_fluxes(),
    bare_fluxes      = land_fluxes(),
    vegetated_surface_properties = merge(static.vegetated_roughness,
                                         (leaf_albedo = static.albedo,
                                          ground_albedo = static.albedo)),
    bare_surface_properties      = merge(static.bare_roughness,
                                         (ground_albedo = static.albedo,)))

# ## Initial state
#
# Skin temperature from ERA5 (SST over the ocean); soil water from ERA5-Land clamped
# inside the pedotransfer pore space; open water saturated.

skin_temperature = Metadatum(:skin_temperature; dataset = ERA5HourlySingleLevel(),
                             date = start_date, region = era5_forcing_region, dir = era5_datadir)
set!(land.temperature, skin_temperature)

ν  = static.hydraulic_fields.porosity
θʳ = static.hydraulic_fields.residual_liquid_fraction
initial_storage = new_surface_field(land_grid)
θ₀ = clamp.(Array(interior(on_cpu(static.soil_water), :, :, 1)),
            1.05 .* Array(interior(on_cpu(θʳ), :, :, 1)),
            0.95 .* Array(interior(on_cpu(ν), :, :, 1)))
θ₀[static.water] .= Array(interior(on_cpu(ν), :, :, 1))[static.water]
set!(initial_storage, θ₀ .* (1000 * static.slab_depth))
set!(land; M = initial_storage, canopy_water_storage = 0, surface_water_storage = 0)

@info @sprintf("initial soil wetness 𝒮 ∈ [%.3f, %.3f], mean %.3f",
               minimum(land.saturation), maximum(land.saturation), mean(land.saturation))

model = AtmosphereLandModel(atmosphere, land; radiation,
                            atmosphere_land_interface = interface)

child = model.atmosphere.model.child

stop_time = parse(Float64, get(ENV, "STOP_TIME", SMOKE ? "1200" : "345600"))   # 96 h
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
land_fields = (Tˡᵃ = land.temperature,
               𝒮   = land.saturation,
               W   = land.water_storage,
               Wᶜ  = land.prognostic.canopy_water_storage,
               Wᵖ  = land.prognostic.surface_water_storage,
               LST = interface.temperature.effective,
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
        leaf_area_index = on_cpu(static.leaf_area_index),
        tile_lai = on_cpu(static.tile_lai),
        vegetation_fraction = on_cpu(static.vegetation_fraction),
        canopy_height = on_cpu(static.canopy_height),
        porosity = on_cpu(static.hydraulic_fields.porosity),
        conductivity = on_cpu(static.hydraulic_fields.matching_point_conductivity),
        inverse_air_entry_head = on_cpu(static.hydraulic_fields.inverse_air_entry_head),
        pore_size_uniformity = on_cpu(static.hydraulic_fields.pore_size_uniformity),
        albedo = on_cpu(static.albedo),
        dry_heat_capacity = on_cpu(static.dry_heat_capacity),
        momentum_roughness = on_cpu(static.vegetated_roughness.momentum_roughness_length),
        displacement = on_cpu(static.vegetated_roughness.zero_plane_displacement),
        bare_roughness = on_cpu(static.bare_roughness.momentum_roughness_length),
        initial_soil_water = on_cpu(static.soil_water),
        deep_temperature = on_cpu(static.deep_temperature),
        igbp = static.igbp,
        water = static.water)

# ## Progress

wall_time = Ref(time_ns())
function progress(sim)
    child = sim.model.atmosphere.model.child
    u, v, w = child.velocities
    ρ  = child.dynamics.total_density
    qᵛ = specific_humidity(child)
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
