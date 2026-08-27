# Shared configuration and dataset ingestion for the Central Borneo slab-canopy calibration.
#
# A 2° × 2° box over the Central Borneo highlands (0.5–2.5 °N, 113–115 °E) at
# Δ = 1/(9 REFINEMENT)° (≈ 12 km / 6 km / 3 km for REFINEMENT = 1 / 2 / 4), forced by ERA5
# hourly single-level fields over 1–8 April 2020. Every surface parameter is ingested from
# a satellite product as in `experiments/conus_slab_canopy_v2` and cached per product and
# refinement as CPU arrays, so the model scripts (column, map, GPU) just load them:
#
#   soil        OpenLandMap-soilDB texture + bulk density → Weynants van Genuchten curves
#   land cover  MODIS MCD12Q1 class fractions and majority vegetated class (2020)
#   leaf area   MODIS MCD15A2H 8-day composites around the window, gap-filled
#   canopy      ETH Sentinel-2 10 m canopy height, class height as the floor
#   urban       GHSL building height + built fraction → morphometric roughness
#   radiation   Copernicus Global Land blue-sky albedo, ASTER GED emissivity
#   soil state  ERA5-Land hourly soil water (initial state and calibration target),
#               ERA5-Land monthly deep soil temperature
#
# Everything here runs on the CPU. `NUMERICALEARTH_DATA_DIRECTORY` should point at the
# run directory so the downloads land beside the caches.

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.Fields: interpolate!
using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.DataWrangling.MODISLand: modis_composite_dates
using CopernicusClimateDataStore   # ERA5 + ERA5-Land downloads
using ArchGDAL                     # COG / GeoTIFF / HDF readers
using JLD2
using Printf
using Statistics: mean, median
import Dates
import Dates: DateTime, Hour

include(joinpath(@__DIR__, "ingest_surface.jl"))

# ## Configuration

include(joinpath(@__DIR__, "borneo_config.jl"))

cpu_land_grid = land_grid()
pedotransfer_slab_depth = 0.3   # layer weights for the hydraulic parameters (0–30 cm)

# ## Dataset-specific ingestion

# Seven 8-day composites from late March to early May 2020; the run starts inside the
# 29 March composite (index 2). Cloud gaps over Borneo are filled from each cell's own
# series and same-class neighbors; cells the fill cannot reach get a class-typical value.
lai_stamps = modis_composite_dates(DateTime(2020, 3, 20), DateTime(2020, 5, 10), 8)
lai_case_index = findlast(<=(start_date), lai_stamps)

class_typical_lai = (evergreen_needleleaf_forest = 4.0, evergreen_broadleaf_forest = 5.0,
                     deciduous_needleleaf_forest = 3.0, deciduous_broadleaf_forest = 4.0,
                     mixed_forest = 4.0, closed_shrubland = 2.0, open_shrubland = 1.0,
                     woody_savanna = 2.5, savanna = 1.5, grassland = 1.5, permanent_wetland = 3.0,
                     cropland = 2.0, cropland_natural_mosaic = 2.5)

function ingest_modis_borneo(grid, region, fill_cache)
    classes = Field(Metadatum(:landcover_class; dataset = MCD12Q1(), region, date = DateTime(landcover_year)), CPU())
    codes = Array(interior(classes, :, :, 1))
    codes .= ifelse.(isfinite.(codes), codes, igbp_class_names.water)

    indicator = surface_field(classes.grid)
    fractions = map(igbp_class_names) do code
        interior(indicator, :, :, 1) .= codes .== code
        fraction = surface_field(grid)
        regrid!(fraction, indicator)
        return fraction
    end

    series = FieldTimeSeries{Center, Center, Nothing}(classes.grid, Dates.datetime2unix.(lai_stamps))
    for (n, date) in enumerate(lai_stamps)
        composite = Field(Metadatum(:leaf_area_index; dataset = MCD15A2H(), region, date), CPU())
        parent(series[n]) .= parent(composite)
        composite = nothing
        GC.gc()
    end
    valid_before = count(isfinite, interior(series[lai_case_index], :, :, 1))
    fill_seasonal_gaps!(series, classes; cyclic = false, cache = fill_cache,
                        max_gap = class_maximum_gap(classes), valid_range = (0, 10),
                        unfilled_classes = igbp_non_vegetated_classes)
    zero_non_vegetated!(series, classes)
    valid_after = count(isfinite, interior(series[lai_case_index], :, :, 1))
    @info @sprintf("MODIS LAI: %.1f%% of native cells valid before the fill, %.1f%% after",
                   100 * valid_before / length(codes), 100 * valid_after / length(codes))

    leaf_area_index = surface_field(grid)
    masked_regrid!(leaf_area_index, series[lai_case_index])
    return (; fractions, leaf_area_index)
end

# The two 3° ETH tiles covering the box, area-averaged onto their 0.01° lattices and cached.
function ensure_eth_tiles!(region)
    for corner in worldcover_tile_corners(region.longitude, region.latitude)
        path = eth_canopy_tile_path(corner)
        isfile(path) && continue
        @info "ETH canopy tile $corner"
        grid = eth_canopy_tile_grid(corner)
        height = canopy_height_field(grid, ETHSentinel2CanopyHeight())
        mkpath(dirname(path))
        jldsave(path; height = Array(interior(height, :, :, 1)), corner)
        GC.gc()
    end
    return nothing
end

# ERA5-Land hourly volumetric soil water in the three upper layers (0–7, 7–28, 28–100 cm),
# interpolated onto the land grid for every hour of the window: the initial soil water and
# the calibration target. ERA5-Land is land-only; sea cells are filled with the land mean.
function ingest_era5_land_hourly(grid, region, dates)
    names = (:volumetric_soil_water_layer_1, :volumetric_soil_water_layer_2, :volumetric_soil_water_layer_3)
    layers = map(names) do name
        series = zeros(Float64, length(dates), size(grid, 1), size(grid, 2))
        for (n, date) in enumerate(dates)
            native = Field(Metadatum(name; dataset = ERA5HourlyLand(), region, date), CPU())
            data = interior(native, :, :, 1)
            fill_invalid!(native, mean(filter(isfinite, data)))
            target = surface_field(grid)
            interpolate!(target, native)
            series[n, :, :] .= interior(target, :, :, 1)
        end
        return series
    end
    return (; times = Dates.datetime2unix.(dates) .- Dates.datetime2unix(first(dates)),
              layer_1 = layers[1], layer_2 = layers[2], layer_3 = layers[3])
end

# ERA5 forcing: every hourly slice of the ERA5 atmosphere and radiation interpolated onto
# the land grid and stored as plain arrays (parent layout, halos included) so the model
# scripts can rebuild an in-memory `PrescribedAtmosphere`/`PrescribedRadiation` on any
# architecture without touching the disk-backed series again.
function ingest_forcing(grid, region, dates)
    dataset = ERA5HourlySingleLevel()
    start_date, end_date = first(dates), last(dates)
    era5_atmosphere = ERA5PrescribedAtmosphere(CPU(); dataset, start_date, end_date, region,
                                               surface_layer_height, boundary_layer_height)
    era5_radiation = ERA5PrescribedRadiation(CPU(); dataset, start_date, end_date, region,
                                             land_surface = SurfaceRadiationProperties(0.15, 0.95),
                                             ocean_surface = nothing, sea_ice_surface = nothing)
    times = era5_atmosphere.velocities.u.times
    slices(fts) = [(target = surface_field(grid); interpolate!(target, fts[n]); Array(parent(target)))
                   for n in eachindex(times)]
    forcing = (; times = collect(times),
                 u    = slices(era5_atmosphere.velocities.u),
                 v    = slices(era5_atmosphere.velocities.v),
                 T    = slices(era5_atmosphere.temperature),
                 q    = slices(era5_atmosphere.specific_humidity),
                 p    = slices(era5_atmosphere.pressure),
                 rain = slices(era5_atmosphere.precipitation_flux.rain),
                 sw   = slices(era5_radiation.downwelling_shortwave),
                 lw   = slices(era5_radiation.downwelling_longwave))

    skin_temperature = Field(Metadatum(:skin_temperature; dataset, date = start_date, region), grid)
    era5_elevation   = Field(Metadatum(:topography; dataset, date = start_date, region), grid)
    land_elevation   = regrid_topography(grid; dataset = ETOPO2022())
    return merge(forcing, (; skin_temperature = Array(interior(skin_temperature, :, :, 1)),
                             era5_elevation   = Array(interior(era5_elevation, :, :, 1)),
                             land_elevation   = Array(interior(land_elevation, :, :, 1))))
end

# ## Ingestion (cached per product and refinement)

soil = cached(cache_file("soil")) do
    ingest_soil(cpu_land_grid, ingest_region; slab_depth = pedotransfer_slab_depth)
end
modis = cached(cache_file("modis")) do
    ingest_modis_borneo(cpu_land_grid, ingest_region, joinpath(cache_directory, "modis_lai_fill.jld2"))
end
urban = cached(cache_file("urban")) do
    ingest_urban(cpu_land_grid)
end
optics = cached(cache_file("optics")) do
    albedo = Field(Metadatum(:albedo; dataset = CopernicusAlbedo(), region = ingest_region, date = DateTime(2020, 4, 10)), CPU())
    target = surface_field(cpu_land_grid)
    masked_regrid!(target, albedo)
    merge((; albedo = target), ingest_emissivity(cpu_land_grid, ingest_region))
end
era5_land = cached(cache_file("era5_land")) do
    deep = Field(Metadatum(:soil_temperature_level_3; dataset = ERA5MonthlyLand(), region = ingest_region,
                           date = DateTime(2020, 4, 1)), CPU())
    fill_invalid!(deep, mean(filter(isfinite, interior(deep, :, :, 1))))
    deep_temperature = surface_field(cpu_land_grid)
    interpolate!(deep_temperature, deep)
    merge((; deep_temperature), ingest_era5_land_hourly(cpu_land_grid, ingest_region, start_date:Hour(1):end_date))
end
eth = cached(cache_file("eth_canopy")) do
    ensure_eth_tiles!(ingest_region)
    ingest_canopy_height(cpu_land_grid, ingest_region)
end
forcing = cached(cache_file("forcing")) do
    ingest_forcing(cpu_land_grid, era5_region, start_date:Hour(1):end_date)
end

# ## Derived surface fields (CPU arrays)

array(field) = Array(interior(field, :, :, 1))

vegetated_igbp_classes = keys(class_typical_lai)
tree_classes = (:evergreen_needleleaf_forest, :evergreen_broadleaf_forest, :deciduous_needleleaf_forest,
                :deciduous_broadleaf_forest, :mixed_forest, :woody_savanna, :savanna, :permanent_wetland)

roughness_class(class) = class == :cropland_natural_mosaic ? :cropland_vegetation_mosaic :
                         class == :permanent_snow_and_ice  ? :snow_and_ice : class

vegetated_cover = sum(array(modis.fractions[class]) for class in vegetated_igbp_classes)
water_cover = array(modis.fractions.water)
urban_cover = array(modis.fractions.urban)
water = water_cover .> 0.5

vegetated_stack = cat((array(modis.fractions[class]) for class in vegetated_igbp_classes)...; dims = 3)
canopy_class = [vegetated_igbp_classes[argmax(view(vegetated_stack, i, j, :))]
                for i in axes(vegetated_stack, 1), j in axes(vegetated_stack, 2)]

leaf_area_index = array(modis.leaf_area_index)
lai_filled_from_class = .!isfinite.(leaf_area_index) .& (vegetated_cover .> 0.05)
leaf_area_index .= ifelse.(isfinite.(leaf_area_index), leaf_area_index,
                           [class_typical_lai[class] for class in canopy_class] .* vegetated_cover)
clamp!(leaf_area_index, 0, 10)
@info @sprintf("LAI: %d of %d cells took the class-typical value", count(lai_filled_from_class), length(leaf_area_index))

vegetation_fraction = ifelse.(leaf_area_index .> 0.1, vegetated_cover, 0.0)
vegetation_fraction[water] .= 0
tile_lai = clamp.(leaf_area_index ./ max.(vegetation_fraction, 0.05), 0.1, 8)

class_canopy_height = [representative_canopy_height(Float64, roughness_class(class)) for class in canopy_class]
eth_canopy_height = array(eth.eth_canopy_height)
canopy_height = max.(ifelse.(isfinite.(eth_canopy_height), eth_canopy_height, 0), min.(class_canopy_height, 1.6))

# Vegetated tile: Raupach drag-partition roughness of the majority class at the tile leaf
# area and the measured height, capped below the 10 m forcing height.
cpu_tile_lai = surface_field(cpu_land_grid); set!(cpu_tile_lai, tile_lai)
cpu_canopy_height = surface_field(cpu_land_grid); set!(cpu_canopy_height, canopy_height)
vegetated_roughness_length = fill(0.03, Nx, Ny)
for class in unique(canopy_class)
    class_roughness_length, _ = canopy_roughness(DragPartitionRoughness(Float64; vegetation_type = roughness_class(class)),
                                                 cpu_tile_lai, cpu_canopy_height)
    cells = canopy_class .== class
    vegetated_roughness_length[cells] .= array(class_roughness_length)[cells]
end
replace!(vegetated_roughness_length, NaN => 0.03)
clamp!(vegetated_roughness_length, 1e-4, 1.5)

# Bare tile: log-mean of the urban (GHSL morphometric), open-water and bare-soil roughness,
# weighted by each surface's share of the non-vegetated area.
default_urban_roughness_length, _ = nonvegetated_roughness(Float64, :urban)
water_roughness_length, _ = nonvegetated_roughness(Float64, :water)
soil_roughness_length, _ = nonvegetated_roughness(Float64, :barren)
urban_roughness_length = array(urban.urban_roughness)
urban_roughness_length .= ifelse.(isfinite.(urban_roughness_length), urban_roughness_length, default_urban_roughness_length)
nonvegetated = max.(1 .- vegetation_fraction, 0.01)
urban_weight = min.(urban_cover, nonvegetated)
water_weight = min.(water_cover, nonvegetated .- urban_weight)
soil_weight  = nonvegetated .- urban_weight .- water_weight
bare_roughness_length = exp.((urban_weight .* log.(urban_roughness_length) .+ water_weight .* log(water_roughness_length) .+
                              soil_weight .* log(soil_roughness_length)) ./ nonvegetated)

albedo = array(optics.albedo)
albedo[water] .= 0.07
replace!(albedo, NaN => median(filter(isfinite, albedo[.!water])))
clamp!(albedo, 0.03, 0.6)
emissivity = array(optics.emissivity)
emissivity[water] .= 0.98
replace!(emissivity, NaN => median(filter(isfinite, emissivity[.!water])))
clamp!(emissivity, 0.9, 1)

hydraulics = map(array, soil.hydraulic_fields)
porosity = hydraulics.porosity
residual = hydraulics.residual_liquid_fraction

# ERA5-Land 0–28 cm volumetric water (layers 1 and 2, thickness-weighted): the initial
# soil water and the calibration target the slab depth is fitted against.
initial_soil_water = clamp.(era5_land_soil_water(era5_land, 1), 1.05 .* residual, 0.95 .* porosity)
initial_soil_water[water] .= porosity[water]

static = (; leaf_area_index, tile_lai, vegetation_fraction, canopy_height, class_canopy_height, eth_canopy_height,
            canopy_class = String.(canopy_class), vegetated_cover, water_cover, urban_cover, water,
            vegetated_roughness_length, bare_roughness_length,
            urban_roughness = array(urban.urban_roughness), urban_fraction = array(urban.urban_fraction),
            building_height = array(urban.building_height),
            albedo, emissivity, hydraulics...,
            infiltration_capacity = soil.infiltration_capacity, scalar_porosity = soil.scalar_porosity,
            dry_heat_capacity = array(soil.dry_heat_capacity),
            sand = array(soil.texture_fields.sand), clay = array(soil.texture_fields.clay),
            bulk_density = array(soil.texture_fields.bulk_density),
            deep_temperature = array(era5_land.deep_temperature),
            initial_soil_water,
            longitude = Array(λnodes(cpu_land_grid, Center())),
            latitude  = Array(φnodes(cpu_land_grid, Center())))

jldsave(cache_file("static"); static...)
@info "Borneo surface fields ready at ≈ $(resolution_km) km ($(Nx) × $(Ny))"
