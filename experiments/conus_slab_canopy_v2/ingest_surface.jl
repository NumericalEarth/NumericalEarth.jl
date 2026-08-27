# Surface parameters for the CONUS canopy-land run: each product is read at (or aggregated
# to) a resolution matched to the land grid, landed on the grid as a Field, and cached as a
# JLD2 file of CPU fields so later runs load the fields instead of the rasters.
#
#   OpenLandMap-soilDB    texture + bulk density → Weynants van Genuchten parameters
#   MODIS MCD12Q1/MCD15A2H IGBP class fractions, majority class, gap-filled leaf area index
#   ESA WorldCover        10 m class fractions from pre-aggregated 3° tiles (optional)
#   GHSL                  building height + built fraction → morphometric urban roughness
#   Copernicus Global Land blue-sky albedo
#   ASTER GED             broadband emissivity
#   ERA5-Land             initial soil water and deep soil temperature

using NumericalEarth
using Oceananigans
using Oceananigans.Fields: interpolate!, regrid!
using NumericalEarth.DataWrangling: NearestNeighborInpainting, inpaint_mask!, metadata_path
using NumericalEarth.DataWrangling.WorldCover: ESA_WORLDCOVER_CLASS_NAMES
using JLD2
using Statistics: mean, median
import Dates
import Dates: DateTime

include(joinpath(@__DIR__, "worldcover_tiles.jl"))
include(joinpath(@__DIR__, "eth_canopy_tiles.jl"))

surface_field(grid) = Field{Center, Center, Nothing}(grid)

function cached(build, path)
    if isfile(path)
        @info "loading cached $(basename(path))"
        return jldopen(file -> file["data"], path)
    end
    @info "building $(basename(path))"
    data = build()
    mkpath(dirname(path))
    jldsave(path; data)
    return data
end

flat_grid(size, longitude, latitude) =
    LatitudeLongitudeGrid(CPU(); size, longitude, latitude, topology = (Bounded, Bounded, Flat))

finite_mean(values) = (finite = filter(isfinite, values); isempty(finite) ? NaN32 : Float32(mean(finite)))
finite_median(values) = median(filter(isfinite, values))

function block_reduce(f, a, n)
    Nx, Ny = size(a) .÷ n
    return [f(view(a, (n * (i - 1) + 1):(n * i), (n * (j - 1) + 1):(n * j))) for i in 1:Nx, j in 1:Ny]
end

# Area-weighted mean of the valid source cells under each target cell (conservative
# regrid of the field and of its validity mask); target cells with no valid source stay NaN.
function masked_regrid!(target, source)
    values = surface_field(source.grid)
    valid  = surface_field(source.grid)
    data = Array(interior(source, :, :, 1))
    interior(values, :, :, 1) .= ifelse.(isfinite.(data), data, 0)
    interior(valid, :, :, 1)  .= isfinite.(data)
    weight = surface_field(target.grid)
    regrid!(weight, valid)
    regrid!(target, values)
    interior(target) .= ifelse.(interior(weight) .> 0, interior(target) ./ interior(weight), NaN)
    return target
end

fill_invalid!(field, value) = (parent(field) .= ifelse.(isfinite.(parent(field)), parent(field), value); field)

##### Soil hydraulics

function ingest_soil(grid, region; slab_depth)
    Nx, Ny = size(grid)
    domain = BoundingBox(grid)
    lattice = LatitudeLongitudeGrid(CPU(), Float64; size = (Nx, Ny, 3),
                                    longitude = domain.longitude, latitude = domain.latitude,
                                    z = [-1.0, -0.6, -0.3, 0.0],
                                    topology = (Bounded, Bounded, Bounded))

    dataset = OpenLandMapSoilDB(aggregation_factor = nothing)
    texture = map((:sand_fraction, :silt_fraction, :clay_fraction, :bulk_density)) do name
        field = Field(Metadatum(name; dataset, region), lattice; cache = true)
        gaps = Field{Center, Center, Center}(lattice, Bool)
        interior(gaps) .= .!isfinite.(interior(field))
        inpaint_mask!(field, gaps)
        return field
    end
    sand, silt, clay, bulk_density = texture

    hydraulics = soil_hydraulic_properties(sand, silt, clay, bulk_density; slab_depth)
    to_surface(f, k = 1) = (g = surface_field(grid); interior(g, :, :, 1) .= interior(f, :, :, k); g)
    hydraulic_fields = map(to_surface, hydraulics)

    ## Macropore-inclusive Cosby conductivity of the 0–30 cm layer caps infiltration (m s⁻¹ × ρˡ).
    cosby = compute!(Field(saturated_conductivity(CosbyConductivity(Float64), sand)))
    infiltration_capacity = 1000 * finite_median(filter(>(0), Array(interior(cosby, :, :, 3))))

    ## Dry areal heat capacity of a 0.15 m diurnal skin from the 0–30 cm bulk density.
    dry_heat_capacity = surface_field(grid)
    interior(dry_heat_capacity, :, :, 1) .= 840 * 0.15 .* interior(bulk_density, :, :, 3)

    texture_fields = map(f -> to_surface(f, 3), (; sand, silt, clay, bulk_density))

    return (; hydraulic_fields, infiltration_capacity, dry_heat_capacity, texture_fields,
              scalar_porosity = finite_median(filter(>(0), Array(interior(hydraulics.porosity)))))
end

##### MODIS land cover and leaf area index

function ingest_modis(grid, region, fill_cache)
    classes = Field(Metadatum(:landcover_class; dataset = MCD12Q1(), region, date = DateTime(2011)), CPU())
    codes = Array(interior(classes, :, :, 1))
    codes .= ifelse.(isfinite.(codes), codes, igbp_class_names.water)

    indicator = surface_field(classes.grid)
    fractions = map(igbp_class_names) do code
        interior(indicator, :, :, 1) .= codes .== code
        fraction = surface_field(grid)
        regrid!(fraction, indicator)
        return fraction
    end

    stacked = cat((interior(f, :, :, 1) for f in fractions)...; dims = 3)
    majority = [keys(igbp_class_names)[argmax(view(stacked, i, j, :))] for i in axes(stacked, 1), j in axes(stacked, 2)]

    ## Five 8-day composites around the case day, cloud gaps filled from each cell's own
    ## series and same-class neighbors, non-vegetated classes zeroed, case composite regridded.
    stamps = [DateTime(2011, 5, 1), DateTime(2011, 5, 9), DateTime(2011, 5, 17),
              DateTime(2011, 5, 25), DateTime(2011, 6, 2)]
    series = FieldTimeSeries{Center, Center, Nothing}(classes.grid, Dates.datetime2unix.(stamps))
    for (n, date) in enumerate(stamps)
        composite = Field(Metadatum(:leaf_area_index; dataset = MCD15A2H(), region, date), CPU())
        parent(series[n]) .= parent(composite)
        composite = nothing
        GC.gc()
    end
    fill_seasonal_gaps!(series, classes; cyclic = false, cache = fill_cache,
                        max_gap = class_maximum_gap(classes), valid_range = (0, 10),
                        unfilled_classes = igbp_non_vegetated_classes)
    zero_non_vegetated!(series, classes)

    leaf_area_index = surface_field(grid)
    masked_regrid!(leaf_area_index, series[3])
    fill_invalid!(leaf_area_index, 0)
    parent(leaf_area_index) .= clamp.(parent(leaf_area_index), 0, 10)

    return (; fractions, majority, leaf_area_index)
end

##### ESA WorldCover class fractions, stitched from the cached 3° tiles

function ingest_worldcover(grid, region)
    corners = worldcover_tile_corners(region.longitude, region.latitude)
    available = filter(c -> isfile(metadata_path(worldcover_tile_metadatum(:vegetation_fraction, c))), corners)
    @info "WorldCover: $(length(available)) of $(length(corners)) tiles cached (the rest are ocean or missing)"
    isempty(available) && return nothing

    Δ = worldcover_lattice_step
    λ₀, φ₀ = minimum(first, corners), minimum(last, corners)
    nλ = round(Int, (maximum(first, corners) + 3 - λ₀) / Δ)
    nφ = round(Int, (maximum(last, corners) + 3 - φ₀) / Δ)
    lattice = flat_grid((nλ, nφ), (λ₀, λ₀ + nλ * Δ), (φ₀, φ₀ + nφ * Δ))
    source = surface_field(lattice)

    names = (map(name -> Symbol(name, :_fraction), keys(ESA_WORLDCOVER_CLASS_NAMES))..., :vegetation_fraction)
    fields = map(names) do name
        parent(source) .= NaN
        for corner in available
            tile = Field(worldcover_tile_metadatum(name, corner), CPU())
            λ = λnodes(tile.grid, Center()); φ = φnodes(tile.grid, Center())
            data = Array(interior(tile, :, :, 1))
            for (b, φb) in enumerate(φ), (a, λa) in enumerate(λ)
                corner[1] < λa < corner[1] + 3 && corner[2] < φb < corner[2] + 3 || continue
                i = round(Int, (λa - λ₀) / Δ + 1//2); j = round(Int, (φb - φ₀) / Δ + 1//2)
                source[i, j, 1] = data[a, b]
            end
        end
        target = surface_field(grid)
        return masked_regrid!(target, source)
    end
    return NamedTuple{(keys(ESA_WORLDCOVER_CLASS_NAMES)..., :vegetation_fraction)}(fields)
end

##### GHSL urban morphometry → roughness

# Building height and built fraction are read in sub-boxes at their native 100 m and binned
# onto a lattice 10× finer than the land grid (~1.2 km, the scale of the morphometric
# regressions): the plan-area index is the pixel mean of the built fraction and the building
# height is the built-area-weighted mean (building volume over built area). The roughness
# closure is evaluated there, then reduced to the land grid: the built-up land fraction, the
# log-mean roughness of the built pixels, and the mean building height.
function ingest_urban(grid; fine = 30, coarse = 10, box = (2700, 2160))
    Nx, Ny = size(grid)
    domain = BoundingBox(grid)
    λ₁, λ₂ = domain.longitude
    φ₁, φ₂ = domain.latitude
    nx, ny = fine * Nx, fine * Ny
    δλ, δφ = (λ₂ - λ₁) / nx, (φ₂ - φ₁) / ny

    n = fine ÷ coarse
    Δλ, Δφ = n * δλ, n * δφ
    volume = zeros(Float64, nx ÷ n, ny ÷ n)
    area   = zeros(Float64, nx ÷ n, ny ÷ n)
    pixels = zeros(Int, nx ÷ n, ny ÷ n)
    for j in Iterators.partition(1:ny, box[2]), i in Iterators.partition(1:nx, box[1])
        longitude = (λ₁ + (first(i) - 1) * δλ, λ₁ + last(i) * δλ)
        latitude  = (φ₁ + (first(j) - 1) * δφ, φ₁ + last(j) * δφ)
        box_region = BoundingBox(flat_grid((length(i), length(j)), longitude, latitude); padding = 0.01)
        h  = Field(Metadatum(:building_height;   dataset = GHSBuiltH(), region = box_region), CPU())
        λᵖ = Field(Metadatum(:built_up_fraction; dataset = GHSBuiltS(), region = box_region), CPU())
        size(h) == size(λᵖ) || error("GHSL height and built-fraction rasters differ in size for $box_region")
        λn = λnodes(h.grid, Center()); φn = φnodes(h.grid, Center())
        heights = interior(h, :, :, 1); fractions = interior(λᵖ, :, :, 1)
        for b in eachindex(φn), a in eachindex(λn)
            longitude[1] <= λn[a] < longitude[2] && latitude[1] <= φn[b] < latitude[2] || continue
            f = fractions[a, b]
            isfinite(f) || continue
            ic = floor(Int, (λn[a] - λ₁) / Δλ) + 1
            jc = floor(Int, (φn[b] - φ₁) / Δφ) + 1
            pixels[ic, jc] += 1
            area[ic, jc] += f
            volume[ic, jc] += ifelse(isfinite(heights[a, b]), f * heights[a, b], 0)
        end
        @info "GHSL box λ ∈ $longitude, φ ∈ $latitude done"
        h = λᵖ = nothing
        GC.gc()
    end

    km_grid = flat_grid((nx ÷ n, ny ÷ n), domain.longitude, domain.latitude)
    height_km = surface_field(km_grid); interior(height_km, :, :, 1) .= volume ./ area
    built_km  = surface_field(km_grid); interior(built_km, :, :, 1)  .= ifelse.(pixels .> 0, area ./ pixels, NaN)

    closure = MorphometricRoughness(eltype(km_grid))
    ℓᵐ, d = urban_roughness(height_km, built_km; closure)
    fill_aerodynamic_roughness_gaps!(ℓᵐ, d, closure)

    ## Built-area-weighted means over each land cell, so the dense core sets the roughness
    ## and the height rather than the many sparsely built pixels around it.
    plan_area = Array(interior(built_km, :, :, 1))
    urban = plan_area .>= 0.01
    weights = ifelse.(urban, plan_area, 0)
    weighted_mean(a) = block_reduce(sum, ifelse.(isfinite.(a), a .* weights, 0), coarse) ./ block_reduce(sum, weights, coarse)

    to_field(a) = (f = surface_field(grid); interior(f, :, :, 1) .= a; f)
    return (; urban_fraction   = to_field(block_reduce(mean, urban, coarse)),
              urban_roughness  = to_field(exp.(weighted_mean(log.(Array(interior(ℓᵐ, :, :, 1)))))),
              building_height  = to_field(weighted_mean(Array(interior(height_km, :, :, 1)))),
              plan_area_index  = to_field(block_reduce(finite_mean, plan_area, coarse)))
end

##### Radiative surface properties

function ingest_albedo(grid, region)
    native = Field(Metadatum(:albedo; dataset = CopernicusAlbedo(), region, date = DateTime(2011, 5, 20)), CPU())
    albedo = surface_field(grid)
    masked_regrid!(albedo, native)
    return (; albedo)
end

function ingest_emissivity(grid, region)
    native = Field(Metadatum(:emissivity; dataset = ASTERGEDv3(), region), CPU();
                   inpainting = NearestNeighborInpainting(20))
    emissivity = surface_field(grid)
    masked_regrid!(emissivity, native)
    return (; emissivity)
end

##### ERA5-Land initial and deep soil state (0.1°, ocean masked → filled before the regrid)

function ingest_era5_land(grid, region, dir)
    function land_field(name, fallback)
        native = Field(Metadatum(name; dataset = ERA5MonthlyLand(), region, date = DateTime(2011, 5, 1), dir), CPU())
        fill_invalid!(native, fallback)
        target = surface_field(grid)
        interpolate!(target, native)
        return fill_invalid!(target, fallback)
    end

    ## ERA5-Land layers 0–7, 7–28, 28–100 cm sampled over a 0–50 cm slab.
    soil_water = surface_field(grid)
    for (name, weight) in zip((:volumetric_soil_water_layer_1, :volumetric_soil_water_layer_2,
                               :volumetric_soil_water_layer_3), (0.14, 0.42, 0.44))
        parent(soil_water) .+= weight .* parent(land_field(name, 0.25))
    end
    deep_temperature = land_field(:soil_temperature_level_3, 288)

    return (; soil_water, deep_temperature)
end

##### ETH canopy height, stitched from the cached 0.01° tiles

# Cell-mean tree height (pixels without trees count as 0 m, no-data is excluded) and the
# share of the cell's ~1 km lattice cells whose mean canopy stands at least 2 m tall.
function ingest_canopy_height(grid, region)
    corners = worldcover_tile_corners(region.longitude, region.latitude)
    available = filter(c -> isfile(eth_canopy_tile_path(c)), corners)
    @info "ETH canopy: $(length(available)) of $(length(corners)) tiles cached (the rest are ocean or missing)"
    isempty(available) && return nothing

    Δ = eth_canopy_lattice_step
    λ₀, φ₀ = minimum(first, corners), minimum(last, corners)
    nλ = round(Int, (maximum(first, corners) + 3 - λ₀) / Δ)
    nφ = round(Int, (maximum(last, corners) + 3 - φ₀) / Δ)
    lattice = flat_grid((nλ, nφ), (λ₀, λ₀ + nλ * Δ), (φ₀, φ₀ + nφ * Δ))
    height = surface_field(lattice); parent(height) .= NaN
    for corner in available
        tile = jldopen(f -> f["height"], eth_canopy_tile_path(corner))
        i₀ = round(Int, (corner[1] - λ₀) / Δ); j₀ = round(Int, (corner[2] - φ₀) / Δ)
        interior(height, (i₀ + 1):(i₀ + size(tile, 1)), (j₀ + 1):(j₀ + size(tile, 2)), 1) .= tile
    end

    tall = surface_field(lattice)
    interior(tall, :, :, 1) .= ifelse.(isfinite.(interior(height, :, :, 1)), interior(height, :, :, 1) .>= 2, NaN)

    eth_canopy_height = surface_field(grid); masked_regrid!(eth_canopy_height, height)
    tall_canopy_fraction = surface_field(grid); masked_regrid!(tall_canopy_fraction, tall)
    return (; eth_canopy_height, tall_canopy_fraction)
end
