# Maps of every ingested surface parameter the Borneo model reads, from the cached
# `static_r<REFINEMENT>.jld2` and `forcing_r<REFINEMENT>.jld2`.

include(joinpath(@__DIR__, "borneo_config.jl"))
using CairoMakie
using Statistics: mean
using Printf

static  = load_static()
forcing = load_cache("forcing")
λ, φ = static.longitude, static.latitude
land = .!static.water
mask(a) = ifelse.(land, a, NaN)

panels = [
    ("Elevation, ETOPO 2022 (m)",            forcing.land_elevation,                        :terrain,  nothing),
    ("ERA5 model elevation (m)",              forcing.era5_elevation,                        :terrain,  nothing),
    ("MODIS leaf area index",                 mask(static.leaf_area_index),                  :YlGn,     (0, 7)),
    ("Vegetated fraction (MODIS classes)",    mask(static.vegetation_fraction),              :YlGn,     (0, 1)),
    ("Canopy height, ETH Sentinel-2 (m)",     mask(static.canopy_height),                    :viridis,  nothing),
    ("Class canopy height, IGBP (m)",         mask(static.class_canopy_height),              :viridis,  nothing),
    ("Vegetated ℓᵐ, Raupach (m)",             mask(static.vegetated_roughness_length),       :viridis,  nothing),
    ("Bare-tile ℓᵐ, GHSL/water/soil (m)",     mask(log10.(static.bare_roughness_length)),    :viridis,  nothing),
    ("Built-up fraction, GHSL",               mask(static.urban_fraction),                   :magma,    nothing),
    ("Albedo, Copernicus (Apr 2020)",         mask(static.albedo),                           :cividis,  nothing),
    ("Emissivity, ASTER GED",                 mask(static.emissivity),                       :cividis,  nothing),
    ("Porosity ν, Weynants PTF",              mask(static.porosity),                         :tempo,    nothing),
    ("Sand fraction 0–30 cm, OpenLandMap",    mask(static.sand),                             :copper,   nothing),
    ("log₁₀ Kₘ (m s⁻¹)",                      mask(log10.(static.matching_point_conductivity)), :viridis, nothing),
    ("van Genuchten n",                       mask(static.pore_size_uniformity),             :viridis,  nothing),
    ("Initial θ, ERA5-Land 0–28 cm",          mask(static.initial_soil_water),               :tempo,    nothing),
    ("Deep soil temperature, ERA5-Land (K)",  mask(static.deep_temperature),                 :thermal,  nothing),
    ("ERA5 skin temperature at start (K)",    forcing.skin_temperature,                      :thermal,  nothing),
]

fig = Figure(size = (2400, 1500), fontsize = 14)
Label(fig[0, 1:12], @sprintf("Central Borneo surface ingestion at ≈ %d km (%d × %d cells); grey = water", resolution_km, Nx, Ny); fontsize = 20)
for (k, (title, data, colormap, colorrange)) in enumerate(panels)
    row, col = fldmod1(k, 6)
    ax = Axis(fig[row, 2col - 1]; title, aspect = DataAspect())
    hm = isnothing(colorrange) ? heatmap!(ax, λ, φ, data; colormap) : heatmap!(ax, λ, φ, data; colormap, colorrange)
    heatmap!(ax, λ, φ, ifelse.(land, NaN, 1.0); colormap = [:gray70, :gray70])
    Colorbar(fig[row, 2col], hm)
    (row < 3) && hidexdecorations!(ax; grid = false)
    (col > 1) && hideydecorations!(ax; grid = false)
end
save("ingestion_r$(refinement).png", fig)

classes = sort(collect(Set(vec(static.canopy_class[land]))); by = c -> -count(==(c), static.canopy_class[land]))
@info "majority vegetated classes: " * join(["$(c) ($(count(==(c), static.canopy_class[land])))" for c in classes], ", ")
@info @sprintf("land cells %d / %d; LAI mean %.2f; canopy height mean %.1f m (ETH valid %.0f%%); f_veg mean %.2f; ℓᵐ veg median %.2f m; θ₀ mean %.3f; ν mean %.3f",
               count(land), length(land), mean(static.leaf_area_index[land]), mean(static.canopy_height[land]),
               100 * count(isfinite, static.eth_canopy_height[land]) / count(land), mean(static.vegetation_fraction[land]),
               median(static.vegetated_roughness_length[land]), mean(static.initial_soil_water[land]), mean(static.porosity[land]))
@info "saved ingestion_r$(refinement).png"
