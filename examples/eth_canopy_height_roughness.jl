# # Aerodynamic roughness from ETH canopy height
#
# Derive momentum roughness length `z₀` and zero-plane displacement `d₀` from the ETH global
# 10 m canopy-height model with the Raupach (1994) drag-partition closure. Canopy height is a
# native per-cell field: the measured height sets the roughness scale where observed, and the
# class-average height fills cells where it is missing. The headline contrasts `z₀` built from
# the measured height field against `z₀` built from a single class-average height.
#
# Only the canopy height is real data. Leaf-area index and IGBP land cover are synthetic
# stand-ins for MODIS (`MCD15A3H`, `MCD12Q1(:IGBP)`), which the closure would otherwise take
# unchanged. Needs `using ArchGDAL` (anonymous ETH COG read; network, no credentials) and
# `using CairoMakie` (figures).

using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, Metadatum
using NumericalEarth.DataWrangling.CanopyHeight: ETHCanopyHeight
using NumericalEarth.DataWrangling.CanopyRoughness: compute_canopy_roughness!,
                                                    canopy_roughness, canopy_drag_parameters,
                                                    class_canopy_height
using Oceananigans
using Oceananigans.Fields: set!, interior
using Oceananigans.Grids: λnodes, φnodes
using Statistics: mean
using ArchGDAL, CairoMakie

# ## Inputs
# A 0.5°×0.5° box over the Bavarian pre-Alps (Alpine forest + farmland), on a 500×500 grid
# matching the ETH mosaic's 0.001° (~110 m) native resolution.
region = BoundingBox(longitude = (11.0, 11.5), latitude = (47.5, 48.0))
grid = LatitudeLongitudeGrid(CPU(); size = (500, 500), longitude = (11.0, 11.5),
                             latitude = (47.5, 48.0), topology = (Bounded, Bounded, Flat))

canopy_height = Field(Metadatum(:canopy_height; dataset = ETHCanopyHeight(), region), grid)

land_cover = Field{Center, Center, Nothing}(grid); set!(land_cover, 4)   # deciduous broadleaf (synthetic)
lai = Field{Center, Center, Nothing}(grid)
set!(lai, (λ, φ) -> 0.5 + 6.5 * (φ - 47.5) / 0.5)                         # synthetic south→north sweep

# ## Roughness: measured height field vs the class-average height alone
z0, d0 = Field{Center, Center, Nothing}(grid), Field{Center, Center, Nothing}(grid)
compute_canopy_roughness!(z0, d0, lai, land_cover, canopy_height, grid)

z0_class, d0_class = Field{Center, Center, Nothing}(grid), Field{Center, Center, Nothing}(grid)
compute_canopy_roughness!(z0_class, d0_class, lai, land_cover, grid)      # omit height → class-average height

# ## Figures
outdir = joinpath(@__DIR__, "eth_canopy_roughness_figures"); mkpath(outdir)
λ, φ = Array(λnodes(grid, Center())), Array(φnodes(grid, Center()))
arr(f) = Array(interior(f, :, :, 1))
finite(a) = filter(isfinite, vec(a))
hclass = class_canopy_height(Float64, 4)   # deciduous-broadleaf class height (m)

function heat!(fig, pos, title, A, crange, cmap)
    ax = Axis(fig[pos...]; title, aspect = DataAspect()); hidedecorations!(ax)
    hm = heatmap!(ax, λ, φ, A; colorrange = crange, colormap = cmap, nan_color = :gray82)
    Colorbar(fig[pos[1], pos[2] + 1], hm; width = 11)
end

# (1) Headline — measured height moves the roughness. Gray in `h_c` is ETH no-data (lakes,
# unobserved pixels); the closure fills those cells with the class-average height, so
# `z₀` carries no gaps.
H, Z, Zc = arr(canopy_height), arr(z0), arr(z0_class)
zrange = (0, maximum(finite(Z)))
Δ = Z .- Zc; Δmax = maximum(abs, finite(Δ))
fig = Figure(size = (1250, 1150))
Label(fig[0, 1:4], "Measured ETH canopy height moves the roughness"; fontsize = 20, font = :bold)
heat!(fig, (1, 1), "canopy height h_c (ETH)",                                  H,  (0, maximum(finite(H))), :YlGn)
heat!(fig, (1, 3), "z₀ from measured h_c",                                     Z,  zrange, :viridis)
heat!(fig, (2, 1), "z₀ from class-average height ($(round(hclass, digits=1)) m, no h_c field)", Zc, zrange, :viridis)
heat!(fig, (2, 3), "z₀ difference (measured − class)",                         Δ,  (-Δmax, Δmax), :balance)
save(joinpath(outdir, "fig1_roughness_maps.png"), fig)

# (2) The closure's LAI regime dependence, isolated at fixed heights (deciduous broadleaf →
# drag group 2). z₀ peaks near Λ≈1.2 then is capped at Λₘₐₓ; d₀ rises monotonically then caps.
p = canopy_drag_parameters(Float64, 2); Λs = 0:0.05:7
z0_of(h) = [canopy_roughness(Λ, h, p, 0.4, 0.193, 20)[1] for Λ in Λs]
d0_of(h) = [canopy_roughness(Λ, h, p, 0.4, 0.193, 20)[2] for Λ in Λs]
fig = Figure(size = (1000, 440))
axz = Axis(fig[1, 1]; xlabel = "LAI (m²/m²)", ylabel = "z₀ (m)", title = "z₀ vs LAI: skimming peak, then Λ-capped")
axd = Axis(fig[1, 2]; xlabel = "LAI (m²/m²)", ylabel = "d₀ (m)", title = "d₀ vs LAI: monotone, then Λ-capped")
for h in (5.0, 15.0, 30.0)
    lines!(axz, Λs, z0_of(h); linewidth = 2, label = "h = $(Int(h)) m")
    lines!(axd, Λs, d0_of(h); linewidth = 2, label = "h = $(Int(h)) m")
end
vlines!(axz, [p.maximum_area_index]; color = :gray50, linestyle = :dash)
vlines!(axd, [p.maximum_area_index]; color = :gray50, linestyle = :dash)
axislegend(axz; position = :rt); axislegend(axd; position = :rb)
save(joinpath(outdir, "fig2_lai_regime.png"), fig)

@info "canopy roughness" h_c_max = round(maximum(finite(H)), digits = 1) z0_measured_mean = round(mean(finite(Z)), digits = 2) z0_class = round(mean(finite(Zc)), digits = 2) Δz0_mean = round(mean(finite(Δ)), digits = 2)
