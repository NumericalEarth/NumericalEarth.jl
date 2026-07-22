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
using Statistics: mean
using ArchGDAL, CairoMakie

# ## Inputs
# A 0.1°×0.1° box over the Bavarian pre-Alps (forest + farmland). The canopy height is read on
# the ETH product's native ~10 m grid (a windowed COG) and everything downstream runs on that
# same grid — no resampling, so the maps show the true 10 m detail.
region = BoundingBox(longitude = (11.0, 11.1), latitude = (47.6, 47.7))
canopy_height = Field(Metadatum(:canopy_height; dataset = ETHCanopyHeight(), region), CPU())
grid = canopy_height.grid   # the ETH native 10 m grid over the window (~1200 × 1200)

φ₁, φ₂ = region.latitude
land_cover = Field{Center, Center, Nothing}(grid); set!(land_cover, 4)   # deciduous broadleaf (synthetic)
lai = Field{Center, Center, Nothing}(grid)
set!(lai, (λ, φ) -> clamp(0.5 + 6.5 * (φ - φ₁) / (φ₂ - φ₁), 0.5, 7.0))    # synthetic south→north sweep

# ## Roughness: measured height field vs the class-average height alone
z0, d0 = Field{Center, Center, Nothing}(grid), Field{Center, Center, Nothing}(grid)
compute_canopy_roughness!(z0, d0, lai, land_cover, canopy_height, grid)

z0_class, d0_class = Field{Center, Center, Nothing}(grid), Field{Center, Center, Nothing}(grid)
compute_canopy_roughness!(z0_class, d0_class, lai, land_cover, grid)      # omit height → class-average height

# ## Figures
outdir = joinpath(@__DIR__, "eth_canopy_roughness_figures"); mkpath(outdir)
finite(f) = filter(isfinite, vec(Array(interior(f))))   # for color-range scalars only
hclass = class_canopy_height(Float64, 4)                # deciduous-broadleaf class height (m)

function heat!(fig, pos, title, field, crange, cmap)
    ax = Axis(fig[pos...]; title, aspect = DataAspect()); hidedecorations!(ax)
    hm = heatmap!(ax, field; colorrange = crange, colormap = cmap, nan_color = :gray82)
    Colorbar(fig[pos[1], pos[2] + 1], hm; width = 11)
end

# (1) Headline — measured height moves the roughness. Gray in `h_c` is ETH no-data (lakes,
# unobserved pixels); the closure fills those cells with the class-average height, so
# `z₀` carries no gaps.
zmax = maximum(finite(z0))
Δz0 = z0 - z0_class
Δmax = maximum(abs, filter(isfinite, vec(Array(interior(z0)) .- Array(interior(z0_class)))))
fig = Figure(size = (1250, 1150))
Label(fig[0, 1:4], "Measured ETH canopy height moves the roughness (native 10 m)"; fontsize = 20, font = :bold)
heat!(fig, (1, 1), "canopy height h_c (ETH)",                                  canopy_height, (0, maximum(finite(canopy_height))), :YlGn)
heat!(fig, (1, 3), "z₀ from measured h_c",                                     z0,       (0, zmax), :viridis)
heat!(fig, (2, 1), "z₀ from class-average height ($(round(hclass, digits=1)) m, no h_c field)", z0_class, (0, zmax), :viridis)
heat!(fig, (2, 3), "z₀ difference (measured − class)",                         Δz0,      (-Δmax, Δmax), :balance)
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

@info "canopy roughness (native 10 m)" h_c_max = round(maximum(finite(canopy_height)), digits = 1) z0_measured_mean = round(mean(finite(z0)), digits = 2) z0_class = round(mean(finite(z0_class)), digits = 2)
