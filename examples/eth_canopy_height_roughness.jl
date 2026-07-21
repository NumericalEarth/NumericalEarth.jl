# # Aerodynamic roughness from ETH canopy height
#
# Ingest the ETH global 10 m canopy-height model over a forest/farmland region, then derive
# the momentum roughness length `z₀` and zero-plane displacement `d₀` per cell with the
# Raupach (1994) drag-partition closure. The canopy height enters the closure as a native
# per-cell field: the measured height sets the roughness *scale* where it is observed, and
# the class-average height fills in only where the observation is missing. The headline is
# the difference between roughness built from the measured height field and roughness built
# from the class-average height alone.
#
# Requirements:
#   * `using ArchGDAL` — windowed anonymous read of the ETH Cloud-Optimized GeoTIFF (network,
#     no credentials)
#   * `using CairoMakie` — figures
#
# Leaf-area index and IGBP land cover would come from MODIS (`MCD15A3H`, `MCD12Q1(:IGBP)`).
# Here land cover is a uniform deciduous-broadleaf class and LAI a smooth gradient, so the
# maps isolate the effect of the measured canopy-height field; swapping in the MODIS fields
# makes the land-cover and LAI inputs data-driven with no change to the closure call.

using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, Metadatum
using NumericalEarth.DataWrangling.CanopyHeight: ETHCanopyHeight
using NumericalEarth.DataWrangling.CanopyRoughness: compute_canopy_roughness!
using Oceananigans
using Oceananigans.Fields: set!, interior
using Oceananigans.Grids: λnodes, φnodes
using ArchGDAL
using CairoMakie
using Statistics: mean

outdir = joinpath(@__DIR__, "eth_canopy_roughness_figures")
mkpath(outdir)

# ## Inputs
# A 0.5°×0.5° box over the Bavarian pre-Alps: Alpine forest to the south, farmland to the
# north — a legible tall-canopy / short-canopy mix. The target grid coarsens the ETH mosaic
# (0.001°, ~111 m) to ~500 m cells.
region = BoundingBox(longitude = (11.0, 11.5), latitude = (47.5, 48.0))
grid = LatitudeLongitudeGrid(CPU(), Float64; size = (100, 100),
                             longitude = (11.0, 11.5), latitude = (47.5, 48.0),
                             topology = (Bounded, Bounded, Flat))

# Measured canopy height `h_c` from ETH (anonymous COG, windowed to the region).
canopy_height = Field(Metadatum(:canopy_height; dataset = ETHCanopyHeight(), region), grid)

# Land cover: uniform deciduous broadleaf forest (IGBP 4). LAI: a smooth gradient spanning
# sparse-to-dense canopy so the skimming behaviour is visible across the domain.
land_cover = Field{Center, Center, Nothing}(grid)
set!(land_cover, 4)

leaf_area_index = Field{Center, Center, Nothing}(grid)
set!(leaf_area_index, (λ, φ) -> 0.5 + 6.5 * (φ - 47.5) / 0.5)

# ## Roughness with the measured height field vs the class-average height alone
z0m, d0 = Field{Center, Center, Nothing}(grid), Field{Center, Center, Nothing}(grid)
compute_canopy_roughness!(z0m, d0, leaf_area_index, land_cover, canopy_height, grid)

z0m_class, d0_class = Field{Center, Center, Nothing}(grid), Field{Center, Center, Nothing}(grid)
compute_canopy_roughness!(z0m_class, d0_class, leaf_area_index, land_cover, grid)

# ## Figures
λ = Array(λnodes(grid, Center())); φ = Array(φnodes(grid, Center()))
arr(f) = Array(interior(f, :, :, 1))
finite(a) = filter(isfinite, vec(a))

hc = arr(canopy_height)
lai = arr(leaf_area_index)
Z, D = arr(z0m), arr(d0)
Zc = arr(z0m_class)
ΔZ = Z .- Zc

@info "ETH canopy height (m)" min=minimum(finite(hc)) mean=mean(finite(hc)) max=maximum(finite(hc))
@info "z₀ from measured height (m)" min=minimum(finite(Z)) mean=mean(finite(Z)) max=maximum(finite(Z))
@info "z₀ measured − class (m)" mean=mean(finite(ΔZ)) max=maximum(finite(ΔZ))

function panel!(fig, pos, title, A, crange, cmap, clabel)
    ax = Axis(fig[pos...]; title, aspect = DataAspect()); hidedecorations!(ax)
    hm = heatmap!(ax, λ, φ, A; colorrange = crange, colormap = cmap, nan_color = :gray82)
    Colorbar(fig[pos[1], pos[2] + 1], hm; label = clabel, width = 11)
end

# (1) Inputs and derived roughness: h_c, LAI, z₀, d₀.
hc_range = (0, maximum(finite(hc)))
z0_range = (0, maximum(finite(Z)))
fig = Figure(size = (1400, 900))
Label(fig[0, 1:4], "ETH canopy height → Raupach roughness — Bavarian pre-Alps"; fontsize = 20, font = :bold)
panel!(fig, (1, 1), "canopy height h_c",         hc,  hc_range, :YlGn,    "m")
panel!(fig, (1, 3), "leaf-area index Λ",         lai, (0, 7),   :YlGn,    "m²/m²")
panel!(fig, (2, 1), "z₀ (measured height)",      Z,   z0_range, :viridis, "m")
panel!(fig, (2, 3), "d₀ (measured height)",      D,   (0, maximum(finite(D))), :magma, "m")
save(joinpath(outdir, "fig1_inputs_roughness.png"), fig)

# (2) The headline: measured-height z₀ vs class-height z₀, and their difference.
Δmax = maximum(abs, finite(ΔZ))
fig = Figure(size = (1500, 470))
Label(fig[0, 1:6], "Measured canopy height moves the roughness"; fontsize = 20, font = :bold)
panel!(fig, (1, 1), "z₀ (measured h_c)",                 Z,  z0_range, :viridis, "m")
panel!(fig, (1, 3), "z₀ (class-average height)",         Zc, z0_range, :viridis, "m")
panel!(fig, (1, 5), "z₀ measured − class",               ΔZ, (-Δmax, Δmax), :balance, "m")
save(joinpath(outdir, "fig2_measured_vs_class.png"), fig)

# (3) Scale and regime dependence: z₀ vs h_c, z₀ vs LAI, d₀ vs LAI.
mask = isfinite.(vec(hc)) .& isfinite.(vec(Z))
fig = Figure(size = (1500, 460))
ax1 = Axis(fig[1, 1]; xlabel = "canopy height h_c (m)", ylabel = "z₀ (m)", title = "z₀ scales with height")
scatter!(ax1, vec(hc)[mask], vec(Z)[mask]; markersize = 4, color = (:navy, 0.3))
ax2 = Axis(fig[1, 2]; xlabel = "LAI (m²/m²)", ylabel = "z₀ (m)", title = "z₀ non-monotonic in LAI (skimming)")
scatter!(ax2, vec(lai)[mask], vec(Z)[mask]; markersize = 4, color = (:seagreen, 0.3))
ax3 = Axis(fig[1, 3]; xlabel = "LAI (m²/m²)", ylabel = "d₀ (m)", title = "d₀ monotone in LAI")
scatter!(ax3, vec(lai)[mask], vec(D)[mask]; markersize = 4, color = (:firebrick, 0.3))
save(joinpath(outdir, "fig3_scale_regime.png"), fig)

@info "wrote figures" outdir
