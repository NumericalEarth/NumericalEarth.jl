# # Vegetation roughness climatology from MODIS LAI
#
# End-to-end demo of this PR: download MODIS leaf-area index + IGBP land cover, derive the
# momentum roughness length `z₀` and zero-plane displacement `d₀` per pixel with the Raupach
# (1994) drag-partition closure (parameterized after Jasinski et al. 2005; reproduces Borak
# et al. 2025), gap-fill the seasonal cycle, and render the maps + animation.
#
# Requirements:
#   * `using ArchGDAL` with a GDAL_jll HDF4 driver (`"HDF4" in keys(ArchGDAL.listdrivers())`)
#   * NASA Earthdata credentials in `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`
#     (register free at https://urs.earthdata.nasa.gov)
#   * `using CairoMakie` for the figures
# The first run downloads ~13 MODIS tiles (cached afterwards); the rest is seconds.

using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, Metadatum
using NumericalEarth.DataWrangling.MODISLand: MCD15A2H, MCD12Q1
using NumericalEarth.DataWrangling.CanopyRoughness: canopy_roughness_climatology, fill_temporal_gaps!
using Oceananigans
using Oceananigans.Fields: set!, interior
using Oceananigans.Grids: λnodes, φnodes
using Oceananigans.OutputReaders: FieldTimeSeries
using ArchGDAL
using CairoMakie
using Statistics: mean
using Printf
using Dates

outdir = joinpath(@__DIR__, "lai_roughness_figures")
mkpath(outdir)

# ## Inputs
# A small demonstration box over the Missouri Ozarks (deciduous forest + cropland — strong
# seasonality), on a target grid near the 500 m native resolution.
region = BoundingBox(longitude = (-92.0, -91.0), latitude = (37.0, 38.0))
grid = LatitudeLongitudeGrid(CPU(), Float32; size = (96, 96),
                             longitude = (-92.0, -91.0), latitude = (37.0, 38.0),
                             topology = (Bounded, Bounded, Flat))

# Static IGBP land cover; the `:IGBP` legend matches the closure's parameter tables.
land_cover = Field(Metadatum(:landcover_igbp; dataset = MCD12Q1(legend = :IGBP),
                             region, date = DateTime(2020, 1, 1)), grid)

# One 8-day LAI composite per ~month across 2020 (composite start DOYs, spaced 32 days).
doys  = collect(9:32:361)
dates = [DateTime(2020) + Day(d - 1) for d in doys]
month_index = month.(dates)

lai = FieldTimeSeries{Center, Center, Nothing}(grid, Float64.(doys .* 86400))
for (n, date) in enumerate(dates)
    set!(lai[n], Field(Metadatum(:leaf_area_index; dataset = MCD15A2H(), region, date), grid))
end

# ## Closure + gap-fill
# Roughness before gap-filling — cloud/snow-screened LAI leaves honest NaN gaps.
z0_raw, _ = canopy_roughness_climatology(lai, land_cover)

# Mask out-of-range LAI to NaN, then repair per pixel by cyclic temporal interpolation across
# the seasonal cycle, and rebuild. `missing_fraction` is reported, never silently dropped.
for n in eachindex(dates)
    A = interior(lai[n])
    A[.!isfinite.(A) .| (A .< 0) .| (A .> 10)] .= NaN
end
missing_fraction = fill_temporal_gaps!(lai)
@info "LAI missing fraction before gap-fill" missing_fraction

z0, d0 = canopy_roughness_climatology(lai, land_cover)

# ## Figures
λ = Array(λnodes(grid, Center())); φ = Array(φnodes(grid, Center()))
names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
lab(m) = names[month_index[m]]
sl(fts, m) = Array(interior(fts[m], :, :, 1))
finite(a) = filter(isfinite, vec(a))
lai_range, z0_range, d0_range = (0, 6), (0, 2), (0, 15)

function panel!(fig, pos, title, A, crange, cmap, clabel)
    ax = Axis(fig[pos...]; title, aspect = DataAspect()); hidedecorations!(ax)
    hm = heatmap!(ax, λ, φ, A; colorrange = crange, colormap = cmap, nan_color = :gray82)
    Colorbar(fig[pos[1], pos[2] + 1], hm; label = clabel, width = 11)
end

# (1) Gap-filling before/after, for the two cloudiest raw months.
worst = sortperm([count(!isfinite, sl(z0_raw, m)) for m in eachindex(dates)]; rev = true)[1:2]
fig = Figure(size = (1250, 720))
Label(fig[0, 1:4], "Gap-filling: MODIS QC leaves winter holes; cyclic temporal fill repairs them"; fontsize = 18, font = :bold)
for (r, m) in enumerate(worst)
    panel!(fig, (r, 1), "z₀ raw — $(lab(m))",    sl(z0_raw, m), z0_range, :viridis, "m")
    panel!(fig, (r, 3), "z₀ filled — $(lab(m))", sl(z0, m),     z0_range, :viridis, "m")
end
save(joinpath(outdir, "fig0_gapfill.png"), fig)

# (2) Overview: land cover + winter/summer LAI, z₀, d₀.
land_cover_array = Array(interior(land_cover, :, :, 1))
fig = Figure(size = (1500, 900))
Label(fig[0, 1:6], "MODIS vegetation roughness — Missouri Ozarks (37–38°N, 91–92°W), 2020"; fontsize = 20, font = :bold)
axlc = Axis(fig[1, 1]; title = "IGBP land cover", aspect = DataAspect()); hidedecorations!(axlc)
hmlc = heatmap!(axlc, λ, φ, land_cover_array; colormap = :tab20, colorrange = (1, 17), nan_color = :gray82)
Colorbar(fig[1, 2], hmlc; label = "IGBP", width = 11)
panel!(fig, (1, 3), "LAI — Jan", sl(lai, 1), lai_range, :YlGn, "m²/m²")
panel!(fig, (1, 5), "LAI — Jul", sl(lai, 7), lai_range, :YlGn, "m²/m²")
panel!(fig, (2, 1), "z₀ — Jan",  sl(z0, 1),  z0_range,  :viridis, "m")
panel!(fig, (2, 3), "z₀ — Jul",  sl(z0, 7),  z0_range,  :viridis, "m")
panel!(fig, (2, 5), "d₀ — Jul",  sl(d0, 7),  d0_range,  :magma, "m")
save(joinpath(outdir, "fig1_overview.png"), fig)

# (3) Monthly climatology grids.
function monthly_grid(fts, crange, cmap, title, clabel, fname)
    fig = Figure(size = (1500, 1150)); Label(fig[0, 1:4], title; fontsize = 20, font = :bold)
    for m in eachindex(dates)
        ax = Axis(fig[(m - 1) ÷ 4 + 1, (m - 1) % 4 + 1]; title = lab(m), aspect = DataAspect()); hidedecorations!(ax)
        heatmap!(ax, λ, φ, sl(fts, m); colorrange = crange, colormap = cmap, nan_color = :gray82)
    end
    Colorbar(fig[1:3, 5]; colorrange = crange, colormap = cmap, label = clabel)
    save(joinpath(outdir, fname), fig)
end
monthly_grid(z0, z0_range, :viridis, "Monthly z₀ (m) — Ozarks 2020", "z₀ (m)", "fig2_z0m_monthly.png")
monthly_grid(d0, d0_range, :magma,   "Monthly d₀ (m) — Ozarks 2020", "d₀ (m)", "fig3_d0_monthly.png")

# (4) Domain-mean seasonal cycle.
fmean(fts) = [mean(finite(sl(fts, m))) for m in eachindex(dates)]
fig = Figure(size = (900, 450))
ax1 = Axis(fig[1, 1]; xlabel = "month", ylabel = "LAI (m²/m²)", title = "Domain-mean seasonal cycle — Ozarks 2020", xticks = (1:12, lab.(1:12)))
ax2 = Axis(fig[1, 1]; ylabel = "roughness (m)", yaxisposition = :right); hidespines!(ax2); hidexdecorations!(ax2)
l1 = lines!(ax1, 1:12, fmean(lai); color = :seagreen,  linewidth = 3)
l2 = lines!(ax2, 1:12, fmean(z0);  color = :navy,      linewidth = 3)
l3 = lines!(ax2, 1:12, fmean(d0);  color = :firebrick, linewidth = 3)
axislegend(ax1, [l1, l2, l3], ["LAI", "z₀", "d₀"]; position = :lt)
save(joinpath(outdir, "fig4_seasonal_cycle.png"), fig)

# (5) Animated monthly climatology (LAI | z₀ | d₀).
m_obs = Observable(1)
fig = Figure(size = (1400, 520))
Label(fig[0, 1:3], @lift("MODIS vegetation roughness climatology — Ozarks 2020 — " * lab($m_obs)); fontsize = 18, font = :bold)
for (col, (fts, crange, cmap, name, unit)) in enumerate(((lai, lai_range, :YlGn, "LAI", "m²/m²"),
                                                         (z0,  z0_range,  :viridis, "z₀", "m"),
                                                         (d0,  d0_range,  :magma, "d₀", "m")))
    ax = Axis(fig[1, col]; title = name, aspect = DataAspect()); hidedecorations!(ax)
    hm = heatmap!(ax, λ, φ, @lift(sl(fts, $m_obs)); colorrange = crange, colormap = cmap, nan_color = :gray82)
    Colorbar(fig[2, col], hm; label = unit, vertical = false, flipaxis = false)
end
record(fig, joinpath(outdir, "canopy_roughness_climatology.mp4"), 1:12; framerate = 2) do m
    m_obs[] = m
end

@info "wrote figures + animation" outdir
