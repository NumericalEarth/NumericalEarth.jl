# # Urban aerodynamic roughness from GHSL building morphometry
#
# Ingest the Global Human Settlement Layer (GHSL R2023A) mean building height and
# built-up fraction over a metropolitan area, derive the momentum roughness length
# `z₀ₘ` and zero-plane displacement `d₀` per cell with the urban morphometric closure
# (Macdonald 1998 / Kanda 2013), and render diagnostic maps + profiles.
#
# Requirements:
#   * `using ArchGDAL` (for the World-Mollweide → EPSG:4326 reprojection)
#   * `using CairoMakie` for the figures
# GHSL is open access — no authentication. The first run downloads two ~20 MB
# Mollweide tiles (cached afterwards); the rest is seconds.

using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, Metadatum
using NumericalEarth.DataWrangling.GHSL: GHSBuiltH, GHSBuiltS
using NumericalEarth.DataWrangling.UrbanRoughness: urban_roughness
using Oceananigans
using Oceananigans.Grids: λnodes, φnodes
using Oceananigans.Fields: interior
using ArchGDAL
using CairoMakie
using Statistics: mean

outdir = joinpath(@__DIR__, "ghsl_urban_roughness_figures")
mkpath(outdir)

# ## Region and target grid
# Greater London: a dense core (City / Westminster) grading out through suburbs to
# rural surroundings — a clean isolated → skimming built-fraction gradient.
region = BoundingBox(longitude = (-0.55, 0.35), latitude = (51.25, 51.75))
grid = LatitudeLongitudeGrid(CPU(), Float64; size = (450, 250),
                             longitude = (-0.55, 0.35), latitude = (51.25, 51.75),
                             topology = (Bounded, Bounded, Flat))

# ## Ingest the building morphometry
# `H` — mean net building height (ANBH, 100 m); `λp` — plan-area built fraction
# (BUILT-S, converted from m²/cell). Both are reprojected from Mollweide in the adapter.
H  = Field(Metadatum(:building_height;   dataset = GHSBuiltH(), region), grid)
λp = Field(Metadatum(:built_up_fraction; dataset = GHSBuiltS(), region), grid)

# ## Urban roughness closure
# Kanda (2013) is the default (height-heterogeneity aware); Macdonald (1998) for
# comparison. Both consume the same `(H, λp)` fields.
z0m_kanda, d0_kanda = urban_roughness(H, λp; method = :kanda)
z0m_macd,  d0_macd  = urban_roughness(H, λp; method = :macdonald)

# ## Figures
λ = Array(λnodes(grid, Center())); φ = Array(φnodes(grid, Center()))
sl(f) = Array(interior(f, :, :, 1))
finite(a) = filter(isfinite, vec(a))

Hm, λpm = sl(H), sl(λp)
z0k, d0k = sl(z0m_kanda), sl(d0_kanda)
z0mac, d0mac = sl(z0m_macd), sl(d0_macd)

@info "GHSL urban roughness over Greater London" building_height_range=extrema(finite(Hm)) built_fraction_range=extrema(finite(λpm)) z0m_kanda_range=extrema(finite(z0k)) d0_kanda_range=extrema(finite(d0k))

function panel!(fig, pos, title, A, crange, cmap, clabel)
    ax = Axis(fig[pos...]; title, aspect = DataAspect()); hidedecorations!(ax)
    hm = heatmap!(ax, λ, φ, A; colorrange = crange, colormap = cmap, nan_color = :gray82)
    Colorbar(fig[pos[1], pos[2] + 1], hm; label = clabel, width = 11)
    return ax
end

# (1) Kanda and Macdonald side by side. Top row: the closure inputs (H, λp). Middle
# rows: z₀ₘ and d₀ from each closure (same columns → same closure). Bottom row: the
# Kanda − Macdonald anomaly, largest over the dense, height-heterogeneous core.
z0_range = (0, 2.5); d0_range = (0, 20)
fig = Figure(size = (1150, 1500))
Label(fig[0, 1:4], "GHSL urban morphometry → roughness — Greater London\nKanda (2013) vs Macdonald (1998)"; fontsize = 20, font = :bold)
panel!(fig, (1, 1), "building height H (m)",  Hm,    (0, 25),  :inferno, "m")
panel!(fig, (1, 3), "built fraction λp",       λpm,   (0, 1),   :turbo,   "–")
panel!(fig, (2, 1), "z₀ₘ — Kanda (m)",         z0k,   z0_range, :viridis, "m")
panel!(fig, (2, 3), "z₀ₘ — Macdonald (m)",     z0mac, z0_range, :viridis, "m")
panel!(fig, (3, 1), "d₀ — Kanda (m)",          d0k,   d0_range, :magma,   "m")
panel!(fig, (3, 3), "d₀ — Macdonald (m)",      d0mac, d0_range, :magma,   "m")
panel!(fig, (4, 1), "z₀ₘ anomaly — Kanda − Macdonald (m)", z0k .- z0mac, (-1, 1),   :balance, "Δm")
panel!(fig, (4, 3), "d₀ anomaly — Kanda − Macdonald (m)",  d0k .- d0mac, (-10, 10), :balance, "Δm")
save(joinpath(outdir, "fig1_overview.png"), fig)

# (2) The diagnostic curve: z₀ₘ rises then falls with λp (isolated → wake → skimming),
# peaking at intermediate coverage — binned mean over the domain.
edges = range(0, 1; length = 21)
centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
function binned(z0)
    v = [Float64[] for _ in centers]
    for (l, z) in zip(vec(λpm), vec(z0))
        (isfinite(l) && isfinite(z)) || continue
        b = clamp(searchsortedlast(edges, l), 1, length(centers))
        push!(v[b], z)
    end
    return [isempty(x) ? NaN : mean(x) for x in v]
end
fig = Figure(size = (760, 500))
ax = Axis(fig[1, 1]; xlabel = "built fraction λp", ylabel = "domain-mean z₀ₘ (m)",
          title = "Roughness peaks at intermediate built fraction")
lines!(ax, centers, binned(z0k);   linewidth = 3, label = "Kanda")
lines!(ax, centers, binned(z0mac); linewidth = 3, label = "Macdonald")
axislegend(ax; position = :rt)
save(joinpath(outdir, "fig2_z0_vs_lambda.png"), fig)

# (3) West→east transect through the dense core, at the central latitude.
jmid = size(grid, 2) ÷ 2
fig = Figure(size = (1100, 640))
Label(fig[0, 1:1], "Transect at $(round(φ[jmid], digits = 3))°N — core → suburb → rural"; fontsize = 16, font = :bold)
ax1 = Axis(fig[1, 1]; ylabel = "H (m) / d₀ (m)", xlabel = "longitude")
ax2 = Axis(fig[1, 1]; ylabel = "λp / z₀ₘ (m)", yaxisposition = :right); hidespines!(ax2); hidexdecorations!(ax2)
lH  = lines!(ax1, λ, Hm[:, jmid];  color = :black,     linewidth = 2)
ld  = lines!(ax1, λ, d0k[:, jmid]; color = :firebrick, linewidth = 2)
lλ  = lines!(ax2, λ, λpm[:, jmid]; color = :seagreen,  linewidth = 2)
lz  = lines!(ax2, λ, z0k[:, jmid]; color = :navy,      linewidth = 2)
axislegend(ax1, [lH, ld, lλ, lz], ["H", "d₀", "λp", "z₀ₘ"]; position = :rt)
save(joinpath(outdir, "fig3_transect.png"), fig)

# ## Sanity check against literature ranges
# Dense-core z₀ₘ ~1–3 m and d₀ ~10–20 m; the peak is at intermediate λp, not maximum
# coverage; λp → 0 reduces to a bare-soil roughness (~0.03 m).
dense = λpm .> 0.4
@info "Dense-core (λp > 0.4) morphometry" n_cells=count(dense) z0m_kanda=mean(finite(z0k[dense])) d0_kanda=mean(finite(d0k[dense]))
@info "Bare (λp < 0.02) roughness" z0m=mean(finite(z0k[λpm .< 0.02]))
@info "wrote figures" outdir
