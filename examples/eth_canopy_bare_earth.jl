# # Real ETH canopy in the DSM bare-earth workflow — central Amazon
#
# This is the `bare_earth_terrain.jl` scene (Rio Negro–Solimões confluence near Manaus),
# but with the **measured ETH canopy height** in place of the synthetic elevation-gated
# canopy that example stands in with. The same canopy field then feeds two consumers:
#
#   1. **Bare-earth terrain** — `bare_earth_elevation(z_DSM, h_c)` = `max(z_DSM − h_c, 0)`.
#   2. **Aerodynamic roughness** — `compute_aerodynamic_roughness!` (Raupach drag partition).
#
# The canopy the DSM overstates the ground by is exactly the canopy that sets the surface
# roughness — the terrain subtraction and the roughness closure share one measured field.
# A final section verifies the roughness closure itself against idealized cases.
#
# DSM stand-in: ETOPO 2022 (token-free; `GLO30()` is the commercial-use 30 m surface model).
# The ETH product is 10 m; `canopy_height_field` area-averages it onto the land grid straight
# from the windowed COGs (anonymous `/vsicurl/`). Needs `using ArchGDAL` and `using CairoMakie`.

using NumericalEarth
using NumericalEarth.DataWrangling.ETHSentinel2Canopy: ETHSentinel2CanopyHeight, canopy_height_field
using NumericalEarth.Lands: DragPartitionRoughness, compute_aerodynamic_roughness!,
                            canopy_roughness, canopy_wind_ratio,
                            canopy_drag_parameters, representative_canopy_height
using Oceananigans
using Oceananigans.Fields: set!, interior
using ArchGDAL   # activates the COG-read extension used by canopy_height_field
using CairoMakie

# ## Domain and DSM (identical to `bare_earth_terrain.jl`)
latitude  = -3.5, -2.4
longitude = -60.5, -59.0
grid = LatitudeLongitudeGrid(CPU(); latitude, longitude, size = (1500, 1100),
                             topology = (Bounded, Bounded, Flat))   # ~110 m across the basin

z_dsm = regrid_topography(grid; dataset = ETOPO2022())

# ## Real canopy height on the model grid
#
# `canopy_height_field` area-averages the ETH 10 m COG pixels within each model cell
# (coarse-graining, not point interpolation), reading only the windowed COG blocks via
# `/vsicurl/`. Canopy height over non-forest is a valid `0`; only the no-data byte is
# masked to `NaN`.
canopy_height = canopy_height_field(grid, ETHSentinel2CanopyHeight())

# ## (1) Bare-earth terrain — DSM minus the measured canopy
z_bare  = bare_earth_elevation(z_dsm, canopy_height)
removed = compute!(Field(z_dsm - z_bare))      # the canopy lift removed from the terrain

# ## (2) Roughness — the same canopy through the Raupach closure
# The basin is evergreen broadleaf forest, a uniform dense canopy (LAI 5) here.
# `DragPartitionRoughness()` defaults to the `:evergreen_broadleaf_forest` class.
lai = Field{Center, Center, Nothing}(grid); set!(lai, 5)
z0, d0 = Field{Center, Center, Nothing}(grid), Field{Center, Center, Nothing}(grid)
compute_aerodynamic_roughness!(z0, d0, DragPartitionRoughness(), (; lai, canopy_height), grid)

# ## Figures
outdir = joinpath(@__DIR__, "eth_canopy_bare_earth_figures"); mkpath(outdir)
finite(f) = filter(isfinite, vec(Array(interior(f))))
zmax = maximum(finite(z_dsm))

function heat!(fig, pos, title, field, crange, cmap)
    ax = Axis(fig[pos...]; title, aspect = DataAspect()); hidedecorations!(ax)
    hm = heatmap!(ax, field; colorrange = crange, colormap = cmap, nan_color = :gray82)
    Colorbar(fig[pos[1], pos[2] + 1], hm; width = 11)
end

fig = Figure(size = (1500, 950))
Label(fig[0, 1:4], "Real ETH canopy in the DSM bare-earth workflow — Rio Negro–Solimões, Amazon"; fontsize = 19, font = :bold)
heat!(fig, (1, 1), "DSM elevation (ETOPO stand-in)", z_dsm,         (0, zmax), :terrain)
heat!(fig, (1, 3), "canopy height h_c (ETH)",        canopy_height, (0, maximum(finite(canopy_height))), :YlGn)
heat!(fig, (2, 1), "bare-earth DTM = DSM − canopy",  z_bare,        (0, zmax), :terrain)
heat!(fig, (2, 3), "z₀ roughness from the same h_c", z0,            (0, maximum(finite(z0))), :viridis)
save(joinpath(outdir, "fig1_canopy_dsm_roughness.png"), fig)

# A west–east transect at 3.0°S: the DSM rides a canopy height above the bare-earth line.
jrow = size(grid, 2) ÷ 2
x = 1:size(grid, 1)
fig = Figure(size = (1200, 440))
ax = Axis(fig[1, 1]; xlabel = "west → east (grid cells)", ylabel = "elevation (m)",
          title = "DSM vs bare-earth along 3°S: the gap is the ETH canopy")
lines!(ax, x, Array(interior(z_dsm,  :, jrow, 1)); linewidth = 2, label = "DSM")
lines!(ax, x, Array(interior(z_bare, :, jrow, 1)); linewidth = 2, label = "bare-earth")
lines!(ax, x, Array(interior(removed, :, jrow, 1)); linewidth = 2, color = :seagreen, label = "removed canopy")
axislegend(ax; position = :rt)
save(joinpath(outdir, "fig2_transect.png"), fig)

@info "canopy + DSM" h_c_max = round(maximum(finite(canopy_height)), digits = 1) dsm_range = round.((minimum(finite(z_dsm)), zmax), digits = 1) removed_canopy_max = round(maximum(finite(removed)), digits = 1) z0_mean = round(sum(finite(z0)) / length(finite(z0)), digits = 2)

# ## (3) Is the roughness closure right? Idealized verification
#
# The `z₀` map above is only as trustworthy as the closure behind it. Driving that same
# closure with idealized inputs recovers its mathematical signatures and its physical
# magnitudes — the behavior expected from Raupach (1994) and the class values tabulated by
# Borak et al. (2025). `canopy_roughness(closure, Λ, h)` is the scalar form of the operator
# that filled the `z₀` field above; it also accepts a `Field` or a `FieldTimeSeries`.

closure = DragPartitionRoughness()                  # evergreen broadleaf forest defaults
Λmax    = closure.parameters.maximum_area_index
h₀      = representative_canopy_height(Float64, :evergreen_broadleaf_forest)

fig = Figure(size = (1500, 1080), fontsize = 15)
Label(fig[0, 1:2], "Drag-partition roughness closure — idealized verification"; fontsize = 21, font = :bold)

# **(1) Λ-response.** Sweeping leaf-area index Λ at the reference height (lengths as a fraction
# of canopy height h): displacement `d₀/h` climbs monotonically toward the canopy top, while
# roughness `z₀/h` is non-monotonic — it peaks then *keeps falling* as the canopy densifies.
# That falling branch is the signature of skimming flow: a denser canopy is aerodynamically
# smoother. Λₘₐₓ caps only the wind ratio `u★/Uh`; both lengths use the full Λ.
Λs  = range(0, 6; length = 241)
d0h = [canopy_roughness(closure, Λ, h₀)[2] / h₀ for Λ in Λs]
z0h = [canopy_roughness(closure, Λ, h₀)[1] / h₀ for Λ in Λs]

ax1 = Axis(fig[1, 1]; xlabel = "leaf-area index Λ", ylabel = "fraction of canopy height h",
           title = "(1) Λ-response: d₀/h monotone, z₀/h skims past Λₘₐₓ")
lines!(ax1, Λs, d0h; linewidth = 3, color = :navy,    label = "d₀/h  (displacement)")
lines!(ax1, Λs, z0h; linewidth = 3, color = :crimson, label = "z₀/h  (roughness)")
vlines!(ax1, [Λmax]; color = :gray50, linestyle = :dash)
text!(ax1, Λmax, 0.02; text = " Λₘₐₓ (u★/Uh cap)", color = :gray40, align = (:left, :bottom))
axislegend(ax1; position = :rc)

# **(2) Wind ratio.** The internal ratio γ = Uh/u★ *falls* as the canopy densifies (form drag
# raises u★/Uh) and is floored at each class's cap 1/(u★/Uh)ₘₐₓ, so the friction ratio never
# runs away; every vegetation class relaxes onto its own floor.
ax2 = Axis(fig[1, 2]; xlabel = "leaf-area index Λ", ylabel = "wind ratio  γ = Uh/u★",
           title = "(2) γ falls with Λ, floored at its cap 1/(u★/Uh)ₘₐₓ")
for (class, color) in ((:evergreen_broadleaf_forest, :seagreen),
                       (:cropland, :sienna), (:grassland, :goldenrod))
    parameters = canopy_drag_parameters(Float64, class)
    lines!(ax2, Λs, [canopy_wind_ratio(Λ, parameters, closure.iterations) for Λ in Λs];
           linewidth = 3, color, label = replace(String(class), "_" => " "))
    hlines!(ax2, [1 / parameters.maximum_friction_ratio]; color, linestyle = :dash)
end
axislegend(ax2; position = :rb)

# **(3) Height scaling.** At fixed Λ both lengths are *exactly* linear in canopy height h
# (`z₀ = h·f(Λ)`, `d₀ = h·g(Λ)`), passing through the origin, and sit close to the height-only
# rules of thumb `d₀ ≈ 2h/3` and `z₀ ≈ d₀/5` (Brutsaert 1982).
hs   = range(0, 30; length = 121)
d0_h = [canopy_roughness(closure, 5.0, h)[2] for h in hs]
z0_h = [canopy_roughness(closure, 5.0, h)[1] for h in hs]

ax3 = Axis(fig[2, 1]; xlabel = "canopy height h (m)", ylabel = "length (m)",
           title = "(3) z₀, d₀ scale linearly with h  (vs semi-empirical rules)")
lines!(ax3, hs, d0_h; linewidth = 3, color = :navy,    label = "d₀ closure (Λ = 5)")
lines!(ax3, hs, z0_h; linewidth = 3, color = :crimson, label = "z₀ closure (Λ = 5)")
lines!(ax3, hs, 2 .* hs ./ 3;  linewidth = 2, linestyle = :dash, color = :navy,    label = "d₀ = 2h/3 (Brutsaert)")
lines!(ax3, hs, 2 .* hs ./ 15; linewidth = 2, linestyle = :dash, color = :crimson, label = "z₀ = d₀/5")
axislegend(ax3; position = :lt)

# **(4) Magnitudes.** Evaluated at each IGBP class's representative (Λ, h), the closure `z₀`
# sits at the class-integrated end (`z₀ⁱ`) of the range spanned by the two reference estimates
# of Borak et al. (2025, Table 5), reproducing the forest ≫ crop, grass ordering.
oracle = [("EBF", :evergreen_broadleaf_forest,  24.72, 6.0, 1.16, 3.30),
          ("DBF", :deciduous_broadleaf_forest,  17.43, 5.0, 0.98, 2.32),
          ("ENF", :evergreen_needleleaf_forest, 16.62, 4.0, 1.36, 2.22),
          ("GRS", :grassland,                    1.39, 1.5, 0.14, 0.19),
          ("CRP", :cropland,                     1.32, 3.0, 0.13, 0.18)]
x        = 1:length(oracle)
z0_class = [canopy_roughness(DragPartitionRoughness(Float64; vegetation_type = class), Λ, h)[1]
            for (_, class, h, Λ, _, _) in oracle]

ax4 = Axis(fig[2, 2]; ylabel = "momentum roughness z₀ (m)",
           xticks = (x, [row[1] for row in oracle]),
           title = "(4) magnitudes vs Borak et al. (2025) Table 5")
rangebars!(ax4, x, [min(row[5], row[6]) for row in oracle], [max(row[5], row[6]) for row in oracle];
           whiskerwidth = 18, linewidth = 3, color = :gray70, label = "Borak z₀ range")
scatter!(ax4, x, z0_class; markersize = 16, color = :black, label = "closure")
axislegend(ax4; position = :rt)

save(joinpath(outdir, "fig3_closure_verification.png"), fig)

# ![](eth_canopy_bare_earth_figures/fig3_closure_verification.png)

fig
