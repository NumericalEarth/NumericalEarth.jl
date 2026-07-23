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
#
# DSM stand-in: ETOPO 2022 (token-free; `GLO30()` is the commercial-use 30 m surface model).
# The ETH product is 10 m; `canopy_height_field` area-averages it onto the land grid straight
# from the windowed COGs (anonymous `/vsicurl/`). Needs `using ArchGDAL` and `using CairoMakie`.

using NumericalEarth
using NumericalEarth.DataWrangling.ETHSentinel2Canopy: ETHSentinel2CanopyHeight, canopy_height_field
using NumericalEarth.Lands: DragPartitionRoughness, compute_aerodynamic_roughness!
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
