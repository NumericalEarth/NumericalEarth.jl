# # Bare-earth terrain from a surface model minus object heights
#
# A Digital Surface Model (DSM) measures the top of whatever sits on the ground —
# tree canopy over forest, roofs over cities — not the ground itself. At the
# resolution of a regional model those objects are sub-grid, so a DSM aggregated to a
# cell reports the surface *raised* by the mean object height. That lift is the same
# effect the surface-layer roughness closure already carries as a displacement
# height, so it belongs in the roughness parameterization, not the terrain. This
# example recovers a bare-earth elevation by subtracting object heights from a DSM,
#
#     z_bare = max(z_DSM − maxₖ object_heightₖ, 0),
#
# with [`bare_earth_elevation`](@ref), and shows the field feeding its current
# consumer, the atmosphere elevation correction.
#
# We run across the central Amazon near Manaus (the Rio Negro–Solimões confluence),
# because that is where the correction matters most. The terrain is nearly flat — a
# ~120 m spread of low terra-firme plateaus dissected by river floodplains — while the
# forest canopy stands ~35 m tall. Removing the canopy is therefore a *large fraction*
# of the surface signal, not a rounding error. This is the regime bare-earth DTMs like
# FABDEM were built for: over intact tropical forest a DSM sits roughly a canopy-height
# above true ground, and the flat terrain makes that offset dominate.
#
# !!! note "DSM source"
#     The commercial-use DSM for this workflow is Copernicus GLO-30 (30 m); pass
#     `dataset = GLO30()` with a `BoundingBox` region and set `DESTINE_ACCESS_TOKEN`
#     (see [`GLO30`](@ref)). Here we use ETOPO 2022 so the script runs without a
#     token — the subtraction, the figures, and the correction wiring are identical.
#
# !!! note "Object heights"
#     A canopy-height dataset supplies the object height over vegetation (a
#     building-height dataset would add the built-up areas). That adapter is a
#     separate piece of the land pipeline; here we stand in a canopy over the
#     terra-firme forest, gated by elevation so the river floodplains stay bare.

using NumericalEarth
using Oceananigans
using CairoMakie
using Statistics
using Oceananigans.Fields: interpolate!

# ## Domain and DSM
#
# A ~1 km land grid over the central Amazon, `Flat` in the vertical (terrain enters as
# a 2-D elevation field, not grid geometry). `regrid_topography` lands the DSM on the
# grid as a positive land-surface elevation, antialiased when it coarsens.

latitude  = -3.5, -2.4
longitude = -60.5, -59.0

grid = LatitudeLongitudeGrid(CPU(); latitude, longitude, size = (150, 110),
                            topology = (Bounded, Bounded, Flat))

dsm_dataset = ETOPO2022()   # stand-in for GLO30() — see the note above

z_dsm = regrid_topography(grid; dataset = dsm_dataset)

# ## Object heights (canopy)
#
# We place a ~35 m forest canopy over the terra-firme uplands and leave the river
# floodplains bare, gated by elevation — 0 height is a valid value, not a gap. In the
# full pipeline this field comes from a canopy-height dataset.

λ, φ, _ = nodes(grid, Center(), Center(), Center())
elevation = Array(interior(z_dsm, :, :, 1))

canopy_height          = 35.0   # m, tall tropical canopy
terra_firme_elevation  = 30.0   # m, above this is upland forest; below is river floodplain

canopy = Field{Center, Center, Nothing}(grid)
canopy_data = [elevation[i, j] ≥ terra_firme_elevation ? canopy_height : 0.0
               for i in eachindex(λ), j in eachindex(φ)]
set!(canopy, canopy_data)

# ## Bare-earth terrain
#
# `bare_earth_elevation` subtracts the object heights from the DSM and clamps at sea
# level. The DSM-minus-bare-earth difference *is* the removed canopy (where the DSM
# stands above it) — the signal that belongs to roughness, not terrain.

z_bare = bare_earth_elevation(z_dsm, canopy)

dsm_minus_bare = z_dsm - z_bare

# ## Antialiasing check
#
# Coarsening a fine surface onto a coarse cell must smooth, not alias. Regridding the
# same DSM onto a grid four times coarser keeps the elevation range intact.

coarse_grid = LatitudeLongitudeGrid(CPU(); latitude, longitude, size = (38, 28),
                                   topology = (Bounded, Bounded, Flat))
z_coarse_dsm = regrid_topography(coarse_grid; dataset = dsm_dataset)

@info "Elevation range (m): model grid $(round.(extrema(z_dsm); digits=1)), " *
      "4× coarser $(round.(extrema(z_coarse_dsm); digits=1))"

# ## Feeding the atmosphere elevation correction
#
# `SlabLand` uses terrain through [`ElevationCorrection`](@ref): the near-surface
# atmosphere is lapse-corrected over `Δz = z_surface − z_atmosphere`, the gap between
# the model's surface elevation and the coarse elevation the atmosphere data assumes.
# A coarse reference elevation stands in for that atmosphere elevation.

z_reference = Field{Center, Center, Nothing}(grid)
interpolate!(z_reference, z_coarse_dsm)

Δz_bare = z_bare - z_reference

# The correction is a one-liner over the bare-earth field. Over this flat basin the
# terrain relief is only ~100 m, so the ~35 m canopy the subtraction removes is a
# large fraction of the correction — the piece that belongs to the roughness closure,
# not the terrain.

correction = ElevationCorrection(z_bare, z_reference)

# ## Transect across the basin
#
# A west–east cut crosses river floodplains and terra-firme forest, so the DSM sits a
# full canopy height above the bare-earth line over the forest and drops onto it over
# the rivers.

transect_latitude = -3.0
jrow = searchsortedfirst(φ, transect_latitude)
transect_dsm  = interior(z_dsm,  :, jrow, 1)
transect_bare = interior(z_bare, :, jrow, 1)

# ## Visualization

fig = Figure(size = (1600, 1150), fontsize = 15)

zlim = extrema(z_dsm)
olim = (0, canopy_height)

ax_dsm = Axis(fig[1, 1]; title = "DSM elevation (m)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hm_dsm = heatmap!(ax_dsm, z_dsm; colormap = :terrain, colorrange = zlim)
Colorbar(fig[1, 2], hm_dsm)

ax_bare = Axis(fig[1, 3]; title = "Bare-earth DTM (m)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hm_bare = heatmap!(ax_bare, z_bare; colormap = :terrain, colorrange = zlim)
Colorbar(fig[1, 4], hm_bare)

ax_obj = Axis(fig[1, 5]; title = "Synthetic canopy height (m)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hm_obj = heatmap!(ax_obj, canopy; colormap = :speed, colorrange = olim)
Colorbar(fig[1, 6], hm_obj)

ax_diff = Axis(fig[2, 1]; title = "DSM − bare-earth (m)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hm_diff = heatmap!(ax_diff, dsm_minus_bare; colormap = :speed, colorrange = olim)
Colorbar(fig[2, 2], hm_diff)

ax_ref = Axis(fig[2, 3]; title = "Coarse reference elevation (m)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hm_ref = heatmap!(ax_ref, z_reference; colormap = :terrain, colorrange = zlim)
Colorbar(fig[2, 4], hm_ref)

Δzmax = maximum(abs, Δz_bare)
ax_cb = Axis(fig[2, 5]; title = "Elevation correction Δz (m)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hm_cb = heatmap!(ax_cb, Δz_bare; colormap = :balance, colorrange = (-Δzmax, Δzmax))
Colorbar(fig[2, 6], hm_cb)

ax_t = Axis(fig[3, 1:6]; title = "West–east transect at $(transect_latitude)°N",
            xlabel = "longitude", ylabel = "elevation (m)")
band!(ax_t, λ, transect_bare, transect_dsm; color = (:seagreen, 0.35), label = "removed canopy (synthetic)")
lines!(ax_t, λ, transect_dsm;  color = :black,     label = "DSM")
lines!(ax_t, λ, transect_bare; color = :firebrick, label = "bare-earth DTM")
axislegend(ax_t; position = :lt)

Label(fig[0, 1:6], rich("Bare-earth terrain over the central Amazon — DSM minus canopy height\n",
                        rich("canopy height is SYNTHETIC (an elevation-gated stand-in, not measured); terrain is real",
                             fontsize = 14, color = :firebrick)),
      fontsize = 20)

save("bare_earth_terrain.png", fig)
nothing #hide

# ![](bare_earth_terrain.png)
#
# The DSM and bare-earth maps differ over the whole forested basin — everywhere the
# canopy stands, the bare-earth surface drops a full tree height, and the DSM − bare-earth
# panel reproduces the canopy map exactly. Because the terrain here spans only ~100 m,
# that ~35 m correction is a large fraction of the relief, not a rounding error — the
# elevation correction `Δz` shifts visibly wherever forest sits. The transect shows the
# DSM riding a canopy height above the bare-earth line over terra-firme forest and
# dropping onto it over the river floodplains.

correction
