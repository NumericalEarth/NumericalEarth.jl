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
#     `dataset = GLO30()` and set `DESTINE_ACCESS_TOKEN` (see [`GLO30`](@ref)) — the
#     grid-derived window is added automatically. Here we use ETOPO 2022 so the script
#     runs without a token; ETOPO reads globally while GLO-30 is windowed to the grid,
#     but the subtraction, the figures, and the correction wiring are identical.
#
# !!! note "Object heights"
#     A canopy-height dataset supplies the object height over vegetation (a
#     building-height dataset would add the built-up areas). That adapter is a
#     separate piece of the land pipeline; here we stand in a canopy over the
#     terra-firme forest, gated by elevation so the river floodplains stay bare.

using NumericalEarth
using Oceananigans
using CairoMakie

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

dsm_elevation = regrid_topography(grid; dataset = dsm_dataset)

# ## Object heights (canopy)
#
# We place a ~35 m forest canopy over the terra-firme uplands and leave the river
# floodplains bare, gated by elevation — 0 height is a valid value, not a gap. In the
# full pipeline this field comes from a canopy-height dataset.

λ, φ, _ = nodes(grid, Center(), Center(), Center())
elevation = interior(dsm_elevation, :, :, 1)

canopy_height          = 35.0   # m, tall tropical canopy
terra_firme_elevation  = 30.0   # m, above this is upland forest; below is river floodplain

canopy = Field{Center, Center, Nothing}(grid)
set!(canopy, ifelse.(elevation .≥ terra_firme_elevation, canopy_height, 0.0))

# ## Bare-earth terrain
#
# `bare_earth_elevation` subtracts the object heights from the DSM and clamps at sea
# level. The DSM-minus-bare-earth difference *is* the removed canopy (where the DSM
# stands above it) — the signal that belongs to roughness, not terrain.

bare_elevation = bare_earth_elevation(dsm_elevation, canopy)

removed_object_height = dsm_elevation - bare_elevation

# ## Antialiasing check
#
# Coarsening a fine surface onto a coarse cell must smooth, not alias. Regridding the
# same DSM onto a grid four times coarser keeps the elevation range intact.

coarse_grid = LatitudeLongitudeGrid(CPU(); latitude, longitude, size = (38, 28),
                                   topology = (Bounded, Bounded, Flat))
coarse_dsm_elevation = regrid_topography(coarse_grid; dataset = dsm_dataset)

@info "Elevation range (m): model grid $(round.(extrema(dsm_elevation); digits=1)), " *
      "4× coarser $(round.(extrema(coarse_dsm_elevation); digits=1))"

# ## Feeding the atmosphere elevation correction
#
# `SlabLand` uses terrain through [`ElevationCorrection`](@ref): the near-surface
# atmosphere is lapse-corrected over `Δz = z_surface − z_atmosphere`, the gap between
# the model's surface elevation and the coarse elevation the atmosphere data assumes.
# A coarse reference elevation stands in for that atmosphere elevation.

reference_elevation = Field{Center, Center, Nothing}(grid)
set!(reference_elevation, coarse_dsm_elevation)

elevation_difference = bare_elevation - reference_elevation

# The correction is a one-liner over the bare-earth field. Over this flat basin the
# terrain relief is only ~100 m, so the ~35 m canopy the subtraction removes is a
# large fraction of the correction — the piece that belongs to the roughness closure,
# not the terrain.

correction = ElevationCorrection(bare_elevation, reference_elevation)

# ## Transect across the basin
#
# A west–east cut crosses river floodplains and terra-firme forest, so the DSM sits a
# full canopy height above the bare-earth line over the forest and drops onto it over
# the rivers.

transect_latitude = -3.0
transect_index = searchsortedfirst(φ, transect_latitude)
dsm_transect   = view(dsm_elevation,  :, transect_index, 1)
bare_transect  = view(bare_elevation, :, transect_index, 1)

# ## Visualization

fig = Figure(size = (1600, 1150), fontsize = 15)

elevation_limits     = extrema(dsm_elevation)
object_height_limits = (0, canopy_height)

dsm_axis = Axis(fig[1, 1]; title = "DSM elevation (m)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
dsm_heatmap = heatmap!(dsm_axis, dsm_elevation; colormap = :terrain, colorrange = elevation_limits)
Colorbar(fig[1, 2], dsm_heatmap)

bare_axis = Axis(fig[1, 3]; title = "Bare-earth DTM (m)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
bare_heatmap = heatmap!(bare_axis, bare_elevation; colormap = :terrain, colorrange = elevation_limits)
Colorbar(fig[1, 4], bare_heatmap)

canopy_axis = Axis(fig[1, 5]; title = "Synthetic canopy height (m)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
canopy_heatmap = heatmap!(canopy_axis, canopy; colormap = :speed, colorrange = object_height_limits)
Colorbar(fig[1, 6], canopy_heatmap)

removed_axis = Axis(fig[2, 1]; title = "DSM − bare-earth (m)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
removed_heatmap = heatmap!(removed_axis, removed_object_height; colormap = :speed, colorrange = object_height_limits)
Colorbar(fig[2, 2], removed_heatmap)

reference_axis = Axis(fig[2, 3]; title = "Coarse reference elevation (m)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
reference_heatmap = heatmap!(reference_axis, reference_elevation; colormap = :terrain, colorrange = elevation_limits)
Colorbar(fig[2, 4], reference_heatmap)

maximum_elevation_difference = maximum(abs, elevation_difference)
correction_axis = Axis(fig[2, 5]; title = "Elevation correction Δz (m)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
correction_heatmap = heatmap!(correction_axis, elevation_difference; colormap = :balance,
                              colorrange = (-maximum_elevation_difference, maximum_elevation_difference))
Colorbar(fig[2, 6], correction_heatmap)

transect_axis = Axis(fig[3, 1:6]; title = "West–east transect at $(transect_latitude)°N",
                     xlabel = "longitude", ylabel = "elevation (m)")
band!(transect_axis, λ, vec(interior(bare_transect)), vec(interior(dsm_transect));
      color = (:seagreen, 0.35), label = "removed canopy (synthetic)")
lines!(transect_axis, dsm_transect;  color = :black,     label = "DSM")
lines!(transect_axis, bare_transect; color = :firebrick, label = "bare-earth DTM")
axislegend(transect_axis; position = :lt)

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
