# # Bare-earth terrain from a surface model minus object heights
#
# A Digital Surface Model (DSM) measures the top of whatever sits on the ground — tree
# canopy over forest, roofs over cities — not the ground itself. At the resolution of a
# regional model those objects are sub-grid, so a DSM aggregated to a cell reports the
# surface *raised* by the mean object height. That lift is the displacement height the
# surface-layer roughness closure already carries, so it belongs in the roughness
# parameterization, not the terrain. This example recovers a bare-earth elevation with
# [`bare_earth_elevation`](@ref),
#
#     z_bare = max(z_DSM − maxₖ object_heightₖ, 0),
#
# and feeds it to the atmosphere elevation correction.
#
# We run across the central Amazon near Manaus (the Rio Negro–Solimões confluence),
# where the terrain is nearly flat — a ~160 m spread of low terra-firme plateaus
# dissected by river floodplains — while the forest canopy stands ~35 m tall. This is
# the regime bare-earth DTMs like FABDEM were built for: over intact tropical forest a
# DSM sits roughly a canopy height above true ground, and the flat terrain makes that
# offset dominate.
#
# !!! note "Access"
#     The DSM is Copernicus GLO-30 (30 m), read from the DestinE Earth Data Hub: set a
#     (free) token in `DESTINE_ACCESS_TOKEN` (see [`GLO30`](@ref)) and load `Zarr`.
#     `regrid_topography` derives the read window from the grid, so only the Amazon block
#     is fetched, not the globe.
#
# !!! note "Object heights"
#     A canopy-height dataset supplies the object height over vegetation, a
#     building-height dataset the built-up areas. Those adapters are a separate piece of
#     the land pipeline; here we stand in a canopy over the terra-firme forest, gated by
#     elevation so the river floodplains stay bare.

using NumericalEarth
using Oceananigans
using Oceananigans.Fields: interpolate!
using Zarr          ## GLO-30 Zarr store read
using CairoMakie

# ## Domain and DSM
#
# A ~1 km land grid, `Flat` in the vertical: terrain enters as a 2-D elevation field, not
# grid geometry. `regrid_topography` lands the DSM on it as a positive land-surface
# elevation.
#
# Caveat: coarsening here is 37× from GLO-30's native 30 m, and `regrid_topography` samples
# rather than cell-averages, so the elevation carries some aliased sub-grid texture;
# conservative regridding is tracked in
# [ClimaOcean #253](https://github.com/CliMA/ClimaOcean.jl/issues/253).

latitude  = -3.5, -2.4
longitude = -60.5, -59.0

grid = LatitudeLongitudeGrid(CPU(); latitude, longitude, size = (150, 110),
                            topology = (Bounded, Bounded, Flat))

dsm_dataset = GLO30()

dsm_elevation = regrid_topography(grid; dataset = dsm_dataset)

# ## Object heights (canopy)
#
# A ~35 m forest canopy over the terra-firme uplands, with the river floodplains left
# bare — 0 height is a valid value, not a gap. In the full pipeline this field comes from
# a canopy-height dataset.

λ, φ, _ = nodes(grid, Center(), Center(), Center())
elevation = interior(dsm_elevation, :, :, 1)

canopy_height          = 35.0   # m, tall tropical canopy
terra_firme_elevation  = 30.0   # m, above this is upland forest; below is river floodplain

canopy = Field{Center, Center, Nothing}(grid)
set!(canopy, ifelse.(elevation .≥ terra_firme_elevation, canopy_height, 0.0))

# ## Bare-earth terrain
#
# `bare_earth_elevation` subtracts the object heights from the DSM and clamps at sea level.
# The DSM-minus-bare-earth difference *is* the removed canopy — the signal that belongs to
# roughness, not terrain.

bare_elevation = bare_earth_elevation(dsm_elevation, canopy)

removed_object_height = dsm_elevation - bare_elevation

# ## Coarse reference elevation
#
# The atmosphere data assumes its own, coarser elevation. With a driving reanalysis in hand
# that is a dataset read — `Field(Metadatum(:topography; dataset, date, region), grid)` lands
# ERA5's own orography on the model grid, as the differentiable ERA5-forced slab land example
# does. Here we stand in for it with the same two-step at ~4 km: `regrid_topography` onto a
# coarse grid, then interpolated up. The stand-in coarsens the DSM rather than the bare-earth
# field because a global elevation product carries surface heights, canopy included.

atmosphere_grid = LatitudeLongitudeGrid(CPU(); latitude, longitude, size = (37, 27),
                                        topology = (Bounded, Bounded, Flat))

coarse_elevation = regrid_topography(atmosphere_grid; dataset = dsm_dataset)

reference_elevation = Field{Center, Center, Nothing}(grid)
interpolate!(reference_elevation, coarse_elevation)

# ## Feeding the atmosphere elevation correction
#
# `SlabLand` uses terrain through [`ElevationCorrection`](@ref): the near-surface atmosphere
# is lapse-corrected over `Δz = z_surface − z_atmosphere`, the gap between the model's
# surface elevation and the coarse elevation the atmosphere data assumes.

elevation_difference = bare_elevation - reference_elevation

correction = ElevationCorrection(bare_elevation, reference_elevation)

# ## Transect across the basin
#
# A west–east cut crosses both river floodplains and terra-firme forest.

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

transect_axis = Axis(fig[3, 1:6]; title = "West–east transect at $(transect_latitude)°",
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
# The DSM and bare-earth maps differ over the whole forested basin — everywhere the canopy
# stands the bare-earth surface drops a full tree height, and the DSM − bare-earth panel
# reproduces the canopy map apart from the ~5% of low cells where the DSM is shorter than the
# canopy the gate assigns and the sea-level clamp truncates the drop. Because the terrain here
# spans only ~160 m, that ~35 m correction is a large fraction of the relief, and the elevation
# correction `Δz` shifts visibly wherever forest sits. The transect shows the DSM riding a
# canopy height above the bare-earth line over terra-firme forest and dropping onto it over
# the river floodplains.

correction
