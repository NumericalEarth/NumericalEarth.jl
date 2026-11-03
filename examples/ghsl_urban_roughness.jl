# # Urban aerodynamic roughness from GHSL building morphometry
#
# Compute momentum roughness length and zero-plane displacement over inner London
# from GHSL built-up surface and building height. The closure uses 1 km neighborhoods,
# and the reduction onto the model grid preserves their area-mean neutral drag.
#
# `GHSBuilt()` pairs the 10 m built fraction with the 100 m building height, both
# for 2018. The first run downloads the intersecting Mollweide tiles and caches them.

using NumericalEarth
using Oceananigans
using ArchGDAL
using CairoMakie

# ## Model grid and neighborhood roughness

grid = LatitudeLongitudeGrid(size = (24, 16),
                             longitude = (-0.28, 0.06), latitude = (51.42, 51.56),
                             topology = (Bounded, Bounded, Flat))
dataset = GHSBuilt()
surface_layer_height = 500
neighborhood = 1000

# Both closures share the raster read and neighborhood morphometry.
# `UniformHeight` uses the equal-height obstacle-array closure.

closures = (; variable_height = MorphometricRoughness(),
              uniform_height = MorphometricRoughness(height_distribution = UniformHeight()))
variable_height, uniform_height = urban_roughness(grid, dataset; surface_layer_height, neighborhood, closure = closures)

# ## Roughness and displacement maps
# `urban_fraction` is the area fraction occupied by urban neighborhoods in each
# model cell. `building_height` is weighted by built area within those neighborhoods.

function panel!(figure, position, title, field, colorrange, colormap, label)
    axis = Axis(figure[position...]; title, aspect = DataAspect())
    hidedecorations!(axis)
    heatmap_plot = heatmap!(axis, field; colorrange, colormap)
    Colorbar(figure[position[1], position[2] + 1], heatmap_plot; label, width = 11)
    return axis
end

roughness_range = (0, 3.5)
displacement_range = (0, 40)
fig = Figure(size = (1150, 1100))
Label(fig[0, 1:4], "GHSL urban roughness over inner London (1 km neighborhoods)"; fontsize = 20)
panel!(fig, (1, 1), "building height (m)", variable_height.building_height, (0, 25), :inferno, "m")
panel!(fig, (1, 3), "urban fraction", variable_height.urban_fraction, (0, 1), :turbo, "1")
panel!(fig, (2, 1), "roughness, variable height (m)", variable_height.momentum_roughness_length,
        roughness_range, :viridis, "m")
panel!(fig, (2, 3), "roughness, uniform height (m)", uniform_height.momentum_roughness_length,
        roughness_range, :viridis, "m")
panel!(fig, (3, 1), "displacement, variable height (m)", variable_height.zero_plane_displacement,
        displacement_range, :magma, "m")
panel!(fig, (3, 3), "displacement, uniform height (m)", uniform_height.zero_plane_displacement,
        displacement_range, :magma, "m")
save("ghsl_urban_roughness.png", fig)
fig
