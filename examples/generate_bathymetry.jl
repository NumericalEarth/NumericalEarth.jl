# # Generate bathymetry data for the Mediterranean Sea
#
# This example shows how to generate realistic bathymetry for the Mediterranean Sea
# with NumericalEarth.jl, ready to be used for an immersed boundary grid.
#
# We need Oceananigans for the `LatitudeLongitudeGrid`, NumericalEarth to download and
# regrid the bathymetry, and CairoMakie to visualize it.

using NumericalEarth
using Oceananigans
using CairoMakie

# We start by defining a `LatitudeLongitudeGrid` that covers the Mediterranean Sea.
#
# The Mediterranean Sea lies roughly between 28ᵒ N and 48ᵒ N and between 0ᵒ and 42ᵒ E.
# We use a horizontal resolution of 1/25ᵒ in both latitude and longitude.

latitude_range = (28, 48)
longitude_range = (0, 42)

Nφ = 25 * (latitude_range[2] - latitude_range[1])
Nλ = 25 * (longitude_range[2] - longitude_range[1])

grid = LatitudeLongitudeGrid(size = (Nλ, Nφ, 1),
                             latitude = latitude_range,
                             longitude = longitude_range,
                             z = (0, 1),
                             halo = (7, 7, 1))

# Next, we generate the bathymetry with `regrid_bathymetry`, which downloads the ETOPO2022
# dataset, regrids it onto `grid`, and returns the bottom height as a `Field`.
# The three calls below show how the keyword arguments shape the result:
#
# - `rough_bathymetry` uses the default parameters, with a single interpolation pass.
# - `smooth_bathymetry` uses 40 interpolation passes, which smooths the bathymetry.
# - `one_basin_bathymetry` uses `major_basins = 1`, which retains only the largest connected
#   basin and fills disconnected regions (e.g., lakes) with land.

rough_bathymetry = regrid_bathymetry(grid)
smooth_bathymetry = regrid_bathymetry(grid; interpolation_passes = 40)
one_basin_bathymetry = regrid_bathymetry(grid; major_basins = 1)
nothing #hide

# Finally, we visualize the three bathymetries, masking land in gray.

for bathymetry in (rough_bathymetry, smooth_bathymetry, one_basin_bathymetry)
    land = interior(bathymetry) .≥ 0
    interior(bathymetry)[land] .= NaN
end

fig = Figure(size=(850, 1150))

ax = Axis(fig[1, 1], title = "Rough bathymetry", xlabel = "Longitude", ylabel = "Latitude")
hm = heatmap!(ax, rough_bathymetry, nan_color=:lightgray, colormap = Reverse(:deep))

ax = Axis(fig[2, 1], title = "Smooth bathymetry", xlabel = "Longitude", ylabel = "Latitude")
hm = heatmap!(ax, smooth_bathymetry, nan_color=:lightgray, colormap = Reverse(:deep))

ax = Axis(fig[3, 1], title = "Bathymetry with only one basin", xlabel = "Longitude", ylabel = "Latitude")
hm = heatmap!(ax, one_basin_bathymetry, nan_color=:lightgray, colormap = Reverse(:deep))

Colorbar(fig[1:3, 2], hm, height = Relative(3/4), label = "Bottom height (m)")

save("different_bottom_heights.png", fig)
nothing #hide

# ![](different_bottom_heights.png)
