# # Inspecting World Ocean Atlas (WOA) Temperature and Salinity
#
# This example demonstrates how to load and visualize WOA climatological
# temperature and salinity data using NumericalEarth.jl.
# The World Ocean Atlas provides objectively analyzed climatological mean
# fields for various ocean properties at 1° resolution.

using Oceananigans
using NumericalEarth
using WorldOceanAtlasTools
using CairoMakie

arch = CPU()

# ## Loading WOA annual climatology
#
# We create metadata for WOA annual temperature and salinity climatology,
# then load each as an Oceananigans `Field` on the native WOA grid.

T_metadata = Metadatum(:temperature; dataset=WOAAnnual())
S_metadata = Metadatum(:salinity;    dataset=WOAAnnual())

T = Field(T_metadata, arch)
S = Field(S_metadata, arch)

# ## Surface fields
#
# Let's visualize the surface (top-most level) of temperature and salinity.

Nz = size(T.grid, 3)

fig = Figure(size=(1200, 800))

axT = Axis(fig[1, 1], title="WOA Annual Surface Temperature (°C)")
hmT = heatmap!(axT, interior(T, :, :, Nz), colorrange=(-2, 30), colormap=:thermal)
Colorbar(fig[1, 2], hmT)

axS = Axis(fig[2, 1], title="WOA Annual Surface Salinity (PSU)")
hmS = heatmap!(axS, interior(S, :, :, Nz), colorrange=(31, 37), colormap=:haline)
Colorbar(fig[2, 2], hmS)

current_figure()

# ## Loading WOA monthly climatology
#
# WOA also provides monthly climatological fields. The `WOAMonthly()` dataset
# has 12 dates representing January through December. A `Metadatum` with the
# default date corresponds to January (the first month).

T_jan = Field(Metadatum(:temperature; dataset=WOAMonthly()), arch)
Nz = size(T_jan.grid, 3)

fig = Figure(size=(1200, 400))
ax_jan = Axis(fig[1, 1], title="WOA January Surface Temperature (°C)")
hm_jan = heatmap!(ax_jan, interior(T_jan, :, :, Nz), colorrange=(-2, 30), colormap=:thermal)
Colorbar(fig[1, 2], hm_jan)

current_figure()

# ## Setting WOA data on a custom grid
#
# We can also interpolate WOA data onto a coarser Oceananigans grid.

grid = LatitudeLongitudeGrid(arch;
                             size = (90, 45, 20),
                             latitude = (-80, 80),
                             longitude = (0, 360),
                             z = (-2000, 0))

T_interp = CenterField(grid)
S_interp = CenterField(grid)

set!(T_interp, T_metadata)
set!(S_interp, S_metadata)

Nz_interp = size(grid, 3)

fig = Figure(size=(1200, 400))

ax1 = Axis(fig[1, 1], title="Interpolated WOA Temperature (°C) at surface")
hm1 = heatmap!(ax1, interior(T_interp, :, :, Nz_interp), colorrange=(-2, 30), colormap=:thermal)
Colorbar(fig[1, 2], hm1)

ax2 = Axis(fig[1, 3], title="Interpolated WOA Salinity (PSU) at surface")
hm2 = heatmap!(ax2, interior(S_interp, :, :, Nz_interp), colorrange=(31, 37), colormap=:haline)
Colorbar(fig[1, 4], hm2)

current_figure()
