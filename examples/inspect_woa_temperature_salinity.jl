# # Inspecting World Ocean Atlas (WOA) temperature and salinity
#
# This example loads and visualizes the WOA climatological temperature and salinity
# with NumericalEarth.jl. The World Ocean Atlas provides objectively analyzed
# climatological means of various ocean properties at 1° resolution.

using Oceananigans
using NumericalEarth
using WorldOceanAtlasTools
using CairoMakie

arch = CPU()

# ## Loading the WOA annual climatology
#
# We create metadata for the WOA annual temperature and salinity climatology,
# then load each as an Oceananigans `Field` on the native WOA grid.

woa = MetadataSet(:temperature, :salinity; dataset=WOAAnnual())

fields = Field(woa, arch) ## (; temperature, salinity)
T, S = fields.temperature, fields.salinity

# ## Surface fields
#
# We plot the top level, that is, the sea surface, of temperature and salinity.

Nz = size(T.grid, 3)

fig = Figure(size=(1200, 800))

axT = Axis(fig[1, 1], title="WOA annual surface temperature (°C)")
hmT = heatmap!(axT, view(T, :, :, Nz), colorrange=(-2, 30), colormap=:thermal)
Colorbar(fig[1, 2], hmT)

axS = Axis(fig[2, 1], title="WOA annual surface salinity (PSU)")
hmS = heatmap!(axS, view(S, :, :, Nz), colorrange=(31, 37), colormap=:haline)
Colorbar(fig[2, 2], hmS)

fig

# ## Loading the WOA monthly climatology
#
# WOA also provides monthly climatologies. The `WOAMonthly()` dataset has 12 dates,
# January through December. A `Metadatum` without a `date` defaults to the first
# month, January.

january_temperature = Field(Metadatum(:temperature; dataset=WOAMonthly()), arch)
Nz = size(january_temperature.grid, 3)

fig = Figure(size=(1200, 400))
ax = Axis(fig[1, 1], title="WOA January surface temperature (°C)")
hm = heatmap!(ax, view(january_temperature, :, :, Nz), colorrange=(-2, 30), colormap=:thermal)
Colorbar(fig[1, 2], hm)

fig

# ## Setting WOA data on a custom grid
#
# We can also interpolate the WOA data onto a coarser Oceananigans grid.

grid = LatitudeLongitudeGrid(arch;
                             size = (90, 45, 20),
                             latitude = (-80, 80),
                             longitude = (0, 360),
                             z = (-2000, 0))

coarse_temperature = CenterField(grid)
coarse_salinity = CenterField(grid)

set!((; temperature = coarse_temperature, salinity = coarse_salinity), woa)

Nz = size(grid, 3)

fig = Figure(size=(1200, 400))

axT = Axis(fig[1, 1], title="Interpolated WOA surface temperature (°C)")
hmT = heatmap!(axT, view(coarse_temperature, :, :, Nz), colorrange=(-2, 30), colormap=:thermal)
Colorbar(fig[1, 2], hmT)

axS = Axis(fig[1, 3], title="Interpolated WOA surface salinity (PSU)")
hmS = heatmap!(axS, view(coarse_salinity, :, :, Nz), colorrange=(31, 37), colormap=:haline)
Colorbar(fig[1, 4], hmS)

fig
