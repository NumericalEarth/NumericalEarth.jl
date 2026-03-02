# # ERA5 winds and Stokes drift
#
# In this example, we download ERA5 10-meter wind and Stokes drift data
# from the Copernicus Climate Data Store, and plot global maps of the
# wind speed and Stokes drift speed side by side.
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add Oceananigans, NumericalEarth, CDSAPI, CairoMakie"
# ```
#
# You also need CDS API credentials in `~/.cdsapirc`.
# See <https://cds.climate.copernicus.eu/how-to-api> for setup instructions.

using NumericalEarth
using NumericalEarth.DataWrangling: Metadatum
using NumericalEarth.DataWrangling.ERA5: ERA5Hourly
using CDSAPI

using Oceananigans
using CairoMakie
using Dates

# ## Define metadata
#
# ERA5 atmospheric variables (wind) live on a 0.25° grid (1440×721),
# while ocean wave variables (Stokes drift) live on a 0.5° grid (720×361).
# We define metadata for each variable at a single date.

dataset = ERA5Hourly()
date = DateTime(2020, 1, 15, 12) # January 15, 2020 at 12:00 UTC

u_stokes_meta = Metadatum(:eastward_stokes_drift;  dataset, date)
v_stokes_meta = Metadatum(:northward_stokes_drift; dataset, date)
u_wind_meta   = Metadatum(:eastward_velocity;      dataset, date)
v_wind_meta   = Metadatum(:northward_velocity;     dataset, date)

# ## Build a grid and create fields
#
# We build a single `LatitudeLongitudeGrid` and use `set!` to download
# and interpolate all four variables onto it.

grid = LatitudeLongitudeGrid(size = (1440, 721, 1),
                             longitude = (0, 360),
                             latitude = (-90, 90),
                             z = (0, 1))

u_stokes = CenterField(grid)
v_stokes = CenterField(grid)
u_wind   = CenterField(grid)
v_wind   = CenterField(grid)

set!(u_stokes, u_stokes_meta)
set!(v_stokes, v_stokes_meta)
set!(u_wind,   u_wind_meta)
set!(v_wind,   v_wind_meta)

# ## Compute speed and plot
#
# We use Oceananigans abstract operations to compute the speed fields,
# then plot them directly as heatmaps on latitude–longitude axes.

stokes_speed = sqrt(u_stokes^2 + v_stokes^2)
wind_speed   = sqrt(u_wind^2   + v_wind^2)

lon, lat, _ = nodes(u_stokes)

fig = Figure(size=(1200, 600))

ax1 = Axis(fig[1, 1]; title="Stokes drift speed (m/s)",
           xlabel="Longitude", ylabel="Latitude")
ax2 = Axis(fig[1, 2]; title="10m wind speed (m/s)",
           xlabel="Longitude", ylabel="Latitude")

hm1 = heatmap!(ax1, lon, lat, stokes_speed; colormap=:solar, colorrange=(0, 0.3))
hm2 = heatmap!(ax2, lon, lat, wind_speed;   colormap=:solar, colorrange=(0, 20))

Colorbar(fig[2, 1], hm1; vertical=false, width=Relative(0.8), label="m/s")
Colorbar(fig[2, 2], hm2; vertical=false, width=Relative(0.8), label="m/s")

Label(fig[0, :],
      "ERA5 Stokes Drift and Surface Wind — $(Dates.format(date, "yyyy-mm-dd HH:MM")) UTC";
      fontsize=20)

save("ERA5_stokes_drift_and_wind.png", fig)

display(fig)
