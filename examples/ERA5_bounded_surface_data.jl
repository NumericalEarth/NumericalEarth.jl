# Download 3-D snapshots of the atmospheric state from ERA5, e.g., to be used with FieldTimeSeries
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add NumericalEarth CDSAPI"
# ```
#
# You also need CDS API credentials in `~/.cdsapirc`.
# See <https://cds.climate.copernicus.eu/how-to-api> for setup instructions.
#
# See <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download> for
# more information about single-level data

using NumericalEarth.DataWrangling: Metadata, BoundingBox, download_dataset
using NumericalEarth.DataWrangling.ERA5
using CDSAPI
using Dates
using Oceananigans

using Statistics: mean
using CairoMakie
#precip_cmap = :rain
precip_cmap = cgrad([:indigo, :darkblue, :blue, :deepskyblue, :cyan,
                     :palegreen, :green, :yellow, :orange, :red, :pink])


dataset = ERA5HourlySingleLevel()

#dates = DateTime(2004, 12, 16):Hour(1):DateTime(2005, 01, 09)  # van Zanten et al 2011 study period
dates = DateTime(2004, 12, 16):Hour(1):DateTime(2004, 12, 23)  # shorter time range for demo

# Rauber et al 2007, Fig 1: precip map
region = BoundingBox(latitude=(-25, 35), longitude=(-110, 30))

# OPTIONAL: download all variables at once (fewer CDS API requests)
# variables = [:total_precipitation,
#              :sea_surface_temperature,
#              :temperature, # at 2 m
#              :dewpoint_temperature, # at 2 m
#              :surface_pressure]
# download_dataset(variables, dataset, dates; region)

# ## Create time series

precip_meta = Metadata(:total_precipitation; dataset, dates, region)
precip_series = FieldTimeSeries(precip_meta)

# ## Plotting

Nt = length(precip_series.times)
λ, φ, _ = nodes(precip_series[1])

# ERA5 total_precipitation is in metres per hour; convert to mm/day
to_mm_day = 1000 * 24

# ### Time-averaged precipitation map

precip_avg = mean(interior(precip_series[n], :, :, 1) for n in 1:Nt) .* to_mm_day

fig1 = Figure(size=(900, 400))
ax1 = Axis(fig1[1, 1],
           title = "Mean precipitation (2004-12-16 to 2005-01-08)",
           xlabel = "Longitude (°)",
           ylabel = "Latitude (°)")
ax1.xticks = -90:30:30
hm = heatmap!(ax1, λ, φ, precip_avg; colormap=precip_cmap, colorrange=(0, 12))
Colorbar(fig1[1, 2], hm, label="Precipitation (mm/day)")
save("ERA5_mean_precipitation.png", fig1)

# ### Precipitation animation

fig2 = Figure(size=(900, 400))

n = Observable(1)
precip_n = @lift interior(precip_series[$n], :, :, 1) .* to_mm_day
anim_title = @lift "Precipitation (mm/day), " * string(first(dates) + Second(round(Int, precip_series.times[$n])))

ax2 = Axis(fig2[1, 1],
           title = anim_title,
           xlabel = "Longitude (°)",
           ylabel = "Latitude (°)")
ax2.xticks = -90:30:30

hm2 = heatmap!(ax2, λ, φ, precip_n; colormap=precip_cmap, colorrange=(0, 12))
Colorbar(fig2[1, 2], hm2, label="Precipitation (mm/day)")

record(fig2, "ERA5_precipitation.mp4", 1:Nt; framerate=12) do nn
    n[] = nn
end

