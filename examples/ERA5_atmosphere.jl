# # ERA5 hourly atmospheric data on single- and pressure-levels
#
# This walkthrough covers downloading ERA5 reanalysis fields from the
# Copernicus Climate Data Store (CDS), with the Rain in Shallow Cumulus Over
# the Ocean (RICO) trade-wind cumulus campaign (Rauber et al. 2007) as a
# unifying case study. We consider both single-level (2-D) and pressure-level
# (3-D) fields with two subsetting approaches (bounding box and column).
#
# Our focus is on a week within van Zanten et al.'s *undisturbed period*
# (Dec 27 2004 – Jan 2 2005), during which a mean precipitation of 21 W m⁻²
# (∼1 mm d⁻¹) was observed. Note that the day-to-day variability measured by
# RICO instrumentation was not necessarily captured by numerical weather
# prediction models. Here, we briefly analyze and present the ERA5 data,
# referring to published material where appropriate.
#
# Two scales are demonstrated:
#
# 1. **Synoptic context** — surface precipitation over an Atlantic-centered
#    region covering the tropics
# 2. **Microscale** — single- and pressure-level u, v, T, qᵛ over the
#    RICO study box; pressure-level qᶜˡ, qʳ in a single column.
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add NumericalEarth CDSAPI Oceananigans CairoMakie"
# ```
#
# You also need CDS API credentials in `~/.cdsapirc`.
# See <https://cds.climate.copernicus.eu/how-to-api> for setup instructions.

using NumericalEarth
using NumericalEarth.DataWrangling: download_dataset
using NumericalEarth.DataWrangling.ERA5
using CDSAPI
using Dates
using Oceananigans
using Statistics
using CairoMakie

# ## Study definition
#
# For demonstration purposes, we select one week within van Zanten et al.
# (2011)'s undisturbed period, giving us 168 hourly snapshots. This is used
# by both sections below.

dates = DateTime(2004, 12, 27):Hour(1):DateTime(2005, 1, 2, 23)
nothing #hide

# To subset the ERA5 data, we define two types of `region`s.
# We introduce two `BoundingBox`es:

## Synoptic-scale region, cf. Fig. 1 in Rauber et al. 2007
synoptic_region = BoundingBox(latitude=(-25, 35), longitude=(-110, 30))

## RICO study area near Antigua and Barbuda
rico_region = BoundingBox(latitude=(17, 18.5), longitude=(-62.5, -61))
nothing #hide

# And a single `Column`, which has no lateral dimensionality:

rico_column = Column(-61.5, 18) # longitude, latitude
nothing #hide

# ## §1 Synoptic context: regional precipitation
#
# This part of the analysis is based on [ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels),
# available from 1940 to present.
dataset = ERA5HourlySingleLevel()
nothing #hide

# We use `Metadata` (multi-date) + `BoundingBox` (region restriction) + the
# `FieldTimeSeries(metadata)` constructor as the basic building blocks
# for constructing an ERA5 time series. Field data are downloaded on the fly
# by `FieldTimeSeries`.

precip_meta   = Metadata(:total_precipitation; dataset, dates, region=synoptic_region)
precip_series = FieldTimeSeries(precip_meta)
nothing #hide

# For brevity, we plot the time-averaged (rather than instantaneous) precipitation over
# the region.

Nt = length(dates)
λ, φ, _ = nodes(precip_series[1])

## ERA5 `total_precipitation` is in m/hour; convert to mm/day.
to_mm_day = 1000 * 24
precip_avg = mean(interior(precip_series[n], :, :, 1) for n in 1:Nt) * to_mm_day

fig1 = Figure(size=(900, 400))
ax1 = Axis(fig1[1, 1],
           title = "Mean precipitation, $(first(dates)) to $(last(dates))",
           xlabel = "Longitude (°)", ylabel = "Latitude (°)",
           xticks = -90:30:30)
hm = heatmap!(ax1, λ, φ, precip_avg; colormap=:rain, colorrange=(0, 12))
Colorbar(fig1[1, 2], hm, label="Precipitation (mm/day)")

fig1

# ## §2 Microscale conditions
#
# ### Time history of precipitation at the RICO location
#
# New in this section:
#
# - `Column` replaces `BoundingBox` as the region restriction. This issues
#   a smaller CDS request that downloads only the cells needed to linearly
#   interpolate (by default) to the requested (longitude, latitude) coordinate.
#   The `Column(...; interpolation = Nearest())` option is also available.
# - We explicitly load all fields in the timeseries into memory through
#   `FieldTimeSeries...; time_indices_in_memory = Nt)` whereas the default is
#   to load only two snapshots at a time. Having the full timeseries in memory
#   facilitates any operations we want to perform on the data — in this case,
#   a units conversion.
#
# *Note: We could have sliced the `precip_series` from above, but we 
# illustrate here a seperate data retrieval path.*

precip_col_meta   = Metadata(:total_precipitation; dataset, dates, region = rico_column)
precip_col_series = FieldTimeSeries(precip_col_meta; time_indices_in_memory = Nt)
nothing #hide

# ERA5 `total_precipitation` is in m/hr (liquid-water-equivalent depth);
# we convert to a latent-heat-equivalent flux in W/m² to compare with
# van Zanten et al.'s reported 21 W m⁻² mean.

ρᴸ = 1000   # kg/m³
Lᵛ = 2.5e6  # J/kg, latent heat of vaporization

to_W_per_m2 = ρᴸ * Lᵛ / 3600  # m/hr → W/m²

precip_W_m2 = interior(precip_col_series, 1, 1, 1, :)
precip_W_m2 .*= to_W_per_m2
nothing #hide

# Now, plot the precipitation time history.

fig2 = Figure(size=(900, 300))

## Tick at each day boundary (00:00 of each day in the window).
day_dts    = first(dates):Day(1):last(dates)
day_ticks  = (0:length(day_dts)-1) .* 86400.0   # seconds since first(dates)
day_labels = Dates.format.(day_dts, dateformat"u d")

ax2 = Axis(fig2[1, 1],
           title  = "Precipitation at $(rico_column.longitude)°E, $(rico_column.latitude)°N",
           ylabel = "Precipitation [W m⁻²]",
           xticks = (day_ticks, day_labels))
lines!(ax2, precip_col_series.times, precip_W_m2; color=:steelblue, label="ERA5 hourly data")
hlines!(ax2, [21.0]; color=:black, linestyle=:dash, label="van Zanten mean (21 W m⁻²)")
axislegend(ax2; position=:rt)

fig2

# ### Time-height of cloud liquid and rain water content at the RICO location
#
# This part of the analysis is based on [ERA5 hourly data on pressure levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels),
# also available from 1940 to present. What's new:
#
# - We restrict the data retrieval to the lower troposphere (≥ 250 hPa,
#   surface up to ~10 km). This returns data with 21 vertical levels
#   instead of all 37 standard pressure levels — the full list is given
#   by `ERA5_all_pressure_levels`.
# - `download_dataset(variables, dataset, dates; region)` bundles
#   multi-variable requests into a single CDS API call — fewer round trips
#   than calling `download_dataset` per variable, which is what
#   `FieldTimeSeries` does automatically on demand.

selected_levels = filter(≥(250hPa), ERA5_all_pressure_levels)
ds_pl = ERA5HourlyPressureLevels(selected_levels)

## Selected pressure levels [hPa]
ds_pl.pressure_levels' / hPa

# 3-D data are downloaded in a `Column` region, resulting in one 1-D field
# per snapshot.
#
# The download list also includes`:geopotential` because a pressure-level
# `FieldTimeSeries` grid derives its `z`-coordinate from the time-mean
# geopotential height. `FieldTimeSeries` would also have auomatically
# downloaded this field on demand, but pre-downloading saves API calls.
# _Note: If the geopotential field isn't available, the fallback is to
# estimate geopotential heights from the international standard atmosphere._

variables = [:specific_cloud_liquid_water_content,
             :specific_rain_water_content,
             :geopotential]
download_dataset(variables, ds_pl, dates; region = rico_column)
nothing #hide

# Load the downloaded data and stack the column profile from each time into
# a (Nt × Nz) matrix.

qᶜ_col_meta   = Metadata(:specific_cloud_liquid_water_content; dataset = ds_pl, dates, region = rico_column)
qʳ_col_meta   = Metadata(:specific_rain_water_content;         dataset = ds_pl, dates, region = rico_column)
qᶜ_col_series = FieldTimeSeries(qᶜ_col_meta)
qʳ_col_series = FieldTimeSeries(qʳ_col_meta)

z_col  = znodes(qᶜ_col_series[1])
Nz_col = length(z_col)

qᶜ_data = zeros(Nt, Nz_col)
qʳ_data = zeros(Nt, Nz_col)
for n in 1:Nt
    qᶜ_data[n, :] = vec(interior(qᶜ_col_series[n]))
    qʳ_data[n, :] = vec(interior(qʳ_col_series[n]))
end
qᶜ_data .*= 1000   # kg/kg → g/kg
qʳ_data .*= 1000   # kg/kg → g/kg
nothing #hide

# Render as a heatmap, sharing the day-boundary x-ticks with `fig2`.

fig3 = Figure(size=(900, 600))
ax_qc = Axis(fig3[1, 1],
             title  = "Specific cloud liquid water content at $(rico_column.longitude)°E, $(rico_column.latitude)°N",
             ylabel = "Height [m]",
             xticks = (day_ticks, day_labels))
ax_qr = Axis(fig3[2, 1],
             title  = "Specific rain water content at $(rico_column.longitude)°E, $(rico_column.latitude)°N",
             ylabel = "Height [m]",
             xticks = (day_ticks, day_labels))
hm_qc = heatmap!(ax_qc, qᶜ_col_series.times, z_col, qᶜ_data; colormap=:Blues)
hm_qr = heatmap!(ax_qr, qʳ_col_series.times, z_col, qʳ_data; colormap=:Blues)
Colorbar(fig3[1, 2], hm_qc, label="qᶜˡ [g kg⁻¹]")
Colorbar(fig3[2, 2], hm_qr, label="qʳ [g kg⁻¹]")

linkaxes!(ax_qc, ax_qr)
ylims!(ax_qc, 0, 4000)
hidexdecorations!(ax_qc, grid=false)

fig3

# ### Profiles at the RICO location
#
# We use the filtered pressure-levels dataset from before, as well as the
# previously defined `BoundingBox`. As before, we bundle API requests to
# expedite the data retrieval.

variables = [:temperature, :specific_humidity,
             :eastward_velocity, :northward_velocity,
             :geopotential]
download_dataset(variables, ds_pl, dates; region = rico_region)

T_meta = Metadata(:temperature;        dataset=ds_pl, dates, region=rico_region)
q_meta = Metadata(:specific_humidity;  dataset=ds_pl, dates, region=rico_region)
u_meta = Metadata(:eastward_velocity;  dataset=ds_pl, dates, region=rico_region)
v_meta = Metadata(:northward_velocity; dataset=ds_pl, dates, region=rico_region)

T_series = FieldTimeSeries(T_meta)
q_series = FieldTimeSeries(q_meta)
u_series = FieldTimeSeries(u_meta)
v_series = FieldTimeSeries(v_meta)
nothing #hide

# Calculate mean profiles and quantities of interest.

z       = znodes(T_series[1])
Nz      = length(z)
p_levs  = sort(selected_levels, rev=true) ./ hPa   # Pa → hPa, from bottom-to-top

function horizontal_mean_profiles(series)
    profiles = zeros(Nz, Nt)
    for n in 1:Nt
        profiles[:, n] = mean(interior(series[n], :, :, :), dims=(1, 2))
    end
    return profiles
end

T_profiles = horizontal_mean_profiles(T_series)
q_profiles = horizontal_mean_profiles(q_series)
u_profiles = horizontal_mean_profiles(u_series)
v_profiles = horizontal_mean_profiles(v_series)

## T → potential temperature: θ = T (p₀/p)^(R/cₚ)
Rᵈ_over_cᵖ = 0.286
pˢᵗ = 1000
θ_profiles = T_profiles .* (pˢᵗ ./ p_levs) .^ Rᵈ_over_cᵖ

## specific humidity: kg/kg → g/kg
q_profiles .*= 1000
nothing #hide

# Lastly, plot the profiles (cf. van Zanten et al. 2011, Fig. 2).

fig4 = Figure(size=(900, 540), fontsize=12)

fig4_title = string("Mean ± IQR vertical profiles over the RICO box, ",
                    Dates.format(first(dates), dateformat"u d yyyy"), " – ",
                    Dates.format(last(dates),  dateformat"u d yyyy"))
Label(fig4[0, 1:4], fig4_title;
      fontsize=14, font=:bold, halign=:center, tellwidth=false)
ax_θ = Axis(fig4[1, 1], xlabel="θ [K]",       ylabel="Height [m]")
ax_q = Axis(fig4[1, 2], xlabel="qᵛ [g kg⁻¹]", ylabel="Height [m]")
ax_u = Axis(fig4[1, 3], xlabel="u [m s⁻¹]",   ylabel="Height [m]", xticks=-10:2:2)
ax_v = Axis(fig4[1, 4], xlabel="v [m s⁻¹]",   ylabel="Height [m]", xticks=-10:2:0)

for (ax, profiles) in [(ax_θ, θ_profiles), (ax_q, q_profiles),
                       (ax_u, u_profiles), (ax_v, v_profiles)]
    μ  = vec(mean(profiles, dims=2))
    lo = [quantile(r, 0.25) for r in eachrow(profiles)]
    hi = [quantile(r, 0.75) for r in eachrow(profiles)]
    band!(ax, z, lo, hi; direction=:y, color=(:gray, 0.4))
    lines!(ax, μ, z; color=:black, linewidth=2)
end

xlims!(ax_θ, 295, 320)
xlims!(ax_q,   0,  15)
xlims!(ax_u, -10,   2)
xlims!(ax_v,  -9,  -1)
linkyaxes!(ax_θ, ax_q, ax_u, ax_v)
ylims!(ax_θ, 0, 4000)
hideydecorations!(ax_q, grid=false)
hideydecorations!(ax_u, grid=false)
hideydecorations!(ax_v, grid=false)

fig4
