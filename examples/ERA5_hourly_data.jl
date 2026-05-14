# # ERA5 hourly atmospheric data on single- and pressure-levels
#
# This walkthrough covers downloading ERA5 reanalysis fields from the
# Copernicus Climate Data Store (CDS), with the Rain in Shallow Cumulus Over
# the Ocean (RICO) trade-wind cumulus campaign [rauber2007rain](@citep) as a
# unifying case study. We consider both single-level (2-D) and pressure-level
# (3-D) fields with two subsetting approaches (bounding box and column) that
# restrict the amount of data requested through the CDS API.
#
# Our focus is on a week within the  *undisturbed period* (Dec 27 2004 – Jan 2 2005)
# by [vanZanten2011](@citet), during which a mean precipitation was observed.
# Here, we briefly analyze and present the ERA5 data, referring to published
# material where appropriate.
#
# Three scales are demonstrated:
#
# 1. **Global scale** - surface winds and Stokes drift over the entire globe
# 2. **Synoptic scale** — surface precipitation over an Atlantic-centered
#    region covering the tropics
# 3. **Microscale** — single- and pressure-level ``u``, ``v``, ``T``, ``qᵛ`` over the
#    RICO study box; pressure-level ``qᶜˡ``, ``qʳ`` in a single column.
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
using Suppressor

# ## Study definition
#
# For demonstration purposes, we select one week within [vanZanten2011](@citet)'s
# undisturbed period, giving us 168 hourly snapshots. This is used
# by both sections below.

dates = DateTime(2004, 12, 27):Hour(1):DateTime(2005, 1, 2, 23)
nothing #hide

# To subset the ERA5 data, we define two types of `region`s.
# We introduce two `BoundingBox`es:

## Synoptic-scale region, cf. Fig. 1 by [rauber2007rain](@citet)
synoptic_region = BoundingBox(latitude=(-25, 35), longitude=(-110, 30))

## RICO study area near Antigua and Barbuda
rico_region = BoundingBox(latitude=(17, 18.5), longitude=(-62.5, -61))
nothing #hide

# And a single `Column`, which has no lateral dimensionality:

rico_column = Column(-61.5, 18) # longitude, latitude
nothing #hide

# ## §1 Global conditions
#
# This part of the analysis is based on [ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels),
# available from 1940 to present. In this section, we evaluate ocean
# surface conditions based on the near-surface wind velocity at 10-m ASL
# and Stokes drift.

dataset = ERA5HourlySingleLevel()
nothing #hide

# Note that ERA5 atmospheric variables (wind) live on a **0.25°** grid (1440×721),
# whereas ocean wave variables (Stokes drift) live on a **0.5°** grid (720×361).

# ### Metadata definition
#
# We first define metadata for each variable at a single date.
#
# !!! note "Metadatum vs Metadata"
#     `Metadatum` is used for single dates while `Metadata` handles
#      multiple dates.
#
# No `region` kwarg is specified because we want to download global fields.

date = first(dates)
u_stokes_meta = Metadatum(:eastward_stokes_drift;  dataset, date)
v_stokes_meta = Metadatum(:northward_stokes_drift; dataset, date)
u_wind_meta   = Metadatum(:eastward_velocity;      dataset, date)
v_wind_meta   = Metadatum(:northward_velocity;     dataset, date)

# ### Build a grid and create fields
#
# We build a single `LatitudeLongitudeGrid` and use `set!` to download
# and interpolate all four variables onto it.

grid = LatitudeLongitudeGrid(size = (1440, 720, 1),
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

# ### Compute speeds and plot
#
# We use Oceananigans abstract operations to compute the speed fields,
# then plot them directly as heatmaps on latitude–longitude axes.

stokes_speed = sqrt(u_stokes^2 + v_stokes^2)
wind_speed   = sqrt(u_wind^2   + v_wind^2)

lon, lat, _ = nodes(u_stokes)

fig = Figure(size=(1200, 600))

ax1 = Axis(fig[1, 1]; title="Stokes drift speed (m/s)",
           xlabel="Longitude", ylabel="Latitude")
ax2 = Axis(fig[1, 2]; title="10-m wind speed (m/s)",
           xlabel="Longitude", ylabel="Latitude")

hm1 = heatmap!(ax1, lon, lat, stokes_speed; colormap=:speed, colorrange=(0, 0.3))
hm2 = heatmap!(ax2, lon, lat, wind_speed;   colormap=:speed, colorrange=(0, 20))

Colorbar(fig[2, 1], hm1; vertical=false, width=Relative(0.8), label="m/s")
Colorbar(fig[2, 2], hm2; vertical=false, width=Relative(0.8), label="m/s")

Label(fig[0, :],
      "ERA5 Stokes Drift and Surface Wind — $(Dates.format(date, "yyyy-mm-dd HH:MM")) UTC";
      fontsize=20)

fig

# ## §2 Synoptic conditions
#
# New in this section:
#
# - A `BoundingBox` defined by latitude and longitude ranges is introduced for
#   region restriction.
# - We work with a `FieldTimeSeries`, constructed from metadata, to construct
#   an ERA5 time series. Field data are downloaded on the fly.

precip_meta = Metadata(:total_precipitation; dataset, dates, region = synoptic_region)
precip_series = @suppress_out FieldTimeSeries(precip_meta)
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

# ## §3 Microscale conditions
#
# ### Time history of precipitation at the RICO location
#
# New in this section:
#
# - `Column` replaces `BoundingBox` as the region restriction. This issues
#   a smaller CDS request that downloads only the cells needed to linearly
#   interpolate (by default) to the requested (longitude, latitude) coordinate.
#   A `Column(...; interpolation = Nearest())` option is also available.
# - We explicitly load all fields in the timeseries into memory through
#   `FieldTimeSeries...; time_indices_in_memory = Nt)` whereas the default is
#   to load only two snapshots at a time. Having the full timeseries in memory
#   facilitates any operations we want to perform on the data — in this case,
#   a units conversion.
#
# !!! note "Slicing"
#     We could have sliced the `precip_series` from above, but we illustrate here
#     a seperate data retrieval path.

precip_col_meta = Metadata(:total_precipitation; dataset, dates, region = rico_column)
precip_col_series = @suppress_out FieldTimeSeries(precip_col_meta; time_indices_in_memory = Nt)
nothing #hide

# ERA5 `total_precipitation` is in m (liquid-water-equivalent depth). Note
# that this is an *accumulated* rather than instantaneous quantity (more
# discussion [here](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Meanrates/fluxesandaccumulations))
# with an accumulation period of 1 hour. We convert to a latent-heat-equivalent
# flux in W/m² to compare with [vanZanten2011](@citet)'s reported 21 W m⁻² mean.

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

# In comparison with observations ([vanZanten2011](@citet)'s Fig. 1), we see different
# day-to-day variability in the reanalysis, with a wider range of values and greater
# mean precipitation.

@show mean(precip_W_m2)

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
# geopotential height. `FieldTimeSeries` would also have automatically
# downloaded this field on demand, but pre-downloading saves API calls.
#
# !!! note "Geopotential"
#     If the geopotential field isn't available, the fallback is to
#     estimate geopotential heights from the international standard atmosphere.

variables = [:specific_cloud_liquid_water_content,
             :specific_rain_water_content,
             :geopotential]
@suppress_out download_dataset(variables, ds_pl, dates; region = rico_column)
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

# Render the Hovmöller diagram using `heatmap`, with the same x-ticks as `fig2`.

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

# This figure tells the same story as `fig2` in a different way.

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

T_series = @suppress_out FieldTimeSeries(T_meta)
q_series = @suppress_out FieldTimeSeries(q_meta)
u_series = @suppress_out FieldTimeSeries(u_meta)
v_series = @suppress_out FieldTimeSeries(v_meta)
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

# Lastly, plot the profiles (cf. [vanZanten2011](@citet), Fig. 2).

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

# ## §4 Per-column vertical interpolation and downscaling
#
# `ERA5HourlyPressureLevels` builds its native vertical coordinate from the
# instantaneous geopotential Φ(λ,φ,p)/g, with sub-surface levels clipped at the
# surface (see issue #236) — `Oceananigans.interpolate!` bisects each column's
# actual heights inside the kernel. We use this to drop ERA5 specific humidity
# onto a 4×-downscaled LAM grid at a fixed altitude (RICO cloud base ≈ 800 m)
# and animate the result.

target_grid = LatitudeLongitudeGrid(CPU();
                                    size = (24, 24, 1),
                                    longitude = rico_region.longitude,
                                    latitude  = rico_region.latitude,
                                    z = (800.0, 801.0),
                                    halo = (2, 2, 1),
                                    topology = (Bounded, Bounded, Bounded))

q2d   = CenterField(target_grid)
q_obs = Observable(interior(q2d)[:, :, 1])

fig5 = Figure(size = (700, 600))
ax5  = Axis(fig5[1, 1]; xlabel = "Longitude [°]", ylabel = "Latitude [°]",
            aspect = DataAspect(), title = "ERA5 qᵛ at z = 800 m")
hm = heatmap!(ax5, λnodes(target_grid, Center(), Center(), Center()),
                   φnodes(target_grid, Center(), Center(), Center()),
                   q_obs; colormap = :viridis)
Colorbar(fig5[1, 2], hm; label = "qᵛ [kg kg⁻¹]")

## Each frame: load qᵛ on its per-column-z grid (one CDS file per hour, cached
## after the first run), interpolate down onto the 2-D target at z = 800 m.
record(fig5, "rico_qv_animation.mp4", dates; framerate = 12) do date
    set!(q2d, Metadatum(:specific_humidity; dataset = ds_pl, date, region = rico_region))
    q_obs[] = interior(q2d)[:, :, 1]
    ax5.title = "ERA5 qᵛ at 800 m — " * Dates.format(date, dateformat"u d HH:MM") * " UTC"
end

@info "Wrote rico_qv_animation.mp4"
fig5
