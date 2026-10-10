# # ERA5 and GloFAS reanalysis data
#
# This walkthrough covers downloading ERA5 reanalysis fields from the
# Copernicus Climate Data Store (CDS), with the Rain in Shallow Cumulus Over
# the Ocean (RICO) trade-wind cumulus campaign [rauber2007rain](@citep) as a
# unifying case study. We consider both single-level (2-D) and pressure-level
# (3-D) fields with two subsetting approaches (bounding box and column) that
# restrict the amount of data requested through the CDS API.
#
# The final section turns to **GloFAS** river discharge — ERA5 runoff routed to
# river mouths by a hydrological model — and shows how `GloFASPrescribedLand`
# places that freshwater on the ocean coastline.
#
# We focus on the first four days of the *undisturbed period*
# (Dec 27 2004 – Jan 2 2005) defined by [vanZanten2011](@citet), for which they
# report the observed mean precipitation. We briefly analyze and present the
# ERA5 data, referring to published material where appropriate.
#
# Three scales are demonstrated:
#
# 1. **Global scale** — surface winds and Stokes drift over the entire globe
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

using Downloads: download
using NumericalEarth
using NumericalEarth.DataWrangling.ERA5
using CDSAPI
using Dates
using Oceananigans
using Oceananigans.Units
using Oceananigans.Fields: interpolate!
using Oceananigans.Grids: x_domain
using Statistics
using CairoMakie
using Suppressor

# ## Study definition
#
# For demonstration purposes, we select four days within
# [vanZanten2011](@citet)'s undisturbed period, which gives 96 hourly snapshots
# used throughout the sections below.

dates = DateTime(2004, 12, 27):Hour(1):DateTime(2004, 12, 30, 23)
nothing #hide

# To subset the ERA5 data we use two kinds of `region`. First, two `BoundingBox`es:

## Synoptic-scale region, cf. Fig. 1 by [rauber2007rain](@citet)
synoptic_region = BoundingBox(latitude=(-25, 35), longitude=(-110, 30))

## RICO study area near Antigua and Barbuda
rico_region = BoundingBox(latitude=(17.5, 18.5), longitude=(-62, -61))
nothing #hide

# Second, a single `Column`, which has no horizontal extent:

rico_column = Column(-61.5, 18) # longitude, latitude
nothing #hide

# ## §1 Global conditions
#
# This part of the analysis is based on [ERA5 hourly data on single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels),
# available from 1940 to present. In this section, we evaluate ocean
# surface conditions using the wind velocity 10 m above sea level and the
# Stokes drift.

dataset = ERA5HourlySingleLevel()
nothing #hide

# Note that ERA5 atmospheric variables (wind) live on a **0.25°** grid (1440×721),
# whereas ocean wave variables (Stokes drift) live on a **0.5°** grid (720×361).

# ### Metadata definition
#
# We first define metadata for each variable at a single date.
#
# !!! note "Metadatum vs Metadata"
#     `Metadatum` describes a single date, while `Metadata` describes
#     multiple dates.
#
# We omit the `region` keyword argument because we want global fields.

date = first(dates)
metadata = (uˢ  = Metadatum(:eastward_stokes_drift;  dataset, date),
            vˢ  = Metadatum(:northward_stokes_drift; dataset, date),
            u₁₀ = Metadatum(:eastward_velocity;      dataset, date),
            v₁₀ = Metadatum(:northward_velocity;     dataset, date))
nothing #hide

# ### Build a grid and create fields
#
# We build a single `LatitudeLongitudeGrid` and use `set!` to download
# and interpolate all four variables onto it.

grid = LatitudeLongitudeGrid(size = (1440, 720, 1),
                             longitude = (0, 360),
                             latitude = (-90, 90),
                             z = (0, 1))

uˢ  = CenterField(grid)
vˢ  = CenterField(grid)
u₁₀ = CenterField(grid)
v₁₀ = CenterField(grid)

set!(uˢ,  metadata.uˢ)
set!(vˢ,  metadata.vˢ)
set!(u₁₀, metadata.u₁₀)
set!(v₁₀, metadata.v₁₀)

# ### Compute speeds and plot
#
# The speeds are Oceananigans `AbstractOperation`s, which we plot directly as
# heatmaps on latitude–longitude axes.

stokes_speed = sqrt(uˢ^2  + vˢ^2)
wind_speed   = sqrt(u₁₀^2 + v₁₀^2)

fig = Figure(size=(1200, 600))

ax1 = Axis(fig[1, 1]; title="Stokes drift speed (m/s)",
           xlabel="Longitude (°)", ylabel="Latitude (°)")
ax2 = Axis(fig[1, 2]; title="10-m wind speed (m/s)",
           xlabel="Longitude (°)", ylabel="Latitude (°)")

hm1 = heatmap!(ax1, stokes_speed; colormap=:speed, colorrange=(0, 0.3))
hm2 = heatmap!(ax2, wind_speed;   colormap=:speed, colorrange=(0, 20))

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
# - A `BoundingBox`, defined by latitude and longitude ranges, restricts the
#   region.
# - We build an ERA5 time series as a `FieldTimeSeries` constructed from
#   metadata. Field data are downloaded on the fly.
# - We interpolate specific humidity, column by column, from its native
#   pressure levels onto a fixed altitude (~800 m, the RICO cloud base).
#
# We download two fields over the same synoptic-scale box: surface
# precipitation (single-level, 2-D) and specific humidity on pressure levels
# (3-D). The pressure-level data live on a `PressureLevelGrid` whose
# z-coordinate is built from the instantaneous geopotential, and
# `Oceananigans.interpolate!` bisects the actual heights of each column inside
# the kernel — see [issue #236](https://github.com/NumericalEarth/NumericalEarth.jl/issues/236).

precipitation_metadata = Metadata(:total_precipitation; dataset, dates, region = synoptic_region)
precipitation_ts = @suppress_out FieldTimeSeries(precipitation_metadata)
nothing #hide

# Next, pressure-level ``qᵛ`` over the same region. We restrict the data retrieval
# to the lower troposphere (≥ 250 hPa, from the surface up to ~10 km), which gives
# 21 vertical levels instead of the 37 standard pressure levels.

selected_levels = filter(≥(250hPa), ERA5_all_pressure_levels)
pressure_level_dataset = ERA5HourlyPressureLevels(selected_levels)

humidity_metadata = Metadata(:specific_humidity; dataset = pressure_level_dataset, dates, region = synoptic_region)
humidity_ts = @suppress_out FieldTimeSeries(humidity_metadata)
nothing #hide

# For each frame we interpolate the pressure-level ``qᵛ`` onto a single-level
# grid at z = 800 m, the RICO cloud-base altitude.

cloud_base_grid = LatitudeLongitudeGrid(CPU();
                                        size = (140, 60, 1),
                                        longitude = x_domain(humidity_ts.grid),
                                        latitude  = synoptic_region.latitude,
                                        z = (800.0, 801.0),
                                        halo = (2, 2, 1),
                                        topology = (Bounded, Bounded, Bounded))
cloud_base_humidity = CenterField(cloud_base_grid)

# We build a two-panel animation: precipitation on top and ``qᵛ`` at z = 800 m
# below. NumericalEarth loads ERA5 `total_precipitation` as a mass flux
# (kg m⁻² s⁻¹), which is the liquid-water depth in mm s⁻¹; multiplying by one
# day gives mm/day. Each frame sets the plotted fields, `P` and `qᵛ`, which carry
# their own longitude and latitude.

Nt = length(dates)
n = Observable(1)

P  = similar(precipitation_ts[1])
qᵛ = CenterField(cloud_base_grid)

Pn  = @lift set!(P, precipitation_ts[$n] * day)
qᵛn = @lift begin
    interpolate!(cloud_base_humidity, humidity_ts[$n])
    set!(qᵛ, 1000 * cloud_base_humidity) # kg/kg → g/kg
end

fig1 = Figure(size=(900, 700))
title = @lift "ERA5 synoptic conditions — " * Dates.format(dates[$n], dateformat"u d HH:MM") * " UTC"
Label(fig1[0, 1:2], title; fontsize=14, font=:bold, tellwidth=false)

ax_p = Axis(fig1[1, 1]; title="Total precipitation",
            xlabel="Longitude (°)", ylabel="Latitude (°)",
            xticks=-90:30:30)
ax_q = Axis(fig1[2, 1]; title="Specific humidity at z = 800 m",
            xlabel="Longitude (°)", ylabel="Latitude (°)",
            xticks=-90:30:30)

hm_p = heatmap!(ax_p, Pn;  colormap=:rain,    colorrange=(0, 12))
hm_q = heatmap!(ax_q, qᵛn; colormap=:viridis, colorrange=(0, 20))

Colorbar(fig1[1, 2], hm_p, label="mm/day")
Colorbar(fig1[2, 2], hm_q, label="qᵛ [g/kg]")

linkaxes!(ax_p, ax_q)

CairoMakie.record(fig1, "synoptic_animation.mp4", 1:Nt; framerate=12) do nn
    n[] = nn
end
nothing #hide

# ![](synoptic_animation.mp4)

# ## §3 Microscale conditions
#
# ### Time history of precipitation at the RICO location
#
# New in this section:
#
# - `Column` replaces `BoundingBox` as the region restriction. This issues
#   a smaller CDS request that downloads only the cells needed to interpolate
#   (linearly, by default) to the requested (longitude, latitude) coordinate.
#   The option `Column(...; interpolation = Nearest())` is also available.
# - We load the whole time series into memory with
#   `FieldTimeSeries(...; time_indices_in_memory = Nt)`, whereas by default only
#   two snapshots are kept in memory at a time. With the full time series in
#   memory we can operate on the data in place — here, to convert units.
#
# !!! note "Slicing"
#     We could have sliced `precipitation_ts` from above; instead, we illustrate
#     a separate data retrieval path.

column_precipitation_metadata = Metadata(:total_precipitation; dataset, dates, region = rico_column)
column_precipitation_ts = @suppress_out FieldTimeSeries(column_precipitation_metadata; time_indices_in_memory = Nt)
nothing #hide

# ERA5 `total_precipitation` is an *accumulated* rather than an instantaneous
# quantity (more discussion [here](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Meanrates/fluxesandaccumulations)),
# with an accumulation period of 1 hour. NumericalEarth converts the hourly
# accumulated depth into a mean mass flux (kg m⁻² s⁻¹) when loading it. We
# multiply by the latent heat of vaporization to obtain a latent-heat-equivalent
# flux in W m⁻², which we compare with the 21 W m⁻² mean reported by
# [vanZanten2011](@citet).

ℒˡ = 2.5e6 # J kg⁻¹

interior(column_precipitation_ts) .*= ℒˡ
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

lines!(ax2, column_precipitation_ts; color=:steelblue, label="ERA5 hourly data")
hlines!(ax2, [21.0]; color=:black, linestyle=:dash, label="van Zanten mean (21 W m⁻²)")

axislegend(ax2; position=:rt)

fig2

# Compared with the observations ([vanZanten2011](@citet), Fig. 1), the reanalysis
# shows different day-to-day variability, a wider range of values, and a greater
# mean precipitation:

mean(column_precipitation_ts)

# ### Time-height of cloud liquid and rain water content at the RICO location
#
# This part of the analysis is based on [ERA5 hourly data on pressure levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels),
# also available from 1940 to present. What's new:
#
# - We restrict the data retrieval to the lower troposphere (≥ 250 hPa,
#   surface up to ~10 km). This returns data with 21 vertical levels
#   instead of all 37 standard pressure levels — the full list is given
#   by `ERA5_all_pressure_levels`.
# - `download(variables, dataset, dates; region)` bundles requests for several
#   variables into a single CDS API call. This needs fewer round trips than
#   calling `download` once per variable, which is what `FieldTimeSeries` does
#   automatically on demand.

## Selected pressure levels [hPa] (filtered to the lower troposphere in §2).
pressure_level_dataset.pressure_levels' / hPa

# We download 3-D data in a `Column` region, which yields one 1-D field per
# snapshot.
#
# The download list also includes `:geopotential`, because the grid of a
# pressure-level `FieldTimeSeries` derives its `z`-coordinate from the
# time-mean geopotential height. `FieldTimeSeries` would also download this
# field automatically on demand, but downloading it up front saves API calls.
#
# !!! note "Geopotential"
#     If the geopotential field isn't available, the fallback is to
#     estimate geopotential heights from the international standard atmosphere.

variables = [:specific_cloud_liquid_water_content,
             :specific_rain_water_content,
             :geopotential]
@suppress_out download(variables, pressure_level_dataset, dates; region = rico_column)
nothing #hide

# We load the downloaded data fully into memory and convert from kg/kg to g/kg.
# Each column `FieldTimeSeries` lives on a `(Flat, Flat, Bounded)` grid, so it
# plots directly as a time–height (Hovmöller) diagram.

cloud_set = MetadataSet(:specific_cloud_liquid_water_content,
                        :specific_rain_water_content;
                        dataset = pressure_level_dataset, dates, region = rico_column)
cloud_ts = NamedTuple(name => FieldTimeSeries(cloud_set[name]; time_indices_in_memory = Nt)
                      for name in cloud_set.names)
qᶜˡ_ts = cloud_ts.specific_cloud_liquid_water_content
qʳ_ts  = cloud_ts.specific_rain_water_content

interior(qᶜˡ_ts) .*= 1000
interior(qʳ_ts)  .*= 1000
nothing #hide

# We render the Hovmöller diagrams with `heatmap`, using the same x-ticks as `fig2`.

fig3 = Figure(size=(900, 600))

ax_qc = Axis(fig3[1, 1],
             title  = "Specific cloud liquid water content at $(rico_column.longitude)°E, $(rico_column.latitude)°N",
             ylabel = "Height [m]",
             xticks = (day_ticks, day_labels))
ax_qr = Axis(fig3[2, 1],
             title  = "Specific rain water content at $(rico_column.longitude)°E, $(rico_column.latitude)°N",
             ylabel = "Height [m]",
             xticks = (day_ticks, day_labels))

hm_qc = heatmap!(ax_qc, qᶜˡ_ts; colormap=:Blues)
hm_qr = heatmap!(ax_qr, qʳ_ts;  colormap=:Blues)

Colorbar(fig3[1, 2], hm_qc, label="qᶜˡ [g kg⁻¹]")
Colorbar(fig3[2, 2], hm_qr, label="qʳ [g kg⁻¹]")

linkaxes!(ax_qc, ax_qr)
ylims!(ax_qc, 0, 4000)
hidexdecorations!(ax_qc, grid=false)

fig3

# The cloud and rain water tell the same story as the precipitation in `fig2`.

# ### Profiles at the RICO location
#
# We reuse the filtered pressure-level dataset and the RICO `BoundingBox`
# defined above. As before, we bundle the API requests to speed up the data
# retrieval.

variables = [:temperature, :specific_humidity,
             :eastward_velocity, :northward_velocity,
             :geopotential]
download(variables, pressure_level_dataset, dates; region = rico_region)

rico_set = MetadataSet(:temperature, :specific_humidity,
                       :eastward_velocity, :northward_velocity;
                       dataset = pressure_level_dataset, dates, region = rico_region)

rico_ts = @suppress_out NamedTuple(name => FieldTimeSeries(rico_set[name]) for name in rico_set.names)
T_ts = rico_ts.temperature
q_ts = rico_ts.specific_humidity
u_ts = rico_ts.eastward_velocity
v_ts = rico_ts.northward_velocity
nothing #hide

# We compute the horizontal-mean profiles at every snapshot and convert
# temperature to potential temperature.

z  = znodes(T_ts.grid, nothing, nothing, Center())
Nz = length(z)
p  = sort(selected_levels, rev=true) / hPa # from bottom to top

## ERA5 pressure-level fields are NaN-filled below the local surface, so we
## skip NaNs when averaging horizontally.
function horizontal_mean_profiles(series)
    profiles = zeros(Nz, Nt)
    for n in 1:Nt
        slab = interior(series[n], :, :, :)
        for k in 1:Nz
            column = filter(!isnan, @view slab[:, :, k])
            profiles[k, n] = isempty(column) ? NaN : mean(column)
        end
    end
    return profiles
end

T̄ = horizontal_mean_profiles(T_ts)
q̄ = horizontal_mean_profiles(q_ts) * 1000 # kg/kg → g/kg
ū = horizontal_mean_profiles(u_ts)
v̄ = horizontal_mean_profiles(v_ts)

## θ = T (p₀ / p)^κ, with κ = Rᵈ / cᵖᵈ
p₀ = 1000 # hPa
κ  = 0.286
θ̄  = @. T̄ * (p₀ / p)^κ
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

for (ax, profiles) in [(ax_θ, θ̄), (ax_q, q̄), (ax_u, ū), (ax_v, v̄)]
    rows = [filter(!isnan, r) for r in eachrow(profiles)]
    μ  = [isempty(r) ? NaN : mean(r)           for r in rows]
    lo = [isempty(r) ? NaN : quantile(r, 0.25) for r in rows]
    hi = [isempty(r) ? NaN : quantile(r, 0.75) for r in rows]
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

# ## §4 River runoff from GloFAS
#
# ERA5's own surface runoff is generated locally over the whole land surface and
# is *not* routed downstream — interpolating it onto an ocean grid and masking
# land would discard most of the water. The [Global Flood Awareness System
# (GloFAS)](https://www.globalfloods.eu/) solves this: it forces the LISFLOOD
# hydrological and channel-routing model with ERA5 runoff to produce **river
# discharge already accumulated to river mouths** [harrigan2020glofas](@citep) —
# the ERA5-consistent analog of JRA55's pre-routed river freshwater flux.
#
# GloFAS lives on the Copernicus Early Warning Data Store (EWDS), a separate
# endpoint from the ERA5 CDS. The download automatically targets the EWDS url
# (https://ewds.climate.copernicus.eu/api) while reusing the ECMWF token from
# `~/.cdsapirc` — the same token works across both data stores — so the ERA5
# sections above and this GloFAS section run in one session without editing
# `~/.cdsapirc`. You still need to accept the `cems-glofas-historical` license
# once on the dataset page (see <https://ewds.climate.copernicus.eu/how-to-api>).
#
# We focus on the mouth of the Amazon, the largest freshwater source to the
# global ocean.

glofas = GloFASReanalysis()
amazon_region = BoundingBox(latitude = (-2, 5), longitude = (-53, -45))
nothing #hide

# GloFAS river discharge is a daily volume flux (m³ s⁻¹) on a 0.05° grid, with
# ocean cells left undefined (`NaN`). We download a single day over the region
# and load it on its native grid.

discharge_date = DateTime(2004, 12, 27)
discharge_meta = Metadatum(:river_discharge; dataset = glofas,
                           date = discharge_date, region = amazon_region)
discharge = @suppress_out Field(discharge_meta)
nothing #hide

# Plotting the discharge on a log scale reveals the routed river network feeding
# the coast — the discharge concentrates into channels that grow downstream and
# terminate at the river mouths.

λd, φd, _ = nodes(discharge)

## Mask non-positive values so `log10` is well defined for the heatmap.
discharge_data = interior(discharge, :, :, 1)
log_discharge = map(q -> (isnan(q) || q ≤ 0) ? NaN : log10(q), discharge_data)

fig5 = Figure(size=(800, 600))
ax5 = Axis(fig5[1, 1]; title = "GloFAS river discharge — $(Dates.format(discharge_date, "yyyy-mm-dd"))",
           xlabel = "Longitude (°)", ylabel = "Latitude (°)")
hm5 = heatmap!(ax5, λd, φd, log_discharge; colormap = :viridis)
Colorbar(fig5[1, 2], hm5; label = "log₁₀ discharge [m³ s⁻¹]")

fig5

# ### Routing discharge onto the ocean coastline
#
# To force an ocean model we need the discharge on the *ocean* grid, located on
# wet coastal cells. We build a regional ocean grid from ETOPO bathymetry over
# the same region. The vertical grid needs enough near-surface resolution that
# shallow shelf cells stay *active* — a single thick level would place every
# cell center below the coastal seafloor, leaving no wet cells for the routing.

ocean_grid = LatitudeLongitudeGrid(size = (80, 70, 30),
                                   longitude = amazon_region.longitude,
                                   latitude  = amazon_region.latitude,
                                   z = (-200, 0))

bottom_height = regrid_bathymetry(ocean_grid; minimum_depth = 5)
ocean_grid = ImmersedBoundaryGrid(ocean_grid, GridFittedBottom(bottom_height))
nothing #hide

# `GloFASPrescribedLand` downloads the discharge, locates the river mouths from
# the land/ocean boundary, and spreads each mouth over the active ocean cells of
# `ocean_grid` around it — conserving the total volume of freshwater (see
# [`build_river_routing`](@ref)).

land = @suppress_out GloFASPrescribedLand(ocean_grid; dataset = glofas,
                                          start_date = discharge_date,
                                          end_date = discharge_date + Day(2),
                                          region = amazon_region,
                                          maximum_search_radius = 8)
nothing #hide

# The resulting `RiverRouting` records, for each river mouth, the coastal ocean
# cells that receive its discharge. We overlay the mouths (on the GloFAS network)
# and their destination ocean cells (on the coastline) on the discharge map to
# visualize the relocation.

routing = land.river_routing.rivers

λn = λnodes(land.grid, Center(), Center(), Center())
φn = φnodes(land.grid, Center(), Center(), Center())
mouth_λ = [λn[i] for i in Array(routing.contribution_outlet_i)]
mouth_φ = [φn[j] for j in Array(routing.contribution_outlet_j)]

λo = λnodes(ocean_grid, Center(), Center(), Center())
φo = φnodes(ocean_grid, Center(), Center(), Center())
target_λ = [λo[i] for i in Array(routing.target_i)]
target_φ = [φo[j] for j in Array(routing.target_j)]

fig6 = Figure(size=(800, 600))
ax6 = Axis(fig6[1, 1]; title = "GloFAS river mouths routed to the ocean coastline",
           xlabel = "Longitude (°)", ylabel = "Latitude (°)")
hm6 = heatmap!(ax6, λd, φd, log_discharge; colormap = :grays)
scatter!(ax6, mouth_λ, mouth_φ; color = :dodgerblue, markersize = 4, label = "river mouths")
scatter!(ax6, target_λ, target_φ; color = :crimson, marker = :xcross,
         markersize = 8, label = "ocean injection cells")
Colorbar(fig6[1, 2], hm6; label = "log₁₀ discharge [m³ s⁻¹]")
axislegend(ax6; position = :rb)

fig6

# The red crosses mark where freshwater enters the ocean — on the coastline,
# regardless of the (coarser) ocean grid resolution. Passing `land` to a coupled
# ocean simulation injects this discharge as a conservative surface freshwater
# flux, lowering coastal sea-surface salinity near the Amazon plume.
