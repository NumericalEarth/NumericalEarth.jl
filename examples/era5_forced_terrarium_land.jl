# # ERA5-forced Terrarium land over Central Borneo
#
# This example couples a process-based [Terrarium](https://github.com/CliMA/Terrarium.jl)
# `LandModel` — solving the coupled heat and Richards equations in the soil column — into
# NumericalEarth's `EarthSystemModel` as the land component, forced by ERA5 reanalysis.
# The surface turbulent fluxes are computed by NumericalEarth's Monin–Obukhov similarity
# theory and consumed by Terrarium; Terrarium diagnoses radiation locally and owns the
# subsurface water budget (infiltration, runoff, evapotranspiration).
#
# Terrarium is column-based: its land grid is a set of laterally independent, georeferenced
# vertical columns (`ColumnRingGrid`, whose lateral layout is a `RingGrids` grid). Because
# the ERA5 forcing is *prescribed* (data, not a prognostic atmosphere), we interpolate it
# onto the columns **up-front** — physically identical to regridding it every step — and
# build a column-grid `PrescribedAtmosphere` that shares the land exchange grid, so the
# coupler's air–land exchange is a direct 1:1 mapping.
#
# We run a short (~5 day) forward simulation over a snow-free equatorial box on the CPU.
#
# ## CDS API credentials
#
# Downloading ERA5 fields requires CDS API credentials at `~/.cdsapirc`;
# see <https://cds.climate.copernicus.eu/how-to-api>.

# ## Load packages

using NumericalEarth
using Terrarium

using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Fields: interpolate, interior

using CDSAPI                              # activates the CDS-API extension
using Printf
using Statistics                         # mean

import CairoMakie: Makie
import Terrarium.RingGrids # would be better to add as explicit dependency?
import Dates: DateTime, Hour

arch = Oceananigans.CPU()
NF   = Float64                           # match the EarthSystemModel clock precision

# ## Domain: 2° × 2° Central Borneo box
#
# The region is equatorial (snow-free), heavy-rainfall, and fully inland — a clean case for
# a soil-column land model with no snow or sea-ice coupling.

latitude  = lat_min, lat_max = 0.5, 2.5
longitude = lon_min, lon_max = 113.0, 115.0
region    = BoundingBox(; latitude, longitude)

# ## Terrarium land grid: georeferenced columns over the box
#
# `ColumnRingGrid` places its columns on a global `RingGrids` grid; we take a fine global
# Gaussian grid and **mask** it to the Borneo box, so only the columns inside the box are
# simulated (`Nh` active columns). The soil column uses an exponentially stretched 12-layer
# discretization (finer near the surface).

rings        = RingGrids.FullGaussianGrid(256)          # ~0.35° globally
londs, latds = RingGrids.get_londlatds(rings)
inbox        = (lon_min .<= londs .<= lon_max) .& (lat_min .<= latds .<= lat_max)
mask         = convert.(Bool, RingGrids.Field(rings)); mask .= inbox

# A 10-layer soil column with a 10 cm surface layer. The surface layer thickness sets the
# explicit stability limit on the coupled time step (Δt ≲ Δz²/2κ); 10 cm keeps a 5-minute step
# comfortably stable.
vertical  = ExponentialSpacing(Δz_min = 0.1, Δz_max = 1.0, N = 10)
land_grid = ColumnRingGrid(arch, NF, vertical, mask)

## Per-column geographic coordinates (used to sample ERA5 at each column).
col_lon = londs[land_grid.mask.data]
col_lat = latds[land_grid.mask.data]
Nh      = length(col_lon)
@info "Terrarium land grid: $Nh columns in the Borneo box"

# ## Terrarium land model (deferred surface energy balance, Richards hydrology)
#
# `land_simulation` builds the `LandModel` in the deferred-flux configuration (NumericalEarth
# computes the turbulent fluxes; Terrarium closes the ground heat flux as a residual and owns
# the water budget), initializes it, and returns an Oceananigans `Simulation`. We use the
# variably saturated Richards equation for the soil water.

soil = SoilEnergyWaterCarbon(NF; hydrology = SoilHydrology(NF, RichardsEq()))
land = NumericalEarth.land_simulation(land_grid;
                                      soil,
                                      vegetation  = nothing,
                                      initializers = (temperature           = (x, z) -> 25.0,   # °C, warm tropical soil
                                                      saturation_water_ice  = (x, z) -> 0.6))   # moist

# ## ERA5 forcing over the box
#
# `ERA5PrescribedAtmosphere` / `ERA5PrescribedRadiation` download the required ERA5 single-level
# fields over the region (10 m wind, 2 m temperature, dewpoint → specific humidity, surface
# pressure, total precipitation, downwelling shortwave/longwave) and convert accumulated
# radiation/precipitation to fluxes. They live on ERA5's native `LatitudeLongitudeGrid`.

dataset    = ERA5HourlySingleLevel()
start_date = DateTime(2020, 4, 1)
end_date   = DateTime(2020, 4, 5, 23)

era5_atmos = ERA5PrescribedAtmosphere(arch; dataset, start_date, end_date, region,
                                      surface_layer_height = 10, boundary_layer_height = 800)
era5_rad   = ERA5PrescribedRadiation(arch; dataset, start_date, end_date, region,
                                     land_surface = SurfaceRadiationProperties(0.18, 0.95))
times = era5_atmos.velocities.u.times
Nt    = length(times)

# ## Interpolate ERA5 onto the Terrarium columns
#
# We build a `PrescribedAtmosphere` / `PrescribedRadiation` **on the Terrarium exchange grid**
# (`land.model.grid`, the flattened `(Nh, 1, Nz)` column field grid) so `atmosphere.grid ===
# exchange_grid` and the coupler's air–land regridding is the identity. Each time slice is
# filled by sampling the ERA5 lat-lon field at every column's (lon, lat) with Oceananigans'
# native point interpolation — orientation-safe for a regional grid. The atmosphere fields
# inherit the exchange grid's vertical levels, so we broadcast each column value across them.

exchange_grid = land.model.grid

## The exchange grid is the 3-D column field grid (`Nz>1`), for which `PrescribedAtmosphere`
## omits precipitation by default; supply a 2-D rain/snow flux explicitly (`Center,Center,Nothing`
## fields are surface fields even on a 3-D grid).
rain = Oceananigans.FieldTimeSeries{Oceananigans.Center, Oceananigans.Center, Nothing}(exchange_grid, times)
snow = Oceananigans.FieldTimeSeries{Oceananigans.Center, Oceananigans.Center, Nothing}(exchange_grid, times)
precipitation_flux = NumericalEarth.PrescribedPrecipitationFlux(rain, snow)

atmosphere = NumericalEarth.PrescribedAtmosphere(exchange_grid, times; precipitation_flux,
                                                 surface_layer_height = 10, boundary_layer_height = 800)
radiation  = NumericalEarth.PrescribedRadiation(exchange_grid, times; land_surface = SurfaceRadiationProperties(0.18, 0.95),
                                 ocean_surface = nothing, sea_ice_surface = nothing)

## Sample a lat-lon ERA5 slice at the column locations and write it into `dst` (all z-levels).
function fill_columns!(dst, era5_slice, col_lon, col_lat)
    vals = [interpolate((col_lon[i], col_lat[i]), era5_slice) for i in eachindex(col_lon)]
    d = interior(dst)
    for k in axes(d, 3)
        d[:, 1, k] .= vals
    end
    return nothing
end

@info "Interpolating ERA5 onto $Nh columns for $Nt hourly slices..."
for n in 1:Nt
    fill_columns!(atmosphere.temperature[n],             era5_atmos.temperature[n],             col_lon, col_lat)
    fill_columns!(atmosphere.specific_humidity[n],       era5_atmos.specific_humidity[n],       col_lon, col_lat)
    fill_columns!(atmosphere.velocities.u[n],            era5_atmos.velocities.u[n],            col_lon, col_lat)
    fill_columns!(atmosphere.velocities.v[n],            era5_atmos.velocities.v[n],            col_lon, col_lat)
    fill_columns!(atmosphere.pressure[n],                era5_atmos.pressure[n],                col_lon, col_lat)
    fill_columns!(atmosphere.precipitation_flux.rain[n], era5_atmos.precipitation_flux.rain[n], col_lon, col_lat)
    fill_columns!(radiation.downwelling_shortwave[n],    era5_rad.downwelling_shortwave[n],     col_lon, col_lat)
    fill_columns!(radiation.downwelling_longwave[n],     era5_rad.downwelling_longwave[n],      col_lon, col_lat)
end
update_state!(atmosphere)
update_state!(radiation)

# ## Coupled model
#
# `AtmosphereLandModel` wires the Terrarium land to the column atmosphere/radiation through
# NumericalEarth's `InterfaceComputations`. Each step: the exchanger publishes the land
# surface temperature and saturation, Monin–Obukhov computes the turbulent fluxes, and the
# coupler pushes skin temperature, turbulent fluxes, downwelling radiation, precipitation, and
# near-surface forcing into Terrarium, which closes the ground heat flux and steps the soil.

model      = AtmosphereLandModel(atmosphere, land; radiation)
simulation = Oceananigans.Simulation(model; Δt = 5minutes, stop_time = (Nt - 1) * 3600.0)

# ## Diagnostics
#
# Record the domain skin-temperature statistics and mean ground heat flux each simulated hour.

t_hours   = Float64[]
T_mean    = Float64[]
T_min     = Float64[]
T_max     = Float64[]
G_mean    = Float64[]
wall_time = Ref(time_ns())

function record!(sim)
    state = sim.model.land.model.state
    Tsurf = vec(Array(interior(state.skin_temperature)))     # °C, prescribed interface temperature
    G     = vec(Array(interior(state.ground_heat_flux)))     # W m⁻², positive upward
    push!(t_hours, sim.model.clock.time / 3600)
    push!(T_mean, mean(Tsurf)); push!(T_min, minimum(Tsurf)); push!(T_max, maximum(Tsurf))
    push!(G_mean, mean(G))
    elapsed = 1e-9 * (time_ns() - wall_time[]); wall_time[] = time_ns()
    @info @sprintf("t = %6.1f h   ⟨Tₛ⟩ %.2f °C  (%.2f–%.2f)   ⟨G⟩ %+6.1f W m⁻²   wall Δ %.1fs",
                   sim.model.clock.time / 3600, mean(Tsurf), minimum(Tsurf), maximum(Tsurf), mean(G), elapsed)
    return nothing
end
add_callback!(simulation, record!, TimeInterval(1hour))

# ## Run

@info "Running ERA5-forced Terrarium land over Central Borneo (~5 days)..."
run!(simulation)
@info "Simulation complete."

# ## Visualization
#
# Left: domain skin-temperature envelope over time. Right: final skin temperature scattered at
# the column locations over the box.

state    = land.model.state
Tsurf_f  = vec(Array(interior(state.skin_temperature)))

fig = Makie.Figure(size = (1400, 600), fontsize = 16)

ax_t = Makie.Axis(fig[1, 1]; title = "Domain skin temperature", xlabel = "t (hours)", ylabel = "Tₛ (°C)")
Makie.band!(ax_t, t_hours, T_min, T_max; color = (:orange, 0.25))
Makie.lines!(ax_t, t_hours, T_mean; color = :firebrick, label = "mean")
Makie.lines!(ax_t, t_hours, T_min;  color = :steelblue, linestyle = :dash, label = "min")
Makie.lines!(ax_t, t_hours, T_max;  color = :orangered, linestyle = :dash, label = "max")
Makie.axislegend(ax_t; position = :rb)

ax_m = Makie.Axis(fig[1, 2]; title = "Final skin temperature", xlabel = "longitude", ylabel = "latitude",
            aspect = DataAspect())
sc = Makie.scatter!(ax_m, col_lon, col_lat; color = Tsurf_f, colormap = :thermal, markersize = 18)
Makie.Colorbar(fig[1, 3], sc; label = "Tₛ (°C)")

Makie.Label(fig[0, 1:3], "ERA5-forced Terrarium land — Central Borneo, $Nh columns")

save("era5_forced_terrarium_land.png", fig)
@info "Saved era5_forced_terrarium_land.png"

nothing #hide

# ![](era5_forced_terrarium_land.png)
