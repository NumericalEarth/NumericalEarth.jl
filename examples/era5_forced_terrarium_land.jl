# # ERA5-forced Terrarium land over Central Borneo
#
# This example couples a process-based [Terrarium](https://github.com/NumericalEarth/Terrarium.jl)
# `LandModel`, solving the coupled heat and Richards equations in the soil column, into
# NumericalEarth's `EarthSystemModel` as the land component, forced by ERA5 reanalysis.
#
# NumericalEarth owns the surface energy balance: it computes the turbulent fluxes with
# Monin–Obukhov similarity theory, the radiative fluxes from the prescribed downwelling
# radiation, and assembles their sum into the ground heat flux. Terrarium receives that flux as
# the top boundary condition of the soil column and owns the subsurface water budget
# (infiltration, runoff, and evapotranspiration).
#
# The land runs on an ordinary Oceananigans `LatitudeLongitudeGrid`, so the ERA5 atmosphere
# and radiation are passed to the coupler directly on their native grid and regridded onto the
# land each step.
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

using CopernicusClimateDataStore         # activates the ERA5 download extension
using Printf
using Statistics

import CairoMakie: Makie
import Dates: DateTime

arch = Oceananigans.CPU()
NF   = Float64                           # match the EarthSystemModel clock precision

# ## Domain: 2° × 2° Central Borneo box
#
# The region is equatorial (snow-free), heavy-rainfall, and fully inland, which makes it a clean
# case for a soil-column land model with no snow or sea-ice coupling.

latitude  = lat_min, lat_max = 0.5, 2.5
longitude = lon_min, lon_max = 113.0, 115.0

# ERA5 is loaded over a slightly larger region so that every land cell lies strictly inside the
# ERA5 grid. Cells on the edge of the ERA5 grid would otherwise interpolate into its halo.

era5_pad    = 0.75
era5_region = BoundingBox(latitude  = (lat_min - era5_pad, lat_max + era5_pad),
                          longitude = (lon_min - era5_pad, lon_max + era5_pad))

# ## Terrarium land grid
#
# A 0.25° `LatitudeLongitudeGrid` over the box, matching the ERA5 resolution. Each horizontal
# cell holds a soil column with an exponentially stretched 10-layer discretization. The 10 cm
# surface layer sets the explicit stability limit on the time step. Heat conduction alone would
# allow several minutes, but the Richards equation is far stiffer: when the surface layer dries
# under strong midday heating, a 5-minute step lets the soil water oscillate between dry and
# saturated in alternating steps. We therefore use a 1-minute step.

vertical  = ExponentialSpacing(Δz_min = 0.1, Δz_max = 1.0, N = 10)
land_grid = LatitudeLongitudeGrid(arch, NF;
                                  size = (8, 8, num_layers(vertical)),
                                  longitude,
                                  latitude,
                                  z = vertical,
                                  topology = (Bounded, Bounded, Bounded))

# ## Terrarium land model
#
# `land_simulation` builds the `LandModel` with a fully prescribed surface energy balance,
# initializes it, and returns an Oceananigans `Simulation`. We use the variably saturated
# Richards equation for the soil water.
#
# `initializers` sets the initial state of the land model. Each entry names a Terrarium state
# variable and gives its initial value, either as a constant or as a function of position
# `(x, y, z)`. Terrarium works in degrees Celsius, so we start from a uniform soil column at
# 25 °C that is 60% saturated with water and ice.

soil = SoilEnergyWaterCarbon(NF; hydrology = SoilHydrology(NF, RichardsEq()))
land = NumericalEarth.land_simulation(land_grid;
                                      soil,
                                      vegetation   = nothing,
                                      initializers = (temperature          = 25.0,
                                                      saturation_water_ice = 0.6))

# ## ERA5 forcing
#
# `ERA5PrescribedAtmosphere` and `ERA5PrescribedRadiation` download the required ERA5
# single-level fields over the region (10 m wind, 2 m temperature, dewpoint, surface pressure,
# total precipitation, and downwelling shortwave and longwave radiation) and convert the
# accumulated radiation and precipitation to fluxes.

dataset    = ERA5HourlySingleLevel()
start_date = DateTime(2020, 4, 1)
end_date   = DateTime(2020, 4, 5, 23)

atmosphere = ERA5PrescribedAtmosphere(arch; dataset, start_date, end_date, region = era5_region,
                                      surface_layer_height = 10, boundary_layer_height = 800)
radiation  = ERA5PrescribedRadiation(arch; dataset, start_date, end_date, region = era5_region,
                                     land_surface = SurfaceRadiationProperties(0.18, 0.95))

Nt = length(atmosphere.velocities.u.times)

# ## Coupled model
#
# Each step, the exchanger publishes the uppermost soil temperature and saturation, the coupler
# computes the turbulent and radiative fluxes at the interface, and pushes the skin temperature,
# the individual fluxes, their sum as the ground heat flux, the precipitation, and the
# near-surface forcing into Terrarium, which then steps the soil.

model      = AtmosphereLandModel(atmosphere, land; radiation)
simulation = Oceananigans.Simulation(model; Δt = 1minute, stop_time = (Nt - 1) * hour)

# ## Output and progress
#
# A `JLD2Writer` saves the skin temperature and the ground heat flux every simulated hour, and a
# callback reports the domain skin-temperature range twice a day.

state    = land.model.state
outputs  = (; Tₛ = state.skin_temperature, G = state.ground_heat_flux)
filename = "era5_forced_terrarium_land"

simulation.output_writers[:land] = JLD2Writer(model, outputs;
                                              filename,
                                              schedule = TimeInterval(1hour),
                                              overwrite_files = true)

function progress(sim)
    Tₛ = state.skin_temperature
    @info @sprintf("t = %s, ⟨Tₛ⟩ = %.2f °C (%.2f to %.2f), wall time: %s",
                   prettytime(sim), mean(Tₛ), minimum(Tₛ), maximum(Tₛ), prettytime(sim.run_wall_time))
    return nothing
end

add_callback!(simulation, progress, TimeInterval(12hours))

# ## Run

@info "Running ERA5-forced Terrarium land over Central Borneo (~5 days)..."
run!(simulation)
@info "Simulation complete."

# ## Visualization
#
# Left: the domain skin-temperature envelope and the domain-mean ground heat flux (positive
# upward) over time. Right: the final skin temperature over the box.

Tₛ_ts = FieldTimeSeries("$filename.jld2", "Tₛ")
G_ts  = FieldTimeSeries("$filename.jld2", "G")

t_hours = Tₛ_ts.times ./ hour
Nₜ      = length(t_hours)
T_mean  = [mean(Tₛ_ts[n]) for n in 1:Nₜ]
T_min   = [minimum(Tₛ_ts[n]) for n in 1:Nₜ]
T_max   = [maximum(Tₛ_ts[n]) for n in 1:Nₜ]
G_mean  = [mean(G_ts[n]) for n in 1:Nₜ]

λ, φ, _ = Oceananigans.nodes(land_grid, Center(), Center(), Center())
Tₛ_final = interior(Tₛ_ts[Nₜ], :, :, 1)

let fig = Makie.Figure(size = (1400, 800), fontsize = 16)
    ax_T = Makie.Axis(fig[1, 1]; title = "Domain skin temperature", ylabel = "Tₛ (°C)")
    Makie.band!(ax_T, t_hours, T_min, T_max; color = (:orange, 0.25))
    Makie.lines!(ax_T, t_hours, T_mean; color = :firebrick, label = "mean")
    Makie.lines!(ax_T, t_hours, T_min;  color = :steelblue, linestyle = :dash, label = "min")
    Makie.lines!(ax_T, t_hours, T_max;  color = :orangered, linestyle = :dash, label = "max")
    Makie.axislegend(ax_T; position = :rb)

    ax_G = Makie.Axis(fig[2, 1]; title = "Domain-mean ground heat flux", xlabel = "t (hours)", ylabel = "G (W m⁻²)")
    Makie.lines!(ax_G, t_hours, G_mean; color = :black)
    Makie.linkxaxes!(ax_T, ax_G)

    ax_m = Makie.Axis(fig[1:2, 2]; title = "Final skin temperature", xlabel = "longitude", ylabel = "latitude",
                      aspect = Makie.DataAspect())
    hm = Makie.heatmap!(ax_m, λ, φ, Tₛ_final; colormap = :thermal)
    Makie.Colorbar(fig[1:2, 3], hm; label = "Tₛ (°C)")

    Makie.Label(fig[0, 1:3], "ERA5-forced Terrarium land, Central Borneo")

    Makie.save("era5_forced_terrarium_land.png", fig)
    fig
end

nothing #hide

# ![](era5_forced_terrarium_land.png)
