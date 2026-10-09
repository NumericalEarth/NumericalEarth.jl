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

soil = SoilEnergyWaterCarbon(NF; hydrology = SoilHydrology(NF, RichardsEq()))
land = NumericalEarth.land_simulation(land_grid;
                                      soil,
                                      vegetation   = nothing,
                                      initializers = (temperature          = 25.0,   # °C, warm tropical soil
                                                      saturation_water_ice = 0.6))   # moist

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

# ## Diagnostics
#
# Record the domain skin-temperature statistics and the mean ground heat flux each simulated hour.

t_hours   = Float64[]
T_mean    = Float64[]
T_min     = Float64[]
T_max     = Float64[]
G_mean    = Float64[]
wall_time = Ref(time_ns())

function record!(sim)
    state = sim.model.land.model.state
    Tsurf = Array(interior(state.skin_temperature))     # °C
    G     = Array(interior(state.ground_heat_flux))     # W m⁻², positive upward
    push!(t_hours, sim.model.clock.time / hour)
    push!(T_mean, mean(Tsurf))
    push!(T_min, minimum(Tsurf))
    push!(T_max, maximum(Tsurf))
    push!(G_mean, mean(G))
    elapsed = 1e-9 * (time_ns() - wall_time[])
    wall_time[] = time_ns()
    @info @sprintf("t = %6.1f h   ⟨Tₛ⟩ %.2f °C  (%.2f–%.2f)   ⟨G⟩ %+6.1f W m⁻²   wall Δ %.1fs",
                   sim.model.clock.time / hour, mean(Tsurf), minimum(Tsurf), maximum(Tsurf), mean(G), elapsed)
    return nothing
end

add_callback!(simulation, record!, TimeInterval(1hour))

# ## Run

@info "Running ERA5-forced Terrarium land over Central Borneo (~5 days)..."
run!(simulation)
@info "Simulation complete."

# ## Visualization
#
# Left: domain skin-temperature envelope over time. Right: final skin temperature over the box.

λ, φ, _ = Oceananigans.nodes(land_grid, Center(), Center(), Center())
Tsurf_f = Array(interior(land.model.state.skin_temperature))[:, :, 1]

let fig = Makie.Figure(size = (1400, 600), fontsize = 16)
    ax_t = Makie.Axis(fig[1, 1]; title = "Domain skin temperature", xlabel = "t (hours)", ylabel = "Tₛ (°C)")
    Makie.band!(ax_t, t_hours, T_min, T_max; color = (:orange, 0.25))
    Makie.lines!(ax_t, t_hours, T_mean; color = :firebrick, label = "mean")
    Makie.lines!(ax_t, t_hours, T_min;  color = :steelblue, linestyle = :dash, label = "min")
    Makie.lines!(ax_t, t_hours, T_max;  color = :orangered, linestyle = :dash, label = "max")
    Makie.axislegend(ax_t; position = :rb)

    ax_m = Makie.Axis(fig[1, 2]; title = "Final skin temperature", xlabel = "longitude", ylabel = "latitude",
                      aspect = Makie.DataAspect())
    hm = Makie.heatmap!(ax_m, λ, φ, Tsurf_f; colormap = :thermal)
    Makie.Colorbar(fig[1, 3], hm; label = "Tₛ (°C)")

    Makie.Label(fig[0, 1:3], "ERA5-forced Terrarium land, Central Borneo")

    Makie.save("era5_forced_terrarium_land.png", fig)
    @info "Saved era5_forced_terrarium_land.png"
    fig
end

nothing #hide

# ![](era5_forced_terrarium_land.png)
