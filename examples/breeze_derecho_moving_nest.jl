# # A moving nest following the 11 August 2026 Midwest derecho (ERA5 → 3 km Breeze)
#
# On the morning of 11 August 2026 a forward-propagating storm complex crossed Iowa, northern
# Illinois and northwest Indiana, with measured gusts of 85–99 mph from Sugar Grove to Gary and
# damage-survey estimates above 100 mph in northern Lake and Porter counties. The complex moved at
# roughly 100 km/h, so a fixed convection-permitting nest around it would have to span the whole
# Corn Belt. Here the nest moves instead: a 3 km Breeze child a few hundred kilometers across is
# relocated along the derecho's track as the simulation runs, with ERA5 driving its lateral
# boundaries wherever it happens to be.
#
# `nested_atmosphere_model(grid, dataset; dates, mover, …)` builds the moving nest. The parent is one
# `PrescribedAtmosphere` of ERA5 over the whole corridor the nest will visit, streamed three hourly
# levels at a time, and the `mover` is a `NestMover`: on its schedule, it reads the tracked position,
# rounds the offset from the nest's center to a whole number of cells and, if the nest has drifted, calls
# `translate_domain!`. That rebuilds the child on the shifted grid (terrain re-materialized from
# ETOPO2022, boundary conditions and relaxation rebuilt on the same ERA5-derived state), copies the
# state over the cells the two grids share, fills the exposed strip from ERA5, and hands the simulation
# the new child under the old clock.
#
# ## What this example does
# - Downloads ERA5 pressure levels over the derecho corridor and regrids ETOPO2022 terrain onto a
#   terrain-following 3 km `LatitudeLongitudeGrid` child.
# - Prescribes the nest track from the NWS Chicago damage-report timeline (Polo → Sugar Grove →
#   Gary), extended upstream into Iowa and downstream into Michigan by hand.
# - Relocates the child every five minutes of simulated time, by at most a few cells, so each
#   exposed strip lies inside the boundary relaxation zone.
# - Writes near-surface wind speed, the cold pool (near-surface virtual potential temperature) and
#   rain water at 2 km with the nest's current position, and animates them on the corridor map.
#
# ## What it does NOT do (yet)
# - No surface coupling: the child feels the ground through bulk drag only (no sensible or latent
#   heat fluxes, no radiation). The surface temperature entering the drag is one static mean value.
# - The track is prescribed; nothing in the child diagnoses where the gust front actually is.
# - Output writers hold the nest's state by value at every save, so no `FieldTimeSeries` is written:
#   the slices come back as arrays with the nest origin saved alongside.

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Breeze
using CopernicusClimateDataStore          # activates NumericalEarthCopernicusClimateDataStoreExt (ERA5 downloads)
using CloudMicrophysics                   # nested_atmosphere_model's default microphysics → 1-moment mixed-phase
using CairoMakie                          # loads Makie → NumericalEarthMakieExt (`visualize_nested_domain`)
using NaturalEarth, GeoInterface          # together → NumericalEarthNaturalEarthExt (`natural_earth_lines`)
using CUDA
using JLD2
using Printf
using Dates: DateTime, Second

# The 3 km nest (225×135×50 ≈ 1.5M cells, split-explicit) targets a CUDA GPU; switch to `CPU()` only
# for a small smoke test.
arch = GPU()

Oceananigans.defaults.FloatType = Float32

# ## The track
#
# Times are seconds after the start of the run. The three interior waypoints are the NWS Chicago
# damage reports: the first gust at Polo, Illinois (75 mph, 9:21 am CDT), Sugar Grove (85 mph ASOS,
# 10:11 am) and Gary/Chicago International Airport (99 mph, 11:09 am). The complex covered the 180 km
# from Polo to Gary in under two hours. The Iowa waypoints before dawn and the Michigan waypoint after
# are hand extrapolations at that speed along the line's heading, to be replaced by radar-derived
# positions where available.

name = "derecho"
start_date = DateTime(2026, 8, 11, 6)        # 01:00 CDT, the complex over central Iowa
duration = 12hours
stop_date = start_date + Second(duration)
dates = (start_date, stop_date)

seconds_after_start(date) = Second(date - start_date).value

waypoints = [DateTime(2026, 8, 11,  6, 00) => (-94.5, 42.3),   # central Iowa (extrapolated)
             DateTime(2026, 8, 11,  9, 00) => (-92.8, 42.1),   # eastern Iowa (extrapolated)
             DateTime(2026, 8, 11, 12, 00) => (-91.3, 41.9),   # Quad Cities (extrapolated)
             DateTime(2026, 8, 11, 14, 21) => (-89.58, 41.99), # Polo, IL: first 75 mph gust
             DateTime(2026, 8, 11, 15, 11) => (-88.44, 41.76), # Sugar Grove, IL: 85 mph ASOS
             DateTime(2026, 8, 11, 16, 09) => (-87.41, 41.62), # Gary/Chicago airport: 99 mph
             DateTime(2026, 8, 11, 18, 00) => (-85.9, 41.5)]   # southwest Michigan (extrapolated)

track = PrescribedTrack(seconds_after_start.(first.(waypoints)),
                        first.(last.(waypoints)),
                        last.(last.(waypoints)))

# ## The nest and its corridor
#
# The child is 7.5° × 4.5° (≈ 620 × 500 km) at 1/30° (≈ 2.8 km in latitude, 3.7 km in
# longitude), centered on the track's first waypoint. The parent box is the whole corridor the child
# will visit, with margin for the interpolation stencils and relocation rounding.

Δλ = Δφ = 1/30
Lλ, Lφ = 7.5, 4.5
Nx = round(Int, Lλ / Δλ)
Ny = round(Int, Lφ / Δφ)

λ₀, φ₀ = track(nothing, 0)
λ_west, λ_east   = λ₀ - Lλ / 2, λ₀ + Lλ / 2
φ_south, φ_north = φ₀ - Lφ / 2, φ₀ + Lφ / 2

corridor = BoundingBox(longitude = (-100, -81), latitude = (37, 47))

z = ReferenceToStretchedDiscretization(extent = 19525.0,
                                       bias = :left,
                                       bias_edge = 0,
                                       constant_spacing = 60,
                                       constant_spacing_extent = 60,
                                       maximum_spacing = 490,
                                       stretching = LinearStretching(0.15))
Nz = length(z)

grid = LatitudeLongitudeGrid(arch;
                             longitude = (λ_west,  λ_east),
                             latitude  = (φ_south, φ_north),
                             z = TerrainFollowingVerticalDiscretization(z),
                             size = (Nx, Ny, Nz),
                             halo = (5, 5, 5),
                             topology = (Bounded, Bounded, Bounded))

# ## Build the moving nest
#
# The parent holds ERA5 over the corridor with three hourly levels resident, enough for the
# exchanger's window. The mover checks the track every five minutes; at the derecho's ≈ 1.2° h⁻¹ that
# is at most three cells per relocation, inside the five-cell relaxation zone, so the strip the
# relocation exposes is always one the parent already relaxes toward. The surface temperature in the
# bulk drag is the corridor's mean skin temperature at the start, since a field on the child grid
# would not survive a relocation.

era5_datadir = "era5"
dataset = ERA5HourlyPressureLevels()
relax_width = 5

parent = PrescribedAtmosphere(corridor, dates, dataset; dir = era5_datadir,
                              architecture = arch, time_indices_in_memory = 3)

mover = NestMover(track; schedule = TimeInterval(5minutes), tolerance = 1)

nest = nested_atmosphere_model(grid, dataset;
                               dates,
                               dir = era5_datadir,
                               parent_atmosphere = parent,
                               mover,
                               terrain = ETOPO2022(),
                               terrain_blend_width = relax_width,
                               relaxation_rate = 1/300,
                               relaxation_width = relax_width,
                               bottom_drag_coefficient = 2e-3,
                               momentum_advection = WENO(order = 5))

# ## The corridor map
#
# The nest at its starting position inside the ERA5 corridor, with the track it will follow.

fig = visualize_nested_domain(grid;
                              parent = corridor,
                              padding = 1,
                              title = "ERA5 → 3 km moving nest, 11 August 2026 derecho",
                              label = "3 km nest at 06 UTC",
                              parent_label = "ERA5 corridor",
                              landmarks = tuple("Gary" => (-87.41, 41.62)))

lines!(fig.content[1], track.longitude, track.latitude; color = :red, linewidth = 2, label = "track")
save("derecho_domains.png", fig)

fig

# ## Simulation
#
# The adaptive outer Δt is bounded by the advective CFL (acoustic modes are substepped).

Δt = 10
simulation = Simulation(nest; Δt, stop_time = duration)
conjure_time_step_wizard!(simulation, IterationInterval(3); cfl = 0.5, max_Δt = Δt)

# ## Output
#
# Relocation replaces the child, so a writer cannot hold on to the child's fields: each output is a
# function of the model that reads the *current* child at save time. The near-surface slices show
# the gust front (wind speed) and the cold pool (virtual potential temperature); rain water at 2 km
# shows the convective line itself. The nest's origin goes into the file with every frame.

k_aloft = searchsortedfirst(Array(znodes(grid, Center())), 2000)

slice(operation, k) = Array(interior(compute!(Field(operation; indices = (:, :, k)))))[:, :, 1]

outputs = (U  = model -> slice(sqrt(model.child.velocities.u^2 + model.child.velocities.v^2), 1),
           θᵛ = model -> slice(VirtualPotentialTemperature(model.child), 1),
           qʳ = model -> slice(model.child.microphysical_fields.qʳ, k_aloft),
           λ₀ = model -> Oceananigans.Grids.x_domain(model.child.grid)[1],
           φ₀ = model -> Oceananigans.Grids.y_domain(model.child.grid)[1])

filename = name * "_slices.jld2"
simulation.output_writers[:slices] = JLD2Writer(nest, outputs; schedule = TimeInterval(10minutes),
                                                filename, overwrite_files = true)

function progress(sim)
    child = sim.model.child
    u, v, w = child.velocities
    ρ  = child.dynamics.total_density
    λ₀, φ₀ = Oceananigans.Grids.x_domain(child.grid)[1], Oceananigans.Grids.y_domain(child.grid)[1]
    @info @sprintf("iter=%4d, t=%s, Δt=%s, nest origin=(%.2f°, %.2f°), max|u|=(%6.1f, %6.1f, %5.1f), ρ ∈ [%.3f, %.3f]",
                   sim.model.clock.iteration, prettytime(sim), prettytime(sim.Δt), λ₀, φ₀,
                   maximum(abs, u), maximum(abs, v), maximum(abs, w), minimum(ρ), maximum(ρ))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(100))

# ## Run

run!(simulation)

# ## Animation
#
# Three panels on the corridor map: near-surface wind speed, near-surface θᵛ and rain water at 2 km,
# each drawn at the nest's position in that frame, with the state borders and the prescribed track
# for orientation. The frames are the saved arrays; their coordinates follow from the saved origin and
# the (fixed) grid spacing.

file = jldopen(filename)
iterations = parse.(Int, keys(file["timeseries/t"]))
times = [file["timeseries/t/$iteration"] for iteration in iterations]
frame(name, iteration) = file["timeseries/$name/$iteration"]

λ_nodes(λ₀) = λ₀ .+ Δλ .* ((1:Nx) .- 1/2)
φ_nodes(φ₀) = φ₀ .+ Δφ .* ((1:Ny) .- 1/2)

borders = natural_earth_lines("admin_1_states_provinces_lines")

n = Observable(1)
λ = @lift λ_nodes(frame("λ₀", iterations[$n]))
φ = @lift φ_nodes(frame("φ₀", iterations[$n]))
U  = @lift frame("U",  iterations[$n])
θᵛ = @lift frame("θᵛ", iterations[$n])
qʳ = @lift 1e3 * frame("qʳ", iterations[$n])

fig = Figure(size = (1500, 500))
Label(fig[0, 1:6], @lift(@sprintf("11 August 2026 derecho — ERA5 → 3 km moving nest — %s UTC",
                                  start_date + Second(round(Int, times[$n])))),
      fontsize = 20, tellwidth = false)

panels = (("|U| at the surface (m s⁻¹)", U,  :speed,   (0, 35)),
          ("θᵛ at the surface (K)",      θᵛ, :thermal, (295, 315)),
          ("qʳ at 2 km (g kg⁻¹)",        qʳ, :dense,   (0, 3)))

for (i, (title, data, colormap, colorrange)) in enumerate(panels)
    ax = Axis(fig[1, 2i - 1]; title, xlabel = "longitude (°)", ylabel = i == 1 ? "latitude (°)" : "",
              limits = (corridor.longitude..., corridor.latitude...), aspect = DataAspect())
    hm = heatmap!(ax, λ, φ, data; colormap, colorrange)
    lines!(ax, borders...; color = :gray40, linewidth = 0.8)
    lines!(ax, track.longitude, track.latitude; color = :red, linewidth = 1.5, linestyle = :dash)
    Colorbar(fig[1, 2i], hm)
end

CairoMakie.record(fig, "derecho_moving_nest.mp4", eachindex(iterations); framerate = 8) do nn
    n[] = nn   ## `record` is also exported by CUDA, so qualify it
end
close(file)
nothing #hide

# ![](derecho_moving_nest.mp4)
