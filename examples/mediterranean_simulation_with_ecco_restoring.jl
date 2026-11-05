# # Mediterranean simulation with restoring to ECCO
#
# This example sets up and runs a high-resolution ocean simulation of the Mediterranean Sea
# with Oceananigans and NumericalEarth. Temperature and salinity are restored toward
# the ECCO (Estimating the Circulation and Climate of the Ocean) state estimate.
#
# ## Initial setup with package imports
#
# We begin by importing the Julia packages for visualization (CairoMakie), ocean modeling
# (Oceananigans, NumericalEarth), dates and times (Dates), and running on CUDA-enabled GPUs (CUDA).

using CairoMakie
using Oceananigans
using Oceananigans.Units
using NumericalEarth
using Printf
using Dates
using CUDA

# ## Grid configuration for the Mediterranean Sea
#
# We build a `LatitudeLongitudeGrid` that covers the Mediterranean Sea, with the domain bounded
# in longitude by (λ₁, λ₂) and in latitude by (φ₁, φ₂). The vertical grid has constant spacing
# near the surface and stretches with depth. The horizontal resolution is 1/15th of a degree,
# which corresponds to about 7 kilometers.

λ₁, λ₂ = ( 0, 42)
φ₁, φ₂ = (30, 45)

z = ReferenceToStretchedDiscretization(; extent = 5000,
                                       constant_spacing = 2.5,
                                       constant_spacing_extent = 50,
                                       stretching = PowerLawStretching(1.07))

Nx = 15 * Int(λ₂ - λ₁)
Ny = 15 * Int(φ₂ - φ₁)
Nz = length(z)

grid = LatitudeLongitudeGrid(GPU();
                             size = (Nx, Ny, Nz),
                             latitude  = (φ₁, φ₂),
                             longitude = (λ₁, λ₂),
                             z,
                             halo = (7, 7, 7))

# ### Bathymetry interpolation
#
# `regrid_bathymetry` interpolates bathymetric data onto the grid. The keyword arguments
# `minimum_depth` and `interpolation_passes` control how shallow regions are filled
# and how much the bathymetry is smoothed. We keep only the largest basin.

bottom_height = regrid_bathymetry(grid,
                                  height_above_water = 1,
                                  minimum_depth = 10,
                                  interpolation_passes = 25,
                                  major_basins = 1)

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))

# ## Restoring to ECCO
#
# `DatasetRestoring` builds forcing terms that nudge temperature and salinity toward
# the ECCO monthly fields with a timescale of 2 days.

dates = (DateTime(1993, 1, 1), DateTime(1993, 12, 1))

temperature_metadata = Metadata(:temperature; dataset = ECCO4Monthly(), dates)
salinity_metadata    = Metadata(:salinity;    dataset = ECCO4Monthly(), dates)

FT = DatasetRestoring(temperature_metadata, GPU(); rate = 1 / 2days)
FS = DatasetRestoring(salinity_metadata,    GPU(); rate = 1 / 2days)

# ## Constructing the simulation
#
# We construct an ocean simulation that evolves temperature and salinity,
# and pass it the restoring forcings.

ocean = ocean_simulation(grid; forcing = (T = FT, S = FS))

# ## Initializing the model
#
# We initialize temperature and salinity from ECCO at the start date and plot them at the surface.

set!(ocean.model, MetadataSet(:temperature, :salinity;
                              dataset = ECCO4Monthly(), date = first(dates)))

T, S = ocean.model.tracers

fig = Figure()
ax = Axis(fig[1, 1])
heatmap!(ax, view(T, :, :, Nz), colorrange = (10, 20), colormap = :thermal)
ax = Axis(fig[1, 2])
heatmap!(ax, view(S, :, :, Nz), colorrange = (35, 40), colormap = :haline)
fig

# We add a callback that prints a progress message while the simulation runs.

function progress(sim)
    u, v, w = sim.model.velocities
    T, S = sim.model.tracers

    @info @sprintf("Time: %s, iteration: %d, Δt: %s, max|u|: (%.2e, %.2e, %.2e) m s⁻¹, max(T, S): %.2f ᵒC, %.2f g kg⁻¹",
                   prettytime(sim), iteration(sim), prettytime(sim.Δt),
                   maximum(abs, u), maximum(abs, v), maximum(abs, w),
                   maximum(T), maximum(S))

    return nothing
end

ocean.callbacks[:progress] = Callback(progress, IterationInterval(10))

# ## Simulation warm-up
#
# We have regridded the ECCO solution from a coarse grid (half a degree) to a
# fine grid (1/15th of a degree), and small mismatches with the bathymetry
# might crash the simulation. We therefore warm up the simulation with a small
# time step for a few iterations, which lets the solution adjust to the new grid and
# bathymetry.

ocean.Δt = 10
ocean.stop_iteration = 1000
run!(ocean)

# ## Running the simulation
#
# Now that the solution has adjusted to the bathymetry, we can increase the time
# step. A `TimeStepWizard` adapts the time step to keep the CFL number at 0.2.

wizard = TimeStepWizard(; cfl = 0.2, max_Δt = 10minutes, max_change = 1.1)

ocean.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

# We remove the iteration limit and run for 200 days, saving the surface fields every day.

ocean.stop_iteration = Inf
ocean.stop_time = 200days

ocean.output_writers[:surface_fields] = JLD2Writer(ocean.model, merge(ocean.model.velocities, ocean.model.tracers);
                                                   indices = (:, :, Nz),
                                                   schedule = TimeInterval(1days),
                                                   overwrite_files = true,
                                                   including = [:grid],
                                                   filename = "med_surface_field")

run!(ocean)

# ## Recording a video
#
# We read the output and record a video of the Mediterranean Sea's surface
# zonal velocity, meridional velocity, temperature, and salinity.

u_ts = FieldTimeSeries("med_surface_field.jld2", "u")
v_ts = FieldTimeSeries("med_surface_field.jld2", "v")
T_ts = FieldTimeSeries("med_surface_field.jld2", "T")
S_ts = FieldTimeSeries("med_surface_field.jld2", "S")

n = Observable(1)

uₙ = @lift u_ts[$n]
vₙ = @lift v_ts[$n]
Tₙ = @lift T_ts[$n]
Sₙ = @lift S_ts[$n]

fig = Figure()
ax = Axis(fig[1, 1], title = "Surface zonal velocity (m s⁻¹)")
heatmap!(ax, uₙ)
ax = Axis(fig[1, 2], title = "Surface meridional velocity (m s⁻¹)")
heatmap!(ax, vₙ)
ax = Axis(fig[2, 1], title = "Surface temperature (ᵒC)")
heatmap!(ax, Tₙ)
ax = Axis(fig[2, 2], title = "Surface salinity (g kg⁻¹)")
heatmap!(ax, Sₙ)

CairoMakie.record(fig, "mediterranean_video.mp4", 1:length(u_ts.times); framerate = 5) do nn
    n[] = nn
end
nothing #hide

# ![](mediterranean_video.mp4)
