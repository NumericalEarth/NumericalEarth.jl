# # Near-global ocean simulation
#
# This example sets up and runs a near-global ocean simulation using Oceananigans.jl and
# NumericalEarth.jl. The simulation covers latitudes from 75°S to 75°N, with a horizontal
# resolution of 1/4 degree and 40 vertical levels.
#
# We visualize the results with CairoMakie.jl.
#
# ## Initial setup with package imports
#
# We begin by importing the Julia packages for visualization (CairoMakie),
# ocean modeling (Oceananigans, NumericalEarth), dates and times (CFTime, Dates),
# and running on CUDA-enabled GPUs (CUDA).

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using CairoMakie
using CFTime
using Dates
using Printf
using CUDA

# ### Grid configuration
#
# We define a `LatitudeLongitudeGrid` that spans latitudes from 75°S to 75°N with a horizontal
# resolution of 1/4 degree and 40 vertical levels. Exponential vertical spacing resolves the
# upper ocean better. The domain is 6000 meters deep, and the simulation runs on a GPU.

arch = GPU()
Nx = 1440
Ny = 600
Nz = 40

depth = 6000meters
z = ExponentialDiscretization(Nz, -depth, 0, mutable=true)

grid = LatitudeLongitudeGrid(arch;
                             size = (Nx, Ny, Nz),
                             halo = (7, 7, 7),
                             z,
                             latitude  = (-75, 75),
                             longitude = (0, 360))

# ### Bathymetry and immersed boundary
#
# We use `regrid_bathymetry` to derive the bottom height from ETOPO data.
# We smooth the interpolated data with 5 interpolation passes. We also fill in
# * all enclosed basins except the 3 largest (`major_basins`), and
# * regions that are shallower than `minimum_depth`.

bottom_height = regrid_bathymetry(grid;
                                  minimum_depth = 10meters,
                                  interpolation_passes = 5,
                                  major_basins = 3)

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)

# Let's see what the bathymetry looks like:

fig, ax, hm = heatmap(bottom_height_field(grid), colormap=:deep, colorrange=(-depth, 0))
Colorbar(fig[0, 1], hm, label="Bottom height (m)", vertical=false)
save("bathymetry.png", fig)
nothing #hide

# ![](bathymetry.png)

# ### Ocean model configuration
#
# We build the ocean simulation with `ocean_simulation`,

ocean = ocean_simulation(grid)

# which builds a default ocean model:

ocean.model

# We initialize the ocean model with ECCO4 temperature and salinity on January 1, 1992.

date = DateTime(1992, 1, 1)
set!(ocean.model, MetadataSet(:temperature, :salinity; dataset=ECCO4Monthly(), date))

# ### Prescribed atmosphere, radiation, and land
#
# Next we build the prescribed atmosphere, radiation, and land components that drive
# the ocean simulation. The atmospheric state, the downwelling shortwave and longwave
# radiation, and the river and iceberg runoff all come from JRA55.

atmosphere = JRA55PrescribedAtmosphere(arch)
radiation  = JRA55PrescribedRadiation(arch)
land       = JRA55PrescribedLand(grid)

# ## The coupled simulation

# We assemble the ocean, atmosphere, land, and radiation into a coupled model,

coupled_model = OceanOnlyModel(ocean; atmosphere, land, radiation)

# We then create a coupled simulation.

simulation = Simulation(coupled_model; Δt=25minutes, stop_time=60days)

# We define a callback function to monitor the simulation's progress,

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    u, v, w = ocean.model.velocities
    T = ocean.model.tracers.T

    Tmax = maximum(T)
    Tmin = minimum(T)

    umax = (maximum(abs, u), maximum(abs, v), maximum(abs, w))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg = @sprintf("Iter: %d, time: %s, Δt: %s", iteration(sim), prettytime(sim), prettytime(sim.Δt))
    msg *= @sprintf(", max|u|: (%.2e, %.2e, %.2e) m s⁻¹, extrema(T): (%.2f, %.2f) ᵒC, wall time: %s",
                    umax..., Tmin, Tmax, prettytime(step_time))

    @info msg

    wall_time[] = time_ns()

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, TimeInterval(5days))

# ### Set up output writers
#
# We save the surface velocities and tracers every day. The `indices` keyword argument
# saves only a slice of each three-dimensional field: here, the surface level `k = grid.Nz`.

outputs = merge(ocean.model.tracers, ocean.model.velocities)
ocean.output_writers[:surface] = JLD2Writer(ocean.model, outputs;
                                            schedule = TimeInterval(1days),
                                            including = [:grid],
                                            filename = "near_global_surface_fields",
                                            indices = (:, :, grid.Nz),
                                            with_halos = true,
                                            overwrite_files = true,
                                            array_type = Array{Float32})

# ### Running the simulation

run!(simulation)

# ## A pretty movie
#
# It's time to make a pretty movie of the simulation. First we load the output we saved
# on disk and plot the final snapshot:

u_ts = FieldTimeSeries("near_global_surface_fields.jld2", "u"; backend = OnDisk())
v_ts = FieldTimeSeries("near_global_surface_fields.jld2", "v"; backend = OnDisk())
T_ts = FieldTimeSeries("near_global_surface_fields.jld2", "T"; backend = OnDisk())
e_ts = FieldTimeSeries("near_global_surface_fields.jld2", "e"; backend = OnDisk())

times = u_ts.times
Nt = length(times)

n = Observable(Nt)

Tₙ = @lift T_ts[$n]
eₙ = @lift e_ts[$n]

# The surface speed is a field computed from snapshot velocities that we `set!` at every frame.

uₙ = u_ts[1]
vₙ = v_ts[1]
s = Field(sqrt(uₙ^2 + vₙ^2))

sₙ = @lift begin
    set!(uₙ, u_ts[$n])
    set!(vₙ, v_ts[$n])
    s
end

title = @lift string("Near-global 1/4 degree ocean simulation after ",
                     prettytime(times[$n] - times[1]))

fig = Figure(size = (800, 1200))

axs = Axis(fig[1, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
axT = Axis(fig[2, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")
axe = Axis(fig[3, 1], xlabel="Longitude (deg)", ylabel="Latitude (deg)")

hm = heatmap!(axs, sₙ, colorrange = (0, 0.5), colormap = :deep, nan_color = :lightgray)
Colorbar(fig[1, 2], hm, label = "Surface speed (m s⁻¹)")

hm = heatmap!(axT, Tₙ, colorrange = (-1, 30), colormap = :magma, nan_color = :lightgray)
Colorbar(fig[2, 2], hm, label = "Surface temperature (ᵒC)")

hm = heatmap!(axe, eₙ, colorrange = (0, 1e-3), colormap = :solar, nan_color = :lightgray)
Colorbar(fig[3, 2], hm, label = "Turbulent kinetic energy (m² s⁻²)")

Label(fig[0, :], title)

save("snapshot.png", fig)
nothing #hide

# ![](snapshot.png)

# And now we make a movie:

CairoMakie.record(fig, "near_global_ocean_surface.mp4", 1:Nt, framerate = 8) do nn
    n[] = nn
end
nothing #hide

# ![](near_global_ocean_surface.mp4)
