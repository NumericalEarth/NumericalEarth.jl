# # Meridional heat transport of a one-degree ocean--sea ice simulation
#
# We run a one-degree ocean--sea ice simulation initialized from ECCO and forced by JRA55,
# save its meridional heat transport, and plot the time-mean transport against latitude.

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Dates
using Printf

using CUDA

arch = GPU()
Nx = 360
Ny = 180
Nz = 50

depth = 5000meters
z = ExponentialDiscretization(Nz, -depth, 0; scale = depth/4)

underlying_grid = LatitudeLongitudeGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 4), z, longitude = (0, 360), latitude = (-80, 80))
bottom_height = regrid_bathymetry(underlying_grid;
                                  minimum_depth = 10,
                                  interpolation_passes = 10,
                                  major_basins = 2)
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height);
                            active_cells_map=true)

free_surface       = SplitExplicitFreeSurface(grid; substeps=70)
momentum_advection = WENOVectorInvariant(order=5)
tracer_advection   = WENO(order=5)
vertical_mixing = NumericalEarth.Oceans.default_ocean_closure()
ocean = ocean_simulation(grid; momentum_advection, tracer_advection, free_surface,
                         closure=(vertical_mixing,))
sea_ice = sea_ice_simulation(grid, ocean; advection=tracer_advection)

date = DateTime(1993, 1, 1)
ecco_set = MetadataSet(:temperature, :salinity,
                       :sea_ice_thickness, :sea_ice_concentration;
                       dataset = ECCO4Monthly(), date)

set!(ocean.model,   ecco_set)   # T, S
set!(sea_ice.model, ecco_set)   # h, ℵ

atmosphere = JRA55PrescribedAtmosphere(arch)
land       = JRA55PrescribedLand(grid)
radiation  = JRA55PrescribedRadiation(arch)
esm = OceanSeaIceModel(ocean, sea_ice; atmosphere, land, radiation)

simulation = Simulation(esm; Δt=20minutes, stop_time=5*365days)

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    u, v, w = ocean.model.velocities
    e = ocean.model.tracers.e
    emax = maximum(e)
    umax = (maximum(abs, u), maximum(abs, v), maximum(abs, w))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iter: %d", prettytime(sim), iteration(sim))
    msg2 = @sprintf(", max|u|: (%.1e, %.1e, %.1e) m s⁻¹", umax...)
    msg3 = @sprintf(", max(e): %.2f m² s⁻²", emax)
    msg4 = @sprintf(", wall time: %s \n", prettytime(step_time))

    @info msg1 * msg2 * msg3 * msg4

    wall_time[] = time_ns()

    return nothing
end

# We add the progress message as a callback to the simulation.

add_callback!(simulation, progress, IterationInterval(200))

mht = meridional_heat_transport(esm)

ocean.output_writers[:mht] = JLD2Writer(ocean.model, (; mht);
                                        schedule = TimeInterval(3hours),
                                        filename = "ocean_one_degree_mht",
                                        overwrite_existing = true)

run!(simulation)

# ## Time-mean meridional heat transport
#
# We load the saved transport and average it over all saved snapshots.

using CairoMakie

mht_ts = FieldTimeSeries("ocean_one_degree_mht.jld2", "mht"; backend = OnDisk())
Nt = length(mht_ts.times)

mean_mht = sum(mht_ts; dims=4)[1] / Nt

φ = φnodes(mht_ts.grid, Face())

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Latitude (deg)", ylabel="Meridional heat transport (PW)")
lines!(ax, φ, mean_mht / 1e15, linewidth=4)

save("mht.png", fig)

fig
