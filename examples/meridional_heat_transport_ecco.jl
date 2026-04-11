using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Dates
using Statistics
using Printf

using CUDA; CUDA.device!(3)

arch = GPU()
Nx = 360
Ny = 180
Nz = 50

depth = 5000meters
z = ExponentialDiscretization(Nz, -depth, 0; scale = depth/4)

underlying_grid = TripolarGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 4), z)
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
dataset = ECCO4Monthly()
ecco_temperature           = Metadatum(:temperature; date, dataset)
ecco_salinity              = Metadatum(:salinity; date, dataset)
ecco_sea_ice_thickness     = Metadatum(:sea_ice_thickness; date, dataset)
ecco_sea_ice_concentration = Metadatum(:sea_ice_concentration; date, dataset)

set!(ocean.model, T=ecco_temperature, S=ecco_salinity)
set!(sea_ice.model, h=ecco_sea_ice_thickness, ℵ=ecco_sea_ice_concentration)

radiation  = Radiation(arch)
atmosphere = JRA55PrescribedAtmosphere(arch; backend=JRA55NetCDFBackend(80),
                                       include_rivers_and_icebergs = false)
esm = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

simulation = Simulation(esm; Δt=20minutes, stop_time=5*365days)

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    u, v, w = ocean.model.velocities
    T = ocean.model.tracers.T
    e = ocean.model.tracers.e
    Tmin, Tmax, Tavg = minimum(T), maximum(T), mean(view(T, :, :, ocean.model.grid.Nz))
    emax = maximum(e)
    umax = (maximum(abs, u), maximum(abs, v), maximum(abs, w))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iter: %d", prettytime(sim), iteration(sim))
    msg2 = @sprintf(", max|uo|: (%.1e, %.1e, %.1e) m s⁻¹", umax...)
    msg3 = @sprintf(", max(e): %.2f m² s⁻²", emax)
    msg4 = @sprintf(", wall time: %s \n", prettytime(step_time))

    @info msg1 * msg2 * msg3 * msg4

    wall_time[] = time_ns()

     return nothing
end

# And add it as a callback to the simulation.
add_callback!(simulation, progress, IterationInterval(200))

mht = Field(meridional_heat_transport(esm))

ocean.output_writers[:mth] = JLD2Writer(ocean.model, (; mht);
                                        schedule = TimeInterval(3hours),
                                        filename = "ocean_one_degree_mht",
                                        overwrite_existing = true)

run!(simulation)

##

using Oceananigans

mht  = FieldTimeSeries("ocean_one_degree_mht.jld2", "mht"; backend = OnDisk())

times = mht.times
Nt = length(times)

grid = mht.grid
Ny = size(mht.grid, 2)

mht_mean  = deepcopy(mht[1][1, :, 1])

for iter in 1:Nt
    @info "iteration $iter out of $Nt"
    mht_mean  +=  mht[iter][1, :, 1]
end

@. mht_mean = mht_mean / Nt

using CairoMakie

fig = Figure()
ax = Axis(fig[1, 1], xlabel="latitude (deg)", ylabel="MHT (PW)")

φ = φnodes(grid, Face())

lines!(ax, φ, mht_mean[1:Ny+1]  / 1e15, linewidth=4)

save("mht.png", fig)
