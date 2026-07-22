using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Dates
using Statistics
using Printf

using CUDA

arch = GPU()
Nx = 360
Ny = 180
Nz = 50

depth = 5000meters
z = MutableVerticalDiscretization(ExponentialDiscretization(Nz, -depth, 0; scale = depth/4))

underlying_grid = TripolarGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 4), z)

destination_grid = LatitudeLongitudeGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 4), z, longitude = (0, 360), latitude = (-89, 89))

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
land       = JRA55PrescribedLand(arch)
radiation  = JRA55PrescribedRadiation(arch)
esm = OceanSeaIceModel(ocean, sea_ice; atmosphere, land, radiation)

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

# mht_vT = Field(meridional_heat_transport(simulation, MeridionalFluxMethod())) # This currently is not supported with Othrogonal grids, so we use the OHC method instead.
temperature_budget = BudgetComputation(:temperature, esm)
add_callback!(simulation, temperature_budget)
mht_OHC = Field(meridional_heat_transport(simulation; destination_grid))

ocean.output_writers[:mth] = JLD2Writer(ocean.model, (; mht_OHC);
                                        schedule = TimeInterval(3hours),
                                        filename = "ocean_one_degree_mht",
                                        overwrite_existing = true)

run!(simulation)

##

using Oceananigans

# mht_vT  = FieldTimeSeries("ocean_one_degree_mht.jld2", "mht_vT"; backend = OnDisk())
mht_OHC = FieldTimeSeries("ocean_one_degree_mht.jld2", "mht_OHC"; backend = OnDisk())

times = mht_OHC.times
Nt = length(times)

grid = mht_OHC.grid
Ny = size(mht_OHC.grid, 2)

# mht_vT_mean  = deepcopy(mht_vT[1][1, :, 1])
mht_OHC_mean = deepcopy(mht_OHC[1][1, :, 1])

for iter in 1:Nt
    @info "iteration $iter out of $Nt"
    # mht_vT_mean  +=  mht_vT[iter][1, :, 1]
    mht_OHC_mean += mht_OHC[iter][1, :, 1]
end

@. mht_OHC_mean = mht_OHC_mean / Nt
# @. mht_vT_mean = mht_vT_mean / Nt

using CairoMakie

fig = Figure()
ax = Axis(fig[1, 1], xlabel="latitude (deg)", ylabel="MHT (PW)")

φ = φnodes(grid, Face())

# lines!(ax, φ, mht_vT_mean[1:Ny+1]  / 1e15, linewidth=4, label="via vT")
lines!(ax, φ, mht_OHC_mean[1:Ny+1] / 1e15, linewidth=4, label="via OHC")
Legend(fig[2, :], ax, orientation=:horizontal)
Label(fig[0, :], "Meridional heat transport", fontsize=16, tellwidth=false)

save("mht.png", fig)
