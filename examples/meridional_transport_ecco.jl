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
z = ExponentialDiscretization(Nz, -depth, 0, mutable=true; scale = depth/4)

underlying_grid = TripolarGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 4), z)

bottom_height = regrid_bathymetry(underlying_grid;
                                  minimum_depth = 10,
                                  interpolation_passes = 10,
                                  major_basins = 2)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height);
                            active_cells_map=true)

underlying_grid = LatitudeLongitudeGrid(arch;
                                    size = (Nx, Ny, 1),
                                    halo = (5, 5, 4),
                                    z = (-1, 0),
                                    longitude = (0, 360),
                                    latitude = (-89, 89))

bottom_height = regrid_bathymetry(underlying_grid;
                                  minimum_depth = 10,
                                  interpolation_passes = 10,
                                  major_basins = 2)

destination_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height);
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

simulation = Simulation(esm; Δt=20minutes, stop_time=365days)

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
budget_salinity = BudgetComputation(:salinity, esm)
budget_mass = BudgetComputation(:mass, esm)

add_callback!(simulation, temperature_budget)
add_callback!(simulation, budget_salinity)
add_callback!(simulation, budget_mass)

mht_field = Field(meridional_transport(simulation, :temperature, TendencyMethod(); destination_grid))
mst_field = Field(meridional_transport(simulation, :salinity, TendencyMethod(); destination_grid))
mt_field = Field(meridional_transport(simulation, :mass, TendencyMethod(); destination_grid))

budget_outputs = (
    mht                   = mht_field,
    mst                   = mst_field,
    mt                    = mt_field
)

ocean.output_writers[:merid_trans] = JLD2Writer(ocean.model, budget_outputs;
                                        schedule = TimeInterval(3hours),
                                        filename = "ocean_one_degree_mt",
                                        overwrite_existing = true)

run!(simulation)

##

using Oceananigans

mht = FieldTimeSeries("ocean_one_degree_mt.jld2", "mht"; backend = OnDisk())
mst = FieldTimeSeries("ocean_one_degree_mt.jld2", "mst"; backend = OnDisk())
mt = FieldTimeSeries("ocean_one_degree_mt.jld2", "mt"; backend = OnDisk())

times = mht.times
Nt = length(times)

grid = mht.grid
Ny = size(mht.grid, 2)

mht_avg = zeros(eltype(mht[1]), size(interior(mht[1])[1, :, 1]))
mst_avg = zeros(eltype(mst[1]), size(interior(mst[1])[1, :, 1]))
mt_avg = zeros(eltype(mt[1]), size(interior(mt[1])[1, :, 1]))

for iter in 1:Nt
    @info "iteration $iter out of $Nt"
    mht_values = vec(Array(interior(mht[iter])))
    mst_values = vec(Array(interior(mst[iter])))
    mt_values = vec(Array(interior(mt[iter])))

    mht_avg .+= mht_values
    mst_avg .+= mst_values
    mt_avg .+= mt_values
end

mht_avg ./= Nt
mst_avg ./= Nt
mt_avg ./= Nt

using CairoMakie

fig = Figure()
ax1 = Axis(fig[1, 1], xlabel="latitude (deg)", ylabel="MHT (PW)")
ax2 = Axis(fig[2, 1], xlabel="latitude (deg)", ylabel="MST (kg/s)")
ax3 = Axis(fig[3, 1], xlabel="latitude (deg)", ylabel="MT (m³ s⁻¹)")

φ = φnodes(grid, Face())

lines!(ax1, φ, mht_avg / 1e15, linewidth=4)
xlims!(ax1, extrema(φ))
lines!(ax2, φ, mst_avg, linewidth=4)
xlims!(ax2, extrema(φ))
lines!(ax3, φ, mt_avg, linewidth=4)
xlims!(ax3, extrema(φ))

save("merid_trans_ecco.png", fig)
