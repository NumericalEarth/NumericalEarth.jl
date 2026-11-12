# # [A one-degree ocean and sea ice under the Ai2 Climate Emulator](@id ace-coupled-simulation)
#
# This example couples the [Ai2 Climate Emulator](https://github.com/ai2cm/ace) (ACE2-ERA5), a neural network
# atmosphere trained on the ERA5 reanalysis, to the global ocean--sea ice simulation of the
# [one-degree example](@ref one-degree-ocean-seaice). Every 6 hours ACE advances the atmosphere on its 1ᵒ Gaussian
# grid and returns the 10-meter wind, the 2-meter air temperature and specific humidity, the surface pressure,
# the precipitation, and the downwelling shortwave and longwave radiation. NumericalEarth computes the turbulent
# fluxes into the ocean and sea ice from this near-surface state with its bulk formulas, and hands ACE the skin
# temperature and the sea ice concentration of the surface underneath.
#
# ACE computes its own surface fluxes internally, so the heat and water leaving the atmosphere are not those
# entering the ocean and sea ice: the coupled system conserves neither energy nor water.
#
# ACE runs in Python through the `fme` package, which must be available in the active CondaPkg environment.
# For a fresh local environment, run
#
# ```julia
# ACEModule = Base.get_extension(NumericalEarth, :NumericalEarthPythonCallExt)
# ACEModule.install_ace()
# ```
#
# once and restart Julia before executing this example.

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using ConservativeRegridding
using PythonCall, CondaPkg
using Dates
using Printf
using CUDA

# ### Ocean and sea ice
#
# The grid, bathymetry, closures and numerics are those of the one-degree example.

arch = GPU()
Nx = 360
Ny = 180
Nz = 50

depth = 5000meters
z = ExponentialDiscretization(Nz, -depth, 0; scale = depth/4, mutable = true)
underlying_grid = TripolarGrid(arch; size = (Nx, Ny, Nz), halo = (5, 5, 4), z)

bottom_height = regrid_bathymetry(underlying_grid; minimum_depth = 10, interpolation_passes = 10, major_basins = 2)
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map = true)

using Oceananigans.TurbulenceClosures: IsopycnalSkewSymmetricDiffusivity, AdvectiveFormulation

eddy_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=1e3, κ_symmetric=1e3, skew_flux_formulation=AdvectiveFormulation())
vertical_mixing = NumericalEarth.Oceans.default_ocean_closure()

free_surface       = SplitExplicitFreeSurface(grid; substeps = 70)
momentum_advection = WENOVectorInvariant(order = 5)
tracer_advection   = WENO(order = 5)

ocean = ocean_simulation(grid; momentum_advection, tracer_advection, free_surface,
                         closure = (eddy_closure, vertical_mixing))

sea_ice = sea_ice_simulation(grid, ocean; advection = tracer_advection)

# ACE starts from initial conditions on the first of each month of 1940, 1950, 1979, 2001 and 2020. We pick
# January 2001, which the ECCO state estimate also covers, and initialize the ocean and sea ice from ECCO.

date = DateTime(2001, 1, 1)
ecco_set = MetadataSet(:temperature, :salinity, :sea_ice_thickness, :sea_ice_concentration;
                       dataset = ECCO4Monthly(), date)

set!(ocean.model, ecco_set)
set!(sea_ice.model, ecco_set)

# ### The emulated atmosphere and the river runoff
#
# `atmosphere_simulation` downloads the ACE2-ERA5 checkpoint, the initial condition on `date`, and the forcing
# for the year that follows. ACE2-ERA5 has no runoff, so river and iceberg runoff come from JRA55-do.

atmosphere = atmosphere_simulation(ACE2ERA5(), arch; dates = (date, date + Year(1)))
land = JRA55PrescribedLand(grid)

# ### Coupled model
#
# Without a `radiation` component, the coupled model takes ACE's downwelling radiation, which the ocean and
# sea ice absorb with their own albedo and emissivity. ACE holds its state between its 6-hourly steps, so the
# coupled time step must divide 6 hours.

coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, land)
simulation = Simulation(coupled_model; Δt = 30minutes, stop_time = 365days)

wall_time = Ref(time_ns())

function progress(sim)
    sea_surface_temperature = view(sim.model.ocean.model.tracers.T, :, :, Nz)
    air_temperature = sim.model.atmosphere.surface_state.T
    elapsed = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iteration: %d", prettytime(sim), iteration(sim))
    msg2 = @sprintf(", SST: (%.1f, %.1f) ᵒC", minimum(sea_surface_temperature), maximum(sea_surface_temperature))
    msg3 = @sprintf(", 2-meter air temperature: (%.1f, %.1f) K", minimum(air_temperature), maximum(air_temperature))
    msg4 = @sprintf(", wall time: %s", prettytime(elapsed))
    @info msg1 * msg2 * msg3 * msg4

    wall_time[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, TimeInterval(10days))

# ### Output
#
# We save annual means of the sea surface temperature, the sea ice concentration and ACE's 2-meter air temperature,
# which lives on ACE's grid, and 30-day averages of the horizontally averaged ocean temperature, whose top level is
# the global-mean sea surface temperature.

surface_outputs = (sea_surface_temperature = ocean.model.tracers.T,
                   sea_ice_concentration = sea_ice.model.ice_concentration,
                   air_temperature = atmosphere.surface_state.T)

simulation.output_writers[:annual_mean] = JLD2Writer(coupled_model, surface_outputs;
                                                     schedule = AveragedTimeInterval(365days),
                                                     filename = "ace_annual_mean_surface",
                                                     indices = (:, :, Nz),
                                                     overwrite_files = true)

horizontal_mean_temperature = Average(ocean.model.tracers.T, dims = (1, 2))

simulation.output_writers[:horizontal_mean] = JLD2Writer(coupled_model, (; horizontal_mean_temperature);
                                                         schedule = AveragedTimeInterval(30days),
                                                         filename = "ace_horizontal_mean_temperature",
                                                         overwrite_files = true)

run!(simulation)

# ### Annual means and the global-mean sea surface temperature
#
# The last record of each annual-mean output is the average over the whole year.

using CairoMakie

sea_surface_temperature = FieldTimeSeries("ace_annual_mean_surface.jld2", "sea_surface_temperature")
sea_ice_concentration   = FieldTimeSeries("ace_annual_mean_surface.jld2", "sea_ice_concentration")
air_temperature         = FieldTimeSeries("ace_annual_mean_surface.jld2", "air_temperature")

horizontal_mean_temperature = FieldTimeSeries("ace_horizontal_mean_temperature.jld2", "horizontal_mean_temperature")
times = horizontal_mean_temperature.times

fig = Figure(size = (1000, 900))

axsst  = Axis(fig[1, 1], title = "Sea surface temperature (ᵒC)")
axice  = Axis(fig[1, 3], title = "Sea ice concentration")
axair  = Axis(fig[2, 1:3], title = "ACE 2-meter air temperature (K)")
axmean = Axis(fig[3, 1:3], xlabel = "Time (days)", ylabel = "Global-mean SST (ᵒC)")

hm = heatmap!(axsst, sea_surface_temperature[end], colormap = :magma, colorrange = (-2, 30), nan_color = :lightgray)
Colorbar(fig[1, 2], hm)
hm = heatmap!(axice, sea_ice_concentration[end], colormap = :blues, colorrange = (0, 1), nan_color = :lightgray)
Colorbar(fig[1, 4], hm)
hm = heatmap!(axair, air_temperature[end], colormap = :thermal, colorrange = (230, 305))
Colorbar(fig[2, 4], hm)
lines!(axmean, times ./ days, horizontal_mean_temperature[1, 1, Nz, :])

hidedecorations!(axsst)
hidedecorations!(axice)

fig
