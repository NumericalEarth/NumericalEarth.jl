# # An Ocean Simulation at 4ᵒ Resolution Forced by JRA55 Reanalysis
#
# This example uses NumericalEarth's PythonCall extension to run a near-global
# ocean simulation at 4-degree resolution with the Veros ocean model.
# The ocean is forced by the JRA55 reanalysis.
#
# For this example, we need NumericalEarth, PythonCall, Oceananigans, and
# CairoMakie to visualize the simulation.

using NumericalEarth
using PythonCall
using Oceananigans, Oceananigans.Units
using CairoMakie
using Printf

# We import the Veros 4-degree ocean setup: a near-global ocean with a uniform resolution
# of 4 degrees in both latitude and longitude, spanning from 80°S to 80°N. The setup is
# defined in the `veros.setups.global_4deg` module.

# Before importing the setup, the Veros Python package must be available in the
# active CondaPkg environment. The documentation CI workflow installs it ahead
# of time. For a fresh local environment, run
#
# ```julia
# VerosModule = Base.get_extension(NumericalEarth, :NumericalEarthVerosExt)
# VerosModule.install_veros()
# ```
#
# once before executing this example.

VerosModule = Base.get_extension(NumericalEarth, :NumericalEarthVerosExt)

VerosModule.remove_outputs(:global_4deg)

# We now load and instantiate the Veros setup as `ocean`.

ocean = VerosModule.VerosOceanSimulation("global_4deg", :GlobalFourDegreeSetup)

# The loaded Veros setup contains a `set_forcing` method that computes the surface fluxes as a
# restoring toward climatology. We replace it with a function that computes only the TKE forcing,
# which depends on the wind stresses that NumericalEarth sets, so that our u, v, T, and S forcings
# are not overwritten. The `set_forcing_tke_only` function below is adapted from the `set_forcing`
# method in https://github.com/team-ocean/veros/blob/main/veros/setups/global_4deg/global_4deg.py

pyexec("""
def set_forcing_tke_only(state):
    from veros.core.operators import numpy as npx, update, at
    from veros import KernelOutput

    vs = state.variables
    settings = state.settings

    if settings.enable_tke:
        vs.forc_tke_surface = update(
            vs.forc_tke_surface,
            at[1:-1, 1:-1],
            npx.sqrt(
                (0.5 * (vs.surface_taux[1:-1, 1:-1] + vs.surface_taux[:-2, 1:-1]) / settings.rho_0) ** 2
                + (0.5 * (vs.surface_tauy[1:-1, 1:-1] + vs.surface_tauy[1:-1, :-2]) / settings.rho_0) ** 2
            ) ** 1.5,
        )

    return KernelOutput(
        surface_taux=vs.surface_taux,
        surface_tauy=vs.surface_tauy,
        forc_tke_surface=vs.forc_tke_surface,
        forc_temp_surface=vs.forc_temp_surface,
        forc_salt_surface=vs.forc_salt_surface,
    )

ocean.set_forcing = set_forcing_tke_only
""", Main, (ocean=ocean.setup,))

# We force the 4-degree setup with a prescribed atmosphere based on the JRA55 reanalysis.
# The atmosphere supplies the 2-meter wind velocity, temperature, and humidity, as well as the
# freshwater fluxes; the prescribed radiation supplies the downwelling longwave and shortwave radiation.

atmosphere = JRA55PrescribedAtmosphere()
radiation = JRA55PrescribedRadiation()

# We couple the ocean to the atmosphere; for simplicity, we do not include sea ice.

coupled_model = OceanSeaIceModel(ocean, nothing; atmosphere, radiation)
simulation = Simulation(coupled_model; Δt = 30minutes, stop_time = 60days)

# We set up a progress callback that prints the current time, iteration, and maximum velocities
# every 10 days.

wall_time = Ref(time_ns())

function progress(sim)
    ocean = sim.model.ocean
    umax = maximum(PyArray(ocean.setup.state.variables.u))
    vmax = maximum(PyArray(ocean.setup.state.variables.v))
    wmax = maximum(PyArray(ocean.setup.state.variables.w))

    step_time = 1e-9 * (time_ns() - wall_time[])

    msg1 = @sprintf("time: %s, iteration: %d, Δt: %s, ", prettytime(sim), iteration(sim), prettytime(sim.Δt))
    msg2 = @sprintf("maximum(u): (%.2f, %.2f, %.2f) m s⁻¹, ", umax, vmax, wmax)
    msg3 = @sprintf("wall time: %s \n", prettytime(step_time))

    @info msg1 * msg2 * msg3

    wall_time[] = time_ns()

    return nothing
end

add_callback!(simulation, progress, TimeInterval(10days))

# We also save the surface velocities that the ocean exchanges with the atmosphere.

(; u, v) = coupled_model.interfaces.exchanger.ocean.state

simulation.output_writers[:surface] = JLD2Writer(coupled_model, (; u, v);
                                                 schedule = IterationInterval(10),
                                                 filename = "veros_ocean_surface_fields",
                                                 overwrite_existing = true)

# Let's run the simulation!

run!(simulation)

# After the simulation is done, we animate the surface zonal and meridional velocities.

u_ts = FieldTimeSeries("veros_ocean_surface_fields.jld2", "u")
v_ts = FieldTimeSeries("veros_ocean_surface_fields.jld2", "v")
Nt = length(u_ts.times)

n  = Observable(1)
un = @lift u_ts[$n]
vn = @lift v_ts[$n]

fig = Figure(size = (900, 630))
ax1 = Axis(fig[1, 1]; title = "Surface zonal velocity (m s⁻¹)", ylabel = "Latitude")
ax2 = Axis(fig[2, 1]; title = "Surface meridional velocity (m s⁻¹)", ylabel = "Latitude")
hm1 = heatmap!(ax1, un, colormap = :bwr, colorrange = (-0.2, 0.2))
hm2 = heatmap!(ax2, vn, colormap = :bwr, colorrange = (-0.2, 0.2))

Colorbar(fig[1, 2], hm1)
Colorbar(fig[2, 2], hm2)

CairoMakie.record(fig, "veros_ocean_surface.mp4", 1:Nt, framerate = 8) do nn
    n[] = nn
end
nothing #hide

# ![](veros_ocean_surface.mp4)
