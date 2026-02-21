# # Atmospheric convection over a slab ocean
#
# This example demonstrates coupling a Breeze atmospheric large eddy simulation (LES)
# with a slab ocean model using NumericalEarth's `EarthSystemModel` framework.
#
# The atmosphere drives convective turbulence over a warm ocean surface. The coupling
# framework computes turbulent surface fluxes (sensible heat, latent heat, and momentum)
# using Monin--Obukhov similarity theory. These fluxes cool the ocean and heat
# the atmosphere, creating a two-way feedback loop.
#
# The slab ocean model represents a well-mixed ocean layer of fixed depth ``H``,
# with temperature evolving in response to the net surface heat flux:
#
# ```math
# \frac{∂T}{∂t} = \frac{Q}{ρ \, c_p \, H}
# ```
#
# where ``Q`` is the net downward surface heat flux (W/m²), ``ρ`` is the seawater density,
# ``c_p`` is the heat capacity, and ``H`` is the slab depth. A shallow slab depth (here 1 m)
# gives a fast SST response, making the air-sea coupling visible within hours.

using NumericalEarth
using Breeze
using Oceananigans
using Oceananigans.Units
using Printf

# ## Grid setup
#
# We use a 2D domain in the x-z plane: 20 km wide and 10 km tall with
# 128 × 128 grid points. The `Periodic` x-topology allows convective cells
# to wrap around, and `Flat` y-topology makes this a 2D simulation.

grid = RectilinearGrid(size = (128, 128), halo = (5, 5),
                       x = (-10kilometers, 10kilometers),
                       z = (0, 10kilometers),
                       topology = (Periodic, Flat, Bounded))

# ## Atmosphere
#
# `atmosphere_simulation` constructs a Breeze `AtmosphereModel` with sensible defaults:
# anelastic dynamics, warm-phase saturation adjustment microphysics, and WENO advection.
# The `potential_temperature` keyword sets the reference potential temperature for
# the anelastic reference state. Surface fluxes are not prescribed directly;
# instead they are computed by the `EarthSystemModel` coupling framework using
# similarity theory.

θ₀ = 270 # K
atmosphere = atmosphere_simulation(grid; potential_temperature=θ₀)

# ## Atmospheric initial conditions
#
# We initialize the atmosphere with the reference potential temperature profile
# plus small random perturbations below 500 m. These perturbations seed convective
# instability, which develops into turbulent convection driven by surface heat fluxes.
# A background zonal wind of 1 m/s provides a nonzero wind speed for the
# similarity theory flux computation.

reference_state = atmosphere.dynamics.reference_state

θᵢ(x, z) = reference_state.potential_temperature + 0.1 * randn() * (z < 500)
set!(atmosphere, θ=θᵢ, u=1)

# ## Slab ocean
#
# The slab ocean lives on a 1D horizontal grid that matches the atmosphere's
# x-direction. We use a 1 m slab depth so that the SST responds quickly to
# surface fluxes — a 50 m slab would take days to show appreciable temperature
# change, making a 4-hour simulation appear nearly constant.
#
# The ocean starts 20 K warmer than the atmosphere (290 K vs 270 K), creating
# a strong air-sea temperature contrast that drives vigorous sensible and
# latent heat fluxes.

sst_grid = RectilinearGrid(grid.architecture,
                           size = grid.Nx,
                           halo = grid.Hx,
                           x = (-10kilometers, 10kilometers),
                           topology = (Periodic, Flat, Flat))

ocean = SlabOcean(sst_grid, depth=1, density=1025, heat_capacity=4000)
set!(ocean, T=θ₀ + 20)

# ## Coupled model
#
# `AtmosphereOceanModel` builds an `EarthSystemModel` that couples the atmosphere
# and ocean. At each coupling step the framework:
#
# 1. Interpolates atmosphere and ocean states onto a shared exchange grid.
# 2. Computes turbulent fluxes via Monin--Obukhov similarity theory.
# 3. Assembles net heat, moisture, and momentum fluxes for each component.
#
# The fluxes feed back into both the atmosphere (as bottom boundary conditions)
# and the ocean (as a temperature tendency).

model = AtmosphereOceanModel(atmosphere, ocean)

# ## Simulation
#
# We run for 4 hours with a 1-second timestep. This is long enough for
# convective turbulence to develop and for the SST to cool appreciably.

simulation = Simulation(model, Δt=1, stop_time=4hours)

# ## Progress callback
#
# Print a summary of the simulation state every 400 iterations (~7 minutes),
# including the maximum wind speed and the SST range across the domain.

function progress(sim)
    atmos = sim.model.atmosphere
    u, v, w = atmos.velocities

    umax = maximum(abs, u)
    wmax = maximum(abs, w)

    sst = sim.model.ocean.temperature
    sst_min = minimum(sst)
    sst_max = maximum(sst)

    msg = @sprintf("Iter: %d, t = %s, max|u|: %.2e, max|w|: %.2e, SST: (%.3f, %.3f)",
                    iteration(sim), prettytime(sim), umax, wmax, sst_min, sst_max)
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(400))

# ## Output writers
#
# We define two `JLD2Writer`s: one for atmospheric fields and one for SST.
# Atmosphere outputs include wind speed ``s = \sqrt{u^2 + w^2}``,
# vorticity ``ξ = ∂w/∂x - ∂u/∂z``, temperature ``T``,
# liquid-ice potential temperature ``θ_{li}``, cloud liquid water ``q^ℓ``,
# and total specific humidity ``q^t``. These are saved every 2 minutes.

u, v, w = atmosphere.velocities
T = atmosphere.temperature
θ = liquid_ice_potential_temperature(atmosphere)
qˡ = atmosphere.microphysical_fields.qˡ
s = sqrt(u^2 + w^2)
ξ = ∂x(w) - ∂z(u)
ρqᵗ = atmosphere.moisture_density
ρ₀ = atmosphere.dynamics.reference_state.density
qᵗ = ρqᵗ / ρ₀

simulation.output_writers[:atmos] = JLD2Writer(model, (; s, ξ, T, θ, qˡ, qᵗ),
                                               filename = "atmosphere_slab_ocean",
                                               schedule = TimeInterval(2minutes),
                                               overwrite_existing = true)

# SST is saved to a separate file since it lives on a different grid.

simulation.output_writers[:sst] = JLD2Writer(model, (; SST=ocean.temperature),
                                             filename = "sst_slab_ocean",
                                             schedule = TimeInterval(2minutes),
                                             overwrite_existing = true)

# ## Run

@info "Running atmosphere–slab ocean coupled simulation..."
run!(simulation)
@info "Simulation complete."

# ## Animation
#
# After the simulation, we load the saved data with `FieldTimeSeries`
# and animate the evolving atmospheric fields alongside the SST.

using CairoMakie

# Since the `EarthSystemModel` does not have a single `.grid` property,
# we pass the appropriate grid explicitly when loading each `FieldTimeSeries`.

s_ts  = FieldTimeSeries("atmosphere_slab_ocean.jld2", "s"; grid)
ξ_ts  = FieldTimeSeries("atmosphere_slab_ocean.jld2", "ξ"; grid)
θ_ts  = FieldTimeSeries("atmosphere_slab_ocean.jld2", "θ"; grid)
qˡ_ts = FieldTimeSeries("atmosphere_slab_ocean.jld2", "qˡ"; grid)
sst_ts = FieldTimeSeries("sst_slab_ocean.jld2", "SST"; grid=sst_grid)

times = s_ts.times
Nt = length(times)

# Set up a five-panel figure: wind speed, vorticity, potential temperature,
# cloud water, and SST.

fig = Figure(size = (1200, 800), fontsize = 14)

ax_s = Axis(fig[1, 1], title="Wind speed (m/s)", xlabel="x (km)", ylabel="z (km)")
ax_ξ = Axis(fig[1, 2], title="Vorticity (1/s)", xlabel="x (km)", ylabel="z (km)")
ax_θ = Axis(fig[2, 1], title="θₗᵢ (K)", xlabel="x (km)", ylabel="z (km)")
ax_q = Axis(fig[2, 2], title="Cloud water (g/kg)", xlabel="x (km)", ylabel="z (km)")
ax_sst = Axis(fig[3, 1:2], title="Sea surface temperature (K)", xlabel="x (km)", ylabel="SST (K)")

# Use Observables to drive the animation. Indexing a `FieldTimeSeries` with an
# integer returns a `Field` that Oceananigans' Makie recipes can plot directly.

n = Observable(1)

sn  = @lift s_ts[$n]
ξn  = @lift ξ_ts[$n]
θn  = @lift θ_ts[$n]
qˡn = @lift qˡ_ts[$n]
sstn = @lift sst_ts[$n]

heatmap!(ax_s, sn; colormap=:speed, colorrange=(0, 10))
heatmap!(ax_ξ, ξn; colormap=:balance, colorrange=(-0.05, 0.05))
heatmap!(ax_θ, θn; colormap=:thermal, colorrange=(θ₀ - 1, θ₀ + 2))
heatmap!(ax_q, qˡn; colormap=:dense, colorrange=(0, 0.5))
lines!(ax_sst, sstn; color=:red, linewidth=2)
ylims!(ax_sst, θ₀, θ₀ + 22)

title = @lift "Atmosphere–slab ocean coupling, t = " * prettytime(times[$n])
Label(fig[0, 1:2], title, fontsize=18)

# Record the animation by sweeping through the saved snapshots.

@info "Rendering animation..."
record(fig, "atmosphere_slab_ocean.mp4", 1:Nt; framerate=12) do nn
    n[] = nn
end

@info "Animation saved."
nothing #hide

# ![](atmosphere_slab_ocean.mp4)
