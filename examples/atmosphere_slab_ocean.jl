# # Atmospheric convection over a slab ocean
#
# This example demonstrates coupling a Breeze atmospheric LES with a slab ocean model
# using NumericalEarth's `EarthSystemModel` framework. The sea surface temperature (SST)
# evolves in response to the atmospheric surface heat fluxes computed by similarity theory.
#
# The slab ocean model represents a well-mixed ocean layer of fixed depth ``H`` with the
# SST tendency equation:
#
# ```math
# \frac{∂T}{∂t} = \frac{Q}{ρ \, c_p \, H}
# ```
#
# where ``Q`` is the net downward surface heat flux (W/m²), ``ρ`` is the seawater density,
# ``c_p`` is the heat capacity, and ``H`` is the slab depth.

using NumericalEarth
using Breeze
using Oceananigans
using Oceananigans.Units
using Printf

# ## Grid setup
#
# A 2D domain (x-z plane): 20 km wide × 10 km tall with 128×128 resolution.

grid = RectilinearGrid(size = (128, 128), halo = (5, 5),
                       x = (-10kilometers, 10kilometers),
                       z = (0, 10kilometers),
                       topology = (Periodic, Flat, Bounded))

# ## Atmosphere
#
# `atmosphere_simulation` constructs a Breeze `AtmosphereModel` with sensible defaults
# (anelastic dynamics, warm-phase saturation adjustment microphysics, WENO advection).
# Surface fluxes are handled by the `EarthSystemModel` coupling framework.

θ₀ = 285 # K
atmosphere = atmosphere_simulation(grid; potential_temperature=θ₀)

# ## Initial conditions

reference_state = atmosphere.dynamics.reference_state
set!(atmosphere, θ=reference_state.potential_temperature, u=1)

# ## Slab ocean
#
# A 50 m deep slab ocean with constant initial temperature, on a 1D horizontal grid.

sst_grid = RectilinearGrid(grid.architecture,
                           size = grid.Nx,
                           halo = grid.Hx,
                           x = (-10kilometers, 10kilometers),
                           topology = (Periodic, Flat, Flat))

ocean = SlabOcean(sst_grid, depth=50, density=1025, heat_capacity=4000)
set!(ocean, T=θ₀)

# ## Coupled model
#
# `AtmosphereOceanModel` returns an `EarthSystemModel` that couples the atmosphere
# and ocean through the similarity theory turbulent flux computation.

model = AtmosphereOceanModel(atmosphere, ocean)

# ## Simulation
#
# Run for 4 hours with a fixed timestep.

simulation = Simulation(model, Δt=10, stop_time=4hours)

# ## Diagnostics

T = atmosphere.temperature
θ = liquid_ice_potential_temperature(atmosphere)
qˡ = atmosphere.microphysical_fields.qˡ
u, v, w = atmosphere.velocities

# ## Progress callback

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

add_callback!(simulation, progress, IterationInterval(200))

# ## Output
#
# Collect snapshots of atmosphere fields and SST via callbacks for visualization.

s = sqrt(u^2 + w^2)
ξ = ∂x(w) - ∂z(u)
ρqᵗ = atmosphere.moisture_density
ρ₀ = atmosphere.dynamics.reference_state.density
qᵗ = ρqᵗ / ρ₀

saved_fields = (; s, ξ, T, θ, qˡ, qᵗ)
computed_fields = [compute!(Field(f)) for f in saved_fields]

timeseries = (s=[], ξ=[], T=[], θ=[], qˡ=[], qᵗ=[], SST=[], t=Float64[])

function save_output(sim)
    for f in computed_fields
        compute!(f)
    end
    push!(timeseries.s,  Array(interior(computed_fields[1], :, 1, :)))
    push!(timeseries.ξ,  Array(interior(computed_fields[2], :, 1, :)))
    push!(timeseries.T,  Array(interior(computed_fields[3], :, 1, :)))
    push!(timeseries.θ,  Array(interior(computed_fields[4], :, 1, :)))
    push!(timeseries.qˡ, Array(interior(computed_fields[5], :, 1, :)))
    push!(timeseries.qᵗ, Array(interior(computed_fields[6], :, 1, :)))
    push!(timeseries.SST, Array(interior(ocean.temperature, :, 1, 1)))
    push!(timeseries.t, time(sim))
    return nothing
end

add_callback!(simulation, save_output, TimeInterval(2minutes))

# ## Run

@info "Running atmosphere–slab ocean coupled simulation..."
run!(simulation)
SST = ocean.temperature
@info "Simulation complete. Final SST range: $(minimum(SST)) to $(maximum(SST)) K"

# ## Animation
#
# Create an animation showing the atmospheric state and SST evolution.

using CairoMakie

Nt = length(timeseries.t)

fig = Figure(size = (1200, 800), fontsize = 14)

ax_s = Axis(fig[1, 1], title="Wind speed (m/s)", xlabel="x (km)", ylabel="z (km)")
ax_ξ = Axis(fig[1, 2], title="Vorticity (1/s)", xlabel="x (km)", ylabel="z (km)")
ax_θ = Axis(fig[2, 1], title="θₗᵢ (K)", xlabel="x (km)", ylabel="z (km)")
ax_q = Axis(fig[2, 2], title="Cloud water (g/kg)", xlabel="x (km)", ylabel="z (km)")
ax_sst = Axis(fig[3, 1:2], title="Sea surface temperature (K)", xlabel="x (km)", ylabel="SST (K)")

n = Observable(1)

x_atmo = xnodes(grid, Center()) ./ 1e3
z_atmo = znodes(grid, Center()) ./ 1e3
x_sst = xnodes(sst_grid, Center()) ./ 1e3

sn  = @lift timeseries.s[$n]
ξn  = @lift timeseries.ξ[$n]
θn  = @lift timeseries.θ[$n]
qˡn = @lift timeseries.qˡ[$n] .* 1e3

sstn = @lift timeseries.SST[$n]

heatmap!(ax_s, x_atmo, z_atmo, sn; colormap=:speed, colorrange=(0, 5))
heatmap!(ax_ξ, x_atmo, z_atmo, ξn; colormap=:balance, colorrange=(-0.05, 0.05))
heatmap!(ax_θ, x_atmo, z_atmo, θn; colormap=:thermal, colorrange=(θ₀ - 1, θ₀ + 3))
heatmap!(ax_q, x_atmo, z_atmo, qˡn; colormap=:dense, colorrange=(0, 1))
lines!(ax_sst, x_sst, sstn; color=:red, linewidth=2)
ylims!(ax_sst, θ₀ - 2, θ₀ + 2)

title = @lift "Atmosphere–slab ocean coupling, t = " * prettytime(timeseries.t[$n])
Label(fig[0, 1:2], title, fontsize=18)

@info "Rendering animation..."
record(fig, "atmosphere_slab_ocean.mp4", 1:Nt; framerate=12) do i
    n[] = i
end

@info "Animation saved."
nothing #hide

# ![](atmosphere_slab_ocean.mp4)
