# # Atmospheric convection over a slab ocean vs a hydrostatic ocean
#
# This example demonstrates coupling a Breeze atmospheric large eddy simulation (LES)
# with two different ocean models using NumericalEarth's `EarthSystemModel` framework:
#
# 1. **Slab ocean** (10m depth) — a well-mixed layer that responds uniformly to surface fluxes
# 2. **Full hydrostatic ocean** (50m depth) — with CATKE turbulent mixing and stratification
#
# The atmosphere drives convective turbulence over a warm ocean surface. The coupling
# framework computes turbulent surface fluxes (sensible heat, latent heat, and momentum)
# using Monin--Obukhov similarity theory. These fluxes cool the ocean and heat
# the atmosphere, creating a two-way feedback loop.
#
# By comparing the two models, we can see how ocean vertical mixing and stratification
# affect the SST response to atmospheric forcing.

using NumericalEarth
using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using Statistics: mean

# ## Grid setup
#
# We use a 2D domain in the x-z plane: 20 km wide and 10 km tall with
# 128 × 128 grid points. The `Periodic` x-topology allows convective cells
# to wrap around, and `Flat` y-topology makes this a 2D simulation.

grid = RectilinearGrid(size = (128, 128), halo = (5, 5),
                       x = (-10kilometers, 10kilometers),
                       z = (0, 10kilometers),
                       topology = (Periodic, Flat, Bounded))

# ## Two independent atmospheres
#
# Each coupled model needs its own Breeze atmosphere instance because the
# `EarthSystemModel` writes boundary conditions into the atmosphere.
# Both are initialized identically.

Tᵒᶜ = 290 # K
θᵃᵗ = 250 # K
U₀ = 10 # m/s
coriolis = FPlane(latitude=33)

slab_ocean_atmos = atmosphere_simulation(grid; potential_temperature=θᵃᵗ, coriolis)
full_ocean_atmos = atmosphere_simulation(grid; potential_temperature=θᵃᵗ, coriolis)

# ## Atmospheric initial conditions
#
# We initialize both atmospheres with the reference potential temperature profile
# plus small random perturbations below 500 m. These perturbations seed convective
# instability, which develops into turbulent convection driven by surface heat fluxes.
# A background zonal wind `U₀` provides a nonzero wind speed for the
# similarity theory flux computation.

reference_state = slab_ocean_atmos.dynamics.reference_state

θᵢ(x, z) = reference_state.potential_temperature + 0.1 * randn() * (z < 500)
set!(slab_ocean_atmos, θ=θᵢ, u=U₀)
set!(full_ocean_atmos, θ=θᵢ, u=U₀)

# ## Slab ocean (10m depth)
#
# The slab ocean represents a well-mixed ocean layer of fixed depth.
# Its temperature is in Kelvin (the coupling framework handles the conversion).

sst_grid = RectilinearGrid(grid.architecture,
                           size = grid.Nx,
                           halo = grid.Hx,
                           x = (-10kilometers, 10kilometers),
                           topology = (Periodic, Flat, Flat))

slab_ocean = SlabOcean(sst_grid, depth=10)
set!(slab_ocean, T=Tᵒᶜ)

# ## Full hydrostatic ocean (50m depth with CATKE mixing)
#
# The full ocean uses a `HydrostaticFreeSurfaceModel` with the default TEOS-10
# equation of state and CATKE vertical mixing parameterization. The grid has 20
# vertical levels (2.5m vertical resolution). We disable advection since this is
# primarily a 1D vertical mixing problem.
#
# Since TEOS-10 expects temperature in degrees Celsius, the ocean temperature
# is initialized accordingly. The coupling framework automatically converts
# from Celsius to Kelvin for the flux computation.

Nz_ocean = 20

ocean_grid = RectilinearGrid(grid.architecture,
                             size = (grid.Nx, Nz_ocean),
                             halo = (grid.Hx, 5),
                             x = (-10kilometers, 10kilometers),
                             z = (-50, 0),
                             topology = (Periodic, Flat, Bounded))

ocean = ocean_simulation(ocean_grid; coriolis,
                         momentum_advection = nothing,
                         tracer_advection = nothing,
                         Δt = 2,
                         warn = false)

celsius_to_kelvin = 273.15
T₀ = Tᵒᶜ - celsius_to_kelvin                 # surface temperature in °C
Tᵢ(x, z) = z > -10 ? T₀ : T₀ + (z + 10) /50  # linear cooling below 10m
set!(ocean.model, T=Tᵢ, S=35)

# ## Coupled models
#
# We disable gustiness in the similarity theory flux computation so the surface
# wind speed is determined entirely by the resolved velocity field.

atmosphere_ocean_fluxes = SimilarityTheoryFluxes(gustiness_parameter=0, minimum_gustiness=0)

slab_interfaces = ComponentInterfaces(slab_ocean_atmos, slab_ocean; atmosphere_ocean_fluxes)
full_interfaces = ComponentInterfaces(full_ocean_atmos, ocean; atmosphere_ocean_fluxes)

slab_model = AtmosphereOceanModel(slab_ocean_atmos, slab_ocean; interfaces = slab_interfaces)
full_model = AtmosphereOceanModel(full_ocean_atmos, ocean; interfaces = full_interfaces)

Δt = 2seconds
stop_time = 4hours
slab_sim = Simulation(slab_model; Δt, stop_time)
full_sim = Simulation(full_model; Δt, stop_time)

# ## Progress callbacks

function slab_progress(sim)
    atmos = sim.model.atmosphere
    u, v, w = atmos.velocities
    umax = maximum(abs, u)
    wmax = maximum(abs, w)
    sst = sim.model.ocean.temperature
    sst_min = minimum(sst)
    sst_max = maximum(sst)
    msg = @sprintf("[Slab] Iter: %d, t = %s, max|u|: %.2e, max|w|: %.2e, SST: (%.3f, %.3f)",
                    iteration(sim), prettytime(sim), umax, wmax, sst_min, sst_max)
    @info msg
    return nothing
end

function full_progress(sim)
    atmos = sim.model.atmosphere
    u, v, w = atmos.velocities
    umax = maximum(abs, u)
    wmax = maximum(abs, w)
    T = sim.model.ocean.model.tracers.T
    Nz = size(sim.model.ocean.model.grid, 3)
    sst_surface = view(interior(T), :, 1, Nz)
    sst_min = minimum(sst_surface)
    sst_max = maximum(sst_surface)
    msg = @sprintf("[Full] Iter: %d, t = %s, max|u|: %.2e, max|w|: %.2e, SST: (%.3f, %.3f)",
                    iteration(sim), prettytime(sim), umax, wmax, sst_min, sst_max)
    @info msg
    return nothing
end

add_callback!(slab_sim, slab_progress, IterationInterval(600))
add_callback!(full_sim, full_progress, IterationInterval(600))

# ## Output writers
#
# * Slab simulation: atmospheric θ and u (for heatmaps and profiles), plus slab SST.
# * Full simulation: atmospheric θ, u, cloud water, and w, plus full-depth ocean temperature T.

u_s, v_s, w_s = slab_ocean_atmos.velocities
θ_s = liquid_ice_potential_temperature(slab_ocean_atmos)

slab_sim.output_writers[:atmos] = JLD2Writer(slab_model, (; θ=θ_s, u=u_s),
                                             filename = "slab_ocean_atmos",
                                             schedule = TimeInterval(1minute),
                                             overwrite_existing = true)

slab_sim.output_writers[:sst] = JLD2Writer(slab_model, (; SST=slab_ocean.temperature),
                                           filename = "sst_slab",
                                           schedule = TimeInterval(1minute),
                                           overwrite_existing = true)

u_f, v_f, w_f = full_ocean_atmos.velocities
θ_f = liquid_ice_potential_temperature(full_ocean_atmos)
qˡ_f = full_ocean_atmos.microphysical_fields.qˡ

full_sim.output_writers[:atmos] = JLD2Writer(full_model, (; θ=θ_f, u=u_f, qˡ=qˡ_f, w=w_f),
                                             filename = "full_ocean_atmos",
                                             schedule = TimeInterval(1minute),
                                             overwrite_existing = true)

full_sim.output_writers[:ocean] = JLD2Writer(full_model, (; T=ocean.model.tracers.T),
                                             filename = "ocean_full",
                                             schedule = TimeInterval(1minute),
                                             overwrite_existing = true)

# ## Run both simulations sequentially

@info "Running slab ocean coupled simulation..."
run!(slab_sim)
@info "Slab ocean simulation complete."

@info "Running full ocean coupled simulation..."
run!(full_sim)
@info "Full ocean simulation complete."

# ## Animation
#
# The animation has three columns:
# - **Left**: atmosphere over slab ocean (θ, u) and SST comparison line plot
# - **Middle**: atmosphere over full ocean (cloud water, w) and ocean temperature cross-section
# - **Right** (narrow): horizontal-mean vertical profiles from both simulations
#
# The SST comparison converts the full ocean surface temperature from °C to K
# so both curves are on the same Kelvin scale as the slab ocean.

using CairoMakie

θ_slab_ts = FieldTimeSeries("slab_ocean_atmos.jld2", "θ"; grid)
u_slab_ts = FieldTimeSeries("slab_ocean_atmos.jld2", "u"; grid)
sst_slab_ts = FieldTimeSeries("sst_slab.jld2", "SST"; grid=sst_grid)

θ_full_ts = FieldTimeSeries("full_ocean_atmos.jld2", "θ"; grid)
u_full_ts = FieldTimeSeries("full_ocean_atmos.jld2", "u"; grid)
qˡ_full_ts = FieldTimeSeries("full_ocean_atmos.jld2", "qˡ"; grid)
w_full_ts = FieldTimeSeries("full_ocean_atmos.jld2", "w"; grid)
T_ocean_ts = FieldTimeSeries("ocean_full.jld2", "T"; grid=ocean_grid)

times = θ_slab_ts.times
Nt = length(times)
Nz_ocean = size(ocean_grid, 3)

# Coordinate arrays for manual line plots.

x_ocean = xnodes(ocean_grid, Center())

# ### Figure layout

fig = Figure(size = (1600, 900), fontsize = 12)

ax_θ   = Axis(fig[1, 1], title="θₗᵢ (K) — atmos (slab ocean)", ylabel="z (km)")
ax_u   = Axis(fig[2, 1], title="u (m/s) — atmos (slab ocean)", ylabel="z (km)")
ax_sst = Axis(fig[3, 1], title="SST (K)",                      xlabel="x (km)", ylabel="SST (K)")

ax_qˡ = Axis(fig[1, 2], title="Cloud water (kg/kg) — atmos (full ocean)", ylabel="z (km)")
ax_w  = Axis(fig[2, 2], title="w (m/s) — atmos (full ocean)",             ylabel="z (km)")
ax_oT = Axis(fig[3, 2], title="Ocean T (°C)",            xlabel="x (km)", ylabel="z (km)")

ax_θp = Axis(fig[1, 3], title="⟨θ⟩(z)", xlabel="θ (K)",   ylabel="z (km)", limits=((θᵃᵗ-1, θᵃᵗ+4), nothing))
ax_up = Axis(fig[2, 3], title="⟨u⟩(z)", xlabel="u (m/s)", ylabel="z (km)", limits=((-10, 25), nothing))
ax_Tp = Axis(fig[3, 3], title="⟨T⟩(z)", xlabel="T (°C)",  ylabel="z (km)")

colsize!(fig.layout, 3, Relative(0.15))

for ax in (ax_θ, ax_u, ax_qˡ, ax_w)
    hidexdecorations!(ax, ticks=false)
end

# ### Observables

n = Observable(1)

# Left column
θn = @lift θ_slab_ts[$n]
un = @lift u_slab_ts[$n]
sstn_slab = @lift sst_slab_ts[$n]
# Convert full ocean surface T from °C to K for the SST comparison
ocean_sst_kelvin = @lift interior(T_ocean_ts[$n], :, 1, Nz_ocean) .+ celsius_to_kelvin

# Middle column
qˡn = @lift qˡ_full_ts[$n]
wn  = @lift w_full_ts[$n]
oTn = @lift T_ocean_ts[$n]

# Right column — horizontal-mean profiles
θ_avg_slab = @lift Field(Average(θ_slab_ts[$n], dims=1))
θ_avg_full = @lift Field(Average(θ_full_ts[$n], dims=1))
u_avg_slab = @lift Field(Average(u_slab_ts[$n], dims=1))
u_avg_full = @lift Field(Average(u_full_ts[$n], dims=1))
T_avg_ocean = @lift Field(Average(T_ocean_ts[$n], dims=1))
# Convert slab SST from K to °C for the ocean T profile comparison
sst_avg_celsius = @lift fill(mean(sst_slab_ts[$n]) - celsius_to_kelvin, 2)

# ### Plot

heatmap!(ax_θ,  θn;  colormap=:thermal,          colorrange=(θᵃᵗ - 1, θᵃᵗ + 3))
heatmap!(ax_u,  un;  colormap=:balance,          colorrange=(-25, 25))
heatmap!(ax_qˡ, qˡn; colormap=Reverse(:Blues_4), colorrange=(0, 5e-4))
heatmap!(ax_w,  wn;  colormap=:balance,          colorrange=(-20, 20))
heatmap!(ax_oT, oTn; colormap=:thermal,          colorrange=(T₀ - 1.5, T₀ + 0.5))

lines!(ax_sst, sstn_slab;                 color=:red,  linewidth=2, label="Slab (10m)")
lines!(ax_sst, x_ocean, ocean_sst_kelvin; color=:blue, linewidth=2, label="Full (CATKE)")
axislegend(ax_sst, position=:rb)
ylims!(ax_sst, Tᵒᶜ - 0.7, Tᵒᶜ + 0.2)

lines!(ax_θp, θ_avg_slab; color=:red,  linewidth=1.5, label="Slab")
lines!(ax_θp, θ_avg_full; color=:blue, linewidth=1.5, label="Full")
axislegend(ax_θp, position=:rt)

lines!(ax_up, u_avg_slab; color=:red,  linewidth=1.5)
lines!(ax_up, u_avg_full; color=:blue, linewidth=1.5)

lines!(ax_Tp, T_avg_ocean;                   color=:blue, linewidth=1.5, label="Full")
lines!(ax_Tp, sst_avg_celsius, [-50.0, 0.0]; color=:red,  linewidth=1.5, label="Slab")
xlims!(ax_Tp, T₀ - 1, T₀ + 0.5)

title = @lift "Atmosphere–ocean coupling comparison, t = " * prettytime(times[$n])
Label(fig[0, 1:3], title, fontsize=16)

# ### Record

@info "Rendering animation..."
record(fig, "breeze_over_two_oceans.mp4", 1:Nt; framerate=12) do nn
    n[] = nn
end

@info "Animation saved."
nothing #hide

# ![](breeze_over_two_oceans.mp4)
