# # Atmospheric convection over four ocean models
#
# This example demonstrates coupling a Breeze atmospheric large eddy simulation (LES)
# with four different ocean models using NumericalEarth's `EarthSystemModel` framework:
#
# 1. **Prescribed ocean** — constant SST that does not respond to surface fluxes
# 2. **Slab ocean** (10m depth) — a well-mixed layer that responds uniformly to surface fluxes
# 3. **Hydrostatic ocean** (50m depth) — with CATKE turbulent mixing and stratification
# 4. **Nonhydrostatic ocean** (50m depth) — resolved LES turbulence with WENO advection
#
# The atmosphere drives convective turbulence over a warm ocean surface. The coupling
# framework computes turbulent surface fluxes (sensible heat, latent heat, and momentum)
# using Monin--Obukhov similarity theory. These fluxes cool the ocean and heat
# the atmosphere, creating a two-way feedback loop.
#
# By comparing the four models, we can see how ocean vertical mixing and stratification
# affect the SST response to atmospheric forcing — or in the prescribed case, how the
# atmosphere evolves when the SST is held fixed.

using NumericalEarth
using Breeze
using Oceananigans
using Oceananigans.Units
using Printf
using Statistics: mean

# ## Grid setup
#
# We use a 2D domain in the x-z plane: 20 km wide and 10 km tall with
# 64 × 64 grid points. The `Periodic` x-topology allows convective cells
# to wrap around, and `Flat` y-topology makes this a 2D simulation.

Nxᵃᵗ = 64 # Atmosphere horizontal resolution (shared with ocean)
Nzᵃᵗ = 64 # Atmosphere vertical resolution
Nzᵒᶜ = 20 # Hydrostatic ocean vertical resolution
Nzⁿʰ = 50 # Nonhydrostatic ocean vertical resolution (1m spacing)

grid = RectilinearGrid(size = (Nxᵃᵗ, Nzᵃᵗ), halo = (5, 5),
                       x = (-10kilometers, 10kilometers),
                       z = (0, 10kilometers),
                       topology = (Periodic, Flat, Bounded))

# ## Four independent atmospheres
#
# Each coupled model needs its own Breeze atmosphere instance because the
# `EarthSystemModel` writes boundary conditions into the atmosphere.
# All are initialized identically.

Tᵒᶜ = 290 # K
θᵃᵗ = 250 # K
U₀ = 10 # m/s
coriolis = FPlane(latitude=33)

prescribed_ocean_atmos = atmosphere_simulation(grid; potential_temperature=θᵃᵗ, coriolis)
slab_ocean_atmos       = atmosphere_simulation(grid; potential_temperature=θᵃᵗ, coriolis)
full_ocean_atmos       = atmosphere_simulation(grid; potential_temperature=θᵃᵗ, coriolis)
nh_ocean_atmos         = atmosphere_simulation(grid; potential_temperature=θᵃᵗ, coriolis)

# ## Atmospheric initial conditions
#
# We initialize all atmospheres with the reference potential temperature profile
# plus small random perturbations below 500 m. These perturbations seed convective
# instability, which develops into turbulent convection driven by surface heat fluxes.
# A background zonal wind `U₀` provides a nonzero wind speed for the
# similarity theory flux computation.

reference_state = slab_ocean_atmos.dynamics.reference_state

θᵢ(x, z) = reference_state.potential_temperature + 0.1 * randn() * (z < 500)
set!(prescribed_ocean_atmos, θ=θᵢ, u=U₀)
set!(slab_ocean_atmos,       θ=θᵢ, u=U₀)
set!(full_ocean_atmos,       θ=θᵢ, u=U₀)
set!(nh_ocean_atmos,         θ=θᵢ, u=U₀)

# ## Prescribed ocean (constant SST)
#
# The prescribed ocean holds a fixed temperature that does not evolve.
# Surface fluxes are still computed (so the atmosphere feels the ocean),
# but the ocean temperature is pinned. Its temperature is in Kelvin.

sst_grid = RectilinearGrid(grid.architecture,
                           size = grid.Nx,
                           halo = grid.Hx,
                           x = (-10kilometers, 10kilometers),
                           topology = (Periodic, Flat, Flat))

prescribed_ocean = PrescribedOcean(sst_grid)
set!(prescribed_ocean, T=Tᵒᶜ)

# ## Slab ocean (10m depth)
#
# The slab ocean represents a well-mixed ocean layer of fixed depth.
# Its temperature is in Kelvin (the coupling framework handles the conversion).

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

ocean_grid = RectilinearGrid(grid.architecture,
                             size = (grid.Nx, Nzᵒᶜ),
                             halo = (grid.Hx, 5),
                             x = (-10kilometers, 10kilometers),
                             z = (-50, 0),
                             topology = (Periodic, Flat, Bounded))

ocean = ocean_simulation(ocean_grid; coriolis,
                         closure = CATKEVerticalDiffusivity(), # note ocean_simulation default does not work.
                         momentum_advection = nothing,
                         tracer_advection = nothing,
                         Δt = 2,
                         warn = false)

celsius_to_kelvin = 273.15
T₀ = Tᵒᶜ - celsius_to_kelvin               # surface temperature in °C
Tᵢ(x, z) = T₀ + (z + 10) / 50 * (z < -10)  # linear stratification below 10m
set!(ocean.model, T=Tᵢ, S=35)

# ## Nonhydrostatic ocean LES (50m depth)
#
# The nonhydrostatic ocean uses a `NonhydrostaticModel` that resolves the full
# 3D pressure field. With 1m vertical resolution and WENO(order=9) advection,
# it performs implicit LES. No turbulence closure is needed.

nh_ocean_grid = RectilinearGrid(grid.architecture,
                                size = (grid.Nx, Nzⁿʰ),
                                halo = (grid.Hx, 5),
                                x = (-10kilometers, 10kilometers),
                                z = (-50, 0),
                                topology = (Periodic, Flat, Bounded))

nh_ocean = nonhydrostatic_ocean_simulation(nh_ocean_grid; coriolis, Δt=2)
set!(nh_ocean.model, T=Tᵢ, S=35)

# ## Coupled models
#
# We disable gustiness in the similarity theory flux computation so the surface
# wind speed is determined entirely by the resolved velocity field.

atmosphere_ocean_fluxes = SimilarityTheoryFluxes(gustiness_parameter=0, minimum_gustiness=0)

prescribed_interfaces = ComponentInterfaces(prescribed_ocean_atmos, prescribed_ocean; atmosphere_ocean_fluxes)
slab_interfaces       = ComponentInterfaces(slab_ocean_atmos, slab_ocean; atmosphere_ocean_fluxes)
full_interfaces       = ComponentInterfaces(full_ocean_atmos, ocean; atmosphere_ocean_fluxes)
nh_interfaces         = ComponentInterfaces(nh_ocean_atmos, nh_ocean; atmosphere_ocean_fluxes)

prescribed_model = AtmosphereOceanModel(prescribed_ocean_atmos, prescribed_ocean; interfaces = prescribed_interfaces)
slab_model       = AtmosphereOceanModel(slab_ocean_atmos, slab_ocean; interfaces = slab_interfaces)
full_model       = AtmosphereOceanModel(full_ocean_atmos, ocean; interfaces = full_interfaces)
nh_model         = AtmosphereOceanModel(nh_ocean_atmos, nh_ocean; interfaces = nh_interfaces)

Δt = 5seconds
stop_time = 4hours
prescribed_sim = Simulation(prescribed_model; Δt, stop_time)
slab_sim       = Simulation(slab_model; Δt, stop_time)
full_sim       = Simulation(full_model; Δt, stop_time)
nh_sim         = Simulation(nh_model; Δt, stop_time)

# ## Progress callbacks

function prescribed_progress(sim)
    atmos = sim.model.atmosphere
    u, v, w = atmos.velocities
    umax = maximum(abs, u)
    wmax = maximum(abs, w)
    msg = @sprintf("[Prescribed] Iter: %d, t = %s, max|u|: %.2e, max|w|: %.2e, SST: %.3f (fixed)",
                    iteration(sim), prettytime(sim), umax, wmax, Tᵒᶜ)
    @info msg
    return nothing
end

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

function nh_progress(sim)
    atmos = sim.model.atmosphere
    u, v, w = atmos.velocities
    umax = maximum(abs, u)
    wmax = maximum(abs, w)
    T = sim.model.ocean.model.tracers.T
    Nz = size(sim.model.ocean.model.grid, 3)
    sst_surface = view(interior(T), :, 1, Nz)
    sst_min = minimum(sst_surface)
    sst_max = maximum(sst_surface)
    msg = @sprintf("[NH] Iter: %d, t = %s, max|u|: %.2e, max|w|: %.2e, SST: (%.3f, %.3f)",
                    iteration(sim), prettytime(sim), umax, wmax, sst_min, sst_max)
    @info msg
    return nothing
end

add_callback!(prescribed_sim, prescribed_progress, IterationInterval(400))
add_callback!(slab_sim,       slab_progress,       IterationInterval(400))
add_callback!(full_sim,       full_progress,       IterationInterval(400))
add_callback!(nh_sim,         nh_progress,         IterationInterval(400))

# ## Output writers
#
# * Prescribed simulation: atmospheric θ and u (SST is constant, no need to save).
# * Slab simulation: atmospheric θ and u, plus slab SST.
# * Full simulation: atmospheric θ, u, cloud water, and w, plus full-depth ocean temperature T.
# * Nonhydrostatic simulation: atmospheric θ, u, cloud water, and w, plus full-depth ocean temperature T.

u_p, v_p, w_p = prescribed_ocean_atmos.velocities
θ_p = liquid_ice_potential_temperature(prescribed_ocean_atmos)

prescribed_sim.output_writers[:atmos] = JLD2Writer(prescribed_model, (; θ=θ_p, u=u_p),
                                                   filename = "prescribed_ocean_atmos",
                                                   schedule = TimeInterval(1minute),
                                                   overwrite_existing = true)

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

u_nh, v_nh, w_nh = nh_ocean_atmos.velocities
θ_nh = liquid_ice_potential_temperature(nh_ocean_atmos)
qˡ_nh = nh_ocean_atmos.microphysical_fields.qˡ

nh_sim.output_writers[:atmos] = JLD2Writer(nh_model, (; θ=θ_nh, u=u_nh, qˡ=qˡ_nh, w=w_nh),
                                           filename = "nh_ocean_atmos",
                                           schedule = TimeInterval(1minute),
                                           overwrite_existing = true)

nh_sim.output_writers[:ocean] = JLD2Writer(nh_model, (; T=nh_ocean.model.tracers.T),
                                           filename = "ocean_nh",
                                           schedule = TimeInterval(1minute),
                                           overwrite_existing = true)

# ## Run all four simulations sequentially

@info "Running prescribed ocean coupled simulation..."
run!(prescribed_sim)
@info "Prescribed ocean simulation complete."

@info "Running slab ocean coupled simulation..."
run!(slab_sim)
@info "Slab ocean simulation complete."

@info "Running full ocean coupled simulation..."
run!(full_sim)
@info "Full ocean simulation complete."

@info "Running nonhydrostatic ocean coupled simulation..."
run!(nh_sim)
@info "Nonhydrostatic ocean simulation complete."

# ## Animation
#
# The animation has five columns:
# - **Column 1**: atmosphere over prescribed ocean (θ, u) and SST comparison line plot
# - **Column 2**: atmosphere over slab ocean (θ, u) — same layout
# - **Column 3**: atmosphere over hydrostatic ocean (cloud water, w) and ocean T cross-section
# - **Column 4**: atmosphere over nonhydrostatic ocean (cloud water, w) and ocean T cross-section
# - **Column 5** (narrow): horizontal-mean vertical profiles from all four simulations
#
# The SST comparison converts the full/NH ocean surface temperature from °C to K
# so all curves are on the same Kelvin scale as the slab and prescribed oceans.

using CairoMakie

θ_prescribed_ts = FieldTimeSeries("prescribed_ocean_atmos.jld2", "θ"; grid)
u_prescribed_ts = FieldTimeSeries("prescribed_ocean_atmos.jld2", "u"; grid)

θ_slab_ts = FieldTimeSeries("slab_ocean_atmos.jld2", "θ"; grid)
u_slab_ts = FieldTimeSeries("slab_ocean_atmos.jld2", "u"; grid)
sst_slab_ts = FieldTimeSeries("sst_slab.jld2", "SST"; grid=sst_grid)

θ_full_ts = FieldTimeSeries("full_ocean_atmos.jld2", "θ"; grid)
u_full_ts = FieldTimeSeries("full_ocean_atmos.jld2", "u"; grid)
qˡ_full_ts = FieldTimeSeries("full_ocean_atmos.jld2", "qˡ"; grid)
w_full_ts = FieldTimeSeries("full_ocean_atmos.jld2", "w"; grid)
T_ocean_ts = FieldTimeSeries("ocean_full.jld2", "T"; grid=ocean_grid)

θ_nh_ts = FieldTimeSeries("nh_ocean_atmos.jld2", "θ"; grid)
u_nh_ts = FieldTimeSeries("nh_ocean_atmos.jld2", "u"; grid)
qˡ_nh_ts = FieldTimeSeries("nh_ocean_atmos.jld2", "qˡ"; grid)
w_nh_ts = FieldTimeSeries("nh_ocean_atmos.jld2", "w"; grid)
T_nh_ts = FieldTimeSeries("ocean_nh.jld2", "T"; grid=nh_ocean_grid)

times = θ_slab_ts.times
Nt = length(times)
Nzᵒᶜ = size(ocean_grid, 3)
Nzⁿʰ = size(nh_ocean_grid, 3)

# Coordinate arrays for manual line plots.

x_ocean = xnodes(ocean_grid, Center())
x_nh = xnodes(nh_ocean_grid, Center())

# ### Figure layout

fig = Figure(size = (2400, 900), fontsize = 12)

ax_θ_p = Axis(fig[1, 1], title="θₗᵢ (K) — atmos (prescribed)", ylabel="z (m)")
ax_u_p = Axis(fig[2, 1], title="u (m/s) — atmos (prescribed)", ylabel="z (m)")
ax_sst = Axis(fig[3, 1], title="SST (K)",                       xlabel="x (m)", ylabel="SST (K)")

ax_θ   = Axis(fig[1, 2], title="θₗᵢ (K) — atmos (slab)", ylabel="z (m)")
ax_u   = Axis(fig[2, 2], title="u (m/s) — atmos (slab)", ylabel="z (m)")
ax_oT  = Axis(fig[3, 2], title="Ocean T (°C) — hydrostatic", xlabel="x (m)", ylabel="z (m)")

ax_qˡ = Axis(fig[1, 3], title="Cloud water — atmos (hydrostatic)", ylabel="z (m)")
ax_w  = Axis(fig[2, 3], title="w (m/s) — atmos (hydrostatic)",     ylabel="z (m)")
ax_oT_nh = Axis(fig[3, 3], title="Ocean T (°C) — NH",              xlabel="x (m)", ylabel="z (m)")

ax_qˡ_nh = Axis(fig[1, 4], title="Cloud water — atmos (NH)", ylabel="z (m)")
ax_w_nh  = Axis(fig[2, 4], title="w (m/s) — atmos (NH)",     ylabel="z (m)")
ax_Tp    = Axis(fig[3, 4], title="⟨T⟩(z)", xlabel="T (°C)",  ylabel="z (m)")

ax_θp = Axis(fig[1, 5], title="⟨θ⟩(z)", xlabel="θ (K)",   ylabel="z (m)", limits=((θᵃᵗ-1, θᵃᵗ+4), nothing))
ax_up = Axis(fig[2, 5], title="⟨u⟩(z)", xlabel="u (m/s)", ylabel="z (m)", limits=((-10, 25), nothing))

colsize!(fig.layout, 5, Relative(0.10))

for ax in (ax_θ_p, ax_u_p, ax_θ, ax_u, ax_qˡ, ax_w, ax_qˡ_nh, ax_w_nh)
    hidexdecorations!(ax, ticks=false)
end

# ### Observables

n = Observable(1)

# Column 1 — prescribed atmosphere
θn_p = @lift θ_prescribed_ts[$n]
un_p = @lift u_prescribed_ts[$n]

# Column 2 — slab atmosphere
θn = @lift θ_slab_ts[$n]
un = @lift u_slab_ts[$n]
sstn_slab = @lift sst_slab_ts[$n]

# SST comparison (convert °C to K for full/NH)
ocean_sst_kelvin = @lift interior(T_ocean_ts[$n], :, 1, Nzᵒᶜ) .+ celsius_to_kelvin
nh_sst_kelvin    = @lift interior(T_nh_ts[$n], :, 1, Nzⁿʰ) .+ celsius_to_kelvin

# Column 3 — hydrostatic atmosphere
qˡn = @lift qˡ_full_ts[$n]
wn  = @lift w_full_ts[$n]
oTn = @lift T_ocean_ts[$n]

# Column 4 — nonhydrostatic atmosphere
qˡn_nh = @lift qˡ_nh_ts[$n]
wn_nh  = @lift w_nh_ts[$n]
oTn_nh = @lift T_nh_ts[$n]

# Profile column — horizontal-mean profiles
θ_avg_prescribed = @lift Field(Average(θ_prescribed_ts[$n], dims=1))
θ_avg_slab       = @lift Field(Average(θ_slab_ts[$n], dims=1))
θ_avg_full       = @lift Field(Average(θ_full_ts[$n], dims=1))
θ_avg_nh         = @lift Field(Average(θ_nh_ts[$n], dims=1))
u_avg_prescribed = @lift Field(Average(u_prescribed_ts[$n], dims=1))
u_avg_slab       = @lift Field(Average(u_slab_ts[$n], dims=1))
u_avg_full       = @lift Field(Average(u_full_ts[$n], dims=1))
u_avg_nh         = @lift Field(Average(u_nh_ts[$n], dims=1))
T_avg_ocean      = @lift Field(Average(T_ocean_ts[$n], dims=1))
T_avg_nh         = @lift Field(Average(T_nh_ts[$n], dims=1))
# Convert slab SST from K to °C for the ocean T profile comparison
sst_avg_celsius        = @lift fill(mean(sst_slab_ts[$n]) - celsius_to_kelvin, 2)
prescribed_avg_celsius = fill(Tᵒᶜ - celsius_to_kelvin, 2) # constant

# ### Plot

heatmap!(ax_θ_p, θn_p; colormap=:thermal, colorrange=(θᵃᵗ - 1, θᵃᵗ + 3))
heatmap!(ax_u_p, un_p; colormap=:balance,  colorrange=(-30, 30))

heatmap!(ax_θ,  θn;  colormap=:thermal,          colorrange=(θᵃᵗ - 1, θᵃᵗ + 3))
heatmap!(ax_u,  un;  colormap=:balance,          colorrange=(-30, 30))
heatmap!(ax_oT, oTn; colormap=:thermal,          colorrange=(T₀ - 1.5, T₀ + 0.5))

heatmap!(ax_qˡ, qˡn; colormap=Reverse(:Blues_4), colorrange=(0, 5e-4))
heatmap!(ax_w,  wn;  colormap=:balance,          colorrange=(-25, 25))
heatmap!(ax_oT_nh, oTn_nh; colormap=:thermal,    colorrange=(T₀ - 1.5, T₀ + 0.5))

heatmap!(ax_qˡ_nh, qˡn_nh; colormap=Reverse(:Blues_4), colorrange=(0, 5e-4))
heatmap!(ax_w_nh,  wn_nh;  colormap=:balance,          colorrange=(-25, 25))

hlines!(ax_sst, Tᵒᶜ;                       color=:black, linewidth=2, label="Prescribed")
lines!(ax_sst, sstn_slab;                  color=:red,   linewidth=2, label="Slab (10m)")
lines!(ax_sst, x_ocean, ocean_sst_kelvin;  color=:blue,  linewidth=2, label="Hydrostatic")
lines!(ax_sst, x_nh, nh_sst_kelvin;        color=:green, linewidth=2, label="NH")
axislegend(ax_sst, position=:rb)
ylims!(ax_sst, Tᵒᶜ - 0.7, Tᵒᶜ + 0.2)

lines!(ax_θp, θ_avg_prescribed; color=:black, linewidth=1.5, label="Prescribed")
lines!(ax_θp, θ_avg_slab;       color=:red,   linewidth=1.5, label="Slab")
lines!(ax_θp, θ_avg_full;       color=:blue,  linewidth=1.5, label="Hydro")
lines!(ax_θp, θ_avg_nh;         color=:green, linewidth=1.5, label="NH")
axislegend(ax_θp, position=:rt)

lines!(ax_up, u_avg_prescribed; color=:black, linewidth=1.5)
lines!(ax_up, u_avg_slab;       color=:red,   linewidth=1.5)
lines!(ax_up, u_avg_full;       color=:blue,  linewidth=1.5)
lines!(ax_up, u_avg_nh;         color=:green, linewidth=1.5)

lines!(ax_Tp, T_avg_ocean;                              color=:blue,  linewidth=1.5, label="Hydro")
lines!(ax_Tp, T_avg_nh;                                 color=:green, linewidth=1.5, label="NH")
lines!(ax_Tp, sst_avg_celsius,        [-50.0, 0.0];     color=:red,   linewidth=1.5, label="Slab")
lines!(ax_Tp, prescribed_avg_celsius,  [-50.0, 0.0];    color=:black, linewidth=1.5, label="Prescribed")
axislegend(ax_Tp, position=:lb)
xlims!(ax_Tp, T₀ - 1, T₀ + 0.5)

title = @lift "Atmosphere–ocean coupling comparison, t = " * prettytime(times[$n])
Label(fig[0, 1:5], title, fontsize=16)

# ### Record

@info "Rendering animation..."
record(fig, "breeze_over_four_oceans.mp4", 1:Nt; framerate=12) do nn
    n[] = nn
end

@info "Animation saved."
nothing #hide

# ![](breeze_over_four_oceans.mp4)
