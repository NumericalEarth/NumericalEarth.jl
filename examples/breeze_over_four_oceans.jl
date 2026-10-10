# # Atmospheric convection over four ocean models
#
# This example demonstrates coupling a Breeze atmospheric large eddy simulation (LES)
# with four different ocean models using NumericalEarth's `EarthSystemModel` framework:
#
# 1. **Prescribed ocean** — constant SST that does not respond to surface fluxes
# 2. **Slab ocean** (10 m deep) — a well-mixed layer that responds uniformly to surface fluxes
# 3. **Hydrostatic ocean** (50 m deep) — with CATKE turbulent mixing and stratification
# 4. **Nonhydrostatic ocean** (50 m deep) — resolved LES turbulence with WENO advection
#
# The atmosphere drives convective turbulence over a warm ocean surface. The coupling
# framework computes turbulent surface fluxes (sensible heat, latent heat, and momentum)
# using Monin–Obukhov similarity theory. These fluxes cool the ocean and heat
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
# We use a two-dimensional domain in the x-z plane, 20 km wide and 10 km tall, with
# 64 × 64 grid points. The `Periodic` x-topology lets convective cells
# wrap around, and the `Flat` y-topology makes the simulation two-dimensional.

Nxᵃᵗ = 64 # Atmosphere horizontal resolution (shared with ocean)
Nzᵃᵗ = 64 # Atmosphere vertical resolution
Nzᵒᶜ = 20 # Hydrostatic ocean vertical resolution
Nzⁿʰ = 50 # Nonhydrostatic ocean vertical resolution (1 m spacing)

grid = RectilinearGrid(size = (Nxᵃᵗ, Nzᵃᵗ), halo = (5, 5),
                       x = (-10kilometers, 10kilometers),
                       z = (0, 10kilometers),
                       topology = (Periodic, Flat, Bounded))

# ## Four independent atmospheres
#
# Each coupled model needs its own Breeze atmosphere because the
# `EarthSystemModel` writes boundary conditions into it.
# All four atmospheres are initialized identically.

Tᵒᶜ = 290 # K
θᵃᵗ = 250 # K
U₀ = 10 # m/s
coriolis = FPlane(latitude=33)

reference_state = ReferenceState(grid, ThermodynamicConstants(); potential_temperature=θᵃᵗ)
dynamics = AnelasticDynamics(reference_state)

prescribed_ocean_atmosphere     = atmosphere_simulation(grid; dynamics, coriolis)
slab_ocean_atmosphere           = atmosphere_simulation(grid; dynamics, coriolis)
hydrostatic_ocean_atmosphere    = atmosphere_simulation(grid; dynamics, coriolis)
nonhydrostatic_ocean_atmosphere = atmosphere_simulation(grid; dynamics, coriolis)

# ## Atmospheric initial conditions
#
# We initialize all atmospheres with the reference potential temperature profile
# plus small random perturbations below 500 m. These perturbations seed convective
# instability, which develops into turbulent convection driven by surface heat fluxes.
# A background zonal wind `U₀` provides a nonzero wind speed for the
# similarity theory flux computation.

θᵢ(x, z) = reference_state.potential_temperature + 0.1 * randn() * (z < 500)
set!(prescribed_ocean_atmosphere.model,     θ=θᵢ, u=U₀)
set!(slab_ocean_atmosphere.model,           θ=θᵢ, u=U₀)
set!(hydrostatic_ocean_atmosphere.model,    θ=θᵢ, u=U₀)
set!(nonhydrostatic_ocean_atmosphere.model, θ=θᵢ, u=U₀)

# ## Prescribed ocean (constant SST)
#
# The prescribed ocean holds a fixed temperature, in Kelvin, that does not evolve.
# Surface fluxes are still computed, so the atmosphere feels the ocean,
# but the ocean temperature does not respond to them.

sst_grid = RectilinearGrid(grid.architecture,
                           size = grid.Nx,
                           halo = grid.Hx,
                           x = (-10kilometers, 10kilometers),
                           topology = (Periodic, Flat, Flat))

prescribed_ocean = PrescribedOcean(sst_grid)
set!(prescribed_ocean, T=Tᵒᶜ)

# ## Slab ocean (10 m deep)
#
# The slab ocean represents a well-mixed ocean layer of fixed depth.
# Its temperature is also in Kelvin.

slab_ocean = SlabOcean(sst_grid, depth=10)
set!(slab_ocean, T=Tᵒᶜ)

# ## Hydrostatic ocean (50 m deep, with CATKE mixing)
#
# The hydrostatic ocean uses a `HydrostaticFreeSurfaceModel` with the default TEOS-10
# equation of state and the CATKE vertical mixing parameterization. The grid has 20
# vertical levels (2.5 m vertical resolution). We disable advection since this is
# primarily a one-dimensional vertical mixing problem.
#
# TEOS-10 expects temperature in degrees Celsius, so we initialize the ocean temperature
# in Celsius. The coupling framework converts it to Kelvin for the flux computation.

hydrostatic_ocean_grid = RectilinearGrid(grid.architecture,
                                         size = (grid.Nx, Nzᵒᶜ),
                                         halo = (grid.Hx, 5),
                                         x = (-10kilometers, 10kilometers),
                                         z = (-50, 0),
                                         topology = (Periodic, Flat, Bounded))

hydrostatic_ocean = ocean_simulation(hydrostatic_ocean_grid; coriolis,
                                     closure = CATKEVerticalDiffusivity(), # the default closure of `ocean_simulation` does not work here
                                     momentum_advection = nothing,
                                     tracer_advection = nothing,
                                     Δt = 2,
                                     warn = false)

celsius_to_kelvin = 273.15
T₀ = Tᵒᶜ - celsius_to_kelvin               # surface temperature in °C
Tᵢ(x, z) = T₀ + (z + 10) / 50 * (z < -10)  # linear stratification below 10 m
set!(hydrostatic_ocean.model, T=Tᵢ, S=35)

# ## Nonhydrostatic ocean LES (50 m deep)
#
# The nonhydrostatic ocean uses a `NonhydrostaticModel`, which solves for the full
# nonhydrostatic pressure. With 1 m vertical resolution and `WENO(order=9)` advection,
# it is an implicit LES that needs no turbulence closure.

nonhydrostatic_ocean_grid = RectilinearGrid(grid.architecture,
                                            size = (grid.Nx, Nzⁿʰ),
                                            halo = (grid.Hx, 5),
                                            x = (-10kilometers, 10kilometers),
                                            z = (-50, 0),
                                            topology = (Periodic, Flat, Bounded))

nonhydrostatic_ocean = ocean_simulation(nonhydrostatic_ocean_grid; model=:nonhydrostatic, coriolis, Δt=2)
set!(nonhydrostatic_ocean.model, T=Tᵢ, S=35)

# ## Coupled models
#
# We disable gustiness in the similarity theory flux computation so the surface
# wind speed is determined entirely by the resolved velocity field.

atmosphere_ocean_fluxes = SimilarityTheoryFluxes(subgrid_velocities = nothing)

prescribed_interfaces     = ComponentInterfaces(prescribed_ocean_atmosphere, prescribed_ocean; atmosphere_ocean_fluxes)
slab_interfaces           = ComponentInterfaces(slab_ocean_atmosphere, slab_ocean; atmosphere_ocean_fluxes)
hydrostatic_interfaces    = ComponentInterfaces(hydrostatic_ocean_atmosphere, hydrostatic_ocean; atmosphere_ocean_fluxes)
nonhydrostatic_interfaces = ComponentInterfaces(nonhydrostatic_ocean_atmosphere, nonhydrostatic_ocean; atmosphere_ocean_fluxes)

prescribed_model     = AtmosphereOceanModel(prescribed_ocean_atmosphere, prescribed_ocean; interfaces = prescribed_interfaces)
slab_model           = AtmosphereOceanModel(slab_ocean_atmosphere, slab_ocean; interfaces = slab_interfaces)
hydrostatic_model    = AtmosphereOceanModel(hydrostatic_ocean_atmosphere, hydrostatic_ocean; interfaces = hydrostatic_interfaces)
nonhydrostatic_model = AtmosphereOceanModel(nonhydrostatic_ocean_atmosphere, nonhydrostatic_ocean; interfaces = nonhydrostatic_interfaces)

Δt = 5seconds
stop_time = 4hours
prescribed_simulation     = Simulation(prescribed_model; Δt, stop_time)
slab_simulation           = Simulation(slab_model; Δt, stop_time)
hydrostatic_simulation    = Simulation(hydrostatic_model; Δt, stop_time)
nonhydrostatic_simulation = Simulation(nonhydrostatic_model; Δt, stop_time)

# ## Progress callbacks
#
# A single progress function logs the atmospheric velocities and the sea surface temperature
# range of each simulation.

function progress(sim, (label, SST))
    u, v, w = sim.model.atmosphere.model.velocities
    msg = @sprintf("[%s] iteration: %d, time: %s, max|u|: %.2e, max|w|: %.2e, SST: (%.3f, %.3f)",
                   label, iteration(sim), prettytime(sim), maximum(abs, u), maximum(abs, w),
                   minimum(SST), maximum(SST))
    @info msg
    return nothing
end

hydrostatic_SST    = view(hydrostatic_ocean.model.tracers.T, :, :, Nzᵒᶜ)
nonhydrostatic_SST = view(nonhydrostatic_ocean.model.tracers.T, :, :, Nzⁿʰ)

add_callback!(prescribed_simulation,     progress, IterationInterval(400), parameters=("Prescribed", Tᵒᶜ))
add_callback!(slab_simulation,           progress, IterationInterval(400), parameters=("Slab", slab_ocean.temperature))
add_callback!(hydrostatic_simulation,    progress, IterationInterval(400), parameters=("Hydrostatic", hydrostatic_SST))
add_callback!(nonhydrostatic_simulation, progress, IterationInterval(400), parameters=("Nonhydrostatic", nonhydrostatic_SST))

# ## Output writers
#
# * Prescribed ocean: atmospheric θ and u (the SST is constant, so there is no need to save it).
# * Slab ocean: atmospheric θ and u, plus the slab SST.
# * Hydrostatic ocean: atmospheric θ, u, cloud liquid water qˡ, and w, plus the ocean temperature T.
# * Nonhydrostatic ocean: atmospheric θ, u, cloud liquid water qˡ, and w, plus the ocean temperature T.

atmosphere = prescribed_ocean_atmosphere.model
outputs = (θ = liquid_ice_potential_temperature(atmosphere), u = atmosphere.velocities.u)
prescribed_simulation.output_writers[:atmos] = JLD2Writer(prescribed_model, outputs,
                                                          filename = "prescribed_ocean_atmos",
                                                          schedule = TimeInterval(1minute),
                                                          overwrite_files = true)

atmosphere = slab_ocean_atmosphere.model
outputs = (θ = liquid_ice_potential_temperature(atmosphere), u = atmosphere.velocities.u)
slab_simulation.output_writers[:atmos] = JLD2Writer(slab_model, outputs,
                                                    filename = "slab_ocean_atmos",
                                                    schedule = TimeInterval(1minute),
                                                    overwrite_files = true)

slab_simulation.output_writers[:sst] = JLD2Writer(slab_model, (; SST=slab_ocean.temperature),
                                                  filename = "sst_slab",
                                                  schedule = TimeInterval(1minute),
                                                  overwrite_files = true)

atmosphere = hydrostatic_ocean_atmosphere.model
outputs = (θ = liquid_ice_potential_temperature(atmosphere),
           u = atmosphere.velocities.u,
           qˡ = atmosphere.microphysical_fields.qˡ,
           w = atmosphere.velocities.w)
hydrostatic_simulation.output_writers[:atmos] = JLD2Writer(hydrostatic_model, outputs,
                                                           filename = "full_ocean_atmos",
                                                           schedule = TimeInterval(1minute),
                                                           overwrite_files = true)

hydrostatic_simulation.output_writers[:ocean] = JLD2Writer(hydrostatic_model, (; T=hydrostatic_ocean.model.tracers.T),
                                                           filename = "ocean_full",
                                                           schedule = TimeInterval(1minute),
                                                           overwrite_files = true)

atmosphere = nonhydrostatic_ocean_atmosphere.model
outputs = (θ = liquid_ice_potential_temperature(atmosphere),
           u = atmosphere.velocities.u,
           qˡ = atmosphere.microphysical_fields.qˡ,
           w = atmosphere.velocities.w)
nonhydrostatic_simulation.output_writers[:atmos] = JLD2Writer(nonhydrostatic_model, outputs,
                                                              filename = "nh_ocean_atmos",
                                                              schedule = TimeInterval(1minute),
                                                              overwrite_files = true)

nonhydrostatic_simulation.output_writers[:ocean] = JLD2Writer(nonhydrostatic_model, (; T=nonhydrostatic_ocean.model.tracers.T),
                                                              filename = "ocean_nh",
                                                              schedule = TimeInterval(1minute),
                                                              overwrite_files = true)

# ## Run all four simulations, one after the other

run!(prescribed_simulation)
run!(slab_simulation)
run!(hydrostatic_simulation)
run!(nonhydrostatic_simulation)

# ## Animation
#
# The animation is laid out as follows:
# - **Columns 1–4, top two rows**: the atmosphere over the prescribed, slab, hydrostatic, and
#   nonhydrostatic oceans. We show θ and u over the prescribed and slab oceans, and cloud
#   liquid water and w over the hydrostatic and nonhydrostatic oceans.
# - **Bottom row**: the SST of all four oceans, the temperature cross-sections of the hydrostatic
#   and nonhydrostatic oceans, and the horizontally averaged ocean temperature profiles.
# - **Column 5** (narrow): horizontally averaged θ and u profiles from all four simulations.
#
# The SST comparison converts the hydrostatic and nonhydrostatic ocean surface temperatures
# from °C to K, so that all curves share the Kelvin scale of the slab and prescribed oceans.

using CairoMakie

θ_prescribed_ts = FieldTimeSeries("prescribed_ocean_atmos.jld2", "θ"; grid)
u_prescribed_ts = FieldTimeSeries("prescribed_ocean_atmos.jld2", "u"; grid)

θ_slab_ts   = FieldTimeSeries("slab_ocean_atmos.jld2", "θ"; grid)
u_slab_ts   = FieldTimeSeries("slab_ocean_atmos.jld2", "u"; grid)
SST_slab_ts = FieldTimeSeries("sst_slab.jld2", "SST"; grid=sst_grid)

θ_hydrostatic_ts  = FieldTimeSeries("full_ocean_atmos.jld2", "θ"; grid)
u_hydrostatic_ts  = FieldTimeSeries("full_ocean_atmos.jld2", "u"; grid)
qˡ_hydrostatic_ts = FieldTimeSeries("full_ocean_atmos.jld2", "qˡ"; grid)
w_hydrostatic_ts  = FieldTimeSeries("full_ocean_atmos.jld2", "w"; grid)
T_hydrostatic_ts  = FieldTimeSeries("ocean_full.jld2", "T"; grid=hydrostatic_ocean_grid)

θ_nonhydrostatic_ts  = FieldTimeSeries("nh_ocean_atmos.jld2", "θ"; grid)
u_nonhydrostatic_ts  = FieldTimeSeries("nh_ocean_atmos.jld2", "u"; grid)
qˡ_nonhydrostatic_ts = FieldTimeSeries("nh_ocean_atmos.jld2", "qˡ"; grid)
w_nonhydrostatic_ts  = FieldTimeSeries("nh_ocean_atmos.jld2", "w"; grid)
T_nonhydrostatic_ts  = FieldTimeSeries("ocean_nh.jld2", "T"; grid=nonhydrostatic_ocean_grid)

times = θ_slab_ts.times
Nt = length(times)

# ### Figure layout

fig = Figure(size = (1400, 525), fontsize = 12)

ax_θ_prescribed = Axis(fig[1, 1], title="θₗᵢ (K), atmosphere over prescribed ocean", ylabel="z (m)")
ax_u_prescribed = Axis(fig[2, 1], title="u (m s⁻¹), atmosphere over prescribed ocean", ylabel="z (m)")
ax_SST          = Axis(fig[3, 1], title="SST (K)", xlabel="x (m)", ylabel="SST (K)")

ax_θ_slab        = Axis(fig[1, 2], title="θₗᵢ (K), atmosphere over slab ocean", ylabel="z (m)")
ax_u_slab        = Axis(fig[2, 2], title="u (m s⁻¹), atmosphere over slab ocean", ylabel="z (m)")
ax_T_hydrostatic = Axis(fig[3, 2], title="T (°C), hydrostatic ocean", xlabel="x (m)", ylabel="z (m)")

ax_qˡ_hydrostatic   = Axis(fig[1, 3], title="Cloud liquid water, atmosphere over hydrostatic ocean", ylabel="z (m)")
ax_w_hydrostatic    = Axis(fig[2, 3], title="w (m s⁻¹), atmosphere over hydrostatic ocean", ylabel="z (m)")
ax_T_nonhydrostatic = Axis(fig[3, 3], title="T (°C), nonhydrostatic ocean", xlabel="x (m)", ylabel="z (m)")

ax_qˡ_nonhydrostatic = Axis(fig[1, 4], title="Cloud liquid water, atmosphere over nonhydrostatic ocean", ylabel="z (m)")
ax_w_nonhydrostatic  = Axis(fig[2, 4], title="w (m s⁻¹), atmosphere over nonhydrostatic ocean", ylabel="z (m)")
ax_T_profile         = Axis(fig[3, 4], title="⟨T⟩(z)", xlabel="T (°C)", ylabel="z (m)")

ax_θ_profile = Axis(fig[1, 5], title="⟨θ⟩(z)", xlabel="θ (K)", ylabel="z (m)", limits=((θᵃᵗ-1, θᵃᵗ+4), nothing))
ax_u_profile = Axis(fig[2, 5], title="⟨u⟩(z)", xlabel="u (m s⁻¹)", ylabel="z (m)", limits=((-10, 25), nothing))

colsize!(fig.layout, 5, Relative(0.10))

for ax in (ax_θ_prescribed, ax_u_prescribed, ax_θ_slab, ax_u_slab,
           ax_qˡ_hydrostatic, ax_w_hydrostatic, ax_qˡ_nonhydrostatic, ax_w_nonhydrostatic)
    hidexdecorations!(ax, ticks=false)
end

# ### Plot

n = Observable(1)

heatmap!(ax_θ_prescribed, @lift(θ_prescribed_ts[$n]); colormap=:thermal, colorrange=(θᵃᵗ - 1, θᵃᵗ + 3))
heatmap!(ax_u_prescribed, @lift(u_prescribed_ts[$n]); colormap=:balance, colorrange=(-30, 30))

heatmap!(ax_θ_slab,        @lift(θ_slab_ts[$n]);        colormap=:thermal, colorrange=(θᵃᵗ - 1, θᵃᵗ + 3))
heatmap!(ax_u_slab,        @lift(u_slab_ts[$n]);        colormap=:balance, colorrange=(-30, 30))
heatmap!(ax_T_hydrostatic, @lift(T_hydrostatic_ts[$n]); colormap=:thermal, colorrange=(T₀ - 1.5, T₀ + 0.5))

heatmap!(ax_qˡ_hydrostatic,   @lift(qˡ_hydrostatic_ts[$n]);   colormap=Reverse(:Blues_4), colorrange=(0, 5e-4))
heatmap!(ax_w_hydrostatic,    @lift(w_hydrostatic_ts[$n]);    colormap=:balance,          colorrange=(-25, 25))
heatmap!(ax_T_nonhydrostatic, @lift(T_nonhydrostatic_ts[$n]); colormap=:thermal,          colorrange=(T₀ - 1.5, T₀ + 0.5))

heatmap!(ax_qˡ_nonhydrostatic, @lift(qˡ_nonhydrostatic_ts[$n]); colormap=Reverse(:Blues_4), colorrange=(0, 5e-4))
heatmap!(ax_w_nonhydrostatic,  @lift(w_nonhydrostatic_ts[$n]);  colormap=:balance,          colorrange=(-25, 25))

x_hydrostatic    = xnodes(hydrostatic_ocean_grid, Center())
x_nonhydrostatic = xnodes(nonhydrostatic_ocean_grid, Center())
hydrostatic_SST_kelvin    = @lift interior(T_hydrostatic_ts[$n], :, 1, Nzᵒᶜ) .+ celsius_to_kelvin
nonhydrostatic_SST_kelvin = @lift interior(T_nonhydrostatic_ts[$n], :, 1, Nzⁿʰ) .+ celsius_to_kelvin

hlines!(ax_SST, Tᵒᶜ;                                        color=:black, linewidth=2, label="Prescribed")
lines!(ax_SST, @lift(SST_slab_ts[$n]);                      color=:red,   linewidth=2, label="Slab (10 m)")
lines!(ax_SST, x_hydrostatic, hydrostatic_SST_kelvin;       color=:blue,  linewidth=2, label="Hydrostatic")
lines!(ax_SST, x_nonhydrostatic, nonhydrostatic_SST_kelvin; color=:green, linewidth=2, label="Nonhydrostatic")
axislegend(ax_SST, position=:rb)
ylims!(ax_SST, Tᵒᶜ - 0.7, Tᵒᶜ + 0.2)

horizontal_average(fts, n) = Field(Average(fts[n], dims=1))

lines!(ax_θ_profile, @lift(horizontal_average(θ_prescribed_ts, $n));     color=:black, linewidth=1.5, label="Prescribed")
lines!(ax_θ_profile, @lift(horizontal_average(θ_slab_ts, $n));           color=:red,   linewidth=1.5, label="Slab")
lines!(ax_θ_profile, @lift(horizontal_average(θ_hydrostatic_ts, $n));    color=:blue,  linewidth=1.5, label="Hydrostatic")
lines!(ax_θ_profile, @lift(horizontal_average(θ_nonhydrostatic_ts, $n)); color=:green, linewidth=1.5, label="Nonhydrostatic")
axislegend(ax_θ_profile, position=:rt)

lines!(ax_u_profile, @lift(horizontal_average(u_prescribed_ts, $n));     color=:black, linewidth=1.5)
lines!(ax_u_profile, @lift(horizontal_average(u_slab_ts, $n));           color=:red,   linewidth=1.5)
lines!(ax_u_profile, @lift(horizontal_average(u_hydrostatic_ts, $n));    color=:blue,  linewidth=1.5)
lines!(ax_u_profile, @lift(horizontal_average(u_nonhydrostatic_ts, $n)); color=:green, linewidth=1.5)

## The well-mixed slab and the prescribed ocean have vertically uniform temperatures.
slab_temperature = @lift mean(SST_slab_ts[$n]) - celsius_to_kelvin

lines!(ax_T_profile, @lift(horizontal_average(T_hydrostatic_ts, $n));    color=:blue,  linewidth=1.5, label="Hydrostatic")
lines!(ax_T_profile, @lift(horizontal_average(T_nonhydrostatic_ts, $n)); color=:green, linewidth=1.5, label="Nonhydrostatic")
vlines!(ax_T_profile, slab_temperature;                                  color=:red,   linewidth=1.5, label="Slab")
vlines!(ax_T_profile, T₀;                                                color=:black, linewidth=1.5, label="Prescribed")
axislegend(ax_T_profile, position=:lb)
xlims!(ax_T_profile, T₀ - 1, T₀ + 0.5)

title = @lift "Atmosphere–ocean coupling comparison, t = " * prettytime(times[$n])
Label(fig[0, 1:5], title, fontsize=16)

fig

# ### Record

CairoMakie.record(fig, "breeze_over_four_oceans.mp4", 1:Nt; framerate=12) do nn
    n[] = nn
end
nothing #hide

# ![](breeze_over_four_oceans.mp4)
