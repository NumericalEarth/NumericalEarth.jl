# # Atmospheric convection over a homogeneous slab land (3D)
#
# Couples a Breeze atmospheric large eddy simulation (LES) to a
# `SlabLand` composed of:
#   energy    = SlabEnergy(heat_capacity = ρcH_g)
#   hydrology = BucketHydrology(...)
#   surface   = ConstantSurfaceProperties(...)
# through NumericalEarth's `EarthSystemModel` framework in a doubly-
# periodic 3D domain. Turbulent surface fluxes (sensible, latent,
# momentum) are computed with Monin–Obukhov similarity theory through
# `AtmosphereLandModel`, with surface humidity reduced by the slab's
# `moisture_availability` (β-factor).
#
# The land is spatially uniform: a single roughness length applies
# everywhere. (Radiative properties — albedo, emissivity — live on the
# top-level `radiation` component when one is attached; this example
# runs without radiation.) The animation pairs the horizontal 2D
# ground-temperature field with bottom-level wind vectors, and shows
# the domain-mean vertical profiles of θ and u.

using NumericalEarth
using Breeze
using CUDA
using Oceananigans
using Oceananigans.Units
using Printf

# ## Grid setup
#
# Doubly-periodic 3D domain, 20 km × 20 km × 10 km, 64³ grid points.

arch = GPU()

Nxᵃᵗ = 64
Nyᵃᵗ = 64
Nzᵃᵗ = 64

grid = RectilinearGrid(arch,
                       size = (Nxᵃᵗ, Nyᵃᵗ, Nzᵃᵗ), halo = (5, 5, 5),
                       x = (-10kilometers, 10kilometers),
                       y = (-10kilometers, 10kilometers),
                       z = (0, 10kilometers),
                       topology = (Periodic, Periodic, Bounded))

# ## Atmosphere
#
# Warm ground, cool air aloft → unstable PBL with upward sensible and
# latent heat fluxes that cool the ground toward equilibrium and drive
# convective overturning.

Tᵍ₀ = 305 # K, initial ground temperature
θᵃᵗ = 295 # K, atmospheric potential temperature
U₀  = 2   # m/s

atmos = atmosphere_simulation(grid; potential_temperature=θᵃᵗ)

reference_state = atmos.dynamics.reference_state
θᵢ(x, y, z) = reference_state.potential_temperature + 0.1 * randn() * (z < 500)
set!(atmos, θ=θᵢ, u=U₀)

# ## Slab land
#
# Single homogeneous patch — `SlabEnergy + BucketHydrology +
# ConstantSurfaceProperties`. The bucket is 15 cm deep (Manabe's
# original `W_max = 150 kg m⁻²`); the soil thermal slab carries the
# same areal heat capacity as the previous example for comparability.

land_grid = RectilinearGrid(grid.architecture,
                            size = (grid.Nx, grid.Ny),
                            halo = (grid.Hx, grid.Hy),
                            x = (-10kilometers, 10kilometers),
                            y = (-10kilometers, 10kilometers),
                            topology = (Periodic, Periodic, Flat))

ρcH_g = 1500.0 * 1480.0 * 0.10      # J m⁻² K⁻¹

energy    = SlabEnergy(eltype(land_grid); heat_capacity = ρcH_g)
hydrology = BucketHydrology(eltype(land_grid);
                         field_capacity   = 150.0,
                         critical_wetness = 0.75)
surface   = ConstantSurfaceProperties(eltype(land_grid);
                                       momentum_roughness_length = 0.1,
                                       scalar_roughness_length = 0.01)

slab_land = SlabLand(land_grid; energy, hydrology, surface)

set!(slab_land.state.T, Tᵍ₀)
set!(slab_land.state.W, 0.5 * hydrology.field_capacity)   # 50% saturation
Oceananigans.TimeSteppers.update_state!(slab_land)

# ## Coupled model
#
# `AtmosphereLandModel` wires the atmosphere to the slab through
# similarity theory. Roughness lengths come from the surface closure.

model = AtmosphereLandModel(atmos, slab_land)

Δt = 5seconds
stop_time = 4hours
simulation = Simulation(model; Δt, stop_time)

# ## Progress callback

function progress(sim)
    a = sim.model.atmosphere
    u, v, w = a.velocities
    umax = maximum(abs, u)
    wmax = maximum(abs, w)
    Tg   = sim.model.land.state.T
    msg = @sprintf("Iter: %d, t = %s, max|u|: %.2e, max|w|: %.2e, T_g: (%.3f, %.3f)",
                   iteration(sim), prettytime(sim), umax, wmax,
                   minimum(Tg), maximum(Tg))
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(400))

# ## Output writers

u, v, w = atmos.velocities
s = @at (Center, Center, Center) √(u^2 + v^2 + w^2)
θ = liquid_ice_potential_temperature(atmos)

simulation.output_writers[:atmos] = JLD2Writer(model, (; θ, u, v, s);
                                               filename = "breeze_slab_land_atmos",
                                               schedule = TimeInterval(1minute),
                                               overwrite_existing = true)

simulation.output_writers[:land] = JLD2Writer(model, (; Tg=slab_land.state.T);
                                              filename = "breeze_slab_land_surface",
                                              schedule = TimeInterval(1minute),
                                              overwrite_existing = true)

# ## Run

@info "Running coupled simulation..."
run!(simulation)
@info "Simulation complete."

# ## Animation
#
# Layout: top row shows horizontal (x–y) maps of bottom-level θₗᵢ and
# bottom-level wind speed alongside domain-mean vertical profiles of θ;
# bottom row shows the domain-mean T_g time series, the ground
# temperature T_g with surface wind vectors overlaid as quivers, and
# the domain-mean vertical profile of u.

using CairoMakie
using Statistics

θ_ts  = FieldTimeSeries("breeze_slab_land_atmos.jld2",   "θ"; architecture=CPU())
u_ts  = FieldTimeSeries("breeze_slab_land_atmos.jld2",   "u"; architecture=CPU())
v_ts  = FieldTimeSeries("breeze_slab_land_atmos.jld2",   "v"; architecture=CPU())
s_ts  = FieldTimeSeries("breeze_slab_land_atmos.jld2",   "s"; architecture=CPU())
Tg_ts = FieldTimeSeries("breeze_slab_land_surface.jld2", "Tg"; architecture=CPU())

times = θ_ts.times
Nt = length(times)

x_atmos = xnodes(grid, Center())
y_atmos = ynodes(grid, Center())
z_atmos = znodes(grid, Center())
x_land  = xnodes(land_grid, Center())
y_land  = ynodes(land_grid, Center())

qstep = max(grid.Nx ÷ 16, 1)
i_quiver = 1:qstep:grid.Nx
j_quiver = 1:qstep:grid.Ny
x_quiver = x_atmos[i_quiver]
y_quiver = y_atmos[j_quiver]

fig = Figure(size = (1500, 900), fontsize = 12)

ax_θ  = Axis(fig[1, 1], title="θₗᵢ (K), bottom level",               xlabel="x (m)", ylabel="y (m)",
             aspect=DataAspect())
ax_s  = Axis(fig[1, 2], title="√(u² + v² + w²) (m/s), bottom level", xlabel="x (m)", ylabel="y (m)",
             aspect=DataAspect())
ax_θp = Axis(fig[1, 3], title="⟨θ⟩(z)",                              xlabel="θ (K)", ylabel="z (m)")

ax_tg = Axis(fig[2, 1], title="Domain-mean T_g(t)",     xlabel="t (h)",   ylabel="T_g (K)")
ax_hs = Axis(fig[2, 2], title="T_g (K) + surface wind", xlabel="x (m)",   ylabel="y (m)",
             aspect=DataAspect())
ax_up = Axis(fig[2, 3], title="⟨u⟩(z)",                 xlabel="u (m/s)", ylabel="z (m)")

colsize!(fig.layout, 3, Relative(0.2))

n = Observable(1)

θn_bot = @lift view(interior(θ_ts[$n]), :, :, 1)
sn_bot = @lift view(interior(s_ts[$n]), :, :, 1)
Tg2d   = @lift view(interior(Tg_ts[$n]), :, :, 1)
u_bot  = @lift view(interior(u_ts[$n]), i_quiver, j_quiver, 1)
v_bot  = @lift view(interior(v_ts[$n]), i_quiver, j_quiver, 1)

θ_mean = @lift vec(mean(interior(θ_ts[$n]); dims=(1, 2)))
u_mean = @lift vec(mean(interior(u_ts[$n]); dims=(1, 2)))

heatmap!(ax_θ,  x_atmos, y_atmos, θn_bot; colormap=:thermal, colorrange=(θᵃᵗ - 0.5, θᵃᵗ + 8))
heatmap!(ax_s,  x_atmos, y_atmos, sn_bot; colormap=:speed,   colorrange=(0, 5))
heatmap!(ax_hs, x_land,  y_land,  Tg2d;   colormap=:thermal, colorrange=(Tᵍ₀ - 10, Tᵍ₀ + 15))
arrows2d!(ax_hs, x_quiver, y_quiver, u_bot, v_bot; lengthscale=200, color=:white)

times_hours = collect(times) ./ 3600
Tg_mean_ts  = [mean(interior(Tg_ts[k])) for k in 1:Nt]

lines!(ax_tg, times_hours, Tg_mean_ts; linewidth=1.5, color=:black)
t_now = @lift [times_hours[$n]]
vlines!(ax_tg, t_now; color=:black, linewidth=1.0, linestyle=:dash)

lines!(ax_θp, θ_mean, z_atmos; linewidth=1.5, color=:black)
xlims!(ax_θp, θᵃᵗ - 1, θᵃᵗ + 8)

lines!(ax_up, u_mean, z_atmos; linewidth=1.5, color=:black)
xlims!(ax_up, -3, 8)

title = @lift "Breeze LES over homogeneous slab land (3D), t = " * prettytime(times[$n])
Label(fig[0, 1:3], title, fontsize=16)

# ## Record

@info "Rendering animation..."
CairoMakie.record(fig, "breeze_over_slab_land.mp4", 1:Nt; framerate=12) do nn
    n[] = nn
end

@info "Animation saved."
nothing #hide

# ![](breeze_over_slab_land.mp4)
