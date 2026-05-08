# # Atmospheric convection over heterogeneous land use (3D)
#
# Couples a Breeze simulation (LES) to a `RucSlabLand` through
# NumericalEarth's `EarthSystemModel` framework in a doubly-periodic 3D
# domain. Turbulent surface fluxes (sensible heat, latent heat,
# momentum) are computed with Monin--Obukhov similarity theory through
# `AtmosphereLandModel`, with surface humidity reduced by the slab's
# `mavail` (β-factor).
#
# The domain is partitioned into three strips of contrasting USGS classes
# — evergreen broadleaf forest, grassland, and barren / sparsely vegetated.
# The atmosphere-land turbulent-flux path uses the land-class `znt` field in
# MOST and β-reduced surface humidity, so the surface-flux contrast reflects
# both roughness and strip-dependent soil moisture. Albedo, emissivity, and
# LAI are also populated for future radiation and transpiration plumbing. The
# strips are oriented along the x axis (parallel to the geostrophic flow) so
# air parcels remain over a single land-cover class rather than fetching
# across boundaries. The animation pairs the horizontal 2D ground-temperature
# field with bottom-level wind vectors, and reports strip-conditional vertical
# profiles of θ and u.

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
# Heterogeneous USGS land use. Strips are elongated along x and stack
# in y, so each strip is parallel to the geostrophic flow:
#   y ∈ [-10, -10/3) km : class 13 (evergreen broadleaf forest)
#                         z₀=0.80 m, α=0.12, vegfrac=0.95, LAI=6.48
#   y ∈ [-10/3, 10/3) km: class 7  (grassland)
#                         z₀=0.075 m, α=0.19, vegfrac=0.80, LAI=2.90
#   y ∈ [ 10/3, 10] km  : class 19 (barren / sparsely vegetated)
#                         z₀=0.05 m, α=0.25, vegfrac=0.01, LAI=0.75
# `apply_land_classifications!` populates `vegfrac`, `lai`,
# `albedo_veg`, `emissivity_veg`, `z0_veg`, `r_smin` from this
# categorical map. Initial soil moisture is also strip-dependent
# (wet under forest, dry under barren) so the partitioning of net
# turbulent exchange between sensible and latent heat differs across strips.

land_grid = RectilinearGrid(grid.architecture,
                            size = (grid.Nx, grid.Ny),
                            halo = (grid.Hx, grid.Hy),
                            x = (-10kilometers, 10kilometers),
                            y = (-10kilometers, 10kilometers),
                            topology = (Periodic, Periodic, Flat))

slab_land = RucSlabLand(land_grid;
                        parameters = RucSlabLandParameters(eltype(land_grid);
                                                           depth = 0.10,
                                                           density = 1500,
                                                           heat_capacity = 1480,
                                                           soil_depth = 1.0))

strip_south = -10kilometers / 3   # forest | grass boundary
strip_north =  10kilometers / 3   # grass  | barren boundary

y_land_centers = Array(ynodes(land_grid, Center()))
vegtype = fill(7, grid.Nx, grid.Ny)
for j in 1:grid.Ny
    if y_land_centers[j] < strip_south
        vegtype[:, j] .= 13       # evergreen broadleaf forest
    elseif y_land_centers[j] >= strip_north
        vegtype[:, j] .= 19       # barren / sparsely vegetated
    end
end

registry = usgs_land_classifications(eltype(land_grid))
apply_land_classifications!(slab_land, vegtype, registry)

function θ_init(x, y)
    if y < strip_south
        return 0.40       # wet forest soil
    elseif y >= strip_north
        return 0.10       # dry barren soil
    else
        return 0.30       # moderate grassland soil
    end
end

set!(slab_land, T = Tᵍ₀, Tc = Tᵍ₀, θ = θ_init)

slab_land.forcings.solar_irradiance .= 600.0    # W m⁻², bright midday

# `air_temperature` and `air_humidity` are refreshed from the coupled
# atmosphere state by `AtmosphereLandModel`.

# ## Coupled model
#
# `AtmosphereLandModel` wires the atmosphere to the slab through
# similarity theory. The default land flux formulation reads the slab's
# spatially varying `znt` field for momentum roughness and uses `znt / 10`
# for heat and moisture roughness.

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
    Tg   = sim.model.land.temperature
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
                                               filename = "breeze_ruc_slab_land_atmos",
                                               schedule = TimeInterval(1minute),
                                               overwrite_existing = true)

simulation.output_writers[:land] = JLD2Writer(model, (; Tg=slab_land.temperature);
                                              filename = "breeze_ruc_slab_land_surface",
                                              schedule = TimeInterval(1minute),
                                              overwrite_existing = true)

# ## Run

@info "Running coupled simulation..."
run!(simulation)
@info "Simulation complete."

# ## Animation
#
# Layout: top row shows horizontal (x–y) maps of bottom-level θₗᵢ and
# bottom-level wind speed alongside strip-conditional vertical profiles
# of θ; bottom row shows the strip-mean T_g time series, the ground
# temperature T_g with surface wind vectors overlaid as quivers, and
# strip-conditional vertical profiles of u. Dashed strip-boundary
# lines on every horizontal panel mark the underlying land cover; a
# moving black tick on the time-series panel tracks the current
# animation frame.

using CairoMakie
using Statistics

θ_ts  = FieldTimeSeries("breeze_ruc_slab_land_atmos.jld2",   "θ"; architecture=CPU())
u_ts  = FieldTimeSeries("breeze_ruc_slab_land_atmos.jld2",   "u"; architecture=CPU())
v_ts  = FieldTimeSeries("breeze_ruc_slab_land_atmos.jld2",   "v"; architecture=CPU())
s_ts  = FieldTimeSeries("breeze_ruc_slab_land_atmos.jld2",   "s"; architecture=CPU())
Tg_ts = FieldTimeSeries("breeze_ruc_slab_land_surface.jld2", "Tg"; architecture=CPU())

times = θ_ts.times
Nt = length(times)

# Coordinate arrays.

x_atmos = xnodes(grid, Center())
y_atmos = ynodes(grid, Center())
z_atmos = znodes(grid, Center())
x_land  = xnodes(land_grid, Center())
y_land  = ynodes(land_grid, Center())

# Subsampled grid for quiver arrows.
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

ax_tg = Axis(fig[2, 1], title="Strip-mean T_g(t)",      xlabel="t (h)",   ylabel="T_g (K)")
ax_hs = Axis(fig[2, 2], title="T_g (K) + surface wind", xlabel="x (m)",   ylabel="y (m)",
             aspect=DataAspect())
ax_up = Axis(fig[2, 3], title="⟨u⟩(z)",                 xlabel="u (m/s)", ylabel="z (m)")

colsize!(fig.layout, 3, Relative(0.2))

# Observables.

n = Observable(1)

θn_bot = @lift view(interior(θ_ts[$n]), :, :, 1)
sn_bot = @lift view(interior(s_ts[$n]), :, :, 1)
Tg2d   = @lift view(interior(Tg_ts[$n]), :, :, 1)
u_bot  = @lift view(interior(u_ts[$n]), i_quiver, j_quiver, 1)
v_bot  = @lift view(interior(v_ts[$n]), i_quiver, j_quiver, 1)

# Strip-conditional vertical profiles (one curve per land-use strip).
j1 = count(y -> y < strip_south, y_atmos)
j2 = count(y -> y < strip_north, y_atmos)
j_forest = 1:j1
j_grass  = (j1 + 1):j2
j_barren = (j2 + 1):grid.Ny

θ_forest = @lift vec(mean(view(interior(θ_ts[$n]), :, j_forest, :); dims=(1, 2)))
θ_grass  = @lift vec(mean(view(interior(θ_ts[$n]), :, j_grass,  :); dims=(1, 2)))
θ_barren = @lift vec(mean(view(interior(θ_ts[$n]), :, j_barren, :); dims=(1, 2)))

u_forest = @lift vec(mean(view(interior(u_ts[$n]), :, j_forest, :); dims=(1, 2)))
u_grass  = @lift vec(mean(view(interior(u_ts[$n]), :, j_grass,  :); dims=(1, 2)))
u_barren = @lift vec(mean(view(interior(u_ts[$n]), :, j_barren, :); dims=(1, 2)))

heatmap!(ax_θ,  x_atmos, y_atmos, θn_bot; colormap=:thermal, colorrange=(θᵃᵗ - 0.5, θᵃᵗ + 8))
heatmap!(ax_s,  x_atmos, y_atmos, sn_bot; colormap=:speed,   colorrange=(0, 5))
heatmap!(ax_hs, x_land,  y_land,  Tg2d;   colormap=:thermal, colorrange=(Tᵍ₀ - 10, Tᵍ₀ + 15))
arrows2d!(ax_hs, x_quiver, y_quiver, u_bot, v_bot; lengthscale=200, color=:white)

# Strip-mean T_g time series. Pre-computed once over the full series
# (cheap), then animated only by sliding a vertical tick.
times_hours    = collect(times) ./ 3600
T_g_forest_ts  = [mean(view(interior(Tg_ts[k]), :, j_forest, 1)) for k in 1:Nt]
T_g_grass_ts   = [mean(view(interior(Tg_ts[k]), :, j_grass,  1)) for k in 1:Nt]
T_g_barren_ts  = [mean(view(interior(Tg_ts[k]), :, j_barren, 1)) for k in 1:Nt]

lines!(ax_tg, times_hours, T_g_forest_ts; linewidth=1.5, color=:darkgreen, label="Forest")
lines!(ax_tg, times_hours, T_g_grass_ts;  linewidth=1.5, color=:olive,     label="Grass")
lines!(ax_tg, times_hours, T_g_barren_ts; linewidth=1.5, color=:peru,      label="Barren")
t_now = @lift [times_hours[$n]]
vlines!(ax_tg, t_now; color=:black, linewidth=1.0)
axislegend(ax_tg; position=:rb, framevisible=false)

for ax in (ax_θ, ax_s, ax_hs)
    hlines!(ax, [strip_south, strip_north]; color=:black, linestyle=:dash, linewidth=1.5)
end

x_label = minimum(x_land) + 1kilometers
text!(ax_hs, x_label, (minimum(y_land) + strip_south) / 2; text="Forest", color=:white, fontsize=13, align=(:left, :center))
text!(ax_hs, x_label, 0;                                   text="Grass",  color=:white, fontsize=13, align=(:left, :center))
text!(ax_hs, x_label, (strip_north + maximum(y_land)) / 2; text="Barren", color=:white, fontsize=13, align=(:left, :center))

lines!(ax_θp, θ_forest, z_atmos; linewidth=1.5, color=:darkgreen, label="Forest")
lines!(ax_θp, θ_grass,  z_atmos; linewidth=1.5, color=:olive,     label="Grass")
lines!(ax_θp, θ_barren, z_atmos; linewidth=1.5, color=:peru,      label="Barren")
xlims!(ax_θp, θᵃᵗ - 1, θᵃᵗ + 8)
axislegend(ax_θp; position=:rb, framevisible=false)

lines!(ax_up, u_forest, z_atmos; linewidth=1.5, color=:darkgreen, label="Forest")
lines!(ax_up, u_grass,  z_atmos; linewidth=1.5, color=:olive,     label="Grass")
lines!(ax_up, u_barren, z_atmos; linewidth=1.5, color=:peru,      label="Barren")
xlims!(ax_up, -3, 8)

title = @lift "Breeze LES over heterogeneous RUC slab land (3D), t = " * prettytime(times[$n])
Label(fig[0, 1:3], title, fontsize=16)

# ## Record

@info "Rendering animation..."
CairoMakie.record(fig, "breeze_over_ruc_slab_land.mp4", 1:Nt; framerate=12) do nn
    n[] = nn
end

@info "Animation saved."
nothing #hide

# ![](breeze_over_ruc_slab_land.mp4)
