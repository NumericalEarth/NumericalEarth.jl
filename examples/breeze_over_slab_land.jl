# # Diurnal radiative convection over heterogeneous slab land (2D)
#
# A 2D Breeze atmospheric large eddy simulation (LES) coupled to a
# `SlabLand` with spatially-varying surface moisture, driven by full
# RRTMGP all-sky radiation with a diurnal cycle.
#
# The land's central wet patch (`W ≈ W_max`, β ≈ 1) evaporates strongly
# during the day so that incoming radiation is partitioned into latent
# heat. The dry edges (`W = 0`, β = 0) cannot evaporate, so all the net
# radiation goes into sensible heat — producing strong surface heating
# and a vigorous dry convective boundary layer. At the wet/dry boundary
# the contrast drives a low-level "sea breeze"-like circulation.
#
# Coupling lives entirely in the `EarthSystemModel`:
#   * `AtmosphereLandModel(atmos, slab_land; radiation = rtm)` wires
#     turbulent surface fluxes (sensible, latent, momentum) through
#     Monin–Obukhov similarity theory and hands the RRTMGP
#     `RadiativeTransferModel` to the coupled model.
#   * The atmosphere is built with a skeleton `CoupledRadiation`
#     placeholder; the coupled-model constructor materializes it to
#     alias `rtm.flux_divergence` so Breeze's tendency machinery reads
#     directly from the RTM's flux divergence.
#   * The atmosphere's own `update_state!` drives the RRTMGP solve
#     through the proxy (honoring the RTM's `schedule`).
#   * Net surface SW/LW from the RTM feeds the slab's `net_energy_flux`
#     via `apply_air_land_radiative_fluxes!`, closing the surface
#     energy balance — no example-level callbacks required.

using NumericalEarth
using Breeze
using Oceananigans
using Oceananigans.Units
using RRTMGP
using NCDatasets
using Printf, Random, Statistics
using Dates: DateTime
using CairoMakie

Random.seed!(2025)

# ## Grid
#
# A 2D vertical slice: periodic in x, flat in y, bounded in z. Vertical
# stretching gives fine 100 m cells in the boundary layer (z ≤ 3 km),
# transitioning to 1 km cells up to 15 km.

arch = CPU()
Oceananigans.defaults.FloatType = Float32

Nx = 64
Lx = 20kilometers

z = PiecewiseStretchedDiscretization(z  = [0, 3000, 8000, 15000],
                                     Δz = [100,  100, 1000,  1000])

Nz = length(z) - 1

grid = RectilinearGrid(arch;
                       size = (Nx, Nz),
                       x = (-Lx/2, Lx/2),
                       z,
                       halo = (5, 5),
                       topology = (Periodic, Flat, Bounded))

# ## Heterogeneous slab land
#
# A 1D land grid (size Nx, flat in y and z) carrying skin temperature,
# soil moisture, and moisture availability. The land grid spans the
# same x extent as the atmosphere so that the slab T can serve directly
# as the RRTMGP surface temperature.

land_grid = RectilinearGrid(arch;
                            size = Nx,
                            x = (-Lx/2, Lx/2),
                            halo = grid.Hx,
                            topology = (Periodic, Flat, Flat))

hydrology = BucketHydrology(eltype(land_grid);
                            maximum_water_storage  = 150,
                            critical_wetness_ratio = 0.75)

slab_land = SlabLand(land_grid; hydrology)

# ### Moisture availability and the wet/dry contrast
#
# The bucket hydrology stores land water mass per area `Mˡᵃ` (kg m⁻²) with a
# Manabe-style saturation cap `Mˡᵃ⁺` (`maximum_water_storage`, the soil-science
# "field capacity"). The coupler reads a moisture availability factor
# `β ∈ [0, 1]` that linearly scales the surface specific humidity above which
# evaporation can occur:
#
# ```math
# β = \min\!\left(\frac{M^{\mathrm{la}}}{\varepsilon^{w} \, M^{\mathrm{la}\!+}},\ 1\right),
# ```
#
# so cells with `Mˡᵃ ≥ εʷ · Mˡᵃ⁺` (here `εʷ = 0.75`) evaporate as if
# saturated, and cells with `Mˡᵃ = 0` cannot evaporate at all. The coupler
# then uses `qₛ = qᵃ + β · (qᵛ⁺(Tₛ) − qᵃ)` — β = 1 → full latent heat flux,
# β = 0 → zero latent flux (all surface energy goes into sensible heating).
#
# We initialize `Mˡᵃ` as a Gaussian centered at the domain midpoint:
# saturated in the middle (`β ≈ 1`), dry at the edges (`β ≈ 0`). That
# contrast persists through the run because (i) the wet center can
# evaporate but starts deep inside the saturated plateau
# (`Mˡᵃ > εʷ · Mˡᵃ⁺`), so `β` stays pegged at 1 even as `Mˡᵃ` decreases,
# and (ii) the dry edges have no water to gain except via precipitation,
# which is not prescribed here.

T₀     = 295
M_wet  = 0.95 * hydrology.maximum_water_storage
σ_wet  = Lx / 8

M_init(x) = M_wet * exp(-(x/σ_wet)^2)

set!(slab_land.temperature, T₀)
set!(slab_land.water_storage, M_init)
Oceananigans.TimeSteppers.update_state!(slab_land)

# ## RRTMGP radiation
#
# All-sky RRTMGP at 15°N starting at the equinox, midnight local. The
# `surface_temperature` is the slab land's prognostic skin temperature,
# so radiation responds to surface heating and cooling in real time.

background_atmosphere = BackgroundAtmosphere()

latitude = 15
solar_position = ApparentSolarPosition(coordinate = (0, latitude),
                                       epoch = DateTime(2024, 3, 20, 0, 0, 0))

constants = ThermodynamicConstants()

radiation = RadiativeTransferModel(grid, AllSkyOptics(), constants;
                                   solar_position, background_atmosphere,
                                   surface_temperature = slab_land.temperature,
                                   surface_albedo      = 0.20,
                                   surface_emissivity  = 0.95,
                                   solar_constant      = 1361,
                                   schedule = TimeInterval(10minutes),
                                   liquid_effective_radius = ConstantRadiusParticles(10e-6),
                                   ice_effective_radius    = ConstantRadiusParticles(30e-6))

# ## Atmosphere (Simulation wrapping a Breeze `AtmosphereModel`)
#
# The atmosphere is built with a skeleton `CoupledRadiation` placeholder.
# `AtmosphereLandModel` materializes it against the RTM below — no
# `radiation` kwarg is passed here.

atmos = atmosphere_simulation(grid;
                              surface_pressure      = 101325,
                              potential_temperature = 300,
                              coriolis = FPlane(latitude = latitude))

# Initial atmospheric profile: dry-adiabatic sub-cloud layer capped by a
# stably stratified troposphere transitioning to a 210 K stratosphere.
# Small perturbations in the lowest 1 km trigger convection once the
# surface heats up.

function Tᵇᵍ(z)
    T = 300 - 1e-3 * max(z, 1000) - 5e-3 * max(0, z - 1000)
    return max(T, 210)
end

δT = 1
zδ = 1000

Tᵢ(x, z) = Tᵇᵍ(z) + δT * (rand() - 0.5) * (z < zδ)
ℋᵢ(x, z) = (0.5 + 1e-2 * (rand() - 0.5)) * (z < zδ)

set!(atmos.model; T = Tᵢ, ℋ = ℋᵢ)

# ## Coupled model
#
# Passing `radiation = rtm` here triggers `materialize_earth_system_radiation!`,
# which aliases the atmosphere's `CoupledRadiation.flux_divergence` to
# `rtm.flux_divergence`, and installs the Breeze-aware
# `apply_air_land_radiative_fluxes!`.

model = AtmosphereLandModel(atmos, slab_land; radiation)

simulation = Simulation(model; Δt = 10, stop_time = 3days)
conjure_time_step_wizard!(simulation; cfl = 0.7, max_Δt = 60)

# ## Progress

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    atmos_model = sim.model.atmosphere.model
    T = atmos_model.temperature
    _, _, w = atmos_model.velocities
    wmax       = maximum(abs, w)
    Tmin, Tmax = extrema(T)

    Tg = sim.model.land.temperature
    Tg_min, Tg_max = extrema(Tg)

    rtm = sim.model.radiation
    OLR = mean(view(rtm.upwelling_longwave_flux, :, 1, Nz+1))

    @info @sprintf("iter %5d, t %8s, Δt %4.1fs, wall %6s, max|w| %4.2f m/s, T [%5.1f,%5.1f] K, Tg [%5.1f,%5.1f] K, OLR %5.1f W/m²",
                   iteration(sim), prettytime(sim), sim.Δt, prettytime(elapsed),
                   wmax, Tmin, Tmax, Tg_min, Tg_max, OLR)

    wall_clock[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(1000))

# ## Output

_, _, w = atmos.model.velocities
T  = atmos.model.temperature
qᵛ = atmos.model.microphysical_fields.qᵛ
qˡ = atmos.model.microphysical_fields.qˡ

simulation.output_writers[:atmos] = JLD2Writer(model, (; w, T, qᵛ, qˡ);
                                               filename = "breeze_slab_land_atmos",
                                               schedule = TimeInterval(10minutes),
                                               overwrite_existing = true)

simulation.output_writers[:land] = JLD2Writer(model,
                                              (; Tg = slab_land.temperature,
                                                  W  = slab_land.water_storage,
                                                  β  = slab_land.moisture_availability);
                                              filename = "breeze_slab_land_surface",
                                              schedule = TimeInterval(10minutes),
                                              overwrite_existing = true)

# ## Run

@info "Running coupled simulation..."
run!(simulation)
@info "Simulation complete."

# ## Animation
#
# Top row: x–z vertical slices of vertical velocity, temperature anomaly,
# and cloud liquid water. Bottom row: 1D land state along x — skin
# temperature, soil moisture, and moisture availability β.

w_ts  = FieldTimeSeries("breeze_slab_land_atmos.jld2",   "w")
T_ts  = FieldTimeSeries("breeze_slab_land_atmos.jld2",   "T")
qˡ_ts = FieldTimeSeries("breeze_slab_land_atmos.jld2",   "qˡ")
Tg_ts = FieldTimeSeries("breeze_slab_land_surface.jld2", "Tg")
W_ts  = FieldTimeSeries("breeze_slab_land_surface.jld2", "W")
β_ts  = FieldTimeSeries("breeze_slab_land_surface.jld2", "β")

times = w_ts.times
Nt    = length(times)

x_atmos  = xnodes(grid, Center())
z_face   = znodes(grid, Face())
z_center = znodes(grid, Center())
x_land   = xnodes(land_grid, Center())

wlim  = maximum(abs, w_ts) / 2
qˡlim = max(1e-6, maximum(qˡ_ts) / 2)

fig = Figure(size = (1500, 800), fontsize = 13)

ax_w  = Axis(fig[1, 1], title = "w (m/s)",       ylabel = "z (m)", limits = (nothing, (0, 5e3)))
ax_T  = Axis(fig[1, 2], title = "T anomaly (K)",                   limits = (nothing, (0, 5e3)))
ax_qˡ = Axis(fig[1, 3], title = "qˡ (kg/kg)",                      limits = (nothing, (0, 5e3)))

ax_Tg = Axis(fig[2, 1], title = "Skin temperature (K)",  xlabel = "x (m)", ylabel = "T_g (K)")
ax_W  = Axis(fig[2, 2], title = "Soil water (kg/m²)",    xlabel = "x (m)", ylabel = "W")
ax_β  = Axis(fig[2, 3], title = "Moisture availability", xlabel = "x (m)", ylabel = "β")

n = Observable(1)

wn  = @lift view(interior(w_ts[$n]),  :, 1, :)
Tn  = @lift begin
    T_xz = view(interior(T_ts[$n]), :, 1, :)
    T_xz .- mean(T_xz, dims = 1)
end
qˡn  = @lift view(interior(qˡ_ts[$n]), :, 1, :)
Tg_n = @lift vec(interior(Tg_ts[$n], :, 1, 1))
W_n  = @lift vec(interior(W_ts[$n],  :, 1, 1))
β_n  = @lift vec(interior(β_ts[$n],  :, 1, 1))

heatmap!(ax_w,  x_atmos, z_face,   wn;  colormap = :balance, colorrange = (-wlim, wlim))
heatmap!(ax_T,  x_atmos, z_center, Tn;  colormap = :balance, colorrange = (-2, 2))
heatmap!(ax_qˡ, x_atmos, z_center, qˡn; colormap = :dense,   colorrange = (0, qˡlim))

lines!(ax_Tg, x_land, Tg_n; color = :black, linewidth = 2)
lines!(ax_W,  x_land, W_n;  color = :black, linewidth = 2)
lines!(ax_β,  x_land, β_n;  color = :black, linewidth = 2)

ylims!(ax_W, 0, hydrology.maximum_water_storage * 1.05)
ylims!(ax_β, 0, 1.05)

title = @lift "Diurnal convection over heterogeneous slab land, t = " * prettytime(times[$n])
Label(fig[0, 1:3], title, fontsize = 16)

@info "Rendering animation..."
CairoMakie.record(fig, "breeze_over_slab_land.mp4", 1:Nt; framerate = 12) do nn
    n[] = nn
end

@info "Animation saved."
nothing #hide

# ![](breeze_over_slab_land.mp4)
