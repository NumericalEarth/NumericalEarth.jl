# # Diurnal radiative convection over heterogeneous slab land (2D)
#
# A 2D Breeze atmospheric large eddy simulation (LES) coupled to a
# `SlabLand` with spatially-varying surface moisture, driven by full
# RRTMGP all-sky radiation with a diurnal cycle.
#
# The land's central wet patch (`𝒮 ≈ 1`, full evaporation efficiency)
# evaporates strongly during the day so that incoming radiation is partitioned
# into latent heat. The dry edges (`𝒮 = 0`) cannot evaporate, so all the net
# radiation goes into sensible heat — producing strong surface heating
# and a vigorous dry convective boundary layer. At the wet/dry boundary
# the contrast drives a low-level "sea breeze"-like circulation.
#
# Coupling lives entirely in the `EarthSystemModel`:
#   * `AtmosphereLandModel(atmos, slab_land; radiation)` wires turbulent
#     surface fluxes (sensible, latent, momentum) through Monin–Obukhov
#     similarity theory — using land stability functions
#     (`atmosphere_land_stability_functions`, the Businger–Dyer /
#     Large–Yeager form) rather than the ocean Edson default — and hands
#     the RRTMGP `RadiativeTransferModel` to the coupled model.
#   * The atmosphere is built with a skeleton `CoupledRadiation`
#     placeholder; the coupled-model constructor materializes it to alias
#     `radiative_transfer_model.flux_divergence` so Breeze's tendency
#     machinery reads directly from the RTM's flux divergence.
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
# soil water, and surface saturation. The land grid spans the same x
# extent as the atmosphere so that the slab T can serve directly as the
# RRTMGP surface temperature.

land_grid = RectilinearGrid(arch;
                            size = Nx,
                            x = (-Lx/2, Lx/2),
                            halo = grid.Hx,
                            topology = (Periodic, Flat, Flat))

hydrology = BucketHydrology(maximum_water_storage = 150)
slab_land = SlabLand(land_grid; hydrology)

# ### Surface saturation and the wet/dry contrast
#
# The bucket hydrology stores land water mass per area `Mˡᵃ` (kg m⁻²) with a
# saturation cap `Mˡᵃ⁺` (`maximum_water_storage`, the soil-science "field
# capacity"), and exposes the continuous surface saturation
# `𝒮 = Mˡᵃ/Mˡᵃ⁺ ∈ [0, 1]`. The interface's `FractionalHumidity` model with a
# Manabe `CriticalSaturation(𝒮ᶜ)` efficiency scales the saturation specific humidity
# by the evaporation efficiency `β(𝒮) = min(𝒮/𝒮ᶜ, 1)`:
#
# ```math
# q_s = β(𝒮) \, q^{v+}(T_s),  \qquad β(𝒮) = \min(𝒮/𝒮_c, 1).
# ```
#
# The wet center (`𝒮 ≥ 𝒮ᶜ`) evaporates at full efficiency (`qₛ = qᵛ⁺`, strong
# latent-heat flux), while the dry edges (`𝒮 = 0`) cannot evaporate (no latent
# flux ⇒ all surface energy goes into sensible heating).
#
# We initialize `Mˡᵃ` as a Gaussian centered at the domain midpoint: wet in
# the middle (`qₛ = qᵛ⁺`), bone-dry at the edges (`qₛ = 0`). The contrast
# persists because the wet center retains water through the run while the dry
# edges have no source (no precipitation is prescribed here).

T₀    = 295 # K
M_wet = 0.95 * hydrology.maximum_water_storage
σ_wet = Lx / 8

M_init(x) = M_wet * exp(-(x/σ_wet)^2)

set!(slab_land.temperature, T₀)
set!(slab_land.water_storage, M_init)
Oceananigans.TimeSteppers.update_state!(slab_land)

# ## Reference state, dynamics, and a stratospheric sponge
#
# The 15 km column gives RRTMGP a realistic atmosphere, but the initial
# stratosphere is not in radiative equilibrium and the coarse upper cells
# respond strongly once radiation switches on. A Newtonian relaxation of
# temperature toward the reference profile above 8 km anchors the
# stratosphere without affecting the troposphere (as in Breeze's
# `radiative_convection` example). We build the reference state explicitly
# so the sponge and the radiation share the same thermodynamic constants.

p₀ = 101325    # Pa
θ₀ = 300       # K
latitude = 15

constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants;
                                 surface_pressure = p₀,
                                 potential_temperature = θ₀)
dynamics = AnelasticDynamics(reference_state)

Tᵣ  = reference_state.temperature
ρᵣ  = reference_state.density
cᵖᵈ = constants.dry_air.heat_capacity / constants.dry_air.molar_mass
τ_sponge = 6hours

@inline function stratospheric_relaxation(i, j, k, grid, clock, model_fields, p)
    @inbounds T  = model_fields.T[i, j, k]
    @inbounds Tᵣ = p.Tᵣ[i, j, k]
    @inbounds ρ  = p.ρᵣ[i, j, k]
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Center())
    α = clamp((z - 8000) / 4000, 0, 1)
    return ρ * p.cᵖᵈ * (-α * (T - Tᵣ) / p.τ)
end

sponge = Forcing(stratospheric_relaxation; discrete_form = true,
                 parameters = (; Tᵣ, ρᵣ, cᵖᵈ, τ = τ_sponge))

# ## RRTMGP radiation
#
# All-sky RRTMGP at 15°N starting at the equinox, midnight local. The
# `surface_temperature` is the slab land's prognostic skin temperature,
# so radiation responds to surface heating and cooling in real time.
#
# A tropical ozone profile is required for stratospheric radiative balance:
# without it the upper column is far from radiative equilibrium and
# destabilizes when the spectral fluxes recompute over the convecting
# troposphere.

@inline function tropical_ozone(z)
    troposphere_O₃  = 3e-8 * (1 + 0.5 * z / 1e3)
    stratosphere_O₃ = 8e-6 * exp(-((z - 25e3) / 5e3)^2)
    χˢᵗ = 1 / (1 + exp(-(z - 15e3) / 2))
    return troposphere_O₃ * (1 - χˢᵗ) + stratosphere_O₃ * χˢᵗ
end

background_atmosphere = BackgroundAtmosphere(O₃ = tropical_ozone)
solar_position = ApparentSolarPosition(coordinate = (0, latitude),
                                       epoch = DateTime(2024, 3, 20, 0, 0, 0))

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

atmos = atmosphere_simulation(grid; dynamics,
                              forcing  = (; ρe = sponge),
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

# Recompute the reference state from the horizontal mean. In a tall Float32
# column the default dry-adiabat reference diverges from the actual
# stratospheric profile, producing density errors that overwhelm Float32
# precision; `set_to_mean!` anchors `ρᵣ` to the current state.

set_to_mean!(reference_state, atmos.model, rescale_densities = true)

# ## Coupled model
#
# Passing `radiation = radiative_transfer_model` here triggers
# `materialize_earth_system_radiation!`, which aliases the atmosphere's
# `CoupledRadiation.flux_divergence` to
# `radiative_transfer_model.flux_divergence` and installs the Breeze-aware
# `apply_air_land_radiative_fluxes!`.

# The surface specific humidity uses a Manabe evaporation efficiency: saturated
# above the critical saturation `𝒮ᶜ = 0.75`, scaling down linearly below it.
interface_specific_humidity = FractionalHumidity(efficiency = CriticalSaturation(0.75))
al_interface = atmosphere_land_interface(slab_land.grid, atmos, slab_land;
                                         specific_humidity = interface_specific_humidity)

# The coupled model's clock is authoritative for all components, and a
# `Simulation`'s clock type is fixed by its grid — since the grids here are
# `Float32`, the coupled model needs a matching `Float32` clock.
model = AtmosphereLandModel(atmos, slab_land; radiation,
                            clock = Clock{Float32}(time=0),
                            atmosphere_land_interface = al_interface)

# The wizard recomputes Δt every iteration so the step always tracks the
# current CFL — important for a convective LES on a 100 m grid, where a
# cumulus updraft can tighten the vertical CFL within a few steps. `max_Δt`
# caps the step during the quiescent cold-start (velocities ≈ 0 ⇒ unbounded
# advective timescale).
simulation = Simulation(model; Δt = 1e-6, stop_time = 3days)
conjure_time_step_wizard!(simulation, IterationInterval(1); cfl = 0.7, max_Δt = 6)

# ## Progress

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    atmos_model = sim.model.atmosphere.model
    Tᵃᵗ = atmos_model.temperature
    _, _, w = atmos_model.velocities
    wmax            = maximum(abs, w)
    Tᵃᵗ_min, Tᵃᵗ_max = extrema(Tᵃᵗ)

    Tˡᵃ = sim.model.land.temperature
    Tˡᵃ_min, Tˡᵃ_max = extrema(Tˡᵃ)

    radiative_transfer_model = sim.model.radiation
    OLR = mean(view(radiative_transfer_model.upwelling_longwave_flux, :, 1, Nz+1))

    @info @sprintf("iter %5d, t %8s, Δt %4.1fs, wall %6s, max|w| %4.2f m/s, Tᵃᵗ [%5.1f,%5.1f] K, Tˡᵃ [%5.1f,%5.1f] K, OLR %5.1f W/m²",
                   iteration(sim), prettytime(sim), sim.Δt, prettytime(elapsed),
                   wmax, Tᵃᵗ_min, Tᵃᵗ_max, Tˡᵃ_min, Tˡᵃ_max, OLR)

    wall_clock[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(1000))

# ## Output

_, _, w = atmos.model.velocities
T  = atmos.model.temperature
qˡ = atmos.model.microphysical_fields.qˡ

simulation.output_writers[:atmos] = JLD2Writer(model, (; w, T, qˡ);
                                               filename = "breeze_slab_land_atmos",
                                               schedule = TimeInterval(10minutes),
                                               overwrite_existing = true)

simulation.output_writers[:land] = JLD2Writer(model,
                                              (; T = slab_land.temperature,
                                                  M = slab_land.water_storage,
                                                  𝒮 = slab_land.saturation);
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
# temperature, soil water, and surface saturation 𝒮.

w_ts   = FieldTimeSeries("breeze_slab_land_atmos.jld2",   "w")
Tᵃᵗ_ts = FieldTimeSeries("breeze_slab_land_atmos.jld2",   "T")
qˡ_ts  = FieldTimeSeries("breeze_slab_land_atmos.jld2",   "qˡ")
Tˡᵃ_ts = FieldTimeSeries("breeze_slab_land_surface.jld2", "T")
M_ts   = FieldTimeSeries("breeze_slab_land_surface.jld2", "M")
𝒮_ts   = FieldTimeSeries("breeze_slab_land_surface.jld2", "𝒮")

times = w_ts.times
Nt    = length(times)

x_atmos  = xnodes(grid, Center())
z_face   = znodes(grid, Face())
z_center = znodes(grid, Center())
x_land   = xnodes(land_grid, Center())

wlim  = maximum(abs, w_ts) / 2

# Cloud liquid water is sparse and the single peak value is much larger than a
# typical cloudy cell, so scaling the colorbar to the global maximum washes the
# clouds out. Anchor the upper bound to a high quantile of the *cloudy* (qˡ > 0)
# values instead, so the bulk of the cloud field spans the colormap.
qˡ_cloudy = filter(>(0), interior(qˡ_ts))
qˡlim     = isempty(qˡ_cloudy) ? 1e-6 : quantile(qˡ_cloudy, 0.99)

fig = Figure(size = (1200, 640))

ax_w   = Axis(fig[1, 1][1, 1], title = "w (m/s)",         ylabel = "z (m)", limits = (nothing, (0, 5e3)))
ax_Tᵃᵗ = Axis(fig[1, 2][1, 1], title = "Tᵃᵗ anomaly (K)",                   limits = (nothing, (0, 5e3)))
ax_qˡ  = Axis(fig[1, 3][1, 1], title = "qˡ (kg/kg)",                        limits = (nothing, (0, 5e3)))

ax_Tˡᵃ = Axis(fig[2, 1], title = "Skin temperature (K)", xlabel = "x (m)", ylabel = "Tˡᵃ (K)")
ax_M   = Axis(fig[2, 2], title = "Soil water (kg/m²)",   xlabel = "x (m)", ylabel = "M (kg/m²)")
ax_𝒮   = Axis(fig[2, 3], title = "Surface saturation",   xlabel = "x (m)", ylabel = "𝒮")

n = Observable(1)

wn    = @lift view(interior(w_ts[$n]),  :, 1, :)
Tᵃᵗ_n = @lift begin
    T_xz = view(interior(Tᵃᵗ_ts[$n]), :, 1, :)
    T_xz .- mean(T_xz, dims = 1)
end
qˡn   = @lift view(interior(qˡ_ts[$n]), :, 1, :)
Tˡᵃ_n = @lift vec(interior(Tˡᵃ_ts[$n],  :, 1, 1))
M_n   = @lift vec(interior(M_ts[$n],    :, 1, 1))
𝒮_n   = @lift vec(interior(𝒮_ts[$n],    :, 1, 1))

hm_w   = heatmap!(ax_w,   x_atmos, z_face,   wn;    colormap = :balance, colorrange = (-wlim, wlim))
hm_Tᵃᵗ = heatmap!(ax_Tᵃᵗ, x_atmos, z_center, Tᵃᵗ_n; colormap = :balance, colorrange = (-2, 2))
hm_qˡ  = heatmap!(ax_qˡ,  x_atmos, z_center, qˡn;   colormap = :dense,   colorrange = (0, qˡlim))

Colorbar(fig[1, 1][1, 2], hm_w)
Colorbar(fig[1, 2][1, 2], hm_Tᵃᵗ)
Colorbar(fig[1, 3][1, 2], hm_qˡ)

lines!(ax_Tˡᵃ, x_land, Tˡᵃ_n; color = :black, linewidth = 2)
lines!(ax_M,   x_land, M_n;   color = :black, linewidth = 2)
lines!(ax_𝒮,   x_land, 𝒮_n;   color = :black, linewidth = 2)

ylims!(ax_M, 0, hydrology.maximum_water_storage * 1.05)
ylims!(ax_𝒮, 0, 1.05)

title = @lift "Diurnal convection over heterogeneous slab land, t = " * prettytime(times[$n])
Label(fig[0, 1:3], title, fontsize = 16)

@info "Rendering animation..."
CairoMakie.record(fig, "breeze_over_slab_land.mp4", 1:Nt; framerate = 12) do nn
    n[] = nn
end

@info "Animation saved."
nothing #hide

# ![](breeze_over_slab_land.mp4)
