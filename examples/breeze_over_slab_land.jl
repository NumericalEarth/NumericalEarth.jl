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
# Coupling:
#   - `AtmosphereLandModel(atmos, slab_land)` wires turbulent surface
#     fluxes (sensible, latent, momentum) through Monin–Obukhov
#     similarity theory, with the surface specific humidity reduced by
#     the slab's β-factor.
#   - RRTMGP reads `slab_land.state.T` as its surface temperature each
#     step and computes spectrally-resolved SW and LW fluxes.
#   - A callback adds the net surface radiative flux back into the
#     slab's `net_energy_flux`, closing the coupled surface energy
#     balance.

using NumericalEarth
using Breeze
using Oceananigans
using Oceananigans.Units
using RRTMGP
using NCDatasets
using Printf, Random, Statistics
using Oceananigans.Utils.KernelAbstractions: @kernel, @index
using Dates: DateTime
using CairoMakie

Random.seed!(2025)

# ## Grid
#
# A 2D vertical slice: periodic in x, flat in y, bounded in z. Vertical
# stretching gives fine 100 m cells in the boundary layer (z ≤ 3 km),
# transitioning to 1 km cells in the stratosphere (z up to 15 km).

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

# ## Reference state and dynamics

p₀ = 101325  # surface pressure (Pa)
θ₀ = 300     # reference potential temperature (K)

constants = ThermodynamicConstants()

reference_state = ReferenceState(grid, constants;
                                 surface_pressure = p₀,
                                 potential_temperature = θ₀,
                                 vapor_mass_fraction = 0)

dynamics = AnelasticDynamics(reference_state)

# ## Background atmosphere
#
# Modern trace gas concentrations plus a tropical ozone profile,
# required by RRTMGP for spectrally-resolved absorption.

@inline function tropical_ozone(z)
    troposphere_O₃ = 30e-9 * (1 + 0.5 * z / 10_000)
    zˢᵗ = 25e3
    Hˢᵗ = 5e3
    stratosphere_O₃ = 8e-6 * exp(-((z - zˢᵗ) / Hˢᵗ)^2)
    χˢᵗ = 1 / (1 + exp(-(z - 15e3) / 2))
    return troposphere_O₃ * (1 - χˢᵗ) + stratosphere_O₃ * χˢᵗ
end

background_atmosphere = BackgroundAtmosphere(CO₂ = 420e-6,
                                             CH₄ = 1.8e-6,
                                             N₂O = 330e-9,
                                             O₃  = tropical_ozone)

# ## Heterogeneous slab land
#
# A 1D land grid (size Nx, flat in y and z) carrying skin temperature,
# soil moisture, and moisture availability. The land grid spans the
# same x extent as the atmosphere so that the slab T can serve directly
# as RRTMGP's surface temperature.

land_grid = RectilinearGrid(arch;
                            size = Nx,
                            x = (-Lx/2, Lx/2),
                            halo = grid.Hx,
                            topology = (Periodic, Flat, Flat))

ρcH = 1500.0 * 1480.0 * 0.10    # J m⁻² K⁻¹

energy    = SlabEnergy(eltype(land_grid); dry_heat_capacity = ρcH)
hydrology = BucketHydrology(eltype(land_grid);
                            field_capacity   = 150.0,
                            critical_wetness = 0.75)
surface   = ConstantSurfaceProperties(eltype(land_grid);
                                      momentum_roughness_length = 0.1,
                                      scalar_roughness_length   = 0.01)

slab_land = SlabLand(land_grid; energy, hydrology, surface)

# Gaussian moisture pattern: saturated in the center, dry at the edges.

T₀     = 295.0                        # initial skin temperature (K)
W_wet  = 0.95 * hydrology.field_capacity
σ_wet  = Lx / 8

W_init(x) = W_wet * exp(-(x/σ_wet)^2)

set!(slab_land.state.T, T₀)
set!(slab_land.state.W, W_init)
Oceananigans.TimeSteppers.update_state!(slab_land)

# ## Radiation
#
# All-sky RRTMGP at 15°N starting at the equinox, midnight local. The
# `surface_temperature` is the slab land's prognostic skin temperature,
# so radiation responds to surface heating and cooling in real time.

latitude = 15

radiation = RadiativeTransferModel(grid, AllSkyOptics(), constants;
                                   surface_temperature = slab_land.state.T,
                                   surface_albedo      = 0.20,
                                   surface_emissivity  = 0.95,
                                   solar_constant      = 1361,
                                   background_atmosphere,
                                   solar_position = ApparentSolarPosition(coordinate = (0, latitude),
                                                                          epoch      = DateTime(2024, 3, 20, 0, 0, 0)),
                                   schedule = TimeInterval(10minutes),
                                   liquid_effective_radius = ConstantRadiusParticles(10e-6),
                                   ice_effective_radius    = ConstantRadiusParticles(30e-6))

# ## Atmosphere model
#
# `atmosphere_simulation` wires Breeze's `AtmosphereModel` with bottom
# boundary conditions that the `EarthSystemModel` coupler fills with
# similarity-theory turbulent fluxes. Additional Breeze kwargs are
# forwarded — here we pass the RRTMGP radiation and an f-plane Coriolis.

coriolis = FPlane(latitude = latitude)

atmos = atmosphere_simulation(grid;
                              surface_pressure       = p₀,
                              potential_temperature  = θ₀,
                              radiation,
                              coriolis)

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

Tᵢ(x, z)  = Tᵇᵍ(z) + δT * (rand() - 0.5) * (z < zδ)
ℋᵢ(x, z)  = (0.5 + 1e-2 * (rand() - 0.5)) * (z < zδ)

set!(atmos; T = Tᵢ, ℋ = ℋᵢ)

# ## Coupled model

model = AtmosphereLandModel(atmos, slab_land)

Δt        = 2.0
stop_time = 24hours

simulation = Simulation(model; Δt, stop_time)

# ## Radiative coupling to the land
#
# Breeze's sign convention is "positive = upward". The net radiative
# flux *into* the surface is therefore `-(↑LW + ↓LW + ↓SW)` at z = 0.
# We add this contribution to the slab's `net_energy_flux` accumulator
# every step, after the coupler has populated the turbulent
# (sensible + latent) flux.

@kernel function _apply_rrtmgp_to_land!(Q, ℐ_lw_up, ℐ_lw_dn, ℐ_sw_dn)
    i, j = @index(Global, NTuple)
    @inbounds Q[i, j, 1] -= (ℐ_lw_up[i, j, 1] + ℐ_lw_dn[i, j, 1] + ℐ_sw_dn[i, j, 1])
end

function apply_rrtmgp_to_land!(sim)
    cm = sim.model
    rad = cm.atmosphere.radiation
    land = cm.land
    Q = land.fluxes.net_energy_flux
    arch = Oceananigans.Architectures.architecture(land.grid)
    Oceananigans.Utils.launch!(arch, land.grid, :xy,
                               _apply_rrtmgp_to_land!,
                               Q,
                               rad.upwelling_longwave_flux,
                               rad.downwelling_longwave_flux,
                               rad.downwelling_shortwave_flux)
    return nothing
end

add_callback!(simulation, apply_rrtmgp_to_land!, IterationInterval(1))

# ## Progress

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    atmos = sim.model.atmosphere
    T = atmos.temperature
    u, _, w = atmos.velocities
    wmax = maximum(abs, w)
    Tmin, Tmax = extrema(T)

    Tg = sim.model.land.state.T
    Tg_min, Tg_max = extrema(Tg)

    rad = atmos.radiation
    OLR = mean(view(rad.upwelling_longwave_flux, :, 1, Nz+1))

    msg = @sprintf("iter %5d, t %8s, Δt %5.2fs, wall %6s, max|w| %4.2f m/s, T [%5.1f,%5.1f] K, Tg [%5.1f,%5.1f] K, OLR %5.1f W/m²",
                   iteration(sim), prettytime(sim), sim.Δt, prettytime(elapsed),
                   wmax, Tmin, Tmax, Tg_min, Tg_max, OLR)
    @info msg

    wall_clock[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(200))

# ## Output

u, _, w = atmos.velocities
T  = atmos.temperature
qᵛ = atmos.microphysical_fields.qᵛ
qˡ = atmos.microphysical_fields.qˡ

simulation.output_writers[:atmos] = JLD2Writer(model, (; u, w, T, qᵛ, qˡ);
                                               filename = "breeze_slab_land_atmos",
                                               schedule = TimeInterval(10minutes),
                                               overwrite_existing = true)

simulation.output_writers[:land] = JLD2Writer(model,
                                              (; Tg = slab_land.state.T,
                                                  W  = slab_land.state.W,
                                                  β  = slab_land.state.moisture_availability);
                                              filename = "breeze_slab_land_surface",
                                              schedule = TimeInterval(10minutes),
                                              overwrite_existing = true)

# ## Run

@info "Running coupled simulation..."
run!(simulation)
@info "Simulation complete."

# ## Animation
#
# Top row shows the vertical slice (x–z) of moist convection: vertical
# velocity (w), temperature anomaly from the horizontal mean, and cloud
# liquid water. Bottom row shows the land state along x: skin
# temperature, soil moisture, and moisture availability β.

w_ts  = FieldTimeSeries("breeze_slab_land_atmos.jld2", "w"; architecture=CPU())
T_ts  = FieldTimeSeries("breeze_slab_land_atmos.jld2", "T"; architecture=CPU())
qˡ_ts = FieldTimeSeries("breeze_slab_land_atmos.jld2", "qˡ"; architecture=CPU())
Tg_ts = FieldTimeSeries("breeze_slab_land_surface.jld2", "Tg"; architecture=CPU())
W_ts  = FieldTimeSeries("breeze_slab_land_surface.jld2", "W"; architecture=CPU())
β_ts  = FieldTimeSeries("breeze_slab_land_surface.jld2", "β"; architecture=CPU())

times = w_ts.times
Nt = length(times)

x_atmos  = xnodes(grid, Center())
z_center = znodes(grid, Center())
z_face   = znodes(grid, Face())
x_land   = xnodes(land_grid, Center())

wlim  = maximum(abs, w_ts) / 2
qˡlim = max(1e-6, maximum(qˡ_ts) / 2)

fig = Figure(size = (1500, 800), fontsize = 13)

ax_w  = Axis(fig[1, 1], title = "w (m/s)", ylabel = "z (m)",
             limits = (nothing, (0, 5e3)))
ax_T  = Axis(fig[1, 2], title = "T anomaly (K)",
             limits = (nothing, (0, 5e3)))
ax_qˡ = Axis(fig[1, 3], title = "qˡ (kg/kg)",
             limits = (nothing, (0, 5e3)))

ax_Tg = Axis(fig[2, 1], title = "Skin temperature (K)",  xlabel = "x (m)", ylabel = "T_g (K)")
ax_W  = Axis(fig[2, 2], title = "Soil water (kg/m²)",    xlabel = "x (m)", ylabel = "W")
ax_β  = Axis(fig[2, 3], title = "Moisture availability", xlabel = "x (m)", ylabel = "β")

n = Observable(1)

wn  = @lift view(interior(w_ts[$n]),  :, 1, :)
Tn  = @lift begin
    T_xz = view(interior(T_ts[$n]), :, 1, :)
    T_xz .- mean(T_xz, dims = 1)
end
qˡn = @lift view(interior(qˡ_ts[$n]), :, 1, :)

Tg_n = @lift vec(interior(Tg_ts[$n], :, 1, 1))
W_n  = @lift vec(interior(W_ts[$n],  :, 1, 1))
β_n  = @lift vec(interior(β_ts[$n],  :, 1, 1))

heatmap!(ax_w,  x_atmos, z_face,   wn;  colormap = :balance, colorrange = (-wlim, wlim))
heatmap!(ax_T,  x_atmos, z_center, Tn;  colormap = :balance, colorrange = (-2, 2))
heatmap!(ax_qˡ, x_atmos, z_center, qˡn; colormap = :dense,   colorrange = (0, qˡlim))

lines!(ax_Tg, x_land, Tg_n; color = :black, linewidth = 2)
lines!(ax_W,  x_land, W_n;  color = :black, linewidth = 2)
lines!(ax_β,  x_land, β_n;  color = :black, linewidth = 2)

ylims!(ax_W, 0, hydrology.field_capacity * 1.05)
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
