# # Diurnal radiative convection over heterogeneous slab land (2D)
#
# A 2D Breeze atmospheric large eddy simulation (LES) coupled to a
# `SlabLand` with spatially-varying surface moisture, driven by full
# RRTMGP all-sky radiation with a diurnal cycle.
#
# The central wet patch of the land (`𝒮 ≈ 1`, full evaporation efficiency)
# evaporates strongly during the day, so incoming radiation is partitioned
# into latent heat. The dry edges (`𝒮 = 0`) cannot evaporate, so all the net
# radiation goes into sensible heat, which produces strong surface heating
# and a vigorous dry convective boundary layer. At the wet/dry boundary
# the contrast drives a low-level circulation similar to a sea breeze.
#
# Coupling lives entirely in the `EarthSystemModel`:
#   * `AtmosphereLandModel(atmos, slab_land; radiation)` wires turbulent
#     surface fluxes (sensible, latent, momentum) through Monin–Obukhov
#     similarity theory with land stability functions
#     (`atmosphere_land_stability_functions`, the Businger–Dyer /
#     Large–Yeager form), and hands the RRTMGP `RadiativeTransferModel`
#     to the coupled model.
#   * The atmosphere is built with a skeleton `CoupledRadiation`
#     placeholder; the coupled-model constructor materializes it to alias
#     `radiative_transfer_model.flux_divergence` so Breeze's tendency
#     machinery reads directly from the RTM's flux divergence.
#   * The atmosphere's own `update_state!` drives the RRTMGP solve
#     through the proxy, following the RTM's `schedule`.
#   * The net surface shortwave and longwave fluxes from the RTM feed the
#     slab's `surface_energy_flux` via `apply_air_land_radiative_fluxes!`,
#     which closes the surface energy balance without any callbacks.

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
# A 2D vertical slice: periodic in x, flat in y, and bounded in z. The vertical
# grid has fine 100 m cells in the boundary layer (z ≤ 3 km), is stretched
# between 3 km and 8 km, and has 1 km cells from 8 km up to the 15 km top.

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
# A 1D land grid (size Nx, flat in y and z) carries the skin temperature,
# soil water, and surface saturation. The land grid spans the same x
# extent as the atmosphere, so the slab temperature can serve directly as the
# RRTMGP surface temperature.

land_grid = RectilinearGrid(arch;
                            size = Nx,
                            x = (-Lx/2, Lx/2),
                            halo = grid.Hx,
                            topology = (Periodic, Flat, Flat))

# We use a conservative, variably saturated hydrology. Its storage is the
# augmented liquid fraction `ϑˡ = θˡ + max(Π, 0)/hˢˢ`, so wetting beyond
# saturation (`Mˡᵃ > Mˡᵃ⁺ = ρˡ ν hˡᵃ`) is admitted as a positive pressure head.
# We use Van Genuchten retention and conductivity, no deep drainage at the
# bottom, and an infiltration-capacity runoff closure for any precipitation
# that exceeds the soil capacity.

hydrology = VariablySaturatedHydrology(eltype(land_grid);
    slab_depth = 1.0,
    porosity = 0.4,
    residual_liquid_fraction = 0.05,
    storage_height = 1000,
    retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 1, pore_size_uniformity = 2),
    hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 1e-7, pore_size_uniformity = 2),
    deep_liquid_flux = NoDeepLiquidFlux(),
    runoff = InfiltrationCapacityRunoff(infiltration_capacity = 1e-3))

# The energy budget is coupled to the water mass through the areal heat capacity
# `cˡᵃ(Mˡᵃ) = cᵈʳʸ + cˡ Mˡᵃ`, and `Tˡᵃ` is updated conservatively: adding or
# removing water at the slab temperature leaves `Tˡᵃ` unchanged.
#
# The `deep_temperature` should be close to the radiative–convective
# equilibrium of the surface (~310 K here). A much colder restoring target
# holds the thin dry patches, whose heat capacity is low, far below their
# daytime equilibrium and destabilizes the coupled boundary layer.

energy = WaterCoupledEnergy(eltype(land_grid);
    dry_heat_capacity = 1480 * 1500 * 0.10,
    liquid_heat_capacity = 4186,
    reference_temperature = 273.15,
    deep_temperature = 310.0,
    deep_time_scale = 12hours,
    advect_deep_liquid_energy = false,
    advect_surface_liquid_energy = false)

slab_land = SlabLand(land_grid; hydrology, energy)

# ### Surface saturation and the wet/dry contrast
#
# The hydrology exposes the diagnostic surface saturation
# `𝒮 = clamp(θˡ/ν, 0, 1) ∈ [0, 1]`. The interface's
# [`DryLayerHumidity`](@ref) closure further below solves for the
# atmosphere-facing specific humidity `qⁱⁿ` from a vapor-flux balance through
# an unresolved dry layer at saturation-dependent depth
# `δᵛ(𝒮) = δᵛ_max[1 − min(𝒮/𝒮ᶜ, 1)]^η`. The wet center (`𝒮 ≥ 𝒮ᶜ`) has
# `δᵛ = 0` and a saturated skin (`qⁱⁿ = qᵛ⁺`). At the dry edges `δᵛ → δᵛ_max`,
# and the small dry-layer piston velocity `wᵈ = Dᵛ_eff/δᵛ` shuts off
# evaporation entirely. The wet/dry contrast thus emerges from the dry-layer
# physics, with no prescribed evaporation efficiency `β(𝒮)`.
#
# We initialize `Mˡᵃ` as a Gaussian centered at the domain midpoint.

T₀   = 295 # K
Mˡᵃ⁺ = slab_land.hydrology.porosity * slab_land.hydrology.slab_depth * 1000 # ρˡ ν hˡᵃ
M₀   = 0.95 * Mˡᵃ⁺
σ    = Lx / 8

Mᵢ(x) = M₀ * exp(-(x / σ)^2)

set!(slab_land.temperature, T₀)
set!(slab_land.water_storage, Mᵢ)
Oceananigans.TimeSteppers.update_state!(slab_land)

# ## Reference state, dynamics, and a stratospheric sponge
#
# The 15 km column gives RRTMGP a realistic atmosphere, but the initial
# stratosphere is not in radiative equilibrium, and the coarse upper cells
# respond strongly once radiation switches on. A Newtonian relaxation of
# potential temperature toward its initial profile above 8 km anchors the
# stratosphere without affecting the troposphere (as in Breeze's
# `radiative_convection` example). We build the reference state explicitly
# so that the sponge and the radiation share the same thermodynamic constants.

p₀ = 101325    # Pa
θ₀ = 300       # K
latitude = 15

constants = ThermodynamicConstants()
reference_state = ReferenceState(grid, constants;
                                 base_pressure = p₀,
                                 potential_temperature = θ₀)
dynamics = AnelasticDynamics(reference_state)

θᵣ = CenterField(grid) # the initial potential temperature, set below
ρᵣ = reference_state.density
sponge_time_scale = 6hours

@inline function stratospheric_relaxation(i, j, k, grid, clock, model_fields, p)
    @inbounds θ  = model_fields.θ[i, j, k]
    @inbounds θᵣ = p.θᵣ[i, j, k]
    @inbounds ρ  = p.ρᵣ[i, j, k]
    z = Oceananigans.Grids.znode(i, j, k, grid, Center(), Center(), Center())
    α = clamp((z - 8000) / 4000, 0, 1)
    return ρ * (-α * (θ - θᵣ) / p.τ)
end

sponge = Forcing(stratospheric_relaxation; discrete_form = true,
                 parameters = (; θᵣ, ρᵣ, τ = sponge_time_scale))

# ## RRTMGP radiation
#
# All-sky RRTMGP at 15°N, starting at local midnight on the equinox. The
# `surface_temperature` is the prognostic skin temperature of the slab land,
# so the radiation responds to surface heating and cooling as they happen.
#
# Stratospheric radiative balance requires a tropical ozone profile: without
# it the upper column is far from radiative equilibrium and becomes unstable
# when the spectral fluxes are recomputed over the convecting troposphere.

@inline function tropical_ozone(z)
    tropospheric_ozone  = 3e-8 * (1 + 0.5 * z / 1e3)
    stratospheric_ozone = 8e-6 * exp(-((z - 25e3) / 5e3)^2)
    stratospheric_fraction = 1 / (1 + exp(-(z - 15e3) / 2))
    return tropospheric_ozone * (1 - stratospheric_fraction) + stratospheric_ozone * stratospheric_fraction
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
# The atmosphere is built with a skeleton `CoupledRadiation` placeholder,
# which `AtmosphereLandModel` materializes against the RTM below. We therefore
# do not pass a `radiation` keyword argument here.

atmos = atmosphere_simulation(grid; dynamics,
                              forcing  = (; ρθ = sponge),
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

# We recompute the reference state from the horizontal mean. In a tall Float32
# column the default dry-adiabatic reference diverges from the actual
# stratospheric profile and produces density errors that overwhelm Float32
# precision; `set_to_mean!` anchors `ρᵣ` to the current state.

set_to_mean!(reference_state, atmos.model, rescale_densities = true)
set!(θᵣ, liquid_ice_potential_temperature(atmos.model))

# ## Coupled model
#
# Passing `radiation = radiative_transfer_model` here triggers
# `materialize_earth_system_radiation!`, which aliases the atmosphere's
# `CoupledRadiation.flux_divergence` to
# `radiative_transfer_model.flux_divergence` and installs the Breeze-aware
# `apply_air_land_radiative_fluxes!`.

# [`DryLayerHumidity`](@ref) solves for the surface specific humidity from
# a Fickian vapor-flux balance between the saturated soil at depth `δᵛ` and
# the atmosphere.

interface_specific_humidity = DryLayerHumidity(;
    dry_layer_depth = StorageBasedDryLayerDepth(
        maximum_dry_layer_depth = 0.05,
        dry_layer_onset_saturation = 0.5,
        dry_layer_exponent = 2),
    vapor_exchange = DryLayerVaporPistonVelocity(
        minimum_dry_layer_depth = 1e-4,
        molecular_diffusivity = 2.5e-5,
        tortuosity = PowerLawTortuosity()),
    thermal_exchange_depth = 0.10,
    porosity = slab_land.hydrology.porosity)
interface = atmosphere_land_interface(slab_land.grid, atmos, slab_land;
                                      specific_humidity = interface_specific_humidity)

# The atmosphere runs in `Float32` (its clock follows the grid), so the coupled
# model needs a matching `Float32` clock.

model = AtmosphereLandModel(atmos, slab_land; radiation,
                            atmosphere_land_interface = interface,
                            clock = Clock{eltype(grid)}(time = 0))

# The time-step wizard recomputes Δt every iteration, so the time step tracks
# the current CFL number. This matters for a convective LES on a 100 m grid,
# where a cumulus updraft can tighten the vertical CFL constraint within a few
# steps. `max_Δt` caps the time step during the quiescent cold start, when the
# velocities are nearly zero and the advective time scale is unbounded.

simulation = Simulation(model; Δt = 1e-6, stop_time = 3days)
conjure_time_step_wizard!(simulation, IterationInterval(1); cfl = 0.7, max_Δt = 6)

# ## Progress

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    w   = sim.model.atmosphere.model.velocities.w
    Tᵃᵗ = sim.model.atmosphere.model.temperature
    Tˡᵃ = sim.model.land.temperature
    ℐꜛˡʷ = sim.model.radiation.upwelling_longwave_flux

    @info @sprintf("iter %5d, t %8s, Δt %4.1fs, wall %6s, max|w| %4.2f m/s, Tᵃᵗ [%5.1f,%5.1f] K, Tˡᵃ [%5.1f,%5.1f] K, OLR %5.1f W/m²",
                   iteration(sim), prettytime(sim), sim.Δt, prettytime(elapsed),
                   maximum(abs, w), extrema(Tᵃᵗ)..., extrema(Tˡᵃ)...,
                   mean(view(ℐꜛˡʷ, :, 1, Nz+1)))

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
                                               overwrite_files = true)

simulation.output_writers[:land] = JLD2Writer(model,
                                              (; T = slab_land.temperature,
                                                  M = slab_land.water_storage,
                                                  𝒮 = slab_land.saturation);
                                              filename = "breeze_slab_land_surface",
                                              schedule = TimeInterval(10minutes),
                                              overwrite_files = true)

# ## Run

run!(simulation)

# ## Animation
#
# The top row shows x–z slices of the vertical velocity, the temperature anomaly
# from its horizontal mean, and the cloud liquid water. The bottom row shows the
# land state along x: skin temperature, soil water, and surface saturation `𝒮`.

w_ts   = FieldTimeSeries("breeze_slab_land_atmos.jld2",   "w")
Tᵃᵗ_ts = FieldTimeSeries("breeze_slab_land_atmos.jld2",   "T")
qˡ_ts  = FieldTimeSeries("breeze_slab_land_atmos.jld2",   "qˡ")
Tˡᵃ_ts = FieldTimeSeries("breeze_slab_land_surface.jld2", "T")
M_ts   = FieldTimeSeries("breeze_slab_land_surface.jld2", "M")
𝒮_ts   = FieldTimeSeries("breeze_slab_land_surface.jld2", "𝒮")

times = w_ts.times
Nt    = length(times)

wlim = maximum(abs, w_ts) / 2

# Cloud liquid water is sparse, and its peak value is much larger than in a
# typical cloudy cell, so a colorbar scaled to the global maximum washes the
# clouds out. We instead anchor the upper bound to a high quantile of the
# *cloudy* (qˡ > 0) values, so that the bulk of the cloud field spans the colormap.
cloudy_liquid = filter(>(0), interior(qˡ_ts))
qˡlim = isempty(cloudy_liquid) ? 1e-6 : quantile(cloudy_liquid, 0.99)

fig = Figure(size = (1200, 640))

ax_w   = Axis(fig[1, 1][1, 1], limits = (nothing, (0, 5e3)), title = "w (m/s)", ylabel = "z (m)")
ax_Tᵃᵗ = Axis(fig[1, 2][1, 1], limits = (nothing, (0, 5e3)), title = "Tᵃᵗ anomaly (K)")
ax_qˡ  = Axis(fig[1, 3][1, 1], limits = (nothing, (0, 5e3)), title = "qˡ (kg/kg)")

ax_Tˡᵃ = Axis(fig[2, 1], title = "Skin temperature (K)", xlabel = "x (m)", ylabel = "Tˡᵃ (K)")
ax_M   = Axis(fig[2, 2], title = "Soil water (kg/m²)",   xlabel = "x (m)", ylabel = "M (kg/m²)")
ax_𝒮   = Axis(fig[2, 3], title = "Surface saturation",   xlabel = "x (m)", ylabel = "𝒮")

n = Observable(1)

Tᵃᵗ′ = similar(Tᵃᵗ_ts[1])

wn    = @lift w_ts[$n]
Tᵃᵗn  = @lift set!(Tᵃᵗ′, Tᵃᵗ_ts[$n] - Field(Average(Tᵃᵗ_ts[$n], dims=1)))
qˡn   = @lift qˡ_ts[$n]
Tˡᵃn  = @lift Tˡᵃ_ts[$n]
Mn    = @lift M_ts[$n]
𝒮n    = @lift 𝒮_ts[$n]

hm_w   = heatmap!(ax_w,   wn;   colormap = :balance, colorrange = (-wlim, wlim))
hm_Tᵃᵗ = heatmap!(ax_Tᵃᵗ, Tᵃᵗn; colormap = :balance, colorrange = (-2, 2))
hm_qˡ  = heatmap!(ax_qˡ,  qˡn;  colormap = :dense,   colorrange = (0, qˡlim))

Colorbar(fig[1, 1][1, 2], hm_w)
Colorbar(fig[1, 2][1, 2], hm_Tᵃᵗ)
Colorbar(fig[1, 3][1, 2], hm_qˡ)

lines!(ax_Tˡᵃ, Tˡᵃn; color = :black, linewidth = 2)
lines!(ax_M,   Mn;   color = :black, linewidth = 2)
lines!(ax_𝒮,   𝒮n;   color = :black, linewidth = 2)

ylims!(ax_M, 0, Mˡᵃ⁺ * 1.05)
ylims!(ax_𝒮, 0, 1.05)

title = @lift "Diurnal convection over heterogeneous slab land, t = " * prettytime(times[$n])
Label(fig[0, 1:3], title, fontsize = 16)

CairoMakie.record(fig, "breeze_over_slab_land.mp4", 1:Nt; framerate = 12) do nn
    n[] = nn
end
nothing #hide

# ![](breeze_over_slab_land.mp4)
