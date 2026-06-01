# # Breeze atmosphere over evaporation-front slab land
#
# A 2D Breeze atmospheric LES coupled to a `SlabLand` running
# `VariablySaturatedBucketHydrology` + `WaterCoupledForceRestoreEnergy`,
# with the new `EvaporationFrontHumidity` setting the atmosphere-facing
# specific humidity from a dry-layer vapor balance instead of a
# saturation-times-efficiency closure.
#
# The land has a wet patch in the middle (`𝒮 ≈ 1`, the front sits at the
# surface) and bone-dry edges (`𝒮 = 0`, the front retreats to `δᵛ_max`).
# Under the same wind and the same warm surface temperature, the wet
# patch evaporates strongly (large `Jᵛ`, large latent heat) while the
# dry edges cannot: the dry-layer piston velocity
# `wᵈ = Dᵛ_eff/max(δᵛ, δᵛ_min)` collapses with `δᵛ = δᵛ_max`. This is
# the classic moisture-availability contrast, but produced
# self-consistently from a dry-layer Fickian vapor balance rather than
# prescribed `β(𝒮)`.
#
# This example skips the full RRTMGP radiation chain that the legacy
# `breeze_over_slab_land.jl` uses, instead driving the contrast with a
# prescribed sensible-warming surface temperature. That keeps the
# example fast (< 60 s on CPU) while still exercising the full
# coupling stack — atmosphere ↔ interface ↔ hydrology ↔ energy — end-to-end.

using NumericalEarth
using Breeze
using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf, Statistics

import CUDA: @allowscalar

# ## Grids
#
# 2D vertical slice, periodic in `x`, flat in `y`, bounded in `z`. Small
# grid (16 × 16 over 4 km × 1 km) so CI finishes in seconds.

Nx, Nz = 16, 16
Lx, Lz = 4kilometers, 1kilometer

grid = RectilinearGrid(CPU();
                       size = (Nx, Nz), halo = (5, 5),
                       x = (-Lx/2, Lx/2),
                       z = (0, Lz),
                       topology = (Periodic, Flat, Bounded))

land_grid = RectilinearGrid(CPU();
                            size = Nx, halo = grid.Hx,
                            x = (-Lx/2, Lx/2),
                            topology = (Periodic, Flat, Flat))

# ## Slab land with the new closures

hydrology = VariablySaturatedBucketHydrology(eltype(land_grid);
    slab_depth = 1.0,
    porosity = 0.4,
    residual_liquid_fraction = 0.0,
    specific_storage = 1e-3,
    critical_saturation = 0.5,
    retention_curve = VanGenuchtenRetention(α = 1.0, n = 2.0),
    hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-6, n = 2.0),
    deep_liquid_flux = NoDeepLiquidFlux(),
    runoff = NoRunoff())

energy = WaterCoupledForceRestoreEnergy(eltype(land_grid);
    dry_heat_capacity = 1.5e6,
    liquid_heat_capacity = 4186,
    reference_temperature = 273.15,
    deep_temperature = 295.0,
    deep_time_scale = 12hours,
    advect_deep_liquid_energy = false,
    advect_surface_liquid_energy = false)

slab_land = SlabLand(land_grid; hydrology, energy)

# Wet patch in the middle (Gaussian Mˡᵃ), dry edges.
M_wet = 400.0
σ_wet = Lx / 8
M_init(x) = M_wet * exp(-(x / σ_wet)^2)

set!(slab_land; T = 297.0)
set!(slab_land.water_storage, M_init)
Oceananigans.TimeSteppers.update_state!(slab_land)

# ## Atmosphere — neutral profile, light wind, slightly cooler than land

atmos = atmosphere_simulation(grid; potential_temperature = 295.0)
set!(atmos.model;
     θ = atmos.model.dynamics.reference_state.potential_temperature,
     u = 2)

# ## Atmosphere-land interface with `EvaporationFrontHumidity`
#
# Uses default `BulkTemperature()` for `Tⁱⁿ` (= `Tˡᵃ` here); the
# evaporation-front humidity formulation handles the wet/dry contrast
# through variable dry-layer piston velocity `wᵈ`.

al_interface = atmosphere_land_interface(slab_land.grid, atmos, slab_land;
    specific_humidity = EvaporationFrontHumidity(;
        evaporation_front_depth = StorageBasedEvaporationFrontDepth(
            maximum_front_depth = 0.05,
            critical_saturation = 0.5,
            front_depth_exponent = 2.0),
        vapor_exchange = DryLayerVaporPistonVelocity(
            minimum_front_depth = 1e-4,
            molecular_diffusivity = 2.5e-5,
            tortuosity_model = :millington_quirk),
        thermal_exchange_depth = 0.10,
        porosity = 0.4))

model = AtmosphereLandModel(atmos, slab_land;
                            atmosphere_land_interface = al_interface)

# ## Time integration

simulation = Simulation(model; Δt = 1.0, stop_iteration = 60)

run!(simulation)

# Inspect the steady-state(ish) wet/dry contrast.

Jv = Array(interior(slab_land.fluxes.vapor_flux))[:, 1, 1]
𝒮  = Array(interior(slab_land.saturation))[:, 1, 1]
M  = Array(interior(slab_land.water_storage))[:, 1, 1]
T  = Array(interior(slab_land.temperature))[:, 1, 1]
xs = Array(xnodes(slab_land.water_storage))

@printf("Wet center:  𝒮 = %.2f, Jᵛ = %.2e kg/m²/s\n",
        𝒮[Nx÷2], Jv[Nx÷2])
@printf("Dry edge:    𝒮 = %.2f, Jᵛ = %.2e kg/m²/s\n",
        𝒮[1], Jv[1])
@printf("Ratio (wet/dry): Jᵛ_wet / Jᵛ_dry = %.1f\n",
        Jv[Nx÷2] / max(Jv[1], eps()))

# ## Plot

fig = Figure(size = (900, 700))

ax_M = Axis(fig[1, 1]; ylabel = "Mˡᵃ (kg m⁻²)",
            title = "Slab water mass (initial Gaussian)")
lines!(ax_M, xs ./ 1000, M)

ax_𝒮 = Axis(fig[2, 1]; ylabel = "𝒮 (–)",
            title = "Surface saturation")
lines!(ax_𝒮, xs ./ 1000, 𝒮)
hlines!(ax_𝒮, [0.5]; color = :gray, linestyle = :dot, label = "𝒮ᶜ")
axislegend(ax_𝒮; position = :rb)

ax_Jv = Axis(fig[3, 1]; xlabel = "x (km)", ylabel = "Jᵛ (kg m⁻² s⁻¹)",
            title = "Upward vapor flux — wet/dry contrast")
lines!(ax_Jv, xs ./ 1000, Jv)

save("breeze_over_evaporation_front_slab_land.png", fig; px_per_unit = 2)
nothing # hide

# ![](breeze_over_evaporation_front_slab_land.png)
#
# The wet patch (center) evaporates roughly an order of magnitude
# faster than the dry edges. The contrast comes entirely from the
# saturation-dependent dry-layer depth `δᵛ(𝒮)` and the resulting piston
# velocity `wᵈ` — no `β(𝒮)` scaling is prescribed.
