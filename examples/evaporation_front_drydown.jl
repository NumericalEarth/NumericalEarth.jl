# # Evaporation-front drydown
#
# A standalone `SlabLand` configured with `VariablySaturatedBucketHydrology`
# and a prescribed atmospheric evaporative demand. The slab starts wet
# (`𝒮 = 1`) and dries out under a constant prescribed vapor flux. Once
# `𝒮` drops below the critical saturation `𝒮ᶜ`, the evaporation front
# would have begun to retreat downward (`δᵛ > 0`); in this prescribed-flux
# example we visualize the drydown of `Mˡᵃ` and the diagnostic `𝒮`.
#
# The coupled atmosphere-land variant
# (`breeze_over_evaporation_front_slab_land.jl`) replaces the prescribed
# `Jᵛ` with one solved from the dry-layer vapor balance — when `𝒮 < 𝒮ᶜ`,
# `Jᵛ` falls off as the front retreats, slowing the drydown
# self-consistently.
#
# Land-only is fast (< 1 s for the full integration) and isolates the
# hydrology and energy machinery without the atmosphere.

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf

import CUDA: @allowscalar      # works on CPU too (no-op wrapper)

# ## Grid (one cell — single-column land)

grid = RectilinearGrid(CPU();
                       size = 1,
                       x = (0, 1),
                       y = (0, 1),
                       z = (-1, 0),
                       topology = (Flat, Flat, Bounded))

# ## SlabLand
#
# A 1 m slab, porosity 0.4, saturated at `Mˡᵃ⁺ = ρˡ ν D = 400 kg m⁻²`.
# The critical saturation `𝒮ᶜ = 0.75` is the Manabe (1969) value — above
# it the surface evaporates at full efficiency, below it evaporation is
# moisture-limited. `WaterCoupledForceRestoreEnergy` restores the bulk
# temperature toward `Tᵈᵉᵉᵖ` with a 12-hour time scale.

hydrology = VariablySaturatedBucketHydrology(eltype(grid);
    slab_depth = 1.0,
    porosity = 0.4,
    residual_liquid_fraction = 0.0,
    specific_storage = 1e-3,
    critical_saturation = 0.75,
    retention_curve = VanGenuchtenRetention(α = 1.5, n = 2.0),
    hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-5, n = 2.0),
    deep_liquid_flux = NoDeepLiquidFlux(),
    runoff = NoRunoff())

energy = WaterCoupledForceRestoreEnergy(eltype(grid);
    dry_heat_capacity = 1.5e6,
    liquid_heat_capacity = 4186,
    reference_temperature = 273.15,
    deep_temperature = 290.0,
    deep_time_scale = 12hours,
    advect_deep_liquid_energy = false,
    advect_surface_liquid_energy = false)

land = SlabLand(grid; hydrology, energy)
set!(land; T = 295.0, M = 400.0)   # fully saturated and warmer than Tᵈᵉᵉᵖ

# ## Prescribed atmospheric demand
#
# A constant upward vapor flux `Jᵛ = 1e-5 kg m⁻² s⁻¹` ≈ 0.86 mm day⁻¹.
# In a coupled run this would emerge from the atmosphere-land interface;
# here we just prescribe it.

Jᵛ_demand = 1e-5
fill!(land.fluxes.vapor_flux, Jᵛ_demand)
fill!(land.fluxes.surface_energy_flux, 0.0)   # no radiative/turbulent driving
fill!(land.fluxes.liquid_precipitation_flux, 0.0)
fill!(land.fluxes.liquid_precipitation_temperature, 295.0)

# ## Integrate for 60 days

Δt    = 600.0          # 10 minutes
N     = round(Int, 60days / Δt)
times = Vector{Float64}(undef, N+1)
Ms    = Vector{Float64}(undef, N+1)
𝒮s    = Vector{Float64}(undef, N+1)
Ts    = Vector{Float64}(undef, N+1)

times[1] = 0.0
Ms[1]    = @allowscalar interior(land.water_storage)[1, 1, 1]
𝒮s[1]    = @allowscalar interior(land.saturation)[1, 1, 1]
Ts[1]    = @allowscalar interior(land.temperature)[1, 1, 1]

for n in 1:N
    time_step!(land, Δt)
    times[n+1] = land.clock.time
    Ms[n+1]    = @allowscalar interior(land.water_storage)[1, 1, 1]
    𝒮s[n+1]    = @allowscalar interior(land.saturation)[1, 1, 1]
    Ts[n+1]    = @allowscalar interior(land.temperature)[1, 1, 1]
end

@printf("Mˡᵃ: %.1f → %.1f kg/m² over %.0f days\n", Ms[1], Ms[end], times[end]/days)
@printf("𝒮:   %.3f → %.3f\n", 𝒮s[1], 𝒮s[end])
@printf("Tˡᵃ: %.2f → %.2f K\n", Ts[1], Ts[end])

# ## Plot the drydown
#
# The analytic expectation: with constant `Jᵛ` (independent of `Mˡᵃ`),
# `Mˡᵃ(t) = Mˡᵃ₀ − Jᵛ t`. The simulation should match this exactly.

t_days = times ./ days

fig = Figure(size = (900, 600))

ax_M = Axis(fig[1, 1]; ylabel = "Mˡᵃ (kg m⁻²)",
            title = "Slab water storage")
lines!(ax_M, t_days, Ms; label = "simulated")
lines!(ax_M, t_days, Ms[1] .- Jᵛ_demand .* times; linestyle = :dash, label = "analytic")
axislegend(ax_M; position = :lt)

ax_𝒮 = Axis(fig[2, 1]; ylabel = "𝒮 (–)",
            title = "Surface saturation")
lines!(ax_𝒮, t_days, 𝒮s)
hlines!(ax_𝒮, [0.75]; color = :gray, linestyle = :dot, label = "𝒮ᶜ")
axislegend(ax_𝒮; position = :rt)

ax_T = Axis(fig[3, 1]; xlabel = "t (days)", ylabel = "Tˡᵃ (K)",
            title = "Bulk land temperature")
lines!(ax_T, t_days, Ts)
hlines!(ax_T, [290]; color = :gray, linestyle = :dot, label = "Tᵈᵉᵉᵖ")
axislegend(ax_T; position = :rt)

save("evaporation_front_drydown.png", fig; px_per_unit = 2)
nothing # hide

# ![](evaporation_front_drydown.png)
#
# The water storage decays linearly because `Jᵛ` is prescribed constant —
# the model conservation is bit-exact against the analytic line. In the
# coupled version, `Jᵛ` would *decrease* once `𝒮 < 𝒮ᶜ` (the front
# retreats), so the dry-down would slow and asymptote.
