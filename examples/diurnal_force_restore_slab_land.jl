# # Diurnal force-restore slab land
#
# A standalone `SlabLand` with `WaterCoupledForceRestoreEnergy` driven by
# a prescribed sinusoidal net surface energy flux that mimics a diurnal
# radiation cycle. Highlights the `Mˡᵃ`-dependent heat capacity — a wet
# slab has higher thermal inertia and a smaller diurnal temperature
# amplitude than a dry slab, even with the same incoming radiation.
#
# The energy budget:
#
#     dEˡᵃ/dt = Λᵈᵉᵉᵖ(Tᵈᵉᵉᵖ − Tˡᵃ) − Jᴱ_s
#     dTˡᵃ/dt = (dEˡᵃ/dt − cˡ(Tˡᵃ − Tᵣ) dMˡᵃ/dt) / C(Mˡᵃ),  C(Mˡᵃ) = C_dry + cˡ Mˡᵃ.
#
# Here `Jᴱ_s = −F_sun(t)` is a prescribed surface energy flux (positive
# upward in our sign convention; net incoming radiation is negative `Jᴱ_s`),
# `Mˡᵃ` is held constant (no fluxes), and the force-restore relaxes
# `Tˡᵃ → Tᵈᵉᵉᵖ` on the deep time scale `τᵈᵉᵉᵖ`.

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf

import CUDA: @allowscalar

# ## Grid: two columns — one wet (Mˡᵃ = 400), one dry (Mˡᵃ = 0)

grid = RectilinearGrid(CPU();
                       size = (2, 1),
                       x = (0, 2),
                       y = (0, 1),
                       z = (-1, 0),
                       topology = (Bounded, Flat, Bounded))

hydrology = VariablySaturatedBucketHydrology(eltype(grid);
    slab_depth = 1.0, porosity = 0.4, specific_storage = 1e-3,
    critical_saturation = 0.75,
    retention_curve = VanGenuchtenRetention(α = 1.5, n = 2.0),
    hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-6, n = 2.0),
    deep_liquid_flux = NoDeepLiquidFlux(), runoff = NoRunoff())

energy = WaterCoupledForceRestoreEnergy(eltype(grid);
    dry_heat_capacity = 1.5e6,
    liquid_heat_capacity = 4186,
    reference_temperature = 273.15,
    deep_temperature = 290.0,
    deep_time_scale = 24hours,
    advect_deep_liquid_energy = false,
    advect_surface_liquid_energy = false)

land = SlabLand(grid; hydrology, energy)
set!(land; T = 290.0)
@allowscalar begin
    interior(land.water_storage)[1, 1, 1] = 400.0   # wet column
    interior(land.water_storage)[2, 1, 1] = 0.0     # dry column
end
Oceananigans.TimeSteppers.update_state!(land)        # refresh 𝒮 diagnostic

# ## Time loop with sinusoidal forcing
#
# Net surface energy flux mimics a diurnal radiative cycle peaking at
# ±400 W m⁻² (about ±0.4 kW m⁻² midday SW − LW balance). Positive `Jᴱ_s`
# (loss to atmosphere) at night, negative `Jᴱ_s` (gain from sun) at noon.

ω    = 2π / day
F₀   = 400.0     # W m⁻², diurnal amplitude
Δt   = 600.0     # 10 minutes
N    = round(Int, 5days / Δt)

times = Vector{Float64}(undef, N+1)
T_wet = Vector{Float64}(undef, N+1)
T_dry = Vector{Float64}(undef, N+1)
M_wet = Vector{Float64}(undef, N+1)
M_dry = Vector{Float64}(undef, N+1)
flux  = Vector{Float64}(undef, N+1)

times[1] = 0.0
T_wet[1] = @allowscalar interior(land.temperature)[1, 1, 1]
T_dry[1] = @allowscalar interior(land.temperature)[2, 1, 1]
M_wet[1] = @allowscalar interior(land.water_storage)[1, 1, 1]
M_dry[1] = @allowscalar interior(land.water_storage)[2, 1, 1]
flux[1]  = 0.0   # phase: night start ⇒ no SW

fill!(land.fluxes.vapor_flux, 0)
fill!(land.fluxes.liquid_precipitation_flux, 0)
fill!(land.fluxes.liquid_precipitation_temperature, 290.0)

for n in 1:N
    # Sinusoidal net surface energy flux at this step's midpoint.
    t̄ = land.clock.time + Δt/2
    Jᴱs = -F₀ * sin(ω * t̄)   # negative at noon (energy enters)
    fill!(land.fluxes.surface_energy_flux, Jᴱs)
    time_step!(land, Δt)

    times[n+1] = land.clock.time
    T_wet[n+1] = @allowscalar interior(land.temperature)[1, 1, 1]
    T_dry[n+1] = @allowscalar interior(land.temperature)[2, 1, 1]
    M_wet[n+1] = @allowscalar interior(land.water_storage)[1, 1, 1]
    M_dry[n+1] = @allowscalar interior(land.water_storage)[2, 1, 1]
    flux[n+1]  = Jᴱs
end

@printf("Wet column (Mˡᵃ = %.0f): Tˡᵃ amplitude = %.2f K\n",
        M_wet[end], (maximum(T_wet) - minimum(T_wet)) / 2)
@printf("Dry column (Mˡᵃ = %.0f): Tˡᵃ amplitude = %.2f K\n",
        M_dry[end], (maximum(T_dry) - minimum(T_dry)) / 2)

# Expected ratio of amplitudes: C_dry / (C_dry + cˡ Mˡᵃ)
# = 1.5e6 / (1.5e6 + 4186*400) = 1.5e6 / 3.17e6 ≈ 0.47
amp_ratio_expected = 1.5e6 / (1.5e6 + 4186 * 400)
amp_ratio_actual   = (maximum(T_dry) - minimum(T_dry)) / (maximum(T_wet) - minimum(T_wet))
@printf("Amplitude ratio (wet/dry): expected %.2f, simulated %.2f\n",
        1 / amp_ratio_expected, amp_ratio_actual)

# ## Plot

t_hr = times ./ hour

fig = Figure(size = (900, 600))

ax_F = Axis(fig[1, 1]; ylabel = "Jᴱ_s (W m⁻²)",
            title = "Prescribed diurnal forcing (positive = upward)")
lines!(ax_F, t_hr, flux)
hlines!(ax_F, [0]; color = :gray, linestyle = :dot)

ax_T = Axis(fig[2, 1]; ylabel = "Tˡᵃ (K)",
            title = "Diurnal temperature response: M-dependent heat capacity")
lines!(ax_T, t_hr, T_wet; label = "wet (Mˡᵃ = 400)")
lines!(ax_T, t_hr, T_dry; label = "dry (Mˡᵃ = 0)")
hlines!(ax_T, [290]; color = :gray, linestyle = :dot, label = "Tᵈᵉᵉᵖ")
axislegend(ax_T; position = :rt)

ax_M = Axis(fig[3, 1]; xlabel = "t (hours)", ylabel = "Mˡᵃ (kg m⁻²)",
            title = "Water storage (held constant — no surface vapor flux)")
lines!(ax_M, t_hr, M_wet; label = "wet")
lines!(ax_M, t_hr, M_dry; label = "dry")
axislegend(ax_M; position = :rc)

save("diurnal_force_restore_slab_land.png", fig; px_per_unit = 2)
nothing # hide

# ![](diurnal_force_restore_slab_land.png)
#
# The dry column has roughly twice the diurnal amplitude of the wet
# column under identical forcing, because the wet slab's heat capacity
# `C(Mˡᵃ) = C_dry + cˡ Mˡᵃ` is about 2.1× larger when fully saturated.
# Both columns relax toward `Tᵈᵉᵉᵖ = 290 K` over the 24-hour time scale.
