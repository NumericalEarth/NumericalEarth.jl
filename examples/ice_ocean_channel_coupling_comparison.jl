# ===========================================================================
#  Wind-Driven Ice-Ocean Channel: AB2+FE vs RK3+RK3 Coupling Comparison
# ===========================================================================
#
# Idealized Southern Ocean channel with sea ice.
#
# The setup: a zonally periodic channel (~1000 km × 500 km, 3 ocean layers)
# with cold polar atmospheric forcing and an eastward wind.
# - Cold atmosphere grows ice from above
# - Warm deep water (CDW analog) melts ice from below
# - Eastward wind + Coriolis drives northward Ekman transport, creating
#   upwelling of warm water at the southern boundary and downwelling
#   at the northern boundary
# - Where warm water upwells, ice melts faster → polynya-like features
# - Brine rejection from ice formation destabilizes the water column,
#   feeding back on the circulation
#
# This script compares two coupling strategies:
#   1. AB2 ocean + Forward Euler sea ice  (sequential, loosely coupled)
#   2. RK3 ocean + RK3 sea ice           (tightly coupled at each RK3 stage)
#
# The tight coupling recomputes ice-ocean fluxes at every RK3 substage,
# capturing the rapid feedback between ice thickness changes and ocean
# heat/salt fluxes more accurately than the sequential approach.
#
# ===========================================================================

using NumericalEarth
using Oceananigans
using Oceananigans.Units

using ClimaSeaIce
using ClimaSeaIce: SeaIceModel, SlabSeaIceThermodynamics, PhaseTransitions, ConductiveFlux
using ClimaSeaIce.SeaIceThermodynamics: IceWaterThermalEquilibrium, PrescribedTemperature

using NumericalEarth.EarthSystemModels: ocean_surface_salinity
using NumericalEarth.Oceans

using Printf

# =====================
# Physical parameters
# =====================

# Domain
const Lx = 1000kilometers  # zonal extent (periodic)
const Ly = 500kilometers   # meridional extent (bounded)

# Grid resolution
const Nx = 500  # ~20 km in x
const Ny = 250  # ~20 km in y
const Nz = 20   # 20 ocean layers

# Non-uniform vertical: thin surface layer for resolving ice-ocean exchange
const z_faces = ExponentialDiscretization(20, -500, 0; scale = 200)

# Coriolis (Southern Ocean, ~65°S)
const f₀ = -1.3e-4  # s⁻¹

# Atmospheric state (polar winter)
const Tₐ  = 273.15 - 20  # -20°C in Kelvin
const u₁₀ = 10.0         # 10 m/s eastward wind
const qₐ  = 0.001        # specific humidity (dry cold air)
const Qsw = 0.0          # no shortwave (polar winter)
const Qlw = 180.0        # downwelling longwave (W/m²)

# Ocean initial conditions
const T_surface = -1.0    # °C (cold surface, above freezing for S=34)
const T_deep    = 2.0     # °C (warm CDW analog)
const S_surface = 34.0    # psu
const S_deep    = 34.8    # psu

# Sea ice initial conditions
const h₀ = 0.5   # initial ice thickness (m)
const ℵ₀ = 0.9   # initial ice concentration

# Time stepping
const Δt = 10minutes
const stop_time = 200days

# =====================
# Grid (shared)
# =====================

grid = RectilinearGrid(size = (Nx, Ny, Nz),
                       x = (0, Lx),
                       y = (0, Ly),
                       z = z_faces,
                       topology = (Periodic, Bounded, Bounded))

# =====================
# Initial conditions
# =====================

# Exponential stratification: warm/salty deep, cold/fresh surface
# e-folding scale = 100 m
Tᵢ(x, y, z) = T_deep + (T_surface - T_deep) * exp(z / 100)
Sᵢ(x, y, z) = S_deep + (S_surface - S_deep) * exp(z / 100)

# =====================
# Helper functions
# =====================

"""Create a fresh prescribed atmosphere (each run needs its own to avoid clock contamination).

Uses a 1×1 cell grid covering the ocean domain so that the regridder can compute
valid fractional indices (a fully Flat atmosphere grid triggers a bug when paired
with a non-Flat ocean grid)."""
function build_atmosphere()
    atmosphere_grid  = RectilinearGrid(size = (1, 1),
                                       x = (0, Lx),
                                       y = (0, Ly),
                                       topology = (Periodic, Bounded, Flat))
    atmosphere_times = range(0, 365days, length=3)
    atmosphere = PrescribedAtmosphere(atmosphere_grid, atmosphere_times)

    parent(atmosphere.tracers.T)  .= Tₐ
    parent(atmosphere.velocities.u) .= u₁₀
    parent(atmosphere.tracers.q)  .= qₐ
    parent(atmosphere.downwelling_radiation.shortwave) .= Qsw
    parent(atmosphere.downwelling_radiation.longwave)  .= Qlw

    return atmosphere
end

"""
    build_coupled_simulation(; ocean_timestepper, sea_ice_timestepper, output_prefix)

Build a fully coupled ocean-sea ice simulation with the specified time-stepping strategy.

When both `ocean_timestepper` and `sea_ice_timestepper` are `:SplitRungeKutta3`,
the `EarthSystemModel` automatically dispatches to the tightly coupled time-stepping
method that recomputes ice-ocean fluxes at each RK3 substage.

When the ocean uses `:QuasiAdamsBashforth2` and sea ice uses `:ForwardEuler`,
the default sequential coupling is used (sea ice steps first, then ocean).
"""
function build_coupled_simulation(; ocean_timestepper,
                                    sea_ice_timestepper,
                                    output_prefix)

    # --- Atmosphere (fresh instance) ---
    atmosphere = build_atmosphere()
    radiation  = Radiation(ocean_albedo=0.06, sea_ice_albedo=0.7)

    # --- Ocean ---
    ocean = ocean_simulation(grid;
                             Δt,
                             timestepper = ocean_timestepper,
                             coriolis = FPlane(f=f₀),
                             closure = Oceans.default_ocean_closure(),
                             momentum_advection = WENOVectorInvariant(),
                             tracer_advection = WENO(order=7))

    set!(ocean.model, T=Tᵢ, S=Sᵢ)


    sea_ice = sea_ice_simulation(ocean, grid; advection = WENO(order=7))
    set!(sea_ice.model, h=h₀, ℵ=ℵ₀)

    # --- Coupled Model ---
    coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
    simulation    = Simulation(coupled_model; Δt, stop_time)

    # --- Output Writers ---
    ocean_outputs = merge(ocean.model.tracers, ocean.model.velocities)
    ocean.output_writers[:surface] = JLD2Writer(ocean.model, ocean_outputs;
                                                schedule = TimeInterval(1days),
                                                filename = output_prefix * "_ocean_surface",
                                                indices = (:, :, grid.Nz),
                                                overwrite_existing = true,
                                                array_type = Array{Float32})

    sea_ice_outputs = (; h = sea_ice.model.ice_thickness,
                         ℵ = sea_ice.model.ice_concentration)
    sea_ice.output_writers[:fields] = JLD2Writer(sea_ice.model, sea_ice_outputs;
                                                 schedule = TimeInterval(1days),
                                                 filename = output_prefix * "_sea_ice",
                                                 overwrite_existing = true,
                                                 array_type = Array{Float32})

    # --- Progress callback ---
    wall_clock = Ref(time_ns())
    function progress(sim)
        ocean_sim = sim.model.ocean
        T  = ocean_sim.model.tracers.T
        h  = sim.model.sea_ice.model.ice_thickness
        uₒ = ocean_sim.model.velocities.u
        elapsed = (time_ns() - wall_clock[]) * 1e-9
        wall_clock[] = time_ns()

        @info @sprintf("[%s] Time: %s, wall: %.1f s, max|u|: %.2e m/s, T ∈ [%.2f, %.2f] °C, h ∈ [%.3f, %.3f] m",
                       output_prefix,
                       prettytime(sim),
                       elapsed,
                       maximum(abs, interior(uₒ)),
                       minimum(interior(T)),
                       maximum(interior(T)),
                       minimum(interior(h)),
                       maximum(interior(h)))
    end
    add_callback!(simulation, progress, TimeInterval(5days))

    return simulation
end

# ==============================================
#  Case 1: AB2 Ocean + Forward Euler Sea Ice
#  (sequential, loosely coupled)
# ==============================================

@info "═══════════════════════════════════════════════════"
@info " Case 1: AB2 + Forward Euler (loosely coupled)"
@info "═══════════════════════════════════════════════════"

sim_ab2 = build_coupled_simulation(ocean_timestepper = :QuasiAdamsBashforth2,
                                   sea_ice_timestepper = :ForwardEuler,
                                   output_prefix = "ab2_fe")
run!(sim_ab2)

# ==============================================
#  Case 2: RK3 Ocean + RK3 Sea Ice
#  (tightly coupled at each RK3 stage)
# ==============================================

@info "═══════════════════════════════════════════════════"
@info " Case 2: RK3 + RK3 (tightly coupled)"
@info "═══════════════════════════════════════════════════"

sim_rk3 = build_coupled_simulation(ocean_timestepper = :SplitRungeKutta3,
                                   sea_ice_timestepper = :SplitRungeKutta3,
                                   output_prefix = "rk3_rk3")
run!(sim_rk3)

# ==============================================
#  Summary comparison
# ==============================================

@info "═══════════════════════════════════════════════════"
@info " Comparison Summary (final state)"
@info "═══════════════════════════════════════════════════"

# Extract final state from both simulations
T_ab2 = interior(sim_ab2.model.ocean.model.tracers.T)
S_ab2 = interior(sim_ab2.model.ocean.model.tracers.S)
h_ab2 = interior(sim_ab2.model.sea_ice.model.ice_thickness)
ℵ_ab2 = interior(sim_ab2.model.sea_ice.model.ice_concentration)

T_rk3 = interior(sim_rk3.model.ocean.model.tracers.T)
S_rk3 = interior(sim_rk3.model.ocean.model.tracers.S)
h_rk3 = interior(sim_rk3.model.sea_ice.model.ice_thickness)
ℵ_rk3 = interior(sim_rk3.model.sea_ice.model.ice_concentration)

using Statistics: mean

@info @sprintf("  Ocean SST     — AB2+FE: [%.3f, %.3f] °C (mean %.3f)  |  RK3+RK3: [%.3f, %.3f] °C (mean %.3f)",
               minimum(T_ab2[:, :, Nz]), maximum(T_ab2[:, :, Nz]), mean(T_ab2[:, :, Nz]),
               minimum(T_rk3[:, :, Nz]), maximum(T_rk3[:, :, Nz]), mean(T_rk3[:, :, Nz]))

@info @sprintf("  Ocean SSS     — AB2+FE: [%.3f, %.3f] psu (mean %.3f)  |  RK3+RK3: [%.3f, %.3f] psu (mean %.3f)",
               minimum(S_ab2[:, :, Nz]), maximum(S_ab2[:, :, Nz]), mean(S_ab2[:, :, Nz]),
               minimum(S_rk3[:, :, Nz]), maximum(S_rk3[:, :, Nz]), mean(S_rk3[:, :, Nz]))

@info @sprintf("  Ice thickness — AB2+FE: [%.4f, %.4f] m (mean %.4f)  |  RK3+RK3: [%.4f, %.4f] m (mean %.4f)",
               minimum(h_ab2), maximum(h_ab2), mean(h_ab2),
               minimum(h_rk3), maximum(h_rk3), mean(h_rk3))

@info @sprintf("  Ice conc.     — AB2+FE: [%.4f, %.4f] (mean %.4f)  |  RK3+RK3: [%.4f, %.4f] (mean %.4f)",
               minimum(ℵ_ab2), maximum(ℵ_ab2), mean(ℵ_ab2),
               minimum(ℵ_rk3), maximum(ℵ_rk3), mean(ℵ_rk3))

# Differences
ΔT = T_rk3[:, :, Nz] .- T_ab2[:, :, Nz]
Δh = h_rk3 .- h_ab2

@info @sprintf("  SST difference (RK3 - AB2):  max|ΔT| = %.4f °C, mean ΔT = %.4f °C",
               maximum(abs, ΔT), mean(ΔT))

@info @sprintf("  h   difference (RK3 - AB2):  max|Δh| = %.4f m,  mean Δh = %.4f m",
               maximum(abs, Δh), mean(Δh))

@info ""
@info "Output files:"
@info "  AB2+FE:  ab2_fe_ocean_surface.jld2,  ab2_fe_sea_ice.jld2"
@info "  RK3+RK3: rk3_rk3_ocean_surface.jld2, rk3_rk3_sea_ice.jld2"

# ==============================================
#  Visualization
# ==============================================

using CairoMakie

# Convert grid coordinates to km for axis labels
x_km = range(0, Lx / 1e3, length=Nx)
y_km = range(0, Ly / 1e3, length=Ny)

# --- Final-state snapshot: side-by-side + difference ---

SST_ab2 = T_ab2[:, :, Nz]
SST_rk3 = T_rk3[:, :, Nz]
SSS_ab2 = S_ab2[:, :, Nz]
SSS_rk3 = S_rk3[:, :, Nz]
h_ab2_2d = h_ab2[:, :, 1]
h_rk3_2d = h_rk3[:, :, 1]

ΔSST = SST_rk3 .- SST_ab2
ΔSSS = SSS_rk3 .- SSS_ab2
Δh2d = h_rk3_2d .- h_ab2_2d

fig = Figure(size=(1400, 1000), fontsize=14)

Label(fig[0, :], "Ice-Ocean Channel: AB2+FE vs RK3+RK3 — final state at t = $(prettytime(stop_time))",
      fontsize=18, tellwidth=false)

# Row 1: Sea surface temperature
ax1 = Axis(fig[1, 1]; title="SST — AB2+FE",  xlabel="x (km)", ylabel="y (km)")
ax2 = Axis(fig[1, 2]; title="SST — RK3+RK3", xlabel="x (km)", ylabel="y (km)")
ax3 = Axis(fig[1, 3]; title="SST difference (RK3 − AB2)", xlabel="x (km)", ylabel="y (km)")

Tmin = min(minimum(SST_ab2), minimum(SST_rk3))
Tmax = max(maximum(SST_ab2), maximum(SST_rk3))
hm1 = heatmap!(ax1, x_km, y_km, SST_ab2; colorrange=(Tmin, Tmax), colormap=:thermal)
hm2 = heatmap!(ax2, x_km, y_km, SST_rk3; colorrange=(Tmin, Tmax), colormap=:thermal)
Colorbar(fig[1, 4], hm2; label="°C")

dTlim = maximum(abs, ΔSST)
hm3 = heatmap!(ax3, x_km, y_km, ΔSST; colorrange=(-dTlim, dTlim), colormap=:balance)
Colorbar(fig[1, 5], hm3; label="°C")

# Row 2: Ice thickness
ax4 = Axis(fig[2, 1]; title="Ice thickness — AB2+FE",  xlabel="x (km)", ylabel="y (km)")
ax5 = Axis(fig[2, 2]; title="Ice thickness — RK3+RK3", xlabel="x (km)", ylabel="y (km)")
ax6 = Axis(fig[2, 3]; title="Δh (RK3 − AB2)",          xlabel="x (km)", ylabel="y (km)")

hmin = min(minimum(h_ab2_2d), minimum(h_rk3_2d))
hmax = max(maximum(h_ab2_2d), maximum(h_rk3_2d))
hm4 = heatmap!(ax4, x_km, y_km, h_ab2_2d; colorrange=(hmin, hmax), colormap=:ice)
hm5 = heatmap!(ax5, x_km, y_km, h_rk3_2d; colorrange=(hmin, hmax), colormap=:ice)
Colorbar(fig[2, 4], hm5; label="m")

dhlim = maximum(abs, Δh2d)
hm6 = heatmap!(ax6, x_km, y_km, Δh2d; colorrange=(-dhlim, dhlim), colormap=:balance)
Colorbar(fig[2, 5], hm6; label="m")

# Row 3: Sea surface salinity
ax7 = Axis(fig[3, 1]; title="SSS — AB2+FE",  xlabel="x (km)", ylabel="y (km)")
ax8 = Axis(fig[3, 2]; title="SSS — RK3+RK3", xlabel="x (km)", ylabel="y (km)")
ax9 = Axis(fig[3, 3]; title="ΔSSS (RK3 − AB2)", xlabel="x (km)", ylabel="y (km)")

Smin = min(minimum(SSS_ab2), minimum(SSS_rk3))
Smax = max(maximum(SSS_ab2), maximum(SSS_rk3))
hm7 = heatmap!(ax7, x_km, y_km, SSS_ab2; colorrange=(Smin, Smax), colormap=:haline)
hm8 = heatmap!(ax8, x_km, y_km, SSS_rk3; colorrange=(Smin, Smax), colormap=:haline)
Colorbar(fig[3, 4], hm8; label="psu")

dSlim = maximum(abs, ΔSSS)
hm9 = heatmap!(ax9, x_km, y_km, ΔSSS; colorrange=(-dSlim, dSlim), colormap=:balance)
Colorbar(fig[3, 5], hm9; label="psu")

save("coupling_comparison_snapshot.png", fig)
@info "Saved coupling_comparison_snapshot.png"

# --- Time series: domain-averaged ice thickness and SST evolution ---

T_ab2_ts = FieldTimeSeries("ab2_fe_ocean_surface.jld2",  "T"; backend=OnDisk())
T_rk3_ts = FieldTimeSeries("rk3_rk3_ocean_surface.jld2", "T"; backend=OnDisk())
h_ab2_ts = FieldTimeSeries("ab2_fe_sea_ice.jld2",  "h"; backend=OnDisk())
h_rk3_ts = FieldTimeSeries("rk3_rk3_sea_ice.jld2", "h"; backend=OnDisk())

times = T_ab2_ts.times
Nt = length(times)
days = times ./ 86400  # convert seconds to days

mean_SST_ab2 = [mean(interior(T_ab2_ts[n])) for n in 1:Nt]
mean_SST_rk3 = [mean(interior(T_rk3_ts[n])) for n in 1:Nt]
mean_h_ab2   = [mean(interior(h_ab2_ts[n]))  for n in 1:Nt]
mean_h_rk3   = [mean(interior(h_rk3_ts[n]))  for n in 1:Nt]

fig2 = Figure(size=(1000, 700), fontsize=14)

Label(fig2[0, :], "Domain-averaged evolution: AB2+FE vs RK3+RK3", fontsize=18, tellwidth=false)

ax_sst = Axis(fig2[1, 1]; xlabel="Time (days)", ylabel="Mean SST (°C)",
              title="Sea Surface Temperature")
lines!(ax_sst, days, mean_SST_ab2; label="AB2+FE",  linewidth=2, color=:royalblue)
lines!(ax_sst, days, mean_SST_rk3; label="RK3+RK3", linewidth=2, color=:orangered)
axislegend(ax_sst; position=:rt)

ax_h = Axis(fig2[2, 1]; xlabel="Time (days)", ylabel="Mean ice thickness (m)",
            title="Sea Ice Thickness")
lines!(ax_h, days, mean_h_ab2; label="AB2+FE",  linewidth=2, color=:royalblue)
lines!(ax_h, days, mean_h_rk3; label="RK3+RK3", linewidth=2, color=:orangered)
axislegend(ax_h; position=:rt)

ax_diff = Axis(fig2[3, 1]; xlabel="Time (days)", ylabel="RK3 − AB2",
               title="Differences")
lines!(ax_diff, days, mean_SST_rk3 .- mean_SST_ab2;
       label="ΔSST (°C)", linewidth=2, color=:royalblue)

ax_diff_h = Axis(fig2[3, 1]; ylabel="Δh (m)", yaxisposition=:right)
lines!(ax_diff_h, days, mean_h_rk3 .- mean_h_ab2;
       label="Δh (m)", linewidth=2, color=:orangered, linestyle=:dash)
hidespines!(ax_diff_h)
hidexdecorations!(ax_diff_h)
axislegend(ax_diff;   position=:lt)
axislegend(ax_diff_h; position=:rt)

save("coupling_comparison_timeseries.png", fig2)
@info "Saved coupling_comparison_timeseries.png"

@info "Done!"
