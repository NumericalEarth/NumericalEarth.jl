# # JRA55-forced slab land — Greater Yellowstone / Snake River Plain
#
# A coarse-resolution land-only simulation forced by JRA55 reanalysis
# over northwest Wyoming and eastern Idaho. The domain spans roughly
#
#   * Salt Lake City (≈ 40.8°N, 112°W) at the SW corner,
#   * Pocatello / Idaho Falls (≈ 43°N, 112.5°W),
#   * Jackson Hole + the Tetons (≈ 43.5°N, 110.8°W),
#   * Yellowstone Plateau (≈ 44.5°N, 110.5°W),
#   * the Wind River Range out east.
#
# The `SlabLand` composes
#
#     energy    = SlabEnergy(dry_heat_capacity = ρcH_g, liquid_heat_capacity = cˡ)
#     hydrology = BucketHydrology(...)
#     surface   = ConstantSurfaceProperties(...)
#
# coupled to a `JRA55PrescribedAtmosphere` and a `JRA55PrescribedRadiation`
# (with explicit land albedo/emissivity) through `AtmosphereLandModel`.
# Turbulent fluxes use Monin–Obukhov similarity theory; surface humidity
# is β-reduced (`qₛ = qₐ + β·[q⁺(Tₛ) − qₐ]`) using the slab's bucket
# moisture availability.
#
# The terrain itself is not resolved here — `SlabLand` is spatially
# uniform in its closure properties — but the JRA55 atmospheric forcing
# does see the JRA55 native topography, so land surface temperature
# tracks elevation-driven gradients in T₂ₘ, downwelling shortwave, and
# precipitation across the region.

using NumericalEarth
using NumericalEarth.Lands
using NumericalEarth.Radiations: PrescribedRadiation, SurfaceRadiationProperties
using Oceananigans
using Oceananigans.Units
using Dates
using Printf
using Statistics

# ## Domain — 5° × 5° regional patch at 0.25° resolution
#
# 40.5°N–45.5°N, 114°W–109°W. 20 × 20 = 400 cells; trivial on CPU.

arch = CPU()
FT   = Float64

land_grid = LatitudeLongitudeGrid(arch, FT;
                                  size      = (20, 20),
                                  latitude  = (40.5, 45.5),
                                  longitude = (-114.0, -109.0),
                                  topology  = (Bounded, Bounded, Flat))

# ## Slab land closures
#
# A 15 cm thermal soil slab (`dry_heat_capacity = ρ·c·H ≈ 1500·1480·0.1
# J m⁻² K⁻¹`) with a Manabe single-bucket hydrology and uniform
# aerodynamic roughness for mid-vegetation / open shrubland.

ρcH_g = 1500.0 * 1480.0 * 0.10  # J m⁻² K⁻¹ (soil ρ × c × H = 222 kJ m⁻² K⁻¹)

energy    = SlabEnergy(eltype(land_grid);
                       dry_heat_capacity    = ρcH_g,
                       liquid_heat_capacity = 4186.0)
hydrology = BucketHydrology(eltype(land_grid);
                            field_capacity   = 150.0,
                            critical_wetness = 0.75)
surface   = ConstantSurfaceProperties(eltype(land_grid);
                                      momentum_roughness_length = 0.1,
                                      scalar_roughness_length   = 0.01)

slab_land = SlabLand(land_grid; energy, hydrology, surface)

# Initial conditions: 280 K skin temperature (early-spring climatology
# for the Snake River Plain) and a bucket at 50% of field capacity. The
# bucket can hold 150 kg m⁻², so W₀ = 75 kg m⁻². β is recomputed inside
# `update_state!` so we don't have to set it manually.

set!(slab_land.state.T, 280.0)
set!(slab_land.state.W, 0.5 * 150.0)
Oceananigans.TimeSteppers.update_state!(slab_land)

# ## JRA55 atmospheric + radiative forcing
#
# A 30-day window in spring captures the seasonal snowmelt transition
# over the higher terrain, which is when LST gradients across the
# region are most dramatic. JRA55-do native resolution is ≈ 0.5625°,
# so 0.25° land cells over-sample slightly — fine for visualization.

backend    = JRA55NetCDFBackend(10)
start_date = DateTime(1990, 4, 1)
end_date   = DateTime(1990, 5, 1)

atmosphere = JRA55PrescribedAtmosphere(arch, FT; backend, start_date, end_date)
radiation  = JRA55PrescribedRadiation(arch, FT; backend, start_date, end_date,
                                      land_surface = SurfaceRadiationProperties(0.18, 0.95))

# ## Coupled model
#
# `AtmosphereLandModel(atmos, land; radiation)` is a convenience wrapper
# around `EarthSystemModel` with `ocean = sea_ice = nothing`. The
# exchange grid is the land grid; JRA55 atmosphere and radiation are
# regridded onto it through fractional-index interpolators.

model = AtmosphereLandModel(atmosphere, slab_land; radiation)

Δt        = 10minutes
stop_time = 30days
simulation = Simulation(model; Δt, stop_time)

# ## Progress callback

wall_time = Ref(time_ns())

function progress(sim)
    land = sim.model.land
    T = land.state.T
    W = land.state.W
    β = land.state.moisture_availability
    Q = land.fluxes.net_energy_flux

    Tmin, Tmax = minimum(T), maximum(T)
    Wmin, Wmax = minimum(W), maximum(W)
    βmean      = mean(β)
    Qmean      = mean(Q)

    elapsed = 1e-9 * (time_ns() - wall_time[])
    wall_time[] = time_ns()

    msg = @sprintf("Iter: %d, t = %s, T: (%.1f, %.1f) K, W: (%.1f, %.1f) kg m⁻², ⟨β⟩: %.2f, ⟨Q⟩: %+6.1f W m⁻², wall Δ: %.1fs",
                   iteration(sim), prettytime(sim), Tmin, Tmax, Wmin, Wmax, βmean, Qmean, elapsed)
    @info msg
    return nothing
end

add_callback!(simulation, progress, IterationInterval(72))   # every 12 h

# ## Output writers

outputs = (T = slab_land.state.T,
           W = slab_land.state.W,
           β = slab_land.state.moisture_availability,
           Q = slab_land.fluxes.net_energy_flux,
           E = slab_land.fluxes.evaporation,
           P = slab_land.fluxes.precipitation,
           R = slab_land.fluxes.runoff)

simulation.output_writers[:land] = JLD2Writer(model, outputs;
                                              filename = "jra55_forced_slab_land",
                                              schedule = TimeInterval(3hours),
                                              overwrite_existing = true)

# ## Run

@info "Running coupled simulation..."
run!(simulation)
@info "Simulation complete."

# ## Animation
#
# A 2 × 3 panel showing the spatial fields plus a domain-mean time
# series. Coastlines aren't drawn, but the latitude/longitude axes give
# an orientation: Yellowstone is the upper-right blob (≈ 44.5°N,
# −110.5°W), Jackson sits ≈ 43.5°N / −110.8°W, Pocatello sits
# ≈ 42.9°N / −112.5°W, and Salt Lake City pokes the bottom edge near
# −111.9°W.

using CairoMakie

T_ts = FieldTimeSeries("jra55_forced_slab_land.jld2", "T"; architecture = CPU())
W_ts = FieldTimeSeries("jra55_forced_slab_land.jld2", "W"; architecture = CPU())
β_ts = FieldTimeSeries("jra55_forced_slab_land.jld2", "β"; architecture = CPU())
Q_ts = FieldTimeSeries("jra55_forced_slab_land.jld2", "Q"; architecture = CPU())
E_ts = FieldTimeSeries("jra55_forced_slab_land.jld2", "E"; architecture = CPU())
P_ts = FieldTimeSeries("jra55_forced_slab_land.jld2", "P"; architecture = CPU())

times = T_ts.times
Nt    = length(times)

λ = λnodes(land_grid, Center())
φ = φnodes(land_grid, Center())

times_days = collect(times) ./ 86400
T_dom_mean = [mean(interior(T_ts[k]))         for k in 1:Nt]
T_dom_max  = [maximum(interior(T_ts[k]))      for k in 1:Nt]
T_dom_min  = [minimum(interior(T_ts[k]))      for k in 1:Nt]

fig = Figure(size = (1500, 900), fontsize = 12)

ax_T = Axis(fig[1, 1]; title = "Skin temperature T (K)",     xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
ax_β = Axis(fig[1, 2]; title = "Moisture availability β",    xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
ax_Q = Axis(fig[1, 3]; title = "Net energy flux Q (W m⁻²)",  xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())

ax_t = Axis(fig[2, 1:3]; title = "Domain T extrema and mean over time", xlabel = "t (days)", ylabel = "T (K)")

n = Observable(1)

Tn = @lift view(interior(T_ts[$n]), :, :, 1)
βn = @lift view(interior(β_ts[$n]), :, :, 1)
Qn = @lift view(interior(Q_ts[$n]), :, :, 1)

heatmap!(ax_T, λ, φ, Tn; colormap = :thermal, colorrange = (260, 305))
heatmap!(ax_β, λ, φ, βn; colormap = :tempo,   colorrange = (0, 1))
heatmap!(ax_Q, λ, φ, Qn; colormap = :balance, colorrange = (-300, 300))

Colorbar(fig[1, 1, Right()],  limits = (260, 305),  colormap = :thermal,  label = "T (K)")
Colorbar(fig[1, 2, Right()],  limits = (0, 1),      colormap = :tempo,    label = "β")
Colorbar(fig[1, 3, Right()],  limits = (-300, 300), colormap = :balance,  label = "Q (W m⁻²)")

lines!(ax_t, times_days, T_dom_max;  color = :red,    linewidth = 1.5, label = "max")
lines!(ax_t, times_days, T_dom_mean; color = :black,  linewidth = 1.5, label = "mean")
lines!(ax_t, times_days, T_dom_min;  color = :blue,   linewidth = 1.5, label = "min")
axislegend(ax_t; position = :rb)
t_now = @lift [times_days[$n]]
vlines!(ax_t, t_now; color = :black, linewidth = 1.0, linestyle = :dash)

title = @lift "JRA55-forced slab land — Greater Yellowstone, t = " * prettytime(times[$n])
Label(fig[0, 1:3], title, fontsize = 16)

@info "Rendering animation..."
CairoMakie.record(fig, "jra55_forced_slab_land.mp4", 1:Nt; framerate = 12) do nn
    n[] = nn
end
@info "Animation saved."

nothing #hide

# ![](jra55_forced_slab_land.mp4)
