# # ERA5-forced slab land — Greater Yellowstone at ~1 km
#
# A high-resolution land-only simulation over the Greater Yellowstone
# region, forced by ERA5 reanalysis and elevation-corrected with ETOPO
# 2022 topography so subgrid mountains *are* visible in the skin
# temperature field.
#
# The domain covers roughly 43.25–45.25 °N × 111–109 °W (~220 × 156
# km), capturing Yellowstone Plateau, the Tetons, Jackson Hole, the
# northern Wind Rivers, and the Madison/Gallatin valleys at 200 × 200
# = 40 000 cells (≈ 1 km).
#
# `SlabLand` composes
#
#     energy    = SlabEnergy(dry_heat_capacity = ρcH_g)
#     hydrology = BucketHydrology(...)
#
# (roughness lengths are set on the atmosphere-land flux closure, not the land)
#
# coupled through `AtmosphereLandModel` to an [`ERA5PrescribedAtmosphere`](@ref)
# and [`ERA5PrescribedRadiation`](@ref): these download the required ERA5
# single-level fields (T₂ₘ, dewpoint, 10 m wind, surface pressure, total
# precipitation, downwelling SW/LW) over the domain, derive specific humidity
# from the dewpoint, and convert ERA5's accumulated radiation/precipitation to
# fluxes. They live on the ERA5 native grid; the coupled model interpolates them
# onto the 1 km land exchange grid.
#
# ## Elevation downscaling
#
# `SlabLand` itself has no terrain knowledge, and ERA5's T₂ₘ is at its
# own ~28 km grid-cell mean elevation (~2 km in this domain). To make
# the 1 km grid show elevation-driven temperature contrasts we apply a
# moist-environmental lapse-rate correction for the elevation difference
#
#     Δz(λ, φ) = z_ETOPO(λ, φ) − z_ERA5(λ, φ)
#
# where `z_ERA5` is ERA5's own surface elevation (its surface geopotential ÷ g).
# The correction `T ← T − Γ Δz` (Γ = 6.5 K km⁻¹) with a hydrostatic pressure
# adjustment is applied at run time by the coupled model's state exchanger
# ([`ElevationCorrection`](@ref)) — the prescribed atmosphere carries raw ERA5
# fields, and the same correction would apply to a live atmosphere. Specific
# humidity is conserved through the lift.
#
# Net effect: ~2 km elevation contrast between the Snake River Plain
# floor and the Wind River summits → ≈ 13 K skin-temperature spread
# from topography alone, on top of the synoptic + diurnal variability
# ERA5 already carries.
#
# ## CDS API credentials
#
# Downloading ERA5 fields requires CDS API credentials at `~/.cdsapirc`;
# see <https://cds.climate.copernicus.eu/how-to-api>.

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using CDSAPI                         # activates the CDS-API extension
using CairoMakie                     # rendered up-front: see note below the run
using Printf
using Statistics
import Dates: DateTime, Hour         # `Dates.hour` clashes with `Oceananigans.Units.hour`

# ## Domain — 2° × 2° Yellowstone box at ~1 km

arch  = CPU()

latitude = lat_min, lat_max = 43.25, 45.25
longitude = lon_min, lon_max = -111.0, -109.0

# The slab land is a 2D surface, so the grid is `Flat` in the vertical.
land_grid = LatitudeLongitudeGrid(arch; latitude, longitude,
                                  size = (200, 200),
                                  topology  = (Bounded, Bounded, Flat))

# ## ETOPO surface elevation
#
# `regrid_topography` regrids ETOPO 2022 onto the land grid as a positive land
# surface elevation (the topographic counterpart of `regrid_bathymetry`). This
# is the *desired* elevation the atmosphere is corrected to.

z_land = regrid_topography(land_grid; dataset = ETOPO2022())

# ## ERA5 forcing — 3-day window
#
# Three days of hourly data spans two diurnal cycles plus a synoptic pulse and
# keeps the download + simulation under ~10 min on CPU.
# [`ERA5PrescribedAtmosphere`](@ref) and [`ERA5PrescribedRadiation`](@ref)
# download the required single-level fields over `region`, derive specific
# humidity from the 2 m dewpoint, and convert ERA5's hourly-accumulated
# radiation (J m⁻²) and precipitation (m) to fluxes (W m⁻², kg m⁻² s⁻¹) — they
# return standard prescribed components on the ERA5 native grid, which the
# coupled model interpolates onto the 1 km land grid. The region is just the
# land domain; the dataset fetch center-brackets it by one native cell, so the
# downscaling is well-posed at the domain edges.

dataset    = ERA5HourlySingleLevel()
dates      = DateTime(2020, 4, 1):Hour(1):DateTime(2020, 4, 3, 23)
region     = BoundingBox(; latitude, longitude)
start_date = first(dates)
end_date   = last(dates)
Nt         = length(dates)

atmosphere = ERA5PrescribedAtmosphere(arch; dataset, start_date, end_date, region,
                                      surface_layer_height  = 10,
                                      boundary_layer_height = 800)

radiation = ERA5PrescribedRadiation(arch; dataset, start_date, end_date, region,
                                    land_surface = SurfaceRadiationProperties(0.18, 0.95))

# ## Elevation correction
#
# ERA5's near-surface fields correspond to ERA5's own ~28 km grid-cell mean
# elevation (~2 km here). [`ElevationCorrection`](@ref) lifts the regridded
# atmosphere from that elevation (`z_era5`) to the 1 km ETOPO surface (`z_land`)
# with a moist lapse-rate shift + hydrostatic pressure adjustment, applied by the
# state exchanger every step (`q` conserved). `z_era5` is ERA5's model topography
# (its surface geopotential ÷ g → metres); the gravitational acceleration and
# gas constant the pressure adjustment needs are pulled from the atmosphere's
# thermodynamics, not hand-passed.
z_meta = Metadatum(:topography; dataset, date = start_date, region)
z_era5 = Field(z_meta, land_grid)

Δz = z_land - z_era5
@info "Elevation field stats" land=extrema(z_land) era5=extrema(z_era5) Δz=extrema(Δz)

Γ_lapse = 6.5e-3 # K m⁻¹  environmental lapse rate
correction = ElevationCorrection(z_land, z_era5; lapse_rate = Γ_lapse)

# ## Slab land
#
# `SlabLand` is purely energy + hydrology; aerodynamic roughness is a property of
# the atmosphere-land flux closure (set on the model below), not of the land.
dry_heat_capacity = 0.1 * 1500 * 1480
slab_land = SlabLand(land_grid;
                     energy = SlabEnergy(; dry_heat_capacity),
                     hydrology = BucketHydrology(maximum_water_storage = 150))

# Cold-start the skin temperature from the elevation-corrected ERA5 T₂ₘ at the
# first snapshot (interpolated onto the 1 km grid), mirroring the runtime lift.
T₀ = Field{Center, Center, Nothing}(land_grid)
Oceananigans.Fields.interpolate!(T₀, atmosphere.tracers.T[1])
set!(slab_land; T = T₀ - Γ_lapse * Δz, M = 0.5 * 150)

# ## Coupled model
#
# Roughness lengths live with the atmosphere-land flux closure: pass a
# `SimilarityTheoryFluxes` with the desired land roughness (0.1 m momentum,
# 0.01 m scalar here) via `atmosphere_land_fluxes`.

atmosphere_land_fluxes = SimilarityTheoryFluxes(momentum_roughness_length    = 0.1,
                                                temperature_roughness_length = 0.01,
                                                water_vapor_roughness_length = 0.01)

model = AtmosphereLandModel(atmosphere, slab_land;
                            radiation, atmosphere_land_fluxes,
                            exchanger_correction = correction)

simulation = Simulation(model; Δt = 5minutes, stop_time = (Nt - 1) * 3600)

wall_time = Ref(time_ns())

function progress(sim)
    land = sim.model.land
    Tmin, Tmax = minimum(land.temperature), maximum(land.temperature)
    Wmin, Wmax = minimum(land.water_storage), maximum(land.water_storage)
    𝒮mean      = mean(land.saturation)
    Qmean      = mean(land.fluxes.net_energy_flux)
    elapsed    = 1e-9 * (time_ns() - wall_time[]); wall_time[] = time_ns()
    @info @sprintf("Iter %d  t = %s  T %.1f–%.1f K  W %.1f–%.1f kg m⁻²  ⟨𝒮⟩ %.2f  ⟨Q⟩ %+6.1f W m⁻²  wall Δ %.1fs",
                   iteration(sim), prettytime(sim), Tmin, Tmax, Wmin, Wmax, 𝒮mean, Qmean, elapsed)
    return nothing
end
add_callback!(simulation, progress, IterationInterval(144))  # ~12 h

outputs = (T = slab_land.temperature,
           W = slab_land.water_storage,
           𝒮 = slab_land.saturation,
           Q = slab_land.fluxes.net_energy_flux,
           E = slab_land.fluxes.evaporation,
           P = slab_land.fluxes.precipitation)

simulation.output_writers[:land] = JLD2Writer(model, outputs;
                                              filename = "era5_forced_slab_land",
                                              schedule = TimeInterval(1hour),
                                              overwrite_existing = true)

# ## Run

@info "Running ERA5-forced slab land simulation at ~1 km..."
run!(simulation)
@info "Simulation complete."

# Release the JLD2 writer's file handle and drop the working FTS
# before the animation re-opens the file. At 200 × 200 the combined
# working set is large enough that leaving these references live can
# segfault the same Julia session during heatmap setup.
close(simulation.output_writers[:land])
delete!(simulation.output_writers, :land)
atmosphere = radiation = simulation = model = nothing
GC.gc(true); GC.gc(true)

# ## Animation
#
# Three spatial panels — T, 𝒮, Q — plus a static elevation panel and
# a domain-mean T(t) time series. The elevation panel makes the
# lapse-rate signature in T(λ, φ) directly readable.

T_ts = FieldTimeSeries("era5_forced_slab_land.jld2", "T")
𝒮_ts = FieldTimeSeries("era5_forced_slab_land.jld2", "𝒮")
Q_ts = FieldTimeSeries("era5_forced_slab_land.jld2", "Q")

times      = T_ts.times
Nframes    = length(times)
times_days = collect(times) ./ 86400

# Colorrange for 𝒮 covers the actual span over the run (clamped to a sensible
# minimum width so an unusually static field still renders cleanly).
𝒮_lo, 𝒮_hi = extrema(𝒮_ts)
𝒮_range = (𝒮_lo, max(𝒮_hi, 𝒮_lo + 0.1))

fig = Figure(size = (1500, 1000), fontsize = 12)
ax_T = Axis(fig[1, 1]; title = "Skin temperature T (K)",    xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
ax_𝒮 = Axis(fig[1, 2]; title = "Surface saturation 𝒮",   xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
ax_Q = Axis(fig[1, 3]; title = "Net energy flux Q (W m⁻²)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
ax_z = Axis(fig[2, 1]; title = "Elevation (m, ETOPO 2022)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
ax_t = Axis(fig[2, 2:3]; title = "Domain T extrema and mean over time", xlabel = "t (days)", ylabel = "T (K)")

n  = Observable(1)
Tn = @lift T_ts[$n]
𝒮n = @lift 𝒮_ts[$n]
Qn = @lift Q_ts[$n]

hm_T = heatmap!(ax_T, Tn; colormap = :thermal, colorrange = (250, 300))
hm_𝒮 = heatmap!(ax_𝒮, 𝒮n; colormap = :tempo,   colorrange = 𝒮_range)
hm_Q = heatmap!(ax_Q, Qn; colormap = :balance, colorrange = (-400, 400))
hm_z = heatmap!(ax_z, z_land; colormap = :terrain, colorrange = (1000, 3500))

Colorbar(fig[1, 1, Right()], hm_T; label = "T (K)")
Colorbar(fig[1, 2, Right()], hm_𝒮; label = "𝒮")
Colorbar(fig[1, 3, Right()], hm_Q; label = "Q (W m⁻²)")
Colorbar(fig[2, 1, Right()], hm_z; label = "elevation (m)")

T_mean = [mean(T_ts[k])    for k in 1:Nframes]
T_max  = [maximum(T_ts[k]) for k in 1:Nframes]
T_min  = [minimum(T_ts[k]) for k in 1:Nframes]
lines!(ax_t, times_days, T_max;  color = :red,   linewidth = 1.5, label = "max")
lines!(ax_t, times_days, T_mean; color = :black, linewidth = 1.5, label = "mean")
lines!(ax_t, times_days, T_min;  color = :blue,  linewidth = 1.5, label = "min")
axislegend(ax_t; position = :rb)
vlines!(ax_t, @lift([times_days[$n]]); color = :black, linewidth = 1.0, linestyle = :dash)

Label(fig[0, 1:3], @lift("ERA5-forced slab land — Greater Yellowstone at ~1 km, t = " * prettytime(times[$n])), fontsize = 16)

@info "Rendering animation..."
CairoMakie.record(fig, "era5_forced_slab_land.mp4", 1:Nframes; framerate = 12) do nn
    n[] = nn
end
@info "Animation saved."

nothing #hide

# ![](era5_forced_slab_land.mp4)
