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
#     energy    = SlabEnergy(dry_heat_capacity = ρcH_g, liquid_heat_capacity = cˡ)
#     hydrology = BucketHydrology(...)
#     surface   = ConstantSurfaceProperties(...)
#
# coupled to a `PrescribedAtmosphere` and `PrescribedRadiation` through
# `AtmosphereLandModel`. ERA5 single-level fields (T₂ₘ, dewpoint, 10 m
# wind, surface pressure, total precipitation, downwelling SW/LW) are
# bundled into a single [`MetadataSet`](@ref) and loaded onto the 1 km
# land grid in one call — the dataset backend bilinearly downscales
# from ERA5 native to 1 km at load time.
#
# ## Elevation downscaling
#
# `SlabLand` itself has no terrain knowledge, and ERA5's T₂ₘ is at its
# own ~28 km grid-cell mean elevation (~2 km in this domain). To make
# the 1 km grid show elevation-driven temperature contrasts we apply a
# moist-environmental lapse-rate correction for the elevation difference
#
#     Δz(λ, φ) = z_ETOPO(λ, φ) − z_ERA5_eff(λ, φ)
#
# where `z_ERA5_eff` is ETOPO box-averaged onto the ERA5 native grid (≈ what
# ERA5 thinks the surface elevation is in each cell), projected back to 1 km.
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
using NumericalEarth.DataWrangling.ERA5: ERA5HourlySingleLevel        # not reexported at the top level
using NumericalEarth.Atmospheres: PrescribedPrecipitationFlux         # not reexported at the top level
using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: update_state!
using CDSAPI                         # activates the CDS-API extension
using CairoMakie                     # rendered up-front: see note below the run
using Printf
using Statistics
import Dates: DateTime, Hour         # `Dates.hour` clashes with `Oceananigans.Units.hour`

# ## Domain — 2° × 2° Yellowstone box at ~1 km

const Γ_lapse = 6.5e-3        # K m⁻¹  environmental lapse rate
const g_acc   = 9.81          # m s⁻²
const Rd      = 287.052       # J kg⁻¹ K⁻¹
const ε       = 0.62198       # R_d / R_v

arch  = CPU()
FT    = Float64

lat_min, lat_max = 43.25, 45.25
lon_min, lon_max = -111.0, -109.0

land_grid = LatitudeLongitudeGrid(arch, FT;
                                  size      = (200, 200, 1),
                                  latitude  = (lat_min, lat_max),
                                  longitude = (lon_min, lon_max),
                                  z         = (-1.0, 0.0),
                                  topology  = (Bounded, Bounded, Bounded))

# ## ETOPO elevation on the land grid + ERA5-effective elevation
#
# `regrid_bathymetry` returns `bottom_height` — positive over land. We
# regrid ETOPO 2022 onto two grids: the 1 km land grid and a coarse
# 0.25° "ERA5-like" grid covering the same region. Box-averaging ETOPO
# onto the ERA5 grid approximates ERA5's effective grid-cell elevation;
# projecting that field back to 1 km gives the elevation each ERA5
# cell "thinks" applies everywhere within it.

z_land_field = regrid_bathymetry(land_grid; dataset = ETOPO2022(),
                                 interpolation_passes = 1, minimum_depth = 0)

pad = 0.25                    # one ERA5 cell of padding on every side
era5_lat = (lat_min - pad, lat_max + pad)
era5_lon = (lon_min - pad, lon_max + pad)
era5_Ny  = round(Int, (era5_lat[2] - era5_lat[1]) / 0.25)
era5_Nx  = round(Int, (era5_lon[2] - era5_lon[1]) / 0.25)

era5_grid = LatitudeLongitudeGrid(arch, FT;
                                  size      = (era5_Nx, era5_Ny, 1),
                                  latitude  = era5_lat,
                                  longitude = era5_lon,
                                  z         = (-1.0, 0.0),
                                  topology  = (Bounded, Bounded, Bounded))

z_era5_field = regrid_bathymetry(era5_grid; dataset = ETOPO2022(),
                                 interpolation_passes = 1, minimum_depth = 0)

# Project the ERA5-effective elevation onto the 1 km grid by a
# piecewise-constant lookup: each 1 km cell inherits the value of the
# ERA5 cell that contains its centre. (Bilinear `set!` between grids
# of different size is not yet supported in Oceananigans main.)
λ_land = λnodes(land_grid, Center())
φ_land = φnodes(land_grid, Center())
z_land     = max.(interior(z_land_field, :, :, 1), 0.0)
z_era5_raw = interior(z_era5_field, :, :, 1)
z_era5_eff = similar(z_land)
@inbounds for j in eachindex(φ_land), i in eachindex(λ_land)
    i_e = clamp(floor(Int, (λ_land[i] - era5_lon[1]) / 0.25) + 1, 1, era5_Nx)
    j_e = clamp(floor(Int, (φ_land[j] - era5_lat[1]) / 0.25) + 1, 1, era5_Ny)
    z_era5_eff[i, j] = max(z_era5_raw[i_e, j_e], 0.0)
end
Δz = z_land .- z_era5_eff       # m, positive over peaks

@info "Elevation field stats" land=(minimum(z_land), maximum(z_land)) era5_eff=(minimum(z_era5_eff), maximum(z_era5_eff)) Δz=(minimum(Δz), maximum(Δz))

# ## ERA5 forcing — 3-day window
#
# Three days of hourly data is enough for two diurnal cycles plus a
# synoptic pulse and keeps the data download + simulation under
# ~10 min on CPU. Bundle the eight required variables into one
# `MetadataSet` so the same `dataset`, `dates`, and `region` aren't
# repeated; `FieldTimeSeries(mset, land_grid)` returns a `NamedTuple`
# of pre-downscaled FTS keyed by variable name.

dates  = DateTime(2020, 4, 1):Hour(1):DateTime(2020, 4, 3, 23)
region = BoundingBox(latitude = era5_lat, longitude = era5_lon)
Nt     = length(dates)

forcing_set = MetadataSet(:eastward_velocity,
                          :northward_velocity,
                          :temperature,
                          :dewpoint_temperature,
                          :surface_pressure,
                          :total_precipitation,
                          :downwelling_shortwave_radiation,
                          :downwelling_longwave_radiation;
                          dataset = ERA5HourlySingleLevel(),
                          dates, region)

era5 = FieldTimeSeries(forcing_set, land_grid; time_indices_in_memory = Nt)
atmos_times = era5.eastward_velocity.times

# ## Atmosphere forcing fields
#
# The elevation lapse-rate correction is applied lazily by the coupled model's
# state exchanger (`ElevationCorrection`, below), so the prescribed atmosphere
# carries the *raw* ERA5 temperature and pressure — no elevation-corrected copies.
# We still convert the ERA5-specific inputs: dewpoint → specific humidity (at the
# raw surface pressure; the correction conserves `q`), *accumulated* SW/LW
# (J m⁻² per hour) → power (W m⁻²), and total precipitation (m per hour) → kg m⁻² s⁻¹.

@inline saturation_vapor_pressure(T) = 611.2 * exp(17.62 * (T - 273.15) / (T - 30.04))
@inline q_from_dewpoint(Td, p) = (e = saturation_vapor_pressure(Td); ε * e / (p - (1 - ε) * e))

q_local   = FieldTimeSeries{Center, Center, Nothing}(land_grid, atmos_times)
rain      = FieldTimeSeries{Center, Center, Nothing}(land_grid, atmos_times)
ssrd_rate = FieldTimeSeries{Center, Center, Nothing}(land_grid, atmos_times)
strd_rate = FieldTimeSeries{Center, Center, Nothing}(land_grid, atmos_times)

for n in 1:Nt
    p_era5 = interior(era5.surface_pressure[n], :, :, 1)
    Td     = interior(era5.dewpoint_temperature[n], :, :, 1)
    interior(q_local[n], :, :, 1) .= q_from_dewpoint.(Td, p_era5)

    interior(rain[n],      :, :, 1) .= max.(interior(era5.total_precipitation[n],             :, :, 1) .* (1000.0 / 3600.0), 0)
    interior(ssrd_rate[n], :, :, 1) .= max.(interior(era5.downwelling_shortwave_radiation[n], :, :, 1) ./ 3600.0,            0)
    interior(strd_rate[n], :, :, 1) .= max.(interior(era5.downwelling_longwave_radiation[n],  :, :, 1) ./ 3600.0,            0)
end

# The state exchanger differences these two elevations into `Δz` on the exchange
# grid under the hood and applies `T -= Γ Δz` plus the hydrostatic pressure
# adjustment to the regridded atmosphere every step (any atmosphere, prescribed
# or online).
elevation_correction = ElevationCorrection(z_era5_eff, z_land;
                                           lapse_rate = Γ_lapse,
                                           gravitational_acceleration = g_acc,
                                           dry_air_gas_constant = Rd)

# ## Prescribed atmosphere, radiation, and slab land

atmosphere = PrescribedAtmosphere(land_grid, atmos_times;
                                  velocities = (u = era5.eastward_velocity, v = era5.northward_velocity),
                                  tracers    = (T = era5.temperature, q = q_local),
                                  pressure   = era5.surface_pressure,
                                  freshwater_flux = PrescribedPrecipitationFlux(rain = rain),
                                  surface_layer_height  = 10.0,
                                  boundary_layer_height = 800.0)

radiation = PrescribedRadiation(ssrd_rate, strd_rate;
                                ocean_surface   = nothing,
                                sea_ice_surface = nothing,
                                land_surface    = SurfaceRadiationProperties(0.18, 0.95))

slab_land = SlabLand(land_grid;
                     energy    = SlabEnergy(FT;
                                            dry_heat_capacity    = 1500.0 * 1480.0 * 0.10,
                                            liquid_heat_capacity = 4186.0),
                     hydrology = BucketHydrology(FT; maximum_water_storage = 150.0),
                     surface   = ConstantSurfaceProperties(FT;
                                                           momentum_roughness_length = 0.1,
                                                           scalar_roughness_length   = 0.01))

# Initialize T₀ from the elevation-corrected ERA5 T₂ₘ at the first snapshot
# (mirroring the runtime correction for a consistent cold start); fill the
# parent first so halo cells start at the domain-mean rather than uninitialised
# memory.
T_init = interior(era5.temperature[1], :, :, 1) .- Γ_lapse .* Δz
fill!(parent(slab_land.temperature), mean(T_init))
interior(slab_land.temperature, :, :, 1) .= T_init
fill!(parent(slab_land.water_storage), 0.5 * 150.0)
update_state!(slab_land)

# ## Coupled model

model      = AtmosphereLandModel(atmosphere, slab_land; radiation,
                                 atmosphere_state_correction = elevation_correction)
simulation = Simulation(model; Δt = 5minutes, stop_time = (Nt - 1) * 3600.0)

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
atmosphere = simulation = model = nothing
era5 = T_local = p_local = q_local = rain = ssrd_rate = strd_rate = nothing
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

hm_T = heatmap!(ax_T, Tn;           colormap = :thermal, colorrange = (250, 300))
hm_𝒮 = heatmap!(ax_𝒮, 𝒮n;           colormap = :tempo,   colorrange = 𝒮_range)
hm_Q = heatmap!(ax_Q, Qn;           colormap = :balance, colorrange = (-400, 400))
hm_z = heatmap!(ax_z, z_land_field; colormap = :terrain, colorrange = (1000, 3500))

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
