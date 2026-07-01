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
#     energy    = WaterCoupledEnergy(...)   # C(Mˡᵃ) = C_dry + cˡ Mˡᵃ
#     hydrology = VariablySaturatedHydrology(...) # ϑˡ storage, signed-flux budget
#
# with the atmosphere-facing humidity solved by
# [`DryLayerHumidity`](@ref) — a Fickian vapor-flux balance through
# an unresolved dry layer at saturation-dependent depth `δᵛ(𝒮)` —
# instead of the prescribed `β(𝒮) qᵛ⁺(Tˡᵃ)` Manabe efficiency. See the
# [SlabLand tutorial](../land/evaporation_front_slab_land.md) for the model
# derivation. (Roughness lengths are set on the atmosphere–land flux
# closure, not the land.)
#
# The land is coupled through `AtmosphereLandModel` to an
# [`ERA5PrescribedAtmosphere`](@ref) and [`ERA5PrescribedRadiation`](@ref):
# these download the required ERA5 single-level fields (T₂ₘ, dewpoint, 10 m
# wind, surface pressure, total precipitation, downwelling SW/LW) over the
# domain, derive specific humidity from the dewpoint, and convert ERA5's
# accumulated radiation/precipitation to fluxes. They live on the ERA5 native
# grid; the coupled model interpolates them onto the 1 km land exchange grid.
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

# latitude = lat_min, lat_max = 43.25, 45.25
# longitude = lon_min, lon_max = -111.0, -109.0

# latitude = lat_min, lat_max = 3.5, 5.5
# longitude = lon_min, lon_max = 101.2, 103.2

# Central Borneo highlands: equatorial (snow-free), heavy rainfall, ~2,300 m
# relief in the Müller/Schwaner ranges, and fully inland (no ocean).
latitude = lat_min, lat_max = 0.5, 2.5
longitude = lon_min, lon_max = 113.0, 115.0

# Meghalaya / Shillong Plateau: the wettest place on Earth (Cherrapunji),
# fully inland, with peaks below ~1,960 m so snow is impossible.
# latitude = lat_min, lat_max = 24.8, 26.8
# longitude = lon_min, lon_max = 90.5, 92.5

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
dates      = DateTime(2020, 4, 1):Hour(1):DateTime(2020, 4, 5, 23)
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
# the atmosphere-land flux closure (built with the interface below), not of the
# land. We use the systematic-land stack from the
# [differentiable dry-layer example](differentiable_dry_layer_slab_land.md):
# a `WaterCoupledEnergy` slab whose areal heat capacity `C(Mˡᵃ) = C_dry + cˡ Mˡᵃ`
# grows with stored water, over a `VariablySaturatedHydrology` augmented-storage
# budget. We keep `InfiltrationCapacityRunoff`, which sheds rain arriving faster
# than the soil infiltration capacity rather than letting it all enter storage.
# The atmosphere-facing humidity is closed by `DryLayerHumidity` on the interface
# (see the coupled model below).
#
# Unlike the thin idealized slab of that example, this is a real-run slab: a 0.5 m
# soil column with deep-temperature restoring on a 12 h time scale and the
# advective energy terms off.

slab_depth               = 1
porosity                 = 0.4
residual_liquid_fraction = 0.05

slab_land = SlabLand(land_grid;
                     energy = WaterCoupledEnergy(eltype(land_grid);
                                                 dry_heat_capacity = 0.1 * 1500 * 1480,
                                                 liquid_heat_capacity = 4186,
                                                 reference_temperature = 273.15,
                                                 deep_temperature = 280.0,
                                                 deep_time_scale = 12hours,
                                                 advect_deep_liquid_energy = false,
                                                 advect_surface_liquid_energy = false),
                     hydrology = VariablySaturatedHydrology(eltype(land_grid);
                                                            slab_depth,
                                                            porosity,
                                                            residual_liquid_fraction,
                                                            storage_height = 1000,
                                                            retention_curve = VanGenuchtenRetention(α = 1.0, n = 2.0),
                                                            hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-7, n = 2.0),
                                                            deep_liquid_flux = NoDeepLiquidFlux(),
                                                            runoff = InfiltrationCapacityRunoff(infiltration_capacity = 1e-3)))

## --- Classic Manabe bucket slab (disabled) ---
## slab_land = SlabLand(land_grid;
##                      energy = SlabEnergy(; dry_heat_capacity = 0.1 * 1500 * 1480),
##                      hydrology = BucketHydrology(maximum_water_storage = 150))

# Cold-start the skin temperature from the elevation-corrected ERA5 T₂ₘ at the
# first snapshot (interpolated onto the 1 km grid), mirroring the runtime lift.
# Initial soil water `M = 100 kg m⁻²` ≈ half the slab's saturation storage
# `Mˡᵃ⁺ = ρˡ ν hˡᵃ = 1000 · 0.4 · 0.5 = 200 kg m⁻²` (𝒮 ≈ 0.43).
T₀ = Field{Center, Center, Nothing}(land_grid)
Oceananigans.Fields.interpolate!(T₀, atmosphere.temperature[1])
set!(slab_land; T = T₀ - Γ_lapse * Δz, M = 100)

# ## Coupled model
#
# The atmosphere-facing humidity uses `DryLayerHumidity` — a Fickian vapor-flux
# balance through an unresolved dry layer whose depth `δᵛ(𝒮)` vanishes at a
# saturated surface and grows as the slab dries past the onset saturation `𝒮ᶜ`,
# throttling evaporation (the two-stage bare-soil drying of Or et al. 2013).
#
# Roughness lengths still live with the atmosphere-land flux closure, but because
# we pass a *custom* `atmosphere_land_interface` they must be baked into *its*
# `fluxes`: a model-level `atmosphere_land_fluxes` is ignored once
# `atmosphere_land_interface` is supplied.

atmosphere_land_fluxes = SimilarityTheoryFluxes(momentum_roughness_length    = 0.1,
                                                temperature_roughness_length = 0.01,
                                                water_vapor_roughness_length = 0.01)

interface_specific_humidity = DryLayerHumidity(;
    dry_layer_depth = StorageBasedDryLayerDepth(maximum_dry_layer_depth    = 0.05,
                                                dry_layer_onset_saturation = 0.5,
                                                dry_layer_exponent         = 2),
    vapor_exchange = DryLayerVaporPistonVelocity(minimum_dry_layer_depth = 1e-4,
                                                 molecular_diffusivity   = 2.5e-5,
                                                 tortuosity_model        = MillingtonQuirk()),
    thermal_exchange_depth = 0.10,
    porosity)

al_interface = atmosphere_land_interface(slab_land.grid, atmosphere, slab_land;
                                         fluxes = atmosphere_land_fluxes,
                                         specific_humidity = interface_specific_humidity)

model = AtmosphereLandModel(atmosphere, slab_land;
                            radiation,
                            atmosphere_land_interface = al_interface,
                            exchanger_correction = correction)

simulation = Simulation(model; Δt = 5minutes, stop_time = (Nt - 1) * 3600)

wall_time = Ref(time_ns())

function progress(sim)
    land = sim.model.land
    Tmin, Tmax = minimum(land.temperature), maximum(land.temperature)
    Wmin, Wmax = minimum(land.water_storage), maximum(land.water_storage)
    𝒮mean      = mean(land.saturation)
    Qmean      = -mean(land.fluxes.surface_energy_flux)  ## net energy into the slab (positive-upward flux negated)
    elapsed    = 1e-9 * (time_ns() - wall_time[]); wall_time[] = time_ns()
    @info @sprintf("Iter %d  t = %s  T %.1f–%.1f K  W %.1f–%.1f kg m⁻²  ⟨𝒮⟩ %.2f  ⟨Q⟩ %+6.1f W m⁻²  wall Δ %.1fs",
                   iteration(sim), prettytime(sim), Tmin, Tmax, Wmin, Wmax, 𝒮mean, Qmean, elapsed)
    return nothing
end
add_callback!(simulation, progress, IterationInterval(144))  # ~12 h

# The variably-saturated hydrology reports a signed vapor flux `Jᵛ` (positive
# upward — evaporation) and a liquid precipitation flux `Pˡ` (positive downward),
# in place of the bucket's positive-part `evaporation`/`precipitation` accumulators.
outputs = (T = slab_land.temperature,
           W = slab_land.water_storage,
           𝒮 = slab_land.saturation,
           Q = slab_land.fluxes.surface_energy_flux,
           E = slab_land.fluxes.vapor_flux,
           P = slab_land.fluxes.liquid_precipitation_flux)

simulation.output_writers[:land] = JLD2Writer(model, outputs;
                                              filename = "era5_forced_slab_land_lat_$(lat_min)_$(lat_max)_lon_$(lon_min)_$(lon_max)_slabdepth_$(slab_depth)",
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

T_ts = FieldTimeSeries("era5_forced_slab_land_lat_$(lat_min)_$(lat_max)_lon_$(lon_min)_$(lon_max)_slabdepth_$(slab_depth).jld2", "T")
𝒮_ts = FieldTimeSeries("era5_forced_slab_land_lat_$(lat_min)_$(lat_max)_lon_$(lon_min)_$(lon_max)_slabdepth_$(slab_depth).jld2", "𝒮")
Q_ts = FieldTimeSeries("era5_forced_slab_land_lat_$(lat_min)_$(lat_max)_lon_$(lon_min)_$(lon_max)_slabdepth_$(slab_depth).jld2", "Q")
P_ts = FieldTimeSeries("era5_forced_slab_land_lat_$(lat_min)_$(lat_max)_lon_$(lon_min)_$(lon_max)_slabdepth_$(slab_depth).jld2", "P")

times      = T_ts.times
Nframes    = length(times)
times_days = collect(times) ./ 86400

λ, φ, _ = nodes(land_grid, Center(), Center(), Center())

# Colorrange for 𝒮 covers the actual span over the run (clamped to a sensible
# minimum width so an unusually static field still renders cleanly).
𝒮_lo, 𝒮_hi = extrema(𝒮_ts)
𝒮_range = (𝒮_lo, max(𝒮_hi, 𝒮_lo + 0.1))

fig = Figure(size = (1700, 1000), fontsize = 12)
ax_T = Axis(fig[1, 1]; title = "Skin temperature T (K)",    xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
ax_𝒮 = Axis(fig[1, 3]; title = "Surface saturation 𝒮",   xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
ax_Q = Axis(fig[1, 5]; title = "Net energy flux Q (W m⁻²)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
ax_P = Axis(fig[1, 7]; title = "Precipitation P (mm hr⁻¹)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
ax_z = Axis(fig[2, 1]; title = "Elevation (m, ETOPO 2022)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
ax_t = Axis(fig[2, 3:8]; title = "Domain T extrema and mean over time", xlabel = "t (days)", ylabel = "T (K)")

## Ocean cells (z_land == 0) carry no land state, so blank them out with NaN.
ocean = interior(z_land, :, :, 1) .== 0
mask_ocean(field) = ifelse.(ocean, NaN, field)

n  = Observable(1)
Tn = @lift mask_ocean(interior(T_ts[$n], :, :, 1))
𝒮n = @lift mask_ocean(interior(𝒮_ts[$n], :, :, 1))
Qn = @lift mask_ocean(interior(Q_ts[$n], :, :, 1))
## P is stored as a positive-downward mass flux (kg m⁻² s⁻¹); show it as mm hr⁻¹.
Pn = @lift mask_ocean(interior(P_ts[$n], :, :, 1) .* 3600)

Tlim = extrema(T_ts)
𝒮lim = (𝒮_lo, max(𝒮_hi, 𝒮_lo + 0.1))
Qlim = (-maximum(abs.(Q_ts)), maximum(abs.(Q_ts)))
Plim = (0, max(maximum(P_ts) * 3600, 1e-3))
zlim = extrema(interior(z_land, :, :, 1))

hm_T = heatmap!(ax_T, λ, φ, Tn; colormap = :turbo, colorrange = Tlim)
hm_𝒮 = heatmap!(ax_𝒮, λ, φ, 𝒮n; colormap = :tempo,   colorrange = 𝒮lim)
hm_Q = heatmap!(ax_Q, λ, φ, Qn; colormap = :balance, colorrange = Qlim)
hm_P = heatmap!(ax_P, λ, φ, Pn; colormap = :dense,   colorrange = Plim)
hm_z = heatmap!(ax_z, λ, φ, interior(z_land, :, :, 1); colormap = :terrain, colorrange = zlim)

Colorbar(fig[1, 2], hm_T; label = "T (K)")
Colorbar(fig[1, 4], hm_𝒮; label = "𝒮")
Colorbar(fig[1, 6], hm_Q; label = "Q (W m⁻²)")
Colorbar(fig[1, 8], hm_P; label = "P (mm hr⁻¹)")
Colorbar(fig[2, 2], hm_z; label = "elevation (m)")

T_mean = [mean(T_ts[k])    for k in 1:Nframes]
T_max  = [maximum(T_ts[k]) for k in 1:Nframes]
T_min  = [minimum(T_ts[k]) for k in 1:Nframes]
lines!(ax_t, times_days, T_max;  color = :red,   linewidth = 1.5, label = "max")
lines!(ax_t, times_days, T_mean; color = :black, linewidth = 1.5, label = "mean")
lines!(ax_t, times_days, T_min;  color = :blue,  linewidth = 1.5, label = "min")
axislegend(ax_t; position = :rb)
vlines!(ax_t, @lift([times_days[$n]]); color = :black, linewidth = 1.0, linestyle = :dash)

# Label(fig[0, 1:8], @lift("ERA5-forced slab land — Meghalaya at ~1 km, t = " * prettytime(times[$n])), fontsize = 16)
Label(fig[0, 1:8], @lift("ERA5-forced slab land — Central Borneo at ~1 km, t = " * prettytime(times[$n])), fontsize = 16)

trim!(fig.layout)

@info "Rendering animation..."
CairoMakie.record(fig, "era5_forced_slab_land_lat_$(lat_min)_$(lat_max)_lon_$(lon_min)_$(lon_max)_slabdepth_$(slab_depth).mp4", 1:Nframes; framerate = 12) do nn
    n[] = nn
end
@info "Animation saved."

nothing #hide

# ![](era5_forced_slab_land.mp4)
