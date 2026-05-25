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
# downloaded over the bounding box and interpolated bilinearly onto the
# 1 km land grid at load time. The atmosphere grid then *is* the land
# grid — no further regridding at runtime.
#
# ## Elevation downscaling
#
# `SlabLand` itself has no terrain knowledge, and ERA5's T₂ₘ is at its
# own ~28 km grid-cell mean elevation (~2 km in this domain). To make
# the 1 km grid show elevation-driven temperature contrasts we apply a
# dry-environmental lapse-rate correction:
#
#     T_local(λ, φ) = T_ERA5(λ, φ) − Γ · (z_ETOPO(λ, φ) − z_ERA5_eff(λ, φ))
#
# with Γ = 6.5 K km⁻¹. `z_ERA5_eff` is ETOPO box-averaged onto the ERA5
# native grid (≈ what ERA5 thinks the surface elevation is in each
# cell), then bilinearly upsampled back to 1 km. Surface pressure is
# adjusted hydrostatically by the same `Δz`, and specific humidity is
# recomputed from dewpoint at the corrected (T, p).
#
# Net effect: ~2 km elevation contrast between the Snake River Plain
# floor and the Wind River summits → ≈ 13 K skin-temperature spread
# from topography alone, on top of the synoptic + diurnal variability
# ERA5 already carries.
#
# ## Architecture note
#
# The atmosphere `FieldTimeSeries` here live directly on the 1 km land
# grid rather than on ERA5's native ≈ 28 km grid. We pre-bake the
# lapse correction into them at construction time, which means the
# state exchanger's atmosphere → exchange-grid interpolation is the
# identity. A "proper" downscaling exchanger that performs the
# bilinear regrid *and* applies the elevation correction belongs in
# `EarthSystemModels.InterfaceComputations`; this example treats the
# downscaling as a pre-processing step on the inputs.
#
# ## CDS API credentials
#
# Downloading ERA5 fields requires CDS API credentials at `~/.cdsapirc`;
# see <https://cds.climate.copernicus.eu/how-to-api>.

using NumericalEarth
using NumericalEarth.Lands
using NumericalEarth.Radiations: PrescribedRadiation, SurfaceRadiationProperties
using NumericalEarth.Atmospheres: PrescribedAtmosphere, PrescribedPrecipitationFlux
using NumericalEarth.DataWrangling: Metadata, Metadatum, BoundingBox
using NumericalEarth.DataWrangling.ERA5
using NumericalEarth.Bathymetry: regrid_bathymetry
using CDSAPI            # Activates `NumericalEarthCDSAPIExt` (ERA5 downloads).
using Oceananigans
using Oceananigans.Fields: Center, set!
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
# Load CairoMakie up front so its precompile happens before the
# simulation builds 200 × 200 × 72 atmosphere FTS in memory; at that
# resolution, deferring the Makie precompile until after `run!` has
# segfaulted on this machine.
using CairoMakie
# `Dates.hour` clashes with `Oceananigans.Units.hour`; import only what
# the script actually uses by name.
import Dates: DateTime, Hour
using Printf
using Statistics

# ## Domain — 2° × 2° Yellowstone box at ~1 km

arch = CPU()
FT   = Float64

const Γ_lapse  = 6.5e-3            # K m⁻¹ — environmental lapse rate
const g_acc    = 9.81              # m s⁻²
const Rd       = 287.052           # J kg⁻¹ K⁻¹
const ε_Rd_Rv  = 0.62198

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
# regrid ETOPO 2022 onto two grids: the 1 km `land_grid` and a coarse
# 0.25° "ERA5-like" grid covering the same region with the same
# topology. Box-averaging ETOPO onto the ERA5 grid approximates ERA5's
# effective grid-cell elevation; bilinearly upsampling that field
# back to 1 km gives the elevation each ERA5 cell "thinks" applies
# everywhere within it.

z_land_field = regrid_bathymetry(land_grid; dataset = ETOPO2022(),
                                 interpolation_passes = 1, minimum_depth = 0)

era5_pad      = 0.25   # one ERA5 cell of padding on every side
era5_lat_min  = lat_min - era5_pad
era5_lat_max  = lat_max + era5_pad
era5_lon_min  = lon_min - era5_pad
era5_lon_max  = lon_max + era5_pad
era5_Ny       = round(Int, (era5_lat_max - era5_lat_min) / 0.25)
era5_Nx       = round(Int, (era5_lon_max - era5_lon_min) / 0.25)

era5_grid = LatitudeLongitudeGrid(arch, FT;
                                  size      = (era5_Nx, era5_Ny, 1),
                                  latitude  = (era5_lat_min, era5_lat_max),
                                  longitude = (era5_lon_min, era5_lon_max),
                                  z         = (-1.0, 0.0),
                                  topology  = (Bounded, Bounded, Bounded))

z_era5_field = regrid_bathymetry(era5_grid; dataset = ETOPO2022(),
                                 interpolation_passes = 1, minimum_depth = 0)

# Project the ERA5-effective elevation onto the 1 km land grid by a
# piecewise-constant lookup: each 1 km cell inherits the value of the
# ERA5 cell that contains its centre. This is faithful to ERA5's
# grid-mean semantics (a single effective elevation per ERA5 cell).
# `set!(land_field, era5_field)` between different-size fields still
# routes through a same-shape broadcast in Oceananigans main; until
# the proper bilinear `set!` lands we use this explicit lookup.
λ_land = λnodes(land_grid, Center())
φ_land = φnodes(land_grid, Center())

z_land_raw = interior(z_land_field, :, :, 1)
z_era5_raw = interior(z_era5_field, :, :, 1)

z_land = max.(z_land_raw, 0.0)
z_era5_eff = similar(z_land)
@inbounds for j in eachindex(φ_land), i in eachindex(λ_land)
    i_e = clamp(floor(Int, (λ_land[i] - era5_lon_min) / 0.25) + 1, 1, era5_Nx)
    j_e = clamp(floor(Int, (φ_land[j] - era5_lat_min) / 0.25) + 1, 1, era5_Ny)
    z_era5_eff[i, j] = max(z_era5_raw[i_e, j_e], 0.0)
end
Δz = z_land .- z_era5_eff                # m, positive over peaks

@info "Elevation field stats" land_min_m=minimum(z_land) land_max_m=maximum(z_land) era5_min_m=minimum(z_era5_eff) era5_max_m=maximum(z_era5_eff) Δz_min_m=minimum(Δz) Δz_max_m=maximum(Δz)

# ## ERA5 forcing — 3-day window
#
# Three days of hourly data is enough for two diurnal cycles plus a
# synoptic pulse, and keeps the data download + simulation under
# ~10 min on CPU.

dates   = DateTime(2020, 4, 1):Hour(1):DateTime(2020, 4, 3, 23)
region  = BoundingBox(latitude = (era5_lat_min, era5_lat_max),
                      longitude = (era5_lon_min, era5_lon_max))
dataset = ERA5HourlySingleLevel()

# Load ERA5 fields, pre-interpolated onto the land grid. The
# `FieldTimeSeries(metadata, land_grid)` form lets the dataset backend
# bilinearly downscale from ERA5 native to the 1 km land grid as the
# data is loaded.

u10_meta  = Metadata(:eastward_velocity;               dataset, dates, region)
v10_meta  = Metadata(:northward_velocity;              dataset, dates, region)
T2m_meta  = Metadata(:temperature;                     dataset, dates, region)
Td_meta   = Metadata(:dewpoint_temperature;            dataset, dates, region)
sp_meta   = Metadata(:surface_pressure;                dataset, dates, region)
tp_meta   = Metadata(:total_precipitation;             dataset, dates, region)
ssrd_meta = Metadata(:downwelling_shortwave_radiation; dataset, dates, region)
strd_meta = Metadata(:downwelling_longwave_radiation;  dataset, dates, region)

Nt = length(dates)

u10  = FieldTimeSeries(u10_meta,  land_grid; time_indices_in_memory = Nt)
v10  = FieldTimeSeries(v10_meta,  land_grid; time_indices_in_memory = Nt)
T2m  = FieldTimeSeries(T2m_meta,  land_grid; time_indices_in_memory = Nt)
Td2  = FieldTimeSeries(Td_meta,   land_grid; time_indices_in_memory = Nt)
sp   = FieldTimeSeries(sp_meta,   land_grid; time_indices_in_memory = Nt)
tp   = FieldTimeSeries(tp_meta,   land_grid; time_indices_in_memory = Nt)
ssrd = FieldTimeSeries(ssrd_meta, land_grid; time_indices_in_memory = Nt)
strd = FieldTimeSeries(strd_meta, land_grid; time_indices_in_memory = Nt)

atmos_times = u10.times

# ## Elevation-corrected atmosphere
#
# Build new fields on the 1 km grid carrying the lapse- and
# hydrostatic-corrected (T, p, q) plus rate-converted rainfall and
# radiation. The ERA5 *accumulated* SW/LW (J m⁻² over the past hour)
# is divided by 3600 s to give power (W m⁻²) that `PrescribedRadiation`
# expects; total precipitation in m liquid-water-equivalent per hour
# becomes kg m⁻² s⁻¹.

to_kg_m2_per_s = 1000.0 / 3600.0
to_W_per_m2    = 1.0 / 3600.0

@inline saturation_vapor_pressure(T) =
    611.2 * exp(17.62 * (T - 273.15) / (T - 30.04))

@inline specific_humidity_from_dewpoint(Td, p) =
    (e = saturation_vapor_pressure(Td); ε_Rd_Rv * e / (p - (1 - ε_Rd_Rv) * e))

T_local    = FieldTimeSeries{Center, Center, Nothing}(land_grid, atmos_times)
p_local    = FieldTimeSeries{Center, Center, Nothing}(land_grid, atmos_times)
q_local    = FieldTimeSeries{Center, Center, Nothing}(land_grid, atmos_times)
rain       = FieldTimeSeries{Center, Center, Nothing}(land_grid, atmos_times)
ssrd_rate  = FieldTimeSeries{Center, Center, Nothing}(land_grid, atmos_times)
strd_rate  = FieldTimeSeries{Center, Center, Nothing}(land_grid, atmos_times)

ΓΔz = Γ_lapse .* Δz                                 # K, lapse correction

for n in 1:Nt
    T_era5 = interior(T2m[n], :, :, 1)
    p_era5 = interior(sp[n],  :, :, 1)
    Td_n   = interior(Td2[n], :, :, 1)

    ## Lapse + hydrostatic correction.
    T_loc = T_era5 .- ΓΔz
    T_avg = 0.5 .* (T_era5 .+ T_loc)
    p_loc = p_era5 .* exp.(.- g_acc .* Δz ./ (Rd .* T_avg))

    interior(T_local[n], :, :, 1) .= T_loc
    interior(p_local[n], :, :, 1) .= p_loc
    interior(q_local[n], :, :, 1) .= specific_humidity_from_dewpoint.(Td_n, p_loc)

    interior(rain[n],      :, :, 1) .= max.(interior(tp[n],   :, :, 1) .* to_kg_m2_per_s, 0)
    interior(ssrd_rate[n], :, :, 1) .= max.(interior(ssrd[n], :, :, 1) .* to_W_per_m2,    0)
    interior(strd_rate[n], :, :, 1) .= max.(interior(strd[n], :, :, 1) .* to_W_per_m2,    0)
end

# ## Prescribed atmosphere and radiation
#
# Atmosphere grid == land grid, so no further regridding at runtime.

atmosphere = PrescribedAtmosphere(land_grid, atmos_times;
                                  velocities      = (u = u10, v = v10),
                                  tracers         = (T = T_local, q = q_local),
                                  pressure        = p_local,
                                  freshwater_flux = PrescribedPrecipitationFlux(rain = rain, snow = nothing),
                                  surface_layer_height  = 10.0,
                                  boundary_layer_height = 800.0)

radiation = PrescribedRadiation(ssrd_rate, strd_rate;
                                ocean_surface   = nothing,
                                sea_ice_surface = nothing,
                                land_surface    = SurfaceRadiationProperties(0.18, 0.95))

# ## Slab land closures
#
# A 15 cm thermal soil slab, Manabe single-bucket hydrology, uniform
# aerodynamic roughness for mid-vegetation / open shrubland.

ρcH_g = 1500.0 * 1480.0 * 0.10

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

# Initialize T₀ from the elevation-corrected ERA5 T₂ₘ at the first
# snapshot (within ~3 K of equilibrium for early April). Fill the
# parent first so halo cells start at the domain-mean rather than
# uninitialised memory, then overwrite the interior with the
# lapse-corrected field. `fill_halo_regions!` (via `update_state!`)
# will propagate the boundary interior cells back into the halos via
# the NoFlux BCs.
T_init = interior(T_local[1], :, :, 1)
fill!(parent(slab_land.state.T), mean(T_init))
interior(slab_land.state.T, :, :, 1) .= T_init
fill!(parent(slab_land.state.W), 0.5 * 150.0)
Oceananigans.TimeSteppers.update_state!(slab_land)

# ## Coupled model

model      = AtmosphereLandModel(atmosphere, slab_land; radiation)
Δt         = 5minutes
stop_time  = (Nt - 1) * 3600.0
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

add_callback!(simulation, progress, IterationInterval(144))   # ~12 h

# ## Output writers

outputs = (T = slab_land.state.T,
           W = slab_land.state.W,
           β = slab_land.state.moisture_availability,
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
# before the animation re-opens the file for reading. At 200 × 200
# the combined working set is large enough that leaving these
# references live can segfault the same Julia session during heatmap
# setup.
close(simulation.output_writers[:land])
delete!(simulation.output_writers, :land)
atmosphere = simulation = model = nothing
u10 = v10 = T2m = Td2 = sp = tp = ssrd = strd = nothing
T_local = p_local = q_local = rain = ssrd_rate = strd_rate = nothing
GC.gc(true); GC.gc(true)

# ## Animation
#
# Three spatial panels — T, β, Q — over the elevation, plus a static
# elevation reference panel below. The elevation panel uses the same
# colormap range as the T panel difference: viewers can directly read
# off the lapse-rate signature.

T_ts = FieldTimeSeries("era5_forced_slab_land.jld2", "T"; architecture = CPU())
β_ts = FieldTimeSeries("era5_forced_slab_land.jld2", "β"; architecture = CPU())
Q_ts = FieldTimeSeries("era5_forced_slab_land.jld2", "Q"; architecture = CPU())

times = T_ts.times
Nframes = length(times)

λ = λnodes(land_grid, Center())
φ = φnodes(land_grid, Center())

times_days = collect(times) ./ 86400

fig = Figure(size = (1500, 1000), fontsize = 12)

ax_T  = Axis(fig[1, 1]; title = "Skin temperature T (K)",     xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
ax_β  = Axis(fig[1, 2]; title = "Moisture availability β",    xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
ax_Q  = Axis(fig[1, 3]; title = "Net energy flux Q (W m⁻²)",  xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
ax_z  = Axis(fig[2, 1]; title = "Elevation (m, ETOPO 2022)",  xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
ax_t  = Axis(fig[2, 2:3]; title = "Domain T extrema and mean over time", xlabel = "t (days)", ylabel = "T (K)")

n = Observable(1)

Tn = @lift view(interior(T_ts[$n]), :, :, 1)
βn = @lift view(interior(β_ts[$n]), :, :, 1)
Qn = @lift view(interior(Q_ts[$n]), :, :, 1)

heatmap!(ax_T, λ, φ, Tn; colormap = :thermal, colorrange = (250, 300))
heatmap!(ax_β, λ, φ, βn; colormap = :tempo,   colorrange = (0, 1))
heatmap!(ax_Q, λ, φ, Qn; colormap = :balance, colorrange = (-400, 400))
heatmap!(ax_z, λ, φ, z_land; colormap = :terrain, colorrange = (1000, 3500))

Colorbar(fig[1, 1, Right()], limits = (250, 300),  colormap = :thermal, label = "T (K)")
Colorbar(fig[1, 2, Right()], limits = (0, 1),      colormap = :tempo,   label = "β")
Colorbar(fig[1, 3, Right()], limits = (-400, 400), colormap = :balance, label = "Q (W m⁻²)")
Colorbar(fig[2, 1, Right()], limits = (1000, 3500), colormap = :terrain, label = "elevation (m)")

T_dom_mean = [mean(interior(T_ts[k])) for k in 1:Nframes]
T_dom_max  = [maximum(interior(T_ts[k])) for k in 1:Nframes]
T_dom_min  = [minimum(interior(T_ts[k])) for k in 1:Nframes]
lines!(ax_t, times_days, T_dom_max;  color = :red,   linewidth = 1.5, label = "max")
lines!(ax_t, times_days, T_dom_mean; color = :black, linewidth = 1.5, label = "mean")
lines!(ax_t, times_days, T_dom_min;  color = :blue,  linewidth = 1.5, label = "min")
axislegend(ax_t; position = :rb)
t_now = @lift [times_days[$n]]
vlines!(ax_t, t_now; color = :black, linewidth = 1.0, linestyle = :dash)

title = @lift "ERA5-forced slab land — Greater Yellowstone at ~1 km, t = " * prettytime(times[$n])
Label(fig[0, 1:3], title, fontsize = 16)

@info "Rendering animation..."
CairoMakie.record(fig, "era5_forced_slab_land.mp4", 1:Nframes; framerate = 12) do nn
    n[] = nn
end
@info "Animation saved."

nothing #hide

# ![](era5_forced_slab_land.mp4)
