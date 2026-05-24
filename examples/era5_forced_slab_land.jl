# # ERA5-forced slab land — Greater Yellowstone / Snake River Plain
#
# A coarse-resolution land-only simulation forced by ERA5 reanalysis
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
# coupled to a `PrescribedAtmosphere` (ERA5 hourly single-level fields)
# and a `PrescribedRadiation` (ERA5 downwelling SW/LW, with explicit
# land albedo/emissivity) through `AtmosphereLandModel`. Turbulent
# fluxes use Monin–Obukhov similarity theory; surface humidity is
# β-reduced (`qₛ = qₐ + β·[q⁺(Tₛ) − qₐ]`) using the slab's bucket
# moisture availability.
#
# The terrain itself is not resolved here — `SlabLand` is spatially
# uniform in its closure properties — but ERA5's native topography
# drives strong spatial gradients in T₂ₘ, downwelling shortwave, and
# precipitation across the region, so land surface temperature tracks
# elevation-driven contrasts between the Snake River Plain, the
# Wasatch, and the Yellowstone–Wind River high country.
#
# ## CDS API credentials
#
# Downloading ERA5 fields requires CDS API credentials at `~/.cdsapirc`;
# see <https://cds.climate.copernicus.eu/how-to-api>.

using NumericalEarth
using NumericalEarth.Lands
using NumericalEarth.Radiations: PrescribedRadiation, SurfaceRadiationProperties
using NumericalEarth.Atmospheres: PrescribedAtmosphere, PrescribedPrecipitationFlux
using NumericalEarth.DataWrangling: Metadata, BoundingBox
using NumericalEarth.DataWrangling.ERA5
using CDSAPI            # Activates `NumericalEarthCDSAPIExt` (ERA5 downloads).
using Oceananigans
using Oceananigans.Fields: Center
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
# `Dates.hour` clashes with `Oceananigans.Units.hour`; import only what
# the script actually uses by name.
import Dates: DateTime, Hour
using Printf
using Statistics

# ## Domain — 5° × 5° regional patch at ERA5's native 0.25° resolution
#
# 40.5°N–45.5°N, 114°W–109°W. 20 × 20 = 400 cells; trivial on CPU.
# The land grid uses `(Bounded, Bounded, Flat)` because `SlabLand`
# operates as a 2-D component; ERA5 fields live on a 3-D cutout that
# the atmosphere exchanger regrids onto this land grid each step.

arch = CPU()
FT   = Float64

# The land grid carries a degenerate single-cell Bounded z column rather
# than `Flat` in z so the atmosphere and radiation regridders can build
# fractional indices against ERA5's 3-D (single-layer) native grid; the
# `SlabLand` kernels themselves only index `[i, j, 1]`.
land_grid = LatitudeLongitudeGrid(arch, FT;
                                  size      = (20, 20, 1),
                                  latitude  = (40.5, 45.5),
                                  longitude = (-114.0, -109.0),
                                  z         = (-1.0, 0.0),
                                  topology  = (Bounded, Bounded, Bounded))

# ## ERA5 forcing window
#
# A one-week window in early April 2020 captures the spring
# snow/freeze-melt transition over the higher terrain, which is when
# LST gradients across the region are most dramatic. 168 hourly
# snapshots, ~5 MB per variable for this small region.

dates   = DateTime(2020, 4, 1):Hour(1):DateTime(2020, 4, 7, 23)
# The atmospheric forcing region is one ERA5 cell (0.25°) larger than
# the land grid on every side so the exchange-grid halo cells fall
# inside ERA5's regional cutout; otherwise the interpolator dereferences
# uninitialized memory at the halo and the flux solve sees nonsense
# temperatures (JRA55 sidesteps this because it's global).
region  = BoundingBox(latitude = (40.25, 45.75), longitude = (-114.25, -108.75))
dataset = ERA5HourlySingleLevel()

u10_meta  = Metadata(:eastward_velocity;               dataset, dates, region)
v10_meta  = Metadata(:northward_velocity;              dataset, dates, region)
T2m_meta  = Metadata(:temperature;                     dataset, dates, region)
Td_meta   = Metadata(:dewpoint_temperature;            dataset, dates, region)
sp_meta   = Metadata(:surface_pressure;                dataset, dates, region)
tp_meta   = Metadata(:total_precipitation;             dataset, dates, region)
ssrd_meta = Metadata(:downwelling_shortwave_radiation; dataset, dates, region)
strd_meta = Metadata(:downwelling_longwave_radiation;  dataset, dates, region)

# Load into `FieldTimeSeries`. Setting `time_indices_in_memory =
# length(dates)` keeps everything in RAM (the regional cutout is tiny)
# so we can iterate over snapshots to derive q₂ₘ and the rainfall rate.

Nt = length(dates)

u10  = FieldTimeSeries(u10_meta;  time_indices_in_memory = Nt)
v10  = FieldTimeSeries(v10_meta;  time_indices_in_memory = Nt)
T2m  = FieldTimeSeries(T2m_meta;  time_indices_in_memory = Nt)
Td2  = FieldTimeSeries(Td_meta;   time_indices_in_memory = Nt)
sp   = FieldTimeSeries(sp_meta;   time_indices_in_memory = Nt)
tp   = FieldTimeSeries(tp_meta;   time_indices_in_memory = Nt)
ssrd = FieldTimeSeries(ssrd_meta; time_indices_in_memory = Nt)
strd = FieldTimeSeries(strd_meta; time_indices_in_memory = Nt)

atmos_grid  = u10.grid
atmos_times = u10.times

# ## Derived fields
#
# ERA5 single levels expose 2 m dewpoint, not specific humidity. We
# convert with Magnus's saturation-vapor-pressure formula over liquid
# water and Raoult's mixing-ratio relation:
#
#   e_sat(T) ≈ 611.2 · exp(17.62 · (T − 273.15) / (T − 30.04))   [Pa]
#   q = ε · e / (p − (1−ε) · e),   ε = R_d/R_v ≈ 0.622
#
# Total precipitation is the *accumulated* hourly water-equivalent in
# meters; we convert to a kg m⁻² s⁻¹ rate by ×1000/3600.

const ε_Rd_Rv = 0.62198

@inline saturation_vapor_pressure(T) =
    611.2 * exp(17.62 * (T - 273.15) / (T - 30.04))

@inline specific_humidity_from_dewpoint(Td, p) =
    (e = saturation_vapor_pressure(Td); ε_Rd_Rv * e / (p - (1 - ε_Rd_Rv) * e))

to_kg_m2_per_s = 1000.0 / 3600.0   # m / hr  →  kg m⁻² s⁻¹

# Downwelling SW/LW radiation come from CDS as *accumulated* energy
# (J m⁻²) over the preceding hour, not instantaneous W m⁻². Convert by
# /3600 s so `PrescribedRadiation` (which expects W m⁻²) sees a power
# flux.
to_W_per_m2 = 1.0 / 3600.0          # J m⁻² over 1 hour  →  W m⁻²

q2m       = FieldTimeSeries{Center, Center, Nothing}(atmos_grid, atmos_times)
rain      = FieldTimeSeries{Center, Center, Nothing}(atmos_grid, atmos_times)
ssrd_rate = FieldTimeSeries{Center, Center, Nothing}(atmos_grid, atmos_times)
strd_rate = FieldTimeSeries{Center, Center, Nothing}(atmos_grid, atmos_times)

for n in 1:Nt
    Td_n = interior(Td2[n])
    p_n  = interior(sp[n])
    interior(q2m[n])       .= specific_humidity_from_dewpoint.(Td_n, p_n)
    interior(rain[n])      .= max.(interior(tp[n])   .* to_kg_m2_per_s, 0)
    interior(ssrd_rate[n]) .= max.(interior(ssrd[n]) .* to_W_per_m2,    0)
    interior(strd_rate[n]) .= max.(interior(strd[n]) .* to_W_per_m2,    0)
end

# ## Prescribed atmosphere and radiation
#
# ERA5 winds are at 10 m, so `surface_layer_height = 10`. The PBL
# height is a rough constant; for offline land-only use it only enters
# the gustiness floor.

atmosphere = PrescribedAtmosphere(atmos_grid, atmos_times;
                                  velocities      = (u = u10, v = v10),
                                  tracers         = (T = T2m, q = q2m),
                                  pressure        = sp,
                                  freshwater_flux = PrescribedPrecipitationFlux(rain = rain, snow = nothing),
                                  surface_layer_height  = 10.0,
                                  boundary_layer_height = 800.0)

# An albedo of 0.18 / emissivity of 0.95 is appropriate for the mix of
# bare soil, shrubland, and conifer in the Snake River Plain / Wasatch;
# the higher Yellowstone / Wind River terrain has higher snow-modified
# albedo in early April that this constant under-represents — a known
# limitation we'll revisit with a spatial property provider.

radiation = PrescribedRadiation(ssrd_rate, strd_rate;
                                ocean_surface   = nothing,
                                sea_ice_surface = nothing,
                                land_surface    = SurfaceRadiationProperties(0.18, 0.95))

# ## Slab land closures
#
# A 15 cm thermal soil slab (`dry_heat_capacity = ρ · c · H ≈ 1500 ·
# 1480 · 0.1 J m⁻² K⁻¹`) with a Manabe single-bucket hydrology and
# uniform aerodynamic roughness for mid-vegetation / open shrubland.

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

# Initialize interior + halo so the flux kernel never sees uninitialized
# halo cells (it iterates over the full halo-inclusive region).
fill!(parent(slab_land.state.T), 280.0)
fill!(parent(slab_land.state.W), 0.5 * 150.0)
Oceananigans.TimeSteppers.update_state!(slab_land)

# ## Coupled model
#
# `AtmosphereLandModel(atmos, land; radiation)` is a convenience
# wrapper around `EarthSystemModel` with `ocean = sea_ice = nothing`.
# The exchange grid is the land grid; ERA5 atmosphere and radiation
# are regridded onto it through fractional-index interpolators.

model = AtmosphereLandModel(atmosphere, slab_land; radiation)

Δt        = 10minutes
stop_time = (Nt - 1) * 3600.0
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
                                              filename = "era5_forced_slab_land",
                                              schedule = TimeInterval(1hour),
                                              overwrite_existing = true)

# ## Run

@info "Running ERA5-forced slab land simulation..."
run!(simulation)
@info "Simulation complete."

# ## Animation
#
# Three spatial panels — T, β, Q — plus a domain-mean T(t) time series
# at the bottom. Lat/lon axes anchor the geography: Yellowstone sits
# upper-right, Jackson just south of it, Pocatello toward the west
# edge, and Salt Lake City pokes the south.

using CairoMakie

T_ts = FieldTimeSeries("era5_forced_slab_land.jld2", "T"; architecture = CPU())
β_ts = FieldTimeSeries("era5_forced_slab_land.jld2", "β"; architecture = CPU())
Q_ts = FieldTimeSeries("era5_forced_slab_land.jld2", "Q"; architecture = CPU())

times = T_ts.times
Nframes = length(times)

λ = λnodes(land_grid, Center())
φ = φnodes(land_grid, Center())

times_days = collect(times) ./ 86400
T_dom_mean = [mean(interior(T_ts[k])) for k in 1:Nframes]
T_dom_max  = [maximum(interior(T_ts[k])) for k in 1:Nframes]
T_dom_min  = [minimum(interior(T_ts[k])) for k in 1:Nframes]

fig = Figure(size = (1500, 900), fontsize = 12)

ax_T = Axis(fig[1, 1]; title = "Skin temperature T (K)",      xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
ax_β = Axis(fig[1, 2]; title = "Moisture availability β",     xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
ax_Q = Axis(fig[1, 3]; title = "Net energy flux Q (W m⁻²)",   xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())

ax_t = Axis(fig[2, 1:3]; title = "Domain T extrema and mean over time", xlabel = "t (days)", ylabel = "T (K)")

n = Observable(1)

Tn = @lift view(interior(T_ts[$n]), :, :, 1)
βn = @lift view(interior(β_ts[$n]), :, :, 1)
Qn = @lift view(interior(Q_ts[$n]), :, :, 1)

heatmap!(ax_T, λ, φ, Tn; colormap = :thermal, colorrange = (255, 305))
heatmap!(ax_β, λ, φ, βn; colormap = :tempo,   colorrange = (0, 1))
heatmap!(ax_Q, λ, φ, Qn; colormap = :balance, colorrange = (-300, 300))

Colorbar(fig[1, 1, Right()], limits = (255, 305),  colormap = :thermal, label = "T (K)")
Colorbar(fig[1, 2, Right()], limits = (0, 1),      colormap = :tempo,   label = "β")
Colorbar(fig[1, 3, Right()], limits = (-300, 300), colormap = :balance, label = "Q (W m⁻²)")

lines!(ax_t, times_days, T_dom_max;  color = :red,   linewidth = 1.5, label = "max")
lines!(ax_t, times_days, T_dom_mean; color = :black, linewidth = 1.5, label = "mean")
lines!(ax_t, times_days, T_dom_min;  color = :blue,  linewidth = 1.5, label = "min")
axislegend(ax_t; position = :rb)
t_now = @lift [times_days[$n]]
vlines!(ax_t, t_now; color = :black, linewidth = 1.0, linestyle = :dash)

title = @lift "ERA5-forced slab land — Greater Yellowstone, t = " * prettytime(times[$n])
Label(fig[0, 1:3], title, fontsize = 16)

@info "Rendering animation..."
CairoMakie.record(fig, "era5_forced_slab_land.mp4", 1:Nframes; framerate = 12) do nn
    n[] = nn
end
@info "Animation saved."

nothing #hide

# ![](era5_forced_slab_land.mp4)
