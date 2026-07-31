# # Deep soil temperature from ERA5-Land
#
# The `SlabLand` energy budget (`WaterCoupledEnergy` here) restores the land temperature
# `Tˡᵃ` toward a deep temperature `Tᵈᵉᵉᵖ` — the soil temperature at the depth where diurnal
# and synoptic variability is damped out. `Tᵈᵉᵉᵖ` sets the mean state the slab relaxes to,
# so a hand-set constant biases every cell whose true deep temperature differs from it: by
# several K per km of elevation over terrain, since `Tᵈᵉᵉᵖ` tracks the annual-mean surface
# temperature.
#
# This example grabs `Tᵈᵉᵉᵖ` from the [`ERA5MonthlyLand`](@ref) dataset — the monthly-mean
# [ERA5-Land](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means)
# reanalysis on its native 0.1° (~9 km) grid. Soil temperature level 4 (`stl4`, 100–289 cm)
# is the deepest layer of ECMWF's land model and the natural choice for a slab's lower
# thermal boundary. Over the European Alps we
#
# 1. download a five-year monthly climatology of all four soil temperature levels and check
#    that the seasonal cycle damps and lags with depth — the textbook signature of downward
#    heat diffusion;
# 2. time-average `stl4` into a static `Tᵈᵉᵉᵖ` map and lapse-correct it from ERA5-Land's
#    ~9 km cell-mean elevation to a ~1 km model grid, with a ground lapse rate fitted from
#    the product itself;
# 3. hand the corrected field to `WaterCoupledEnergy` and run an ERA5-forced `SlabLand`,
#    against a control run that restores to the hand-set default `Tᵈᵉᵉᵖ = 280` K.
#
# Downloading requires CDS credentials at `~/.cdsapirc`
# (see <https://cds.climate.copernicus.eu/how-to-api>) and acceptance of the ERA5-Land
# licence on the CDS portal (separate from the ERA5 licence).

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using CDSAPI                        # activates the CDS download extension
using CairoMakie
using Printf
using Statistics
using Downloads: download
using Oceananigans.Fields: interpolate!
import Dates: DateTime, Month       # `Dates.hour` clashes with `Oceananigans.Units.hour`

# ## Region and climatology window
#
# A 5° × 2.5° box over the European Alps: Po valley lowlands in the south, the main
# Alpine ridge (Mont Blanc, Monte Rosa, ~4 km of relief) through the middle, and the
# Swiss plateau in the north. Fully inland, so the land-only ERA5-Land fields have no
# ocean gaps here.

latitude  = (45, 47.5)
longitude = (6, 11)
region    = BoundingBox(; latitude, longitude)

dataset = ERA5MonthlyLand()
climatology_dates = DateTime(2020, 1, 1):Month(1):DateTime(2024, 12, 1)

# ## Download the soil temperature climatology
#
# The four ECMWF soil levels span 0–7, 7–28, 28–100, and 100–289 cm. One batched CDS
# request per calendar year fetches all four variables at once (60 months × 4 levels).

soil_temperature_names = [:soil_temperature_level_1, :soil_temperature_level_2,
                          :soil_temperature_level_3, :soil_temperature_level_4]

download(soil_temperature_names, dataset, climatology_dates; region)

# ## The seasonal cycle damps and lags with depth
#
# Averaging each level's domain-mean over the five years gives a 12-month climatology
# per depth. Heat diffusing downward from the surface loses amplitude and arrives late:
# the 0–7 cm layer swings with the surface, while the 100–289 cm layer carries a few-K
# cycle peaking one to two months after midsummer — the physical reason `stl4` is the
# right slowly-varying lower boundary condition.

soil_level_depths = ["0–7 cm", "7–28 cm", "28–100 cm", "100–289 cm"]

monthly_climatologies = map(soil_temperature_names) do name
    metadata = Metadata(name; dataset, dates = climatology_dates, region)
    fts = FieldTimeSeries(metadata, CPU(); time_indices_in_memory = length(climatology_dates))
    domain_means = [mean(fts[n]) for n in eachindex(fts.times)]
    [mean(domain_means[m:12:end]) for m in 1:12]
end

fig = Figure(size = (700, 450), fontsize = 14)
ax = Axis(fig[1, 1];
          title  = "ERA5-Land soil temperature over the Alps (2020–2024 climatology)",
          xlabel = "month", ylabel = "domain-mean T (K)", xticks = 1:12)
for (climatology, depth) in zip(monthly_climatologies, soil_level_depths)
    scatterlines!(ax, 1:12, climatology; label = depth)
end
axislegend(ax, "Soil level depth"; position = :rt)

save("era5_land_soil_temperature_cycle.png", fig)
nothing #hide

# ![](era5_land_soil_temperature_cycle.png)

# ## A static `Tᵈᵉᵉᵖ` map, lapse-corrected to the model terrain
#
# At 100–289 cm the five-year mean is an excellent estimate of the equilibrium deep
# temperature, and it is smooth — 9 km resolution loses nothing except *terrain*. The
# coarse field represents each ERA5-Land cell's mean elevation, so on a ~1 km grid the
# resolved ridges and valleys need the lapse-rate shift
#
#     Tᵈᵉᵉᵖ(z_model) = Tᵈᵉᵉᵖ(z_coarse) − Γ (z_model − z_coarse).
#
# `z_model` is ETOPO 2022 elevation regridded to the model grid; `z_coarse` is the same
# ETOPO elevation smoothed to ERA5-Land's native 0.1° grid — the elevation the coarse
# `Tᵈᵉᵉᵖ` "lives at" — and interpolated back. The correction therefore adds exactly the
# sub-9 km topographic signal.
#
# The right `Γ` for a *soil* field is the ground-temperature lapse rate, which is
# shallower than the free-air 6.5 K km⁻¹ in snowy terrain (the winter snowpack insulates
# high-altitude soil from the air above). Rather than hard-coding an atmospheric
# convention, fit `Γ` from the coarse product itself — the regression slope of the
# native-grid `Tᵈᵉᵉᵖ` against the native-grid elevation. Over this box the fit gives
# ~4.2 K km⁻¹ with correlation −0.89: elevation alone explains most of the spatial
# variance of the deep soil temperature.

stl4_metadata = Metadata(:soil_temperature_level_4; dataset, dates = climatology_dates, region)
stl4_native_grid = native_grid(stl4_metadata, CPU())

Tᵈ_native = time_averaged_field(stl4_metadata, CPU())
z_native  = regrid_topography(stl4_native_grid; dataset = ETOPO2022())

Tᵈ, z = vec(interior(Tᵈ_native)), vec(interior(z_native))
ground_lapse_rate = -cov(z, Tᵈ) / var(z)

@printf("Fitted ground lapse rate: %.2f K km⁻¹ (correlation %.2f)\n",
        1000 * ground_lapse_rate, cor(z, Tᵈ))

function deep_temperature_field(grid)
    Tᵈ_coarse = time_averaged_field(stl4_metadata, grid)
    z_model = regrid_topography(grid; dataset = ETOPO2022())
    z_coarse = Field{Center, Center, Nothing}(grid)
    interpolate!(z_coarse, z_native)
    return compute!(Field(Tᵈ_coarse - ground_lapse_rate * (z_model - z_coarse)))
end

map_grid = LatitudeLongitudeGrid(CPU(); latitude, longitude,
                                 size = (500, 250),
                                 topology = (Bounded, Bounded, Flat))

z_map  = regrid_topography(map_grid; dataset = ETOPO2022())
Tᵈ_raw = time_averaged_field(stl4_metadata, map_grid)
Tᵈ_map = deep_temperature_field(map_grid)

# The raw map only carries the smooth 9 km terrain signal; the corrected map is several K
# colder on the resolved ridges and warmer in the incised valleys.

fig = Figure(size = (1300, 750), fontsize = 14)

ax_z = Axis(fig[1, 1]; title = "Elevation (m, ETOPO 2022)",       xlabel = "longitude", ylabel = "latitude")
ax_r = Axis(fig[1, 3]; title = "Annual-mean stl4 (K), raw 0.1°",  xlabel = "longitude", ylabel = "latitude")
ax_c = Axis(fig[2, 1]; title = "Tᵈᵉᵉᵖ (K), lapse-corrected",      xlabel = "longitude", ylabel = "latitude")
ax_d = Axis(fig[2, 3]; title = "Correction (K): −Γ (z₁ₖₘ − z₉ₖₘ)", xlabel = "longitude", ylabel = "latitude")

Tlim = extrema(interior(Tᵈ_map))

hm_z = heatmap!(ax_z, z_map;  colormap = :terrain)
hm_r = heatmap!(ax_r, Tᵈ_raw; colormap = :turbo, colorrange = Tlim)
hm_c = heatmap!(ax_c, Tᵈ_map; colormap = :turbo, colorrange = Tlim)
hm_d = heatmap!(ax_d, Tᵈ_map - Tᵈ_raw; colormap = :balance, colorrange = (-8, 8))

Colorbar(fig[1, 2], hm_z; label = "z (m)")
Colorbar(fig[1, 4], hm_r; label = "T (K)")
Colorbar(fig[2, 2], hm_c; label = "T (K)")
Colorbar(fig[2, 4], hm_d; label = "ΔT (K)")

save("era5_land_deep_temperature_maps.png", fig)
nothing #hide

# ![](era5_land_deep_temperature_maps.png)

# ## An ERA5-forced slab land restoring to the data-driven `Tᵈᵉᵉᵖ`
#
# `WaterCoupledEnergy` accepts `deep_temperature` as a `Field`, so the corrected map drops
# straight into the constructor. We run four July days at ~2.5 km forced by hourly ERA5
# single-level reanalysis (with the usual `ElevationCorrection` lifting the ~28 km
# atmosphere state onto the 1 km terrain), and repeat the run with the hand-set default
# `deep_temperature = 280` — a value 9 K too cold for the Po valley and 15 K too warm for
# the high ridge.

arch = CPU()

run_grid = LatitudeLongitudeGrid(arch; latitude, longitude,
                                 size = (200, 100),
                                 topology = (Bounded, Bounded, Flat))

Tᵈ_run = deep_temperature_field(run_grid)

forcing_dataset = ERA5HourlySingleLevel()
start_date = DateTime(2020, 7, 1)
end_date   = DateTime(2020, 7, 5)
run_time   = 4days

# The forcing correction between ERA5's own model surface and the resolved terrain. This
# one lifts the *air* state, so it keeps `ElevationCorrection`'s default free-air lapse
# rate (6.5 K km⁻¹) rather than the fitted ground lapse rate:

z_run  = regrid_topography(run_grid; dataset = ETOPO2022())
z_era5 = Field(Metadatum(:topography; dataset = forcing_dataset, date = start_date, region), run_grid)
forcing_correction = ElevationCorrection(z_run, z_era5)

# Both runs share the forcing, hydrology, and initial state `Tˡᵃ = Tᵈᵉᵉᵖ`,
# `Mˡᵃ = 150 kg m⁻²`; only `deep_temperature` differs.

function run_slab_land(deep_temperature, label)
    atmosphere = ERA5PrescribedAtmosphere(arch; dataset = forcing_dataset, start_date, end_date, region,
                                          surface_layer_height = 10, boundary_layer_height = 800)
    radiation = ERA5PrescribedRadiation(arch; dataset = forcing_dataset, start_date, end_date, region,
                                        land_surface = SurfaceRadiationProperties(0.18, 0.95))

    energy = WaterCoupledEnergy(eltype(run_grid); deep_temperature, deep_time_scale = 12hours)
    hydrology = VariablySaturatedHydrology(eltype(run_grid);
                                           slab_depth             = 1,
                                           porosity               = 0.4,
                                           residual_liquid_fraction = 0.05,
                                           storage_height         = 1000,
                                           retention_curve        = VanGenuchtenRetention(α = 1.0, n = 2.0),
                                           hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-7, n = 2.0),
                                           deep_liquid_flux       = NoDeepLiquidFlux(),
                                           runoff                 = InfiltrationCapacityRunoff(infiltration_capacity = 1e-3))

    slab_land = SlabLand(run_grid; energy, hydrology)
    set!(slab_land; T = Tᵈ_run, M = 150)

    model = AtmosphereLandModel(atmosphere, slab_land; radiation,
                                exchanger_correction = forcing_correction)

    simulation = Simulation(model; Δt = 5minutes, stop_time = run_time)

    progress(sim) = @info @sprintf("[%s] iter %d, t = %s, T ∈ [%.1f, %.1f] K",
                                   label, iteration(sim), prettytime(sim),
                                   minimum(slab_land.temperature), maximum(slab_land.temperature))
    add_callback!(simulation, progress, IterationInterval(288))

    simulation.output_writers[:land] = JLD2Writer(model, (; T = slab_land.temperature);
                                                  filename = "era5_land_deep_temperature_$label",
                                                  schedule = TimeInterval(1hour),
                                                  overwrite_existing = true)
    run!(simulation)
    return nothing
end

run_slab_land(Tᵈ_run, "era5land")
run_slab_land(280,    "constant")

# ## The deep temperature sets where the slab equilibrates
#
# Both runs start from the same state, feel the same corrected atmosphere, and differ only
# in the reservoir they restore to. Within a couple of days the control run drifts toward
# its flat 280 K reservoir — warming the high ridge and cooling the lowlands — while the
# data-driven run holds the terrain-following equilibrium. The difference map at the final
# time recovers the imprint of `Tᵈᵉᵉᵖ − 280`.

T_era5land = FieldTimeSeries("era5_land_deep_temperature_era5land.jld2", "T")
T_constant = FieldTimeSeries("era5_land_deep_temperature_constant.jld2", "T")

times_days = T_era5land.times ./ days
Nt = length(times_days)

fig = Figure(size = (1300, 750), fontsize = 14)

ax_t = Axis(fig[1, 1:5];
            title  = "Domain-mean land temperature",
            xlabel = "t (days)", ylabel = "⟨Tˡᵃ⟩ (K)")
lines!(ax_t, times_days, [mean(T_era5land[n]) for n in 1:Nt]; color = :royalblue, label = "Tᵈᵉᵉᵖ from ERA5-Land")
lines!(ax_t, times_days, [mean(T_constant[n]) for n in 1:Nt]; color = :firebrick, label = "Tᵈᵉᵉᵖ = 280 K")
hlines!(ax_t, [mean(Tᵈ_run)]; color = :royalblue, linestyle = :dash, label = "⟨Tᵈᵉᵉᵖ⟩ (data)")
hlines!(ax_t, [280];          color = :firebrick, linestyle = :dash, label = "280 K")
axislegend(ax_t; position = :lt, nbanks = 2)

ax_1 = Axis(fig[2, 1]; title = "Final Tˡᵃ (K), data Tᵈᵉᵉᵖ",     xlabel = "longitude", ylabel = "latitude")
ax_2 = Axis(fig[2, 2]; title = "Final Tˡᵃ (K), Tᵈᵉᵉᵖ = 280 K",  xlabel = "longitude", ylabel = "latitude")
ax_3 = Axis(fig[2, 4]; title = "Difference (K): constant − data", xlabel = "longitude", ylabel = "latitude")

Tlim = extrema(interior(T_era5land[Nt]))

hm_1 = heatmap!(ax_1, T_era5land[Nt]; colormap = :turbo, colorrange = Tlim)
hm_2 = heatmap!(ax_2, T_constant[Nt]; colormap = :turbo, colorrange = Tlim)
hm_3 = heatmap!(ax_3, T_constant[Nt] - T_era5land[Nt]; colormap = :balance, colorrange = (-10, 10))

Colorbar(fig[2, 3], hm_1; label = "T (K)")
Colorbar(fig[2, 5], hm_3; label = "ΔT (K)")

save("era5_land_deep_temperature_run.png", fig)
nothing #hide

# ![](era5_land_deep_temperature_run.png)
