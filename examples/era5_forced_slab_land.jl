# # A differentiable slab-land model: 0D column and 2D ERA5-forced map
#
# This example builds a differentiable land model in two parts: an idealized 0D column first,
# then a more realistic case forced by ERA5 over real topography.
# Both use a `SlabLand` composed of
#
#     energy    = WaterCoupledEnergy(...)          # cˡᵃ(Mˡᵃ) = cᵈʳʸ + cˡ Mˡᵃ, conservative dTˡᵃ/dt
#     hydrology = VariablySaturatedHydrology(...)  # augmented-storage ϑˡ budget
#     humidity  = DryLayerHumidity(...)            # qⁱⁿ from a dry-layer vapor-flux balance
#
# The example proceeds in four stages:
# 1. a 0D dry-layer slab under idealized analytic forcing;
# 2. the sensitivity of its skin temperature to the rain rate, ``∂T / ∂𝒫̇``, from a single reverse pass;
# 3. an ERA5-forced run over Central Borneo at ~1 km with downscaled elevation;
# 4. a pointwise ``∂T / ∂ν`` porosity-sensitivity map over a 2D sub-patch of that domain.
#
# Stages 2 and 4 compile the coupled time step to
# [XLA](https://en.wikipedia.org/wiki/Accelerated_Linear_Algebra) with
# [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) and differentiate it in
# reverse mode with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl).
#
# ## The dry-layer humidity closure
#
# A diagnostic dry-layer depth `δᵛ(𝒮) = δᵛ_max · max(1 − 𝒮/𝒮ᶜ, 0)²` vanishes while
# the surface is saturated (`𝒮 ≥ 𝒮ᶜ`) and grows as the slab dries below the onset
# saturation `𝒮ᶜ`. Vapor escapes through that layer with a piston velocity
# `wᵈ = Dᵛ_eff / δᵛ`, where `Dᵛ_eff` is the effective vapor diffusivity, so a saturated surface
# evaporates at the limit set by atmospheric demand while a dried-out one is throttled.
# [Or et al. (2013)](@cite or2013advances) explain this two-stage bare-soil drying in detail.
# We use a thin slab so that both stages appear within a short run. A rain pulse on day 2
# briefly over-saturates the slab (`𝒮 = 1`) and the latent heat flux rises to the
# saturated regime. Once evaporation draws the slab back down through the dry-layer onset
# `𝒮ᶜ`, the dry layer reopens and the latent heat flux collapses again.

# ## Load packages
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using CDSAPI                         # activates the CDS-API extension (Stages 3–4)
using CairoMakie
using Printf
using Statistics                     # mean
import Dates: DateTime, Hour         # `Dates.hour` clashes with `Oceananigans.Units.hour`
using Oceananigans.TimeSteppers: Clock, update_state!
using Oceananigans.Fields: interpolate!
using Reactant, CUDA                 # CUDA loads the Reactant KernelAbstractions extension (Stages 2 & 4)
using Enzyme
using Oceananigans.Architectures: ReactantState
using Reactant: @trace

# ## Idealized forcing
#
# The idealized forcing represents a warm, dry 280–290 K regime that keeps the evaporative
# demand high. A six-hour rain pulse falls on day 2 with amplitude `𝒫̇` (nominally
# `6e-4 kg m⁻² s⁻¹`, ≈ 13 mm over the six hours).

air_temperature(t)       = 285 - 5 * cos(2π * t / day)               # K, 280–290 diurnal
downwelling_shortwave(t) = max(0, 600 * cos(2π * (t - day/2) / day)) # W m⁻², daytime only
downwelling_longwave(t)  = 320                                       # W m⁻², constant
wind_speed               = 4.0                                       # m s⁻¹
specific_humidity        = 0.004                                     # kg kg⁻¹ (dry air drives evaporation)
surface_pressure         = 101325                                    # Pa

nominal_rain_rate = 6e-4                                             # kg m⁻² s⁻¹

in_rain_pulse(t) = 2days ≤ t ≤ 2days + 6hours
rain_pulse(t, 𝒫̇) = ifelse(in_rain_pulse(t), 𝒫̇, zero(𝒫̇))

# ## Forcing builders
#
# `forced_atmosphere` and `forced_radiation` fill each `FieldTimeSeries` time slice from the
# analytic functions.

function forced_atmosphere(grid, times, 𝒫̇)
    atmosphere = PrescribedAtmosphere(grid, times; surface_layer_height = 10, boundary_layer_height = 512)
    set!(atmosphere; u = wind_speed, q = specific_humidity, p = surface_pressure)
    for n in eachindex(times)
        set!(atmosphere.temperature[n], air_temperature(times[n]))
        set!(atmosphere.precipitation_flux.rain[n], rain_pulse(times[n], 𝒫̇))
    end
    update_state!(atmosphere)
    return atmosphere
end

function forced_radiation(grid, times; albedo = 0.2, emissivity = 0.97)
    radiation = PrescribedRadiation(grid, times;
                                    land_surface  = SurfaceRadiationProperties(albedo, emissivity),
                                    ocean_surface = nothing, sea_ice_surface = nothing)
    for n in eachindex(times)
        set!(radiation.downwelling_shortwave[n], downwelling_shortwave(times[n]))
        set!(radiation.downwelling_longwave[n],  downwelling_longwave(times[n]))
    end
    update_state!(radiation)
    return radiation
end

# ## Define land-model constants

liquid_density             = 1000.0
residual_liquid_fraction   = 0.05
dry_layer_onset_saturation = 0.5   # onset saturation 𝒮ᶜ, read by `make_dry_layer_humidity`

# ## The dry-layer slab
#
# We build a thin (5 cm) loamy slab, so that the daytime evaporative demand dries it back
# across `𝒮ᶜ` within a few days of the pulse and the slab passes from the saturated to the
# throttled evaporation stage. We start *below* `𝒮ᶜ`, so a dry surface layer is already
# present (`δᵛ > 0`) and evaporation is initially throttled. The pulse amplitude stays below
# the soil infiltration capacity, so `InfiltrationCapacityRunoff` sheds no water. With
# `NoDeepLiquidFlux` closing the bottom, evaporation is the only water sink.
# The pulse drives the storage briefly past its saturation value `Mˡᵃ⁺ = ρˡ ν hˡᵃ`. Because
# the prognostic variable is the augmented liquid fraction, the storage can exceed
# saturation; the excess is carried as a positive pressure head (with `𝒮` pinned at 1).

porosity              = 0.4
slab_depth            = 0.05
maximum_water_storage = liquid_density * porosity * slab_depth   # Mˡᵃ⁺ = ρˡ ν hˡᵃ = 20 kg m⁻²

initial_saturation  = 0.4   # below 𝒮ᶜ — a dry layer is already present
initial_temperature = 280.0 # K

# The diagnostic surface saturation `𝒮(Mˡᵃ)` of the variably saturated slab is
# `𝒮 = clamp((θˡ − θʳ)/(ν − θʳ), 0, 1)` with `θˡ = min(Mˡᵃ/(ρˡ hˡᵃ), ν)`.

function saturation_from_storage(M)
    θˡ = min(M / (liquid_density * slab_depth), porosity)
    return clamp((θˡ - residual_liquid_fraction) / (porosity - residual_liquid_fraction), 0, 1)
end

initial_water_storage =
    (initial_saturation * (porosity - residual_liquid_fraction) + residual_liquid_fraction) *
    liquid_density * slab_depth

# ## Model builders

# `WaterCoupledEnergy` is the slab's energy budget. It steps the skin temperature `Tˡᵃ`
# with a force-restore balance toward a deep reservoir at `deep_temperature` over the
# `deep_time_scale`, and folds the water storage into the areal heat capacity
# `cˡᵃ(Mˡᵃ) = cᵈʳʸ + cˡ Mˡᵃ`. The heat capacity is recomputed every step, so a wetter slab
# carries more thermal inertia. `dry_heat_capacity` is `cᵈʳʸ` (depth × density × specific
# heat of the dry soil) and `liquid_heat_capacity` is `cˡ`.
soil_energy(FT; deep_time_scale) = WaterCoupledEnergy(FT;
    dry_heat_capacity     = 0.1 * 1500 * 1480,
    liquid_heat_capacity  = 4186,
    reference_temperature = 273.15,
    deep_temperature      = 280,
    deep_time_scale)

# `VariablySaturatedHydrology` is the soil-water budget. It steps the column water
# storage `Mˡᵃ` with its surface flux (precipitation minus evaporation, limited by
# the runoff scheme) and its bottom flux, and diagnoses the surface saturation
# `𝒮(Mˡᵃ)` through the van Genuchten retention and conductivity curves.
variably_saturated_hydrology(FT, porosity; slab_depth) = VariablySaturatedHydrology(FT;
    slab_depth, porosity, residual_liquid_fraction,
    storage_height         = 1000,
    retention_curve        = VanGenuchtenRetention(inverse_air_entry_head = 1, pore_size_uniformity = 2),
    hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 1e-7, pore_size_uniformity = 2),
    deep_liquid_flux       = NoDeepLiquidFlux(),
    runoff                 = InfiltrationCapacityRunoff(infiltration_capacity = 1e-3))

# In the dry-layer humidity closure, the dry-layer depth grows from 0 (saturated skin)
# toward `δᵛ_max` as the slab dries past `𝒮ᶜ`, and `qⁱⁿ` is solved from the
# vapor-flux balance across the dry layer.
make_dry_layer_humidity(porosity) = DryLayerHumidity(;
    dry_layer_depth = StorageBasedDryLayerDepth(; maximum_dry_layer_depth = 0.05,
                                                dry_layer_onset_saturation,
                                                dry_layer_exponent      = 2),
    vapor_exchange = DryLayerVaporPistonVelocity(minimum_dry_layer_depth = 1e-4,
                                                 molecular_diffusivity   = 2.5e-5,
                                                 tortuosity              = PowerLawTortuosity()),
    thermal_exchange_depth = 0.10,
    porosity)

# Next, we assemble the coupled model used by the *differentiable* stages.
# Its turbulent fluxes use a fixed-iteration Monin–Obukhov solver (`FixedIterations`)
# rather than the default tolerance-based `while` loop. The differentiated pass compiles
# the coupled time step to XLA, where a data-dependent `while` becomes an XLA op that is
# expensive to differentiate, whereas a fixed iteration count unrolls into a static graph
# that Enzyme can traverse. `Clock(grid)` is a plain `Float64` clock on the CPU but a
# Reactant `ConcreteRNumber` clock on a `ReactantState` grid. The latter advances the model
# time *inside* the compiled XLA loop, so the time-dependent forcing is sampled correctly
# at every step.
function coupled_slab_land_model(grid, atmosphere, radiation; energy, hydrology, humidity_porosity,
                                 exchanger_correction = nothing)
    slab_land = SlabLand(grid; energy, hydrology)
    fluxes = NumericalEarth.EarthSystemModels.InterfaceComputations.default_atmosphere_land_fluxes(
                 slab_land, eltype(grid); solver_stop_criteria = FixedIterations(8))
    al_interface = atmosphere_land_interface(grid, atmosphere, slab_land;
                                             specific_humidity = make_dry_layer_humidity(humidity_porosity),
                                             fluxes)
    return AtmosphereLandModel(atmosphere, slab_land; radiation,
                               atmosphere_land_interface = al_interface,
                               exchanger_correction, clock = Clock(grid))
end

# The 0D dry-layer slab restores weakly toward the deep reservoir (over 10 days), so the
# thin slab is free to dry down during the run.
function dry_layer_model(grid, times, 𝒫̇)
    energy     = soil_energy(eltype(grid); deep_time_scale = 10days)
    hydrology  = variably_saturated_hydrology(eltype(grid), porosity; slab_depth)
    atmosphere = forced_atmosphere(grid, times, 𝒫̇)
    radiation  = forced_radiation(grid, times)
    return coupled_slab_land_model(grid, atmosphere, radiation;
                                   energy, hydrology, humidity_porosity = porosity)
end

# ## Integration length
#
# We integrate the model for 5.84 days (`41² = 1681` steps) and differentiate the final
# skin temperature with respect to the rain-pulse amplitude `𝒫̇`, which gives the
# sensitivity of the skin temperature to the rain rate. Reverse-mode gradient
# checkpointing needs a perfect-square step count, hence `41²`.

Δt     = 5minutes
Nsteps = 41^2                              # 1681 steps ≈ 5.84 days
times  = range(0, Nsteps * Δt, step = 1hour)

# `pulse_indices` are the `times` slices inside the rain pulse.

pulse_indices = [n for n in eachindex(times) if in_rain_pulse(times[n])]

# ## Forward run
#
# We are now ready to run the 0D model. A `JLD2Writer` records the column state and the
# surface energy budget at every step.

grid = RectilinearGrid(CPU(); size = (), topology = (Flat, Flat, Flat))

forward_model = dry_layer_model(grid, times, nominal_rain_rate)
set!(forward_model.land; T = initial_temperature, M = initial_water_storage)

land             = forward_model.land
turbulent_fluxes = forward_model.interfaces.atmosphere_land_interface.fluxes
radiative_fluxes = forward_model.radiation.interface_fluxes.land

## H and LE are positive upward; Rₙ and G are positive into the surface.
outputs = (T  = land.temperature,
           M  = land.water_storage,
           𝒮  = land.saturation,
           H  = turbulent_fluxes.sensible_heat,
           LE = turbulent_fluxes.latent_heat,
           Rₙ = radiative_fluxes.downwelling_shortwave + radiative_fluxes.downwelling_longwave -
                radiative_fluxes.upwelling_longwave,
           G  = -land.fluxes.surface_energy_flux)

forward_simulation = Simulation(forward_model; Δt, stop_iteration = Nsteps)

forward_simulation.output_writers[:column] = JLD2Writer(forward_model, outputs;
                                                        filename = "dry_layer_slab_land",
                                                        schedule = IterationInterval(1),
                                                        overwrite_files = true)

run!(forward_simulation)

# ## Differentiating skin temperature against rain rate in the idealized model
#
# To differentiate through the model, we rebuild the same setup on a `ReactantState`
# grid, so that every array is an XLA buffer. We then wrap the coupled time stepping in
# a scalar objective, `final_skin_temperature`, compile it once, and differentiate it
# with respect to the rain-pulse amplitude.

Reactant.set_default_backend("cpu")

reactant_grid  = RectilinearGrid(ReactantState(); size = (), topology = (Flat, Flat, Flat))
reactant_model = dry_layer_model(reactant_grid, times, nominal_rain_rate)
dmodel         = Enzyme.make_zero(reactant_model)

# The differentiated input is the rain-pulse amplitude `𝒫̇`, carried as a
# one-element XLA buffer so that Enzyme can accumulate ``∂T / ∂𝒫̇`` into its shadow,
# `d𝒫̇`.

𝒫̇  = Reactant.to_rarray([nominal_rain_rate])
d𝒫̇ = Enzyme.make_zero(𝒫̇)

# ## The objective
#
# `final_skin_temperature` resets the slab to its initial state, sets the rain pulse
# from the current `𝒫̇`, runs `nsteps` coupled time steps inside a `@trace` loop, and
# returns the final skin temperature. Because the `set!`s happen inside the objective,
# the gradient follows the trajectory that starts from them.

function final_skin_temperature(model, 𝒫̇, pulse_indices, initial_temperature, initial_water_storage, Δt, nsteps)
    set!(model.land.temperature,   initial_temperature)
    set!(model.land.water_storage, initial_water_storage)
    set!(model.land.saturation,    saturation_from_storage(initial_water_storage))

    rate = sum(𝒫̇)  # 1-element reduction → differentiable scalar
    rain = model.atmosphere.precipitation_flux.rain
    for n in pulse_indices
        set!(rain[n], rate)
    end

    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end

    return mean(interior(model.land.temperature))
end

# ## The gradient wrapper
#
# We differentiate in reverse mode, with the model and the rain amplitude passed as
# `Duplicated` (primal and shadow) and the loop length and scalar parameters as `Const`.

function grad_final_skin_temperature(model, dmodel, 𝒫̇, d𝒫̇, pulse_indices, initial_temperature, initial_water_storage, Δt, nsteps)
    parent(d𝒫̇) .= 0
    _, T = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        final_skin_temperature, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(𝒫̇, d𝒫̇),
        Enzyme.Const(pulse_indices),
        Enzyme.Const(initial_temperature),
        Enzyme.Const(initial_water_storage),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps))
    return d𝒫̇, T
end

# ## Compilation and execution

@info "Compiling differentiated dry-layer land model — this may take a minute..."
compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_final_skin_temperature(
    reactant_model, dmodel, 𝒫̇, d𝒫̇, pulse_indices,
    initial_temperature, initial_water_storage, Δt, Nsteps)

d𝒫̇, primal_temperature = compiled_grad(reactant_model, dmodel, 𝒫̇, d𝒫̇, pulse_indices,
                                       initial_temperature, initial_water_storage, Δt, Nsteps)

adjoint_sensitivity = Array(d𝒫̇)[1]   # K / (kg m⁻² s⁻¹)

@info @sprintf("Adjoint:  T(t=%.2f d) = %.4f K,  ∂T/∂𝒫̇ = %.4e K / (kg m⁻² s⁻¹)",
               Nsteps * Δt / day, Reactant.to_number(primal_temperature), adjoint_sensitivity)

# ## Finite-difference check
#
# To check the reverse-mode derivative, we compare it with a centered finite-difference
# approximation of the same derivative. The two agree closely.

function readout_skin_temperature(𝒫̇)
    model = dry_layer_model(grid, times, 𝒫̇)
    set!(model.land; T = initial_temperature, M = initial_water_storage)
    for _ in 1:Nsteps
        time_step!(model, Δt)
    end
    return mean(model.land.temperature)
end

δ = 0.001 * nominal_rain_rate
finite_difference_sensitivity = (readout_skin_temperature(nominal_rain_rate + δ) -
                                 readout_skin_temperature(nominal_rain_rate - δ)) / 2δ

@info @sprintf("Finite difference:  ∂T/∂𝒫̇ ≈ %.4e K / (kg m⁻² s⁻¹)", finite_difference_sensitivity)

# ## Visualization

T_ts  = FieldTimeSeries("dry_layer_slab_land.jld2", "T")
M_ts  = FieldTimeSeries("dry_layer_slab_land.jld2", "M")
𝒮_ts  = FieldTimeSeries("dry_layer_slab_land.jld2", "𝒮")
H_ts  = FieldTimeSeries("dry_layer_slab_land.jld2", "H")
LE_ts = FieldTimeSeries("dry_layer_slab_land.jld2", "LE")
Rₙ_ts = FieldTimeSeries("dry_layer_slab_land.jld2", "Rₙ")
G_ts  = FieldTimeSeries("dry_layer_slab_land.jld2", "G")

t = T_ts.times / day

pulse_span  = (2, 2 + 6/24)
readout_day = Nsteps * Δt / day

## A pulse of amplitude 𝒫̇ lasting τ deposits 𝒫̇ τ kg m⁻² (mm) of rain.
pulse_duration = 6hours
sensitivity_per_millimeter = adjoint_sensitivity / pulse_duration

fig = Figure(size = (1600, 800), fontsize = 20)

## Column 1: the surface energy balance above the air-temperature forcing.
axR = Axis(fig[1, 1]; title = "Surface energy balance\n(positive: atmosphere → surface)",
           xlabel = "t (days)", ylabel = "flux (W m⁻²)")
vspan!(axR, pulse_span...; color = (:skyblue, 0.25))
lines!(axR, t, Rₙ_ts; color = :black,    label = "Net radiation")
lines!(axR, t, G_ts;  color = :seagreen, linestyle = :dash, label = "Energy into land")
axislegend(axR; position = :lt)

axTₐ = Axis(fig[2, 1]; title = "Prescribed air temperature", xlabel = "t (days)", ylabel = "Tₐ (K)")
vspan!(axTₐ, pulse_span...; color = (:skyblue, 0.25))
lines!(axTₐ, t, air_temperature.(T_ts.times); color = :purple)

## Column 2: the surface saturation above the turbulent fluxes it controls
## (as 𝒮 falls through 𝒮ᶜ the latent flux collapses and the sensible flux takes over).
ax𝒮 = Axis(fig[1, 2]; title = "Surface saturation", xlabel = "t (days)", ylabel = "𝒮")
vspan!(ax𝒮, pulse_span...; color = (:skyblue, 0.25))
lines!(ax𝒮, t, 𝒮_ts; color = :darkorange, label = "𝒮")
hlines!(ax𝒮, [dry_layer_onset_saturation]; color = :gray, linestyle = :dash, label = "Dry-layer onset")
ylims!(ax𝒮, 0, 1.05)
axislegend(ax𝒮; position = :rb)

axH = Axis(fig[2, 2]; title = "Turbulent heat fluxes\n(positive: surface → atmosphere)",
           xlabel = "t (days)", ylabel = "flux (W m⁻²)")
vspan!(axH, pulse_span...; color = (:skyblue, 0.25))
lines!(axH, t, LE_ts; color = :navy,   label = "Latent")
lines!(axH, t, H_ts;  color = :orange, label = "Sensible")
axislegend(axH; position = :lt)

## Column 3: the water storage above the skin temperature, whose final value is differentiated.
axM = Axis(fig[1, 3]; title = "Water storage", xlabel = "t (days)", ylabel = "Mˡᵃ (kg m⁻²)")
vspan!(axM, pulse_span...; color = (:skyblue, 0.25))
lines!(axM, t, M_ts; color = :navy, label = "Mˡᵃ")
hlines!(axM, [maximum_water_storage]; color = :gray, linestyle = :dash, label = "Saturation")
axislegend(axM; position = :rt)

axT = Axis(fig[2, 3];
           title  = @sprintf("Skin temperature, ∂T/∂(rain) ≈ %.3f K mm⁻¹", sensitivity_per_millimeter),
           xlabel = "t (days)", ylabel = "T (K)")
vspan!(axT, pulse_span...; color = (:skyblue, 0.25))
lines!(axT, t, T_ts; color = :firebrick)
vlines!(axT, [readout_day]; color = :black, linestyle = :dash, label = "Sensitivity readout time")
axislegend(axT; position = :lt)

Label(fig[0, 1:3], "Differentiable dry-layer slab: rain on day 2, evaporation, and dry-down")

save("differentiable_dry_layer_slab_land.png", fig)
nothing #hide

# ![](differentiable_dry_layer_slab_land.png)

# ## Scaling up: ERA5-forced slab land over Central Borneo
#
# We now run the same `SlabLand` stack as a high-resolution, land-only simulation over
# the Central Borneo highlands, forced by ERA5 reanalysis over the ETOPO 2022 surface
# elevation. The region is equatorial (snow-free), very rainy, fully inland, and has
# about 2 km of relief.
#
# The domain is a 2° × 2° box (0.5–2.5 °N × 113–115 °E, ≈ 220 × 220 km) at
# 200 × 200 = 40 000 cells (≈ 1 km resolution).
#
# `AtmosphereLandModel` couples the land to an [`ERA5PrescribedAtmosphere`](@ref) and an
# [`ERA5PrescribedRadiation`](@ref). These download the required ERA5 single-level fields
# (T₂ₘ, dewpoint, 10 m wind, surface pressure, total precipitation, and downwelling
# shortwave and longwave radiation) over the domain, derive the specific humidity from the
# dewpoint, and convert ERA5's accumulated radiation and precipitation to fluxes. The data
# live on ERA5's native grid, so the coupled model interpolates them onto the 1 km land
# exchange grid.
#
# ## CDS API credentials
#
# Downloading ERA5 fields requires CDS API credentials at `~/.cdsapirc`;
# see <https://cds.climate.copernicus.eu/how-to-api>.

# ## Domain: 2° × 2° Central Borneo box at ~1 km resolution

arch = CPU()

latitude  = (0.5, 2.5)
longitude = (113.0, 115.0)

land_grid = LatitudeLongitudeGrid(arch; latitude, longitude,
                                  size = (200, 200),
                                  topology = (Bounded, Bounded, Flat))

# ## ETOPO surface elevation
#
# `regrid_topography` regrids ETOPO 2022 onto the land grid as a non-negative land
# surface elevation. The atmosphere is corrected to this elevation.

surface_elevation = regrid_topography(land_grid; dataset = ETOPO2022())

# ## ERA5 forcing

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

# ## Altitude correction and downscaling
#
# `SlabLand` itself knows nothing about terrain, and ERA5's near-surface fields
# correspond to the mean elevation of ERA5's own ~28 km grid cells. To resolve
# elevation-driven temperature contrasts on the 1 km grid, [`AltitudeCorrection`](@ref)
# moves the regridded atmosphere from that elevation (`atmosphere_elevation`) to the
# 1 km ETOPO surface (`surface_elevation`), across the elevation difference
#
#     Δz(λ, φ) = z_ETOPO(λ, φ) − z_ERA5(λ, φ).
#
# The correction shifts the temperature along the environmental lapse rate,
# `T ← T − Γ Δz` with the lapse rate Γ = 6.5 K km⁻¹, and adjusts the pressure hydrostatically; the
# specific humidity `q` is unchanged. The state exchanger applies it at every step.
# `atmosphere_elevation` is ERA5's own model topography (its surface geopotential
# divided by g); the gravitational acceleration and gas constant needed for the pressure
# adjustment come from the atmosphere's thermodynamics.

atmosphere_elevation = Field(Metadatum(:topography; dataset, date = start_date, region), land_grid)

Δz = surface_elevation - atmosphere_elevation

lapse_rate = 6.5e-3 # K m⁻¹
correction = AltitudeCorrection(surface_elevation, atmosphere_elevation; lapse_rate)

# ## Slab land
#
# We simulate a 1 m soil column (`slab_depth = 1`) that restores toward the deep
# reservoir over 12 hours. Under heavy equatorial rainfall, `InfiltrationCapacityRunoff`
# sheds the rain that arrives faster than the soil can absorb it.

slab_depth = 1

# For simplicity, we derive the soil porosity from the elevation. Terrain is a strong
# predictor of soil texture and organic content, so soil properties vary systematically
# with topographic position. In interior Borneo, the lowlands are rich in peat and organic
# matter (high porosity), while the mineral uplands of the Müller and Schwaner ranges are
# less porous. We therefore let the porosity decrease linearly with elevation. This is an
# illustration rather than a physical law, but it gives the hydrology a spatially varying
# porosity field `ν(λ, φ)`.

upland_porosity  = 0.35   # mineral upland soils
lowland_porosity = 0.55   # peat/organic-rich valley soils

zmin = minimum(surface_elevation)
zmax = maximum(surface_elevation)

elevation_porosity(z) = lowland_porosity + (upland_porosity - lowland_porosity) * (z - zmin) / (zmax - zmin)

porosity = Field(elevation_porosity(surface_elevation))

# The humidity closure takes a representative scalar porosity, which only enters its
# power-law tortuosity. The porosity mostly acts through the hydrology's spatially
# varying field above.
nominal_porosity = 0.4

slab_land = SlabLand(land_grid;
                     energy    = soil_energy(eltype(land_grid); deep_time_scale = 12hours),
                     hydrology = variably_saturated_hydrology(eltype(land_grid), porosity; slab_depth))

# We initialize the skin temperature from the elevation-corrected ERA5 T₂ₘ. The initial
# soil water, `Mˡᵃ = 100 kg m⁻²`, is well below the slab's saturation storage
# `Mˡᵃ⁺ = ρˡ ν hˡᵃ` (350–550 kg m⁻² across the porosity range).

T₀ = Field{Center, Center, Nothing}(land_grid)
interpolate!(T₀, atmosphere.temperature[1])
set!(slab_land; T = T₀ - lapse_rate * Δz, M = 100)

# ## Coupled model
#
# The interface specific humidity uses the same `DryLayerHumidity` closure as the
# 0D column. The roughness lengths belong to the atmosphere–land flux formulation;
# because we pass a custom `atmosphere_land_interface`, we set them in its `fluxes`.

atmosphere_land_fluxes = SimilarityTheoryFluxes(momentum_roughness_length    = 0.1,
                                                temperature_roughness_length = 0.01,
                                                water_vapor_roughness_length = 0.01)

interface_specific_humidity = make_dry_layer_humidity(nominal_porosity)

al_interface = atmosphere_land_interface(slab_land.grid, atmosphere, slab_land;
                                         fluxes = atmosphere_land_fluxes,
                                         specific_humidity = interface_specific_humidity)

model = AtmosphereLandModel(atmosphere, slab_land;
                            radiation,
                            atmosphere_land_interface = al_interface,
                            exchanger_correction = correction)

simulation = Simulation(model; Δt = 5minutes, stop_time = (Nt - 1) * hour)

wall_time = Ref(time_ns())

function progress(sim)
    land = sim.model.land
    Tmin, Tmax = minimum(land.temperature), maximum(land.temperature)
    Mmin, Mmax = minimum(land.water_storage), maximum(land.water_storage)
    𝒮mean      = mean(land.saturation)
    Qmean      = -mean(land.fluxes.surface_energy_flux)  # positive into the slab
    elapsed    = 1e-9 * (time_ns() - wall_time[]); wall_time[] = time_ns()
    @info @sprintf("Iter %d  t = %s  T %.1f–%.1f K  M %.1f–%.1f kg m⁻²  ⟨𝒮⟩ %.2f  ⟨Q⟩ %+6.1f W m⁻²  wall Δ %.1fs",
                   iteration(sim), prettytime(sim), Tmin, Tmax, Mmin, Mmax, 𝒮mean, Qmean, elapsed)
    return nothing
end
add_callback!(simulation, progress, IterationInterval(144))  # every 12 hours

# The variably saturated hydrology reports a signed vapor flux `Jᵛ` (positive upward,
# i.e., evaporation) and a liquid precipitation flux `Pˡ` (positive downward). We save the
# precipitation in mm hr⁻¹, since 1 kg m⁻² of liquid water is 1 mm deep.

outputs = (T = slab_land.temperature,
           M = slab_land.water_storage,
           𝒮 = slab_land.saturation,
           Q = slab_land.fluxes.surface_energy_flux,
           E = slab_land.fluxes.vapor_flux,
           P = slab_land.fluxes.liquid_precipitation_flux * hour)

filename = "era5_forced_slab_land"

simulation.output_writers[:land] = JLD2Writer(model, outputs;
                                              filename,
                                              schedule = TimeInterval(1hour),
                                              overwrite_files = true)

# ## Run

run!(simulation)

close(simulation.output_writers[:land])
delete!(simulation.output_writers, :land)
atmosphere = radiation = simulation = model = nothing
GC.gc(true); GC.gc(true)

# ## Animation

T_ts = FieldTimeSeries("$filename.jld2", "T")
𝒮_ts = FieldTimeSeries("$filename.jld2", "𝒮")
Q_ts = FieldTimeSeries("$filename.jld2", "Q")
P_ts = FieldTimeSeries("$filename.jld2", "P")

times   = T_ts.times
Nframes = length(times)

fig = Figure(size = (1700, 1000), fontsize = 12)

axT = Axis(fig[1, 1]; title = "Skin temperature T (K)",    xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
ax𝒮 = Axis(fig[1, 3]; title = "Surface saturation 𝒮",     xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
axQ = Axis(fig[1, 5]; title = "Net energy flux Q (W m⁻²)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
axP = Axis(fig[1, 7]; title = "Precipitation P (mm hr⁻¹)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
axz = Axis(fig[2, 1]; title = "Elevation (m, ETOPO 2022)", xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
axt = Axis(fig[2, 3:8]; title = "Domain minimum, mean, and maximum of T", xlabel = "t (days)", ylabel = "T (K)")

n  = Observable(1)
Tn = @lift T_ts[$n]
𝒮n = @lift 𝒮_ts[$n]
Qn = @lift Q_ts[$n]
Pn = @lift P_ts[$n]

𝒮min, 𝒮max = extrema(𝒮_ts)
Qmax = maximum(abs, Q_ts)

Tlim = extrema(T_ts)
𝒮lim = (𝒮min, max(𝒮max, 𝒮min + 0.1))
Qlim = (-Qmax, Qmax)
Plim = (0, max(maximum(P_ts), 1e-3))

hmT = heatmap!(axT, Tn; colormap = :turbo,   colorrange = Tlim)
hm𝒮 = heatmap!(ax𝒮, 𝒮n; colormap = :tempo,   colorrange = 𝒮lim)
hmQ = heatmap!(axQ, Qn; colormap = :balance, colorrange = Qlim)
hmP = heatmap!(axP, Pn; colormap = :dense,   colorrange = Plim)
hmz = heatmap!(axz, surface_elevation; colormap = :terrain)

Colorbar(fig[1, 2], hmT; label = "T (K)")
Colorbar(fig[1, 4], hm𝒮; label = "𝒮")
Colorbar(fig[1, 6], hmQ; label = "Q (W m⁻²)")
Colorbar(fig[1, 8], hmP; label = "P (mm hr⁻¹)")
Colorbar(fig[2, 2], hmz; label = "elevation (m)")

mean_temperature    = [mean(T_ts[m])    for m in 1:Nframes]
maximum_temperature = [maximum(T_ts[m]) for m in 1:Nframes]
minimum_temperature = [minimum(T_ts[m]) for m in 1:Nframes]

t = times / day

lines!(axt, t, maximum_temperature; color = :red,   linewidth = 1.5, label = "maximum")
lines!(axt, t, mean_temperature;    color = :black, linewidth = 1.5, label = "mean")
lines!(axt, t, minimum_temperature; color = :blue,  linewidth = 1.5, label = "minimum")
axislegend(axt; position = :rb)
vlines!(axt, @lift([t[$n]]); color = :black, linewidth = 1.0, linestyle = :dash)

title = @lift "ERA5-forced slab land over Central Borneo at ~1 km, t = " * prettytime(times[$n])
Label(fig[0, 1:8], title, fontsize = 16)

trim!(fig.layout)

CairoMakie.record(fig, "$filename.mp4", 1:Nframes; framerate = 12) do nn
    n[] = nn
end

nothing #hide

# ![](era5_forced_slab_land.mp4)

# ## Differentiating the ERA5 run: a pointwise porosity sensitivity map
#
# The forward run above depends on the land parameters. We now ask a calibration-style
# question: **how sensitive is the skin temperature to the soil porosity, cell by cell?**
# The forward run uses the elevation-derived porosity field `ν(λ, φ)`. We differentiate
# the coupled run with respect to every cell of that field at once, so that a single
# reverse pass returns the whole map ``∂T(λ, φ)/∂ν(λ, φ)`` of the local skin
# temperature's response to the local porosity.
#
# Differentiating over the full 200 × 200 grid is expensive, so for demonstration we
# restrict the computation to a 20 × 20 sub-patch in the interior of the same Central
# Borneo domain (≈ 1 km cells, land only). As in the 0D case, the Monin–Obukhov solver
# runs a fixed number of iterations (`FixedIterations`) so that the compiled time step
# stays differentiable.
#
# Reading the map pointwise relies on the columns being independent. The hydrology has
# no lateral coupling: `InfiltrationCapacityRunoff` is a per-cell sink whose cap depends
# only on precipitation, not on ν, and `NoDeepLiquidFlux` closes the bottom. With the
# scalar objective `L = Σ T(λ, φ)`, the cross terms therefore vanish and
# `dL/dν(λ, φ) = ∂T(λ, φ)/∂ν(λ, φ)` is exactly the pointwise sensitivity. A per-cell
# finite difference checks both the adjoint and this column independence.

patch_latitude  = (1.4, 1.6)
patch_longitude = (113.9, 114.1)
patch_size      = (20, 20)

make_land_grid(arch) = LatitudeLongitudeGrid(arch; latitude = patch_latitude, longitude = patch_longitude,
                                             size = patch_size,
                                             topology = (Bounded, Bounded, Flat))

# ## ERA5 forcing window
#
# Reverse-mode gradient checkpointing needs a perfect-square step count; `38² = 1444`
# steps of 5 minutes span 5.01 days, so the ERA5 window extends comfortably beyond the
# run. `patch_region` extends a few native ERA5 cells beyond the sub-patch.

Δt       = 5minutes
Nsteps   = 38^2
run_time = Nsteps * Δt

patch_start_date = DateTime(2020, 4, 1)
patch_end_date   = DateTime(2020, 4, 6, 12)
patch_region     = BoundingBox(latitude = (1.0, 2.0), longitude = (113.5, 114.5))

surface_layer_height  = 10
boundary_layer_height = 800
land_surface          = SurfaceRadiationProperties(0.18, 0.95)

initial_water_storage = 100.0 # kg m⁻²

# ## Moving the ERA5 forcing into memory
#
# An ERA5 `FieldTimeSeries` with a `DatasetBackend` reads files from the host and cannot
# be stepped inside a compiled XLA loop. We therefore load the ERA5 data for the whole run
# once, on the CPU, and interpolate every hourly slice onto an in-memory
# `PrescribedAtmosphere` and `PrescribedRadiation` on the patch grid.
#
# `map_forcing_slices!` applies `op!(destination, source)` to every pair of matching
# forcing slices.

function map_forcing_slices!(op!, times, dst_atmos, dst_rad, src_atmos, src_rad)
    for n in eachindex(times)
        op!(dst_atmos.velocities.u[n],            src_atmos.velocities.u[n])
        op!(dst_atmos.velocities.v[n],            src_atmos.velocities.v[n])
        op!(dst_atmos.temperature[n],             src_atmos.temperature[n])
        op!(dst_atmos.specific_humidity[n],       src_atmos.specific_humidity[n])
        op!(dst_atmos.pressure[n],                src_atmos.pressure[n])
        op!(dst_atmos.precipitation_flux.rain[n], src_atmos.precipitation_flux.rain[n])
        op!(dst_rad.downwelling_shortwave[n],     src_rad.downwelling_shortwave[n])
        op!(dst_rad.downwelling_longwave[n],      src_rad.downwelling_longwave[n])
    end
    return nothing
end

function inmemory_forcing(land_grid)
    era5_atmos = ERA5PrescribedAtmosphere(CPU(); dataset, start_date = patch_start_date, end_date = patch_end_date,
                                          region = patch_region, surface_layer_height, boundary_layer_height)
    era5_rad   = ERA5PrescribedRadiation(CPU(); dataset, start_date = patch_start_date, end_date = patch_end_date,
                                         region = patch_region, land_surface)
    times = era5_atmos.velocities.u.times

    atmos = PrescribedAtmosphere(land_grid, times; surface_layer_height, boundary_layer_height)
    rad   = PrescribedRadiation(land_grid, times; land_surface,
                                ocean_surface = nothing, sea_ice_surface = nothing)

    ## Regridding every slice up front leaves only time interpolation for the compiled loop.
    map_forcing_slices!(interpolate!, times, atmos, rad, era5_atmos, era5_rad)
    update_state!(atmos); update_state!(rad)
    return (; atmos, rad, times)
end

# We then copy the CPU forcing, buffer by buffer, onto the `ReactantState` grid. Copying
# the already-regridded slices, rather than regridding again, gives the compiled run
# exactly the forcing of the CPU forward run, so the adjoint and finite-difference
# derivatives are comparable.

function transfer_forcing(grid, cpu)
    times = cpu.times
    atmos = PrescribedAtmosphere(grid, times; surface_layer_height, boundary_layer_height)
    rad   = PrescribedRadiation(grid, times; land_surface,
                                ocean_surface = nothing, sea_ice_surface = nothing)
    copy_slice!(dst, src) = (parent(dst) .= Array(parent(src)); nothing)
    map_forcing_slices!(copy_slice!, times, atmos, rad, cpu.atmos, cpu.rad)
    update_state!(atmos); update_state!(rad)
    return (; atmos, rad, times)
end

# ## Model builder
#
# The porosity is a `(Center, Center, Nothing)` field set to `value`.

function porosity_field_on(grid, value)
    ν = Field{Center, Center, Nothing}(grid)
    set!(ν, value)
    return ν
end

function era5_slab_land_model(grid, forcing, porosity_field, porosity_scalar; exchanger_correction = nothing)
    energy    = soil_energy(eltype(grid); deep_time_scale = 12hours)
    hydrology = variably_saturated_hydrology(eltype(grid), porosity_field; slab_depth)
    return coupled_slab_land_model(grid, forcing.atmos, forcing.rad;
                                   energy, hydrology, humidity_porosity = porosity_scalar,
                                   exchanger_correction)
end

# The initial skin temperature is the ERA5 T₂ₘ at the first snapshot.

cold_start_temperature(grid, forcing) =
    (T₀ = Field{Center, Center, Nothing}(grid);
     interpolate!(T₀, forcing.atmos.temperature[1]); T₀)

# ## Forward run
#
# We now run the model on the patch under the ERA5 forcing, with the porosity given as a
# `Field`. The same routine drives both the nominal run and the finite-difference
# perturbations.

cpu_grid                = make_land_grid(CPU())
cpu_forcing             = inmemory_forcing(cpu_grid)
cpu_initial_temperature = cold_start_temperature(cpu_grid, cpu_forcing)

patch_surface_elevation = regrid_topography(cpu_grid; dataset = ETOPO2022())
cpu_porosity            = Field(elevation_porosity(patch_surface_elevation))

# The ETOPO surface and ERA5's own topography are read on the CPU and passed to
# `AltitudeCorrection` as plain arrays. The correction then needs no host I/O and runs
# on both the CPU and the `ReactantState` grid.

patch_atmosphere_elevation = Field(Metadatum(:topography; dataset, date = patch_start_date, region = patch_region), cpu_grid)

patch_correction = AltitudeCorrection(Array(interior(patch_surface_elevation, :, :, 1)),
                                      Array(interior(patch_atmosphere_elevation, :, :, 1));
                                      lapse_rate)

function run_forward(grid, forcing, T₀, porosity_field)
    model = era5_slab_land_model(grid, forcing, porosity_field, nominal_porosity;
                                 exchanger_correction = patch_correction)
    set!(model.land; T = T₀, M = initial_water_storage)

    for _ in 1:Nsteps
        time_step!(model, Δt)
    end

    return model.land.temperature
end

final_temperature = run_forward(cpu_grid, cpu_forcing, cpu_initial_temperature, cpu_porosity)

@info @sprintf("Forward:  ⟨T(t=%.2f d)⟩ = %.4f K", run_time / day, mean(final_temperature))

# ## A differentiable workflow with Reactant and Enzyme
#
# We rebuild the patch on a `ReactantState` grid, so that every array is an XLA buffer,
# and differentiate with respect to the porosity field.

reactant_grid    = make_land_grid(ReactantState())
reactant_forcing = transfer_forcing(reactant_grid, cpu_forcing)
reactant_model   = era5_slab_land_model(reactant_grid, reactant_forcing,
                                        porosity_field_on(reactant_grid, nominal_porosity), nominal_porosity;
                                        exchanger_correction = patch_correction)

# The state exchanger's regridder is populated before the compiled run.

Oceananigans.initialize!(reactant_model)

dmodel                       = Enzyme.make_zero(reactant_model)
reactant_initial_temperature = cold_start_temperature(reactant_grid, reactant_forcing)

# The differentiated input is the porosity field `ν`. Enzyme accumulates `∂L/∂ν(λ, φ)`
# into its shadow `dν`, a field of the same shape: the sensitivity map.

ν = porosity_field_on(reactant_grid, nominal_porosity)
parent(ν) .= Array(parent(cpu_porosity))
dν = Enzyme.make_zero(ν)

# ## The objective
#
# `final_skin_temperature_era5` copies the porosity field into the model, resets the
# initial state, runs `nsteps` coupled time steps inside a `@trace` loop, and returns the
# sum of the final skin temperature over the patch. Because the columns are independent,
# the gradient of this sum is the pointwise sensitivity and compares directly with the
# finite-difference map.

function final_skin_temperature_era5(model, ν, T₀, M₀, Δt, nsteps)
    parent(model.land.hydrology.porosity) .= parent(ν)

    set!(model.land.water_storage, M₀)
    νp = parent(ν)
    θˡ = min.(M₀ / (liquid_density * slab_depth), νp)
    parent(model.land.saturation) .= clamp.((θˡ .- residual_liquid_fraction) ./
                                            (νp .- residual_liquid_fraction), 0, 1)
    parent(model.land.temperature) .= parent(T₀)

    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end

    return sum(interior(model.land.temperature))
end

# ## The gradient wrapper
#
# As before, the model and the porosity *field* are `Duplicated` (primal and shadow),
# while the initial state, the time step, and the loop length are `Const`.

function grad_final_skin_temperature_era5(model, dmodel, ν, dν, T₀, M₀, Δt, nsteps)
    parent(dν) .= 0
    _, L = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        final_skin_temperature_era5, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(ν, dν),
        Enzyme.Const(T₀),
        Enzyme.Const(M₀),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps))
    return dν, L
end

# ## Compilation and execution

@info "Compiling differentiated ERA5 slab-land model — this may take a few minutes..."
compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_final_skin_temperature_era5(
    reactant_model, dmodel, ν, dν, reactant_initial_temperature, initial_water_storage, Δt, Nsteps)

dν, L = compiled_grad(reactant_model, dmodel, ν, dν, reactant_initial_temperature, initial_water_storage, Δt, Nsteps);

adjoint_map = Field{Center, Center, Nothing}(cpu_grid)
set!(adjoint_map, Array(interior(dν, :, :, 1)))

@info @sprintf("Adjoint:  ⟨T(t=%.2f d)⟩ = %.4f K,  ⟨∂T/∂ν⟩ = %+.4e K",
               run_time / day, Reactant.to_number(L) / prod(patch_size), mean(adjoint_map))

# ## Per-cell finite-difference check
#
# A single reverse pass gave the whole map. We check it against a per-cell centered
# finite difference: we perturb the porosity of *every* cell by `±δν` and read off the
# local skin-temperature response. Under column independence, the uniform perturbation
# recovers each diagonal entry `∂T(λ, φ)/∂ν(λ, φ)`. The adjoint and finite-difference
# maps agree in magnitude and sign.

δν = 0.001 * nominal_porosity

perturbed_temperature_plus  = run_forward(cpu_grid, cpu_forcing, cpu_initial_temperature, Field(cpu_porosity + δν))
perturbed_temperature_minus = run_forward(cpu_grid, cpu_forcing, cpu_initial_temperature, Field(cpu_porosity - δν))

finite_difference_map = Field((perturbed_temperature_plus - perturbed_temperature_minus) / 2δν)
maximum_absolute_error = maximum(abs, adjoint_map - finite_difference_map)

@info @sprintf("Finite difference:  ⟨∂T/∂ν⟩ = %+.4e K", mean(finite_difference_map))

# ## Visualization

fig = Figure(size = (1700, 950), fontsize = 16)

axz = Axis(fig[1, 1]; title = "Elevation (m, ETOPO 2022)",
           xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hmz = heatmap!(axz, patch_surface_elevation; colormap = :terrain)
Colorbar(fig[1, 2], hmz; label = "elevation (m)")

axT = Axis(fig[2, 1]; title = "Final skin temperature T (K)",
           xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hmT = heatmap!(axT, final_temperature; colormap = :thermal)
Colorbar(fig[2, 2], hmT; label = "T (K)")

axν = Axis(fig[1, 3]; title = "Porosity ν (elevation-derived)",
           xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hmν = heatmap!(axν, cpu_porosity; colormap = :viridis)
Colorbar(fig[1, 4], hmν; label = "ν")

sensitivity_limit = max(maximum(abs, adjoint_map), maximum(abs, finite_difference_map))

axA = Axis(fig[2, 3]; title = "Adjoint ∂T/∂ν (K) from one reverse pass",
           xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hmA = heatmap!(axA, adjoint_map; colormap = :balance, colorrange = (-sensitivity_limit, sensitivity_limit))
Colorbar(fig[2, 4], hmA; label = "∂T/∂ν (K)")

axF = Axis(fig[2, 5]; title = "Finite-difference ∂T/∂ν (K)",
           xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hmF = heatmap!(axF, finite_difference_map; colormap = :balance, colorrange = (-sensitivity_limit, sensitivity_limit))
Colorbar(fig[2, 6], hmF; label = "∂T/∂ν (K)")

Label(fig[0, 1:6], @sprintf("Differentiable ERA5 slab land: pointwise ∂T/∂ν map (max |adjoint − finite difference| = %.2e K)",
                            maximum_absolute_error))

save("era5_forced_slab_land_porosity_map.png", fig)
nothing #hide

# ![](era5_forced_slab_land_porosity_map.png)
