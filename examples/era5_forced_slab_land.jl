# # A differentiable slab-land model: 0D column and 2D ERA5-forced map
#
# This page builds a differentiable land model in two parts, starting from an idealized 0D
# example, then demonstrating a more realistic case with ERA5 forcing on realistic topography.
# We use a `SlabLand` composed of
#
#     energy    = WaterCoupledEnergy(...)          # C(Mˡᵃ) = C_dry + cˡ Mˡᵃ, conservative dTˡᵃ/dt
#     hydrology = VariablySaturatedHydrology(...)  # augmented-storage ϑˡ budget
#     humidity  = DryLayerHumidity(...)            # qⁱⁿ from a dry-layer vapor-flux balance
#
# This page is split into four stages:
# 1. A 0D dry-layer slab under idealized analytic forcing;
# 2. its skin temperature's sensitivity to the rain rate``∂T / ∂𝒫̇`` from a single reverse pass;
# 3. an ERA5-forced example in Central Borneo example run at ~1 km with downscaled elevation;
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
# `wᵈ = Dᵛ_eff / δᵛ`, so a saturated surface evaporates at the atmospheric-demand
# limit while a dried-out one is throttled. This two-stage bare-soil drying is
# explained in detail in [Or et al. (2013)](@cite or2013advances). 
# In this example, we use a thin slab to demonstrate the different stages over
# a short roll out for demonstration purposes. A rain pulse on day 2 briefly over-saturates the slab
# (`𝒮 = 1`) such that the latent heat flux increases to the wet/saturated regime.
# Once evaporation draws the slab back down through the dry layer onser`𝒮ᶜ`, the dry layer 
# reopens and the latent heat collapses again.

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
# The idealized forcings simulate a warm, dry 280–290 K regime that keeps evaporative
# demand high. A six-hour rain pulse falls on day 2 with amplitude `𝒫̇` (nominally
# `6e-4 kg m⁻² s⁻¹`, ≈ 13 mm over the six hours)

const day = 1days

air_temperature(t)       = 285 - 5 * cos(2π * t / day)               # K, 280–290 diurnal
downwelling_shortwave(t) = max(0, 600 * cos(2π * (t - day/2) / day)) # W m⁻², daytime only
downwelling_longwave(t)  = 320                                       # W m⁻², constant
wind_speed               = 4.0                                       # m s⁻¹
specific_humidity         = 0.004                                     # kg kg⁻¹ (dry air drives evaporation)
surface_pressure         = 101325                                    # Pa

nominal_rain_rate = 6e-4                                             # kg m⁻² s⁻¹

in_rain_pulse(t) = 2days ≤ t ≤ 2days + 6hours
rain_pulse(t, 𝒫̇) = ifelse(in_rain_pulse(t), 𝒫̇, zero(𝒫̇))

# ## Forcing and model builders
#
# `forced_atmosphere` fills each `FieldTimeSeries` time slice from the analytic functions.

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
# We build a thin (5 cm) loamy slab, so the daytime evaporative demand dries it back
# across `𝒮ᶜ` within a few days of the pulse, transitioning between 
# saturated and arrested evaporation phases. We start *below* `𝒮ᶜ`, so a dry surface layer is
# already present (`δᵛ > 0`) and evaporation is initially throttled. The pulse amplitude
# stays below the soil infiltration capacity, so the `InfiltrationCapacityRunoff` model does not
# shed water. With `NoDeepLiquidFlux` closing the bottom, evaporation is the only water sink. 
# The pulse drives storage briefly past the saturation value `Mˡᵃ⁺ = ρˡ ν hˡᵃ`. Using the 
# augmented liquid fraction as the tracer, it can go above saturation, carrying the excess 
# as a positive pressure head (`𝒮` pinned at 1).

porosity              = 0.4
slab_depth            = 0.05
maximum_water_storage = liquid_density * porosity * slab_depth   # Mˡᵃ⁺ = ρˡ ν hˡᵃ = 20 kg m⁻²

initial_saturation  = 0.4   # below 𝒮ᶜ — a dry layer is already present
initial_temperature = 280.0 # K

# Diagnostic surface saturation `𝒮(Mˡᵃ)` for the variably-saturated slab:
# `θˡ = min(Mˡᵃ/(ρˡ hˡᵃ), ν)` and `𝒮 = clamp((θˡ − θʳ)/(ν − θʳ), 0, 1)`.

function saturation_from_storage(M)
    θˡ = min(M / (liquid_density * slab_depth), porosity)
    return clamp((θˡ - residual_liquid_fraction) / (porosity - residual_liquid_fraction), 0, 1)
end

initial_water_storage =
    (initial_saturation * (porosity - residual_liquid_fraction) + residual_liquid_fraction) *
    liquid_density * slab_depth

# ## Model builders

# `WaterCoupledEnergy` is the slab's energy budget which steps the skin temperature `Tˡᵃ`
#  with a force-restore balance toward a deep reservoir at `deep_temperature` on the 
# `deep_time_scale`, and folds the water storage into the areal heat capacity 
# `C(Mˡᵃ) = C_dry + cˡ Mˡᵃ`, recomputed each step so that a wetter slab carries more 
# thermal inertia. `dry_heat_capacity` is `C_dry` (depth × density × specific heat of 
# the dry soil), `liquid_heat_capacity` is `cˡ`.
soil_energy(FT; deep_time_scale) = WaterCoupledEnergy(FT;
    dry_heat_capacity     = 0.1 * 1500 * 1480,
    liquid_heat_capacity  = 4186,
    reference_temperature = 273.15,
    deep_temperature      = 280,
    deep_time_scale)

# `VariablySaturatedHydrology` is the soil-water budget: it steps the column water
# storage `Mˡᵃ` from its surface flux (precipitation minus evaporation, throttled by
# the runoff scheme) and its bottom flux, and diagnoses the surface saturation
# `𝒮(Mˡᵃ)` through the van Genuchten retention/conductivity curves.
variably_saturated_hydrology(FT, porosity; slab_depth) = VariablySaturatedHydrology(FT;
    slab_depth, porosity, residual_liquid_fraction,
    storage_height         = 1000,
    retention_curve        = VanGenuchtenRetention(α = 1.0, n = 2.0),
    hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-7, n = 2.0),
    deep_liquid_flux       = NoDeepLiquidFlux(),
    runoff                 = InfiltrationCapacityRunoff(infiltration_capacity = 1e-3))

# The dry-layer humidity closure: the front depth grows from 0 (saturated skin)
# toward `δᵛ_max` as the slab dries past `𝒮ᶜ`, and `qⁱⁿ` is solved from the
# vapor-flux balance through it.
make_dry_layer_humidity(porosity) = DryLayerHumidity(;
    dry_layer_depth = StorageBasedDryLayerDepth(; maximum_dry_layer_depth = 0.05,
                                                dry_layer_onset_saturation,
                                                dry_layer_exponent      = 2),
    vapor_exchange = DryLayerVaporPistonVelocity(minimum_dry_layer_depth = 1e-4,
                                                 molecular_diffusivity   = 2.5e-5,
                                                 tortuosity_model        = MillingtonQuirk()),
    thermal_exchange_depth = 0.10,
    porosity)

# Now, we assemble the coupled model used by the *differentiable* stages.
# For differentiability purposes, the turbulent fluxes use a fixed-iteration Monin–Obukhov solver 
# (`FixedIterations`) instead of the default tolerance-based `while` loop. 
# This is done because the differentiated pass compiles the coupled
# time step to XLA, where a data-dependent `while` becomes an XLA op that does not
# differentiate cheaply. On the other hand, a fixed iteration count unrolls into a static
# graph Enzyme can traverse. `Clock(grid)` is a plain `Float64` clock on the CPU
# but a Reactant `ConcreteRNumber` clock on a `ReactantState` grid — the latter
# advances model time *inside* the compiled XLA loop, so the time-dependent forcing
# is sampled correctly each step.
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

# The 0D dry-layer slab uses a weak deep restoring (`τᵈᵉᵉᵖ = 10 days`) so the thin
# slab is free to dry down over the run.
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
# We integrate the model for 5.84 days (`41² = 1681` steps). The final time 
# skin temperature is differentiated against the rain pulse amplitude `𝒫̇`
# in order to compute the sensitivity of skin temperature against rain rate.
# Reverse gradient checkpointing needs a perfect-square step count, hence we chose `41²`.

Δt     = 5minutes
Nsteps = 41^2                              # 1681 steps ≈ 5.84 days
times  = range(0, Nsteps * Δt, step = 1hour)

# `pulse_indices` are the `times` slices inside the rain pulse.

pulse_indices = [n for n in eachindex(times) if in_rain_pulse(times[n])]

# ## Forward run
#
# Now we are ready to run the 0D model.

scalar(field) = first(interior(field))

grid = RectilinearGrid(CPU(); size = (), topology = (Flat, Flat, Flat))

forward_model = dry_layer_model(grid, times, nominal_rain_rate)
set!(forward_model.land; T = initial_temperature, M = initial_water_storage)

t  = zeros(Nsteps)
T  = zeros(Nsteps)
M  = zeros(Nsteps)
𝒮  = zeros(Nsteps)
Rₙ = zeros(Nsteps)
H  = zeros(Nsteps)
LE = zeros(Nsteps)
G  = zeros(Nsteps)

@info "Running forward dry-layer slab..."
for n in 1:Nsteps
    time_step!(forward_model, Δt)
    land      = forward_model.land
    interface = forward_model.interfaces.atmosphere_land_interface.fluxes
    rad       = forward_model.radiation.interface_fluxes.land

    t[n]  = forward_model.clock.time / day
    T[n]  = scalar(land.temperature)
    M[n]  = scalar(land.water_storage)
    𝒮[n]  = scalar(land.saturation)
    H[n]  = scalar(interface.sensible_heat)   # positive upward
    LE[n] = scalar(interface.latent_heat)     # positive upward
    Rₙ[n] = scalar(rad.downwelling_shortwave) + scalar(rad.downwelling_longwave) -
            scalar(rad.upwelling_longwave)    # net radiation into the surface
    G[n]  = -scalar(land.fluxes.surface_energy_flux)  # storage residual Rₙ − H − LE
end

# # Differentiating skin temperature against rain rate in the idealized model
#
# To differentiate through the model, we rebuild the same setup on
# a `ReactantState` grid so every array is an XLA buffer, wrap the
# coupled time-stepping in a scalar objective `final_skin_temperature`, compile it
# once, and differentiate it with respect to the rain-pulse amplitude.

Reactant.set_default_backend("cpu")

# TODO: remove this line once Reactant has been bumped
Reactant.LLVM.clopts("-vectorize-slp=false", "-vectorize-loops=false")

grid_ad  = RectilinearGrid(ReactantState(); size = (), topology = (Flat, Flat, Flat))
model_ad = dry_layer_model(grid_ad, times, nominal_rain_rate)
dmodel   = Enzyme.make_zero(model_ad)

# The differentiated input is the rain-pulse amplitude `𝒫̇`, carried as a
# one-element XLA buffer so Enzyme can accumulate ``∂T / ∂𝒫̇`` into its shadow
# `d𝒫̇`.

𝒫̇  = Reactant.to_rarray([nominal_rain_rate])
d𝒫̇ = Enzyme.make_zero(𝒫̇)

# ## The objective
#
# `final_skin_temperature` re-initializes the slab from the initial state, sets the
# rain pulse from the current `𝒫̇`, runs `nsteps` of the coupled time step inside a
# `@trace` loop, and returns the final skin temperature. The `set!`s inside it make
# the gradient track the live trajectory.

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
# Reverse mode with the model and rain amplitude as `Duplicated` (primal +
# shadow); the loop length and scalar parameters are `Const`.

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
    model_ad, dmodel, 𝒫̇, d𝒫̇, pulse_indices,
    initial_temperature, initial_water_storage, Δt, Nsteps)

@info "Running gradient..."
d𝒫̇, T_readout = compiled_grad(model_ad, dmodel, 𝒫̇, d𝒫̇, pulse_indices,
                              initial_temperature, initial_water_storage, Δt, Nsteps)

dT_drain = Array(d𝒫̇)[1]   # K / (kg m⁻² s⁻¹)

@info @sprintf("Adjoint:  T(t=%.2f d) = %.4f K,  ∂T/∂𝒫̇ = %.4e K / (kg m⁻² s⁻¹)",
               Nsteps * Δt / day, Reactant.to_number(T_readout), dT_drain)

# ## Finite-difference check
#
# To check that our reverse-mode AD gives the right answer, we run a finite-difference 
# approximation of the same derivative.

function readout_skin_temperature(𝒫̇)
    model = dry_layer_model(grid, times, 𝒫̇)
    set!(model.land; T = initial_temperature, M = initial_water_storage)
    for _ in 1:Nsteps
        time_step!(model, Δt)
    end
    return scalar(model.land.temperature)
end

δ = 0.001 * nominal_rain_rate
dT_drain_fd = (readout_skin_temperature(nominal_rain_rate + δ) -
               readout_skin_temperature(nominal_rain_rate - δ)) / 2δ

@info @sprintf("Finite difference:  ∂T/∂𝒫̇ ≈ %.4e K / (kg m⁻² s⁻¹)", dT_drain_fd)

# ## Visualization

fig = Figure(size = (1600, 800), fontsize = 20)

pulse_span  = (2, 2 + 6/24)
readout_day = Nsteps * Δt / day
Tₐ          = air_temperature.(t .* day)   # prescribed diurnal forcing at the sample times

pulse_duration = 6hours
dT_per_mm      = dT_drain / pulse_duration   # K per mm of pulse rain (W = 𝒫̇ Δt ⟹ ÷Δt)

## Column 1 — the surface energy balance over the air-temperature forcing.
ax_flux = Axis(fig[1, 1]; title = "Surface energy balance\n(positive: atmosphere → surface)",
               xlabel = "t (days)", ylabel = "flux (W m⁻²)")
vspan!(ax_flux, pulse_span...; color = (:skyblue, 0.25))
lines!(ax_flux, t, Rₙ; color = :black,    label = "Net radiation")
lines!(ax_flux, t, G;  color = :seagreen, linestyle = :dash, label = "Energy into land")
axislegend(ax_flux; position = :lt)

ax_Ta = Axis(fig[2, 1]; title = "Prescribed air temperature", xlabel = "t (days)", ylabel = "Tₐ (K)")
vspan!(ax_Ta, pulse_span...; color = (:skyblue, 0.25))
lines!(ax_Ta, t, Tₐ; color = :purple)

## Column 2 — the dry-layer onset: surface saturation over the turbulent fluxes
## it governs (as 𝒮 falls through 𝒮ᶜ the latent flux collapses, sensible takes over).
ax_S = Axis(fig[1, 2]; title = "Surface saturation", xlabel = "t (days)", ylabel = "𝒮")
vspan!(ax_S, pulse_span...; color = (:skyblue, 0.25))
lines!(ax_S, t, 𝒮; color = :darkorange, label = "𝒮")
hlines!(ax_S, [dry_layer_onset_saturation]; color = :gray, linestyle = :dash, label = "Dry layer onset")
ylims!(ax_S, 0, 1.05)
axislegend(ax_S; position = :rb)

ax_LE = Axis(fig[2, 2]; title = "Turbulent heat fluxes\n(positive: surface → atmosphere)",
             xlabel = "t (days)", ylabel = "flux (W m⁻²)")
vspan!(ax_LE, pulse_span...; color = (:skyblue, 0.25))
lines!(ax_LE, t, LE; color = :navy,   label = "Latent")
lines!(ax_LE, t, H;  color = :orange, label = "Sensible")
axislegend(ax_LE; position = :lt)

## Column 3 — the slab water storage over the skin-temperature response (whose
## dried-out end value is the differentiated quantity).
ax_M = Axis(fig[1, 3]; title = "Water storage", xlabel = "t (days)", ylabel = "Mˡᵃ (kg m⁻²)")
vspan!(ax_M, pulse_span...; color = (:skyblue, 0.25))
lines!(ax_M, t, M; color = :navy, label = "Mˡᵃ")
hlines!(ax_M, [maximum_water_storage]; color = :gray, linestyle = :dash, label = "Saturation")
axislegend(ax_M; position = :rt)

ax_T = Axis(fig[2, 3];
            title  = @sprintf("Skin temperature, ∂T/∂(rain) ≈ %.3f K mm⁻¹", dT_per_mm),
            xlabel = "t (days)", ylabel = "T (K)")
vspan!(ax_T, pulse_span...; color = (:skyblue, 0.25))
lines!(ax_T, t, T; color = :firebrick)
vlines!(ax_T, [readout_day]; color = :black, linestyle = :dash, label="Sensitivity readout time")
axislegend(ax_T; position = :lt)

Label(fig[0, 1:3], "Differentiable dry-layer slab — rain at day 2, evaporation, and drydown")

save("differentiable_dry_layer_slab_land.png", fig)
@info "Saved differentiable_dry_layer_slab_land.png"

nothing #hide

# ![](differentiable_dry_layer_slab_land.png)

# # Scaling up: ERA5-forced slab land over Central Borneo
#
# We now run the same `SlabLand` stack as a high-resolution, land-only simulation over 
# the Central Borneo highlands, forced by ERA5 reanalysis and realistic elevation with
# ETOPO 2022 topography. The region is equatorial (snow-free), heavy-rainfall, fully inland,
# and carries ~2 km of relief.
#
# The domain is a 2° × 2° box (0.5–2.5 °N × 113–115 °E, ≈ 220 × 220 km) at
# 200 × 200 = 40 000 cells (≈ 1 km resolution).
#
# The land is coupled through `AtmosphereLandModel` to an
# [`ERA5PrescribedAtmosphere`](@ref) and [`ERA5PrescribedRadiation`](@ref): these
# download the required ERA5 single-level fields (T₂ₘ, dewpoint, 10 m wind,
# surface pressure, total precipitation, downwelling shortwave/longwave) over the domain,
# derive specific humidity from the dewpoint, and convert ERA5's accumulated
# radiation/precipitation to fluxes. Since the raw datasets live on the ERA5 native grid, the
# coupled model interpolates them onto the 1 km land exchange grid.
#
# ## CDS API credentials
#
# Downloading ERA5 fields requires CDS API credentials at `~/.cdsapirc`;
# see <https://cds.climate.copernicus.eu/how-to-api>.

# ## Domain: 2° × 2° Central Borneo box at ~1 km resolution

arch = CPU()

latitude  = lat_min, lat_max = 0.5, 2.5
longitude = lon_min, lon_max = 113.0, 115.0

land_grid = LatitudeLongitudeGrid(arch; latitude, longitude,
                                  size = (200, 200),
                                  topology = (Bounded, Bounded, Flat))

# ## ETOPO surface elevation
#
# `regrid_topography` regrids ETOPO 2022 onto the land grid as a positive land
# surface elevation. This is the elevation the atmosphere is corrected to.

z_land = regrid_topography(land_grid; dataset = ETOPO2022())

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

# ## Elevation correction and downscaling
#
# `SlabLand` itself has no terrain knowledge, and ERA5's near-surface fields
# correspond to ERA5's own ~28 km grid-cell mean elevation. To make the 1 km grid
# show elevation-driven temperature contrasts, [`ElevationCorrection`](@ref) lifts
# the regridded atmosphere from that elevation (`z_era5`) to the 1 km ETOPO surface
# (`z_land`) over the elevation difference
#
#     Δz(λ, φ) = z_ETOPO(λ, φ) − z_ERA5(λ, φ)
#
# with a moist-environmental lapse-rate shift `T ← T − Γ Δz` (Γ = 6.5 K km⁻¹) and a
# hydrostatic pressure adjustment, applied by the state exchanger every step
# (specific humidity `q` conserved through the lift). `z_era5` is ERA5's own model
# topography (its surface geopotential ÷ g → metres); the gravitational
# acceleration and gas constant the pressure adjustment needs are pulled from the
# atmosphere's thermodynamics.

z_meta = Metadatum(:topography; dataset, date = start_date, region)
z_era5 = Field(z_meta, land_grid)

Δz = z_land - z_era5

Γ_lapse    = 6.5e-3 # K m⁻¹, environmental lapse rate
correction = ElevationCorrection(z_land, z_era5; lapse_rate = Γ_lapse)

# ## Slab land
#
# For this example, we simulate a 1 m soil column (`slab_depth = 1`) with the deep
# reservoir restoring on a 12 h time scale. Under heavy equatorial rainfall,
# the `InfiltrationCapacityRunoff` sheds rain that arrives faster than the
# soil infiltration capacity rather than letting it all enter storage.

slab_depth = 1

# For simplicity, the soil porosity is altitude-dependent. Since terrain is a strong
# predictor of soil texture and organic content, soil properties vary
# systematically with topographic position. In interior Borneo the lowlands are
# peat- and organic-rich (high porosity) while the mineral uplands of the
# Müller/Schwaner ranges are less porous, so porosity decreases with the ETOPO
# elevation. This is illustrative rather than a physical law, but it makes the
# hydrology read a spatially-varying porosity field `ν(λ, φ)`.

upland_porosity  = 0.35   # mineral upland soils
lowland_porosity = 0.55   # peat/organic-rich valley soils

zmin, zmax = extrema(interior(z_land, :, :, 1))
elevation_porosity(z) = clamp(lowland_porosity +
                              (upland_porosity - lowland_porosity) * (z - zmin) / (zmax - zmin),
                              residual_liquid_fraction, 1)

porosity = Field{Center, Center, Nothing}(land_grid)
parent(porosity) .= elevation_porosity.(parent(z_land))

# A representative scalar porosity for the humidity closure, which feeds the
# Millington–Quirk tortuosity (a minor pathway). The dominant porosity dependence
# runs through the hydrology's per-cell field above.
nominal_porosity = 0.4

slab_land = SlabLand(land_grid;
                     energy    = soil_energy(eltype(land_grid); deep_time_scale = 12hours),
                     hydrology = variably_saturated_hydrology(eltype(land_grid), porosity; slab_depth))

# Initialize skin temperature from the elevation-corrected ERA5 T₂ₘ.
# Initial soil water `Mˡᵃ = 100 kg m⁻²` sits well below the slab's saturation
# storage `Mˡᵃ⁺ = ρˡ ν hˡᵃ` (350–550 kg m⁻² across the porosity range).

T₀ = Field{Center, Center, Nothing}(land_grid)
interpolate!(T₀, atmosphere.temperature[1])
set!(slab_land; T = T₀ - Γ_lapse * Δz, M = 100)

# ## Coupled model
#
# The atmosphere-facing humidity uses the same `DryLayerHumidity` closure as the
# 0D column. Roughness lengths live with the atmosphere–land flux closure, but
# because we pass a custom `atmosphere_land_interface` they must be baked into
# its `fluxes`.

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
# upward — evaporation) and a liquid precipitation flux `Pˡ` (positive downward).

outputs = (T = slab_land.temperature,
           W = slab_land.water_storage,
           𝒮 = slab_land.saturation,
           Q = slab_land.fluxes.surface_energy_flux,
           E = slab_land.fluxes.vapor_flux,
           P = slab_land.fluxes.liquid_precipitation_flux)

filename = "era5_forced_slab_land"

simulation.output_writers[:land] = JLD2Writer(model, outputs;
                                              filename,
                                              schedule = TimeInterval(1hour),
                                              overwrite_existing = true)

# ## Run

@info "Running ERA5-forced slab land simulation at ~1 km..."
run!(simulation)
@info "Simulation complete."

close(simulation.output_writers[:land])
delete!(simulation.output_writers, :land)
atmosphere = radiation = simulation = model = nothing
GC.gc(true); GC.gc(true)

# ## Animation

T_ts = FieldTimeSeries("$filename.jld2", "T")
𝒮_ts = FieldTimeSeries("$filename.jld2", "𝒮")
Q_ts = FieldTimeSeries("$filename.jld2", "Q")
P_ts = FieldTimeSeries("$filename.jld2", "P")

times      = T_ts.times
Nframes    = length(times)
times_days = collect(times) ./ 86400

λ, φ, _ = nodes(land_grid, Center(), Center(), Center())

fig = Figure(size = (1700, 1000), fontsize = 12)
ax_T = Axis(fig[1, 1]; title = "Skin temperature T (K)",    xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
ax_𝒮 = Axis(fig[1, 3]; title = "Surface saturation 𝒮",     xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
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

𝒮_lo, 𝒮_hi = extrema(𝒮_ts)

Tlim = extrema(T_ts)
𝒮lim = (𝒮_lo, max(𝒮_hi, 𝒮_lo + 0.1))
Qlim = (-maximum(abs.(Q_ts)), maximum(abs.(Q_ts)))
Plim = (0, max(maximum(P_ts) * 3600, 1e-3))
zlim = extrema(interior(z_land, :, :, 1))

hm_T = heatmap!(ax_T, λ, φ, Tn; colormap = :turbo,   colorrange = Tlim)
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

Label(fig[0, 1:8], @lift("ERA5-forced slab land — Central Borneo at ~1 km, t = " * prettytime(times[$n])), fontsize = 16)

trim!(fig.layout)

@info "Rendering animation..."
CairoMakie.record(fig, "$filename.mp4", 1:Nframes; framerate = 12) do nn
    n[] = nn
end
@info "Animation saved."

nothing #hide

# ![](era5_forced_slab_land.mp4)

# # Differentiating the ERA5 run: a pointwise porosity sensitivity map
#
# The forward run above is dependent on the land parameters. We now ask a
# calibration-style question: **how sensitive is the skin temperature to the
# soil porosity, cell by cell?** The forward run contains an
# altitude-dependent porosity field `ν(λ, φ)`. Now, we differentiate the coupled
# run with respect to every cell of it at once, so a single reverse pass returns
# the entire sensitivity map ``∂T(λ, φ)/∂ν(λ, φ)`` of the local skin
# temperature's response to the local porosity.
#
# Differentiating across the full 200 × 200 grid is computationally expensive, so we
# restrict to a 20 × 20 sub-patch in the interior of the same Central Borneo
# domain (≈ 1 km cells, land only) for demonstration purposes. As beforer, the 
# **fixed-iteration** Monin–Obukhov solver (`FixedIterations`) replaces the 
# default tolerance-based `while` loop, for the same XLA reason as the 0D case.
#
# The pointwise reading of the map relies on columns being independent: the
# hydrology has no lateral coupling (`InfiltrationCapacityRunoff` is a per-cell
# sink whose cap depends only on precipitation, not on ν, and `NoDeepLiquidFlux`
# closes the bottom), so with the scalar objective `L = Σ T(λ, φ)` the cross terms
# vanish and `dL/dν(λ, φ) = ∂T(λ, φ)/∂ν(λ, φ)` is exactly the pointwise
# sensitivity. A per-cell finite difference checks both the adjoint and this
# column-independence.

patch_latitude  = patch_lat_min, patch_lat_max = 1.4, 1.6
patch_longitude = patch_lon_min, patch_lon_max = 113.9, 114.1
patch_size      = (20, 20)

make_land_grid(arch) = LatitudeLongitudeGrid(arch; latitude = patch_latitude, longitude = patch_longitude,
                                             size = patch_size,
                                             topology = (Bounded, Bounded, Flat))

# ## ERA5 forcing window
#
# Reverse gradient checkpointing needs a perfect-square step count; `38² = 1444`
# steps at `Δt = 5 min` is 5.01 days, so the ERA5 window runs comfortably past
# it. `patch_region` brackets the sub-patch by a few native cells.

Δt       = 5minutes
Nsteps   = 38^2
run_time = Nsteps * Δt

patch_start_date = DateTime(2020, 4, 1)
patch_end_date   = DateTime(2020, 4, 6, 12)
patch_region     = BoundingBox(latitude = (1.0, 2.0), longitude = (113.5, 114.5))

surface_layer_height  = 10
boundary_layer_height = 800
land_surface          = SurfaceRadiationProperties(0.18, 0.95)

# ## Soil / closure parameters

initial_water_storage = 100.0      # kg m⁻²

# ## Moving ERA5 forcing into memory
#
# A `DatasetBackend` ERA5 `FieldTimeSeries` does host file I/O and cannot be
# time-stepped inside a compiled XLA loop, so we load the ERA5 required for
#  the simulation length once on the CPU and interpolate every hourly slice 
# onto an in-memory `PrescribedAtmosphere` / `PrescribedRadiation` on the patch grid.

# This is done by applying `op!(dst, src)` to every forcing field in turn, pairing a patch-grid
# atmosphere / radiation destination with a matching source set.
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

    ## Interpolate every hourly slice onto the patch grid up front: this drops the
    ## disk-backed `DatasetBackend` and bakes in the ERA5→patch regrid, leaving only
    ## time interpolation between resident slices for the compiled loop to do.
    map_forcing_slices!(interpolate!, times, atmos, rad, era5_atmos, era5_rad)
    update_state!(atmos); update_state!(rad)
    return (; atmos, rad, times)
end

# Then, we move the CPU forcing set onto the `ReactantState` grid buffer-by-buffer.
function transfer_forcing(grid, cpu)
    times = cpu.times
    atmos = PrescribedAtmosphere(grid, times; surface_layer_height, boundary_layer_height)
    rad   = PrescribedRadiation(grid, times; land_surface,
                                ocean_surface = nothing, sea_ice_surface = nothing)
    ## Copy the already-interpolated CPU arrays straight into the XLA buffers rather
    ## than re-`interpolate!`: the ERA5 load and regrid are host work that cannot run
    ## on a Reactant grid, and reusing the arrays makes the compiled run see forcing
    ## identical to the CPU forward run, so the adjoint and finite-difference compare.
    copy_slice!(dst, src) = (parent(dst) .= Array(parent(src)); nothing)
    map_forcing_slices!(copy_slice!, times, atmos, rad, cpu.atmos, cpu.rad)
    update_state!(atmos); update_state!(rad)
    return (; atmos, rad, times)
end

# ## Model builder

# The porosity is represented with a `(Center, Center, Nothing)` field. `value` may be a scalar or a
# function `(λ, φ) -> …` while `set!` evaluates it at the cell centers.
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

# We initialize the skin temperature from the ERA5 T₂ₘ at the first snapshot.
cold_start_T₀(grid, forcing) =
    (T₀ = Field{Center, Center, Nothing}(grid);
     interpolate!(T₀, forcing.atmos.temperature[1]); T₀)

# ## Forward run
#
# We now run the model on the patch under the real ERA5 forcing, taking the
# porosity as a `Field`. The same routine drives both the nominal run and the
# per-cell finite-difference perturbations.

cpu_grid     = make_land_grid(CPU())
cpu_forcing  = inmemory_forcing(cpu_grid)
T₀_cpu       = cold_start_T₀(cpu_grid, cpu_forcing)

z_patch      = regrid_topography(cpu_grid; dataset = ETOPO2022())
cpu_porosity = porosity_field_on(cpu_grid, 0)
parent(cpu_porosity) .= elevation_porosity.(parent(z_patch))

# The ETOPO surface (`z_patch`) and ERA5's own topography are read on the CPU and 
# passed as plain arrays, so the correction materializes its exchange-grid `Δz` 
# through a device-safe array write and runs on either a CPU or a `ReactantState` 
# grid without host I/O on the XLA buffers. The unmaterialized correction is 
# grid-free.

z_era5_patch         = Field(Metadatum(:topography; dataset, date = patch_start_date, region = patch_region), cpu_grid)
surface_elevation    = Array(interior(z_patch, :, :, 1))
atmosphere_elevation = Array(interior(z_era5_patch, :, :, 1))
patch_correction     = ElevationCorrection(surface_elevation, atmosphere_elevation; lapse_rate = Γ_lapse)

function run_forward(grid, forcing, T₀, porosity_field; collect_series = false)
    model = era5_slab_land_model(grid, forcing, porosity_field, nominal_porosity;
                                 exchanger_correction = patch_correction)
    parent(model.land.temperature) .= parent(T₀)
    set!(model.land; M = initial_water_storage)

    T_mean = collect_series ? zeros(Nsteps) : Float64[]
    for n in 1:Nsteps
        time_step!(model, Δt)
        collect_series && (T_mean[n] = mean(interior(model.land.temperature, :, :, 1)))
    end
    T_final = Array(interior(model.land.temperature, :, :, 1))
    return (; T_final, T_mean)
end

@info "Running forward ERA5 slab-land patch on the CPU..."
forward = run_forward(cpu_grid, cpu_forcing, T₀_cpu, cpu_porosity; collect_series = true)
@info @sprintf("Forward done: ⟨T(t=%.2f d)⟩ = %.4f K", run_time / 86400, mean(forward.T_final))

# ## Building a differentiable workflow with Reactant and Enzyme
#
# Rebuild the patch on a `ReactantState` grid so that every array is an XLA buffer
# and differentiate with the porosity field as the variable.

grid_ad    = make_land_grid(ReactantState())
forcing_ad = transfer_forcing(grid_ad, cpu_forcing)
model_ad   = era5_slab_land_model(grid_ad, forcing_ad,
                                  porosity_field_on(grid_ad, nominal_porosity), nominal_porosity;
                                  exchanger_correction = patch_correction)

# Populate the state-exchanger regridder explicitly before the compiled run.
Oceananigans.initialize!(model_ad)

dmodel = Enzyme.make_zero(model_ad)
T₀_ad  = cold_start_T₀(grid_ad, forcing_ad)

# The differentiated input is the porosity field `ν`. Enzyme accumulates `∂L/∂ν(λ, φ)`
#  into its shadow `dν` which is a field of the same shape, i.e. the sensitivity map.

ν = porosity_field_on(grid_ad, nominal_porosity)
parent(ν) .= Array(parent(cpu_porosity))   # carry the CPU spatial pattern onto the XLA buffer
dν = Enzyme.make_zero(ν)

# ## The objective
#
# `final_skin_temperature_era5` copies the porosity field into the model and runs
# `nsteps` of the coupled time step inside a `@trace` loop, and returns the summed
# final skin temperature. The per-cell gradient of this sum is the pointwise sensitivity
# (columns are independent), so no rescaling is needed to compare with the finite-difference map.

function final_skin_temperature_era5(model, ν, T₀_field, M0, Δt, nsteps)
    parent(model.land.hydrology.porosity) .= parent(ν)

    set!(model.land.water_storage, M0)
    νp = parent(ν)
    θˡ = min.(M0 / (1000 * slab_depth), νp)
    parent(model.land.saturation) .= clamp.((θˡ .- residual_liquid_fraction) ./
                                            (νp .- residual_liquid_fraction), 0, 1)
    parent(model.land.temperature) .= parent(T₀_field)

    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end

    return sum(interior(model.land.temperature))
end

# ## The gradient wrapper
#
# Reverse mode with the model and the porosity *field* as `Duplicated` (primal +
# shadow); the initial state, step, and loop length are `Const`.

function grad_final_skin_temperature_era5(model, dmodel, ν, dν, T₀_field, M0, Δt, nsteps)
    parent(dν) .= 0
    _, L = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        final_skin_temperature_era5, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(ν, dν),
        Enzyme.Const(T₀_field),
        Enzyme.Const(M0),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps))
    return dν, L
end

# ## Compilation and execution

@info "Compiling differentiated ERA5 slab-land model — this may take a few minutes..."
compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_final_skin_temperature_era5(
    model_ad, dmodel, ν, dν, T₀_ad, initial_water_storage, Δt, Nsteps)

@info "Running gradient..."
dν, L_readout = compiled_grad(model_ad, dmodel, ν, dν, T₀_ad, initial_water_storage, Δt, Nsteps);

adjoint_map = Array(interior(dν, :, :, 1))   # ∂T(λ,φ)/∂ν(λ,φ), one reverse pass
N           = prod(patch_size)

@info @sprintf("Adjoint:  ⟨T(t=%.2f d)⟩ = %.4f K,  ⟨∂T/∂ν⟩ = %+.4e K",
               run_time / 86400, Reactant.to_number(L_readout) / N, mean(adjoint_map))

# ## Per-cell finite-difference check
#
# One reverse pass gave the whole map. We check it against a per-cell central
# finite difference: perturb *every* cell's porosity by `±δν` (uniformly, so each
# diagonal entry `∂T(λ,φ)/∂ν(λ,φ)` is recovered under column independence) and read
# the local skin-temperature response.

δν = 0.001 * nominal_porosity
ν_plus  = porosity_field_on(cpu_grid, 0); parent(ν_plus)  .= parent(cpu_porosity) .+ δν
ν_minus = porosity_field_on(cpu_grid, 0); parent(ν_minus) .= parent(cpu_porosity) .- δν
T_plus  = run_forward(cpu_grid, cpu_forcing, T₀_cpu, ν_plus)
T_minus = run_forward(cpu_grid, cpu_forcing, T₀_cpu, ν_minus)
fd_map  = (T_plus.T_final .- T_minus.T_final) ./ 2δν

max_abs_error = maximum(abs.(adjoint_map .- fd_map))
@info @sprintf("Adjoint vs finite-difference map:  max|Δ| = %.3e K", max_abs_error)

# ## Visualization

fig = Figure(size = (1700, 950), fontsize = 16)

λ, φ, _ = nodes(cpu_grid, Center(), Center(), Center())

ax_z = Axis(fig[1, 1]; title = "Elevation (m, ETOPO 2022)",
            xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hm_z = heatmap!(ax_z, λ, φ, Array(interior(z_patch, :, :, 1)); colormap = :terrain)
Colorbar(fig[1, 2], hm_z; label = "elevation (m)")

ax_T = Axis(fig[2, 1]; title = "Final skin temperature T (K)",
            xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hm_T = heatmap!(ax_T, λ, φ, forward.T_final; colormap = :thermal)
Colorbar(fig[2, 2], hm_T; label = "T (K)")

ax_ν = Axis(fig[1, 3]; title = "Porosity ν (elevation-derived)",
            xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hm_ν = heatmap!(ax_ν, λ, φ, Array(interior(cpu_porosity, :, :, 1)); colormap = :viridis)
Colorbar(fig[1, 4], hm_ν; label = "ν")

slim = max(maximum(abs.(adjoint_map)), maximum(abs.(fd_map)), eps())

ax_A = Axis(fig[2, 3]; title = "Adjoint ∂T/∂ν (K) — one reverse pass",
            xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hm_A = heatmap!(ax_A, λ, φ, adjoint_map; colormap = :balance, colorrange = (-slim, slim))
Colorbar(fig[2, 4], hm_A; label = "∂T/∂ν (K)")

ax_F = Axis(fig[2, 5]; title = "Finite-difference ∂T/∂ν (K)",
            xlabel = "longitude", ylabel = "latitude", aspect = DataAspect())
hm_F = heatmap!(ax_F, λ, φ, fd_map; colormap = :balance, colorrange = (-slim, slim))
Colorbar(fig[2, 6], hm_F; label = "∂T/∂ν (K)")

Label(fig[0, 1:6], @sprintf("Differentiable ERA5 slab land — pointwise ∂T/∂ν map (max|adjoint − FD| = %.2e K)",
                            max_abs_error))

save("era5_forced_slab_land_porosity_map.png", fig)
@info "Saved era5_forced_slab_land_porosity_map.png"

nothing #hide

# ![](era5_forced_slab_land_porosity_map.png)
