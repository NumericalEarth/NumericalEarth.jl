# # Rain → skin-temperature sensitivity — a differentiable dry-layer slab
#
# The [differentiable bucket example](differentiable_bucket_slab_land.md) takes
# the gradient of the land skin temperature with respect to the rain rate
# through the *classic* Manabe bucket (`SlabEnergy` + `BucketHydrology`,
# evaporation efficiency `β(𝒮)`). This example asks the same question — *how
# sensitive is the skin temperature to the rain rate?* — but through the
# **systematic-land** stack introduced by this PR:
#
#     energy    = WaterCoupledEnergy(...)          # C(Mˡᵃ) = C_dry + cˡ Mˡᵃ, conservative dTˡᵃ/dt
#     hydrology = VariablySaturatedHydrology(...)  # augmented-storage ϑˡ budget
#     humidity  = DryLayerHumidity(...)            # qⁱⁿ solved from a dry-layer vapor-flux balance
#
# The **dry-layer humidity closure** is the new physics on display. A diagnostic
# dry-layer depth `δᵛ(𝒮) = δᵛ_max · max(1 − 𝒮/𝒮ᶜ, 0)²` vanishes while the surface
# is saturated (`𝒮 ≥ 𝒮ᶜ`) and grows as the slab dries below the onset saturation
# `𝒮ᶜ`. Vapor escapes through that layer with a piston velocity
# `wᵈ = Dᵛ_eff / δᵛ`, so a saturated surface evaporates at the atmospheric-demand
# limit while a dried-out one is throttled — the two-stage bare-soil drying of
# [Or et al. (2013)](@cite or2013advances). A *thin* slab makes both stages
# visible within days: a rain pulse on day 2 briefly over-saturates the slab
# (`𝒮 = 1`), the latent heat flux jumps to its demand-limited daytime peaks, and
# once evaporation draws the slab back down through `𝒮ᶜ` the dry layer reopens
# and the latent heat collapses again. That is the headline of the forward run.
#
# The same wetting both *cools* the skin — more rain ⇒ higher `𝒮` ⇒ smaller `δᵛ`
# ⇒ freer evaporation ⇒ ``∂T / ∂\dot P < 0`` — and *raises* the heat capacity
# `C(Mˡᵃ) = C_dry + cˡ Mˡᵃ`, a competing thermal-inertia term the constant-capacity
# bucket lacks. We differentiate the skin temperature at the *end* of the run —
# five days on, with the slab dried back below `𝒮ᶜ` — and find it still
# *remembers* the pulse: more rain delayed the dry-down and left the surface
# cooler, so ``∂T / ∂\dot P < 0``. The gradient survives the whole drydown, the
# `𝒮ᶜ` crossing included, and a central finite difference reproduces it.
#
# **First** we run the dry-layer slab on the CPU under idealized analytic forcing.
# **Second** we treat that forward model as a *function* of the rain-pulse
# amplitude ``\dot P``, compile the coupled time-stepping with
# [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) down to
# [XLA](https://en.wikipedia.org/wiki/Accelerated_Linear_Algebra), and
# differentiate it with [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl) in
# reverse mode. The whole sensitivity ``∂T / ∂\dot P`` comes out of a single
# backward pass; a central finite difference at the end checks the adjoint.

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using CairoMakie
using Printf

using Oceananigans.TimeSteppers: update_state!, Clock

# ## Idealized forcing
#
# The diurnal air-temperature and shortwave cycle and the constant longwave /
# wind / humidity match the
# [idealized dry-layer example](idealized_dry_layer_slab_land.md) — the warm,
# dry 280–290 K regime that keeps evaporative demand high. A six-hour rain pulse
# falls on day 2; its amplitude `𝒫̇` (nominally `6e-4 kg m⁻² s⁻¹`, ≈ 13 mm over
# the six hours) is the *parameter* we differentiate with respect to — heavy
# enough to briefly *over-saturate* the thin slab.

const day = 86400

air_temperature(t)       = 285 - 5 * cos(2π * t / day)               # K, 280–290 diurnal
downwelling_shortwave(t) = max(0, 600 * cos(2π * (t - day/2) / day)) # W m⁻², daytime only
downwelling_longwave(t)  = 320                                       # W m⁻², constant
wind_speed               = 4.0                                       # m s⁻¹
specific_humidity        = 0.004                                     # kg kg⁻¹ (dry air drives evaporation)
surface_pressure         = 101325                                    # Pa

nominal_rain_rate = 6e-4                                             # kg m⁻² s⁻¹

in_rain_pulse(t) = 2days ≤ t ≤ 2days + 6hours
rain_pulse(t, 𝒫̇) = ifelse(in_rain_pulse(t), 𝒫̇, zero(𝒫̇))

# ## Forcing and model builders
#
# `forced_atmosphere` fills each `FieldTimeSeries` time slice from the analytic
# functions; the rain slices are set from the pulse amplitude `𝒫̇`. The builders
# are spatially uniform, so they work unchanged on the single-column grid used
# both for the forward run and for the differentiated run.

function forced_atmosphere(grid, times, 𝒫̇)
    atmosphere = PrescribedAtmosphere(grid, times; surface_layer_height = 10, boundary_layer_height = 512)
    set!(atmosphere; u = wind_speed, q = specific_humidity, p = surface_pressure)
    for n in eachindex(times)
        set!(atmosphere.tracers.T[n], air_temperature(times[n]))
        set!(atmosphere.freshwater_flux.rain[n], rain_pulse(times[n], 𝒫̇))
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

# ## The dry-layer slab
#
# A thin (5 cm) loamy slab, so the daytime evaporative demand dries it back
# across `𝒮ᶜ` within a few days of the pulse — exposing the stage-1 → stage-2
# drying a deep slab would hide. We start *below* `𝒮ᶜ`, so a dry surface layer
# is already present (`δᵛ > 0`) and the pre-pulse evaporation is throttled; the
# pulse then lifts `𝒮` above `𝒮ᶜ` and collapses `δᵛ`. We use `NoRunoff` and
# `NoDeepLiquidFlux` so evaporation is the only water sink and the full rain
# input reaches the storage budget (no infiltration-cap `clamp` to zero the
# sensitivity). The pulse drives the storage briefly *past* the saturation value
# `Mˡᵃ⁺ = ρˡ ν hˡᵃ`; with `NoRunoff` the augmented-storage budget carries the
# excess as a positive pressure head (`𝒮` pinned at 1) rather than shedding it,
# and evaporation then draws the slab back down across `𝒮ᶜ`.

liquid_density             = 1000.0
porosity                   = 0.4
slab_depth                 = 0.05
residual_liquid_fraction   = 0.05
dry_layer_onset_saturation = 0.5
maximum_water_storage      = liquid_density * porosity * slab_depth   # Mˡᵃ⁺ = ρˡ ν hˡᵃ = 20 kg m⁻²

initial_saturation  = 0.4   # below 𝒮ᶜ — a dry layer is already present
initial_temperature = 280.0 # K

# Diagnostic surface saturation `𝒮(Mˡᵃ)` for the variably-saturated slab:
# `θˡ = min(Mˡᵃ/(ρˡ hˡᵃ), ν)` and `𝒮 = clamp((θˡ − θʳ)/(ν − θʳ), 0, 1)`. Used
# both to convert the initial saturation to a storage and to seed `saturation`
# consistently inside the differentiated `loss`.

function saturation_from_storage(M)
    θˡ = min(M / (liquid_density * slab_depth), porosity)
    return clamp((θˡ - residual_liquid_fraction) / (porosity - residual_liquid_fraction), 0, 1)
end

initial_water_storage =
    (initial_saturation * (porosity - residual_liquid_fraction) + residual_liquid_fraction) *
    liquid_density * slab_depth

# The dry-layer humidity closure: the front depth grows from 0 (saturated skin)
# toward `δᵛ_max` as the slab dries past `𝒮ᶜ`, and `qⁱⁿ` is solved from the
# vapor-flux balance through it. Parameters follow the CLM5 / ClimaLand bare-soil
# scheme.

dry_layer_humidity = DryLayerHumidity(;
    dry_layer_depth = StorageBasedDryLayerDepth(;
        maximum_dry_layer_depth = 0.05,
        dry_layer_onset_saturation,
        dry_layer_exponent      = 2),
    vapor_exchange = DryLayerVaporPistonVelocity(
        minimum_dry_layer_depth = 1e-4,
        molecular_diffusivity   = 2.5e-5,
        tortuosity              = PowerLawTortuosity()),
    thermal_exchange_depth = 0.10,
    porosity)

# The turbulent-flux closure uses a **fixed-trip** Monin–Obukhov solver
# (`FixedIterations`) instead of the default tolerance-based `while` loop: the
# differentiated pass compiles the coupled time step to XLA and a data-dependent
# `while` loop becomes an XLA `while` op that does not differentiate cheaply,
# whereas a fixed iteration count unrolls into a static graph Enzyme can
# traverse. The same fixed-point solver also closes the dry-layer humidity
# balance. The forward and adjoint runs use the identical closure.

function dry_layer_model(grid, times, 𝒫̇)
    slab_land = SlabLand(grid;
        energy = WaterCoupledEnergy(eltype(grid);
            dry_heat_capacity         = 0.1 * 1500 * 1480,
            liquid_heat_capacity      = 4186,
            reference_temperature     = 273.15,
            deep_temperature          = 285.0,
            deep_time_scale           = 10days,
            advect_deep_liquid_energy = true),
        hydrology = VariablySaturatedHydrology(eltype(grid);
            slab_depth,
            porosity,
            residual_liquid_fraction,
            storage_height         = 1000,
            retention_curve        = VanGenuchtenRetention(α = 1.0, n = 2.0),
            hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-7, n = 2.0),
            deep_liquid_flux       = NoDeepLiquidFlux(),
            runoff                 = NoRunoff()))

    atmosphere = forced_atmosphere(grid, times, 𝒫̇)
    radiation  = forced_radiation(grid, times)

    fluxes = NumericalEarth.EarthSystemModels.InterfaceComputations.default_atmosphere_land_fluxes(
                 slab_land, eltype(grid); solver_stop_criteria = FixedIterations(8))

    al_interface = atmosphere_land_interface(grid, atmosphere, slab_land;
                                             specific_humidity = dry_layer_humidity,
                                             fluxes)

    ## `Clock(grid)` is a plain `Float64` clock on the CPU but a Reactant
    ## `ConcreteRNumber` clock on a `ReactantState` grid — the latter lets the
    ## model time advance *inside* the compiled XLA loop, so the time-dependent
    ## forcing (and the day-2 rain pulse) is sampled correctly each step.
    return AtmosphereLandModel(atmosphere, slab_land; radiation,
                               atmosphere_land_interface = al_interface,
                               clock = Clock(grid))
end

# ## Integration length
#
# Forward-Euler slab stepping is stable at `Δt = 10 minutes`. We integrate 5.84
# days (`29² = 841` steps) to trace the *whole* drydown — the day-2 pulse, the
# saturated evaporative plateau, and the stage-2 collapse — so that by the final
# step the slab has dried back well below `𝒮ᶜ` (`𝒮 ≈ 0.40`). That dried-out
# final temperature is the quantity we differentiate: a clean test that the
# gradient survives the entire drydown, the `𝒮ᶜ` crossing included. Reverse
# gradient checkpointing needs a perfect-square step count, hence `29²`.

Δt     = 10minutes
Nsteps = 29^2                              # 841 steps ≈ 5.84 days
times  = range(0, Nsteps * Δt, step = 1hour)

# `pulse_indices` are the `times` slices inside the rain pulse — the only slices
# whose value depends on `𝒫̇`. The differentiated `loss` re-sets exactly these.

pulse_indices = [n for n in eachindex(times) if in_rain_pulse(times[n])]

# ## Forward run
#
# A single-column slab integrated on the CPU, collecting the slab state (skin
# temperature, water storage, surface saturation) and the full surface energy
# budget `Rₙ = H + LE + G` — the same diagnostics the idealized dry-layer
# example reports. The single-column grid is zero-dimensional (`size = ()`).

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

# ## A differentiable workflow
#
# We now rebuild the *same* setup on a `ReactantState` grid so every array is an
# XLA buffer, wrap the coupled time-stepping in a scalar `loss`, compile it once,
# and differentiate it.

using Reactant, CUDA    # CUDA is required to load the Reactant KernelAbstractions extension (even on CPU)
using Enzyme
using Statistics: mean
using Oceananigans.Architectures: ReactantState
using Reactant: @trace

Reactant.set_default_backend("cpu")

# Disable LLVM's auto-vectorizers before any kernel is compiled. They pack the
# scalar per-grid-point arithmetic in the similarity solver, the dry-layer
# humidity balance, and the variably-saturated step into `vector<2xf64>` ops
# that EnzymeXLA's raise-to-StableHLO pass cannot lower — which here crashes
# LLVM outright during the differentiated compile. With the kernels kept scalar
# the raise succeeds; XLA re-vectorizes across grid points anyway, so the
# executable is not slowed down. This sets global LLVM state for the session, so
# prefer a session dedicated to this differentiable workflow.
Reactant.LLVM.clopts("-vectorize-slp=false", "-vectorize-loops=false")

grid_ad  = RectilinearGrid(ReactantState(); size = (), topology = (Flat, Flat, Flat))
model_ad = dry_layer_model(grid_ad, times, nominal_rain_rate)
dmodel   = Enzyme.make_zero(model_ad)

# The differentiated input is the rain-pulse amplitude `𝒫̇`, carried as a
# one-element XLA buffer so Enzyme can accumulate ``∂T / ∂\dot P`` into its
# shadow `d𝒫̇`.

𝒫̇  = Reactant.to_rarray([nominal_rain_rate])
d𝒫̇ = Enzyme.make_zero(𝒫̇)

# ### The objective
#
# `loss` re-initializes the slab from the (rain-independent) initial state, sets
# the rain pulse from the current `𝒫̇`, runs `nsteps` of the coupled time step
# inside a `@trace` loop, and returns the final (dried-out, day-5.84) skin
# temperature. The `set!`s *inside* `loss` are what make the gradient track the
# live trajectory rather than a stale one.

function loss(model, 𝒫̇, pulse_indices, initial_temperature, initial_water_storage, Δt, nsteps)
    ## Set the prognostic land state by filling the fields directly. We avoid
    ## `set!(model.land; …)` here because it also calls `update_state!(land)`,
    ## which launches the saturation diagnostic *kernel* — a bare kernel launch
    ## in straight-line autodiff code (outside the `@trace` loop) crashes the
    ## differentiated compile. The diagnostics are refreshed inside the loop by
    ## `time_step!` anyway; we seed `saturation` consistently for the first step.
    set!(model.land.temperature,   initial_temperature)
    set!(model.land.water_storage, initial_water_storage)
    set!(model.land.saturation,    saturation_from_storage(initial_water_storage))

    rate = sum(𝒫̇)  # 1-element reduction → differentiable scalar
    rain = model.atmosphere.freshwater_flux.rain
    for n in pulse_indices
        set!(rain[n], rate)
    end

    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end

    return mean(interior(model.land.temperature))
end

# ### The gradient wrapper
#
# Reverse mode with the model and rain amplitude as `Duplicated` (primal +
# shadow); the loop length and scalar parameters are `Const`.

function grad_loss(model, dmodel, 𝒫̇, d𝒫̇, pulse_indices, initial_temperature, initial_water_storage, Δt, nsteps)
    parent(d𝒫̇) .= 0
    _, T = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        loss, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(𝒫̇, d𝒫̇),
        Enzyme.Const(pulse_indices),
        Enzyme.Const(initial_temperature),
        Enzyme.Const(initial_water_storage),
        Enzyme.Const(Δt),
        Enzyme.Const(nsteps))
    return d𝒫̇, T
end

# ### Compilation and execution

@info "Compiling differentiated dry-layer land model — this may take a minute..."
compiled_grad = Reactant.@compile raise=true raise_first=true sync=true grad_loss(
    model_ad, dmodel, 𝒫̇, d𝒫̇, pulse_indices,
    initial_temperature, initial_water_storage, Δt, Nsteps)

@info "Running gradient..."
d𝒫̇, T_readout = compiled_grad(model_ad, dmodel, 𝒫̇, d𝒫̇, pulse_indices,
                              initial_temperature, initial_water_storage, Δt, Nsteps)

dT_drain = Array(d𝒫̇)[1]   # K / (kg m⁻² s⁻¹)

@info @sprintf("Adjoint:  T(t=%.2f d) = %.4f K,  ∂T/∂𝒫̇ = %.4e K / (kg m⁻² s⁻¹)",
               Nsteps * Δt / day, Reactant.to_number(T_readout), dT_drain)

# ### Finite-difference check
#
# A single reverse pass gave the whole sensitivity. To trust it, we compare
# against a central finite difference.
# The finite difference scheme converges and reproduces the adjoint to four significant figures.

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
#
# Six panels trace the forward run (the rain-pulse window shaded, the readout
# marked), in three columns of paired top/bottom rows. **Column 1** — the surface
# energy balance (`Rₙ` and the energy `G` into the slab) above the diurnal
# air-temperature forcing. **Column 2** — the dry-layer onset: the surface
# saturation `𝒮` above the turbulent heat fluxes it governs; as `𝒮` falls back
# through `𝒮ᶜ` the latent flux collapses and the sensible flux takes over (the
# Bowen-ratio flip). **Column 3** — the slab water storage `Mˡᵃ` (driven briefly
# *past* the saturation value `Mˡᵃ⁺` — augmented storage, no runoff, `𝒮` pinned at
# 1) above the skin temperature, which carries the differentiated quantity: its
# dried-out end-of-run value still *remembers* the pulse — more rain leaves it
# cooler (`∂T / ∂𝒫̇ < 0`, adjoint ≈ finite difference).

fig = Figure(size = (1600, 800), fontsize = 20)

pulse_span  = (2, 2 + 6/24)
readout_day = Nsteps * Δt / day
Tₐ          = air_temperature.(t .* day)   # prescribed diurnal forcing at the sample times

## The raw gradient ∂T/∂𝒫̇ is per unit *rain rate*, whose SI unit
## (1 kg m⁻² s⁻¹ ≈ 3.6 m hr⁻¹) is absurdly large — so the bare number is huge.
## Per millimetre of rain *delivered by the pulse* (depth = 𝒫̇ × pulse duration,
## and 1 kg m⁻² ≡ 1 mm of water) it reads as an intuitive ~0.04 K mm⁻¹.
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
