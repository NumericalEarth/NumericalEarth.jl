# # A canopy-air-space column: two-source vegetation, transpiration, and a rain pulse
#
# This idealized 0D example exercises the [`CanopyAirSpace`](@ref) land-surface
# component under an analytic diurnal forcing with a rain pulse dropped on day 2.
# It sweeps leaf-area index and stomatal conductance, then adds subgrid tiling
# ([`TiledLandInterface`](@ref)).

# ## Load packages
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: update_state!
using CairoMakie

# ## Idealized diurnal forcing with a rain pulse
#
# A warm, humid regime with a strong daytime shortwave pulse; longwave, wind, humidity,
# and pressure held constant. A 12-hour rain pulse falls on day 2 (after one full
# diurnal cycle) and refills the soil, momentarily saturating it.

day = 86400
air_temperature(t)       = 288 - 6 * cos(2π * t / day)                # K, 282–294 diurnal
downwelling_shortwave(t) = max(0, 800 * cos(2π * (t - day/2) / day))  # W m⁻², daytime only
downwelling_longwave(t)  = 330                                        # W m⁻²
wind_speed               = 3.0                                        # m s⁻¹
specific_humidity        = 0.007                                       # kg kg⁻¹ (dry air keeps evaporative demand high)
surface_pressure         = 101325                                     # Pa

rain_rate = 8e-4                                                      # kg m⁻² s⁻¹ (≈ 35 mm over 12 h)
in_rain_pulse(t) = 2day ≤ t ≤ 2.5day
rain_pulse(t)    = ifelse(in_rain_pulse(t), rain_rate, zero(rain_rate))

# ## Forcing builders
#
# Each `FieldTimeSeries` slice is filled from the analytic functions; the coupled
# model then interpolates between hourly slices as it steps. The rain enters as the
# atmosphere's precipitation flux, which the land hydrology reads as a water source.

function forced_atmosphere(grid, times)
    atmosphere = PrescribedAtmosphere(grid, times; surface_layer_height = 10, boundary_layer_height = 512)
    for n in eachindex(times)
        set!(atmosphere.velocities.u[n],            wind_speed)
        set!(atmosphere.specific_humidity[n],       specific_humidity)
        set!(atmosphere.pressure[n],                surface_pressure)
        set!(atmosphere.temperature[n],             air_temperature(times[n]))
        set!(atmosphere.precipitation_flux.rain[n], rain_pulse(times[n]))
    end
    update_state!(atmosphere)
    return atmosphere
end

function forced_radiation(grid, times)
    radiation = PrescribedRadiation(grid, times;
                                    land_surface  = SurfaceRadiationProperties(0.2, 0.95),
                                    ocean_surface = nothing, sea_ice_surface = nothing)
    for n in eachindex(times)
        set!(radiation.downwelling_shortwave[n], downwelling_shortwave(times[n]))
        set!(radiation.downwelling_longwave[n],  downwelling_longwave(times[n]))
    end
    update_state!(radiation)
    return radiation
end

# ## Canopy, soil, and slab constants
#
# Beer–Lambert and two-face-longwave optics constants, shared by the model below and
# the post-run radiation partition.

leaf_albedo           = 0.15
ground_albedo         = 0.15
max_canopy_emissivity = 0.98
ground_emissivity     = 0.96
extinction            = 0.5   # Beer–Lambert K
clumping              = 1.0   # foliage clumping Ω
stefan_boltzmann      = 5.670374419e-8

# The critical saturation `𝒮ᶜ` sets where the bare-soil branch becomes water-limited,
# opening the soil's dry layer and throttling ground evaporation below it.

critical_saturation = 0.5

# The soil vapor branch is a `DryLayerHumidity`: a dry surface layer that grows as the
# soil dries past `𝒮ᶜ`, throttling bare-soil evaporation. A thicker maximum layer makes
# the dry surface strongly evaporation-limited (and therefore hot).

soil_branch() = DryLayerHumidity(;
    dry_layer_depth = StorageBasedDryLayerDepth(maximum_dry_layer_depth    = 0.025,
                                                dry_layer_onset_saturation = critical_saturation,
                                                dry_layer_exponent         = 2),
    vapor_exchange  = DryLayerVaporPistonVelocity(minimum_dry_layer_depth = 1e-3,
                                                  molecular_diffusivity   = 2.4e-5,
                                                  tortuosity              = ConstantTortuosity()),
    thermal_exchange_depth = 0.05, porosity = soil_porosity)

# The soil is a `VariablySaturatedHydrology` column wrapped by an `InterceptingHydrology`
# canopy interception store, stepped with the conservative `WaterCoupledEnergy`. Every
# column below shares this land, starting dry (below `𝒮ᶜ`).

vegetated_leaf_area_index   = 4.0
soil_slab_depth             = 0.1
soil_porosity               = 0.4
soil_residual_fraction      = 0.05
liquid_density              = 1000.0
soil_retention_α            = 1.0     # van Genuchten inverse air-entry (m⁻¹)
soil_retention_n            = 2.0     # van Genuchten pore-size shape
soil_retention              = VanGenuchtenRetention(inverse_air_entry_head = soil_retention_α, pore_size_uniformity = soil_retention_n)
soil_conductivity_saturated = 1e-6    # K_sat (m s⁻¹)
interception_capacity       = 0.1     # c — canopy water capacity per unit LAI (kg m⁻², BATS/CLM)

initial_temperature     = 286.0
initial_soil_saturation = 0.3
initial_soil_storage    = (initial_soil_saturation * (soil_porosity - soil_residual_fraction) +
                           soil_residual_fraction) * liquid_density * soil_slab_depth

variably_saturated_soil() = VariablySaturatedHydrology(;
    slab_depth = soil_slab_depth, porosity = soil_porosity,
    residual_liquid_fraction = soil_residual_fraction, storage_height = 1000,
    retention_curve        = soil_retention,
    hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = soil_conductivity_saturated, pore_size_uniformity = soil_retention_n),
    deep_liquid_flux       = NoDeepLiquidFlux(),
    runoff                 = InfiltrationCapacityRunoff(infiltration_capacity = 5e-4))

conservative_energy() = WaterCoupledEnergy(;
    dry_heat_capacity = 0.1 * 1500 * 1480, liquid_heat_capacity = 4186,
    reference_temperature = 273.15, deep_temperature = 286, deep_time_scale = 10day)

# The `InterceptingHydrology` wraps the soil with the canopy store; its `leaf_area_index`
# sets the caught fraction and capacity `Wᶜᵐᵃˣ = c·LAI`.
intercepting_hydrology(leaf_area_index) = InterceptingHydrology(;
    soil = variably_saturated_soil(), leaf_area_index,
    capacity_per_leaf_area = interception_capacity)

# ## Model builder
#
# Both interface slots take the same `CanopyAirSpace` object, carrying a
# `CanopyInterception` so the wet leaf evaporates intercepted rain. Every column shares
# one soil branch, moisture stress, PAR, skin flux, and optics; only the leaf-area
# index and stomatal `conductance` vary.

canopy_with_interception(; leaf_area_index, conductance = JarvisConductance()) = CanopyAirSpace(;
    soil   = soil_branch(),
    canopy = CanopyConductanceHumidity(; leaf_area_index, conductance,
                                       moisture_stress = PlantAvailableWaterStress(),
                                       absorbed_par    = InteractiveAbsorbedPAR()),
    soil_skin_flux = SoilConductiveFlux(1.5, 0.05),
    leaf_albedo, ground_albedo, max_canopy_emissivity, ground_emissivity,
    extinction, clumping,
    interception = CanopyInterception())

# The land is the shared variably-saturated soil with the interception store; the two knobs
# enter here — `leaf_area_index` upscales the leaf conductance and sets the shading, and
# `conductance` selects the stomatal model.

function canopy_air_space_column(; leaf_area_index, conductance, label)
    grid       = RectilinearGrid(CPU(); size = (), topology = (Flat, Flat, Flat))
    atmosphere = forced_atmosphere(grid, times)
    radiation  = forced_radiation(grid, times)

    land = SlabLand(grid; energy = conservative_energy(),
                    hydrology = intercepting_hydrology(leaf_area_index))
    set!(land; T = initial_temperature, M = initial_soil_storage, canopy_water_storage = 0.0)

    canopy_air_space = canopy_with_interception(; leaf_area_index, conductance)

    model = AtmosphereLandModel(atmosphere, land; radiation,
                                atmosphere_land_interface_temperature       = canopy_air_space,
                                atmosphere_land_interface_specific_humidity = canopy_air_space)

    ## Diagnostics: the CAS stores its node/leaf/soil-skin/effective temperatures, the
    ## skin→bulk ground heat flux, and the two-source latent/sensible shares plus the
    ## wet-canopy evaporation in `interface.temperature`; the atmosphere-facing turbulent
    ## totals live in `interface.fluxes`; the water fluxes/stores in `land`.
    interface = model.interfaces.atmosphere_land_interface

    T   = zeros(Nsteps)   # time (days)
    Tₐ  = zeros(Nsteps)   # air (forcing)
    Tˡᵉᵃᶠ  = zeros(Nsteps)   # leaf
    Tᵃᶜ = zeros(Nsteps)   # canopy-air node
    Tᵍ = zeros(Nsteps)   # soil skin
    Tˡᵃ = zeros(Nsteps)   # bulk slab
    Tₑ  = zeros(Nsteps)   # effective radiating (LST)
    H   = zeros(Nsteps)   # sensible heat, atmosphere total (positive up)
    LE  = zeros(Nsteps)   # latent heat, atmosphere total (positive up)
    LEˡᵉᵃᶠ = zeros(Nsteps)   # leaf latent (transpiration + wet-canopy, positive up)
    LEᵍ = zeros(Nsteps)   # soil evaporation — ground latent share (positive up)
    Hˡᵉᵃᶠ  = zeros(Nsteps)   # leaf sensible share
    Hᵍ  = zeros(Nsteps)   # ground sensible share
    𝒬ᵍ  = zeros(Nsteps)   # skin → bulk ground heat flux
    SW  = zeros(Nsteps)   # incident shortwave
    LW  = zeros(Nsteps)   # incident (downwelling) longwave
    E   = zeros(Nsteps)   # soil + transpiration vapor flux Jᵛ − Eʷᵉᵗ (kg m⁻² s⁻¹, up)
    Ewet = zeros(Nsteps)  # wet-canopy evaporation (kg m⁻² s⁻¹, up)
    LEwet = zeros(Nsteps) # wet-canopy latent heat ℒ·Eʷᵉᵗ (W m⁻², up)
    P   = zeros(Nsteps)   # incident rain (kg m⁻² s⁻¹, down)
    Pˡ  = zeros(Nsteps)   # throughfall reaching the soil (kg m⁻² s⁻¹, down)
    𝒮   = zeros(Nsteps)   # surface saturation
    M   = zeros(Nsteps)   # soil water storage (kg m⁻²)
    Wᶜ  = zeros(Nsteps)   # canopy water store (mm)

    @info "Running canopy-air-space column: $label ..."
    for n in 1:Nsteps
        time_step!(model, Δt)
        Ts   = interface.temperature
        time = model.clock.time
        T[n]   = time / day
        Tₐ[n]  = air_temperature(time)
        Tˡᵉᵃᶠ[n]  = scalar(Ts.canopy)
        Tᵃᶜ[n] = scalar(Ts.interface)
        Tᵍ[n] = scalar(Ts.soil_skin)
        Tˡᵃ[n] = scalar(land.temperature)
        Tₑ[n]  = scalar(Ts.effective)
        H[n]   = scalar(interface.fluxes.sensible_heat)
        LE[n]  = scalar(interface.fluxes.latent_heat)
        LEˡᵉᵃᶠ[n] = scalar(Ts.canopy_latent_heat)
        LEᵍ[n] = scalar(Ts.soil_latent_heat)
        Hˡᵉᵃᶠ[n]  = scalar(Ts.canopy_sensible_heat)
        Hᵍ[n]  = scalar(Ts.soil_sensible_heat)
        𝒬ᵍ[n]  = scalar(Ts.ground_heat_flux)
        SW[n]  = downwelling_shortwave(time)
        LW[n]  = downwelling_longwave(time)
        E[n]     = scalar(land.fluxes.vapor_flux)
        Ewet[n]  = scalar(Ts.canopy_evaporation)
        LEwet[n] = scalar(Ts.canopy_wet_latent_heat)
        P[n]     = applied_rain(model)
        Pˡ[n]    = scalar(land.diagnostics.throughfall)
        𝒮[n]   = scalar(land.saturation)
        M[n]   = scalar(land.water_storage)
        Wᶜ[n]  = scalar(land.prognostic.canopy_water_storage)
    end

    ## The leaf latent `LEˡᵉᵃᶠ` is transpiration + wet-canopy; the model reports the wet share
    ## `LEwet = ℒ·Eʷᵉᵗ` with the *same* `ℒ` as `LEˡᵉᵃᶠ`, so `transpiration` is the exact
    ## stomatal flux (and stays ≥ 0, unlike a split against a fixed reference `ℒ`).
    transpiration = LEˡᵉᵃᶠ .- LEwet

    return (; label, leaf_area_index,
              t = T, Tₐ, Tˡᵉᵃᶠ, Tᵃᶜ, Tᵍ, Tˡᵃ, Tₑ, H, LE, LEˡᵉᵃᶠ, transpiration, LEᵍ, Hˡᵉᵃᶠ, Hᵍ, 𝒬ᵍ, SW, LW,
              E, Ewet, LEwet, P, Pˡ, 𝒮, M, Wᶜ)
end

scalar(field) = first(interior(field))

# The rain the land actually received this step — the atmosphere's hourly series interpolated
# to the coupler — so incident rain and the model's throughfall are read from one source.
applied_rain(model) = scalar(model.interfaces.exchanger.atmosphere.state.Jʳⁿ)

# ## Integration length and the case set
#
# Six days at `Δt = 10 min`, one full diurnal cycle before the pulse and three-plus
# days of drydown after it.

Δt     = 10minutes
Nsteps = 6 * 144                     # 6 days
times  = range(0, Nsteps * Δt, step = 1hour)

# The bare/moderate/dense sweep uses the default Jarvis conductance; the dense case is
# repeated with Medlyn conductance at the same LAI.

medlyn = MedlynConductance()
jarvis = JarvisConductance()

bare     = canopy_air_space_column(leaf_area_index = 0.0, conductance = jarvis, label = "bare (LAI 0)")
moderate = canopy_air_space_column(leaf_area_index = 2.0, conductance = jarvis, label = "moderate (LAI 2)")
dense    = canopy_air_space_column(leaf_area_index = 4.0, conductance = jarvis, label = "dense (LAI 4)")
dense_m  = canopy_air_space_column(leaf_area_index = 4.0, conductance = medlyn, label = "dense, Medlyn")

# ## Radiation partition (reconstructed from the stored surface temperatures)
#
# The CAS keeps only the surface temperatures and turbulent totals, so the radiative
# shares are reconstructed here from the stored leaf `Tˡᵉᵃᶠ` and soil-skin `Tᵍ`
# temperatures using the Beer–Lambert shortwave split and two-face longwave ledger.

function radiation_partition(case)
    LAI = case.leaf_area_index
    σ   = stefan_boltzmann
    εˡᵉᵃᶠ = max_canopy_emissivity * (1 - exp(-LAI))
    εᵍ = ground_emissivity
    transmittance = exp(-extinction * LAI * clumping)

    SW, LW, Tˡᵉᵃᶠ, Tᵍ = case.SW, case.LW, case.Tˡᵉᵃᶠ, case.Tᵍ

    ## Shortwave: the canopy absorbs `(1−α_leaf)(1−f_trans)`, the shaded soil gets the
    ## transmitted remainder `f_trans(1−α_ground)`.
    SWˡᵉᵃᶠ = @. (1 - leaf_albedo) * (1 - transmittance) * SW
    SWᵍ = @. transmittance * (1 - ground_albedo) * SW

    ## Longwave: below-canopy downwelling (reaching the soil), the canopy's own
    ## downward emission, the upwelling above the soil, and the total escaping to space.
    LWꜜᵍ      = @. (1 - εˡᵉᵃᶠ) * LW + εˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^4     # reaching soil
    canopy_emission = @. εˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^4                     # canopy emission (one face)
    LWꜛᵍ      = @. εᵍ * σ * Tᵍ^4 + (1 - εᵍ) * LWꜜᵍ # upwelling from ground
    LWu        = @. (1 - εˡᵉᵃᶠ) * LWꜛᵍ + εˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^4  # to space

    ## Net radiation absorbed by the whole land column (canopy + soil).
    Rₙ = @. SWˡᵉᵃᶠ + SWᵍ + LW - LWu

    return (; SWˡᵉᵃᶠ, SWᵍ, LWꜜᵍ, canopy_emission, LWꜛᵍ, LWu, Rₙ)
end

# ## Plot helpers

mm_per_day(flux) = flux .* 86400   # kg m⁻² s⁻¹ → mm day⁻¹
pulse_span = (2, 2.5)              # rain-pulse span in days
mark_pulse!(ax) = vspan!(ax, pulse_span...; color = (:skyblue, 0.25))

# A right-hand twin axis sharing the x-axis, for panels that carry two scales.
function twin_axis(cell; ylabel, color)
    ax = Axis(cell; ylabel, yaxisposition = :right, ygridvisible = false,
              ylabelcolor = color, yticklabelcolor = color)
    hidespines!(ax, :t, :l, :b)
    hidexdecorations!(ax)
    return ax
end

# ## Figure 1 — anatomy of the dense canopy through the rain pulse
#
# The full component breakdown for the dense (`LAI = 4`, Jarvis) column: how each
# temperature, radiative share, turbulent flux, and water variable responds before,
# during, and after the pulse (shaded band).

ref  = dense
rad  = radiation_partition(ref)
t    = ref.t

fig = Figure(size = (1700, 1850), fontsize = 17)

## (1,1) Temperatures — leaf leads and swings widest, shaded soil skin is damped,
## bulk slab lags, node sits between the surfaces and the air.
ax = Axis(fig[1, 1]; title = "Temperatures", xlabel = "t (days)", ylabel = "T (K)")
mark_pulse!(ax)
lines!(ax, t, ref.Tₐ;  color = :gray,        linewidth = 3, label = "air Tₐ")
lines!(ax, t, ref.Tˡᵉᵃᶠ;  color = :seagreen,    label = "leaf Tˡᵉᵃᶠ")
lines!(ax, t, ref.Tᵍ; color = :saddlebrown, label = "soil skin Tᵍ")
lines!(ax, t, ref.Tᵃᶜ; color = :steelblue,   linestyle = :dash, label = "canopy air Tᵃᶜ")
lines!(ax, t, ref.Tˡᵃ; color = :firebrick,   label = "bulk Tˡᵃ")
lines!(ax, t, ref.Tₑ;  color = :black,       linestyle = :dot, label = "LST effective_temperature")
axislegend(ax; position = :lt, nbanks = 2, labelsize = 12)

## (1,2) Two-source temperature contrasts.
ax = Axis(fig[1, 2]; title = "Source temperature contrasts", xlabel = "t (days)", ylabel = "ΔT (K)")
mark_pulse!(ax)
hlines!(ax, [0]; color = :gray, linestyle = :dash)
lines!(ax, t, ref.Tˡᵉᵃᶠ  .- ref.Tᵍ; color = :seagreen,    label = "leaf − soil skin")
lines!(ax, t, ref.Tᵃᶜ .- ref.Tₐ;  color = :steelblue,   label = "node − air")
lines!(ax, t, ref.Tᵍ .- ref.Tˡᵃ; color = :saddlebrown, label = "soil skin − bulk")
axislegend(ax; position = :lt, labelsize = 12)

## (2,1) Shortwave partition — the canopy shades the soil.
ax = Axis(fig[2, 1]; title = "Downwelling shortwave partition", xlabel = "t (days)", ylabel = "SW (W m⁻²)")
mark_pulse!(ax)
lines!(ax, t, ref.SW;        color = :goldenrod, linewidth = 3, label = "incident on land")
lines!(ax, t, rad.SWˡᵉᵃᶠ; color = :seagreen,    label = "absorbed by canopy")
lines!(ax, t, rad.SWᵍ; color = :saddlebrown, label = "reaching soil")
axislegend(ax; position = :lt, labelsize = 12)

## (2,2) Longwave down partition.
ax = Axis(fig[2, 2]; title = "Downwelling longwave partition", xlabel = "t (days)", ylabel = "LW (W m⁻²)")
mark_pulse!(ax)
lines!(ax, t, ref.LW;         color = :black,     linewidth = 3, label = "incident on land")
lines!(ax, t, rad.LWꜜᵍ;      color = :saddlebrown, label = "reaching soil")
lines!(ax, t, rad.canopy_emission; color = :seagreen,  label = "emitted down by canopy")
axislegend(ax; position = :lt, labelsize = 12)

## (3,1) Longwave up partition.
ax = Axis(fig[3, 1]; title = "Upwelling longwave partition", xlabel = "t (days)", ylabel = "LW (W m⁻²)")
mark_pulse!(ax)
lines!(ax, t, rad.LWꜛᵍ;      color = :saddlebrown, label = "from ground")
lines!(ax, t, rad.canopy_emission; color = :seagreen,   label = "from canopy")
lines!(ax, t, rad.LWu;        color = :black,        linewidth = 2, label = "to space (LST)")
axislegend(ax; position = :lt, labelsize = 12)

## (3,2) Surface energy balance — net radiation into the land splits into turbulent
## sensible + latent to the atmosphere and conduction into the slab.
ax = Axis(fig[3, 2]; title = "Surface energy fluxes\n(positive: surface → atmosphere; 𝒬ᵍ into slab)",
          xlabel = "t (days)", ylabel = "flux (W m⁻²)")
mark_pulse!(ax)
hlines!(ax, [0]; color = :gray, linestyle = :dash)
lines!(ax, t, rad.Rₙ;  color = :black,     linewidth = 2, label = "net radiation Rₙ")
lines!(ax, t, ref.LE;  color = :navy,      label = "latent H₂O (LE)")
lines!(ax, t, ref.H;   color = :orange,    label = "sensible (H)")
lines!(ax, t, ref.𝒬ᵍ;  color = :seagreen,  linestyle = :dash, label = "ground heat 𝒬ᵍ")
axislegend(ax; position = :lt, labelsize = 12)

## (4,1) Latent-heat pathways — the atmosphere feels only the total LE, but the CAS
## resolves it into leaf transpiration, soil evaporation, and (while the canopy is wet)
## wet-canopy evaporation from the interception store.
ax = Axis(fig[4, 1]; title = "Latent-heat pathways", xlabel = "t (days)", ylabel = "LE (W m⁻²)")
mark_pulse!(ax)
lines!(ax, t, ref.LE;            color = :navy,        linewidth = 2, label = "total LE")
lines!(ax, t, ref.transpiration; color = :seagreen,    label = "transpiration")
lines!(ax, t, ref.LEᵍ;           color = :saddlebrown, label = "soil evaporation")
lines!(ax, t, ref.LEwet; color = :steelblue, label = "wet-canopy evaporation")
axislegend(ax; position = :lt, labelsize = 12)

## (4,2) Two-source sensible split — leaf and under-canopy ground shares of the sensible
## flux the atmosphere feels.
ax = Axis(fig[4, 2]; title = "Two-source sensible heat", xlabel = "t (days)", ylabel = "H (W m⁻²)")
mark_pulse!(ax)
hlines!(ax, [0]; color = :gray, linestyle = :dash)
lines!(ax, t, ref.H;  color = :orange,      linewidth = 2, label = "total H")
lines!(ax, t, ref.Hˡᵉᵃᶠ; color = :seagreen,    label = "leaf")
lines!(ax, t, ref.Hᵍ; color = :saddlebrown, label = "ground")
axislegend(ax; position = :lt, labelsize = 12)

## (5,1) Moisture fluxes — evaporation (soil + transpiration) against the rain pulse,
## on a second axis (the pulse dwarfs the background ET).
ax = Axis(fig[5, 1]; title = "Moisture fluxes", xlabel = "t (days)",
          ylabel = "evaporation (mm day⁻¹)", ylabelcolor = :navy, yticklabelcolor = :navy)
mark_pulse!(ax)
lines!(ax, t, mm_per_day(ref.E); color = :navy, label = "evaporation")
axislegend(ax; position = :lt, labelsize = 12)
axP = twin_axis(fig[5, 1]; ylabel = "precipitation (mm day⁻¹)", color = :skyblue)
linkxaxes!(ax, axP)
lines!(axP, t, mm_per_day(ref.P); color = :skyblue, linewidth = 2)

## (5,2) Water state — surface saturation and the storage it tracks; the pulse pins
## 𝒮 at 1, then evaporation draws it back down through the dry-layer onset 𝒮ᶜ.
ax = Axis(fig[5, 2]; title = "Water state", xlabel = "t (days)",
          ylabel = "saturation 𝒮", ylabelcolor = :darkorange, yticklabelcolor = :darkorange)
mark_pulse!(ax)
lines!(ax, t, ref.𝒮; color = :darkorange, linewidth = 2, label = "𝒮")
hlines!(ax, [critical_saturation]; color = :gray, linestyle = :dash, label = "dry-layer onset 𝒮ᶜ")
ylims!(ax, 0, 1.05)
axislegend(ax; position = :rc, labelsize = 12)
axM = twin_axis(fig[5, 2]; ylabel = "soil storage (kg m⁻²)", color = :navy)
linkxaxes!(ax, axM)
lines!(axM, t, ref.M; color = :navy, linestyle = :dot)
ylims!(axM, 0, 1.05 * maximum(ref.M))

Label(fig[0, 1:2], "Canopy-air-space column — dense canopy (LAI 4, Jarvis), rain pulse on day 2", fontsize = 22)

save("canopy_air_space_column.png", fig)
@info "Saved canopy_air_space_column.png"

nothing #hide

# ![](canopy_air_space_column.png)

# ## Figure 2 — turning the two knobs
#
# The left column sweeps leaf-area index (bare → dense, Jarvis); the right column
# fixes `LAI = 4` and compares Jarvis against Medlyn stomata. Rows show leaf
# temperature, transpiration, soil evaporation, and surface saturation.

lai_cases  = (bare, moderate, dense)
lai_colors = (:saddlebrown, :seagreen, :darkgreen)
con_cases  = (dense, dense_m)
con_colors = (:darkgreen, :purple)

fig = Figure(size = (1500, 1500), fontsize = 17)

function knob_panel(cell, cases, colors, field, title, ylabel; transform = identity, legend = :lt)
    ax = Axis(cell; title, xlabel = "t (days)", ylabel)
    mark_pulse!(ax)
    for (case, color) in zip(cases, colors)
        lines!(ax, case.t, transform(getproperty(case, field)); color, label = case.label)
    end
    axislegend(ax; position = legend, labelsize = 12)
    return ax
end

knob_panel(fig[1, 1], lai_cases, lai_colors, :Tˡᵉᵃᶠ, "Leaf temperature — LAI sweep", "Tˡᵉᵃᶠ (K)")
knob_panel(fig[1, 2], con_cases, con_colors, :Tˡᵉᵃᶠ, "Leaf temperature — Jarvis vs Medlyn", "Tˡᵉᵃᶠ (K)")

knob_panel(fig[2, 1], lai_cases, lai_colors, :transpiration, "Transpiration — LAI sweep", "transpiration (W m⁻²)")
knob_panel(fig[2, 2], con_cases, con_colors, :transpiration, "Transpiration — Jarvis vs Medlyn", "transpiration (W m⁻²)")

knob_panel(fig[3, 1], lai_cases, lai_colors, :LEᵍ, "Soil evaporation — LAI sweep", "LEᵍ (W m⁻²)")
knob_panel(fig[3, 2], con_cases, con_colors, :LEᵍ, "Soil evaporation — Jarvis vs Medlyn", "LEᵍ (W m⁻²)")

ax = knob_panel(fig[4, 1], lai_cases, lai_colors, :𝒮, "Surface saturation — LAI sweep", "𝒮"; legend = :rc)
hlines!(ax, [critical_saturation]; color = :gray, linestyle = :dash)
ylims!(ax, 0, 1.05)
ax = knob_panel(fig[4, 2], con_cases, con_colors, :𝒮, "Surface saturation — Jarvis vs Medlyn", "𝒮"; legend = :rc)
hlines!(ax, [critical_saturation]; color = :gray, linestyle = :dash)
ylims!(ax, 0, 1.05)

Label(fig[0, 1:2], "Canopy-air-space column — turning the LAI and stomatal-conductance knobs", fontsize = 22)

save("canopy_air_space_knobs.png", fig)
@info "Saved canopy_air_space_knobs.png"

nothing #hide

# ![](canopy_air_space_knobs.png)

# ## Sweeping the bare-soil fraction: a vegetation/bare mosaic
#
# Subgrid tiling ([`TiledLandInterface`](@ref)) makes the cell a mosaic of a vegetated
# fraction `f_veg` and bare-soil fraction `1 − f_veg`, area-weighted into one boundary
# condition `𝒬 = f_veg·𝒬_veg + (1 − f_veg)·𝒬_bare`. It reuses the same soil, energy, and
# interception store as above, with only the tiling and the shared store's cell-average
# LAI `f_veg·LAI_veg` changing.

# The vegetated tile is the same `canopy_with_interception` as above (full `LAI_veg`);
# its bare (`LAI = 0`) counterpart is derived automatically by `TiledLandInterface`. The
# interception store is a cell-average quantity (LAI `f_veg·LAI_veg`, capacity
# `c·f_veg·LAI_veg`) set on the land's `InterceptingHydrology`.

function tiled_intercepting_column(; fraction, label)
    grid       = RectilinearGrid(CPU(); size = (), topology = (Flat, Flat, Flat))
    atmosphere = forced_atmosphere(grid, times)
    radiation  = forced_radiation(grid, times)

    land = SlabLand(grid; energy = conservative_energy(),
                    hydrology = intercepting_hydrology(fraction * vegetated_leaf_area_index))
    set!(land; T = initial_temperature, M = initial_soil_storage, canopy_water_storage = 0.0)

    tiled = TiledLandInterface(grid, atmosphere, land;
                               vegetated = canopy_with_interception(; leaf_area_index = vegetated_leaf_area_index),
                               fraction)
    model = AtmosphereLandModel(atmosphere, land; radiation, atmosphere_land_interface = tiled)
    interface = model.interfaces.atmosphere_land_interface

    T    = zeros(Nsteps)   # time (days)
    H    = zeros(Nsteps)   # blended sensible heat
    LE   = zeros(Nsteps)   # blended latent heat
    effective_temperature = zeros(Nsteps)   # land-surface (radiometric) temperature
    𝒮    = zeros(Nsteps)   # shared-column saturation
    Wᶜ   = zeros(Nsteps)   # canopy water store (mm)
    Ewet = zeros(Nsteps)   # wet-canopy evaporation (mass flux, positive up)
    LEwet = zeros(Nsteps)  # wet-canopy latent heat ℒ·Eʷᵉᵗ (W m⁻², up)
    LEˡᵉᵃᶠ  = zeros(Nsteps)   # blended leaf latent (transpiration + wet-canopy)
    LEᵍ  = zeros(Nsteps)   # blended soil latent
    P    = zeros(Nsteps)   # incident rain (mass flux, down)
    Pˡ   = zeros(Nsteps)   # throughfall reaching the soil (mass flux, down)

    @info "Running tiled interception column: $label ..."
    for n in 1:Nsteps
        time_step!(model, Δt)
        time    = model.clock.time
        T[n]    = time / day
        H[n]    = scalar(interface.fluxes.sensible_heat)
        LE[n]   = scalar(interface.fluxes.latent_heat)
        effective_temperature[n] = scalar(interface.temperature.effective)
        𝒮[n]    = scalar(land.saturation)
        Wᶜ[n]   = scalar(land.prognostic.canopy_water_storage)
        Ewet[n]  = scalar(interface.temperature.canopy_evaporation)
        LEwet[n] = scalar(interface.temperature.canopy_wet_latent_heat)
        LEˡᵉᵃᶠ[n]   = scalar(interface.temperature.canopy_latent_heat)
        LEᵍ[n]   = scalar(interface.temperature.soil_latent_heat)
        P[n]     = applied_rain(model)
        Pˡ[n]    = scalar(land.diagnostics.throughfall)
    end

    return (; label, fraction, t = T, H, LE, effective_temperature, 𝒮, Wᶜ, Ewet, LEwet, LEˡᵉᵃᶠ, LEᵍ, P, Pˡ)
end

# ## The bare-soil-fraction sweep

fractions   = (0.0, 0.25, 0.5, 0.75, 1.0)
tiled_cases = map(f -> tiled_intercepting_column(fraction = f, label = "f_veg = $f"), fractions)
full_canopy = tiled_cases[end]   # f_veg = 1 — the interception store's anatomy

# ## Figure 3 — sweeping the vegetation fraction with an active interception store
#
# As `f_veg` rises the partition shifts from a hot, sensible-dominated bare surface toward a
# cooler, transpiring canopy — and the interception store and wet-canopy evaporation, absent
# over bare soil, grow with the vegetated area.

frac_colors = cgrad(:viridis, length(fractions); categorical = true)

fig = Figure(size = (1500, 1200), fontsize = 17)

function frac_panel(cell, field, title, ylabel; transform = identity, legend = :lt)
    ax = Axis(cell; title, xlabel = "t (days)", ylabel)
    mark_pulse!(ax)
    for (case, color) in zip(tiled_cases, frac_colors)
        lines!(ax, case.t, transform(getproperty(case, field)); color, label = case.label)
    end
    axislegend(ax; position = legend, labelsize = 11)
    return ax
end

frac_panel(fig[1, 1], :LE,   "Latent heat (blended)",        "LE (W m⁻²)")
frac_panel(fig[1, 2], :H,    "Sensible heat (blended)",      "H (W m⁻²)")
frac_panel(fig[2, 1], :effective_temperature, "Land-surface temperature",     "effective_temperature (K)")
frac_panel(fig[2, 2], :Ewet, "Wet-canopy evaporation Eʷᵉᵗ", "Eʷᵉᵗ (mm day⁻¹)"; transform = mm_per_day)
frac_panel(fig[3, 1], :Wᶜ,   "Canopy water store Wᶜ",         "Wᶜ (mm)"; legend = :rt)
ax = frac_panel(fig[3, 2], :𝒮, "Shared-column saturation 𝒮",  "𝒮"; legend = :rc)
hlines!(ax, [critical_saturation]; color = :gray, linestyle = :dash)
ylims!(ax, 0, 1.05)

Label(fig[0, 1:2], "Tiled canopy/bare mosaic with interception — sweeping the vegetation fraction", fontsize = 22)

save("canopy_air_space_tiled_interception.png", fig)
@info "Saved canopy_air_space_tiled_interception.png"

nothing #hide

# ![](canopy_air_space_tiled_interception.png)

# ## Figure 4 — anatomy of the interception store (full canopy, f_veg = 1)
#
# The store buffers the rain and opens a third latent pathway. During the pulse the canopy
# catches part of the rain and fills `Wᶜ`; the wet leaf then evaporates at the potential
# rate, so wet-canopy evaporation spikes while the store drains.

ref   = full_canopy
Wᶜᵐᵃˣ = interception_capacity * vegetated_leaf_area_index
wet_LE           = ref.LEwet             # wet-canopy latent heat ℒ·Eʷᵉᵗ (W m⁻²)
transpiration_LE = ref.LEˡᵉᵃᶠ .- ref.LEwet  # dry (stomatal) leaf latent

fig = Figure(size = (1500, 950), fontsize = 17)

## (1,1) Rain partition — the canopy intercepts part of the pulse; the rest is throughfall.
ax = Axis(fig[1, 1]; title = "Rain partition", xlabel = "t (days)", ylabel = "flux (mm day⁻¹)")
mark_pulse!(ax)
lines!(ax, ref.t, mm_per_day(ref.P);  color = :skyblue,     linewidth = 3, label = "incident rain")
lines!(ax, ref.t, mm_per_day(ref.Pˡ); color = :saddlebrown, label = "throughfall → soil")
axislegend(ax; position = :lt, labelsize = 12)

## (1,2) The store filling during the pulse and draining after.
ax = Axis(fig[1, 2]; title = "Canopy water store Wᶜ", xlabel = "t (days)", ylabel = "Wᶜ (mm)")
mark_pulse!(ax)
lines!(ax, ref.t, ref.Wᶜ; color = :navy, linewidth = 2, label = "Wᶜ")
hlines!(ax, [Wᶜᵐᵃˣ]; color = :gray, linestyle = :dash, label = "capacity Wᶜᵐᵃˣ = c·LAI")
axislegend(ax; position = :rt, labelsize = 12)

## (2,1) The three latent pathways — wet-canopy evaporation joins transpiration and soil
## evaporation while the canopy is wet.
ax = Axis(fig[2, 1]; title = "Latent-heat pathways", xlabel = "t (days)", ylabel = "flux (W m⁻²)")
mark_pulse!(ax)
lines!(ax, ref.t, ref.LE;           color = :navy,        linewidth = 2, label = "total LE")
lines!(ax, ref.t, transpiration_LE; color = :seagreen,    label = "transpiration")
lines!(ax, ref.t, ref.LEᵍ;          color = :saddlebrown, label = "soil evaporation")
lines!(ax, ref.t, wet_LE;           color = :steelblue,   label = "wet-canopy evaporation")
axislegend(ax; position = :lt, labelsize = 12)

## (2,2) The Deardorff wet fraction fʷᵉᵗ = (Wᶜ/Wᶜᵐᵃˣ)^(2/3) driving the wet/dry leaf blend.
fʷᵉᵗ = @. clamp((max(ref.Wᶜ, 0) / Wᶜᵐᵃˣ)^(2/3), 0, 1)
ax = Axis(fig[2, 2]; title = "Wet-canopy fraction fʷᵉᵗ", xlabel = "t (days)", ylabel = "fʷᵉᵗ")
mark_pulse!(ax)
lines!(ax, ref.t, fʷᵉᵗ; color = :steelblue, linewidth = 2)
ylims!(ax, 0, 1.05)

Label(fig[0, 1:2], "Interception store anatomy — full canopy (f_veg = 1)", fontsize = 22)

save("canopy_air_space_interception.png", fig)
@info "Saved canopy_air_space_interception.png"

nothing #hide

# ![](canopy_air_space_interception.png)
