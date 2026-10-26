# # A canopy-air-space column: two-source vegetation, transpiration, and a rain pulse
#
# This idealized 0D example exercises the [`CanopyGroundSurface`](@ref) land-surface
# component under an analytic diurnal forcing with a rain pulse dropped on day 2.
# It sweeps leaf-area index and stomatal conductance, then looks inside the canopy
# interception store.

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

leaf_albedo               = 0.15
ground_albedo             = 0.15
maximum_canopy_emissivity = 0.98
ground_emissivity         = 0.96
extinction                = 0.5   # Beer–Lambert K
clumping                  = 1.0   # foliage clumping Ω
stefan_boltzmann_constant = 5.670374419e-8

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
                                                  tortuosity              = ConstantTortuosity()))

# The soil is a `VariablySaturatedHydrology` column wrapped by an `InterceptingHydrology`
# canopy interception store, stepped with the conservative `WaterCoupledEnergy`. Every
# column below shares this land, starting dry (below `𝒮ᶜ`).

vegetated_leaf_area_index   = 4.0
soil_slab_depth             = 0.1
soil_porosity               = 0.4
soil_residual_fraction      = 0.05
liquid_density              = 1000.0
inverse_air_entry_head      = 1.0     # van Genuchten inverse air-entry (m⁻¹)
pore_size_uniformity        = 2.0     # van Genuchten pore-size shape
soil_retention              = VanGenuchtenRetention(; inverse_air_entry_head, pore_size_uniformity)
matching_point_conductivity = 1e-6    # K_sat (m s⁻¹)
interception_capacity       = 0.1     # c — canopy water capacity per unit LAI (kg m⁻², BATS/CLM)

initial_temperature     = 286.0
initial_soil_saturation = 0.3
initial_water_storage   = (initial_soil_saturation * (soil_porosity - soil_residual_fraction) +
                           soil_residual_fraction) * liquid_density * soil_slab_depth

variably_saturated_soil() = VariablySaturatedHydrology(;
    slab_depth = soil_slab_depth, porosity = soil_porosity,
    residual_liquid_fraction = soil_residual_fraction, storage_height = 1000,
    retention_curve        = soil_retention,
    hydraulic_conductivity = VanGenuchtenConductivity(; matching_point_conductivity, pore_size_uniformity),
    deep_liquid_flux       = NoDeepLiquidFlux(),
    runoff                 = InfiltrationCapacityRunoff(infiltration_capacity = 5e-4))

conservative_energy() = WaterCoupledEnergy(; deep_temperature = initial_temperature, deep_time_scale = 10day)

# The `InterceptingHydrology` wraps the soil with the canopy store; its `leaf_area_index`
# sets the caught fraction and capacity `Mᶜ⁺ = c·LAI`.
intercepting_hydrology() = InterceptingHydrology(;
    soil = variably_saturated_soil(),
    capacity_per_leaf_area = interception_capacity)

# ## Model builder
#
# The land surface uses a `CanopyGroundSurface`; over the intercepting hydrology its wet
# leaf evaporates intercepted rain. Every column shares one soil branch, moisture stress,
# PAR, skin flux, and optics; only the leaf-area index and stomatal `conductance` vary.

canopy_with_interception(; leaf_area_index, conductance = JarvisConductance()) = CanopyGroundSurface(;
    soil_humidity   = soil_branch(),
    canopy_humidity = CanopyConductanceHumidity(; leaf_area_index, conductance,
                                                moisture_stress = PlantAvailableWaterStress(),
                                                absorbed_par    = InteractiveAbsorbedPAR()),
    ground_heat_flux = SoilConductiveFlux(1.5, 0.05),
    leaf_albedo, ground_albedo, maximum_canopy_emissivity, ground_emissivity,
    extinction, clumping)

# The land is the shared variably-saturated soil with the interception store; the two knobs
# enter here — `leaf_area_index` upscales the leaf conductance and sets the shading, and
# `conductance` selects the stomatal model.

function canopy_air_space_column(; leaf_area_index, conductance, label)
    grid       = RectilinearGrid(CPU(); size = (), topology = (Flat, Flat, Flat))
    atmosphere = forced_atmosphere(grid, times)
    radiation  = forced_radiation(grid, times)

    canopy_air_space = canopy_with_interception(; leaf_area_index, conductance)
    land = SlabLand(grid; energy = conservative_energy(),
                    hydrology = intercepting_hydrology(),
                    surface = LandSurface(canopy_air_space))
    set!(land; T = initial_temperature, M = initial_water_storage, canopy_water_storage = 0.0)

    model = AtmosphereLandModel(atmosphere, land; radiation)

    ## Diagnostics: the surface stores its node/leaf/ground/effective temperatures, the
    ## skin→bulk ground heat flux, and the two-source latent/sensible shares plus the
    ## wet-canopy evaporation in `interface.temperature`; the atmosphere-facing turbulent
    ## totals live in `interface.fluxes`; the water fluxes/stores in `land`.
    interface = model.interfaces.atmosphere_land_interface

    t                      = zeros(number_of_steps)   # time (days)
    Tᵃᵗ                    = zeros(number_of_steps)   # air (forcing)
    Tˡᵉᵃᶠ                  = zeros(number_of_steps)   # leaf
    Tᵃᶜ                    = zeros(number_of_steps)   # canopy-air node
    Tᵍ                     = zeros(number_of_steps)   # ground
    Tˡᵃ                    = zeros(number_of_steps)   # bulk slab
    Tᵉᶠᶠ                   = zeros(number_of_steps)   # effective radiating (LST)
    𝒬ᵀ                     = zeros(number_of_steps)   # sensible heat, atmosphere total (positive up)
    𝒬ᵛ                     = zeros(number_of_steps)   # latent heat, atmosphere total (positive up)
    𝒬ᵛˡᵉᵃᶠ                 = zeros(number_of_steps)   # leaf latent (transpiration + wet-canopy, positive up)
    𝒬ᵛᵍ                    = zeros(number_of_steps)   # soil evaporation — ground latent share (positive up)
    𝒬ᵀˡᵉᵃᶠ                 = zeros(number_of_steps)   # leaf sensible share
    𝒬ᵀᵍ                    = zeros(number_of_steps)   # ground sensible share
    𝒬ᵍ                     = zeros(number_of_steps)   # skin → bulk ground heat flux
    incident_shortwave     = zeros(number_of_steps)   # incident shortwave
    incident_longwave      = zeros(number_of_steps)   # incident longwave
    Jᵛ                     = zeros(number_of_steps)   # soil + transpiration vapor flux (kg m⁻² s⁻¹, up)
    Jʷᵉᵗ                   = zeros(number_of_steps)   # wet-canopy evaporation (kg m⁻² s⁻¹, up)
    wet_canopy_latent_heat = zeros(number_of_steps)   # wet-canopy latent heat ℒ·Jʷᵉᵗ (W m⁻², up)
    rain                   = zeros(number_of_steps)   # incident rain (kg m⁻² s⁻¹, down)
    throughfall            = zeros(number_of_steps)   # throughfall reaching the soil (kg m⁻² s⁻¹, down)
    𝒮                      = zeros(number_of_steps)   # surface saturation
    M                      = zeros(number_of_steps)   # soil water storage (kg m⁻²)
    Mᶜ                     = zeros(number_of_steps)   # canopy water store (mm)

    @info "Running canopy-air-space column: $label ..."
    for n in 1:number_of_steps
        Tᵃᵗ[n]                   = scalar(model.interfaces.exchanger.atmosphere.state.T)
        incident_shortwave[n] = -scalar(model.interfaces.exchanger.radiation.state.ℐꜜˢʷ)
        incident_longwave[n]  = -scalar(model.interfaces.exchanger.radiation.state.ℐꜜˡʷ)
        time_step!(model, Δt)
        interface_temperature = interface.temperature
        time                  = model.clock.time
        t[n]                      = time / day
        Tˡᵉᵃᶠ[n]                  = scalar(interface_temperature.canopy)
        Tᵃᶜ[n]                    = scalar(interface_temperature.interface)
        Tᵍ[n]                     = scalar(interface_temperature.ground)
        Tˡᵃ[n]                    = scalar(land.temperature)
        Tᵉᶠᶠ[n]                   = scalar(interface_temperature.effective)
        𝒬ᵀ[n]                     = scalar(interface.fluxes.sensible_heat)
        𝒬ᵛ[n]                     = scalar(interface.fluxes.latent_heat)
        𝒬ᵛˡᵉᵃᶠ[n]                 = scalar(interface_temperature.canopy_latent_heat)
        𝒬ᵛᵍ[n]                    = scalar(interface_temperature.ground_latent_heat)
        𝒬ᵀˡᵉᵃᶠ[n]                 = scalar(interface_temperature.canopy_sensible_heat)
        𝒬ᵀᵍ[n]                    = scalar(interface_temperature.ground_sensible_heat)
        𝒬ᵍ[n]                     = scalar(interface_temperature.ground_heat_flux)
        Jᵛ[n]                     = scalar(land.fluxes.vapor_flux)
        Jʷᵉᵗ[n]                   = scalar(interface_temperature.canopy_evaporation)
        wet_canopy_latent_heat[n] = scalar(interface_temperature.wet_canopy_latent_heat)
        rain[n]                   = scalar(land.fluxes.liquid_precipitation_flux)
        throughfall[n]            = scalar(land.diagnostics.throughfall)
        𝒮[n]                      = scalar(land.saturation)
        M[n]                      = scalar(land.water_storage)
        Mᶜ[n]                     = scalar(land.prognostic.canopy_water_storage)
    end

    ## The leaf latent `𝒬ᵛˡᵉᵃᶠ` is transpiration + wet-canopy; the model reports the wet share
    ## `wet_canopy_latent_heat = ℒ·Jʷᵉᵗ` with the *same* `ℒ` as `𝒬ᵛˡᵉᵃᶠ`, so `transpiration`
    ## is the exact stomatal flux (and stays ≥ 0, unlike a split against a fixed reference `ℒ`).
    transpiration = 𝒬ᵛˡᵉᵃᶠ .- wet_canopy_latent_heat

    return (; label, leaf_area_index,
              t, Tᵃᵗ, Tˡᵉᵃᶠ, Tᵃᶜ, Tᵍ, Tˡᵃ, Tᵉᶠᶠ, 𝒬ᵀ, 𝒬ᵛ, 𝒬ᵛˡᵉᵃᶠ, transpiration, 𝒬ᵛᵍ, 𝒬ᵀˡᵉᵃᶠ, 𝒬ᵀᵍ, 𝒬ᵍ,
              incident_shortwave, incident_longwave,
              Jᵛ, Jʷᵉᵗ, wet_canopy_latent_heat, rain, throughfall, 𝒮, M, Mᶜ)
end

scalar(field) = first(interior(field))

# ## Integration length and the case set
#
# Six days at `Δt = 10 min`, one full diurnal cycle before the pulse and three-plus
# days of drydown after it.

Δt              = 10minutes
number_of_steps = 6 * 144                     # 6 days
times           = range(0, number_of_steps * Δt, step = 1hour)

# The bare/moderate/dense sweep uses the default Jarvis conductance; the dense case is
# repeated with Medlyn conductance at the same LAI.

medlyn = MedlynConductance()
jarvis = JarvisConductance()

bare         = canopy_air_space_column(leaf_area_index = 0.0, conductance = jarvis, label = "bare (LAI 0)")
moderate     = canopy_air_space_column(leaf_area_index = 2.0, conductance = jarvis, label = "moderate (LAI 2)")
dense        = canopy_air_space_column(leaf_area_index = 4.0, conductance = jarvis, label = "dense (LAI 4)")
dense_medlyn = canopy_air_space_column(leaf_area_index = 4.0, conductance = medlyn, label = "dense, Medlyn")

# ## Radiation partition (reconstructed from the stored surface temperatures)
#
# The interface keeps only the surface temperatures and turbulent totals, so the radiative
# shares are reconstructed here from the stored leaf `Tˡᵉᵃᶠ` and ground `Tᵍ`
# temperatures using the Beer–Lambert shortwave split and two-face longwave ledger.

function radiation_partition(case)
    LAI   = case.leaf_area_index
    σ     = stefan_boltzmann_constant
    ϵˡᵉᵃᶠ = maximum_canopy_emissivity * (1 - exp(-LAI))
    ϵᵍ    = ground_emissivity
    transmittance = exp(-extinction * LAI * clumping)

    incident_shortwave, incident_longwave, Tˡᵉᵃᶠ, Tᵍ =
        case.incident_shortwave, case.incident_longwave, case.Tˡᵉᵃᶠ, case.Tᵍ

    ## Shortwave: the canopy absorbs `(1−α_leaf)(1−f_trans)`, the shaded soil gets the
    ## transmitted remainder `f_trans(1−α_ground)`.
    leaf_absorbed_shortwave   = @. (1 - leaf_albedo) * (1 - transmittance) * incident_shortwave
    ground_absorbed_shortwave = @. transmittance * (1 - ground_albedo) * incident_shortwave

    ## Longwave: below-canopy downwelling (reaching the soil), the canopy's own
    ## downward emission, the upwelling above the soil, and the total escaping to space.
    ground_downwelling_longwave = @. (1 - ϵˡᵉᵃᶠ) * incident_longwave + ϵˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^4  # reaching soil
    canopy_emission             = @. ϵˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^4                                       # canopy emission (one face)
    ℐꜛˡʷᵍ                       = @. ϵᵍ * σ * Tᵍ^4 + (1 - ϵᵍ) * ground_downwelling_longwave  # upwelling from ground
    ℐꜛˡʷ                        = @. (1 - ϵˡᵉᵃᶠ) * ℐꜛˡʷᵍ + ϵˡᵉᵃᶠ * σ * Tˡᵉᵃᶠ^4                # to space

    ## Net radiation absorbed by the whole land column (canopy + soil).
    net_radiation = @. leaf_absorbed_shortwave + ground_absorbed_shortwave + incident_longwave - ℐꜛˡʷ

    return (; leaf_absorbed_shortwave, ground_absorbed_shortwave, ground_downwelling_longwave,
              canopy_emission, ℐꜛˡʷᵍ, ℐꜛˡʷ, net_radiation)
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

case      = dense
partition = radiation_partition(case)
t         = case.t

fig = Figure(size = (1700, 1850), fontsize = 17)

## (1,1) Temperatures — leaf leads and swings widest, shaded ground is damped,
## bulk slab lags, node sits between the surfaces and the air.
ax = Axis(fig[1, 1]; title = "Temperatures", xlabel = "t (days)", ylabel = "T (K)")
mark_pulse!(ax)
lines!(ax, t, case.Tᵃᵗ;   color = :gray,        linewidth = 3, label = "air Tᵃᵗ")
lines!(ax, t, case.Tˡᵉᵃᶠ; color = :seagreen,    label = "leaf Tˡᵉᵃᶠ")
lines!(ax, t, case.Tᵍ;    color = :saddlebrown, label = "ground Tᵍ")
lines!(ax, t, case.Tᵃᶜ;   color = :steelblue,   linestyle = :dash, label = "canopy air Tᵃᶜ")
lines!(ax, t, case.Tˡᵃ;   color = :firebrick,   label = "bulk Tˡᵃ")
lines!(ax, t, case.Tᵉᶠᶠ;  color = :black,       linestyle = :dot, label = "LST Tᵉᶠᶠ")
axislegend(ax; position = :lt, nbanks = 2, labelsize = 12)

## (1,2) Two-source temperature contrasts.
ax = Axis(fig[1, 2]; title = "Source temperature contrasts", xlabel = "t (days)", ylabel = "ΔT (K)")
mark_pulse!(ax)
hlines!(ax, [0]; color = :gray, linestyle = :dash)
lines!(ax, t, case.Tˡᵉᵃᶠ .- case.Tᵍ;  color = :seagreen,    label = "leaf − ground")
lines!(ax, t, case.Tᵃᶜ   .- case.Tᵃᵗ; color = :steelblue,   label = "node − air")
lines!(ax, t, case.Tᵍ    .- case.Tˡᵃ; color = :saddlebrown, label = "ground − bulk")
axislegend(ax; position = :lt, labelsize = 12)

## (2,1) Shortwave partition — the canopy shades the soil.
ax = Axis(fig[2, 1]; title = "Downwelling shortwave partition", xlabel = "t (days)", ylabel = "shortwave (W m⁻²)")
mark_pulse!(ax)
lines!(ax, t, case.incident_shortwave;             color = :goldenrod,   linewidth = 3, label = "incident on land")
lines!(ax, t, partition.leaf_absorbed_shortwave;   color = :seagreen,    label = "absorbed by canopy")
lines!(ax, t, partition.ground_absorbed_shortwave; color = :saddlebrown, label = "reaching soil")
axislegend(ax; position = :lt, labelsize = 12)

## (2,2) Longwave down partition.
ax = Axis(fig[2, 2]; title = "Downwelling longwave partition", xlabel = "t (days)", ylabel = "longwave (W m⁻²)")
mark_pulse!(ax)
lines!(ax, t, case.incident_longwave;                color = :black,       linewidth = 3, label = "incident on land")
lines!(ax, t, partition.ground_downwelling_longwave; color = :saddlebrown, label = "reaching soil")
lines!(ax, t, partition.canopy_emission;             color = :seagreen,    label = "emitted down by canopy")
axislegend(ax; position = :lt, labelsize = 12)

## (3,1) Longwave up partition.
ax = Axis(fig[3, 1]; title = "Upwelling longwave partition", xlabel = "t (days)", ylabel = "longwave (W m⁻²)")
mark_pulse!(ax)
lines!(ax, t, partition.ℐꜛˡʷᵍ;           color = :saddlebrown, label = "from ground")
lines!(ax, t, partition.canopy_emission; color = :seagreen,    label = "from canopy")
lines!(ax, t, partition.ℐꜛˡʷ;            color = :black,       linewidth = 2, label = "to space (LST)")
axislegend(ax; position = :lt, labelsize = 12)

## (3,2) Surface energy balance — net radiation into the land splits into turbulent
## sensible + latent to the atmosphere and conduction into the slab.
ax = Axis(fig[3, 2]; title = "Surface energy fluxes\n(positive: surface → atmosphere; 𝒬ᵍ into slab)",
          xlabel = "t (days)", ylabel = "flux (W m⁻²)")
mark_pulse!(ax)
hlines!(ax, [0]; color = :gray, linestyle = :dash)
lines!(ax, t, partition.net_radiation; color = :black,    linewidth = 2, label = "net radiation")
lines!(ax, t, case.𝒬ᵛ;                 color = :navy,     label = "latent H₂O (𝒬ᵛ)")
lines!(ax, t, case.𝒬ᵀ;                 color = :orange,   label = "sensible (𝒬ᵀ)")
lines!(ax, t, case.𝒬ᵍ;                 color = :seagreen, linestyle = :dash, label = "ground heat 𝒬ᵍ")
axislegend(ax; position = :lt, labelsize = 12)

## (4,1) Latent-heat pathways — the atmosphere feels only the total 𝒬ᵛ, but the surface
## resolves it into leaf transpiration, soil evaporation, and (while the canopy is wet)
## wet-canopy evaporation from the interception store.
ax = Axis(fig[4, 1]; title = "Latent-heat pathways", xlabel = "t (days)", ylabel = "𝒬ᵛ (W m⁻²)")
mark_pulse!(ax)
lines!(ax, t, case.𝒬ᵛ;                     color = :navy,        linewidth = 2, label = "total 𝒬ᵛ")
lines!(ax, t, case.transpiration;          color = :seagreen,    label = "transpiration")
lines!(ax, t, case.𝒬ᵛᵍ;                    color = :saddlebrown, label = "soil evaporation")
lines!(ax, t, case.wet_canopy_latent_heat; color = :steelblue,   label = "wet-canopy evaporation")
axislegend(ax; position = :lt, labelsize = 12)

## (4,2) Two-source sensible split — leaf and under-canopy ground shares of the sensible
## flux the atmosphere feels.
ax = Axis(fig[4, 2]; title = "Two-source sensible heat", xlabel = "t (days)", ylabel = "𝒬ᵀ (W m⁻²)")
mark_pulse!(ax)
hlines!(ax, [0]; color = :gray, linestyle = :dash)
lines!(ax, t, case.𝒬ᵀ;     color = :orange,      linewidth = 2, label = "total 𝒬ᵀ")
lines!(ax, t, case.𝒬ᵀˡᵉᵃᶠ; color = :seagreen,    label = "leaf")
lines!(ax, t, case.𝒬ᵀᵍ;    color = :saddlebrown, label = "ground")
axislegend(ax; position = :lt, labelsize = 12)

## (5,1) Moisture fluxes — evaporation (soil + transpiration) against the rain pulse,
## on a second axis (the pulse dwarfs the background ET).
ax = Axis(fig[5, 1]; title = "Moisture fluxes", xlabel = "t (days)",
          ylabel = "evaporation (mm day⁻¹)", ylabelcolor = :navy, yticklabelcolor = :navy)
mark_pulse!(ax)
lines!(ax, t, mm_per_day(case.Jᵛ); color = :navy, label = "evaporation")
axislegend(ax; position = :lt, labelsize = 12)
axP = twin_axis(fig[5, 1]; ylabel = "precipitation (mm day⁻¹)", color = :skyblue)
linkxaxes!(ax, axP)
lines!(axP, t, mm_per_day(case.rain); color = :skyblue, linewidth = 2)

## (5,2) Water state — surface saturation and the storage it tracks; the pulse pins
## 𝒮 at 1, then evaporation draws it back down through the dry-layer onset 𝒮ᶜ.
ax = Axis(fig[5, 2]; title = "Water state", xlabel = "t (days)",
          ylabel = "saturation 𝒮", ylabelcolor = :darkorange, yticklabelcolor = :darkorange)
mark_pulse!(ax)
lines!(ax, t, case.𝒮; color = :darkorange, linewidth = 2, label = "𝒮")
hlines!(ax, [critical_saturation]; color = :gray, linestyle = :dash, label = "dry-layer onset 𝒮ᶜ")
ylims!(ax, 0, 1.05)
axislegend(ax; position = :rc, labelsize = 12)
axM = twin_axis(fig[5, 2]; ylabel = "soil storage (kg m⁻²)", color = :navy)
linkxaxes!(ax, axM)
lines!(axM, t, case.M; color = :navy, linestyle = :dot)
ylims!(axM, 0, 1.05 * maximum(case.M))

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

leaf_area_index_cases  = (bare, moderate, dense)
leaf_area_index_colors = (:saddlebrown, :seagreen, :darkgreen)
conductance_cases      = (dense, dense_medlyn)
conductance_colors     = (:darkgreen, :purple)

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

knob_panel(fig[1, 1], leaf_area_index_cases, leaf_area_index_colors, :Tˡᵉᵃᶠ, "Leaf temperature — LAI sweep", "Tˡᵉᵃᶠ (K)")
knob_panel(fig[1, 2], conductance_cases, conductance_colors, :Tˡᵉᵃᶠ, "Leaf temperature — Jarvis vs Medlyn", "Tˡᵉᵃᶠ (K)")

knob_panel(fig[2, 1], leaf_area_index_cases, leaf_area_index_colors, :transpiration, "Transpiration — LAI sweep", "transpiration (W m⁻²)")
knob_panel(fig[2, 2], conductance_cases, conductance_colors, :transpiration, "Transpiration — Jarvis vs Medlyn", "transpiration (W m⁻²)")

knob_panel(fig[3, 1], leaf_area_index_cases, leaf_area_index_colors, :𝒬ᵛᵍ, "Soil evaporation — LAI sweep", "𝒬ᵛᵍ (W m⁻²)")
knob_panel(fig[3, 2], conductance_cases, conductance_colors, :𝒬ᵛᵍ, "Soil evaporation — Jarvis vs Medlyn", "𝒬ᵛᵍ (W m⁻²)")

ax = knob_panel(fig[4, 1], leaf_area_index_cases, leaf_area_index_colors, :𝒮, "Surface saturation — LAI sweep", "𝒮"; legend = :rc)
hlines!(ax, [critical_saturation]; color = :gray, linestyle = :dash)
ylims!(ax, 0, 1.05)
ax = knob_panel(fig[4, 2], conductance_cases, conductance_colors, :𝒮, "Surface saturation — Jarvis vs Medlyn", "𝒮"; legend = :rc)
hlines!(ax, [critical_saturation]; color = :gray, linestyle = :dash)
ylims!(ax, 0, 1.05)

Label(fig[0, 1:2], "Canopy-air-space column — turning the LAI and stomatal-conductance knobs", fontsize = 22)

save("canopy_air_space_knobs.png", fig)
@info "Saved canopy_air_space_knobs.png"

nothing #hide

# ![](canopy_air_space_knobs.png)

# ## Figure 3 — anatomy of the interception store
#
# The store buffers the rain and opens a third latent pathway. During the pulse the canopy
# catches part of the rain and fills `Mᶜ`; the wet leaf then evaporates at the potential
# rate, so wet-canopy evaporation spikes while the store drains.

case                      = dense
Mᶜ⁺                       = interception_capacity * vegetated_leaf_area_index
wet_canopy_latent_heat    = case.wet_canopy_latent_heat                 # ℒ·Jʷᵉᵗ (W m⁻²)
transpiration_latent_heat = case.𝒬ᵛˡᵉᵃᶠ .- case.wet_canopy_latent_heat  # dry (stomatal) leaf latent

fig = Figure(size = (1500, 950), fontsize = 17)

## (1,1) Rain partition — the canopy intercepts part of the pulse; the rest is throughfall.
ax = Axis(fig[1, 1]; title = "Rain partition", xlabel = "t (days)", ylabel = "flux (mm day⁻¹)")
mark_pulse!(ax)
lines!(ax, case.t, mm_per_day(case.rain);        color = :skyblue,     linewidth = 3, label = "incident rain")
lines!(ax, case.t, mm_per_day(case.throughfall); color = :saddlebrown, label = "throughfall → soil")
axislegend(ax; position = :lt, labelsize = 12)

## (1,2) The store filling during the pulse and draining after.
ax = Axis(fig[1, 2]; title = "Canopy water store Mᶜ", xlabel = "t (days)", ylabel = "Mᶜ (mm)")
mark_pulse!(ax)
lines!(ax, case.t, case.Mᶜ; color = :navy, linewidth = 2, label = "Mᶜ")
hlines!(ax, [Mᶜ⁺]; color = :gray, linestyle = :dash, label = "capacity Mᶜ⁺ = c·LAI")
axislegend(ax; position = :rt, labelsize = 12)

## (2,1) The three latent pathways — wet-canopy evaporation joins transpiration and soil
## evaporation while the canopy is wet.
ax = Axis(fig[2, 1]; title = "Latent-heat pathways", xlabel = "t (days)", ylabel = "flux (W m⁻²)")
mark_pulse!(ax)
lines!(ax, case.t, case.𝒬ᵛ;                  color = :navy,        linewidth = 2, label = "total 𝒬ᵛ")
lines!(ax, case.t, transpiration_latent_heat; color = :seagreen,    label = "transpiration")
lines!(ax, case.t, case.𝒬ᵛᵍ;                 color = :saddlebrown, label = "soil evaporation")
lines!(ax, case.t, wet_canopy_latent_heat;    color = :steelblue,   label = "wet-canopy evaporation")
axislegend(ax; position = :lt, labelsize = 12)

## (2,2) The Deardorff wet fraction fʷᵉᵗ = (Mᶜ/Mᶜ⁺)^(2/3) driving the wet/dry leaf blend.
fʷᵉᵗ = @. clamp((max(case.Mᶜ, 0) / Mᶜ⁺)^(2/3), 0, 1)
ax = Axis(fig[2, 2]; title = "Wet-canopy fraction fʷᵉᵗ", xlabel = "t (days)", ylabel = "fʷᵉᵗ")
mark_pulse!(ax)
lines!(ax, case.t, fʷᵉᵗ; color = :steelblue, linewidth = 2)
ylims!(ax, 0, 1.05)

Label(fig[0, 1:2], "Interception store anatomy — dense canopy (LAI = 4)", fontsize = 22)

save("canopy_air_space_interception.png", fig)
@info "Saved canopy_air_space_interception.png"

nothing #hide

# ![](canopy_air_space_interception.png)
