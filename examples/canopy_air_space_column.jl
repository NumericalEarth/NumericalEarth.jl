# # A canopy-air-space column: two-source vegetation, transpiration, and a rain pulse
#
# This idealized 0D example exercises every piece of the [`CanopyAirSpace`](@ref)
# (CAS) land-surface component under an analytic diurnal forcing with a **rain pulse**
# dropped after the first day, so we can watch the surface energy and water budgets
# *before*, *during*, and *after* rain. A vegetated CAS column carries *four*
# temperatures — the atmosphere never talks to a leaf or the soil directly, only to a
# massless **canopy-air node** `(Tᵃᶜ, qᵃᶜ)` that both surfaces feed and that drains to
# the atmosphere by Monin–Obukhov turbulence:
#
#     Tᵛ   leaf         (diagnostic, massless: Rₙᵛ = Hᵛ + LEᵛ)
#     Tⁱⁿ  soil skin    (diagnostic: Rₙᵍ = Hᵍ + LEᵍ + Λⁱⁿ(Tⁱⁿ − Tˡᵃ))
#     Tᵃᶜ  canopy air    (diagnostic node; what the atmosphere sees)
#     Tˡᵃ  bulk slab     (prognostic; driven by the skin→bulk conduction Λⁱⁿ)
#
# The canopy shades the soil (Beer–Lambert shortwave split), the leaf transpires
# (photosynthesis-coupled stomata), the soil evaporates through a dry surface layer
# ([`DryLayerHumidity`](@ref)), and the two surfaces exchange longwave through a
# two-face ledger. Rain refills the soil bucket, lifting the surface saturation `𝒮`
# that gates both transpiration (moisture stress `β(𝒮)`) and soil evaporation
# (dry-layer depth `δᵛ(𝒮)`).
#
# We turn **two knobs**:
#
# 1. **Leaf-area index** — a bare surface (`LAI = 0`, the smooth bare-ground limit),
#    a moderate canopy (`LAI = 2`), and a dense one (`LAI = 4`).
# 2. **Stomatal conductance** — the photosynthesis-coupled [`MedlynConductance`](@ref)
#    (2011) versus the empirical multiplicative [`JarvisConductance`](@ref) (1976).

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
specific_humidity        = 0.008                                      # kg kg⁻¹
surface_pressure         = 101325                                     # Pa

rain_rate = 1e-3                                                      # kg m⁻² s⁻¹ (≈ 43 mm over 12 h)
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
# The Beer–Lambert / two-face-longwave optics are shared by the model and by the
# post-run radiation partition below, so we name them once. `CanopyAirSpace` reads the
# *downwelling* shortwave and longwave directly (its radiation is internalized), so
# these leaf/ground albedos and emissivities — not the prescribed-radiation surface
# properties — set the canopy's radiative budget.

leaf_albedo           = 0.15
ground_albedo         = 0.15
canopy_emissivity_max = 0.98
ground_emissivity     = 0.96
extinction            = 0.5   # Beer–Lambert K
clumping              = 1.0   # foliage clumping Ω
stefan_boltzmann      = 5.670374419e-8

# The soil vapor branch is a `DryLayerHumidity`: a thin dry surface layer that grows as
# the soil dries past its onset saturation, throttling bare-soil evaporation.

soil_branch() = DryLayerHumidity(;
    dry_layer_depth = StorageBasedDryLayerDepth(maximum_dry_layer_depth    = 0.015,
                                                dry_layer_onset_saturation = 0.5,
                                                dry_layer_exponent         = 2),
    vapor_exchange  = DryLayerVaporPistonVelocity(minimum_dry_layer_depth = 1e-3,
                                                  molecular_diffusivity   = 2.4e-5,
                                                  tortuosity              = ConstantTortuosity()),
    thermal_exchange_depth = 0.05, porosity = 0.4)

# A shallow bucket so the drydown after the pulse plays out within the run. We start
# just above the critical saturation `𝒮ᶜ = 0.5` (where moisture stress sets in) so the
# column begins unstressed, dries toward stress, and the pulse resets it.

maximum_water_storage = 50.0   # kg m⁻²
initial_water_storage = 30.0   # kg m⁻² → 𝒮 = 0.6
initial_temperature   = 286.0  # K

# ## Model builder
#
# Both interface slots take the *same* `CanopyAirSpace` object (it is a combined
# temperature + humidity formulation). The bulk slab uses a pure `SlabEnergy` budget
# and a `BucketHydrology` water store. The two knobs enter here: `leaf_area_index`
# upscales the leaf conductance and sets the shading, and `conductance` selects the
# stomatal model.

function canopy_air_space_column(; leaf_area_index, conductance, label)
    grid       = RectilinearGrid(CPU(); size = (), topology = (Flat, Flat, Flat))
    atmosphere = forced_atmosphere(grid, times)
    radiation  = forced_radiation(grid, times)

    land = SlabLand(grid; energy = SlabEnergy(),
                    hydrology = BucketHydrology(; maximum_water_storage))
    set!(land; T = initial_temperature, M = initial_water_storage)

    canopy_air_space = CanopyAirSpace(;
        soil   = soil_branch(),
        canopy = CanopyConductanceHumidity(; leaf_area_index, conductance,
                                           moisture_stress = CriticalSaturation(0.5),
                                           absorbed_par    = InteractiveAbsorbedPAR()),
        soil_skin_flux = SoilConductiveFlux(1.5, 0.05),
        leaf_albedo, ground_albedo, canopy_emissivity_max, ground_emissivity,
        extinction, clumping)

    model = AtmosphereLandModel(atmosphere, land; radiation,
                                atmosphere_land_interface_temperature       = canopy_air_space,
                                atmosphere_land_interface_specific_humidity = canopy_air_space)

    ## Diagnostics: the CAS stores its node/leaf/soil-skin/effective temperatures and
    ## the skin→bulk ground heat flux as a NamedTuple in `interface.temperature`; the
    ## atmosphere-facing turbulent totals live in `interface.fluxes`; the water fluxes
    ## in `land.fluxes`.
    interface = model.interfaces.atmosphere_land_interface

    T   = zeros(Nsteps)   # time (days)
    Tₐ  = zeros(Nsteps)   # air (forcing)
    Tᵛ  = zeros(Nsteps)   # leaf
    Tᵃᶜ = zeros(Nsteps)   # canopy-air node
    Tⁱⁿ = zeros(Nsteps)   # soil skin
    Tˡᵃ = zeros(Nsteps)   # bulk slab
    Tₑ  = zeros(Nsteps)   # effective radiating (LST)
    H   = zeros(Nsteps)   # sensible heat, atmosphere total (positive up)
    LE  = zeros(Nsteps)   # latent heat, atmosphere total (positive up)
    LEᵛ = zeros(Nsteps)   # transpiration — leaf latent share (positive up)
    LEᵍ = zeros(Nsteps)   # soil evaporation — ground latent share (positive up)
    Hᵛ  = zeros(Nsteps)   # leaf sensible share
    Hᵍ  = zeros(Nsteps)   # ground sensible share
    Gᶜ  = zeros(Nsteps)   # skin → bulk ground heat flux
    SW  = zeros(Nsteps)   # incident shortwave
    LW  = zeros(Nsteps)   # incident (downwelling) longwave
    E   = zeros(Nsteps)   # evaporation (soil + transpiration, kg m⁻² s⁻¹, positive up)
    P   = zeros(Nsteps)   # precipitation (kg m⁻² s⁻¹, positive down)
    𝒮   = zeros(Nsteps)   # surface saturation
    M   = zeros(Nsteps)   # water storage (kg m⁻²)

    @info "Running canopy-air-space column: $label ..."
    for n in 1:Nsteps
        time_step!(model, Δt)
        Ts   = interface.temperature
        time = model.clock.time
        T[n]   = time / day
        Tₐ[n]  = air_temperature(time)
        Tᵛ[n]  = scalar(Ts.canopy)
        Tᵃᶜ[n] = scalar(Ts.interface)
        Tⁱⁿ[n] = scalar(Ts.soil_skin)
        Tˡᵃ[n] = scalar(land.temperature)
        Tₑ[n]  = scalar(Ts.effective)
        H[n]   = scalar(interface.fluxes.sensible_heat)
        LE[n]  = scalar(interface.fluxes.latent_heat)
        LEᵛ[n] = scalar(Ts.canopy_latent_heat)
        LEᵍ[n] = scalar(Ts.soil_latent_heat)
        Hᵛ[n]  = scalar(Ts.canopy_sensible_heat)
        Hᵍ[n]  = scalar(Ts.soil_sensible_heat)
        Gᶜ[n]  = scalar(Ts.ground_heat_flux)
        SW[n]  = downwelling_shortwave(time)
        LW[n]  = downwelling_longwave(time)
        E[n]   = scalar(land.fluxes.evaporation)
        P[n]   = scalar(land.fluxes.precipitation)
        𝒮[n]   = scalar(land.saturation)
        M[n]   = scalar(land.water_storage)
    end

    return (; label, leaf_area_index,
              t = T, Tₐ, Tᵛ, Tᵃᶜ, Tⁱⁿ, Tˡᵃ, Tₑ, H, LE, LEᵛ, LEᵍ, Hᵛ, Hᵍ, Gᶜ, SW, LW, E, P, 𝒮, M)
end

scalar(field) = first(interior(field))

# ## Integration length and the case set
#
# Six days at `Δt = 10 min`, one full diurnal cycle before the pulse and three-plus
# days of drydown after it.

Δt     = 10minutes
Nsteps = 6 * 144                     # 6 days
times  = range(0, Nsteps * Δt, step = 1hour)

# The bare/moderate/dense sweep runs Medlyn; the dense case is repeated with Jarvis so
# the two conductance models can be compared at fixed LAI.

medlyn = MedlynConductance()
jarvis = JarvisConductance()

bare     = canopy_air_space_column(leaf_area_index = 0.0, conductance = medlyn, label = "bare (LAI 0)")
moderate = canopy_air_space_column(leaf_area_index = 2.0, conductance = medlyn, label = "moderate (LAI 2)")
dense    = canopy_air_space_column(leaf_area_index = 4.0, conductance = medlyn, label = "dense (LAI 4)")
dense_j  = canopy_air_space_column(leaf_area_index = 4.0, conductance = jarvis, label = "dense, Jarvis")

# ## Radiation partition (reconstructed from the stored surface temperatures)
#
# The CAS keeps only the surface *temperatures* and the turbulent *totals*; the
# radiative shares are diagnostic functions of those, so we rebuild them here with the
# exact solve formulas — the Beer–Lambert shortwave split and the two-face longwave
# ledger (ClimaLand D13–D17) — using the leaf `Tᵛ` and soil-skin `Tⁱⁿ` each column stored.

function radiation_partition(case)
    LAI = case.leaf_area_index
    σ   = stefan_boltzmann
    ε_c = canopy_emissivity_max * (1 - exp(-LAI))
    ε_g = ground_emissivity
    ftrans = exp(-extinction * LAI * clumping)

    SW, LW, Tᵛ, Tⁱⁿ = case.SW, case.LW, case.Tᵛ, case.Tⁱⁿ

    ## Shortwave: the canopy absorbs `(1−α_leaf)(1−f_trans)`, the shaded soil gets the
    ## transmitted remainder `f_trans(1−α_ground)`.
    canopy_SW = @. (1 - leaf_albedo) * (1 - ftrans) * SW
    ground_SW = @. ftrans * (1 - ground_albedo) * SW

    ## Longwave: below-canopy downwelling (reaching the soil), the canopy's own
    ## downward emission, the upwelling above the soil, and the total escaping to space.
    LWd_c      = @. (1 - ε_c) * LW + ε_c * σ * Tᵛ^4     # reaching soil
    canopy_emit = @. ε_c * σ * Tᵛ^4                     # canopy emission (one face)
    LWu_g      = @. ε_g * σ * Tⁱⁿ^4 + (1 - ε_g) * LWd_c # upwelling from ground
    LWu        = @. (1 - ε_c) * LWu_g + ε_c * σ * Tᵛ^4  # to space

    ## Net radiation absorbed by the whole land column (canopy + soil).
    Rₙ = @. canopy_SW + ground_SW + LW - LWu

    return (; canopy_SW, ground_SW, LWd_c, canopy_emit, LWu_g, LWu, Rₙ)
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
# The full component breakdown for the dense (`LAI = 4`, Medlyn) column: how each
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
lines!(ax, t, ref.Tᵛ;  color = :seagreen,    label = "leaf Tᵛ")
lines!(ax, t, ref.Tⁱⁿ; color = :saddlebrown, label = "soil skin Tⁱⁿ")
lines!(ax, t, ref.Tᵃᶜ; color = :steelblue,   linestyle = :dash, label = "canopy air Tᵃᶜ")
lines!(ax, t, ref.Tˡᵃ; color = :firebrick,   label = "bulk Tˡᵃ")
lines!(ax, t, ref.Tₑ;  color = :black,       linestyle = :dot, label = "LST Teff")
axislegend(ax; position = :lt, nbanks = 2, labelsize = 12)

## (1,2) Two-source temperature contrasts.
ax = Axis(fig[1, 2]; title = "Source temperature contrasts", xlabel = "t (days)", ylabel = "ΔT (K)")
mark_pulse!(ax)
hlines!(ax, [0]; color = :gray, linestyle = :dash)
lines!(ax, t, ref.Tᵛ  .- ref.Tⁱⁿ; color = :seagreen,    label = "leaf − soil skin")
lines!(ax, t, ref.Tᵃᶜ .- ref.Tₐ;  color = :steelblue,   label = "node − air")
lines!(ax, t, ref.Tⁱⁿ .- ref.Tˡᵃ; color = :saddlebrown, label = "soil skin − bulk")
axislegend(ax; position = :lt, labelsize = 12)

## (2,1) Shortwave partition — the canopy shades the soil.
ax = Axis(fig[2, 1]; title = "Downwelling shortwave partition", xlabel = "t (days)", ylabel = "SW (W m⁻²)")
mark_pulse!(ax)
lines!(ax, t, ref.SW;        color = :goldenrod, linewidth = 3, label = "incident on land")
lines!(ax, t, rad.canopy_SW; color = :seagreen,    label = "absorbed by canopy")
lines!(ax, t, rad.ground_SW; color = :saddlebrown, label = "reaching soil")
axislegend(ax; position = :lt, labelsize = 12)

## (2,2) Longwave down partition.
ax = Axis(fig[2, 2]; title = "Downwelling longwave partition", xlabel = "t (days)", ylabel = "LW (W m⁻²)")
mark_pulse!(ax)
lines!(ax, t, ref.LW;         color = :black,     linewidth = 3, label = "incident on land")
lines!(ax, t, rad.LWd_c;      color = :saddlebrown, label = "reaching soil")
lines!(ax, t, rad.canopy_emit; color = :seagreen,  label = "emitted down by canopy")
axislegend(ax; position = :lt, labelsize = 12)

## (3,1) Longwave up partition.
ax = Axis(fig[3, 1]; title = "Upwelling longwave partition", xlabel = "t (days)", ylabel = "LW (W m⁻²)")
mark_pulse!(ax)
lines!(ax, t, rad.LWu_g;      color = :saddlebrown, label = "from ground")
lines!(ax, t, rad.canopy_emit; color = :seagreen,   label = "from canopy")
lines!(ax, t, rad.LWu;        color = :black,        linewidth = 2, label = "to space (LST)")
axislegend(ax; position = :lt, labelsize = 12)

## (3,2) Surface energy balance — net radiation into the land splits into turbulent
## sensible + latent to the atmosphere and conduction into the slab.
ax = Axis(fig[3, 2]; title = "Surface energy fluxes\n(positive: surface → atmosphere; Gᶜ into slab)",
          xlabel = "t (days)", ylabel = "flux (W m⁻²)")
mark_pulse!(ax)
hlines!(ax, [0]; color = :gray, linestyle = :dash)
lines!(ax, t, rad.Rₙ;  color = :black,     linewidth = 2, label = "net radiation Rₙ")
lines!(ax, t, ref.LE;  color = :navy,      label = "latent H₂O (LE)")
lines!(ax, t, ref.H;   color = :orange,    label = "sensible (H)")
lines!(ax, t, ref.Gᶜ;  color = :seagreen,  linestyle = :dash, label = "ground heat Gᶜ")
axislegend(ax; position = :lt, labelsize = 12)

## (4,1) Two-source latent split — the atmosphere feels only the total LE, but the CAS
## resolves it into leaf transpiration and soil evaporation. In a dense canopy
## transpiration dominates; soil evaporation is small (the soil is shaded and its front
## recedes as it dries).
ax = Axis(fig[4, 1]; title = "Two-source latent heat", xlabel = "t (days)", ylabel = "LE (W m⁻²)")
mark_pulse!(ax)
lines!(ax, t, ref.LE;  color = :navy,        linewidth = 2, label = "total LE")
lines!(ax, t, ref.LEᵛ; color = :seagreen,    label = "transpiration (leaf)")
lines!(ax, t, ref.LEᵍ; color = :saddlebrown, label = "soil evaporation (ground)")
axislegend(ax; position = :lt, labelsize = 12)

## (4,2) Two-source sensible split — leaf and under-canopy ground shares of the sensible
## flux the atmosphere feels.
ax = Axis(fig[4, 2]; title = "Two-source sensible heat", xlabel = "t (days)", ylabel = "H (W m⁻²)")
mark_pulse!(ax)
hlines!(ax, [0]; color = :gray, linestyle = :dash)
lines!(ax, t, ref.H;  color = :orange,      linewidth = 2, label = "total H")
lines!(ax, t, ref.Hᵛ; color = :seagreen,    label = "leaf")
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
## 𝒮 at 1, then evaporation draws it back down through the moisture-stress onset 𝒮ᶜ.
ax = Axis(fig[5, 2]; title = "Water state", xlabel = "t (days)",
          ylabel = "saturation 𝒮", ylabelcolor = :darkorange, yticklabelcolor = :darkorange)
mark_pulse!(ax)
lines!(ax, t, ref.𝒮; color = :darkorange, linewidth = 2, label = "𝒮")
hlines!(ax, [0.5]; color = :gray, linestyle = :dash, label = "moisture-stress onset 𝒮ᶜ")
ylims!(ax, 0, 1.05)
axislegend(ax; position = :rc, labelsize = 12)
axM = twin_axis(fig[5, 2]; ylabel = "storage (kg m⁻²)", color = :navy)
linkxaxes!(ax, axM)
lines!(axM, t, ref.M; color = :navy, linestyle = :dot)
ylims!(axM, 0, 1.05 * maximum_water_storage)

Label(fig[0, 1:2], "Canopy-air-space column — dense canopy (LAI 4, Medlyn), rain pulse on day 2", fontsize = 22)

save("canopy_air_space_column.png", fig)
@info "Saved canopy_air_space_column.png"

nothing #hide

# ![](canopy_air_space_column.png)

# ## Figure 2 — turning the two knobs
#
# The left column sweeps leaf-area index (bare → dense, all Medlyn); the right column
# fixes `LAI = 4` and swaps Medlyn for Jarvis stomata. The rows follow the two-source
# partition: leaf temperature, transpiration (the leaf latent share), soil evaporation
# (the ground latent share), and the surface saturation the drydown reads out. Raising
# LAI shifts the latent flux from bare-soil evaporation toward transpiration — the total
# stays energy-limited while its *sources* trade places — and Jarvis transpires
# differently from Medlyn at the same LAI.

lai_cases  = (bare, moderate, dense)
lai_colors = (:saddlebrown, :seagreen, :darkgreen)
con_cases  = (dense, dense_j)
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

knob_panel(fig[1, 1], lai_cases, lai_colors, :Tᵛ, "Leaf temperature — LAI sweep", "Tᵛ (K)")
knob_panel(fig[1, 2], con_cases, con_colors, :Tᵛ, "Leaf temperature — Medlyn vs Jarvis", "Tᵛ (K)")

knob_panel(fig[2, 1], lai_cases, lai_colors, :LEᵛ, "Transpiration — LAI sweep", "LEᵛ (W m⁻²)")
knob_panel(fig[2, 2], con_cases, con_colors, :LEᵛ, "Transpiration — Medlyn vs Jarvis", "LEᵛ (W m⁻²)")

knob_panel(fig[3, 1], lai_cases, lai_colors, :LEᵍ, "Soil evaporation — LAI sweep", "LEᵍ (W m⁻²)")
knob_panel(fig[3, 2], con_cases, con_colors, :LEᵍ, "Soil evaporation — Medlyn vs Jarvis", "LEᵍ (W m⁻²)")

ax = knob_panel(fig[4, 1], lai_cases, lai_colors, :𝒮, "Surface saturation — LAI sweep", "𝒮"; legend = :rc)
hlines!(ax, [0.5]; color = :gray, linestyle = :dash)
ylims!(ax, 0, 1.05)
ax = knob_panel(fig[4, 2], con_cases, con_colors, :𝒮, "Surface saturation — Medlyn vs Jarvis", "𝒮"; legend = :rc)
hlines!(ax, [0.5]; color = :gray, linestyle = :dash)
ylims!(ax, 0, 1.05)

Label(fig[0, 1:2], "Canopy-air-space column — turning the LAI and stomatal-conductance knobs", fontsize = 22)

save("canopy_air_space_knobs.png", fig)
@info "Saved canopy_air_space_knobs.png"

nothing #hide

# ![](canopy_air_space_knobs.png)
