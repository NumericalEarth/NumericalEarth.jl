# # A tiled land column: mosaic of vegetation and bare soil
#
# At 100 m–1 km a land cell is rarely pure canopy or pure bare soil. A
# [`TiledLandInterface`](@ref) treats the cell as a **mosaic**: a vegetated fraction
# `f_veg` and a bare-soil fraction `1 − f_veg`, each running the *same*
# [`CanopyAirSpace`](@ref) surface solve independently against the *same* atmosphere
# (with its own roughness), then area-weighting the fluxes into the boundary condition
# the atmosphere and slab read,
#
#     𝒬 = f_veg · 𝒬_veg + (1 − f_veg) · 𝒬_bare.
#
# This is the SOTA parallel/tiled-flux scheme (Noah-MP, JULES, ClimaLand v1). The two
# tiles share one soil column (one `Tˡᵃ`, `Mˡᵃ`, `𝒮`); the bare tile is a canopy-free
# `CanopyAirSpace` (LAI = 0) derived automatically from the vegetated one, so both emit
# the same currency — internalized radiation and a conduction-driven slab.
#
# We do two things:
#
# 1. **Sweep `f_veg`** from bare (0) to full canopy (1) under a diurnal forcing with a
#    rain pulse, and watch the surface energy partition (Bowen ratio), the land-surface
#    temperature, and the drydown interpolate between the two endpoints.
# 2. **Decompose one mosaic** (`f_veg = 0.5`) into its vegetated and bare tile
#    contributions, and show the **roughness contrast** (forest z₀ ≫ bare z₀) each tile
#    carries — a first-order control on inland wind decay.

# ## Load packages
using NumericalEarth
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    SimilarityTheoryFluxes, atmosphere_land_stability_functions
using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: update_state!
using CairoMakie

# ## Idealized diurnal forcing with a rain pulse
#
# A warm, dry-ish regime with a strong daytime shortwave pulse; a 12-hour rain pulse
# falls on day 2 and refills the shared soil column.

day = 86400
air_temperature(t)       = 292 - 7 * cos(2π * t / day)                # K
downwelling_shortwave(t) = max(0, 850 * cos(2π * (t - day/2) / day))  # W m⁻², daytime only
downwelling_longwave(t)  = 340                                        # W m⁻²
wind_speed               = 5.0                                        # m s⁻¹
specific_humidity        = 0.008                                      # kg kg⁻¹
surface_pressure         = 101325                                     # Pa

rain_rate = 8e-4                                                      # kg m⁻² s⁻¹
in_rain_pulse(t) = 2day ≤ t ≤ 2.5day
rain_pulse(t)    = ifelse(in_rain_pulse(t), rain_rate, zero(rain_rate))

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

# ## The vegetated tile and the roughness contrast
#
# The vegetated tile is a full `CanopyAirSpace`: a transpiring canopy over its own shaded
# soil, with photosynthesis-coupled stomata and a dry-layer soil evaporation branch. The
# bare tile is derived from it (LAI = 0). We give the two tiles a strong roughness
# contrast — a rough forest canopy versus a smooth bare surface.

critical_saturation = 0.5
soil_porosity       = 0.4

soil_branch() = DryLayerHumidity(;
    dry_layer_depth = StorageBasedDryLayerDepth(maximum_dry_layer_depth    = 0.025,
                                                dry_layer_onset_saturation = critical_saturation,
                                                dry_layer_exponent         = 2),
    vapor_exchange  = DryLayerVaporPistonVelocity(minimum_dry_layer_depth = 1e-3,
                                                  molecular_diffusivity   = 2.4e-5,
                                                  tortuosity              = ConstantTortuosity()),
    thermal_exchange_depth = 0.05, porosity = soil_porosity)

# The canopy stress runs on plant-available water: `β` ramps from the permanent wilting
# point to field capacity, both read off the soil column's own retention curve below.
vegetated_tile() = CanopyAirSpace(;
    soil   = soil_branch(),
    canopy = CanopyConductanceHumidity(; leaf_area_index = 4.0,
                                       moisture_stress = PlantAvailableWaterStress(),
                                       absorbed_par    = InteractiveAbsorbedPAR()),
    soil_skin_flux = SoilConductiveFlux(1.5, 0.05))

land_roughness(z₀m, z₀s) = SimilarityTheoryFluxes(;
    stability_functions          = atmosphere_land_stability_functions(Float64),
    momentum_roughness_length    = z₀m,
    temperature_roughness_length = z₀s,
    water_vapor_roughness_length = z₀s)

forest_fluxes() = land_roughness(0.8, 0.08)   # rough canopy
bare_fluxes()   = land_roughness(0.01, 1e-3)  # smooth bare soil

# A shallow variably saturated column on a loam retention curve
# ([Carsel and Parrish (1988)](@cite carsel1988) mean `α`, `n`), started dry (below `𝒮ᶜ`)
# so the pulse produces a clear wet/dry contrast. The canopy stress reads this curve, so
# its wilting point and field capacity are the same soil's.

soil_slab_depth         = 0.075   # m — capacity ρˡ ν hˡᵃ = 30 kg m⁻²
soil_residual_fraction  = 0.05
liquid_density          = 1000.0
soil_retention          = VanGenuchtenRetention(α = 3.6, n = 1.56)
initial_soil_saturation = 0.3
initial_water_storage   = (initial_soil_saturation * (soil_porosity - soil_residual_fraction) +
                           soil_residual_fraction) * liquid_density * soil_slab_depth
initial_temperature     = 288.0

variably_saturated_soil() = VariablySaturatedHydrology(;
    slab_depth = soil_slab_depth, porosity = soil_porosity,
    residual_liquid_fraction = soil_residual_fraction, storage_height = 1000,
    retention_curve        = soil_retention,
    hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-6, n = 1.56),
    deep_liquid_flux       = NoDeepLiquidFlux(),
    runoff                 = InfiltrationCapacityRunoff(infiltration_capacity = 5e-4))

# ## Model builder
#
# The `TiledLandInterface` wraps the vegetated tile, its auto-derived bare counterpart,
# and the vegetation fraction `f_veg`, and is passed straight to the model as the
# atmosphere–land interface.

function tiled_land_column(; fraction, label)
    grid       = RectilinearGrid(CPU(); size = (), topology = (Flat, Flat, Flat))
    atmosphere = forced_atmosphere(grid, times)
    radiation  = forced_radiation(grid, times)

    land = SlabLand(grid; energy = SlabEnergy(),
                    hydrology = variably_saturated_soil())
    set!(land; T = initial_temperature, M = initial_water_storage)

    tiled = TiledLandInterface(grid, atmosphere, land;
                               vegetated        = vegetated_tile(),
                               fraction         = fraction,
                               vegetated_fluxes = forest_fluxes(),
                               bare_fluxes      = bare_fluxes())

    model = AtmosphereLandModel(atmosphere, land; radiation, atmosphere_land_interface = tiled)
    interface = model.interfaces.atmosphere_land_interface

    T    = zeros(Nsteps)   # time (days)
    H    = zeros(Nsteps)   # sensible heat, blended (positive up)
    LE   = zeros(Nsteps)   # latent heat, blended (positive up)
    effective_temperature = zeros(Nsteps)   # land-surface (radiometric) temperature
    Gᵍ   = zeros(Nsteps)   # skin → bulk ground heat flux (into slab)
    𝒮    = zeros(Nsteps)   # shared-column surface saturation
    latent_heat_vegetated  = zeros(Nsteps)   # vegetated-tile latent (weighted contribution)
    latent_heat_bare  = zeros(Nsteps)   # bare-tile latent (weighted contribution)
    friction_velocity_vegetated  = zeros(Nsteps)   # vegetated-tile friction velocity
    friction_velocity_bare  = zeros(Nsteps)   # bare-tile friction velocity

    @info "Running tiled land column: $label ..."
    for n in 1:Nsteps
        time_step!(model, Δt)
        time    = model.clock.time
        T[n]    = time / day
        H[n]    = scalar(interface.fluxes.sensible_heat)
        LE[n]   = scalar(interface.fluxes.latent_heat)
        effective_temperature[n] = scalar(interface.temperature.effective)
        Gᵍ[n]   = scalar(interface.temperature.ground_heat_flux)
        𝒮[n]    = scalar(land.saturation)
        latent_heat_vegetated[n]  = fraction       * scalar(interface.vegetated.fluxes.latent_heat)
        latent_heat_bare[n]  = (1 - fraction) * scalar(interface.bare.fluxes.latent_heat)
        friction_velocity_vegetated[n]  = scalar(interface.vegetated.fluxes.friction_velocity)
        friction_velocity_bare[n]  = scalar(interface.bare.fluxes.friction_velocity)
    end

    return (; label, fraction, t = T, H, LE, effective_temperature, Gᵍ, 𝒮, latent_heat_vegetated, latent_heat_bare, friction_velocity_vegetated, friction_velocity_bare)
end

scalar(field) = first(interior(field))

# ## Integration length and the fraction sweep
#
# Six days at `Δt = 10 min`. We sweep `f_veg` from bare to full canopy, plus a mosaic
# at `f_veg = 0.5` kept for the two-tile decomposition.

Δt     = 10minutes
Nsteps = 6 * 144
times  = range(0, Nsteps * Δt, step = 1hour)

fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
cases     = map(f -> tiled_land_column(fraction = f, label = "f_veg = $f"), fractions)
mosaic    = cases[3]   # f_veg = 0.5

# ## Figure 1 — the mosaic interpolates between bare soil and full canopy
#
# As `f_veg` rises, the partition shifts from a hot, sensible-dominated bare surface
# toward a cooler, transpiration-dominated canopy: sensible heat and the land-surface
# temperature fall, latent heat rises, and the drydown after the pulse slows.

colors = cgrad(:viridis, length(fractions); categorical = true)
pulse_span = (2, 2.5)
mark_pulse!(ax) = vspan!(ax, pulse_span...; color = (:skyblue, 0.25))

fig = Figure(size = (1400, 1050), fontsize = 16)

function sweep_panel(cell, field, title, ylabel; legend = :lt)
    ax = Axis(cell; title, xlabel = "t (days)", ylabel)
    mark_pulse!(ax)
    for (case, color) in zip(cases, colors)
        lines!(ax, case.t, getproperty(case, field); color, label = case.label)
    end
    axislegend(ax; position = legend, labelsize = 11)
    return ax
end

sweep_panel(fig[1, 1], :H,    "Sensible heat (blended)",  "H (W m⁻²)")
sweep_panel(fig[1, 2], :LE,   "Latent heat (blended)",    "LE (W m⁻²)")
sweep_panel(fig[2, 1], :effective_temperature, "Land-surface temperature", "effective_temperature (K)")

## Bowen ratio H/LE: the drier bare endpoint runs a much higher Bowen ratio.
ax = Axis(fig[2, 2]; title = "Midday Bowen ratio H/LE", xlabel = "t (days)", ylabel = "H / LE")
mark_pulse!(ax)
for (case, color) in zip(cases, colors)
    bowen = case.H ./ max.(case.LE, 1.0)
    lines!(ax, case.t, bowen; color, label = case.label)
end
ylims!(ax, -1, 4)
axislegend(ax; position = :lt, labelsize = 11)

ax = sweep_panel(fig[3, 1], :𝒮, "Shared-column saturation 𝒮", "𝒮"; legend = :rc)
hlines!(ax, [critical_saturation]; color = :gray, linestyle = :dash)
ylims!(ax, 0, 1.05)
sweep_panel(fig[3, 2], :Gᵍ, "Ground heat flux (into slab)", "Gᵍ (W m⁻²)")

Label(fig[0, 1:2], "Tiled land column — sweeping the vegetation fraction f_veg", fontsize = 21)

save("tiled_land_fraction_sweep.png", fig)
nothing #hide

# ![](tiled_land_fraction_sweep.png)

# ## Figure 2 — the two-tile decomposition of the f_veg = 0.5 mosaic
#
# The atmosphere feels only the blended flux, but the tiling resolves it into the
# vegetated (transpiration-heavy) and bare (evaporation-limited) contributions —
# `LE = f·LE_veg + (1−f)·LE_bare` — and the two tiles carry a large friction-velocity
# contrast from their roughness difference.

fig2 = Figure(size = (1400, 500), fontsize = 16)

ax = Axis(fig2[1, 1]; title = "Latent heat: blended = veg + bare contributions (f_veg = 0.5)",
          xlabel = "t (days)", ylabel = "LE (W m⁻²)")
mark_pulse!(ax)
lines!(ax, mosaic.t, mosaic.LE;  color = :black,       linewidth = 2, label = "blended")
lines!(ax, mosaic.t, mosaic.latent_heat_vegetated; color = :seagreen,    label = "vegetated (f·LE_veg)")
lines!(ax, mosaic.t, mosaic.latent_heat_bare; color = :saddlebrown, label = "bare ((1−f)·LE_bare)")
axislegend(ax; position = :lt, labelsize = 11)

ax = Axis(fig2[1, 2]; title = "Per-tile friction velocity (roughness contrast)",
          xlabel = "t (days)", ylabel = "u★ (m s⁻¹)")
mark_pulse!(ax)
lines!(ax, mosaic.t, mosaic.friction_velocity_vegetated; color = :seagreen,    label = "vegetated tile (rough)")
lines!(ax, mosaic.t, mosaic.friction_velocity_bare; color = :saddlebrown, label = "bare tile (smooth)")
axislegend(ax; position = :lt, labelsize = 11)

Label(fig2[0, 1:2], "Two-tile decomposition of the mosaic", fontsize = 21)

save("tiled_land_decomposition.png", fig2)
nothing #hide

# ![](tiled_land_decomposition.png)
