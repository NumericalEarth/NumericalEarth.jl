# # A tiled land column: mosaic of vegetation and bare soil
#
# A [`TiledLandInterface`](@ref) treats a land cell as a mosaic of a vegetated fraction
# `f` and bare-soil fraction `1 − f`, each running its own [`CanopyGroundSurface`](@ref)
# surface solve and area-weighted into one boundary condition. This example sweeps
# `f` under a diurnal forcing with a rain pulse, then decomposes one mosaic into
# its vegetated and bare tile contributions.

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
# The vegetated tile is a full `CanopyGroundSurface` (transpiring canopy, dry-layer soil
# evaporation branch); the bare tile is derived from it (LAI = 0). We give the two
# tiles a strong roughness contrast — rough forest canopy versus smooth bare surface.

critical_saturation = 0.5
soil_porosity       = 0.4

soil_branch() = DryLayerHumidity(;
    dry_layer_depth = StorageBasedDryLayerDepth(maximum_dry_layer_depth    = 0.025,
                                                dry_layer_onset_saturation = critical_saturation,
                                                dry_layer_exponent         = 2),
    vapor_exchange  = DryLayerVaporPistonVelocity(minimum_dry_layer_depth = 1e-3,
                                                  molecular_diffusivity   = 2.4e-5,
                                                  tortuosity              = ConstantTortuosity()))

# The canopy stress runs on plant-available water: `β` ramps from the permanent wilting
# point to field capacity, both read off the soil column's own retention curve below.
vegetated_tile() = CanopyGroundSurface(;
    soil_humidity   = soil_branch(),
    canopy_humidity = CanopyConductanceHumidity(; leaf_area_index = 4.0,
                                                moisture_stress = PlantAvailableWaterStress(),
                                                absorbed_par    = InteractiveAbsorbedPAR()),
    ground_heat_flux = SoilConductiveFlux(1.5, 0.05))

land_roughness(ℓᵐ, ℓˢ) = SimilarityTheoryFluxes(;
    stability_functions          = atmosphere_land_stability_functions(Float64),
    momentum_roughness_length    = ℓᵐ,
    temperature_roughness_length = ℓˢ,
    water_vapor_roughness_length = ℓˢ)

forest_fluxes() = land_roughness(0.8, 0.08)   # rough canopy
bare_fluxes()   = land_roughness(0.01, 1e-3)  # smooth bare soil

# A shallow variably saturated column on a loam retention curve
# ([Carsel and Parrish (1988)](@cite carsel1988) mean `α`, `n`), started dry (below `𝒮ᶜ`)
# so the pulse produces a clear wet/dry contrast. The canopy stress reads this curve, so
# its wilting point and field capacity are the same soil's.

soil_slab_depth         = 0.075   # m — capacity ρˡ ν hˡᵃ = 30 kg m⁻²
soil_residual_fraction  = 0.05
liquid_density          = 1000.0
soil_retention          = VanGenuchtenRetention(inverse_air_entry_head = 3.6, pore_size_uniformity = 1.56)
initial_soil_saturation = 0.3
initial_water_storage   = (initial_soil_saturation * (soil_porosity - soil_residual_fraction) +
                           soil_residual_fraction) * liquid_density * soil_slab_depth
initial_temperature     = 288.0

variably_saturated_soil() = VariablySaturatedHydrology(;
    slab_depth = soil_slab_depth, porosity = soil_porosity,
    residual_liquid_fraction = soil_residual_fraction, storage_height = 1000,
    retention_curve        = soil_retention,
    hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 1e-6, pore_size_uniformity = 1.56),
    deep_liquid_flux       = NoDeepLiquidFlux(),
    runoff                 = InfiltrationCapacityRunoff(infiltration_capacity = 5e-4))

# ## Model builder
#
# `TiledLandSurface` owns the tile physics and vegetation fraction. The interface
# supplies separate turbulent transport formulations for the forest and bare soil.

function tiled_land_column(; cover_fraction, label)
    grid       = RectilinearGrid(CPU(); size = (), topology = (Flat, Flat, Flat))
    atmosphere = forced_atmosphere(grid, times)
    radiation  = forced_radiation(grid, times)

    land = SlabLand(grid; energy = SlabEnergy(),
                    hydrology = variably_saturated_soil(),
                    surface = TiledLandSurface(LandSurface(vegetated_tile()); cover_fraction))
    set!(land; T = initial_temperature, M = initial_water_storage)

    tiled = TiledLandInterface(grid, atmosphere, land;
                               vegetated_fluxes = forest_fluxes(),
                               bare_fluxes      = bare_fluxes())

    model = AtmosphereLandModel(atmosphere, land; radiation, atmosphere_land_interface = tiled)
    interface = model.interfaces.atmosphere_land_interface

    t                           = zeros(number_of_steps)   # time (days)
    𝒬ᵀ                          = zeros(number_of_steps)   # sensible heat, blended (positive up)
    𝒬ᵛ                          = zeros(number_of_steps)   # latent heat, blended (positive up)
    Tᵉᶠᶠ                        = zeros(number_of_steps)   # land-surface (radiometric) temperature
    𝒬ᵍ                          = zeros(number_of_steps)   # skin → bulk ground heat flux (into slab)
    𝒮                           = zeros(number_of_steps)   # shared-column surface saturation
    vegetated_latent_heat       = zeros(number_of_steps)   # vegetated-tile latent (weighted contribution)
    bare_latent_heat            = zeros(number_of_steps)   # bare-tile latent (weighted contribution)
    vegetated_friction_velocity = zeros(number_of_steps)   # vegetated-tile friction velocity
    bare_friction_velocity      = zeros(number_of_steps)   # bare-tile friction velocity

    @info "Running tiled land column: $label ..."
    for n in 1:number_of_steps
        time_step!(model, Δt)
        time                           = model.clock.time
        t[n]                           = time / day
        𝒬ᵀ[n]                          = scalar(interface.fluxes.sensible_heat)
        𝒬ᵛ[n]                          = scalar(interface.fluxes.latent_heat)
        Tᵉᶠᶠ[n]                        = scalar(interface.temperature.effective)
        𝒬ᵍ[n]                          = scalar(interface.temperature.ground_heat_flux)
        𝒮[n]                           = scalar(land.saturation)
        vegetated_latent_heat[n]       = cover_fraction       * scalar(interface.vegetated.fluxes.latent_heat)
        bare_latent_heat[n]            = (1 - cover_fraction) * scalar(interface.bare.fluxes.latent_heat)
        vegetated_friction_velocity[n] = scalar(interface.vegetated.fluxes.friction_velocity)
        bare_friction_velocity[n]      = scalar(interface.bare.fluxes.friction_velocity)
    end

    return (; label, cover_fraction, t, 𝒬ᵀ, 𝒬ᵛ, Tᵉᶠᶠ, 𝒬ᵍ, 𝒮,
              vegetated_latent_heat, bare_latent_heat, vegetated_friction_velocity, bare_friction_velocity)
end

scalar(field) = first(interior(field))

# ## Integration length and the fraction sweep
#
# Six days at `Δt = 10 min`. We sweep `f` from bare to full canopy, plus a mosaic
# at `f = 0.5` kept for the two-tile decomposition.

Δt              = 10minutes
number_of_steps = 6 * 144
times           = range(0, number_of_steps * Δt, step = 1hour)

fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
cases     = map(f -> tiled_land_column(cover_fraction = f, label = "f = $f"), fractions)
mosaic    = cases[3]   # f = 0.5

# ## Figure 1 — the mosaic interpolates between bare soil and full canopy
#
# As `f` rises, the partition shifts from a hot, sensible-dominated bare surface
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

sweep_panel(fig[1, 1], :𝒬ᵀ,   "Sensible heat (blended)",  "𝒬ᵀ (W m⁻²)")
sweep_panel(fig[1, 2], :𝒬ᵛ,   "Latent heat (blended)",    "𝒬ᵛ (W m⁻²)")
sweep_panel(fig[2, 1], :Tᵉᶠᶠ, "Land-surface temperature", "Tᵉᶠᶠ (K)")

## Bowen ratio 𝒬ᵀ/𝒬ᵛ: the drier bare endpoint runs a much higher Bowen ratio.
ax = Axis(fig[2, 2]; title = "Midday Bowen ratio 𝒬ᵀ/𝒬ᵛ", xlabel = "t (days)", ylabel = "𝒬ᵀ / 𝒬ᵛ")
mark_pulse!(ax)
for (case, color) in zip(cases, colors)
    bowen = case.𝒬ᵀ ./ max.(case.𝒬ᵛ, 1.0)
    lines!(ax, case.t, bowen; color, label = case.label)
end
ylims!(ax, -1, 4)
axislegend(ax; position = :lt, labelsize = 11)

ax = sweep_panel(fig[3, 1], :𝒮, "Shared-column saturation 𝒮", "𝒮"; legend = :rc)
hlines!(ax, [critical_saturation]; color = :gray, linestyle = :dash)
ylims!(ax, 0, 1.05)
sweep_panel(fig[3, 2], :𝒬ᵍ, "Ground heat flux (into slab)", "𝒬ᵍ (W m⁻²)")

Label(fig[0, 1:2], "Tiled land column — sweeping the vegetation fraction f", fontsize = 21)

save("tiled_land_fraction_sweep.png", fig)
nothing #hide

# ![](tiled_land_fraction_sweep.png)

# ## Figure 2 — the two-tile decomposition of the f = 0.5 mosaic
#
# The atmosphere feels only the blended flux, but the tiling resolves it into the
# vegetated (transpiration-heavy) and bare (evaporation-limited) contributions —
# `𝒬ᵛ` is `f` times the vegetated latent heat plus `1 − f` times the bare latent heat —
# and the two tiles carry a large friction-velocity contrast from their roughness difference.

fig2 = Figure(size = (1400, 500), fontsize = 16)

ax = Axis(fig2[1, 1]; title = "Latent heat: blended = vegetated + bare contributions (f = 0.5)",
          xlabel = "t (days)", ylabel = "𝒬ᵛ (W m⁻²)")
mark_pulse!(ax)
lines!(ax, mosaic.t, mosaic.𝒬ᵛ;                    color = :black,       linewidth = 2, label = "blended")
lines!(ax, mosaic.t, mosaic.vegetated_latent_heat; color = :seagreen,    label = "vegetated, weighted by f")
lines!(ax, mosaic.t, mosaic.bare_latent_heat;      color = :saddlebrown, label = "bare, weighted by 1 − f")
axislegend(ax; position = :lt, labelsize = 11)

ax = Axis(fig2[1, 2]; title = "Per-tile friction velocity (roughness contrast)",
          xlabel = "t (days)", ylabel = "u★ (m s⁻¹)")
mark_pulse!(ax)
lines!(ax, mosaic.t, mosaic.vegetated_friction_velocity; color = :seagreen,    label = "vegetated tile (rough)")
lines!(ax, mosaic.t, mosaic.bare_friction_velocity;      color = :saddlebrown, label = "bare tile (smooth)")
axislegend(ax; position = :lt, labelsize = 11)

Label(fig2[0, 1:2], "Two-tile decomposition of the mosaic", fontsize = 21)

save("tiled_land_decomposition.png", fig2)
nothing #hide

# ![](tiled_land_decomposition.png)
