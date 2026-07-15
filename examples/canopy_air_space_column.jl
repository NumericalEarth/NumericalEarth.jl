# # A canopy-air-space column: leaf, soil-skin, and canopy-air temperatures
#
# This idealized 0D example exercises the [`CanopyAirSpace`](@ref) land-surface
# component under an analytic diurnal forcing, and checks that each of its pieces
# behaves as expected. The canopy and the soil surface exchange with a massless
# **canopy-air node** `(Tᵃᶜ, qᵃᶜ)` that drains to the atmosphere by Monin–Obukhov
# turbulence, so a single column carries *four* temperatures:
#
#     Tᵛ   leaf (diagnostic, massless: Rₙᵛ = Hᵛ + LEᵛ)
#     Tⁱⁿ  soil skin (diagnostic: Rₙᵍ = Hᵍ + LEᵍ + Λⁱⁿ(Tⁱⁿ − Tˡᵃ))
#     Tᵃᶜ  canopy-air node (what the atmosphere sees)
#     Tˡᵃ  bulk slab reservoir (prognostic; driven by the skin→bulk conduction Λⁱⁿ)
#
# The canopy shades the soil (Beer–Lambert), the leaf transpires (Farquhar–Medlyn
# stomata, reusing the same conductance path as `CanopyConductanceHumidity`), and
# the soil evaporates through a dry surface layer (`DryLayerHumidity`). The plotted
# diurnal cycle should show the leaf leading the forcing and swinging widely, the
# shaded soil skin damped, and the bulk slab lagging — with a nighttime inversion
# (the leaf radiatively cools *below* the soil).

# ## Load packages
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: update_state!
using CairoMakie
using Printf

# ## Idealized diurnal forcing
#
# A warm, humid regime with a strong daytime shortwave pulse; longwave, wind, humidity,
# and pressure held constant.

day = 86400
air_temperature(t)       = 288 - 6 * cos(2π * t / day)                # K, 282–294 diurnal
downwelling_shortwave(t) = max(0, 800 * cos(2π * (t - day/2) / day))  # W m⁻², daytime only
downwelling_longwave(t)  = 330                                        # W m⁻²
wind_speed               = 3.0                                        # m s⁻¹
specific_humidity        = 0.008                                      # kg kg⁻¹
surface_pressure         = 101325                                     # Pa

# ## Forcing builders
#
# Each `FieldTimeSeries` slice is filled from the analytic functions; the coupled
# model then interpolates between hourly slices as it steps.

function forced_atmosphere(grid, times)
    atmosphere = PrescribedAtmosphere(grid, times; surface_layer_height = 10, boundary_layer_height = 512)
    for n in eachindex(times)
        set!(atmosphere.velocities.u[n],  wind_speed)
        set!(atmosphere.specific_humidity[n], specific_humidity)
        set!(atmosphere.pressure[n],      surface_pressure)
        set!(atmosphere.temperature[n],   air_temperature(times[n]))
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

# ## The canopy-air-space closure
#
# The two branches are the same objects the standalone humidity closures use: a
# `DryLayerHumidity` soil branch and a photosynthesis-coupled `CanopyConductanceHumidity`
# leaf branch. `CanopyAirSpace` adds the sensible conductances, the two-face longwave,
# the shortwave split, and the coupled leaf/soil-skin solve. The soil skin conducts to
# the bulk through `SoilConductiveFlux` (`Λⁱⁿ = κᵀ/ℓᵀ`).

leaf_area_index = 3.0
extinction      = 0.5
clumping        = 1.0
leaf_albedo     = 0.15
ground_albedo   = 0.15

canopy_air_space = CanopyAirSpace(;
    soil = DryLayerHumidity(;
        dry_layer_depth = StorageBasedDryLayerDepth(maximum_dry_layer_depth = 0.015,
                                                    dry_layer_onset_saturation = 0.5,
                                                    dry_layer_exponent = 2),
        vapor_exchange  = DryLayerVaporPistonVelocity(minimum_dry_layer_depth = 1e-3,
                                                      molecular_diffusivity   = 2.4e-5,
                                                      tortuosity = ConstantTortuosity()),
        thermal_exchange_depth = 0.05, porosity = 0.4),
    canopy = CanopyConductanceHumidity(; leaf_area_index,
                                       moisture_stress = CriticalSaturation(0.5),
                                       absorbed_par    = InteractiveAbsorbedPAR()),
    soil_skin_flux = SoilConductiveFlux(1.5, 0.05),
    leaf_albedo, ground_albedo, extinction, clumping)

# ## Assemble the 0D coupled model
#
# The same `canopy_air_space` object fills both the temperature and the specific-humidity
# interface slots (it is a combined formulation). The bulk slab uses a pure `SlabEnergy`
# budget and a `BucketHydrology` water store.

grid       = RectilinearGrid(CPU(); size = (), topology = (Flat, Flat, Flat))
Δt         = 10minutes
Nsteps     = 3 * 144                 # 3 days
times      = range(0, Nsteps * Δt, step = 1hour)

atmosphere = forced_atmosphere(grid, times)
radiation  = forced_radiation(grid, times)
land       = SlabLand(grid; energy = SlabEnergy(), hydrology = BucketHydrology(maximum_water_storage = 100.0))
set!(land; T = 286.0, M = 50.0)      # 𝒮 = 0.5

model = AtmosphereLandModel(atmosphere, land; radiation,
                            atmosphere_land_interface_temperature       = canopy_air_space,
                            atmosphere_land_interface_specific_humidity = canopy_air_space)

# ## Forward run
#
# The `CanopyAirSpace` interface stores its diagnostic temperatures and the skin→bulk
# ground heat flux as a NamedTuple of fields in `interface.temperature`; the atmosphere
# only ever sees the node `Tᵃᶜ`.

scalar(field) = first(interior(field))
interface = model.interfaces.atmosphere_land_interface

t   = zeros(Nsteps)
Tₐ  = zeros(Nsteps)   # prescribed air temperature
Tᵃᶜ = zeros(Nsteps)   # canopy-air node
Tᵛ  = zeros(Nsteps)   # leaf
Tⁱⁿ = zeros(Nsteps)   # soil skin
Tˡᵃ = zeros(Nsteps)   # bulk slab
Tₑ  = zeros(Nsteps)   # effective radiating (LST)
H   = zeros(Nsteps)   # sensible heat, atmosphere total (positive up)
LE  = zeros(Nsteps)   # latent heat, atmosphere total (positive up)
Gᶜ  = zeros(Nsteps)   # skin → bulk ground heat flux
SW  = zeros(Nsteps)   # incident shortwave

@info "Running the canopy-air-space column..."
for n in 1:Nsteps
    time_step!(model, Δt)
    Ts = interface.temperature
    t[n]   = model.clock.time / day
    Tₐ[n]  = air_temperature(model.clock.time)
    Tᵃᶜ[n] = scalar(Ts.interface)
    Tᵛ[n]  = scalar(Ts.canopy)
    Tⁱⁿ[n] = scalar(Ts.soil_skin)
    Tˡᵃ[n] = scalar(land.temperature)
    Tₑ[n]  = scalar(Ts.effective)
    H[n]   = scalar(interface.fluxes.sensible_heat)
    LE[n]  = scalar(interface.fluxes.latent_heat)
    Gᶜ[n]  = scalar(Ts.ground_heat_flux)
    SW[n]  = downwelling_shortwave(model.clock.time)
end

# ## Component checks
#
# A few quantitative checks that the pieces behave as expected over the run.

daytime   = SW .> 400
nighttime = SW .< 5

@info @sprintf("Leaf diurnal swing:            Tᵛ ∈ [%.1f, %.1f] K (Δ = %.1f K)", minimum(Tᵛ), maximum(Tᵛ), maximum(Tᵛ) - minimum(Tᵛ))
@info @sprintf("Daytime leaf warmer than shaded soil skin (Tᵛ − Tⁱⁿ > 0):  %s (mean %+.2f K)",
               all(Tᵛ[daytime] .> Tⁱⁿ[daytime]), sum((Tᵛ.-Tⁱⁿ)[daytime])/count(daytime))
@info @sprintf("Nighttime leaf radiatively cooler than soil skin (Tᵛ < Tⁱⁿ): %s (mean %+.2f K)",
               all(Tᵛ[nighttime] .< Tⁱⁿ[nighttime]), sum((Tᵛ.-Tⁱⁿ)[nighttime])/count(nighttime))
@info @sprintf("Node between soil skin and atmosphere:  min(Tⁱⁿ,Tᵛ,Tₐ) ≤ Tᵃᶜ ≤ max: %s",
               all(min.(Tⁱⁿ,Tᵛ,Tₐ) .- 1e-3 .≤ Tᵃᶜ .≤ max.(Tⁱⁿ,Tᵛ,Tₐ) .+ 1e-3))

# ## Shortwave partition (the canopy shades the soil)
#
# Beer–Lambert splits the incident shortwave into a canopy-absorbed part and a
# transmitted part reaching the (shaded) soil.

transmitted = exp(-extinction * leaf_area_index * clumping)
canopy_SW = @. (1 - leaf_albedo) * (1 - transmitted) * SW
ground_SW = @. transmitted * (1 - ground_albedo) * SW

# ## Visualization

fig = Figure(size = (1500, 900), fontsize = 18)

ax_T = Axis(fig[1, 1]; title = "Temperatures — leaf leads, soil skin damped, bulk lags",
            xlabel = "t (days)", ylabel = "T (K)")
lines!(ax_T, t, Tₐ;  color = :gray,      linewidth = 3, label = "air Tₐ")
lines!(ax_T, t, Tᵛ;  color = :seagreen,  label = "leaf Tᵛ")
lines!(ax_T, t, Tⁱⁿ; color = :saddlebrown, label = "soil skin Tⁱⁿ")
lines!(ax_T, t, Tᵃᶜ; color = :steelblue, linestyle = :dash, label = "canopy air Tᵃᶜ")
lines!(ax_T, t, Tˡᵃ; color = :firebrick, label = "bulk Tˡᵃ")
lines!(ax_T, t, Tₑ;  color = :black,     linestyle = :dot, label = "LST Teff")
axislegend(ax_T; position = :lt, nbanks = 2)

ax_D = Axis(fig[1, 2]; title = "Two-source contrast\n(canopy heats the air; sunlit leaf > shaded soil)",
            xlabel = "t (days)", ylabel = "ΔT (K)")
hlines!(ax_D, [0]; color = :gray, linestyle = :dash)
lines!(ax_D, t, Tᵛ .- Tⁱⁿ; color = :seagreen,  label = "Tᵛ − Tⁱⁿ (leaf − soil skin)")
lines!(ax_D, t, Tᵃᶜ .- Tₐ; color = :steelblue, label = "Tᵃᶜ − Tₐ (node − air)")
axislegend(ax_D; position = :lt)

ax_F = Axis(fig[2, 1]; title = "Surface fluxes (positive: surface → atmosphere; Gᶜ into slab)",
            xlabel = "t (days)", ylabel = "flux (W m⁻²)")
lines!(ax_F, t, LE; color = :navy,     label = "latent H₂O (LE)")
lines!(ax_F, t, H;  color = :orange,   label = "sensible (H)")
lines!(ax_F, t, Gᶜ; color = :seagreen, linestyle = :dash, label = "ground heat Gᶜ")
axislegend(ax_F; position = :lt)

ax_S = Axis(fig[2, 2]; title = @sprintf("Beer–Lambert shading (LAI = %.0f: soil gets %.0f%% of sun)",
                                        leaf_area_index, 100transmitted),
            xlabel = "t (days)", ylabel = "shortwave (W m⁻²)")
lines!(ax_S, t, SW;        color = :goldenrod, linewidth = 3, label = "incident")
lines!(ax_S, t, canopy_SW; color = :seagreen,    label = "canopy-absorbed")
lines!(ax_S, t, ground_SW; color = :saddlebrown, label = "reaching soil")
axislegend(ax_S; position = :lt)

Label(fig[0, 1:2], "Canopy-air-space column — idealized diurnal forcing", fontsize = 22)

save("canopy_air_space_column.png", fig)
@info "Saved canopy_air_space_column.png"

nothing #hide

# ![](canopy_air_space_column.png)
