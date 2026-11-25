#####
##### Surface exchange and radiation of Breeze's moist `HydrostaticDynamics` on a `YinYangGrid`,
##### with the surfaces on a global latitude-longitude exchange grid
#####
##### PROVISIONAL (AMIP pilot adapter). This file is not yet included by `NumericalEarthBreezeExt`: it
##### depends on `breeze_hydrostatic_atmosphere.jl`, and a driver evaluates it in the extension module with
#####
#####     Base.include(Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt), <this file>)
#####
##### No bulk formula is defined here. The turbulent fluxes are those of `InterfaceComputations`, computed
##### on the exchange grid from the state of the lowest level; this file only moves state and fluxes between
##### the grids and inserts the fluxes into the budgets of the hydrostatic layers.
#####
##### 1. State. The lowest-level wind (averaged to cell centers), temperature, vapor mass fraction, layer
#####    pressure and geopotential height above the ground go from the panels to the exchange grid with the
#####    ownership-aware area-weighted averages of `YinYangRegridder`. The wind is regridded as a Cartesian
#####    3-vector.
##### 2. Fluxes. The fraction-weighted net stress, sensible heat flux and vapor flux are assembled at the cell
#####    centers of the exchange grid, with the weights of `_assemble_net_atmosphere_fluxes!`, and transferred
#####    to the panels with `UniformRedistribution`, which closes their global integrals (and the net force
#####    and torque of the stress) in the ownership quadrature. The transfer is not locally conservative.
##### 3. Insertion. Moist `HydrostaticDynamics` accepts no boundary condition and no forcing of water or
#####    energy, so the sources act once per coupling step before the dynamics, as the column physics of that
#####    model does after it (first-order sequential splitting). With `𝓗 = (Δπ cᵖᵈ + Wᵗ cᵖᵛ) Π θ` the
#####    enthalpy of a layer times g (Breeze's `_sediment_rain!`), `Δπ` its dry pressure thickness and `Wᵗ`
#####    the weight of its water,
#####
#####        every layer:    𝓗⁺ = 𝓗 + g Δt (ℱ⁻ - ℱ⁺)               ℱ the net upward radiative flux at its faces
#####        lowest layer:   𝓗⁺ += g Δt (𝒬ᵀ + Jᵛ cᵖᵛ T₁),   Wᵗ⁺ = Wᵗ + g Δt Jᵛ
#####
#####    and θ is expressed again from `𝓗⁺` at the new composition and pressure of the layer. The vapor
#####    enters at the temperature `T₁` of the air it enters, so that its mass does not change that
#####    temperature: the turbulent heat exchange is the sensible heat flux `𝒬ᵀ` alone ("sensible heat only;
#####    latent heat handled by moisture flux", as for the other Breeze dynamics). The enthalpy of Breeze's
#####    moist air counts vapor at `cᵖᵛ T` and liquid at `cˡ T - ℒˡ₀`, so the latent heat is released where the
#####    vapor condenses and is not a flux through the ground. The dry-mass surface pressure does not change;
#####    the total surface pressure rises by `g Δt Jᵛ`.
#####    The stress accelerates the lowest layer at `τ / (ρᵗ Δz)`, with `ρᵗ Δz` the mass of air and water of
#####    the layer per unit area.
##### 4. Radiation. Breeze's radiative transfer models index one grid, so each panel has its own gray model.
#####    They read the pressures of moist air: the layer pressure and the total pressure of the z-faces of
#####    the dynamics. Their lower boundary is the mosaic of the surfaces on the exchange grid (emissivity
#####    Σ f ϵ, temperature (Σ f ϵ T⁴ / ϵ)^¼, albedo Σ f α, as `_update_radiative_surface!`), interpolated
#####    bilinearly to the panels. Their downwelling fluxes at the ground go to the exchange grid, where the
#####    surfaces absorb them with the kernels of `Radiations`.
##### 5. Land. With a land component the mosaic and the net fluxes weight the land by `1 - ℵᵒᶜ`, the ocean
#####    fraction of the coupled model's surface partition. The land radiates at its interface temperature
#####    with the albedo and emissivity of the `land_surface` of the coupled model's radiation, which are those
#####    it absorbs with. The precipitation that the land takes is a field on the panels that a driver fills
#####    (`yin_yang_surface_precipitation`), averaged to the exchange grid with its global integral kept.
#####

using Oceananigans.Fields: interior
using Oceananigans.Grids: Face, Center
using Oceananigans.MultiRegion: YinYangGrid
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.TimeSteppers: update_state!, time_step!
using Oceananigans.Utils: getregion

using Breeze.HydrostaticEquations: MoistHydrostaticModel, hydrostatic_water_fields, interface_pressure,
                                   layer_pressure_thickness, warm_rain_column_physics!,
                                   column_physics_in_time_step!

using NumericalEarth.YinYangRegridding: YinYangRegridder, GlobalRescaling, UniformRedistribution, transfer_flux!,
                                        interpolate_to_yin_yang!

const MoistHydrostaticAtmosphere = MoistHydrostaticModel

const yin_yang_regions = (1, 2)

#####
##### What the atmosphere carries for the coupling
#####

# Held in the `radiation` slot of the Breeze model, which is the only place an `AtmosphereModel` has for
# what a coupler attaches to it after construction. `radiation` is the per-panel radiative transfer of
# `YinYangGrayRadiation`, or `nothing`; it is attached after the model is built, so the field is untyped.
mutable struct MoistHydrostaticExchange{S, L}
    const surface_fluxes :: S  # net fluxes into the atmosphere on its grid: ρu, ρv (stress), ρE (sensible heat), ρqᵛᵉ (vapor)
    const ledger :: L          # what the sources added to each column, accumulated (two-dimensional)
    radiation                  # YinYangGrayRadiation or nothing
    column_physics_substeps :: Int
end

Base.summary(exchange::MoistHydrostaticExchange) =
    string("MoistHydrostaticExchange(radiation = ", isnothing(exchange.radiation) ? "nothing" : summary(exchange.radiation),
           ", column_physics_substeps = ", exchange.column_physics_substeps, ")")

# Accumulated sources of each column, per unit area. Energies are enthalpies in Breeze's convention.
function surface_exchange_ledger(grid)
    column() = Field{Center, Center, Nothing}(grid)
    return (evaporation = column(),             # kg m⁻²: vapor that entered through the ground
            refused_deposition = column(),      # kg m⁻²: deposition onto the surface beyond the vapor of the lowest layer
            sensible_heat = column(),           # J m⁻²
            vapor_enthalpy = column(),          # J m⁻²: Jᵛ cᵖᵛ T₁
            surface_radiation = column(),       # J m⁻²: net upward radiative flux at the ground
            top_radiation = column())           # J m⁻²: net upward radiative flux at the model top
end

function surface_coupling(dynamics::HydrostaticDynamics, grid, microphysics::Breeze.BulkMicrophysics, thermodynamic_constants)
    surface_fluxes = (ρu   = Field{Face, Center, Nothing}(grid),
                      ρv   = Field{Center, Face, Nothing}(grid),
                      ρE   = Field{Center, Center, Nothing}(grid),
                      ρqᵛᵉ = Field{Center, Center, Nothing}(grid))

    return (; boundary_conditions = NamedTuple(), forcing = NamedTuple(), surface_fluxes)
end

function attach_coupled_radiation(model::MoistHydrostaticAtmosphere, coupling)
    exchange = MoistHydrostaticExchange(coupling.surface_fluxes, surface_exchange_ledger(model.grid), nothing, 1)
    # The adapter applies the column physics of the model itself, after the dynamics, once per coupling step
    column_physics_in_time_step!(model, false)
    return @set model.radiation = exchange
end

# Breeze's hooks for `model.radiation`: the sources are applied by `time_step!` below, not by the tendencies
Breeze.AtmosphereModels.update_radiation!(::MoistHydrostaticExchange, model) = nothing
Breeze.AtmosphereModels.radiation_flux_divergence(::MoistHydrostaticExchange) = nothing

NumericalEarth.EarthSystemModels.materialize_earth_system_radiation!(atmosphere::Simulation{<:MoistHydrostaticAtmosphere}, ::Nothing) = atmosphere
NumericalEarth.EarthSystemModels.materialize_earth_system_radiation!(atmosphere::Simulation{<:MoistHydrostaticAtmosphere}, radiation) = atmosphere

function NumericalEarth.EarthSystemModels.materialize_earth_system_radiation!(atmosphere::Simulation{<:MoistHydrostaticAtmosphere},
                                                                             ::Breeze.RadiativeTransferModel)
    throw(ArgumentError("A moist hydrostatic Breeze atmosphere does not take a Breeze RadiativeTransferModel as the " *
                        "radiation of the coupled model: attach `yin_yang_gray_radiation!` to the atmosphere."))
end

exchange_of(model::MoistHydrostaticAtmosphere) = model.radiation :: MoistHydrostaticExchange

# The pressures a panel's radiative transfer reads: those of moist air, at the layers and at the z-faces
panel_pressures(model::MoistHydrostaticAtmosphere, r) =
    PanelPressures(getregion(model.dynamics.pressure, r), getregion(model.dynamics.moisture.interface_pressure, r))

#####
##### Per-panel gray radiation
#####

struct YinYangGrayRadiation{M, S, D}
    panel_models :: M        # one Breeze gray RadiativeTransferModel per panel
    surface :: S             # temperature, emissivity, albedo of the mosaic, on the atmosphere grid
    downwelling :: D         # shortwave, longwave at the ground on the atmosphere grid (W m⁻², positive upward)
    surface_optics :: @NamedTuple{ocean_albedo::Float64, ocean_emissivity::Float64,
                                  sea_ice_albedo::Float64, sea_ice_emissivity::Float64}
    interval :: Int          # coupling steps between two solves
end

Base.summary(radiation::YinYangGrayRadiation) =
    string("YinYangGrayRadiation(every ", radiation.interval, " coupling steps)")

# What the gray model reads of a model, for one panel: the pressures are those of moist air
struct PanelPressures{P, I}
    layer_pressure :: P
    interface_pressure :: I
end

struct PanelInterfacePressure{I}
    interface_pressure :: I
end

Adapt.adapt_structure(to, pressure::PanelInterfacePressure) = PanelInterfacePressure(Adapt.adapt(to, pressure.interface_pressure))

Breeze.AtmosphereModels.dynamics_pressure(pressures::PanelPressures) = pressures.layer_pressure
Breeze.AtmosphereModels.level_pressure_source(pressures::PanelPressures) = PanelInterfacePressure(pressures.interface_pressure)

@inline Breeze.AtmosphereModels.level_pressure(i, j, k, grid, p, pressure::PanelInterfacePressure) =
    @inbounds pressure.interface_pressure[i, j, k]

"""
$(TYPEDSIGNATURES)

Attach a gray radiative transfer model per panel to `atmosphere`, a moist hydrostatic Breeze atmosphere on
a `YinYangGrid` built by `atmosphere_simulation`, and return the `YinYangGrayRadiation`. `surface_optics`
are the albedos and emissivities of the open ocean and of the sea ice, which must be those of the
`PrescribedRadiation` of the coupled model. `kw` are keyword arguments of Breeze's gray
`RadiativeTransferModel` (`solar_position`, `solar_constant`, `optical_thickness`).
"""
function yin_yang_gray_radiation!(atmosphere; surface_optics, interval = 1, initial_surface_temperature = 288, kw...)
    model = component_model(atmosphere)
    grid = model.grid
    grid isa YinYangGrid || throw(ArgumentError("yin_yang_gray_radiation! needs an atmosphere on a YinYangGrid."))

    column() = Field{Center, Center, Nothing}(grid)
    surface = (temperature = column(), emissivity = column(), albedo = column())
    downwelling = (shortwave = column(), longwave = column())

    fill!(surface.temperature, initial_surface_temperature)
    fill!(surface.emissivity, surface_optics.ocean_emissivity)
    fill!(surface.albedo, surface_optics.ocean_albedo)

    panel_models = map(yin_yang_regions) do r
        Breeze.RadiativeTransferModel(getregion(grid, r), Breeze.GrayOptics(), model.thermodynamic_constants;
                                      surface_temperature = getregion(surface.temperature, r),
                                      surface_emissivity = getregion(surface.emissivity, r),
                                      surface_albedo = getregion(surface.albedo, r),
                                      kw...)
    end

    optics = (ocean_albedo = Float64(surface_optics.ocean_albedo),
              ocean_emissivity = Float64(surface_optics.ocean_emissivity),
              sea_ice_albedo = Float64(surface_optics.sea_ice_albedo),
              sea_ice_emissivity = Float64(surface_optics.sea_ice_emissivity))

    radiation = YinYangGrayRadiation(panel_models, surface, downwelling, optics, interval)
    exchange_of(model).radiation = radiation
    return radiation
end

@kernel function _copy_surface_downwelling!(shortwave, longwave, panel_shortwave, panel_longwave)
    i, j = @index(Global, NTuple)
    @inbounds begin
        shortwave[i, j, 1] = panel_shortwave[i, j, 1]
        longwave[i, j, 1] = panel_longwave[i, j, 1]
    end
end

solve_radiation!(::Nothing, model) = nothing

function solve_radiation!(radiation::YinYangGrayRadiation, model)
    model.clock.iteration % radiation.interval == 0 || return nothing
    grid = model.grid

    for r in yin_yang_regions
        region = getregion(grid, r)
        rtm = radiation.panel_models[r]
        panel = (grid = region, clock = model.clock, dynamics = panel_pressures(model, r), temperature = getregion(model.temperature, r))
        Breeze.AtmosphereModels._update_radiation!(rtm, panel)

        launch!(architecture(region), region, :xy, _copy_surface_downwelling!,
                getregion(radiation.downwelling.shortwave, r), getregion(radiation.downwelling.longwave, r),
                rtm.downwelling_shortwave_flux, rtm.downwelling_longwave_flux)
    end

    return nothing
end

panel_radiative_fluxes(::Nothing, r) = nothing

panel_radiative_fluxes(radiation::YinYangGrayRadiation, r) =
    (upwelling_longwave = radiation.panel_models[r].upwelling_longwave_flux,
     downwelling_longwave = radiation.panel_models[r].downwelling_longwave_flux,
     upwelling_shortwave = radiation.panel_models[r].upwelling_shortwave_flux,
     downwelling_shortwave = radiation.panel_models[r].downwelling_shortwave_flux)

# Net upward radiative flux through the z-face `k` (W m⁻²)
@inline net_upward_radiation(i, j, k, ::Nothing) = 0
@inline net_upward_radiation(i, j, k, ℱ) =
    @inbounds ℱ.upwelling_longwave[i, j, k] + ℱ.downwelling_longwave[i, j, k] +
              ℱ.upwelling_shortwave[i, j, k] + ℱ.downwelling_shortwave[i, j, k]

#####
##### Insertion of the surface fluxes and of the radiative heating
#####

@kernel function _apply_surface_stress!(u, v, grid, τˣ, τʸ, ρᵗ, Δt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Δz = Δzᶜᶜᶜ(i, j, 1, grid)
        u[i, j, 1] += Δt * τˣ[i, j, 1] / (ℑxᶠᵃᵃ(i, j, 1, grid, ρᵗ) * Δz)
        v[i, j, 1] += Δt * τʸ[i, j, 1] / (ℑyᵃᶠᵃ(i, j, 1, grid, ρᵗ) * Δz)
    end
end

@kernel function _apply_column_sources!(ρθ, ρqᵉ, ledger, grid, 𝒬ᵀ, Jᵛ, ℱ, ρ, ρᵗ, ρqʳ, A, B, πˢ, pᶠ, θ, Π, T, qᵛ,
                                        cᵖᵈ, cᵖᵛ, pˢᵗ, κ, g, Δt)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        # Radiative heating of the layers above the lowest: their composition and pressure do not change
        for k in 2:Nz
            Δz = Δzᶜᶜᶜ(i, j, k, grid)
            Δπ = layer_pressure_thickness(i, j, k, A, B, πˢ)
            Wᵗ = g * Δz * (ρqᵉ[i, j, k] + ρqʳ[i, j, k])
            heating = net_upward_radiation(i, j, k, ℱ) - net_upward_radiation(i, j, k+1, ℱ)
            ρθ[i, j, k] += ρ[i, j, k] * g * Δt * heating / ((Δπ * cᵖᵈ + Wᵗ * cᵖᵛ) * Π[i, j, k])
        end

        # The lowest layer also takes the surface fluxes
        Δz = Δzᶜᶜᶜ(i, j, 1, grid)
        Δπ = layer_pressure_thickness(i, j, 1, A, B, πˢ)
        T₁ = T[i, j, 1]

        # Deposition onto the surface is limited to the vapor of the layer: a limit on a rate between two
        # reservoirs, of which what is refused is recorded
        vapor = max(0, qᵛ[i, j, 1]) * ρᵗ[i, j, 1] * Δz
        requested = Δt * Jᵛ[i, j, 1]
        evaporated = max(requested, -vapor)

        Wᵗ = g * Δz * (ρqᵉ[i, j, 1] + ρqʳ[i, j, 1])
        Wᵗ⁺ = Wᵗ + g * evaporated
        heating = net_upward_radiation(i, j, 1, ℱ) - net_upward_radiation(i, j, 2, ℱ)
        𝓗 = (Δπ * cᵖᵈ + Wᵗ * cᵖᵛ) * Π[i, j, 1] * θ[i, j, 1]
        𝓗⁺ = 𝓗 + g * (Δt * (𝒬ᵀ[i, j, 1] + heating) + evaporated * cᵖᵛ * T₁)

        # Pressure and Exner function of the layer under its new water
        p⁻ = pᶠ[i, j, 1] + g * evaporated
        p⁺ = pᶠ[i, j, 2]
        Πᶠ⁻ = (p⁻ / pˢᵗ)^κ
        Πᶠ⁺ = (p⁺ / pˢᵗ)^κ
        Π⁺ = (p⁻ * Πᶠ⁻ - p⁺ * Πᶠ⁺) / ((1 + κ) * (Δπ + Wᵗ⁺))

        ρθ[i, j, 1] = ρ[i, j, 1] * 𝓗⁺ / ((Δπ * cᵖᵈ + Wᵗ⁺ * cᵖᵛ) * Π⁺)
        ρqᵉ[i, j, 1] += evaporated / Δz

        ledger.evaporation[i, j, 1] += evaporated
        ledger.refused_deposition[i, j, 1] += evaporated - requested
        ledger.sensible_heat[i, j, 1] += Δt * 𝒬ᵀ[i, j, 1]
        ledger.vapor_enthalpy[i, j, 1] += evaporated * cᵖᵛ * T₁
        ledger.surface_radiation[i, j, 1] += Δt * net_upward_radiation(i, j, 1, ℱ)
        ledger.top_radiation[i, j, 1] += Δt * net_upward_radiation(i, j, Nz+1, ℱ)
    end
end

regional(fields::NamedTuple, r) = map(field -> getregion(field, r), fields)

"""
$(TYPEDSIGNATURES)

Add to `model`, a moist hydrostatic Breeze atmosphere, what the surface fluxes of its `MoistHydrostaticExchange`
and the radiative fluxes of its radiation supply in `Δt`, and diagnose its state again.
"""
function apply_surface_and_radiative_sources!(model::MoistHydrostaticAtmosphere, Δt)
    exchange = exchange_of(model)
    grid = model.grid
    dynamics = model.dynamics
    moisture = dynamics.moisture
    constants = model.thermodynamic_constants
    water = hydrostatic_water_fields(model.microphysics, model.microphysical_fields)
    fluxes = exchange.surface_fluxes
    u, v, _ = model.velocities

    FT = eltype(grid)
    cᵖᵈ = convert(FT, constants.dry_air.heat_capacity)
    cᵖᵛ = convert(FT, constants.vapor.heat_capacity)
    κ = convert(FT, Breeze.dry_air_gas_constant(constants) / constants.dry_air.heat_capacity)
    g = convert(FT, constants.gravitational_acceleration)
    pˢᵗ = convert(FT, dynamics.standard_pressure)
    Δt = convert(FT, Δt)

    for r in yin_yang_regions
        region = getregion(grid, r)
        arch = architecture(region)

        launch!(arch, region, :xy, _apply_surface_stress!,
                getregion(u, r), getregion(v, r), region, getregion(fluxes.ρu, r), getregion(fluxes.ρv, r),
                getregion(moisture.total_density, r), Δt)

        launch!(arch, region, :xy, _apply_column_sources!,
                getregion(model.formulation.potential_temperature_density, r),
                getregion(model.moisture_density, r),
                regional(exchange.ledger, r),
                region,
                getregion(fluxes.ρE, r),
                getregion(fluxes.ρqᵛᵉ, r),
                panel_radiative_fluxes(exchange.radiation, r),
                getregion(dynamics.density, r),
                getregion(moisture.total_density, r),
                getregion(water.ρqʳ, r),
                getregion(dynamics.pressure_coefficient, r),
                getregion(dynamics.surface_pressure_coefficient, r),
                getregion(dynamics.surface_pressure, r),
                getregion(moisture.interface_pressure, r),
                getregion(model.formulation.potential_temperature, r),
                getregion(dynamics.exner_function, r),
                getregion(model.temperature, r),
                getregion(water.qᵛ, r),
                cᵖᵈ, cᵖᵛ, pˢᵗ, κ, g, Δt)
    end

    update_state!(model)

    return nothing
end

# One coupling step of the atmosphere: radiation of the state at the start of the step, the sources of the
# step, the dynamics, and the column physics of the model.
function Oceananigans.TimeSteppers.time_step!(atmosphere::Simulation{<:MoistHydrostaticAtmosphere}, Δt)
    model = atmosphere.model
    exchange = exchange_of(model)

    solve_radiation!(exchange.radiation, model)
    apply_surface_and_radiative_sources!(model, Δt)
    time_step!(model, Δt)
    warm_rain_column_physics!(model.microphysics, model; Δt, substeps = exchange.column_physics_substeps)

    return nothing
end

#####
##### Exchange of state through the latitude-longitude exchange grid
#####

NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(atmosphere::MoistHydrostaticAtmosphere, exchange_grid;
                                                                          correction = nothing) =
    yin_yang_component_exchanger(atmosphere, exchange_grid, correction)

# Shared by the Breeze atmospheres on a Yin-Yang grid that exchange through a latitude-longitude grid
function yin_yang_component_exchanger(atmosphere, exchange_grid, correction)
    grid = atmosphere.grid
    grid isa YinYangGrid ||
        throw(ArgumentError("This surface exchange of a Breeze atmosphere is implemented on a YinYangGrid " *
                            "with a latitude-longitude exchange grid; got $(summary(grid))."))
    isnothing(correction) || throw(ArgumentError("A Breeze atmosphere on a Yin-Yang grid takes no state correction."))

    regridder = YinYangRegridder(grid, exchange_grid)

    state = (; u    = Oceananigans.CenterField(exchange_grid),
               v    = Oceananigans.CenterField(exchange_grid),
               T    = Oceananigans.CenterField(exchange_grid),
               p    = Oceananigans.CenterField(exchange_grid),
               q    = Oceananigans.CenterField(exchange_grid),
               ℐꜜˢʷ = Oceananigans.CenterField(exchange_grid),
               ℐꜜˡʷ = Oceananigans.CenterField(exchange_grid),
               Jʳⁿ  = Oceananigans.CenterField(exchange_grid),   # the precipitation the land takes; zero without land
               Jˢⁿ  = Oceananigans.CenterField(exchange_grid))

    panel_column() = Field{Center, Center, Nothing}(grid)
    exchange_column() = Field{Center, Center, Nothing}(exchange_grid)

    yin_yang = (u = panel_column(), v = panel_column(), T = panel_column(), p = panel_column(), q = panel_column(),
                height = panel_column(), precipitation = panel_column())

    latitude_longitude = (x_momentum = exchange_column(), y_momentum = exchange_column(),
                          sensible_heat = exchange_column(), water_vapor = exchange_column(),
                          surface_temperature = exchange_column(), surface_emissivity = exchange_column(),
                          surface_albedo = exchange_column(),
                          downwelling_shortwave = exchange_column(), downwelling_longwave = exchange_column())

    return ComponentExchanger(state, (; regridder, yin_yang, latitude_longitude), nothing)
end

# The height on the exchange grid is filled when the state is first exchanged, before any flux is computed.
# Until then it holds the lowest height of the panels, which the checks of the interfaces read when they are built.
function NumericalEarth.EarthSystemModels.surface_layer_height(atmosphere::MoistHydrostaticAtmosphere, exchange_grid)
    dynamics = atmosphere.dynamics
    g = atmosphere.thermodynamic_constants.gravitational_acceleration
    lowest = minimum(yin_yang_regions) do r
        Φ = interior(getregion(dynamics.geopotential, r), :, :, 1)
        Φˢ = interior(getregion(dynamics.surface_geopotential, r), :, :, 1)
        minimum((Φ .- Φˢ) ./ g)
    end

    lowest > 0 ||
        throw(DomainError(lowest, "the lowest level of the hydrostatic atmosphere is not above the ground: set its " *
                                  "state before coupling it."))

    height = Field{Center, Center, Nothing}(exchange_grid)
    fill!(height, lowest)
    return height
end

@kernel function _lowest_level_state!(state, grid, u, v, T, qᵛ, p, Φ, Φˢ, g)
    i, j = @index(Global, NTuple)
    @inbounds begin
        state.u[i, j, 1] = ℑxᶜᵃᵃ(i, j, 1, grid, u)
        state.v[i, j, 1] = ℑyᵃᶜᵃ(i, j, 1, grid, v)
        state.T[i, j, 1] = T[i, j, 1]
        state.q[i, j, 1] = qᵛ[i, j, 1]
        state.p[i, j, 1] = p[i, j, 1]
        state.height[i, j, 1] = (Φ[i, j, 1] - Φˢ[i, j, 1]) / g
    end
end

function NumericalEarth.EarthSystemModels.interpolate_state!(exchanger, exchange_grid, atmosphere::MoistHydrostaticAtmosphere, coupled_model)
    (; regridder, yin_yang, latitude_longitude) = exchanger.regridder
    state = exchanger.state
    grid = atmosphere.grid
    dynamics = atmosphere.dynamics
    u, v, _ = atmosphere.velocities
    water = hydrostatic_water_fields(atmosphere.microphysics, atmosphere.microphysical_fields)
    g = atmosphere.thermodynamic_constants.gravitational_acceleration

    for r in yin_yang_regions
        region = getregion(grid, r)
        launch!(architecture(region), region, :xy, _lowest_level_state!,
                regional(yin_yang, r), region, getregion(u, r), getregion(v, r), getregion(atmosphere.temperature, r),
                getregion(water.qᵛ, r), getregion(dynamics.pressure, r), getregion(dynamics.geopotential, r),
                getregion(dynamics.surface_geopotential, r), g)
    end

    exchange_lowest_level_state!(exchanger, atmosphere, coupled_model)

    return nothing
end

# From the lowest-level state on the panels (`yin_yang`) to the exchange grid
function exchange_lowest_level_state!(exchanger, atmosphere, coupled_model)
    (; regridder, yin_yang, latitude_longitude) = exchanger.regridder
    state = exchanger.state

    transfer_flux!((state.u, state.v), regridder, (yin_yang.u, yin_yang.v))
    transfer_flux!(state.T, regridder, yin_yang.T)
    transfer_flux!(state.p, regridder, yin_yang.p)
    transfer_flux!(state.q, regridder, yin_yang.q)

    height = coupled_model.interfaces.properties.surface_layer_height
    transfer_flux!(height, regridder, yin_yang.height)

    lowest = minimum(height)
    lowest > 0 ||
        throw(DomainError(lowest, "the lowest level of the hydrostatic atmosphere is not above the ground: " *
                                  "its height must be positive in every column."))

    exchange_precipitation!(coupled_model.interfaces.atmosphere_land_interface, state, regridder, yin_yang)
    exchange_downwelling_radiation!(exchange_of(atmosphere).radiation, coupled_model, regridder, latitude_longitude)

    return nothing
end

"""
$(TYPEDSIGNATURES)

Return the field on the grid of the Yin-Yang Breeze atmosphere of `coupled_model` that holds the precipitation
reaching the ground (kg m⁻² s⁻¹, positive downward). The atmosphere does not fill it: a driver writes the
rate of each coupling step into it after the atmosphere is stepped, for example from the column physics
that the exchange applies. When the state is exchanged it is averaged onto the exchange grid, with its
global integral kept, as the rain that the land takes. It is not read by a model without land.
"""
yin_yang_surface_precipitation(coupled_model) =
    coupled_model.interfaces.exchanger.atmosphere.regridder.yin_yang.precipitation

# Only the land takes precipitation: the ocean and the sea ice are prescribed
exchange_precipitation!(::Nothing, state, regridder, yin_yang) = nothing

exchange_precipitation!(land_interface, state, regridder, yin_yang) =
    transfer_flux!(state.Jʳⁿ, regridder, yin_yang.precipitation; correction = GlobalRescaling())

exchange_downwelling_radiation!(::Nothing, coupled_model, regridder, latitude_longitude) = nothing

# The surfaces absorb the downwelling fluxes of the atmosphere's radiation: they replace what the radiation
# component of the coupled model interpolated, in the state its exchanger publishes (W m⁻², positive upward)
function exchange_downwelling_radiation!(radiation::YinYangGrayRadiation, coupled_model, regridder, latitude_longitude)
    radiation_exchanger = coupled_model.interfaces.exchanger.radiation
    isnothing(radiation_exchanger) &&
        throw(ArgumentError("The coupled model has no radiation component to hold the downwelling fluxes of the " *
                            "atmosphere: build it with `radiation = PrescribedRadiation(exchange_grid)`."))

    transfer_flux!(latitude_longitude.downwelling_shortwave, regridder, radiation.downwelling.shortwave)
    transfer_flux!(latitude_longitude.downwelling_longwave, regridder, radiation.downwelling.longwave)

    parent(radiation_exchanger.state.ℐꜜˢʷ) .= parent(latitude_longitude.downwelling_shortwave)
    parent(radiation_exchanger.state.ℐꜜˡʷ) .= parent(latitude_longitude.downwelling_longwave)

    return nothing
end

#####
##### Net fluxes
#####

NumericalEarth.EarthSystemModels.InterfaceComputations.net_fluxes(atmosphere::MoistHydrostaticAtmosphere) =
    exchange_of(atmosphere).surface_fluxes

# The weights of `_assemble_net_atmosphere_fluxes!`, with the stress left at the cell centers
@kernel function _assemble_centered_net_atmosphere_fluxes!(net, ao_fluxes, ai_fluxes, al_fluxes, ℵᵒᶜ, ℵˢⁱ, grid)
    i, j = @index(Global, NTuple)
    ℵᵒᶜᵢ = bounded_fraction(ℵᵒᶜ, i, j, 1)
    ℵˢⁱᵢ = bounded_fraction(ℵˢⁱ, i, j, 1)

    @inbounds begin
        net.x_momentum[i, j, 1] = ρτᶜᶜᶜ(i, j, 1, grid, ao_fluxes.x_momentum, ai_fluxes.x_momentum, al_fluxes.x_momentum, ℵᵒᶜ, ℵˢⁱ)
        net.y_momentum[i, j, 1] = ρτᶜᶜᶜ(i, j, 1, grid, ao_fluxes.y_momentum, ai_fluxes.y_momentum, al_fluxes.y_momentum, ℵᵒᶜ, ℵˢⁱ)
        net.sensible_heat[i, j, 1] = ℵᵒᶜᵢ * ((1 - ℵˢⁱᵢ) * ao_fluxes.sensible_heat[i, j, 1] + ℵˢⁱᵢ * ai_fluxes.sensible_heat[i, j, 1]) + (1 - ℵᵒᶜᵢ) * al_fluxes.sensible_heat[i, j, 1]
        net.water_vapor[i, j, 1] = ℵᵒᶜᵢ * ((1 - ℵˢⁱᵢ) * ao_fluxes.water_vapor[i, j, 1] + ℵˢⁱᵢ * ai_fluxes.water_vapor[i, j, 1]) + (1 - ℵᵒᶜᵢ) * al_fluxes.water_vapor[i, j, 1]
    end
end

NumericalEarth.EarthSystemModels.update_net_fluxes!(coupled_model, atmosphere::MoistHydrostaticAtmosphere) =
    yin_yang_update_net_fluxes!(coupled_model, atmosphere)

function yin_yang_update_net_fluxes!(coupled_model, atmosphere)
    net = coupled_model.interfaces.net_fluxes.atmosphere
    isnothing(net) && return nothing

    interfaces = coupled_model.interfaces
    exchange_grid = interfaces.exchanger.grid
    (; regridder, latitude_longitude) = interfaces.exchanger.atmosphere.regridder

    ao_fluxes = computed_fluxes(interfaces.atmosphere_ocean_interface)
    ai_fluxes = computed_fluxes(interfaces.atmosphere_sea_ice_interface)
    al_fluxes = computed_fluxes(interfaces.atmosphere_land_interface)
    ℵᵒᶜ = interfaces.surface_partition.ocean_fraction
    ℵˢⁱ = sea_ice_concentration(coupled_model.sea_ice)

    launch!(architecture(exchange_grid), exchange_grid, :xy, _assemble_centered_net_atmosphere_fluxes!,
            latitude_longitude, ao_fluxes, ai_fluxes, al_fluxes, ℵᵒᶜ, ℵˢⁱ, exchange_grid)

    correction = UniformRedistribution()
    transfer_flux!((net.ρu, net.ρv), regridder, (latitude_longitude.x_momentum, latitude_longitude.y_momentum); correction)
    transfer_flux!(net.ρE, regridder, latitude_longitude.sensible_heat; correction)
    transfer_flux!(net.ρqᵛᵉ, regridder, latitude_longitude.water_vapor; correction)

    publish_radiative_surface!(exchange_of(atmosphere).radiation, coupled_model, regridder, latitude_longitude)

    return nothing
end

publish_radiative_surface!(::Nothing, coupled_model, regridder, latitude_longitude) = nothing

# The land's radiating surface on the exchange grid: the interface temperature of the land, with the albedo
# and emissivity that the land absorbs the downwelling radiation with. Without land its fraction is zero.
mosaic_land_surface(::Nothing, coupled_model) = (temperature = 0, albedo = 0, emissivity = 0)

function mosaic_land_surface(land_interface, coupled_model)
    exchanger = coupled_model.interfaces.exchanger.radiation
    properties = NumericalEarth.EarthSystemModels.InterfaceComputations.kernel_radiation_properties(coupled_model.radiation,
                                                                                                       exchanger).surface_properties
    land = get(properties, :land, nothing)
    isnothing(land) &&
        throw(ArgumentError("The radiation of the coupled model has no land surface, which the per-panel radiation " *
                            "of a Breeze atmosphere on a Yin-Yang grid needs over land: build it with " *
                            "`PrescribedRadiation(exchange_grid; land_surface = SurfaceRadiationProperties(albedo, emissivity))`."))

    return radiative_surface(land_interface, land.albedo, land.emissivity)
end

@kernel function _mosaic_radiative_surface!(surface, sea, land, optics, ℐꜜˢʷ, grid, clock)
    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)
    time = Time(clock.time)
    location = (Center, Center, Center)

    @inbounds begin
        ℵᵒᶜ = clamp(sea.ocean_fraction[i, j, 1], 0, 1)
        ℵˢⁱ = clamp(sea.sea_ice_concentration[i, j, 1], 0, 1)
        Tᵒᶜ = convert_to_kelvin(sea.ocean_units, sea.ocean_temperature[i, j, 1])
        Tˢⁱ = convert_to_kelvin(sea.sea_ice_units, sea.sea_ice_temperature[i, j, 1])
        ℐꜜˢʷᵢ = ℐꜜˢʷ[i, j, 1]
    end

    # As the land absorbs: `_air_land_surface_radiation_state`
    Tˡᵃ = stateindex(land.temperature, i, j, 1)
    αˡᵃ = stateindex(land.albedo, i, j, kᴺ, grid, time, location, ℐꜜˢʷᵢ)
    ϵˡᵃ = stateindex(land.emissivity, i, j, kᴺ, grid, time, location)

    # The weights of `_assemble_centered_net_atmosphere_fluxes!`
    land_fraction = 1 - ℵᵒᶜ
    ocean_fraction = ℵᵒᶜ * (1 - ℵˢⁱ)
    sea_ice_fraction = ℵᵒᶜ * ℵˢⁱ

    # Emissivity, and the temperature that emits the mosaic's longwave through it
    ϵ = land_fraction * ϵˡᵃ + ocean_fraction * optics.ocean_emissivity + sea_ice_fraction * optics.sea_ice_emissivity
    ϵT⁴ = land_fraction * ϵˡᵃ * Tˡᵃ^4 + ocean_fraction * optics.ocean_emissivity * Tᵒᶜ^4 +
          sea_ice_fraction * optics.sea_ice_emissivity * Tˢⁱ^4
    T⁴ = ifelse(ϵ > 0, ϵT⁴ / ϵ, land_fraction * Tˡᵃ^4 + ocean_fraction * Tᵒᶜ^4 + sea_ice_fraction * Tˢⁱ^4)
    α = land_fraction * αˡᵃ + ocean_fraction * optics.ocean_albedo + sea_ice_fraction * optics.sea_ice_albedo

    @inbounds begin
        surface.surface_temperature[i, j, 1] = sqrt(sqrt(T⁴))
        surface.surface_emissivity[i, j, 1] = ϵ
        surface.surface_albedo[i, j, 1] = α
    end
end

@kernel function _snow_proxy_land_albedo!(albedo, temperature, α⁰, Δα, Tᵐ, ΔT)
    i, j = @index(Global, NTuple)
    @inbounds albedo[i, j, 1] = α⁰ + Δα * clamp((Tᵐ - temperature[i, j, 1]) / ΔT, 0, 1)
end

"""
$(TYPEDSIGNATURES)

Set the land `albedo`, a field on the exchange grid, from the land `temperature` (K): `snow_free_albedo` at
and above `melting_temperature`, rising linearly to `snow_albedo` at `melting_temperature - temperature_range`
and below. The temperature stands in for snow and land ice, which a slab land does not carry: cold land,
the ice sheets included, is bright. A driver calls this between coupling steps on the field it gave to the
`land_surface` of the coupled model's radiation, so that the land absorbs and the mosaic reflects with the
same albedo.
"""
function snow_proxy_land_albedo!(albedo, temperature; snow_free_albedo = 0.2, snow_albedo = 0.65,
                                 melting_temperature = 273.15, temperature_range = 10)
    grid = albedo.grid
    FT = eltype(grid)
    launch!(architecture(grid), grid, :xy, _snow_proxy_land_albedo!, albedo, temperature,
            convert(FT, snow_free_albedo), convert(FT, snow_albedo - snow_free_albedo),
            convert(FT, melting_temperature), convert(FT, temperature_range))
    return nothing
end

# The lower boundary of the radiation: the mosaic of land, open ocean and sea ice on the exchange grid, at the
# interface temperatures of this `update_state!`, interpolated to the panels. The fractions are those that
# weight the net fluxes of the atmosphere, and the land's albedo and emissivity are those it absorbs with.
function publish_radiative_surface!(radiation::YinYangGrayRadiation, coupled_model, regridder, latitude_longitude)
    interfaces = coupled_model.interfaces
    exchange_grid = interfaces.exchanger.grid
    FT = eltype(exchange_grid)
    optics = map(value -> convert(FT, value), radiation.surface_optics)

    launch!(architecture(exchange_grid), exchange_grid, :xy, _mosaic_radiative_surface!,
            latitude_longitude,
            mosaic_sea_surface(coupled_model),
            mosaic_land_surface(interfaces.atmosphere_land_interface, coupled_model),
            optics,
            latitude_longitude.downwelling_shortwave,
            exchange_grid,
            coupled_model.clock)

    interpolate_to_yin_yang!(radiation.surface.temperature, regridder, latitude_longitude.surface_temperature)
    interpolate_to_yin_yang!(radiation.surface.emissivity, regridder, latitude_longitude.surface_emissivity)
    interpolate_to_yin_yang!(radiation.surface.albedo, regridder, latitude_longitude.surface_albedo)

    return nothing
end

#####
##### Checkpointing
#####

# The ledger accumulates over the integration. The downwelling fluxes of the last radiation solve are what
# the surfaces absorb until the next one, so a restored model needs them before its first step.
exchange_restart_fields(exchange::MoistHydrostaticExchange) = exchange_restart_fields(exchange, exchange.radiation)
exchange_restart_fields(exchange, ::Nothing) = exchange.ledger
exchange_restart_fields(exchange, radiation::YinYangGrayRadiation) =
    merge(exchange.ledger, (downwelling_shortwave = radiation.downwelling.shortwave,
                            downwelling_longwave = radiation.downwelling.longwave))

Oceananigans.prognostic_state(exchange::MoistHydrostaticExchange) =
    Oceananigans.prognostic_state(exchange_restart_fields(exchange))

function Oceananigans.restore_prognostic_state!(exchange::MoistHydrostaticExchange, from)
    Oceananigans.restore_prognostic_state!(exchange_restart_fields(exchange), from)
    return exchange
end

Oceananigans.restore_prognostic_state!(exchange::MoistHydrostaticExchange, ::Nothing) = exchange
