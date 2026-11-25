#####
##### Surface exchange and radiation of Breeze's moist `HydrostaticDynamics` on a `YinYangGrid` of pressure levels
##### (`PressureLevelVerticalDiscretization`), with the surfaces on a global latitude-longitude exchange grid
#####
##### PROVISIONAL (AMIP pilot adapter), evaluated after `breeze_yin_yang_moist_hydrostatic_atmosphere.jl` and
##### `breeze_yin_yang_compressible_atmosphere.jl`, whose exchange with the latitude-longitude grid, radiation container,
##### insertion kernels and `CompressibleExchange` it reuses. Only what differs is defined here.
#####
##### On a `PressureLevelGrid` the grid's `Δz` is the physical depth Δp / (g ρ) of each cell and `dynamics.density` is
##### the physical density ρ of moist air, so the insertion of the compressible adapter applies unchanged with ρ in
##### place of both ρᵈ and ρ (the energy flux density of `HydrostaticDynamics` is `DryAirDensity`, whose ratio is 1):
#####
#####     u₁ += Δt τˣ / (ρ Δz),   v₁ += Δt τʸ / (ρ Δz)                  stress (N m⁻²), on the prognostic velocities
#####     ρθₖ += Δt (ℱ⁻ - ℱ⁺ + 𝒬ᵀ δₖ₁) / (ĉᵖ Π Δz)                     sensible heat and radiative flux difference
#####
##### The vapor E = Δt Jᵛ (limited to the vapor of the lowest layer, what is refused recorded) enters through the total-mass
##### coordinate: `Breeze.AtmosphereModels.surface_vapor_mass_change!` adds it to the lowest layer with its enthalpy at the
##### temperature of that layer, changes the surface pressure by g E and re-expresses every layer at its new pressure.
##### No bulk formula is defined here.
#####

using Breeze.PressureLevelDiscretization: PressureLevelVerticalDiscretization
using Breeze.HydrostaticEquations: MoistHydrostaticDynamics
using Oceananigans.Grids: Bounded

# The vapor through the ground of a coupling step, kg m⁻² (one scratch field per atmosphere grid)
const surface_vapor_scratch = IdDict{Any, Any}()
surface_vapor_field(grid) = get!(() -> Field{Center, Center, Nothing}(grid), surface_vapor_scratch, grid)

@kernel function _apply_pressure_level_column_sources!(ρθ, ledger, evaporation, grid, 𝒬ᵀ, Jᵛ, ℱ, ρ, p, qᵛ, qˡ, constants, pˢᵗ, κ, Δt)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        for k in 1:Nz
            Δz = Δzᶜᶜᶜ(i, j, k, grid)
            q = MoistureMassFractions(qᵛ[i, j, k], qˡ[i, j, k])
            ĉᵖ = vapor_equivalent_heat_capacity(q, constants)
            Π = (p[i, j, k] / pˢᵗ)^κ
            heating = net_upward_radiation(i, j, k, ℱ) - net_upward_radiation(i, j, k+1, ℱ) + ifelse(k == 1, 𝒬ᵀ[i, j, 1], zero(Δz))
            ρθ[i, j, k] += Δt * heating / (ĉᵖ * Π * Δz)
        end

        # Deposition onto the surface is limited to the vapor of the lowest layer; what is refused is recorded
        Δz = Δzᶜᶜᶜ(i, j, 1, grid)
        vapor = max(0, qᵛ[i, j, 1]) * ρ[i, j, 1] * Δz
        requested = Δt * Jᵛ[i, j, 1]
        evaporated = max(requested, -vapor)
        evaporation[i, j, 1] = evaporated

        ledger.evaporation[i, j, 1] += evaporated
        ledger.refused_deposition[i, j, 1] += evaporated - requested
        ledger.sensible_heat[i, j, 1] += Δt * 𝒬ᵀ[i, j, 1]
        ledger.surface_radiation[i, j, 1] += Δt * net_upward_radiation(i, j, 1, ℱ)
        ledger.top_radiation[i, j, 1] += Δt * net_upward_radiation(i, j, Nz+1, ℱ)
    end
end

const PressureLevelYinYangGrid = YinYangGrid{<:Any, <:Any, <:Any, <:Bounded, <:PressureLevelVerticalDiscretization}

const YinYangPressureLevelAtmosphere = Breeze.AtmosphereModel{<:MoistHydrostaticDynamics, <:Any, <:Any, <:Any, <:PressureLevelYinYangGrid}

pressure_level_surface_coupling(grid) =
    (; boundary_conditions = NamedTuple(), forcing = NamedTuple(),
       surface_fluxes = (ρu   = Field{Face, Center, Nothing}(grid),
                         ρv   = Field{Center, Face, Nothing}(grid),
                         ρE   = Field{Center, Center, Nothing}(grid),
                         ρqᵛᵉ = Field{Center, Center, Nothing}(grid)))

surface_coupling(::HydrostaticDynamics, grid::PressureLevelYinYangGrid, microphysics, thermodynamic_constants) =
    pressure_level_surface_coupling(grid)

surface_coupling(::HydrostaticDynamics, grid::PressureLevelYinYangGrid, ::Breeze.BulkMicrophysics, thermodynamic_constants) =
    pressure_level_surface_coupling(grid)

# The microphysics of the model runs in its own time stepper (the generic interface): nothing is taken from it
attach_coupled_radiation(model::YinYangPressureLevelAtmosphere, coupling) =
    @set model.radiation = CompressibleExchange(coupling.surface_fluxes, surface_exchange_ledger(model.grid), nothing, nothing)

# The panel radiation that the exchange holds reads the pressures of the coordinate surfaces of each panel
# (`getregion(model, r)`, whose `level_pressure_source` is that of `HydrostaticDynamics`)
Breeze.AtmosphereModels.uses_level_pressure(::CompressibleExchange) = true

NumericalEarth.EarthSystemModels.materialize_earth_system_radiation!(atmosphere::Simulation{<:YinYangPressureLevelAtmosphere}, ::Nothing) = atmosphere
NumericalEarth.EarthSystemModels.materialize_earth_system_radiation!(atmosphere::Simulation{<:YinYangPressureLevelAtmosphere}, radiation) = atmosphere
NumericalEarth.EarthSystemModels.materialize_earth_system_radiation!(atmosphere::Simulation{<:YinYangPressureLevelAtmosphere},
                                                                     ::Breeze.RadiativeTransferModel) = atmosphere

exchange_of(model::YinYangPressureLevelAtmosphere) = model.radiation :: CompressibleExchange

function apply_surface_and_radiative_sources!(model::YinYangPressureLevelAtmosphere, Δt)
    exchange = exchange_of(model)
    grid = model.grid
    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    fluxes = exchange.surface_fluxes
    μ = model.microphysical_fields
    u, v, _ = model.velocities

    FT = eltype(grid)
    κ = convert(FT, Breeze.dry_air_gas_constant(constants) / constants.dry_air.heat_capacity)
    pˢᵗ = convert(FT, dynamics.standard_pressure)
    Δt = convert(FT, Δt)
    evaporation = surface_vapor_field(grid)

    for r in yin_yang_regions
        region = getregion(grid, r)
        arch = architecture(region)
        ρ = getregion(dynamics.density, r)

        launch!(arch, region, :xy, _apply_surface_stress!,
                getregion(u, r), getregion(v, r), region, getregion(fluxes.ρu, r), getregion(fluxes.ρv, r), ρ, Δt)

        # 0M (InstantaneousPrecipitation) retains no condensate, so its microphysical fields have no qˡ
        launch!(arch, region, :xy, _apply_pressure_level_column_sources!,
                getregion(model.formulation.potential_temperature_density, r),
                regional(exchange.ledger, r),
                getregion(evaporation, r),
                region,
                getregion(fluxes.ρE, r),
                getregion(fluxes.ρqᵛᵉ, r),
                panel_radiative_fluxes(exchange.radiation, r),
                ρ,
                getregion(dynamics.pressure, r),
                getregion(μ.qᵛ, r),
                haskey(μ, :qˡ) ? getregion(μ.qˡ, r) : ZeroField(FT),
                constants, pˢᵗ, κ, Δt)
    end

    # The vapor through the total-mass coordinate (re-diagnoses the state)
    Breeze.AtmosphereModels.surface_vapor_mass_change!(dynamics, model, evaporation)

    return nothing
end

function Oceananigans.TimeSteppers.time_step!(atmosphere::Simulation{<:YinYangPressureLevelAtmosphere}, Δt)
    model = atmosphere.model
    exchange = exchange_of(model)
    solve_radiation!(exchange.radiation, model)
    apply_surface_and_radiative_sources!(model, Δt)
    time_step!(model, Δt)
    apply_column_physics!(exchange.column_physics, model, Δt)
    return nothing
end

#####
##### Exchange of state
#####

NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(atmosphere::YinYangPressureLevelAtmosphere, exchange_grid;
                                                                          correction = nothing) =
    yin_yang_component_exchanger(atmosphere, exchange_grid, correction)

# The height of the lowest cell centers above the ground, (Φ₁ - Φˢ) / g, averaged over the owned and halo cells of
# the first panel on the host. The surface layer height of the exchange is one number (as for the height grids).
function pressure_level_lowest_height(atmosphere)
    dynamics = atmosphere.dynamics
    g = atmosphere.thermodynamic_constants.gravitational_acceleration
    Φ = Array(interior(getregion(dynamics.geopotential, 1)))[:, :, 1]
    Φˢ = Array(interior(getregion(dynamics.surface_geopotential, 1)))[:, :, 1]
    return sum(Φ .- Φˢ) / (g * length(Φ))
end

function NumericalEarth.EarthSystemModels.surface_layer_height(atmosphere::YinYangPressureLevelAtmosphere, exchange_grid)
    height = Field{Center, Center, Nothing}(exchange_grid)
    fill!(height, pressure_level_lowest_height(atmosphere))
    return height
end

function NumericalEarth.EarthSystemModels.interpolate_state!(exchanger, exchange_grid, atmosphere::YinYangPressureLevelAtmosphere, coupled_model)
    yin_yang = exchanger.regridder.yin_yang
    grid = atmosphere.grid
    u, v, _ = atmosphere.velocities
    p = atmosphere.dynamics.pressure
    qᵛ = atmosphere.microphysical_fields.qᵛ
    height = convert(eltype(grid), pressure_level_lowest_height(atmosphere))

    for r in yin_yang_regions
        region = getregion(grid, r)
        launch!(architecture(region), region, :xy, _compressible_lowest_level_state!,
                regional(yin_yang, r), region, getregion(u, r), getregion(v, r), getregion(atmosphere.temperature, r),
                getregion(qᵛ, r), getregion(p, r), height)
    end

    exchange_lowest_level_state!(exchanger, atmosphere, coupled_model)

    return nothing
end

NumericalEarth.EarthSystemModels.InterfaceComputations.net_fluxes(atmosphere::YinYangPressureLevelAtmosphere) =
    exchange_of(atmosphere).surface_fluxes

NumericalEarth.EarthSystemModels.update_net_fluxes!(coupled_model, atmosphere::YinYangPressureLevelAtmosphere) =
    yin_yang_update_net_fluxes!(coupled_model, atmosphere)
