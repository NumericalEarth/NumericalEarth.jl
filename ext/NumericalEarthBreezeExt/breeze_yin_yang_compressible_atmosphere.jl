#####
##### Surface exchange and radiation of Breeze's `CompressibleDynamics` on a `YinYangGrid`, with the surfaces
##### on a global latitude-longitude exchange grid
#####
##### PROVISIONAL (AMIP pilot adapter), evaluated after `breeze_yin_yang_moist_hydrostatic_atmosphere.jl`, whose
##### exchange with the latitude-longitude grid, radiation mosaic and per-panel gray radiation it shares.
#####
##### The coupling of the other Breeze dynamics holds the surface fluxes in fields that are the bottom boundary
##### conditions of `ρu`, `ρv`, `ρE` and the moisture density. On a `YinYangGrid` Oceananigans refuses a field as a
##### boundary condition ("A Field used as a boundary condition on dimension 3 must span the whole boundary"), and
##### Breeze's radiative transfer models index one grid. The fluxes are therefore inserted once per coupling
##### step, before the dynamics, with the conversions that Breeze's own boundary conditions apply:
#####
#####     ρu₁ += Δt τˣ / Δz,   ρv₁ += Δt τʸ / Δz                       stress (N m⁻²)
#####     ρqᵉ₁ += Δt Jᵛ / Δz                                           vapor (kg m⁻² s⁻¹), unconverted
#####     ρθ₁ += Δt (ρᵈ / ρ) 𝒬ᵀ / (ĉᵖ Π Δz)                            sensible heat (W m⁻²)
#####     ρθₖ += Δt (ρᵈ / ρ) (ℱ⁻ - ℱ⁺) / (ĉᵖ Π Δz)                     radiative flux difference of layer k
#####
##### (`EnergyFluxBoundaryConditionFunction`: `Jᶿ = (ρᵈ / ρ) 𝒬ᵀ / (ĉᵖ Π)`, "the same conversion the ρE interior
##### forcing and the radiative flux divergence apply"; water enters its prognostic density unconverted.)
##### No bulk formula is defined here.
#####

using Breeze: CompressibleDynamics
using Breeze.Thermodynamics: MoistureMassFractions, vapor_equivalent_heat_capacity
using Oceananigans.Grids: znodes
using Oceananigans.Fields: ZeroField

const YinYangCompressibleAtmosphere = Breeze.AtmosphereModel{<:CompressibleDynamics, <:Any, <:Any, <:Any, <:YinYangGrid}

mutable struct CompressibleExchange{S, L}
    const surface_fluxes :: S  # net fluxes into the atmosphere on its grid: ρu, ρv (stress), ρE (sensible heat), ρqᵛᵉ (vapor)
    const ledger :: L          # what the sources added to each column, accumulated (two-dimensional)
    radiation                  # YinYangGrayRadiation or nothing
    column_physics             # nothing, or `column_physics(model, Δt)` applied after the dynamics of each coupling step
end

Base.summary(exchange::CompressibleExchange) =
    string("CompressibleExchange(radiation = ", isnothing(exchange.radiation) ? "nothing" : summary(exchange.radiation),
           ", column_physics = ", isnothing(exchange.column_physics) ? "nothing" : "after each step", ")")

function surface_coupling(dynamics::CompressibleDynamics, grid::YinYangGrid, microphysics, thermodynamic_constants)
    surface_fluxes = (ρu   = Field{Face, Center, Nothing}(grid),
                      ρv   = Field{Center, Face, Nothing}(grid),
                      ρE   = Field{Center, Center, Nothing}(grid),
                      ρqᵛᵉ = Field{Center, Center, Nothing}(grid))

    return (; boundary_conditions = NamedTuple(), forcing = NamedTuple(), surface_fluxes)
end

attach_coupled_radiation(model::YinYangCompressibleAtmosphere, coupling) =
    @set model.radiation = CompressibleExchange(coupling.surface_fluxes, surface_exchange_ledger(model.grid), nothing, nothing)

# The sources are applied by `time_step!` below, not by the tendencies
Breeze.AtmosphereModels.update_radiation!(::CompressibleExchange, model) = nothing
Breeze.AtmosphereModels.radiation_flux_divergence(::CompressibleExchange) = nothing

NumericalEarth.EarthSystemModels.materialize_earth_system_radiation!(atmosphere::Simulation{<:YinYangCompressibleAtmosphere}, ::Nothing) = atmosphere
NumericalEarth.EarthSystemModels.materialize_earth_system_radiation!(atmosphere::Simulation{<:YinYangCompressibleAtmosphere}, radiation) = atmosphere

exchange_of(model::YinYangCompressibleAtmosphere) = model.radiation :: CompressibleExchange

# The level pressures of a height-coordinate model are the averages of its layer pressures (Breeze's default)
struct PanelLayerPressures{P}
    layer_pressure :: P
end

Breeze.AtmosphereModels.dynamics_pressure(pressures::PanelLayerPressures) = pressures.layer_pressure
Breeze.AtmosphereModels.level_pressure_source(::PanelLayerPressures) = nothing

panel_pressures(model::YinYangCompressibleAtmosphere, r) =
    PanelLayerPressures(getregion(Breeze.AtmosphereModels.dynamics_pressure(model.dynamics), r))

#####
##### Insertion
#####

@kernel function _apply_compressible_surface_stress!(ρu, ρv, grid, τˣ, τʸ, Δt)
    i, j = @index(Global, NTuple)
    @inbounds begin
        Δz = Δzᶜᶜᶜ(i, j, 1, grid)
        ρu[i, j, 1] += Δt * τˣ[i, j, 1] / Δz
        ρv[i, j, 1] += Δt * τʸ[i, j, 1] / Δz
    end
end

@kernel function _apply_compressible_column_sources!(ρθ, ρqᵉ, ledger, grid, 𝒬ᵀ, Jᵛ, ℱ, ρᵈ, ρ, p, qᵛ, qˡ, constants, pˢᵗ, κ, Δt)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        for k in 1:Nz
            Δz = Δzᶜᶜᶜ(i, j, k, grid)
            q = MoistureMassFractions(qᵛ[i, j, k], qˡ[i, j, k])
            ĉᵖ = vapor_equivalent_heat_capacity(q, constants)
            Π = (p[i, j, k] / pˢᵗ)^κ
            heating = net_upward_radiation(i, j, k, ℱ) - net_upward_radiation(i, j, k+1, ℱ) + ifelse(k == 1, 𝒬ᵀ[i, j, 1], zero(Δz))
            ρθ[i, j, k] += Δt * (ρᵈ[i, j, k] / ρ[i, j, k]) * heating / (ĉᵖ * Π * Δz)
        end

        # Deposition onto the surface is limited to the vapor of the lowest layer; what is refused is recorded
        Δz = Δzᶜᶜᶜ(i, j, 1, grid)
        vapor = max(0, qᵛ[i, j, 1]) * ρ[i, j, 1] * Δz
        requested = Δt * Jᵛ[i, j, 1]
        evaporated = max(requested, -vapor)
        ρqᵉ[i, j, 1] += evaporated / Δz

        ledger.evaporation[i, j, 1] += convert(eltype(ledger.evaporation), evaporated)
        ledger.refused_deposition[i, j, 1] += convert(eltype(ledger.refused_deposition), evaporated - requested)
        ledger.sensible_heat[i, j, 1] += convert(eltype(ledger.sensible_heat), Δt * 𝒬ᵀ[i, j, 1])
        ledger.surface_radiation[i, j, 1] += convert(eltype(ledger.surface_radiation), Δt * net_upward_radiation(i, j, 1, ℱ))
        ledger.top_radiation[i, j, 1] += convert(eltype(ledger.top_radiation), Δt * net_upward_radiation(i, j, Nz+1, ℱ))
    end
end

function apply_surface_and_radiative_sources!(model::YinYangCompressibleAtmosphere, Δt)
    exchange = exchange_of(model)
    grid = model.grid
    dynamics = model.dynamics
    constants = model.thermodynamic_constants
    fluxes = exchange.surface_fluxes
    μ = model.microphysical_fields
    # 0M (InstantaneousPrecipitation) retains no condensate, so its microphysical fields have no qˡ

    FT = eltype(grid)
    κ = convert(FT, Breeze.dry_air_gas_constant(constants) / constants.dry_air.heat_capacity)
    pˢᵗ = convert(FT, dynamics.standard_pressure)
    Δt = convert(FT, Δt)

    for r in yin_yang_regions
        region = getregion(grid, r)
        arch = architecture(region)

        launch!(arch, region, :xy, _apply_compressible_surface_stress!,
                getregion(model.momentum.ρu, r), getregion(model.momentum.ρv, r), region,
                getregion(fluxes.ρu, r), getregion(fluxes.ρv, r), Δt)

        launch!(arch, region, :xy, _apply_compressible_column_sources!,
                getregion(model.formulation.potential_temperature_density, r),
                getregion(model.moisture_density, r),
                regional(exchange.ledger, r),
                region,
                getregion(fluxes.ρE, r),
                getregion(fluxes.ρqᵛᵉ, r),
                panel_radiative_fluxes(exchange.radiation, r),
                getregion(dynamics.dry_density, r),
                getregion(dynamics.total_density, r),
                getregion(Breeze.AtmosphereModels.dynamics_pressure(dynamics), r),
                getregion(μ.qᵛ, r),
                haskey(μ, :qˡ) ? getregion(μ.qˡ, r) : ZeroField(FT),
                constants, pˢᵗ, κ, Δt)
    end

    update_state!(model)

    return nothing
end

# One coupling step: radiation of the state at the start of the step, the sources of the step, the dynamics
# (whose time stepper applies the microphysics of the model itself), and column physics that a driver attached, if any.
function Oceananigans.TimeSteppers.time_step!(atmosphere::Simulation{<:YinYangCompressibleAtmosphere}, Δt)
    model = atmosphere.model
    exchange = exchange_of(model)
    solve_radiation!(exchange.radiation, model)
    apply_surface_and_radiative_sources!(model, Δt)
    time_step!(model, Δt)
    apply_column_physics!(exchange.column_physics, model, Δt)
    return nothing
end

apply_column_physics!(::Nothing, model, Δt) = nothing
apply_column_physics!(column_physics, model, Δt) = column_physics(model, Δt)

#####
##### Exchange of state
#####

NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(atmosphere::YinYangCompressibleAtmosphere, exchange_grid;
                                                                          correction = nothing) =
    yin_yang_component_exchanger(atmosphere, exchange_grid, correction)

# The faces are copied to the host: indexing them on a GPU grid is scalar indexing (job 20035)
lowest_level_height(grid::YinYangGrid) = (z = Array(znodes(getregion(grid, 1), Face())); (z[2] - z[1]) / 2)

function NumericalEarth.EarthSystemModels.surface_layer_height(atmosphere::YinYangCompressibleAtmosphere, exchange_grid)
    height = Field{Center, Center, Nothing}(exchange_grid)
    fill!(height, lowest_level_height(atmosphere.grid))
    return height
end

@kernel function _compressible_lowest_level_state!(state, grid, u, v, T, qᵛ, p, height)
    i, j = @index(Global, NTuple)
    @inbounds begin
        state.u[i, j, 1] = ℑxᶜᵃᵃ(i, j, 1, grid, u)
        state.v[i, j, 1] = ℑyᵃᶜᵃ(i, j, 1, grid, v)
        state.T[i, j, 1] = T[i, j, 1]
        state.q[i, j, 1] = qᵛ[i, j, 1]
        state.p[i, j, 1] = p[i, j, 1]
        state.height[i, j, 1] = height
    end
end

function NumericalEarth.EarthSystemModels.interpolate_state!(exchanger, exchange_grid, atmosphere::YinYangCompressibleAtmosphere, coupled_model)
    yin_yang = exchanger.regridder.yin_yang
    grid = atmosphere.grid
    u, v, _ = atmosphere.velocities
    p = Breeze.AtmosphereModels.dynamics_pressure(atmosphere.dynamics)
    qᵛ = atmosphere.microphysical_fields.qᵛ
    height = convert(eltype(grid), lowest_level_height(grid))

    for r in yin_yang_regions
        region = getregion(grid, r)
        launch!(architecture(region), region, :xy, _compressible_lowest_level_state!,
                regional(yin_yang, r), region, getregion(u, r), getregion(v, r), getregion(atmosphere.temperature, r),
                getregion(qᵛ, r), getregion(p, r), height)
    end

    exchange_lowest_level_state!(exchanger, atmosphere, coupled_model)

    return nothing
end

NumericalEarth.EarthSystemModels.InterfaceComputations.net_fluxes(atmosphere::YinYangCompressibleAtmosphere) =
    exchange_of(atmosphere).surface_fluxes

NumericalEarth.EarthSystemModels.update_net_fluxes!(coupled_model, atmosphere::YinYangCompressibleAtmosphere) =
    yin_yang_update_net_fluxes!(coupled_model, atmosphere)

#####
##### Checkpointing
#####

exchange_restart_fields(exchange::CompressibleExchange) = exchange_restart_fields(exchange, exchange.radiation)

Oceananigans.prognostic_state(exchange::CompressibleExchange) =
    Oceananigans.prognostic_state(exchange_restart_fields(exchange))

function Oceananigans.restore_prognostic_state!(exchange::CompressibleExchange, from)
    restore_converted_fields!(exchange_restart_fields(exchange), from)
    return exchange
end

Oceananigans.restore_prognostic_state!(exchange::CompressibleExchange, ::Nothing) = exchange
