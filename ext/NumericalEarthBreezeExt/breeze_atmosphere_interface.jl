using Oceananigans.Grids: Center
using Oceananigans.Operators: ℑxᶠᵃᵃ, ℑyᵃᶠᵃ
using Oceananigans.Fields: compute!
using Breeze.AtmosphereModels: thermodynamic_density, dynamics_pressure,
                               specific_humidity, surface_precipitation_flux
using Breeze.TerrainFollowingDiscretization: TerrainFollowingGrid
using GPUArraysCore: @allowscalar
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters
using NumericalEarth.EarthSystemModels: component_model
using NumericalEarth.EarthSystemModels.InterfaceComputations: interface_kernel_parameters

const BreezeAtmosphere    = Breeze.AtmosphereModel
const BreezeAtmosphereSim = Simulation{<:Breeze.AtmosphereModel}

# Callers in this file work in terms of the underlying `Breeze.AtmosphereModel`,
# and Simulation-typed entry points delegate to the Model-typed methods via the
# generic `component_model` unwrap defined in `EarthSystemModels`.

#####
##### Thermodynamics parameters
#####

# This is a _hack_: the parameters should ideally be derived from Breeze.ThermodynamicConstants,
# but the ESM similarity theory expects CliMA Thermodynamics parameters.
NumericalEarth.EarthSystemModels.thermodynamics_parameters(::BreezeAtmosphere) = AtmosphereThermodynamicsParameters(Float64)
NumericalEarth.EarthSystemModels.thermodynamics_parameters(atmos::BreezeAtmosphereSim) =
    NumericalEarth.EarthSystemModels.thermodynamics_parameters(component_model(atmos))

#####
##### Surface layer and boundary layer height
#####

# The MOST reference height is the lowest cell-center elevation above ground, ½·Δz(i,j,1).
# On a terrain-following grid this varies column-to-column, so fill it on-device; on any
# other grid it is horizontally uniform. Built once and cached in `interfaces.properties`.
@kernel function _fill_surface_layer_height!(z₁, atmos_grid)
    i, j = @index(Global, NTuple)
    @inbounds z₁[i, j, 1] = Oceananigans.zspacing(i, j, 1, atmos_grid, Center(), Center(), Center()) / 2
end

function NumericalEarth.EarthSystemModels.surface_layer_height(atmosphere::BreezeAtmosphere, exchange_grid)
    grid = atmosphere.grid
    if grid isa TerrainFollowingGrid
        # Terrain makes the AGL first-cell height vary per column → a 2-D field.
        z₁ = Oceananigans.Field{Center, Center, Nothing}(exchange_grid)
        launch!(architecture(exchange_grid), exchange_grid, interface_kernel_parameters(exchange_grid),
                _fill_surface_layer_height!, z₁, grid)
        return z₁
    else
        # Horizontally uniform → one scalar. Read once here (cached, not per step), so the
        # single host index into a possibly-stretched device Δz array is fine under @allowscalar.
        return @allowscalar Oceananigans.zspacing(1, 1, 1, grid, Center(), Center(), Center()) / 2
    end
end

NumericalEarth.EarthSystemModels.surface_layer_height(atmos::BreezeAtmosphereSim, exchange_grid) =
    NumericalEarth.EarthSystemModels.surface_layer_height(component_model(atmos), exchange_grid)

# Fallback boundary-layer height for the surface-flux convective gustiness, used when the
# atmosphere's turbulence closure diagnoses no z_i (e.g. closure = nothing).
const default_boundary_layer_height = 600 # m

# The boundary-layer height is the per-column z_i diagnosed by the turbulence closure when
# it provides one (e.g. Breeze's ScaleAdaptiveTKE writes it into its `zi` closure field).
NumericalEarth.EarthSystemModels.boundary_layer_height(atmosphere::BreezeAtmosphere) =
    hasproperty(atmosphere.closure_fields, :zi) ? atmosphere.closure_fields.zi :
                                                  default_boundary_layer_height

NumericalEarth.EarthSystemModels.boundary_layer_height(atmos::BreezeAtmosphereSim) =
    NumericalEarth.EarthSystemModels.boundary_layer_height(component_model(atmos))

#####
##### ComponentExchanger: state fields for flux computations
#####

function NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(atmosphere::BreezeAtmosphere, exchange_grid;
                                                                                   correction = nothing)
    # Breeze's surface rain-flux diagnostic (positive down, kg m⁻² s⁻¹); schemes with no
    # precipitating species define no method — fall back to an inert zero field.
    # TODO: move the fallback into Breeze; add a snow analog (Jˢⁿ stays zero below).
    Jʳⁿ = if applicable(surface_precipitation_flux, atmosphere, atmosphere.microphysics)
        surface_precipitation_flux(atmosphere)
    else
        Oceananigans.CenterField(exchange_grid)
    end

    state = (; u    = Oceananigans.CenterField(exchange_grid),
               v    = Oceananigans.CenterField(exchange_grid),
               T    = Oceananigans.CenterField(exchange_grid),
               p    = Oceananigans.CenterField(exchange_grid),
               q    = Oceananigans.CenterField(exchange_grid),
               ℐꜜˢʷ = Oceananigans.CenterField(exchange_grid),
               ℐꜜˡʷ = Oceananigans.CenterField(exchange_grid),
               Jʳⁿ  = Jʳⁿ,
               Jˢⁿ  = Oceananigans.CenterField(exchange_grid))

    correction = NumericalEarth.EarthSystemModels.InterfaceComputations.materialize_correction(correction, exchange_grid, atmosphere)
    return ComponentExchanger(state, nothing, correction)
end

NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(atmos::BreezeAtmosphereSim, exchange_grid; kw...) =
    NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(component_model(atmos), exchange_grid; kw...)

#####
##### Interpolate atmospheric state onto exchange grid
#####

@kernel function _interpolate_breeze_state!(state, u, v, T, qᵛ, p)
    i, j = @index(Global, NTuple)

    @inbounds begin
        state.u[i, j, 1]    = u[i, j, 1]
        state.v[i, j, 1]    = v[i, j, 1]
        state.T[i, j, 1]    = T[i, j, 1]
        state.q[i, j, 1]    = qᵛ[i, j, 1]
        state.p[i, j, 1]    = p[i, j, 1]
        state.ℐꜜˢʷ[i, j, 1] = 0
        state.ℐꜜˡʷ[i, j, 1] = 0
        state.Jˢⁿ[i, j, 1]  = 0
    end
end

function NumericalEarth.EarthSystemModels.interpolate_state!(exchanger, exchange_grid, atmosphere::BreezeAtmosphere, coupled_model)
    state = exchanger.state
    u, v, w = atmosphere.velocities
    T = atmosphere.temperature
    qᵛ = specific_humidity(atmosphere)

    # Breeze's diagnosed vapor mass fraction, not the scheme-dependent moisture prognostic;
    # `dynamics_pressure` gives the per-column pressure (a single scalar `surface_pressure`
    # would bias fluxes over terrain).
    p = dynamics_pressure(atmosphere.dynamics)

    arch = architecture(exchange_grid)
    kernel_parameters = interface_kernel_parameters(exchange_grid)
    launch!(arch, exchange_grid, kernel_parameters,
            _interpolate_breeze_state!,
            state, u, v, T, qᵛ, p)

    compute!(state.Jʳⁿ)   # refresh the rain diagnostic (no-op for the zero-field fallback)

    return nothing
end

NumericalEarth.EarthSystemModels.interpolate_state!(exchanger, exchange_grid, atmos::BreezeAtmosphereSim, coupled_model) =
    NumericalEarth.EarthSystemModels.interpolate_state!(exchanger, exchange_grid, component_model(atmos), coupled_model)

#####
##### Net fluxes: extract coupling flux fields from Breeze boundary conditions
#####

function NumericalEarth.EarthSystemModels.InterfaceComputations.net_fluxes(atmosphere::BreezeAtmosphere)
    # Momentum flux fields (direct FluxBoundaryCondition on ρu, ρv)
    ρu = atmosphere.momentum.ρu.boundary_conditions.bottom.condition
    ρv = atmosphere.momentum.ρv.boundary_conditions.bottom.condition

    # BulkDrag bottoms would break the extraction below and double-count the coupler's stress.
    ρu isa Oceananigans.Field || throw(ArgumentError(
        "the atmosphere's bottom momentum boundary condition is $(summary(ρu)), not a coupling " *
        "flux field, so its surface stress cannot come from the coupler. Build the atmosphere " *
        "without `bottom_drag_coefficient` (Breeze `BulkDrag`) when coupling to land or ocean."))

    # Energy flux field: ρe BC was converted to ρθ by Breeze's materialization,
    # wrapped in EnergyFluxBoundaryConditionFunction.
    # First .condition unwraps BoundaryCondition, second .condition extracts the
    # original field from EnergyFluxBoundaryConditionFunction.
    ρe = thermodynamic_density(atmosphere.formulation).boundary_conditions.bottom.condition.condition

    # Moisture flux field
    ρqᵛᵉ = atmosphere.moisture_density.boundary_conditions.bottom.condition

    return (; ρu, ρv, ρe, ρqᵛᵉ)
end

NumericalEarth.EarthSystemModels.InterfaceComputations.net_fluxes(atmos::BreezeAtmosphereSim) =
    NumericalEarth.EarthSystemModels.InterfaceComputations.net_fluxes(component_model(atmos))

#####
##### Assemble ESM similarity-theory fluxes into Breeze bottom BCs,
##### weighting each surface's contribution by the surface partition:
##### open ocean θ (1 - ℵ), sea ice θ ℵ, land (1 - θ).
#####

# Center-located partition-weighted stress, interpolated to the face-located ρu/ρv
# only after weighting.
@inline function ρτᶜᶜᶜ(i, j, k, grid, ρτᵃᵒ, ρτᵃⁱ, ρτᵃˡ, θ, ℵ)
    @inbounds begin
        θᵢ = θ[i, j, k]
        ℵᵢ = ℵ[i, j, k]
        return θᵢ * ((1 - ℵᵢ) * ρτᵃᵒ[i, j, k] + ℵᵢ * ρτᵃⁱ[i, j, k]) + (1 - θᵢ) * ρτᵃˡ[i, j, k]
    end
end

@kernel function _assemble_net_atmosphere_fluxes!(net, ao_fluxes, ai_fluxes, al_fluxes, θ, ℵ, grid)
    i, j = @index(Global, NTuple)
    @inbounds begin
        θᵢ = θ[i, j, 1]
        ℵᵢ = ℵ[i, j, 1]

        Qc = θᵢ * ((1 - ℵᵢ) * ao_fluxes.sensible_heat[i, j, 1] + ℵᵢ * ai_fluxes.sensible_heat[i, j, 1]) + (1 - θᵢ) * al_fluxes.sensible_heat[i, j, 1]
        Fv = θᵢ * ((1 - ℵᵢ) * ao_fluxes.water_vapor[i, j, 1]   + ℵᵢ * ai_fluxes.water_vapor[i, j, 1])   + (1 - θᵢ) * al_fluxes.water_vapor[i, j, 1]

        net.ρu[i, j, 1] = ℑxᶠᵃᵃ(i, j, 1, grid, ρτᶜᶜᶜ, ao_fluxes.x_momentum, ai_fluxes.x_momentum, al_fluxes.x_momentum, θ, ℵ)
        net.ρv[i, j, 1] = ℑyᵃᶠᵃ(i, j, 1, grid, ρτᶜᶜᶜ, ao_fluxes.y_momentum, ai_fluxes.y_momentum, al_fluxes.y_momentum, θ, ℵ)
        net.ρe[i, j, 1]  = Qc   # sensible heat only; latent heat handled by moisture flux
        net.ρqᵛᵉ[i, j, 1] = Fv
    end
end

NumericalEarth.EarthSystemModels.update_net_fluxes!(coupled_model, atmos::BreezeAtmosphereSim) =
    NumericalEarth.EarthSystemModels.update_net_fluxes!(coupled_model, component_model(atmos))

function NumericalEarth.EarthSystemModels.update_net_fluxes!(coupled_model, atmosphere::BreezeAtmosphere)
    net = coupled_model.interfaces.net_fluxes.atmosphere
    isnothing(net) && return nothing

    grid = atmosphere.grid
    arch = architecture(grid)
    params = interface_kernel_parameters(grid)

    interfaces = coupled_model.interfaces
    ao_fluxes = computed_fluxes(interfaces.atmosphere_ocean_interface)
    ai_fluxes = computed_fluxes(interfaces.atmosphere_sea_ice_interface)
    al_fluxes = computed_fluxes(interfaces.atmosphere_land_interface)
    θ = interfaces.surface_partition.ocean_fraction
    ℵ = sea_ice_concentration(coupled_model.sea_ice)

    launch!(arch, grid, params, _assemble_net_atmosphere_fluxes!,
            net, ao_fluxes, ai_fluxes, al_fluxes, θ, ℵ, grid)

    return nothing
end

#####
##### CFL wizard support
#####

Oceananigans.Advection.cell_advection_timescale(model::NumericalEarth.EarthSystemModel{<:Any, <:BreezeAtmosphere}) =
    cell_advection_timescale(model.atmosphere)

Oceananigans.Advection.cell_advection_timescale(model::NumericalEarth.EarthSystemModel{<:Any, <:BreezeAtmosphereSim}) =
    cell_advection_timescale(component_model(model.atmosphere))
