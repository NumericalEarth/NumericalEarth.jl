using Oceananigans.Grids: Center
using Breeze.AtmosphereModels: thermodynamic_density, dynamics_density, surface_pressure
using Breeze.TerrainFollowingDiscretization: TerrainFollowingGrid
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

# The MOST reference height is the lowest cell-center elevation, Δz(i,j,1)/2,
# filled per column on-device.
@kernel function _fill_surface_layer_height!(zref, atmos_grid)
    i, j = @index(Global, NTuple)
    @inbounds zref[i, j, 1] = Oceananigans.zspacing(i, j, 1, atmos_grid, Center(), Center(), Center()) / 2
end

function NumericalEarth.EarthSystemModels.surface_layer_height(atmosphere::BreezeAtmosphere, exchange_grid)
    zref = Oceananigans.Field{Center, Center, Nothing}(exchange_grid)
    arch = architecture(exchange_grid)
    launch!(arch, exchange_grid, interface_kernel_parameters(exchange_grid),
            _fill_surface_layer_height!, zref, atmosphere.grid)
    # Per-column field only where terrain makes it vary; otherwise one uniform value,
    # read off the device via a bulk host copy (no scalar GPU indexing).
    if atmosphere.grid isa TerrainFollowingGrid
        return zref
    else
        return first(Array(Oceananigans.interior(zref)))
    end
end

NumericalEarth.EarthSystemModels.surface_layer_height(atmos::BreezeAtmosphereSim, exchange_grid) =
    NumericalEarth.EarthSystemModels.surface_layer_height(component_model(atmos), exchange_grid)

# The boundary-layer height is diagnosed per column by the turbulence closure
# at each step when it provides one. Fall back to a 600 m constant otherwise.
diagnosed_boundary_layer_height(closure_fields) = 600
diagnosed_boundary_layer_height(closure_fields::NamedTuple{names}) where names =
    :zi in names ? closure_fields.zi : 600

NumericalEarth.EarthSystemModels.boundary_layer_height(atmosphere::BreezeAtmosphere) =
    diagnosed_boundary_layer_height(atmosphere.closure_fields)

NumericalEarth.EarthSystemModels.boundary_layer_height(atmos::BreezeAtmosphereSim) =
    NumericalEarth.EarthSystemModels.boundary_layer_height(component_model(atmos))

#####
##### ComponentExchanger: state fields for flux computations
#####

function NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(atmosphere::BreezeAtmosphere, exchange_grid;
                                                                                   correction = nothing)
    state = (; u    = Oceananigans.CenterField(exchange_grid),
               v    = Oceananigans.CenterField(exchange_grid),
               T    = Oceananigans.CenterField(exchange_grid),
               p    = Oceananigans.CenterField(exchange_grid),
               q    = Oceananigans.CenterField(exchange_grid),
               ℐꜜˢʷ = Oceananigans.CenterField(exchange_grid),
               ℐꜜˡʷ = Oceananigans.CenterField(exchange_grid),
               Jʳⁿ  = Oceananigans.CenterField(exchange_grid),
               Jˢⁿ  = Oceananigans.CenterField(exchange_grid))

    correction = NumericalEarth.EarthSystemModels.InterfaceComputations.materialize_correction(correction, exchange_grid, atmosphere)
    return ComponentExchanger(state, nothing, correction)
end

NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(atmos::BreezeAtmosphereSim, exchange_grid; kw...) =
    NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(component_model(atmos), exchange_grid; kw...)

#####
##### Interpolate atmospheric state onto exchange grid
#####

@kernel function _interpolate_breeze_state!(state, u, v, T, ρqᵛᵉ, ρ₀, p₀)
    i, j = @index(Global, NTuple)

    @inbounds begin
        state.u[i, j, 1]    = u[i, j, 1]
        state.v[i, j, 1]    = v[i, j, 1]
        state.T[i, j, 1]    = T[i, j, 1]
        state.q[i, j, 1]    = ρqᵛᵉ[i, j, 1] / ρ₀[i, j, 1]
        state.p[i, j, 1]    = p₀
        state.ℐꜜˢʷ[i, j, 1] = 0
        state.ℐꜜˡʷ[i, j, 1] = 0
        state.Jʳⁿ[i, j, 1]  = 0
        state.Jˢⁿ[i, j, 1]  = 0
    end
end

function NumericalEarth.EarthSystemModels.interpolate_state!(exchanger, exchange_grid, atmosphere::BreezeAtmosphere, coupled_model)
    state = exchanger.state
    u, v, w = atmosphere.velocities
    T = atmosphere.temperature
    ρqᵛᵉ = atmosphere.moisture_density

    # Near-surface density (to convert moisture density ρqᵛ → specific humidity) and
    # surface pressure, via dynamics-generic accessors so coupling works for *both*
    # anelastic atmospheres (reference-state density) and compressible terrain-following
    # atmospheres (prognostic density). Reaching into
    # `dynamics.reference_state` directly is anelastic-only (it is `nothing` for
    # `CompressibleDynamics`).
    ρ₀ = dynamics_density(atmosphere.dynamics)
    p₀ = surface_pressure(atmosphere.dynamics)

    arch = architecture(exchange_grid)
    kernel_parameters = interface_kernel_parameters(exchange_grid)
    launch!(arch, exchange_grid, kernel_parameters,
            _interpolate_breeze_state!,
            state, u, v, T, ρqᵛᵉ, ρ₀, p₀)

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
##### Assemble ESM similarity-theory fluxes into Breeze bottom BCs
#####

@kernel function _assemble_net_atmosphere_fluxes!(net, ao_fluxes)
    i, j = @index(Global, NTuple)
    @inbounds begin
        τx = ao_fluxes.x_momentum[i, j, 1]
        τy = ao_fluxes.y_momentum[i, j, 1]
        Qc = ao_fluxes.sensible_heat[i, j, 1]
        Fv = ao_fluxes.water_vapor[i, j, 1]

        net.ρu[i, j, 1]  = τx
        net.ρv[i, j, 1]  = τy
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

    # Atmosphere-ocean fluxes (when an ocean interface is present).
    ao_interface = coupled_model.interfaces.atmosphere_ocean_interface
    if !isnothing(ao_interface)
        ao_fluxes = computed_fluxes(ao_interface)
        if !isnothing(ao_fluxes)
            launch!(arch, grid, params, _assemble_net_atmosphere_fluxes!, net, ao_fluxes)
        end
    end

    # Atmosphere-land fluxes (when a land interface is present).
    # We assume at most one surface type per cell, so the kernel writes
    # absolute values rather than accumulating; for full coverage with
    # both ocean and land present, a tile-fraction weighted assembly
    # would be needed.
    al_interface = coupled_model.interfaces.atmosphere_land_interface
    if !isnothing(al_interface)
        al_fluxes = computed_fluxes(al_interface)
        if !isnothing(al_fluxes)
            launch!(arch, grid, params, _assemble_net_atmosphere_fluxes!, net, al_fluxes)
        end
    end

    return nothing
end

#####
##### CFL wizard support
#####

Oceananigans.Advection.cell_advection_timescale(model::NumericalEarth.EarthSystemModel{<:Any, <:BreezeAtmosphere}) =
    cell_advection_timescale(model.atmosphere)

Oceananigans.Advection.cell_advection_timescale(model::NumericalEarth.EarthSystemModel{<:Any, <:BreezeAtmosphereSim}) =
    cell_advection_timescale(component_model(model.atmosphere))
