using Oceananigans.Grids: Center
using Breeze.AtmosphereModels: thermodynamic_density
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters
using NumericalEarth.EarthSystemModels.InterfaceComputations: interface_kernel_parameters

const BreezeAtmosphere = Breeze.AtmosphereModel

#####
##### Thermodynamics parameters
#####

# This is a _hack_: the parameters should ideally be derived from Breeze.ThermodynamicConstants,
# but the ESM similarity theory expects CliMA Thermodynamics parameters.
thermodynamics_parameters(::BreezeAtmosphere) = AtmosphereThermodynamicsParameters(Float64)

#####
##### Surface layer and boundary layer height
#####

# Height of the lowest atmospheric cell center (the "surface layer").
# Note: for stretched grids on GPU, this may require allowscalar.
function surface_layer_height(atmosphere::BreezeAtmosphere)
    grid = atmosphere.grid
    return Oceananigans.zspacing(1, 1, 1, grid, Center(), Center(), Center()) / 2
end

boundary_layer_height(::BreezeAtmosphere) = 600

#####
##### ComponentExchanger: state fields for flux computations
#####

function ComponentExchanger(atmosphere::BreezeAtmosphere, exchange_grid)
    state = (; u  = Oceananigans.CenterField(exchange_grid),
               v  = Oceananigans.CenterField(exchange_grid),
               T  = Oceananigans.CenterField(exchange_grid),
               p  = Oceananigans.CenterField(exchange_grid),
               q  = Oceananigans.CenterField(exchange_grid),
               ℐꜜˢʷ = Oceananigans.CenterField(exchange_grid),
               ℐꜜˡʷ = Oceananigans.CenterField(exchange_grid),
               Mp = Oceananigans.CenterField(exchange_grid))

    return ComponentExchanger(state, nothing)
end

#####
##### Interpolate atmospheric state onto exchange grid
#####

@kernel function _interpolate_breeze_state!(state, u, v, T, ρqᵗ, ρ₀, p₀)
    i, j = @index(Global, NTuple)

    @inbounds begin
        state.u[i, j, 1]  = u[i, j, 1]
        state.v[i, j, 1]  = v[i, j, 1]
        state.T[i, j, 1]  = T[i, j, 1]
        state.q[i, j, 1]  = ρqᵗ[i, j, 1] / ρ₀[i, j, 1]
        state.p[i, j, 1]  = p₀
        state.ℐꜜˢʷ[i, j, 1] = 0
        state.ℐꜜˡʷ[i, j, 1] = 0
        state.Mp[i, j, 1] = 0
    end
end

function interpolate_state!(exchanger, exchange_grid, atmosphere::BreezeAtmosphere, coupled_model)
    state = exchanger.state
    u, v, w = atmosphere.velocities
    T = atmosphere.temperature
    ρqᵗ = atmosphere.moisture_density

    # Reference state (anelastic dynamics)
    ref = atmosphere.dynamics.reference_state
    ρ₀ = ref.density
    p₀ = ref.surface_pressure

    arch = architecture(exchange_grid)
    kernel_parameters = interface_kernel_parameters(exchange_grid)
    launch!(arch, exchange_grid, kernel_parameters,
            _interpolate_breeze_state!,
            state, u, v, T, ρqᵗ, ρ₀, p₀)

    return nothing
end

#####
##### Net fluxes: extract coupling flux fields from Breeze boundary conditions
#####

function net_fluxes(atmosphere::BreezeAtmosphere)
    # Momentum flux fields (direct FluxBoundaryCondition on ρu, ρv)
    ρu = atmosphere.momentum.ρu.boundary_conditions.bottom.condition
    ρv = atmosphere.momentum.ρv.boundary_conditions.bottom.condition

    # Energy flux field: ρe BC was converted to ρθ by Breeze's materialization,
    # wrapped in EnergyFluxBoundaryConditionFunction.
    # First .condition unwraps BoundaryCondition, second .condition extracts the
    # original field from EnergyFluxBoundaryConditionFunction.
    ρe = thermodynamic_density(atmosphere.formulation).boundary_conditions.bottom.condition.condition

    # Moisture flux field (direct FluxBoundaryCondition on ρqᵗ)
    ρqᵗ = atmosphere.moisture_density.boundary_conditions.bottom.condition

    return (; ρu, ρv, ρe, ρqᵗ)
end

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
        net.ρqᵗ[i, j, 1] = Fv
    end
end

function update_net_fluxes!(coupled_model, atmosphere::BreezeAtmosphere)
    net = coupled_model.interfaces.net_fluxes.atmosphere
    isnothing(net) && return nothing

    ao_fluxes = computed_fluxes(coupled_model.interfaces.atmosphere_ocean_interface)
    isnothing(ao_fluxes) && return nothing

    grid = atmosphere.grid
    arch = architecture(grid)
    params = interface_kernel_parameters(grid)

    launch!(arch, grid, params, _assemble_net_atmosphere_fluxes!, net, ao_fluxes)
    return nothing
end

#####
##### CFL wizard support
#####

cell_advection_timescale(model::NumericalEarth.EarthSystemModel{<:Any, <:BreezeAtmosphere}) =
    cell_advection_timescale(model.atmosphere)
