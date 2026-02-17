using KernelAbstractions: @kernel, @index

using Oceananigans.TimeSteppers: tick!
using Oceananigans.Utils: launch!
using Oceananigans: Field, compute!

import Oceananigans.TimeSteppers: time_step!
import Oceananigans.Advection: cell_advection_timescale

using NumericalEarth.EarthSystemModels: AtmosphereOceanModel, SlabOcean

const BreezeAtmosphereOceanModel = AtmosphereOceanModel{<:Breeze.AtmosphereModel}

#####
##### Constructor
#####

"""
    AtmosphereOceanModel(atmosphere::Breeze.AtmosphereModel, ocean::SlabOcean)

Construct an `AtmosphereOceanModel` coupling a Breeze `AtmosphereModel` with a `SlabOcean`.

The constructor extracts surface flux operations from the atmosphere boundary conditions
using `BoundaryConditionOperation`, and wraps them in `Field`s for storage.

The SST field in the `SlabOcean` should be the same field used in the atmosphere's
bulk flux boundary conditions, so that updates to SST are automatically seen by the
atmosphere on the next time step.
"""
function NumericalEarth.AtmosphereOceanModel(atmosphere::Breeze.AtmosphereModel, ocean::SlabOcean)
    # Extract surface flux operations using BoundaryConditionOperation
    # Sensible heat flux: Q_sensible from static energy density BCs
    ρe = Breeze.static_energy_density(atmosphere)
    Q_sensible_op = Oceananigans.BoundaryConditionOperation(ρe, :bottom, atmosphere)

    # Latent heat flux: Q_latent = ℒ * J_vapor
    ρqᵗ = atmosphere.moisture_density
    constants = atmosphere.thermodynamic_constants
    θ₀ = atmosphere.dynamics.reference_state.potential_temperature
    ℒˡ = Breeze.Thermodynamics.liquid_latent_heat(θ₀, constants)
    J_vapor_op = Oceananigans.BoundaryConditionOperation(ρqᵗ, :bottom, atmosphere)
    Q_latent_op = ℒˡ * J_vapor_op

    # Net heat flux (sensible + latent)
    Q_net_op = Q_sensible_op + Q_latent_op

    # Wrap in Fields for computation/storage
    Q_sensible = Field(Q_sensible_op)
    Q_latent = Field(Q_latent_op)
    Q_net = Field(Q_net_op)

    ocean_surface_fluxes = (; Q_sensible, Q_latent, Q_net)

    arch = atmosphere.grid.architecture
    clock = deepcopy(atmosphere.clock)

    return AtmosphereOceanModel(arch, clock, atmosphere, ocean, ocean_surface_fluxes)
end

#####
##### Time stepping
#####

function time_step!(model::BreezeAtmosphereOceanModel, Δt; callbacks=[])
    # Step the atmosphere (uses current SST in BCs)
    time_step!(model.atmosphere, Δt)

    # Compute ocean surface fluxes from atmospheric boundary conditions
    compute_ocean_surface_fluxes!(model)

    # Step the slab ocean SST
    step_slab_ocean!(model.ocean, model.ocean_surface_fluxes, Δt)

    # Advance the clock
    tick!(model.clock, Δt)

    return nothing
end

function compute_ocean_surface_fluxes!(model::BreezeAtmosphereOceanModel)
    compute!(model.ocean_surface_fluxes.Q_sensible)
    compute!(model.ocean_surface_fluxes.Q_latent)
    compute!(model.ocean_surface_fluxes.Q_net)
    return nothing
end

#####
##### Slab ocean update
#####

@kernel function _step_slab_ocean!(T, Q_net, Δt, ρ, cₚ, H)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T[i, j, 1] -= Δt * Q_net[i, j, 1] / (ρ * cₚ * H)
    end
end

function step_slab_ocean!(ocean::SlabOcean, fluxes, Δt)
    T = ocean.sea_surface_temperature
    grid = T.grid
    arch = grid.architecture

    launch!(arch, grid, :xy, _step_slab_ocean!,
            T, fluxes.Q_net, Δt,
            ocean.density, ocean.heat_capacity, ocean.mixed_layer_depth)

    return nothing
end

#####
##### CFL wizard support
#####

cell_advection_timescale(model::BreezeAtmosphereOceanModel) =
    cell_advection_timescale(model.atmosphere)
