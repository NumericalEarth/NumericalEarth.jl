module InterfaceComputations

using Oceananigans
using Oceananigans.Fields: AbstractField
using Oceananigans.Utils: KernelParameters
using Adapt

export
    ComponentInterfaces,
    SimilarityTheoryFluxes,
    MomentumRoughnessLength,
    ScalarRoughnessLength,
    CoefficientBasedFluxes,
    SkinTemperature,
    BulkTemperature,
    atmosphere_ocean_stability_functions,
    atmosphere_sea_ice_stability_functions,
    compute_atmosphere_ocean_fluxes!,
    compute_atmosphere_sea_ice_fluxes!,
    compute_sea_ice_ocean_fluxes!,
    # Sea ice-ocean heat flux formulations
    IceBathHeatFlux,
    ThreeEquationHeatFlux,
    # Friction velocity formulations
    MomentumBasedFrictionVelocity

using ..EarthSystemModels: default_gravitational_acceleration,
                           default_freshwater_density,
                           thermodynamics_parameters,
                           surface_layer_height,
                           boundary_layer_height

import NumericalEarth: stateindex
import Oceananigans.Simulations: initialize!

#####
##### Functions extended by component models
#####

net_fluxes(::Nothing) = nothing

#####
##### Radiation hooks: declared here so the turbulent flux kernels can
##### resolve them at parse time. The `Radiations` module extends them
##### with concrete methods for `PrescribedRadiation`.
#####

# `nothing` fallback (radiation is off). Concrete methods for
# `PrescribedRadiation` (and future radiation types) are added in `Radiations`.
@inline kernel_radiation_properties(::Nothing) = nothing

@inline function air_sea_interface_radiation_state(::Nothing, ::Nothing, i, j, k, grid, time)
    z = zero(eltype(grid))
    return (σ = z, α = z, ϵ = z, ℐꜜˢʷ = z, ℐꜜˡʷ = z)
end

@inline function air_sea_ice_interface_radiation_state(::Nothing, ::Nothing, i, j, k, grid, time)
    z = zero(eltype(grid))
    return (σ = z, α = z, ϵ = z, ℐꜜˢʷ = z, ℐꜜˡʷ = z)
end

#####
##### Utilities
#####

function interface_kernel_parameters(grid)
    Nx, Ny, _ = size(grid)
    TX, TY, _ = topology(grid)
    single_column_grid = Nx == 1 && Ny == 1

    if single_column_grid
        kernel_parameters = KernelParameters(1:1, 1:1)
    else
        # Compute fluxes into halo regions (0:N+1) for non-Flat dimensions.
        # Flat dimensions have no halo cells, so only iterate over the interior.
        x_range = TX === Flat ? (1:Nx) : (0:Nx+1)
        y_range = TY === Flat ? (1:Ny) : (0:Ny+1)
        kernel_parameters = KernelParameters(x_range, y_range)
    end

    return kernel_parameters
end

# Turbulent fluxes
include("roughness_lengths.jl")
include("interface_states.jl")
include("compute_interface_state.jl")
include("similarity_theory_turbulent_fluxes.jl")
include("coefficient_based_turbulent_fluxes.jl")

# State exchanger and interfaces
include("state_exchanger.jl")

# Sea ice-ocean heat flux formulations
include("friction_velocity.jl")
include("sea_ice_ocean_heat_flux_formulations.jl")

include("component_interfaces.jl")
include("atmosphere_ocean_fluxes.jl")
include("atmosphere_sea_ice_fluxes.jl")
include("sea_ice_ocean_fluxes.jl")

end # module
