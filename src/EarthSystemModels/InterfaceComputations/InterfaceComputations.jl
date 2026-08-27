module InterfaceComputations

using Adapt: Adapt, adapt
using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans
using Oceananigans.Fields: AbstractField, Field, Face, Center, FractionalIndices
using Oceananigans.Grids: Flat, topology, _node
using Oceananigans.Simulations: Simulation
using Oceananigans.Utils: KernelParameters, worksize

export
    ComponentInterfaces,
    SimilarityTheoryFluxes,
    FixedIterations,
    ConvergenceStopCriteria,
    MomentumRoughnessLength,
    ScalarRoughnessLength,
    LandRoughnessLength,
    LandZeroPlaneDisplacement,
    CoefficientBasedFluxes,
    SimilarityScales,
    PolynomialNeutralDragCoefficient,
    LargeYeagerTransferCoefficients,
    LinearStableStabilityFunction,
    SkinTemperature,
    BulkTemperature,
    DiffusiveFlux,
    InteriorDiffusivity,
    ConvectiveGustiness,
    SubgridVelocityCorrection,
    mahrt_sun_subgrid_velocity,
    atmosphere_ocean_stability_functions,
    atmosphere_land_stability_functions,
    atmosphere_sea_ice_stability_functions,
    large_yeager_stability_functions,
    compute_atmosphere_ocean_fluxes!,
    compute_atmosphere_sea_ice_fluxes!,
    compute_atmosphere_land_fluxes!,
    compute_sea_ice_ocean_fluxes!,
    BulkHumidity,
    SkinHumidity,
    FractionalHumidity,
    CriticalSaturation,
    DryLayerHumidity,
    StorageBasedDryLayerDepth,
    DryLayerVaporPistonVelocity,
    ConstantTortuosity,
    PowerLawTortuosity,
    ElevationCorrection,
    atmosphere_land_interface,
    # Sea ice-ocean heat flux formulations
    IceBathHeatFlux,
    ThreeEquationHeatFlux,
    # Friction velocity formulations
    MomentumBasedFrictionVelocity

using ..EarthSystemModels: EarthSystemModels,
                           default_gravitational_acceleration,
                           default_freshwater_density,
                           thermodynamics_parameters,
                           surface_layer_height,
                           boundary_layer_height

using ...NumericalEarth: stateindex

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

@inline function air_land_interface_radiation_state(::Nothing, ::Nothing, i, j, k, grid, time)
    z = zero(eltype(grid))
    return AirLandRadiationState(z, z, z, z, z)
end

#####
##### Utilities
#####

@kernel function _compute_fractional_indices!(indices_tuple, exchange_grid, source_grid)
    i, j = @index(Global, NTuple)
    kᴺ = size(exchange_grid, 3)
    X = _node(i, j, kᴺ + 1, exchange_grid, Center(), Center(), Face())
    if topology(source_grid) == (Flat, Flat, Flat)
        fractional_indices_ij = FractionalIndices(nothing, nothing, nothing)
    else
        fractional_indices_ij = FractionalIndices(X, source_grid, Center(), Center(), Center())
    end
    fi = indices_tuple.i
    fj = indices_tuple.j
    @inbounds begin
        if !isnothing(fi)
            fi[i, j, 1] = fractional_indices_ij.i
        end

        if !isnothing(fj)
            fj[i, j, 1] = fractional_indices_ij.j
        end
    end
end

function interface_kernel_parameters(grid)
    Sx, Sy, _ = worksize(grid)
    TX, TY, _ = topology(grid)
    single_column_grid = Sx == 1 && Sy == 1

    if single_column_grid
        kernel_parameters = KernelParameters(1:1, 1:1)
    else
        # Compute fluxes into halo regions (0:N+1) for non-Flat dimensions.
        # Flat dimensions have no halo cells, so only iterate over the interior.
        x_range = TX === Flat ? (1:Sx) : (0:Sx+1)
        y_range = TY === Flat ? (1:Sy) : (0:Sy+1)
        kernel_parameters = KernelParameters(x_range, y_range)
    end

    return kernel_parameters
end

# 2-D (surface) specialization of `NumericalEarth.stateindex`, pinning k = 1: a scalar
# (e.g. a prescribed measurement height or the 600 m BL-height fallback) passes through,
# and a 2-D `Field` (Breeze's per-column surface- or boundary-layer height) is read at
# column `(i, j)`. Used by the atmosphere–surface flux kernels to consume
# `surface_layer_height` / `h_bℓ` uniformly.
@inline state2dindex(a, i, j) = stateindex(a, i, j, 1)

# Turbulent fluxes
include("roughness_lengths.jl")
include("interface_states.jl")
include("dry_layer_humidity.jl")
include("compute_interface_state.jl")
include("similarity_theory_turbulent_fluxes.jl")
include("coefficient_based_turbulent_fluxes.jl")

# State exchanger and interfaces
include("state_exchanger.jl")

# Sea ice-ocean heat flux formulations
include("friction_velocity.jl")
include("sea_ice_ocean_heat_flux_formulations.jl")

include("component_interfaces.jl")
include("atmosphere_state_correction.jl")
include("atmosphere_interface_kernels.jl")
include("atmosphere_ocean_fluxes.jl")
include("atmosphere_sea_ice_fluxes.jl")
include("atmosphere_land_fluxes.jl")
include("sea_ice_ocean_fluxes.jl")

end # module
