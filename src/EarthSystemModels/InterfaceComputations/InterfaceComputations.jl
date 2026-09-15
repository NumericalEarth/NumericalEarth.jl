module InterfaceComputations

using Adapt: Adapt
using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: @kernel, @index
using Oceananigans: Oceananigans, location
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: AbstractField, Field, Face, Center, FractionalIndices
using Oceananigans.Grids: Flat, Periodic, halo_size, topology, _node
using Oceananigans.Simulations: Simulation
using Oceananigans.Utils: KernelParameters, worksize

export
    ComponentInterfaces,
    SimilarityTheoryFluxes,
    FixedIterations,
    ConvergenceStopCriteria,
    MomentumRoughnessLength,
    ScalarRoughnessLength,
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
    AltitudeCorrection,
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
    TX, TY, _ = topology(source_grid)
    Nx, Ny, _ = size(source_grid)
    Hx, Hy, _ = halo_size(source_grid)
    Sx, Sy, _ = worksize(exchange_grid)
    halo_column = (i < 1) | (i > Sx) | (j < 1) | (j > Sy)

    fi = indices_tuple.i
    fj = indices_tuple.j
    @inbounds begin
        if !isnothing(fi)
            fi[i, j, 1] = clamp_fractional_index(fractional_indices_ij.i, TX(), Nx, Hx, halo_column)
        end

        if !isnothing(fj)
            fj[i, j, 1] = clamp_fractional_index(fractional_indices_ij.j, TY(), Ny, Hy, halo_column)
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

# Halo columns read the component's own halo, so the index is held where interpolation can
# reach: `⌊f⌋` and `⌊f⌋ + 1` must both lie within `1 - H` and `N + H`.
@inline clamp_fractional_index(::Nothing, topo, N, H, halo_column) = nothing

@inline function clamp_fractional_index(fractional_index, topo, N, H, halo_column)
    FT = typeof(fractional_index)
    lowest = convert(FT, 1 - H)
    highest = prevfloat(convert(FT, N + H))
    clamped = halo_column & !(topo isa Periodic)
    return ifelse(clamped, clamp(fractional_index, lowest, highest), fractional_index)
end

# 2-D (surface) specialization of `NumericalEarth.stateindex`, pinning k = 1
@inline state2dindex(a, i, j) = stateindex(a, i, j, 1)
@inline state2dindex(a, i, j, grid, time) = stateindex(a, i, j, 1, grid, time, (Center, Center, Nothing))

# Functions are resolved at the topmost center: a `Nothing` vertical location yields a two-tuple node.
@inline state2dindex(a::Function, i, j, grid, time) = stateindex(a, i, j, size(grid, 3), grid, time, (Center, Center, Center))

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
