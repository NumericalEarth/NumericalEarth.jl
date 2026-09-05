module InterfaceComputations

using Adapt: Adapt
using Oceananigans: Oceananigans, location
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: AbstractField, Field, Face, Center
using Oceananigans.Grids: Flat, Periodic, topology
using Oceananigans.Simulations: Simulation
using Oceananigans.Utils: KernelParameters, worksize

export
    ComponentInterfaces,
    SurfacePartition,
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
include("surface_partition.jl")

# Sea ice-ocean heat flux formulations
include("friction_velocity.jl")
include("sea_ice_ocean_heat_flux_formulations.jl")

include("component_interfaces.jl")
include("atmosphere_state_correction.jl")
include("atmosphere_ocean_fluxes.jl")
include("atmosphere_sea_ice_fluxes.jl")
include("atmosphere_land_fluxes.jl")
include("sea_ice_ocean_fluxes.jl")

end # module
