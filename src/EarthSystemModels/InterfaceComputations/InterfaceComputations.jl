module InterfaceComputations

using Adapt: Adapt
using Oceananigans: Oceananigans, location
using Oceananigans.Architectures: architecture
using Oceananigans.Fields: AbstractField, Field, Face, Center
using Oceananigans.Grids: Flat, Periodic, topology
using Oceananigans.OutputReaders: FieldTimeSeries, FlavorOfFTS, cpu_interpolating_time_indices
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
    FreeConvectionMomentumStabilityFunction,
    FreeConvectionScalarStabilityFunction,
    SkinTemperature,
    BulkTemperature,
    DiffusiveFlux,
    SoilConductiveFlux,
    EnergyBalanceTemperature,
    SoilSkinTemperature,
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
    PlantAvailableWaterStress,
    DryLayerHumidity,
    StorageBasedDryLayerDepth,
    DryLayerVaporPistonVelocity,
    ConstantTortuosity,
    PowerLawTortuosity,
    CanopyConductanceHumidity,
    CompositeSurfaceHumidity,
    CanopyAirSpace,
    DiagnosticCanopyAir,
    PrognosticCanopyAir,
    DiagnosticSkin,
    PrognosticSkin,
    CanopyInterception,
    AbstractUndercanopyConductance,
    ConstantUndercanopyConductance,
    AreaIndexUndercanopyConductance,
    FrictionVelocityUndercanopyConductance,
    SellersSoilResistance,
    LitterResistance,
    TiledLandInterface,
    bare_canopy_air_space,
    leaf_area_index_cover_fraction,
    FarquharPhotosynthesis,
    AbstractStomatalConductance,
    MedlynConductance,
    JarvisConductance,
    AbstractAbsorbedPAR,
    PrescribedAbsorbedPAR,
    InteractiveAbsorbedPAR,
    PlainArrhenius,
    PeakedArrheniusParameters,
    HeskelParameters,
    PeakedArrhenius,
    AltitudeCorrection,
    atmosphere_land_interface,
    # Sea ice-ocean heat flux formulations
    IceBathHeatFlux,
    ThreeEquationHeatFlux,
    # Friction velocity formulations
    MomentumBasedFrictionVelocity

using ..EarthSystemModels: EarthSystemModels,
                           surface_retention_curve,
                           effective_saturation,
                           default_gravitational_acceleration,
                           default_freshwater_density,
                           default_gas_constant,
                           default_dry_air_molar_mass,
                           celsius_to_kelvin,
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

# A halo column of the exchange grid indexes outside a regional component's interior by design:
# the value there comes from the component's own halo, and so from its boundary conditions.
# Interpolation reads cells `⌊f⌋` and `⌊f⌋ + 1`, so a halo column is held within `1 - H` and
# `N + H`. Interior columns are never clamped, so a component that does not cover the exchange
# grid fails instead of being interpolated over.
@inline clamp_fractional_index(::Nothing, topo, N, H, halo_column) = nothing

@inline function clamp_fractional_index(fractional_index, topo, N, H, halo_column)
    FT = typeof(fractional_index)
    lowest = convert(FT, 1 - H)
    highest = prevfloat(convert(FT, N + H))
    clamped = halo_column & !(topo isa Periodic)
    return ifelse(clamped, clamp(fractional_index, lowest, highest), fractional_index)
end

# 2-D (surface) specialization of `NumericalEarth.stateindex`, pinning k = 1: a 2-D
# `Field` (Breeze's per-column surface- or boundary-layer height, a per-cell roughness
# length) is read at column `(i, j)`, and anything uniform over grid points — a scalar
# height, a flux formulation such as `MomentumRoughnessLength` — passes through.
# Used by the atmosphere–surface flux kernels.
@inline state2dindex(a, i, j) = a
@inline state2dindex(a::AbstractArray, i, j) = stateindex(a, i, j, 1)

# Turbulent fluxes
include("roughness_lengths.jl")
include("moisture_stress.jl")
include("interface_states.jl")
include("dry_layer_humidity.jl")
include("photosynthesis.jl")
include("stomatal_conductance.jl")
include("absorbed_par.jl")
include("canopy_conductance.jl")
include("composite_surface_humidity.jl")
include("canopy_air_space.jl")
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
include("atmosphere_ocean_fluxes.jl")
include("atmosphere_sea_ice_fluxes.jl")
include("atmosphere_land_fluxes.jl")
include("tiled_land_interface.jl")
include("sea_ice_ocean_fluxes.jl")

end # module
