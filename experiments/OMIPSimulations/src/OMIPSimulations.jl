module OMIPSimulations

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znode, Face
using Dates
using NCDatasets
using CUDA

using NumericalEarth
using NumericalEarth.Oceans: ocean_simulation, default_ocean_closure
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities:
    CATKEVerticalDiffusivity, CATKEMixingLength, CATKEEquation
using NumericalEarth.SeaIces: sea_ice_simulation
using NumericalEarth.EarthSystemModels: OceanSeaIceModel, Radiation,
    SimilarityTheoryFluxes,
    LinearStableStabilityFunction,
    MomentumBasedFrictionVelocity,
    ThreeEquationHeatFlux

using NumericalEarth.EarthSystemModels.InterfaceComputations:
    ComponentInterfaces,
    CoefficientBasedFluxes,
    COARELogarithmicSimilarityProfile,
    LargeYeagerTransferCoefficients,
    MomentumRoughnessLength,
    ScalarRoughnessLength,
    WindDependentWaveFormulation,
    TemperatureDependentAirViscosity,
    SimilarityScales,
    SeaIceAlbedo,
    FixedIterations,
    large_yeager_stability_functions,
    atmosphere_sea_ice_stability_functions

using NumericalEarth.Bathymetry: regrid_bathymetry, ORCAGrid
using NumericalEarth.DataWrangling: Metadatum, Metadata, DatasetRestoring,
                                    SurfaceFluxRestoring,
                                    EN4Monthly, ECCO4Monthly
using NumericalEarth.DataWrangling.WOA: WOAMonthly
using NumericalEarth.DataWrangling.ORCA: ORCA1
using NumericalEarth.DataWrangling.JRA55: MultiYearJRA55, JRA55NetCDFBackend,
                                          JRA55PrescribedAtmosphere
using NumericalEarth.Diagnostics: MixedLayerDepthField

export omip_simulation,
       add_omip_diagnostics!,
       compute_report_fields,
       compute_woa_bias

# Backwards-compatible restore for checkpoints saved before ClimaSeaIce 0.4.8
# (which added snow_thickness, snow_thermodynamics, snowfall to SeaIceModel).
# Old checkpoints lack :snow_thickness in the saved state; this override
# silently skips the missing field so pickup works across versions.
using ClimaSeaIce: SeaIceModel
import Oceananigans: restore_prognostic_state!

function restore_prognostic_state!(model::SeaIceModel, state)
    restore_prognostic_state!(model.clock, state.clock)
    restore_prognostic_state!(model.velocities, state.velocities)
    restore_prognostic_state!(model.ice_thickness, state.ice_thickness)
    restore_prognostic_state!(model.ice_concentration, state.ice_concentration)
    restore_prognostic_state!(model.tracers, state.tracers)
    restore_prognostic_state!(model.timestepper, state.timestepper)
    restore_prognostic_state!(model.ice_thermodynamics, state.ice_thermodynamics)
    restore_prognostic_state!(model.dynamics, state.dynamics)

    # New fields in ClimaSeaIce >= 0.4.8 — restore only if checkpoint contains them
    if hasproperty(state, :snow_thickness)
        restore_prognostic_state!(model.snow_thickness, state.snow_thickness)
    end

    return model
end

include("atmosphere.jl")
include("jra55_data_staging.jl")
include("omip_simulation.jl")
include("omip_diagnostics.jl")
include("report_fields.jl")

end # module
