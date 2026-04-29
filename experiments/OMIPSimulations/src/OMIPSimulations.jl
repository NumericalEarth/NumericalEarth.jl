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
       add_ryf_sxthdegree_diagnostics!,
       compute_report_fields,
       compute_woa_bias,
       strait_transports,
       strait_sections,
       StraitSection

include("atmosphere.jl")
include("jra55_data_staging.jl")
include("omip_simulation.jl")
include("omip_diagnostics.jl")
include("report_fields.jl")
include("strait_transports.jl")

end # module
