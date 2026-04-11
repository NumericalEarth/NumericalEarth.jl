module OMIPSimulations

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: znode, Face
using Dates
using NCDatasets
using CUDA

using NumericalEarth
using NumericalEarth.Oceans: ocean_simulation, default_ocean_closure
using NumericalEarth.SeaIces: sea_ice_simulation
using NumericalEarth.EarthSystemModels: OceanSeaIceModel, Radiation
using NumericalEarth.Bathymetry: regrid_bathymetry, ORCAGrid
using NumericalEarth.DataWrangling: Metadatum, Metadata, DatasetRestoring,
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

include("atmosphere.jl")
include("omip_simulation.jl")
include("omip_diagnostics.jl")
include("report_fields.jl")

end # module
