module NumericalEarthZarrExt

using Zarr: Zarr, zopen
using NCDatasets: NCDataset, defDim, defVar
using Oceananigans.Fields: Field, set!
using Oceananigans.Grids: Center, Face, x_domain, y_domain, λnodes, φnodes
using NumericalEarth: NumericalEarth
using NumericalEarth.DataWrangling: native_grid

const Bathymetry = NumericalEarth.Bathymetry
const CopernicusDEM = NumericalEarth.DataWrangling.CopernicusDEM

include("zarr_utils.jl")
include("copernicus_dem.jl")

end # module NumericalEarthZarrExt
