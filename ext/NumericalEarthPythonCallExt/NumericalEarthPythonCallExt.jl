module NumericalEarthPythonCallExt

using CondaPkg: CondaPkg
using Dates: DateTime, Second, year, firstdayofmonth
using DocStringExtensions: TYPEDSIGNATURES
using Downloads: Downloads
using KernelAbstractions: KernelAbstractions
using Oceananigans: Oceananigans
using Oceananigans.Architectures: CPU, GPU, architecture, device
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: Field, ZeroField, interior
using Oceananigans.Grids: Bounded, Flat, Periodic, Center, Face, LatitudeLongitudeGrid, halo_size
using Oceananigans.ImmersedBoundaries: GridFittedBottom, ImmersedBoundaryGrid
using Oceananigans.TimeSteppers: Clock, tick!
using Oceananigans.Utils: prettytime
using NumericalEarth: NumericalEarth
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters
using NumericalEarth.DataWrangling.ACE: ACE2ERA5, ace_file, ace2_era5_initial_condition_years
using NumericalEarth.EarthSystemModels: EarthSystemModels, sea_ice_concentration
using NumericalEarth.EarthSystemModels.InterfaceComputations: InterfaceComputations, ComponentExchanger,
                                                              exchange_over_water, regrid_over_water!
using NumericalEarth.Radiations: AtmosphereRadiation
using PythonCall: Py, PyArray, pybuiltins, pyconvert, pydict, pyexec, pyimport

include("veros_ocean_simulation.jl")
include("veros_state_exchanger.jl")
include("ace_atmosphere.jl")

end # module NumericalEarthPythonCallExt
