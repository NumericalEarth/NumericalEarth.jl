module NumericalEarthVerosExt

using CondaPkg: CondaPkg
using Downloads: Downloads
using Oceananigans: Oceananigans
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: Field, interior
using Oceananigans.Grids: Bounded, Periodic, Center, Face, LatitudeLongitudeGrid, halo_size
using Oceananigans.ImmersedBoundaries: GridFittedBottom, ImmersedBoundaryGrid
using NumericalEarth: NumericalEarth
using PythonCall: Py, PyArray, pyconvert, pyexec, pyimport

include("veros_ocean_simulation.jl")
include("veros_state_exchanger.jl")

end # module NumericalEarthVerosExt
