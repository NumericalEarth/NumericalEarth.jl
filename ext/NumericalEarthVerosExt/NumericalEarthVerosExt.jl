module NumericalEarthVerosExt

using CondaPkg: CondaPkg
using PythonCall: Py, PyArray, pycovert, pyexec, pyimport

using Oceananigans: Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.Fields: Field
using Oceananigans.Grids: Bounded, Flat, Periodic, Center, Face, LatitudeLongitudeGrid

using NumericalEarth: NumericalEarth

include("veros_ocean_simulation.jl")
include("veros_state_exchanger.jl")

end # module NumericalEarthVerosExt
