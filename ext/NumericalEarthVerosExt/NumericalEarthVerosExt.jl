module NumericalEarthVerosExt

using NumericalEarth
using CondaPkg: CondaPkg
using PythonCall: PythonCall, pyimport, pyexec, pyconvert, Py, PyArray
using Oceananigans: Oceananigans
using Oceananigans.Architectures: CPU
using Oceananigans.Grids: Bounded, Center, Face, Periodic, LatitudeLongitudeGrid, Flat
using Oceananigans.Fields: Field
using Oceananigans.Simulations: Simulations
using NumericalEarth: NumericalEarth
using NumericalEarth.Oceans: Oceans


include("veros_ocean_simulation.jl")
include("veros_state_exchanger.jl")

end # module NumericalEarthVerosExt
