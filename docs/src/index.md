# NumericalEarth.jl

🌎 Realistic ocean-only and coupled ocean-sea ice simulations driven by prescribed atmospheres and based on [Oceananigans](https://github.com/CliMA/Oceananigans.jl) and [ClimaSeaIce](https://github.com/CliMA/ClimaSeaIce.jl).

NumericalEarth implements a framework for coupling prescribed or prognostic representations of the ocean, sea ice, and atmosphere state.
Fluxes of heat, momentum, and freshwater are computed across the interfaces of its component models according to either Monin--Obukhov similarity theory,
or coefficient-based "bulk formula".
NumericalEarth builds off Oceananigans, which provides tools for gridded finite-volume computations on CPUs and GPUs and building ocean-flavored fluid dynamics simulations. ClimaSeaIce, which provides software for both stand-alone and coupled sea ice simulations, is also built with Oceananigans.

NumericalEarth's core abstraction is [`EarthSystemModel`](@ref), which encapsulates the ocean, sea ice, atmosphere, and their interfacial flux parameterizations.
The package also provides [`ocean_simulation`](@ref) and [`sea_ice_simulation`](@ref), utilities for building component simulations that can be run standalone or coupled through [`EarthSystemModel`](@ref).

NumericalEarth is written in Julia. The effort started by the [Climate Modeling Alliance](https://clima.caltech.edu) and heroic external collaborators and evolved into a community effort.

## Installation

NumericalEarth is a [registered Julia package](https://julialang.org/packages/). So to install it,

1. [Download Julia](https://julialang.org/downloads/) (version 1.10 or later).

2. Launch Julia and type

```julia
julia> using Pkg

julia> Pkg.add("NumericalEarth")
```

This installs the latest version that's _compatible with your current environment_.

Use `Pkg.add(url="https://github.com/NumericalEarth/NumericalEarth.jl.git", rev="main")` to install the latest development version.

!!! compat "Julia 1.10 is required"
    NumericalEarth requires Julia 1.10 or later.

## Quick start

NumericalEarth can be used in four complementary ways:

1. Build a standalone ocean simulation with [`ocean_simulation`](@ref).
2. Build a standalone atmosphere simulation with [`atmosphere_simulation`](@ref).
3. Build a standalone sea ice simulation with [`sea_ice_simulation`](@ref).
4. Couple ocean, atmosphere, sea ice, and other components with [`EarthSystemModel`](@ref), `OceanOnlyModel`, `OceanSeaIceModel`, or `AtmosphereOceanModel`.

The following script implements a near-global ocean simulation initialized from the [ECCO state estimate](https://doi.org/10.5194/gmd-8-3071-2015) and coupled to a prescribed atmosphere derived from the [JRA55-do reanalysis](@cite tsujino2018jra):

```julia
using Oceananigans
using Oceananigans.Units
using Dates
using CUDA
import NumericalEarth

arch = GPU()
grid = LatitudeLongitudeGrid(arch,
                             size = (1440, 560, 10),
                             halo = (7, 7, 7),
                             longitude = (0, 360),
                             latitude = (-70, 70),
                             z = (-3000, 0))

bathymetry = NumericalEarth.regrid_bathymetry(grid) # builds gridded bathymetry based on ETOPO2022
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bathymetry))

# Build an ocean simulation initialized to the ECCO state estimate version 2 on Jan 1, 1993
ocean = NumericalEarth.ocean_simulation(grid)
start_date = DateTime(1993, 1, 1)
set!(ocean.model,
     NumericalEarth.MetadataSet(:temperature, :salinity;
                                dataset = NumericalEarth.ECCO2Daily(),
                                date    = start_date))

# Build and run an EarthSystemModel (with no sea ice component) forced by JRA55 reanalysis
atmosphere = NumericalEarth.JRA55PrescribedAtmosphere(arch)
coupled_model = NumericalEarth.OceanOnlyModel(ocean; atmosphere)
simulation = Simulation(coupled_model, Δt=20minutes, stop_time=30days)
run!(simulation)
```

The simulation above achieves approximately 8 simulated years per day of wall time on an Nvidia H100 GPU.

We can leverage `Oceananigans` features to plot the surface speed at the end of the simulation:

```julia
u, v, w = ocean.model.velocities
speed = Field(sqrt(u^2 + v^2))

using GLMakie
heatmap(view(speed, :, :, ocean.model.grid.Nz), colorrange=(0, 0.5), colormap=:magma, nan_color=:lightgray)
```

![image](https://github.com/user-attachments/assets/4c484b93-38fe-4840-bf7d-63a3a59d29e1)
