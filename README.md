<!-- Title -->
<h1 align="center">
  NumericalEarth.jl
</h1>

<!-- description -->
<p align="center">
  <strong>🌎 A flexible framework for coupling Earth system model components with prescribed or prognostic data, built on <a href="https://github.com/CliMA/Oceananigans.jl">Oceananigans</a>.</strong>
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.7677442">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7677442.svg?style=flat-square" alt="DOI">
  </a>
  <a href="https://github.com/NumericalEarth/NumericalEarth.jl/actions/workflows/ci.yml">
    <img src="https://github.com/NumericalEarth/NumericalEarth.jl/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://numericalearth.github.io/NumericalEarthDocumentation/stable/">
    <img src="https://img.shields.io/badge/docs-stable-blue.svg?style=flat-square" alt="Stable docs">
  </a>
  <a href="https://numericalearth.github.io/NumericalEarthDocumentation/dev/">
    <img src="https://img.shields.io/badge/docs-dev-orange.svg?style=flat-square" alt="Dev docs">
  </a>
</p>

## Overview

NumericalEarth.jl provides infrastructure for running Earth system model components—ocean, atmosphere, sea ice, and others—coupled together or driven by prescribed datasets. The coupling interface is generic: plug in Oceananigans for ocean dynamics, ClimaSeaIce for sea ice, SpeedyWeather or other atmospheric models, or use reanalysis products like JRA55 and ERA5 as prescribed forcing.

The package handles the complexity of component communication: interpolating between grids, computing air-sea fluxes via similarity theory, managing radiative transfer, and synchronizing time-stepping across components with different temporal resolutions.

NumericalEarth.jl also serves as a sandbox for developing and testing interface parameterizations—bulk flux formulations, roughness length models, albedo schemes, and other boundary layer physics—in a modular environment where they can be validated against observations before deployment in production climate models.

## Installation instructions

NumericalEarth is a [registered Julia package](https://julialang.org/packages/). So to install it,

1. [Download Julia](https://julialang.org/downloads/) (version 1.10 or later).

2. Launch Julia and type

```julia
julia> using Pkg

julia> Pkg.add("NumericalEarth")
```

This installs the latest version that's _compatible with your current environment_.

## Data Wrangling

Running realistic Earth system simulations requires wrangling gigabytes of observational and reanalysis data into formats your model can ingest. NumericalEarth.jl abstracts away this pain. Point the package at a dataset and a target grid, and it handles the downloading, caching, and regridding automatically.

The `Metadatum` abstraction provides a unified interface: whether you're initializing ocean temperature from reanalysis or prescribing atmospheric boundary conditions, the workflow is the same.

```julia
using NumericalEarth
using Dates

# Load temperature from reanalysis on a specific date
T_init = Metadatum(:temperature; date=DateTime(1993, 1, 1), dataset=ECCO2Daily())

# Build a prescribed atmosphere from JRA55
atmosphere = JRA55PrescribedAtmosphere(arch)
```

## A core abstraction: `EarthSystemModel`

The coupling infrastructure is anchored by `EarthSystemModel`, which encapsulates the component models—ocean, sea ice, atmosphere—and specifies how they communicate. Each component can be either prognostic (time-stepped by its own dynamics) or prescribed (interpolated from data). The model handles flux computations at interfaces, grid interpolation between components, and synchronized time-stepping.

We conceive of `EarthSystemModel` as a model in its own right, not just a container for components. This means it works with all the Oceananigans tools you'd use for any other model—`run!(simulation)`, `Callback`, `Checkpointer`, output writers, and the rest.

To illustrate, here's a global ocean simulation driven by prescribed atmospheric reanalysis:

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

bathymetry = NumericalEarth.regrid_bathymetry(grid)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bathymetry))

# Build an ocean simulation initialized to the ECCO state estimate
ocean = NumericalEarth.ocean_simulation(grid)
start_date = DateTime(1993, 1, 1)
set!(ocean.model,
     T=NumericalEarth.Metadatum(:temperature; date=start_date, dataset=NumericalEarth.ECCO2Daily()),
     S=NumericalEarth.Metadatum(:salinity;    date=start_date, dataset=NumericalEarth.ECCO2Daily()))

# Couple the ocean to JRA55 atmospheric forcing
atmosphere = NumericalEarth.JRA55PrescribedAtmosphere(arch)
coupled_model = NumericalEarth.OceanOnlyModel(ocean; atmosphere)
simulation = Simulation(coupled_model, Δt=20minutes, stop_time=30days)
run!(simulation)
```

This simulation achieves approximately 8 simulated years per day of wall time on an Nvidia H100 GPU.

Since `ocean.model` is an `Oceananigans.HydrostaticFreeSurfaceModel`, we can leverage Oceananigans features directly. For example, to plot the surface speed at the end of the simulation:

```julia
u, v, w = ocean.model.velocities
speed = Field(sqrt(u^2 + v^2))
compute!(speed)

using GLMakie
heatmap(view(speed, :, :, ocean.model.grid.Nz), colorrange=(0, 0.5), colormap=:magma, nan_color=:lightgray)
```

which produces

![image](https://github.com/user-attachments/assets/4c484b93-38fe-4840-bf7d-63a3a59d29e1)

## Installation

NumericalEarth is not yet a registered package (we are working on it). To install from a Julia REPL:

```julia
julia> using Pkg

julia> Pkg.add("https://github.com/NumericalEarth/NumericalEarth.jl/")

julia> Pkg.instantiate()
```

Use `Pkg.add(url="https://github.com/NumericalEarth/NumericalEarth.jl.git", rev="main")` to install the latest development version.

## Citing

If you use NumericalEarth for your research, teaching, or fun, we'd be grateful if you cite the Oceananigans overview paper submitted to the Journal of Advances in Modeling Earth Systems:

> "High-level, high-resolution ocean modeling at all scales with Oceananigans"
>
> by Gregory L. Wagner, Simone Silvestri, Navid C. Constantinou, Ali Ramadhan, Jean-Michel Campin,
> Chris Hill, Tomas Chor, Jago Strong-Wright, Xin Kai Lee, Francis Poulin, Andre Souza, Keaton J. Burns,
> Siddhartha Bishnu, John Marshall, and Raffaele Ferrari
>
> arXiv:[2502.14148](https://doi.org/10.48550/arXiv.2502.14148)

<details><summary>bibtex</summary>
  <pre><code>@article{Oceananigans-overview-paper-2025,
  title = {{High-level, high-resolution ocean modeling at all scales with Oceananigans}},
  author = {G. L. Wagner and S. Silvestri and N. C. Constantinou and A. Ramadhan and J.-M. Campin and C. Hill and T. Chor and J. Strong-Wright and X. K. Lee and F. Poulin and A. Souza and K. J. Burns and S. Bishnu and J. Marshall and R. Ferrari},
  journal = {arXiv preprint},
  year = {2025},
  archivePrefix = {arXiv},
  eprint = {2502.14148},
  doi = {10.48550/arXiv.2502.14148},
  notes = {submitted to the Journal of Advances in Modeling Earth Systems},
}</code></pre>
</details>
