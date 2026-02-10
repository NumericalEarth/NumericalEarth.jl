# The `EarthSystemModel`

```@meta
DocTestSetup = quote
    using NumericalEarth
    using Oceananigans
end
```

The `EarthSystemModel` is the central abstraction in NumericalEarth.
It wraps an ocean, a sea ice component, and an atmosphere into a single object
that can be handed to `Simulation` and advanced with `run!`, just like any Oceananigans model!
Internally, it takes care of computing turbulent fluxes between components and communicating
those fluxes as boundary conditions.

Let's start with the simplest possible coupled model: an ocean column with no atmosphere and
no sea ice, we have an alias for that: an `OceanOnlyModel`.

```jldoctest esm
grid = RectilinearGrid(size=10, z=(-100, 0), topology=(Flat, Flat, Bounded))
ocean = ocean_simulation(grid, timestepper = :QuasiAdamsBashforth2)
model = OceanOnlyModel(ocean)

# output
EarthSystemModel{CPU}(time = 0 seconds, iteration = 0)
├── ocean: HydrostaticFreeSurfaceModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── atmosphere: Nothing
├── sea_ice: FreezingLimitedOceanTemperature{ClimaSeaIce.SeaIceThermodynamics.LinearLiquidus{Float64}}
└── interfaces: ComponentInterfaces
```

Even though we didn't specify atmosphere or sea ice, we got both.
The atmosphere is `Nothing` -- no atmospheric forcing at all.
The sea ice is `FreezingLimitedOceanTemperature`, which is not really a sea ice model:
it simply clamps the ocean temperature to the local freezing point, preventing supercooling.
No ice thickness or concentration is tracked.

Since `EarthSystemModel` conforms to the Oceananigans `AbstractModel` interface,
we can build a `Simulation`, attach callbacks and output writers, and run it:

```jldoctest esm
using Oceananigans.Units

simulation = Simulation(model, Δt=20minutes, stop_time=1hour)
run!(simulation)
model

# output
EarthSystemModel{CPU}(time = 1 hour, iteration = 3)
├── ocean: HydrostaticFreeSurfaceModel{CPU, RectilinearGrid}(time = 1 hour, iteration = 3)
├── atmosphere: Nothing
├── sea_ice: FreezingLimitedOceanTemperature{ClimaSeaIce.SeaIceThermodynamics.LinearLiquidus{Float64}}
└── interfaces: ComponentInterfaces
```

Three time steps of 20 minutes each, and we've reached 1 hour.

## Three constructors for three levels of complexity

NumericalEarth provides three constructors.
They all return the same type -- `EarthSystemModel` -- but differ in what they ask you to provide.

### `OceanOnlyModel`

```@docs
OceanOnlyModel
```

We already used this one above. It takes an ocean simulation and optionally a prescribed
atmosphere. Sea ice is set to `FreezingLimitedOceanTemperature` automatically.
This is the right starting point when you don't need sea ice evolution.

### `OceanSeaIceModel`

```@docs
OceanSeaIceModel
```

When you want prognostic sea ice, use `OceanSeaIceModel`. It takes an ocean simulation and
a sea ice component as positional arguments:

```jldoctest esm
sea_ice = FreezingLimitedOceanTemperature()
model = OceanSeaIceModel(ocean, sea_ice)

# output
EarthSystemModel{CPU}(time = 0 seconds, iteration = 0)
├── ocean: HydrostaticFreeSurfaceModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── atmosphere: Nothing
├── sea_ice: FreezingLimitedOceanTemperature{ClimaSeaIce.SeaIceThermodynamics.LinearLiquidus{Float64}}
└── interfaces: ComponentInterfaces
```

Here we passed `FreezingLimitedOceanTemperature` again, but a "real" sea ice simulation, 
built with [`sea_ice_simulation`](@ref) which wraps `ClimaSeaIce.SeaIceModel` in an
Oceananigans `Simulation`, would also work.

### `EarthSystemModel`

```@docs
EarthSystemModel
```

The full constructor takes positional arguments `(atmosphere, ocean, sea_ice)` and
gives access to every knob: radiation parameters, reference densities, heat capacities,
and -- most importantly -- the `interfaces` keyword, which controls how fluxes are computed.

## Customizing flux formulations

The `interfaces` field of an `EarthSystemModel` is a `ComponentInterfaces` object.
If you don't pass one, it gets built automatically with default settings:
similarity-theory fluxes for the atmosphere--surface interface and a three-equation
thermodynamic model for the sea ice--ocean interface.

To change the defaults, construct `ComponentInterfaces` yourself and pass it in.
For example, to use constant transfer coefficients instead of similarity theory:

```jldoctest esm
atmosphere_ocean_fluxes = CoefficientBasedFluxes(drag_coefficient=2e-3)
interfaces = NumericalEarth.EarthSystemModels.ComponentInterfaces(nothing, ocean;
                                                                  atmosphere_ocean_fluxes)
model = OceanOnlyModel(ocean; interfaces)

# output
EarthSystemModel{CPU}(time = 0 seconds, iteration = 0)
├── ocean: HydrostaticFreeSurfaceModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── atmosphere: Nothing
├── sea_ice: FreezingLimitedOceanTemperature{ClimaSeaIce.SeaIceThermodynamics.LinearLiquidus{Float64}}
└── interfaces: ComponentInterfaces
```

Or to use similarity theory with specific stability functions:

```jldoctest esm
stability_functions = NumericalEarth.EarthSystemModels.atmosphere_ocean_stability_functions(Float64)
atmosphere_ocean_fluxes = SimilarityTheoryFluxes(; stability_functions)
interfaces = NumericalEarth.EarthSystemModels.ComponentInterfaces(nothing, ocean;
                                                                  atmosphere_ocean_fluxes)
model = OceanOnlyModel(ocean; interfaces)

# output
EarthSystemModel{CPU}(time = 0 seconds, iteration = 0)
├── ocean: HydrostaticFreeSurfaceModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── atmosphere: Nothing
├── sea_ice: FreezingLimitedOceanTemperature{ClimaSeaIce.SeaIceThermodynamics.LinearLiquidus{Float64}}
└── interfaces: ComponentInterfaces
```

The choices for stability functions are:
- `atmosphere_ocean_stability_functions`: based on Edson et al. (2013), the default for ocean surfaces.
- `atmosphere_sea_ice_stability_functions`: tuned for atmosphere--sea ice interactions.
- `nothing`: neutral boundary layer (no stability correction).

For the sea ice--ocean interface, the two available heat flux formulations are
`ThreeEquationHeatFlux` (default) and `IceBathHeatFlux`.
See the [Interface fluxes](@ref "Turbulent fluxes at component interfaces") page for a full treatment.

## What happens at construction

When you call any of the three constructors, a few things happen behind the scenes:

1. Certain callbacks are stripped from the ocean and sea ice simulations
   (`stop_time_exceeded`, `nan_checker`, etc.), because the coupled model manages
   stopping conditions itself.
2. If no `interfaces` was provided, a `ComponentInterfaces` object is built with defaults.
3. The ocean temperature field is clamped above the local freezing point.
4. Interface fluxes are computed for the initial state.

## How time-stepping works

Each call to `time_step!` advances the coupled system by `Δt`:

1. The sea ice is stepped forward (if present).
2. The ocean is stepped forward.
3. The atmosphere is stepped forward (if prognostic).
4. The clock is ticked.
5. Component states are interpolated onto the shared exchange grid, interface fluxes
   are recomputed, and the resulting boundary conditions are communicated back to each component.

## Checkpointing

`EarthSystemModel` supports Oceananigans' checkpointing infrastructure.
The functions `prognostic_state` and `restore_prognostic_state!` capture and restore
the full state of all components -- ocean, atmosphere, sea ice, clock, and interfaces --
so simulations can be restarted from any saved checkpoint.
