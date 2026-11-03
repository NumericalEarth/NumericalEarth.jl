# Coupling a new component

A new model joins an `EarthSystemModel` by filling one of its roles: atmosphere, ocean, sea ice or land. The coupler owns
the regridding, the interface fluxes, the assembly of the net fluxes and the radiation. The component owns its storage
and its time stepping.

## The contract

Every component implements `time_step!(component, Δt)` and three methods:

| Method | Returns |
|:-------|:--------|
| `grid(model)` | The grid of the model. An Oceananigans `AbstractModel` has this already |
| `surface_state(model)` | A `NamedTuple` of the fields, on `grid(model)`, that the coupler reads |
| `net_fluxes(model)` | A `NamedTuple` of the fields, on `grid(model)`, that the coupler writes |

A model with its own storage or conventions adds up to two more, which do nothing by default:

| Method | Purpose |
|:-------|:--------|
| `update_surface_state!(state, model)` | Refresh `state`, the fields returned by `surface_state`, from the model's storage |
| `update_net_fluxes!(coupled_model, model)` | Hand the model its fluxes, in its own units, signs and staggering |

Here `model` is the bare model: the coupler unwraps a `Simulation` with `component_model`. The coupler calls
`surface_state` when it is built and keeps the fields, and an ocean or sea ice model returns its own fields at every call.

| Component | `update_surface_state!` | `update_net_fluxes!` |
|:----------|:------------------------|:---------------------|
| Slab ocean, ClimaSeaIce | Not needed | Not needed |
| Oceananigans ocean | Converts the temperature to potential temperature | Not needed: `net_fluxes` are its boundary conditions |
| Veros | Copies the current time level from Python | Applies signs and staggering, and copies the fluxes to Python |
| Breeze | Centers the velocities and refreshes the precipitation | Remaps the fluxes and writes the bottom boundary conditions |
| SpeedyWeather | Copies its surface layer | Remaps the fluxes, averaged over water, and writes its prognostic ocean |
| Slab land | Not needed | Splits the fluxes between its stores |
| Terrarium | Reads the top soil layer, in Kelvin | Fills Terrarium's input fields |

## The coupled time step

1. Each component advances with `time_step!(component, Δt)`.
2. The coupler calls `update_surface_state!`, and remaps `surface_state` to the exchange grid if the grids differ.
3. The interfaces compute the turbulent fluxes on the exchange grid.
4. The coupler assembles the net fluxes of each surface and adds the radiation.
5. The coupler calls `update_net_fluxes!`, so a component receives complete fluxes.

## What each role provides

| Role | `surface_state` | `net_fluxes` | Also required |
|:-----|:----------------|:-------------|:--------------|
| Ocean | `u`, `v` (m s⁻¹), `T` (`temperature_units`), `S` (g kg⁻¹) | `u`, `v` (m² s⁻²), `T` (K m s⁻¹), `S` (g kg⁻¹ m s⁻¹), `η` (m s⁻¹), `freshwater_heat_content` (K m s⁻¹), `u_coefficient`, `v_coefficient` (m s⁻¹, or `nothing`) | `reference_density` (kg m⁻³), `heat_capacity` (J kg⁻¹ K⁻¹) |
| Sea ice | `u`, `v` (m s⁻¹), `hi`, `hc`, `hs` (ice, consolidation and snow thickness, m), `ℵ` (concentration), `S` (g kg⁻¹), `T` (°C) | `top` with `heat` (W m⁻²), `snowfall` (kg m⁻² s⁻¹), `u`, `v` (N m⁻²), and `bottom` with `heat` (W m⁻²) | `clock` and `Δt` properties, `reference_density`, `heat_capacity`, `sea_ice_phase_transitions` (with `liquidus` and `reference_latent_heat`), `sea_ice_mass_fluxes` (`ice` and `snow`, kg m⁻² s⁻¹) |
| Atmosphere | `u`, `v` (m s⁻¹), `T` (K), `p` (Pa), `q` (kg kg⁻¹), `ℐꜜˢʷ`, `ℐꜜˡʷ` (W m⁻²), `Jʳⁿ`, `Jˢⁿ` (kg m⁻² s⁻¹), and optionally `h_bℓ` (m) | The model's own, filled in `update_net_fluxes!` | `thermodynamics_parameters`, `surface_layer_height` (m), `boundary_layer_height` (m) |
| Land | `T` (K), `𝒮` (saturation) | Not used: fill the model's inputs in `update_net_fluxes!` | A `fluxes.surface_energy_flux` property, or `land_surface_energy_flux` |

Fluxes and radiation are positive up. Precipitation (`Jʳⁿ`, `Jˢⁿ`, `snowfall`) is positive down, and `η` is positive into the
ocean.

An atmosphere reads `coupled_model.interfaces.surface_partition.fluxes`, which holds the partition-weighted
`sensible_heat`, `water_vapor`, `x_momentum` and `y_momentum` at cell centers of the exchange grid, and
`surface_partition.surface_temperature` in Kelvin. A land model reads `coupled_model.interfaces.atmosphere_land_interface`.

Some functions have defaults that a richer model can extend:

- `temperature_units`, for an ocean in Kelvin.
- `ocean_surface_temperature`, `ocean_surface_salinity`, `ocean_surface_velocities`, `ocean_temperature` and
  `ocean_salinity`, which read `surface_state` by default, for an ocean with more than a surface layer.
- `sea_ice_thickness` and `sea_ice_concentration`, which read `surface_state` by default.
- `intercepted_snowfall`, zero by default, and `sea_ice_ocean_stress`, for sea ice with dynamics.
- `land_temperature` and `land_saturation`, for a land model selecting its fields through the keyword `state_accessors`.

Fields and properties have the floating point type of the grid. Data that need more than this contract, like the time
interpolation of prescribed components, go through their own `ComponentExchanger` and `interpolate_state!`.

## A new ocean

`test/test_new_component.jl` writes one model per role from this contract alone. This mixed layer ocean is one of them. It
stores its fields in the form the coupler reads and writes, so it needs neither `update_surface_state!` nor
`update_net_fluxes!`.

```@example new_ocean
using NumericalEarth
using Oceananigans
using Oceananigans.Fields: ConstantField, ZeroField
using Oceananigans.TimeSteppers: TimeSteppers
using Oceananigans.Units

using NumericalEarth.EarthSystemModels: EarthSystemModels
using NumericalEarth.EarthSystemModels.InterfaceComputations: InterfaceComputations

struct MixedLayerOcean{G, T, F}
    grid :: G
    temperature :: T
    net_fluxes :: F
end

function MixedLayerOcean(grid)
    net_fluxes = (; u = CenterField(grid),
                    v = CenterField(grid),
                    T = CenterField(grid),
                    S = CenterField(grid),
                    η = CenterField(grid),
                    freshwater_heat_content = CenterField(grid),
                    u_coefficient = nothing,
                    v_coefficient = nothing)

    return MixedLayerOcean(grid, CenterField(grid), net_fluxes)
end

function TimeSteppers.time_step!(ocean::MixedLayerOcean, Δt)
    parent(ocean.temperature) .-= parent(ocean.net_fluxes.T) .* Δt ./ ocean.grid.Lz
    return nothing
end

Oceananigans.Grids.grid(ocean::MixedLayerOcean) = ocean.grid
EarthSystemModels.surface_state(ocean::MixedLayerOcean) = (; u = ZeroField(Float64), v = ZeroField(Float64), T = ocean.temperature, S = ConstantField(35.0))
InterfaceComputations.net_fluxes(ocean::MixedLayerOcean) = ocean.net_fluxes

EarthSystemModels.reference_density(::MixedLayerOcean) = 1025.0
EarthSystemModels.heat_capacity(::MixedLayerOcean) = 4000.0

grid = LatitudeLongitudeGrid(size = (8, 4, 1), longitude = (0, 360), latitude = (-40, 40), z = (-50, 0))
ocean = MixedLayerOcean(grid)
set!(ocean.temperature, 20)

atmosphere_grid = LatitudeLongitudeGrid(size = (8, 4), longitude = (0, 360), latitude = (-40, 40),
                                        topology = (Periodic, Bounded, Flat), halo = (3, 3))

atmosphere = PrescribedAtmosphere(atmosphere_grid, [0.0])
set!(atmosphere.velocities.u, 10)
set!(atmosphere.temperature, 318.15)
set!(atmosphere.specific_humidity, 0.06)
set!(atmosphere.pressure, 101325)

model = EarthSystemModel(; atmosphere, ocean)
time_step!(model, 10minutes)
model
```

## Conventions

The fluxes on the exchange grid are positive upward: a positive heat flux cools the surface, and an eastward wind
produces a negative `x_momentum`. A model that takes its fluxes positive into the ocean, like Veros, flips their sign
in `update_net_fluxes!`. A model that stores several time levels reads the current one, and a model with a staggered grid
maps each face to the face of the same location on the exchange grid. A freshwater flux on a model without a surface
volume flux acts on the surface salinity as a virtual salt flux.

## Different horizontal grids

`exchange_regridder(component_grid, exchange_grid)` chooses the regridder. It returns `nothing` when the two grids have
the same type and the same horizontal cell faces, whatever their vertical discretizations, and the coupler then reads
`surface_state` directly. Otherwise it returns a first-order conservative `ConservativeExchange`, which requires
`using ConservativeRegridding`, and the coupler remaps `surface_state` to the exchange grid, with vectors in geographic
components. In `update_net_fluxes!`, a model remaps its fluxes back with
`regrid_surface_field!(destination, regridder.exchange.to_component, source)`.

An atmosphere can lie on any horizontal grid. Ocean, sea ice and land must share the horizontal grid of the exchange grid.
