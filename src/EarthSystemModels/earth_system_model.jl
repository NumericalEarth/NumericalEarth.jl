using Oceananigans
using Oceananigans.TimeSteppers: Clock
using Oceananigans: SeawaterBuoyancy
using ClimaSeaIce.SeaIceThermodynamics: melting_temperature
using KernelAbstractions: @kernel, @index

mutable struct EarthSystemModel{I, A, O, F, C, Arch} <: AbstractModel{Nothing, Arch}
    architecture :: Arch
    clock :: C
    atmosphere :: A
    sea_ice :: I
    ocean :: O
    interfaces :: F
end

const ESM = EarthSystemModel

function Base.summary(model::ESM)
    A = nameof(typeof(architecture(model)))
    return string("EarthSystemModel{$A}",
                  "(time = ", prettytime(model.clock.time), ", iteration = ", model.clock.iteration, ")")
end

function Base.show(io::IO, cm::ESM)

    if cm.sea_ice isa Simulation
        sea_ice_summary = summary(cm.sea_ice.model)
    else
        sea_ice_summary = summary(cm.sea_ice)
    end

    print(io, summary(cm), "\n")

    if cm.ocean isa Simulation
        ocean_summary = summary(cm.ocean.model)
    else
        ocean_summary = summary(cm.ocean)
    end

    print(io, "├── ocean: ", ocean_summary, "\n")
    print(io, "├── atmosphere: ", summary(cm.atmosphere), "\n")
    print(io, "├── sea_ice: ", sea_ice_summary, "\n")
    print(io, "└── interfaces: ", summary(cm.interfaces))
    return nothing
end

# Assumption: We have an ocean!
architecture(model::ESM)           = model.architecture
Base.eltype(model::ESM)            = Base.eltype(model.interfaces.exchanger.grid)
prettytime(model::ESM)             = prettytime(model.clock.time)
iteration(model::ESM)              = model.clock.iteration
timestepper(::ESM)                 = nothing
default_included_properties(::ESM) = tuple()
prognostic_fields(cm::ESM)         = nothing
fields(::ESM)                      = NamedTuple()
default_clock(TT)                   = Oceananigans.TimeSteppers.Clock{TT}(0, 0, 1)

function reset!(model::ESM)
    reset!(model.ocean)
    return nothing
end

# Make sure to initialize the exchanger here
function initialization_update_state!(model::ESM)
    initialize!(model.interfaces.exchanger, model)
    update_state!(model)
    return nothing
end

function initialize!(model::ESM)
    # initialize!(model.ocean)
    initialize!(model.interfaces.exchanger, model)
    return nothing
end

reference_density(unsupported) =
    throw(ArgumentError("Cannot extract reference density from $(typeof(unsupported))"))

heat_capacity(unsupported) =
    throw(ArgumentError("Cannot deduce the heat capacity from $(typeof(unsupported))"))

"""
    EarthSystemModel(atmosphere, ocean, sea_ice;
                     radiation = Radiation(),
                     clock = Clock{Float64}(time=0),
                     ocean_reference_density = reference_density(ocean),
                     ocean_heat_capacity = heat_capacity(ocean),
                     sea_ice_reference_density = reference_density(sea_ice),
                     sea_ice_heat_capacity = heat_capacity(sea_ice),
                     interfaces = nothing)

Construct a coupled earth system model with an atmosphere, ocean, and sea ice component.
For simpler configurations, see [`OceanOnlyModel`](@ref) and [`OceanSeaIceModel`](@ref).

Arguments
==========

- `atmosphere`: A representation of a possibly time-dependent atmospheric state.
                For a prognostic atmosphere, use `atmosphere_simulation`. For prescribed
                atmospheric forcing, use `JRA55PrescribedAtmosphere` or `PrescribedAtmosphere`.
- `ocean`: A representation of a possibly time-dependent ocean state. Currently, only `Oceananigans.Simulation`s
           of `Oceananigans.HydrostaticFreeSurfaceModel` are tested.
- `sea_ice`: A representation of a possibly time-dependent sea ice state.
             For example, the minimalist `FreezingLimitedOceanTemperature` represents
             oceanic latent heating during freezing only, but does not evolve sea ice variables.
             For prognostic sea ice use an `Oceananigans.Simulation` of `ClimaSeaIce.SeaIceModel`.

Keyword Arguments
==================

- `radiation`: Radiation component used to compute surface fluxes at the bottom of the atmosphere.
- `clock`: Keeps track of time.
- `ocean_reference_density`: Reference density for the ocean. Defaults to value from ocean model.
- `ocean_heat_capacity`: Heat capacity for the ocean. Defaults to value from ocean model.
- `sea_ice_reference_density`: Reference density for sea ice. Defaults to value from sea ice model.
- `sea_ice_heat_capacity`: Heat capacity for sea ice. Defaults to value from sea ice model.
- `interfaces`: Component interfaces for coupling. Defaults to `nothing` and will be constructed automatically.
  To customize the sea ice-ocean heat flux formulation, create interfaces manually using `ComponentInterfaces`.

Stability Functions
====================

The model uses similarity theory for turbulent fluxes between components. You can customize the stability functions
by creating a new `SimilarityTheoryFluxes` object with your desired stability functions. For example:

```jldoctest earth_system_model
using NumericalEarth
using Oceananigans

grid = RectilinearGrid(size=10, z=(-100, 0), topology=(Flat, Flat, Bounded))
ocean = ocean_simulation(grid, timestepper = :QuasiAdamsBashforth2)

# Three choices for stability function:
# "No stability function", which also apply to neutral boundary layers
stability_functions = nothing

# Atmosphere-sea ice specific stability functions
stability_functions = NumericalEarth.EarthSystemModels.atmosphere_sea_ice_stability_functions(Float64)

# Edson et al. (2013) stability functions
stability_functions = NumericalEarth.EarthSystemModels.atmosphere_ocean_stability_functions(Float64)

atmosphere_ocean_fluxes = SimilarityTheoryFluxes(; stability_functions)
interfaces = NumericalEarth.EarthSystemModels.ComponentInterfaces(nothing, ocean; atmosphere_ocean_fluxes)
model = OceanOnlyModel(ocean; interfaces)

# output
EarthSystemModel{CPU}(time = 0 seconds, iteration = 0)
├── ocean: HydrostaticFreeSurfaceModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── atmosphere: Nothing
├── sea_ice: FreezingLimitedOceanTemperature{ClimaSeaIce.SeaIceThermodynamics.LinearLiquidus{Float64}}
└── interfaces: ComponentInterfaces
```

The available stability function options include:
- `atmosphere_ocean_stability_functions`: Based on [edson2013exchange](@citet)
- `atmosphere_sea_ice_stability_functions`: Specifically designed for atmosphere-sea ice interactions
- `nothing`: No stability functions will be used
- Custom stability functions can be created by defining functions of the "stability parameter"
  (the flux Richardson number), `ζ`.
"""
function EarthSystemModel(atmosphere, ocean, sea_ice;
                          radiation = Radiation(),
                          clock = Clock{Float64}(time=0),
                          ocean_reference_density = reference_density(ocean),
                          ocean_heat_capacity = heat_capacity(ocean),
                          sea_ice_reference_density = reference_density(sea_ice),
                          sea_ice_heat_capacity = heat_capacity(sea_ice),
                          interfaces = nothing)

    if ocean isa Simulation
        if !isnothing(ocean.callbacks)
            # Remove some potentially irksome callbacks from the ocean simulation
            pop!(ocean.callbacks, :stop_time_exceeded, nothing)
            pop!(ocean.callbacks, :stop_iteration_exceeded, nothing)
            pop!(ocean.callbacks, :wall_time_limit_exceeded, nothing)
            pop!(ocean.callbacks, :nan_checker, nothing)
        end
    end

    if sea_ice isa Simulation
        if !isnothing(sea_ice.callbacks)
            pop!(sea_ice.callbacks, :stop_time_exceeded, nothing)
            pop!(sea_ice.callbacks, :stop_iteration_exceeded, nothing)
            pop!(sea_ice.callbacks, :wall_time_limit_exceeded, nothing)
            pop!(sea_ice.callbacks, :nan_checker, nothing)
        end
    end

    # Contains information about flux contributions: bulk formula, prescribed fluxes, etc.
    if isnothing(interfaces) && !(isnothing(atmosphere) && isnothing(sea_ice))
        interfaces = ComponentInterfaces(atmosphere, ocean, sea_ice;
                                         ocean_reference_density,
                                         ocean_heat_capacity,
                                         sea_ice_reference_density,
                                         sea_ice_heat_capacity,
                                         radiation)
    end

    arch = architecture(interfaces.exchanger.grid)

    earth_system_model = EarthSystemModel(arch,
                                           clock,
                                           atmosphere,
                                           sea_ice,
                                           ocean,
                                           interfaces)

    # Make sure the initial temperature of the ocean
    # is not below freezing and above melting near the surface
    above_freezing_ocean_temperature!(ocean, interfaces.exchanger.grid, sea_ice)
    initialization_update_state!(earth_system_model)

    return earth_system_model
end

"""
    OceanOnlyModel(ocean; atmosphere=nothing, radiation=Radiation(), kw...)

Construct an ocean-only model without a sea ice component.
This is a convenience constructor for [`EarthSystemModel`](@ref) that sets `sea_ice`
to `FreezingLimitedOceanTemperature` (a simple freezing limiter that does not evolve sea ice variables).

The `atmosphere` keyword can be used to specify a prescribed atmospheric forcing
(e.g., `JRA55PrescribedAtmosphere`). All other keyword arguments are forwarded
to `EarthSystemModel`.

```jldoctest
using NumericalEarth
using Oceananigans

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
"""
OceanOnlyModel(ocean; atmosphere = nothing, kw...) = EarthSystemModel(atmosphere, ocean, default_sea_ice(); kw...)

"""
    OceanSeaIceModel(ocean, sea_ice; atmosphere=nothing, radiation=Radiation(), kw...)

Construct a coupled ocean--sea ice model.
This is a convenience constructor for [`EarthSystemModel`](@ref) with an explicit sea ice component
and an optional prescribed atmosphere.

The `atmosphere` keyword can be used to specify a prescribed atmospheric forcing
(e.g., `JRA55PrescribedAtmosphere`). All other keyword arguments are forwarded
to `EarthSystemModel`.

```jldoctest
using NumericalEarth
using Oceananigans

grid = RectilinearGrid(size=10, z=(-100, 0), topology=(Flat, Flat, Bounded))
ocean = ocean_simulation(grid, timestepper = :QuasiAdamsBashforth2)
sea_ice = FreezingLimitedOceanTemperature()
model = OceanSeaIceModel(ocean, sea_ice)

# output
EarthSystemModel{CPU}(time = 0 seconds, iteration = 0)
├── ocean: HydrostaticFreeSurfaceModel{CPU, RectilinearGrid}(time = 0 seconds, iteration = 0)
├── atmosphere: Nothing
├── sea_ice: FreezingLimitedOceanTemperature{ClimaSeaIce.SeaIceThermodynamics.LinearLiquidus{Float64}}
└── interfaces: ComponentInterfaces
```
"""
OceanSeaIceModel(ocean, sea_ice; atmosphere = nothing, kw...) = EarthSystemModel(atmosphere, ocean, sea_ice; kw...)

time(coupled_model::EarthSystemModel) = coupled_model.clock.time

# Check for NaNs in the first prognostic field (generalizes to prescribed velocities).
function default_nan_checker(model::EarthSystemModel)
    T_ocean = ocean_temperature(model.ocean)
    nan_checker = NaNChecker((; T_ocean))
    return nan_checker
end

@kernel function _above_freezing_ocean_temperature!(T, grid, S, ℵ, liquidus)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    @inbounds begin
        for k in 1:Nz
            Tm = melting_temperature(liquidus, S[i, j, k])
            T[i, j, k] = max(T[i, j, k], Tm)
        end
    end
end

function above_freezing_ocean_temperature!(ocean, grid, sea_ice)
    T = ocean_temperature(ocean)
    S = ocean_salinity(ocean)
    ℵ = sea_ice_concentration(sea_ice)
    liquidus = sea_ice.model.ice_thermodynamics.phase_transitions.liquidus

    arch = architecture(grid)
    launch!(arch, grid, :xy, _above_freezing_ocean_temperature!, T, grid, S, ℵ, liquidus)

    return nothing
end

# nothing sea-ice
above_freezing_ocean_temperature!(ocean, grid, ::Nothing) = nothing

#####
##### Checkpointing
#####

function prognostic_state(osm::EarthSystemModel) 
    return (clock = prognostic_state(osm.clock),
            ocean = prognostic_state(osm.ocean),
            atmosphere = prognostic_state(osm.atmosphere),
            sea_ice = prognostic_state(osm.sea_ice),
            interfaces = prognostic_state(osm.interfaces))
end

function restore_prognostic_state!(osm::EarthSystemModel, state)
    restore_prognostic_state!(osm.clock, state.clock)
    restore_prognostic_state!(osm.ocean, state.ocean)
    restore_prognostic_state!(osm.atmosphere, state.atmosphere)
    restore_prognostic_state!(osm.sea_ice, state.sea_ice)
    restore_prognostic_state!(osm.interfaces, state.interfaces)
    # Note: we do NOT call update_state! here because:
    # 1. The checkpoint was saved AFTER update_state! was called at the end of that time step
    # 2. Calling update_state! would recompute interface fluxes and overwrite restored state
    #    (e.g., top_surface_temperature is overwritten by compute_atmosphere_sea_ice_fluxes!)
    return osm
end

restore_prognostic_state!(osm::EarthSystemModel, ::Nothing) = osm
