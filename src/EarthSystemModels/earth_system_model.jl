using Oceananigans
using Oceananigans.TimeSteppers: Clock
using Oceananigans: SeawaterBuoyancy
using ClimaSeaIce.SeaIceThermodynamics: melting_temperature
using KernelAbstractions: @kernel, @index

mutable struct EarthSystemModel{R, A, L, I, O, F, C, Arch} <: AbstractModel{Nothing, Arch}
    architecture :: Arch
    clock :: C
    radiation :: R
    atmosphere :: A
    land :: L
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

    print(io, "├── radiation: ", summary(cm.radiation), "\n")
    print(io, "├── atmosphere: ", summary(cm.atmosphere), "\n")
    print(io, "├── land: ", summary(cm.land), "\n")
    print(io, "├── sea_ice: ", sea_ice_summary, "\n")
    print(io, "├── ocean: ", ocean_summary, "\n")
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
function initialize!(model::ESM)
    initialize!(model.interfaces.exchanger, model)
    return nothing
end

function reconcile_state!(model::ESM)
    initialize!(model.interfaces.exchanger, model)
    update_state!(model)
    return nothing
end

reference_density(unsupported) =
    throw(ArgumentError("Cannot extract reference density from $(typeof(unsupported))"))

heat_capacity(unsupported) =
    throw(ArgumentError("Cannot deduce the heat capacity from $(typeof(unsupported))"))

# Hook called after `interfaces` is constructed and the exchange grid is known.
# Concrete radiation types (e.g. `PrescribedRadiation`) overload this to
# allocate `interface_fluxes` per-surface on the exchange grid.
allocate_interface_fluxes!(::Any, exchange_grid, surfaces) = nothing
allocate_interface_fluxes!(::Nothing, exchange_grid, surfaces) = nothing

"""
    EarthSystemModel(radiation, atmosphere, land, sea_ice, ocean;
                     clock = Clock{Float64}(time=0),
                     ocean_reference_density = reference_density(ocean),
                     ocean_heat_capacity = heat_capacity(ocean),
                     sea_ice_reference_density = reference_density(sea_ice),
                     sea_ice_heat_capacity = heat_capacity(sea_ice),
                     interfaces = nothing)

Construct a coupled earth system model. Components are passed in struct order
(top to bottom): radiation, atmosphere, land, sea_ice, ocean. Pass `nothing`
for components that are absent. For simpler configurations, see
[`OceanOnlyModel`](@ref), [`OceanSeaIceModel`](@ref), and
[`AtmosphereOceanModel`](@ref).

Arguments
==========

- `radiation`: Radiation component, or `nothing` for a radiatively decoupled surface.
               Pass a `PrescribedRadiation` (e.g. `JRA55PrescribedRadiation(...)`) to
               enable radiative forcing.
- `atmosphere`: A representation of a possibly time-dependent atmospheric state, or `nothing`.
                For a prognostic atmosphere, use `atmosphere_simulation`. For prescribed
                atmospheric forcing, use `JRA55PrescribedAtmosphere` or `PrescribedAtmosphere`.
- `land`: Land component, or `nothing`.
- `sea_ice`: A representation of a possibly time-dependent sea ice state, or `nothing`.
             For example, the minimalist `FreezingLimitedOceanTemperature` represents
             oceanic latent heating during freezing only, but does not evolve sea ice variables.
             For prognostic sea ice use an `Oceananigans.Simulation` of `ClimaSeaIce.SeaIceModel`.
- `ocean`: A representation of a possibly time-dependent ocean state. Currently, only `Oceananigans.Simulation`s
           of `Oceananigans.HydrostaticFreeSurfaceModel` are tested.

Keyword Arguments
==================

- `clock`: Keeps track of time.
- `ocean_reference_density`: Reference density for the ocean. Defaults to value from ocean model.
- `ocean_heat_capacity`: Heat capacity for the ocean. Defaults to value from ocean model.
- `sea_ice_reference_density`: Reference density for sea ice. Defaults to value from sea ice model.
- `sea_ice_heat_capacity`: Heat capacity for sea ice. Defaults to value from sea ice model.
- `interfaces`: Component interfaces for coupling. Defaults to `nothing` and will be constructed automatically.
  To customize the sea ice-ocean heat flux formulation, create interfaces manually using `ComponentInterfaces`.
"""
function EarthSystemModel(radiation, atmosphere, land, sea_ice, ocean;
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
                                         radiation,
                                         land,
                                         ocean_reference_density,
                                         ocean_heat_capacity,
                                         sea_ice_reference_density,
                                         sea_ice_heat_capacity)
    end

    arch = architecture(interfaces.exchanger.grid)

    # Allocate per-surface InterfaceRadiationFlux on the exchange grid.
    surfaces = present_surfaces(ocean, sea_ice)
    allocate_interface_fluxes!(radiation, interfaces.exchanger.grid, surfaces)

    earth_system_model = EarthSystemModel(arch,
                                          clock,
                                          radiation,
                                          atmosphere,
                                          land,
                                          sea_ice,
                                          ocean,
                                          interfaces)

    # Make sure the initial temperature of the ocean
    # is not below freezing and above melting near the surface
    above_freezing_ocean_temperature!(ocean, interfaces.exchanger.grid, sea_ice)
    reconcile_state!(earth_system_model)

    return earth_system_model
end

"""
    EarthSystemModel(; radiation = nothing,
                       atmosphere = nothing,
                       land = nothing,
                       sea_ice = nothing,
                       ocean = nothing,
                       kw...)

Keyword-only constructor for `EarthSystemModel`. Equivalent to the positional
form, but lets you pass only the components you actually have:

```julia
EarthSystemModel(; atmosphere, ocean)                       # ocean + atmosphere
EarthSystemModel(; atmosphere, sea_ice, ocean, radiation)   # full coupled
```

All keyword arguments accepted by the positional constructor are forwarded.
"""
EarthSystemModel(; radiation = nothing,
                   atmosphere = nothing,
                   land = nothing,
                   sea_ice = nothing,
                   ocean = nothing,
                   kw...) =
    EarthSystemModel(radiation, atmosphere, land, sea_ice, ocean; kw...)

# Determine which surfaces are present in the model — used to allocate
# per-surface diagnostic radiation flux buffers.
function present_surfaces(ocean, sea_ice)
    surfaces = Symbol[]
    isnothing(ocean)   || push!(surfaces, :ocean)
    isnothing(sea_ice) || push!(surfaces, :sea_ice)
    return Tuple(surfaces)
end

"""
    OceanOnlyModel(ocean; atmosphere=nothing, radiation=nothing, land=nothing, kw...)

Construct an ocean-only model without a sea ice component.
This is a convenience constructor for [`EarthSystemModel`](@ref) that sets `sea_ice`
to `FreezingLimitedOceanTemperature` (a simple freezing limiter that does not evolve sea ice variables).

The `atmosphere` keyword can be used to specify a prescribed atmospheric forcing
(e.g., `JRA55PrescribedAtmosphere`). All other keyword arguments are forwarded
to `EarthSystemModel`.
"""
OceanOnlyModel(ocean; atmosphere=nothing, land=nothing, radiation=nothing, kw...) =
    EarthSystemModel(radiation, atmosphere, land, default_sea_ice(), ocean; kw...)

"""
    OceanSeaIceModel(sea_ice, ocean; atmosphere=nothing, radiation=nothing, land=nothing, kw...)

Construct a coupled ocean--sea ice model.
This is a convenience constructor for [`EarthSystemModel`](@ref) with an explicit sea ice component
and an optional prescribed atmosphere. Positional arguments follow the
struct convention (top→bottom): `sea_ice` then `ocean`.
"""
OceanSeaIceModel(sea_ice, ocean; atmosphere=nothing, land=nothing, radiation=nothing, kw...) =
    EarthSystemModel(radiation, atmosphere, land, sea_ice, ocean; kw...)

"""
    AtmosphereOceanModel(atmosphere, ocean; kw...)

Construct a coupled atmosphere--ocean model.
Convenience constructor for [`EarthSystemModel`](@ref) with an atmosphere and ocean
but no sea ice. All keyword arguments are forwarded to `EarthSystemModel`.
"""
AtmosphereOceanModel(atmosphere, ocean; land=nothing, radiation=nothing, kw...) =
    EarthSystemModel(radiation, atmosphere, land, nothing, ocean; kw...)

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
            radiation = prognostic_state(osm.radiation),
            ocean = prognostic_state(osm.ocean),
            atmosphere = prognostic_state(osm.atmosphere),
            land = prognostic_state(osm.land),
            sea_ice = prognostic_state(osm.sea_ice),
            interfaces = prognostic_state(osm.interfaces))
end

function restore_prognostic_state!(osm::EarthSystemModel, state)
    restore_prognostic_state!(osm.clock, state.clock)
    # Backwards-compatible: older checkpoints may not have a `radiation` entry
    if hasproperty(state, :radiation)
        restore_prognostic_state!(osm.radiation, state.radiation)
    end
    restore_prognostic_state!(osm.ocean, state.ocean)
    restore_prognostic_state!(osm.atmosphere, state.atmosphere)
    restore_prognostic_state!(osm.land, state.land)
    restore_prognostic_state!(osm.sea_ice, state.sea_ice)
    restore_prognostic_state!(osm.interfaces, state.interfaces)
    return osm
end

restore_prognostic_state!(osm::EarthSystemModel, ::Nothing) = osm
