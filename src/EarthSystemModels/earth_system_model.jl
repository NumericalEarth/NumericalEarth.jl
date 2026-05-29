using ClimaSeaIce.SeaIceThermodynamics: melting_temperature
using Oceananigans
using Oceananigans.TimeSteppers: Clock
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

Base.eltype(model::ESM) = Base.eltype(model.interfaces.exchanger.grid)
Oceananigans.prognostic_fields(::ESM) = nothing
Oceananigans.fields(::ESM) = NamedTuple()
Oceananigans.Architectures.architecture(model::ESM) = model.architecture
Oceananigans.Simulations.iteration(model::ESM) = model.clock.iteration
Oceananigans.Simulations.timestepper(::ESM) = nothing
Oceananigans.OutputWriters.default_included_properties(::ESM) = tuple()
Oceananigans.Utils.prettytime(model::ESM) = prettytime(model.clock.time)
default_clock(TT) = Clock{TT}(0, 0, 1)

Oceananigans.Simulations.reset_clock!(::Nothing) = nothing
Oceananigans.TimeSteppers.update_state!(::Nothing) = nothing
Oceananigans.Simulations.reset_clock!(component::Simulation) = reset_clock!(component.model)
Oceananigans.Simulations.reset_clock!(component) = reset!(getproperty(component, :clock))

function Oceananigans.Simulations.reset_clock!(model::ESM)
    reset!(model.clock)

    for component in components(model)
        reset_clock!(component)
    end

    # Keep prescribed atmospheric forcing synchronized during component resets.
    update_state!(model.atmosphere)

    return nothing
end

Oceananigans.TimeSteppers.reset!(model::ESM) = reset_clock!(model)

# Make sure to initialize the exchanger here
function Oceananigans.initialize!(model::ESM)
    initialize!(model.interfaces.exchanger, model)
    return nothing
end

function Oceananigans.TimeSteppers.reconcile_state!(model::ESM)
    initialize!(model.interfaces.exchanger, model)
    update_state!(model)
    return nothing
end

function remove_default_stop_callbacks!(component)
    callbacks = component.callbacks

    if !isnothing(callbacks)
        pop!(callbacks, :stop_time_exceeded, nothing)
        pop!(callbacks, :stop_iteration_exceeded, nothing)
        pop!(callbacks, :wall_time_limit_exceeded, nothing)
        pop!(callbacks, :nan_checker, nothing)
    end

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
    materialize_earth_system_radiation!(atmosphere, radiation)

Return `atmosphere` with any radiation skeletons populated against the
coupled-model `radiation`. Atmospheric components that carry an internal
radiation handle (e.g. Breeze's `AtmosphereModel.radiation`) overload this to
alias their internal flux divergence onto `radiation.flux_divergence`, so the
atmosphere's tendency machinery reads from the same field the coupled
`radiation` writes. Default: no-op, returning `atmosphere` unchanged.

Called from inside the [`EarthSystemModel`](@ref) constructor before
`ComponentInterfaces` is built.
"""
materialize_earth_system_radiation!(atmosphere, radiation) = atmosphere

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
                          interfaces = nothing,
                          interface_kw...) # e.g. `atmosphere_land_interface_specific_humidity`
    if isnothing(radiation) && atmosphere isa AbstractPrescribedComponent
        @warn """
            `EarthSystemModel` was constructed with a `PrescribedAtmosphere` but \
            `radiation = nothing`. This means no upwelling longwave (ϵσT⁴), no \
            absorbed downwelling longwave, and no shortwave absorption — \
            results will be physically inconsistent.

            If you previously relied on `Radiation()` defaults: pass \
            `radiation = JRA55PrescribedRadiation(arch; kwargs...)` (or \
            `ECCOPrescribedRadiation` / `OSPapaPrescribedRadiation`) to restore \
            radiative forcing. Pass `radiation = PrescribedRadiation(grid)` for \
            emission-only mode. To suppress this warning, build the model \
            without a `PrescribedAtmosphere` (radiatively decoupled is the \
            intended `nothing` semantics).
        """ maxlog=1
    end

    ocean   isa Simulation && remove_default_stop_callbacks!(ocean)
    sea_ice isa Simulation && remove_default_stop_callbacks!(sea_ice)

    if atmosphere isa Simulation
        if !isnothing(atmosphere.callbacks)
            pop!(atmosphere.callbacks, :stop_time_exceeded, nothing)
            pop!(atmosphere.callbacks, :stop_iteration_exceeded, nothing)
            pop!(atmosphere.callbacks, :wall_time_limit_exceeded, nothing)
            pop!(atmosphere.callbacks, :nan_checker, nothing)
        end
    end

    # Materialize any radiation skeletons in the atmosphere against the
    # coupled-model radiation. No-op by default; Breeze (or any other
    # atmosphere with an internal radiation handle) overloads this.
    atmosphere = materialize_earth_system_radiation!(atmosphere, radiation)

    # Contains information about flux contributions: bulk formula, prescribed fluxes, etc.
    if isnothing(interfaces) && !(isnothing(atmosphere) && isnothing(sea_ice))
        interfaces = ComponentInterfaces(atmosphere, ocean, sea_ice;
                                         radiation,
                                         land,
                                         ocean_reference_density,
                                         ocean_heat_capacity,
                                         sea_ice_reference_density,
                                         sea_ice_heat_capacity,
                                         interface_kw...)
    end

    arch = architecture(interfaces.exchanger.grid)

    # Allocate per-surface InterfaceRadiationFlux on the exchange grid.
    surfaces = present_surfaces(ocean, sea_ice, land)
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

"""
    components(model::ESM)

Return a named tuple of the non-`nothing` components of an Earth System `model`.
"""
function components(model::ESM)
    all_components = (atmosphere = model.atmosphere,
                      radiation  = model.radiation,
                      ocean      = model.ocean,
                      land       = model.land,
                      sea_ice    = model.sea_ice,)

    return (; filter(p -> !isnothing(last(p)), pairs(all_components))...)
end

# Determine which surfaces are present in the model — used to allocate
# per-surface diagnostic radiation flux buffers.
function present_surfaces(ocean, sea_ice, land)
    surfaces = Symbol[]
    isnothing(ocean)   || push!(surfaces, :ocean)
    isnothing(sea_ice) || push!(surfaces, :sea_ice)
    isnothing(land)    || push!(surfaces, :land)
    return Tuple(surfaces)
end

# Blacklist predicates for catching swapped positional arguments to the convenience
# constructors below. The fallback returns `true` so unfamiliar component types are
# permitted; specializations in `Oceans` and `SeaIces` set the obviously-wrong cases
# (e.g. a sea ice simulation supplied where an ocean is expected) to `false`.
is_sea_ice_component(component) = true
is_ocean_component(component) = true

function invalid_component(constructor, position, expected, received)
    return ArgumentError("$constructor expects $expected as positional argument $position, " *
                         "got a $(typeof(received)) instead")
end

"""
    OceanOnlyModel(ocean; atmosphere=nothing, radiation=nothing, land=nothing, kw...)

Construct an ocean-only model without a sea ice component.
This is a convenience constructor for [`EarthSystemModel`](@ref) that sets `sea_ice`
to `FreezingLimitedOceanTemperature` (a simple freezing limiter that does not evolve sea ice variables).

The `atmosphere`, `radiation`, and `land` keywords can be used to specify prescribed
components (e.g., `JRA55PrescribedAtmosphere`). All other keyword arguments are forwarded
to `EarthSystemModel`.

```jldoctest
using NumericalEarth, Oceananigans

grid = LatitudeLongitudeGrid(size = (20, 20, 4),
                             z = (-100, 0),
                             latitude = (-80, 80),
                             longitude = (0, 360),
                             halo = (6, 6, 3))

ocean = ocean_simulation(grid, closure=nothing)
set!(ocean.model, T=20, S=35, u=0.01, v=-0.005)

ocean = OceanOnlyModel(ocean)
# output

EarthSystemModel{CPU}(time = 0 seconds, iteration = 0)
├── radiation: Nothing
├── atmosphere: Nothing
├── land: Nothing
├── sea_ice: FreezingLimitedOceanTemperature{ClimaSeaIce.SeaIceThermodynamics.LinearLiquidus{Float64}}
├── ocean: HydrostaticFreeSurfaceModel{CPU, LatitudeLongitudeGrid}(time = 0 seconds, iteration = 0)
└── interfaces: ComponentInterfaces
```
"""
function OceanOnlyModel(ocean; atmosphere=nothing, land=nothing, radiation=nothing, kw...)
    is_ocean_component(ocean) || throw(invalid_component(:OceanOnlyModel, 1, "an ocean simulation", ocean))
    return EarthSystemModel(radiation, atmosphere, land, default_sea_ice(), ocean; kw...)
end

"""
    OceanSeaIceModel(ocean, sea_ice; atmosphere=nothing, radiation=nothing, land=nothing, kw...)

Construct a coupled ocean--sea ice model.

This is a convenience constructor for [`EarthSystemModel`](@ref) with explicit ocean and sea ice components
and optional prescribed atmosphere, prescribed radiation, and prescribed land.

Positional arguments are `ocean` then `sea_ice`.

Example
=======

```jldoctest
using NumericalEarth, Oceananigans

grid = LatitudeLongitudeGrid(size = (20, 20, 4),
                             z = (-100, 0),
                             latitude = (-80, 80),
                             longitude = (0, 360),
                             halo = (6, 6, 3))

ocean = ocean_simulation(grid, closure=nothing)
set!(ocean.model, T=20, S=35, u=0.01, v=-0.005)

sea_ice = sea_ice_simulation(grid, ocean)

hi(λ, φ) = φ > 70 || φ < -70
set!(sea_ice.model, h=hi, ℵ=hi)

coupled_model = OceanSeaIceModel(ocean, sea_ice)

# output

EarthSystemModel{CPU}(time = 0 seconds, iteration = 0)
├── radiation: Nothing
├── atmosphere: Nothing
├── land: Nothing
├── sea_ice: SeaIceModel
├── ocean: HydrostaticFreeSurfaceModel{CPU, LatitudeLongitudeGrid}(time = 0 seconds, iteration = 0)
└── interfaces: ComponentInterfaces
```
"""
function OceanSeaIceModel(ocean, sea_ice; atmosphere=nothing, land=nothing, radiation=nothing, kw...)
    is_ocean_component(ocean) || throw(invalid_component(:OceanSeaIceModel, 1, "an ocean simulation", ocean))
    is_sea_ice_component(sea_ice) || throw(invalid_component(:OceanSeaIceModel, 2, "a sea ice simulation", sea_ice))
    return EarthSystemModel(radiation, atmosphere, land, sea_ice, ocean; kw...)
end

"""
    AtmosphereOceanModel(atmosphere, ocean; kw...)

Construct a coupled atmosphere--ocean model.
Convenience constructor for [`EarthSystemModel`](@ref) with an atmosphere and ocean
but no sea ice. All keyword arguments are forwarded to `EarthSystemModel`.
"""
function AtmosphereOceanModel(atmosphere, ocean; land=nothing, radiation=nothing, kw...)
    is_ocean_component(ocean) || throw(invalid_component(:AtmosphereOceanModel, 2, "an ocean simulation", ocean))
    return EarthSystemModel(radiation, atmosphere, land, nothing, ocean; kw...)
end

"""
    AtmosphereLandModel(atmosphere, land; radiation=nothing, kw...)

Construct a coupled atmosphere--land model.
Convenience constructor for [`EarthSystemModel`](@ref) with an atmosphere and
land but no ocean or sea ice. All keyword arguments are forwarded to
`EarthSystemModel`.

The atmosphere--land turbulent fluxes are computed via
`SimilarityTheoryFluxes` using land-side roughness and a β-reduced surface
specific humidity (`qᵃᵗ + β·[qₛ - qᵃᵗ]`).
"""
AtmosphereLandModel(atmosphere, land; radiation=nothing, kw...) =
    EarthSystemModel(radiation, atmosphere, land, nothing, nothing; kw...)

time(coupled_model::EarthSystemModel) = coupled_model.clock.time

# Check for NaNs in the first prognostic field (generalizes to prescribed velocities).
function Oceananigans.Diagnostics.default_nan_checker(model::EarthSystemModel)
    if isnothing(model.ocean)
        # Fall back to the surface skin temperature held at the atmosphere-land
        # interface when there is no ocean. If neither ocean nor the
        # atmosphere-land interface is present, return no NaN checker.
        T_land = surface_temperature(model.interfaces)
        isnothing(T_land) && return nothing
        return NaNChecker((; T_land))
    end

    T_ocean = ocean_temperature(model.ocean)
    return NaNChecker((; T_ocean))
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
    liquidus = sea_ice.model.phase_transitions.liquidus

    arch = architecture(grid)
    launch!(arch, grid, :xy, _above_freezing_ocean_temperature!, T, grid, S, ℵ, liquidus)

    return nothing
end

# nothing sea-ice
above_freezing_ocean_temperature!(ocean, grid, ::Nothing) = nothing

#####
##### Checkpointing
#####

function Oceananigans.prognostic_state(osm::EarthSystemModel)
    return (clock = prognostic_state(osm.clock),
            radiation = prognostic_state(osm.radiation),
            ocean = prognostic_state(osm.ocean),
            atmosphere = prognostic_state(osm.atmosphere),
            land = prognostic_state(osm.land),
            sea_ice = prognostic_state(osm.sea_ice),
            interfaces = prognostic_state(osm.interfaces))
end

function Oceananigans.restore_prognostic_state!(osm::EarthSystemModel, state)
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

Oceananigans.restore_prognostic_state!(osm::EarthSystemModel, ::Nothing) = osm
