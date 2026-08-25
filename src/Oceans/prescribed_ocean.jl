using Oceananigans.Architectures: architecture
using Oceananigans.OutputReaders: update_field_time_series!, FieldTimeSeries
using Oceananigans.TimeSteppers: Clock, tick!
using Oceananigans.Units: Time
using Oceananigans.Utils: prettysummary, prettytime
using NumericalEarth.EarthSystemModels: AbstractPrescribedComponent

"""
    PrescribedOcean(grid, times=[zero(grid)];
                    density = 1025,
                    heat_capacity = 4000,
                    clock = Clock{FT}(time=0),
                    temperature = default_prescribed_temperature(grid, times),
                    salinity = default_prescribed_salinity(grid, times),
                    velocities = default_prescribed_velocities(grid, times),
                    free_surface = default_prescribed_free_surface(grid, times))

An ocean whose state is prescribed as `FieldTimeSeries` rather than evolved.

Surface (z-`Nothing`) and volumetric (z-`Center`) fields are selected from the grid's vertical size
through [`is_three_dimensional`](@ref): a single-level grid builds a surface ocean, the coupling
component of an `EarthSystemModel`, whose state follows the data while surface fluxes are still
computed so the atmosphere feels it; a resolved vertical builds a volumetric ocean, the parent of a
[`nested_ocean_model`](@ref). `free_surface` is two-dimensional either way.

Arguments
=========

- `grid`: An Oceananigans grid for the ocean domain.
- `times`: Time instances for the prescribed data. Default: `[zero(grid)]` (constant).

Keyword Arguments
=================

- `density`: Seawater density in kg/m³. Default: 1025.
- `heat_capacity`: Seawater specific heat capacity in J/(kg·K). Default: 4000.
- `clock`: Clock for tracking ocean time. Default: `Clock{FT}(time=0)`.
- `temperature`: `FieldTimeSeries` for temperature.
- `salinity`: `FieldTimeSeries` for salinity.
- `velocities`: `NamedTuple` of `FieldTimeSeries` for `(u, v)`.
- `free_surface`: `FieldTimeSeries` for the free surface displacement.
"""
mutable struct PrescribedOcean{FT, G, C, T, S, U, E, TI, R, HC} <: AbstractPrescribedComponent
    grid :: G
    clock :: C
    temperature :: T
    salinity :: S
    velocities :: U
    free_surface :: E
    times :: TI
    density :: R
    heat_capacity :: HC

    function PrescribedOcean{FT}(grid::G, clock::C, T::T̃, S::S̃, u::U, η::E, times::TI, ρ::R, cp::HC) where {FT, G, C, T̃, S̃, U, E, TI, R, HC}
        return new{FT, G, C, T̃, S̃, U, E, TI, R, HC}(grid, clock, T, S, u, η, times, ρ, cp)
    end
end

prescribed_ocean_location(grid) = (Center, Center, is_three_dimensional(grid) ? Center : Nothing)

function default_prescribed_temperature(grid, times)
    LX, LY, LZ = prescribed_ocean_location(grid)
    return FieldTimeSeries{LX, LY, LZ}(grid, times)
end

function default_prescribed_salinity(grid, times)
    LX, LY, LZ = prescribed_ocean_location(grid)
    salinity = FieldTimeSeries{LX, LY, LZ}(grid, times)
    parent(salinity) .= 35
    return salinity
end

function default_prescribed_velocities(grid, times)
    LX, LY, LZ = prescribed_ocean_location(grid)
    u = FieldTimeSeries{LX, LY, LZ}(grid, times)
    v = FieldTimeSeries{LX, LY, LZ}(grid, times)
    return (; u, v)
end

default_prescribed_free_surface(grid, times) = FieldTimeSeries{Center, Center, Nothing}(grid, times)

function PrescribedOcean(grid, times=[zero(grid)];
                         FT = eltype(grid),
                         density = 1025,
                         heat_capacity = 4000,
                         clock = Clock{FT}(time=0),
                         temperature = default_prescribed_temperature(grid, times),
                         salinity = default_prescribed_salinity(grid, times),
                         velocities = default_prescribed_velocities(grid, times),
                         free_surface = default_prescribed_free_surface(grid, times))

    return PrescribedOcean{FT}(grid,
                               clock,
                               temperature,
                               salinity,
                               velocities,
                               free_surface,
                               times,
                               convert(FT, density),
                               convert(FT, heat_capacity))
end

Grids.grid(ocean::PrescribedOcean) = ocean.grid

function Oceananigans.set!(ocean::PrescribedOcean; T=nothing, S=nothing, u=nothing, v=nothing, η=nothing)
    !isnothing(T) && (parent(ocean.temperature) .= T)
    !isnothing(S) && (parent(ocean.salinity) .= S)
    !isnothing(u) && (parent(ocean.velocities.u) .= u)
    !isnothing(v) && (parent(ocean.velocities.v) .= v)
    !isnothing(η) && (parent(ocean.free_surface) .= η)
    return nothing
end

function Base.summary(ocean::PrescribedOcean{FT}) where FT
    A = nameof(typeof(architecture(ocean.grid)))
    G = nameof(typeof(ocean.grid))
    Nt = length(ocean.times)
    return string("PrescribedOcean{$FT, $A, $G}",
                  "(Nt = ", Nt, ")")
end

function Base.show(io::IO, ocean::PrescribedOcean)
    print(io, summary(ocean), "\n",
          "├── grid: ", summary(ocean.grid), "\n",
          "├── times: ", prettysummary(ocean.times), "\n",
          "├── density: ", prettysummary(ocean.density), "\n",
          "└── heat_capacity: ", prettysummary(ocean.heat_capacity))
end

Base.eltype(::PrescribedOcean{FT}) where FT = FT

#####
##### EarthSystemModels interface
#####

EarthSystemModels.is_sea_ice_component(::PrescribedOcean) = false

function EarthSystemModels.adopt_clock(ocean::PrescribedOcean{FT}, clock) where FT
    new_clock = EarthSystemModels.matching_clock(ocean.clock, clock)
    isnothing(new_clock) && return ocean
    EarthSystemModels.warn_clock_coercion(ocean, new_clock)
    return PrescribedOcean{FT}(ocean.grid,
                               new_clock,
                               ocean.temperature,
                               ocean.salinity,
                               ocean.velocities,
                               ocean.free_surface,
                               ocean.times,
                               ocean.density,
                               ocean.heat_capacity)
end

EarthSystemModels.reference_density(ocean::PrescribedOcean) = ocean.density
EarthSystemModels.heat_capacity(ocean::PrescribedOcean) = ocean.heat_capacity
EarthSystemModels.temperature_units(::PrescribedOcean) = DegreesKelvin()

EarthSystemModels.ocean_temperature(ocean::PrescribedOcean) = ocean.temperature
EarthSystemModels.ocean_salinity(ocean::PrescribedOcean) = ocean.salinity

# The uppermost level of a prescribed series: the field itself when surface-only, its top level when
# volumetric. The level index stays a range so both forms carry the same `(Nx, Ny, 1, Nt)` shape.
surface_level(series) = (kᴺ = size(series.grid, 3); view(series, :, :, kᴺ:kᴺ, :))
surface_level(series::FieldTimeSeries{<:Any, <:Any, Nothing}) = series

EarthSystemModels.ocean_surface_temperature(ocean::PrescribedOcean) = surface_level(ocean.temperature)
EarthSystemModels.ocean_surface_salinity(ocean::PrescribedOcean) = surface_level(ocean.salinity)
EarthSystemModels.ocean_surface_velocities(ocean::PrescribedOcean) =
    surface_level(ocean.velocities.u), surface_level(ocean.velocities.v)

#####
##### InterfaceComputations interface
#####

function EarthSystemModels.InterfaceComputations.ComponentExchanger(ocean::PrescribedOcean, exchange_grid)
    grid = ocean.grid
    T = Field{Center, Center, Nothing}(grid)
    S = Field{Center, Center, Nothing}(grid)
    u = Field{Center, Center, Nothing}(grid)
    v = Field{Center, Center, Nothing}(grid)

    # Initialize from the first time snapshot
    interior(T) .= interior(ocean.temperature)[:, :, end:end, 1]
    interior(S) .= interior(ocean.salinity)[:, :, end:end, 1]
    interior(u) .= interior(ocean.velocities.u)[:, :, end:end, 1]
    interior(v) .= interior(ocean.velocities.v)[:, :, end:end, 1]

    return ComponentExchanger((; u, v, T, S), nothing)
end

EarthSystemModels.InterfaceComputations.net_fluxes(ocean::PrescribedOcean) = nothing

function EarthSystemModels.interpolate_state!(exchanger, grid, ocean::PrescribedOcean, coupled_model)
    # Copy from FieldTimeSeries to exchanger snapshot fields.
    # For single-time data (constant), time index 1 is always correct.
    # TODO: proper temporal interpolation for multi-time prescribed data.
    n = 1
    interior(exchanger.state.T) .= interior(ocean.temperature)[:, :, end:end, n]
    interior(exchanger.state.S) .= interior(ocean.salinity)[:, :, end:end, n]
    interior(exchanger.state.u) .= interior(ocean.velocities.u)[:, :, end:end, n]
    interior(exchanger.state.v) .= interior(ocean.velocities.v)[:, :, end:end, n]
    return nothing
end

# Prescribed ocean does not evolve, so net flux assembly is not needed.
# The atmosphere still receives its fluxes from compute_atmosphere_ocean_fluxes!.
EarthSystemModels.update_net_fluxes!(coupled_model, ocean::PrescribedOcean) = nothing

#####
##### Time stepping — update FieldTimeSeries backends, tick the clock
#####

function Oceananigans.TimeSteppers.time_step!(ocean::PrescribedOcean, Δt)
    tick!(ocean.clock, Δt)
    time = Time(ocean.clock.time)

    update_prescribed_ocean_series!(ocean, time)

    return nothing
end

# Reposition every prescribed series to `time`. Shared with the nested-ocean exchanger, which advances
# the parent's window ahead of the child's step rather than after it.
function update_prescribed_ocean_series!(ocean::PrescribedOcean, time)
    update_field_time_series!(ocean.temperature, time)
    update_field_time_series!(ocean.salinity, time)
    update_field_time_series!(ocean.velocities.u, time)
    update_field_time_series!(ocean.velocities.v, time)
    update_field_time_series!(ocean.free_surface, time)
    return nothing
end
