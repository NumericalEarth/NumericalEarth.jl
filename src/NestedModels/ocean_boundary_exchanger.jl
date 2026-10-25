#####
##### OceanBoundaryExchanger: parent ocean → the boundary bands of a nested ocean
#####

using Oceananigans.TimeSteppers: time_step!

using ..Oceans: PrescribedOcean
using ..EarthSystemModels: BoundaryPrescribedComponent

"""
    parent_ocean_sources(parent)

Return the parent state a nested ocean samples, a `NamedTuple` with the velocities `u` and `v`, the free
surface `η`, and the tracers `T` and `S`: the `FieldTimeSeries` of a [`PrescribedOcean`](@ref).
"""
parent_ocean_sources(ocean::PrescribedOcean) = (u = ocean.velocities.u,
                                               v = ocean.velocities.v,
                                               η = ocean.free_surface,
                                               T = ocean.temperature,
                                               S = ocean.salinity)

# A strip parent supplies each side from its own strip; any other parent supplies every side.
parent_ocean_sources(parent, sides) = NamedTuple{sides}(ntuple(Returns(parent_ocean_sources(parent)), length(sides)))

function parent_ocean_sources(parent::BoundaryPrescribedComponent, sides)
    for side in sides
        isnothing(getproperty(parent, side)) && throw(ArgumentError("the parent has no strip on the open $side side"))
    end
    return NamedTuple{sides}(map(side -> parent_ocean_sources(getproperty(parent, side)), sides))
end

"""
    OceanBoundaryExchanger(parent, grid, sides; width, time)

Mediates the `parent` ocean state to a nested ocean on `grid`: holds, for each open side in `sides`, the
parent's [`parent_ocean_sources`](@ref) and the [`BoundaryBand`](@ref)s they are sampled onto — `width`
cells wide for the velocities and tracers (the boundary relaxation zone), one cell for the free surface —
and the [`ParentTimes`](@ref) of the two states the bands hold, both initially the parent at `time`.
"""
struct OceanBoundaryExchanger{S, B, T}
    sources :: S
    bands   :: B
    times   :: T
end

function OceanBoundaryExchanger(parent, grid::AbstractGrid, sides::Tuple; width, time)
    sources = parent_ocean_sources(parent, sides)
    times = ParentTimes(time, time)

    function side_bands(side)
        names = keys(sources[side])
        band(name) = BoundaryBand(grid, boundary_location(name), side, name === :η ? 1 : width, times)
        return NamedTuple{names}(map(band, names))
    end

    bands = NamedTuple{sides}(map(side_bands, sides))

    for side in sides, name in keys(bands[side])
        band = bands[side][name]
        sample_parent!(band.current, boundary_location(name), sources[side][name], time)
        Base.parent(band.previous) .= Base.parent(band.current)
    end

    return OceanBoundaryExchanger(sources, bands, times)
end

# Sample the parent at `time` into the bands, keeping the state they held as the earlier time level.
# Idempotent at the current time, so `update_state!` can call it freely.
function exchange_state!(exchanger::OceanBoundaryExchanger, time)
    times = exchanger.times
    time > times.current || return nothing

    for side in keys(exchanger.bands), name in keys(exchanger.bands[side])
        band = exchanger.bands[side][name]
        Base.parent(band.previous) .= Base.parent(band.current)
        sample_parent!(band.current, boundary_location(name), exchanger.sources[side][name], time)
    end

    times.previous = times.current
    times.current = time

    return nothing
end

# The parent steps first, so that the bands bracket the child's step with parent states at its start and
# end, and a streaming parent has loaded the samples that bracket them.
function nested_time_step!(nested, exchanger::OceanBoundaryExchanger, Δt, callbacks)
    time = nested.clock.time + Δt

    Δt_parent = time - nested.parent.clock.time
    Δt_parent > 0 && time_step!(nested.parent, Δt_parent)

    exchange_state!(exchanger, time)
    time_step!(nested.child, Δt; callbacks)

    return nothing
end
