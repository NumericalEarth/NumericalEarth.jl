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
    OceanBoundaryExchanger(parent, grid, sides; width, time, parent_bottom_height = nothing)

Mediates the `parent` ocean state to a nested ocean on `grid`: holds, for each open side in `sides`, the
parent's [`parent_ocean_sources`](@ref) and the [`BoundaryBand`](@ref)s they are sampled onto — `width`
cells wide for the velocities and tracers (the boundary relaxation zone), one cell for the free surface —
and the [`ParentTimes`](@ref) of the two states the bands hold, both initially the parent at `time`.

When the parent's seafloor `parent_bottom_height` is known, each side also holds the parent's depth mean of its
boundary-normal velocity, over the parent's own levels above that seafloor ([`sample_parent_depth_mean!`](@ref)),
on a one-cell band at the boundary face: the barotropic velocity of the Flather exterior.
"""
struct OceanBoundaryExchanger{S, B, D, F, T}
    sources     :: S
    bands       :: B
    depth_means :: D
    floors      :: F
    times       :: T
end

normal_velocity_name(side) = boundary_dimension(side) == 1 ? :u : :v

# The parent's seafloor at the `name` velocity points of `grid`
function boundary_floor(grid, name, parent_bottom_height)
    LX, LY, _ = typeof.(boundary_location(name))
    floor = Field{LX, LY, Nothing}(underlying_grid(grid))
    interpolate!(floor, parent_bottom_height)
    fill_halo_regions!(floor)
    return floor
end

# Every rank builds both floors, whatever its open sides, since the halo fill is collective.
boundary_floors(grid, sides, ::Nothing) = NamedTuple{sides}(map(Returns(nothing), sides))

function boundary_floors(grid, sides, parent_bottom_height)
    floors = (u = boundary_floor(grid, :u, parent_bottom_height),
              v = boundary_floor(grid, :v, parent_bottom_height))
    return NamedTuple{sides}(map(side -> floors[normal_velocity_name(side)], sides))
end

depth_mean_location(side) = (boundary_location(normal_velocity_name(side))[1:2]..., nothing)

depth_mean_band(grid, side, ::Nothing, times) = nothing
depth_mean_band(grid, side, floor, times) = BoundaryBand(grid, depth_mean_location(side), side, 1, times)

sample_depth_mean!(::Nothing, source, floor, side, time) = nothing
sample_depth_mean!(band, source, floor, side, time) =
    sample_parent_depth_mean!(band.current, depth_mean_location(side), source, floor, time)

function OceanBoundaryExchanger(parent, grid::AbstractGrid, sides::Tuple; width, time, parent_bottom_height = nothing)
    sources = parent_ocean_sources(parent, sides)
    times = ParentTimes(time, time)

    function side_bands(side)
        names = keys(sources[side])
        band(name) = BoundaryBand(grid, boundary_location(name), side, name === :η ? 1 : width, times)
        return NamedTuple{names}(map(band, names))
    end

    bands = NamedTuple{sides}(map(side_bands, sides))
    floors = boundary_floors(grid, sides, parent_bottom_height)
    depth_means = NamedTuple{sides}(map(side -> depth_mean_band(grid, side, floors[side], times), sides))

    for side in sides
        for name in keys(bands[side])
            band = bands[side][name]
            sample_parent!(band.current, boundary_location(name), sources[side][name], time)
            Base.parent(band.previous) .= Base.parent(band.current)
        end

        depth_mean = depth_means[side]
        sample_depth_mean!(depth_mean, sources[side][normal_velocity_name(side)], floors[side], side, time)
        isnothing(depth_mean) || (Base.parent(depth_mean.previous) .= Base.parent(depth_mean.current))
    end

    return OceanBoundaryExchanger(sources, bands, depth_means, floors, times)
end

roll_band!(::Nothing) = nothing
roll_band!(band) = (Base.parent(band.previous) .= Base.parent(band.current); nothing)

# Sample the parent at `time` into the bands, keeping the state they held as the earlier time level.
# Idempotent at the current time, so `update_state!` can call it freely.
function exchange_state!(exchanger::OceanBoundaryExchanger, time)
    times = exchanger.times
    time > times.current || return nothing

    for side in keys(exchanger.bands)
        for name in keys(exchanger.bands[side])
            band = exchanger.bands[side][name]
            roll_band!(band)
            sample_parent!(band.current, boundary_location(name), exchanger.sources[side][name], time)
        end

        depth_mean = exchanger.depth_means[side]
        roll_band!(depth_mean)
        sample_depth_mean!(depth_mean, exchanger.sources[side][normal_velocity_name(side)], exchanger.floors[side], side, time)
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
