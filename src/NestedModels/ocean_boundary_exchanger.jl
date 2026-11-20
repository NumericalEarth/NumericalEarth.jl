#####
##### OceanBoundaryExchanger: parent ocean → the boundary bands of a nested ocean
#####

using ..Oceans: PrescribedOcean
using ..EarthSystemModels: BoundaryPrescribedComponent

parent_ocean_sources(ocean::PrescribedOcean) = (u = ocean.velocities.u,
                                               v = ocean.velocities.v,
                                               η = ocean.free_surface,
                                               T = ocean.sea_surface_temperature,
                                               S = ocean.sea_surface_salinity)

# A strip parent supplies each side from its own strip; any other parent supplies every side.
parent_ocean_sources(parent, sides) = NamedTuple{sides}(ntuple(Returns(parent_ocean_sources(parent)), length(sides)))

function parent_ocean_sources(parent::BoundaryPrescribedComponent, sides)
    for side in sides
        isnothing(getproperty(parent, side)) && throw(ArgumentError("the parent has no strip on the open $side side"))
    end
    return NamedTuple{sides}(map(side -> parent_ocean_sources(getproperty(parent, side)), sides))
end

# The parent's sources and bands on each open side, and its depth-mean normal velocity where its seafloor is known
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

# Sample the parent at `time` into the current state of every band
function sample_bands!(exchanger, time)
    for side in keys(exchanger.bands)
        sources = exchanger.sources[side]
        for (name, band) in pairs(exchanger.bands[side])
            sample_parent!(band.current, boundary_location(name), sources[name], time)
        end
        sample_depth_mean!(exchanger.depth_means[side], sources[normal_velocity_name(side)], exchanger.floors[side], side, time)
    end
    return nothing
end

roll_band!(::Nothing) = nothing
roll_band!(band) = (Base.parent(band.previous) .= Base.parent(band.current); nothing)

function roll_bands!(exchanger)
    foreach(side_bands -> foreach(roll_band!, side_bands), exchanger.bands)
    foreach(roll_band!, exchanger.depth_means)
    return nothing
end

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

    exchanger = OceanBoundaryExchanger(sources, bands, depth_means, floors, times)
    sample_bands!(exchanger, time)
    roll_bands!(exchanger)

    return exchanger
end

# Sample the parent at `time` into the bands, keeping their state as the earlier time level; idempotent at the current time
function exchange_state!(exchanger::OceanBoundaryExchanger, time)
    times = exchanger.times
    time > times.current || return nothing

    roll_bands!(exchanger)
    sample_bands!(exchanger, time)

    times.previous = times.current
    times.current = time

    return nothing
end

# The parent steps first, so that the bands bracket the child's step with parent states at its start and end
function time_step!(nested::NestedModel{<:Any, <:Any, <:OceanBoundaryExchanger}, Δt; callbacks=())
    time = nested.clock.time + Δt

    Δt_parent = time - nested.parent.clock.time
    Δt_parent > 0 && time_step!(nested.parent, Δt_parent)

    exchange_state!(nested.exchanger, time)
    time_step!(nested.child, Δt; callbacks = (callbacks..., nested.child_callbacks...))

    return nothing
end
