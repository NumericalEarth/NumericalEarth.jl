#####
##### OceanBoundaryExchanger: parent ocean → the boundary bands of a nested ocean
#####

using ..Oceans: PrescribedOcean
using ..EarthSystemModels: BoundaryPrescribedComponent

using Oceananigans.BoundaryConditions: net_inflow_value
using Oceananigans.Operators: Δxᶜᶠᶜ, Δyᶠᶜᶜ

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

# The parent's sources and bands on each open side, its depth-mean normal velocity where its seafloor is known, and
# the shift of those depth means that sets the exterior net inflow (`nothing` without `net_inflow`)
struct OceanBoundaryExchanger{S, B, D, F, T, N}
    sources             :: S
    bands               :: B
    depth_means         :: D
    floors              :: F
    times               :: T
    exterior_net_inflow :: N
end

# The wet area H Δ of the child's column at each boundary face of every open side (Δ the face's width along the
# boundary), the target `net_inflow` (m³ s⁻¹, positive into the domain; a number or a `Ref`) and the grid
struct ExteriorNetInflow{A, T, G}
    face_areas :: A
    net_inflow :: T
    grid       :: G
end

normal_velocity_name(side) = boundary_dimension(side) == 1 ? :u : :v

# The parent's seafloor at the `name` velocity points of `grid`
function boundary_floor(grid, name, parent_bottom_height)
    LX, LY, _ = typeof.(boundary_location(name))
    floor = Field{LX, LY, Nothing}(underlying_grid(grid))
    interpolate_bottom_height!(floor, parent_bottom_height, parent_bottom_height.grid)
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

inflow_sign(side) = side ∈ (:west, :south) ? 1 : -1

@inline face_width(::Val{1}, face, s, grid) = Δyᶠᶜᶜ(face, s, 1, grid)
@inline face_width(::Val{2}, face, s, grid) = Δxᶜᶠᶜ(s, face, 1, grid)

@kernel function _compute_face_area!(area, grid, dim, series, face, cell)
    i, j = @index(Global, NTuple)
    s = ifelse(dim === Val(1), j, i)
    _, H = wet_column_integral(dim, series, face, cell, s, s, grid, Time(0))
    @inbounds area[i, j, 1] = H * face_width(dim, face, s, grid)
end

function face_area(grid, side, depth_mean, normal_velocity)
    dim = boundary_dimension(side)
    N = size(grid, dim)
    face = boundary_index(Face(), side, N)
    cell = boundary_index(Center(), side, N)
    area = band_field(grid, depth_mean_location(side), depth_mean.current.indices)
    Nx, Ny, _ = size(grid)
    params = dim == 1 ? KernelParameters((1, Ny), (face - 1, 0)) : KernelParameters((Nx, 1), (0, face - 1))
    launch!(architecture(grid), grid, params, _compute_face_area!, area, grid, Val(dim), normal_velocity, face, cell)
    return area
end

exterior_net_inflow(grid, sides, bands, depth_means, ::Nothing) = nothing

function exterior_net_inflow(grid, sides, bands, depth_means, net_inflow)
    areas = map(side -> face_area(grid, side, depth_means[side], bands[side][normal_velocity_name(side)]), sides)
    return ExteriorNetInflow(NamedTuple{sides}(areas), net_inflow, grid)
end

adjust_exterior_transport!(depth_means, ::Nothing) = nothing

# ū ← ū - sᵢ (Q - T) / A on every open side i, with sᵢ = ±1 its inflow sign, Q = Σᵢ sᵢ Σ ū H Δ the exterior net
# inflow, A = Σᵢ Σ H Δ the wet area of the open sides and T the target net inflow
function adjust_exterior_transport!(depth_means, correction::ExteriorNetInflow)
    sides = keys(depth_means)
    areas = correction.face_areas
    arch  = architecture(correction.grid)
    Q = all_reduce(+, sum(side -> inflow_sign(side) * sum(Base.parent(depth_means[side].current) .* Base.parent(areas[side])), sides), arch)
    A = all_reduce(+, sum(side -> sum(Base.parent(areas[side])), sides), arch)
    for side in sides
        Base.parent(depth_means[side].current) .-= inflow_sign(side) * (Q - net_inflow_value(correction.net_inflow)) / A
    end
    return nothing
end

# Sample the parent at `time` into the current state of every band, then shift the depth means to the net inflow
function sample_bands!(exchanger, time)
    for side in keys(exchanger.bands)
        sources = exchanger.sources[side]
        for (name, band) in pairs(exchanger.bands[side])
            sample_parent!(band.current, boundary_location(name), sources[name], time)
        end
        sample_depth_mean!(exchanger.depth_means[side], sources[normal_velocity_name(side)], exchanger.floors[side], side, time)
    end
    adjust_exterior_transport!(exchanger.depth_means, exchanger.exterior_net_inflow)
    return nothing
end

roll_band!(::Nothing) = nothing
roll_band!(band) = (Base.parent(band.previous) .= Base.parent(band.current); nothing)

function roll_bands!(exchanger)
    foreach(side_bands -> foreach(roll_band!, side_bands), exchanger.bands)
    foreach(roll_band!, exchanger.depth_means)
    return nothing
end

function OceanBoundaryExchanger(parent, grid::AbstractGrid, sides::Tuple; width, time, parent_bottom_height = nothing,
                                net_inflow = nothing)
    sources = map(side_sources -> child_frame_sources(grid, side_sources), parent_ocean_sources(parent, sides))
    times = ParentTimes(time, time)

    function side_bands(side)
        names = keys(sources[side])
        band(name) = BoundaryBand(grid, boundary_location(name), side, name === :η ? 1 : width, times)
        return NamedTuple{names}(map(band, names))
    end

    bands = NamedTuple{sides}(map(side_bands, sides))
    floors = boundary_floors(grid, sides, parent_bottom_height)
    depth_means = NamedTuple{sides}(map(side -> depth_mean_band(grid, side, floors[side], times), sides))

    net_inflow_shift = exterior_net_inflow(grid, sides, bands, depth_means, net_inflow)
    exchanger = OceanBoundaryExchanger(sources, bands, depth_means, floors, times, net_inflow_shift)
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
