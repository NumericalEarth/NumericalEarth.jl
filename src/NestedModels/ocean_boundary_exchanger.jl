#####
##### OceanBoundaryExchanger: parent ocean → the boundary bands of a nested ocean
#####

using Oceananigans.Operators: Δxᶜᶠᶜ, Δyᶠᶜᶜ
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
on a one-cell band at the boundary face: the barotropic velocity of the Flather exterior. With
`zero_net_inflow = true`, the same depth-uniform velocity is removed from the depth means of every open side, so
that the exterior transport has no net inflow through the open sides together; with `edge_transports`, a
`NamedTuple` of each open side's inflow (m³ s⁻¹, positive into the domain), each side's depth mean is shifted so that
its exterior transport is that inflow ([`ExteriorTransport`](@ref)).
"""
struct OceanBoundaryExchanger{S, B, D, F, T, N}
    sources     :: S
    bands       :: B
    depth_means :: D
    floors      :: F
    times       :: T
    net_inflow  :: N
end

"""
    ExteriorTransport(face_areas, edge_transports, grid)

Shifts the Flather exterior transport `ū H` of the open sides by depth-uniform velocities. With `edge_transports =
nothing`, one velocity common to every open side, so that together they carry no net inflow; with a `NamedTuple` of
each side's inflow (m³ s⁻¹, positive into the domain), one velocity per side, so that each carries its own.
`face_areas` holds, for each open side, the area `H Δ` of the child's wet column at each boundary face (`Δ` the
face's width along the boundary), on the band of the side's depth mean.
"""
struct ExteriorTransport{A, T, G}
    face_areas      :: A
    edge_transports :: T
    grid            :: G
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

adjust_exterior_transport!(depth_means, ::Nothing) = nothing

# Inflow Qᵢ = sᵢ Σ ū H Δ and wet area Aᵢ = Σ H Δ of open side i, with sᵢ = ±1 its inflow sign
local_exterior_inflow(depth_means, areas, side) = inflow_sign(side) * sum(Base.parent(depth_means[side].current) .* Base.parent(areas[side]))
local_wet_area(areas, side) = sum(Base.parent(areas[side]))

# ū ← ū - sᵢ Q / A on every open side, with Q = Σᵢ Qᵢ and A = Σᵢ Aᵢ
function adjust_exterior_transport!(depth_means, correction::ExteriorTransport{<:Any, Nothing})
    sides = keys(depth_means)
    areas = correction.face_areas
    grid  = correction.grid
    Q = all_reduce(+, sum(side -> local_exterior_inflow(depth_means, areas, side), sides; init = zero(grid)), architecture(grid))
    A = all_reduce(+, sum(side -> local_wet_area(areas, side), sides; init = zero(grid)), architecture(grid))
    for side in sides
        Base.parent(depth_means[side].current) .-= inflow_sign(side) * Q / A
    end
    return nothing
end

# ū ← ū + sᵢ (Tᵢ - Qᵢ) / Aᵢ on every open side, with Tᵢ the side's target inflow
function adjust_exterior_transport!(depth_means, correction::ExteriorTransport)
    areas = correction.face_areas
    arch  = architecture(correction.grid)
    for side in keys(depth_means)
        Qᵢ = all_reduce(+, local_exterior_inflow(depth_means, areas, side), arch)
        Aᵢ = all_reduce(+, local_wet_area(areas, side), arch)
        Base.parent(depth_means[side].current) .+= inflow_sign(side) * (correction.edge_transports[side] - Qᵢ) / Aᵢ
    end
    return nothing
end

function OceanBoundaryExchanger(parent, grid::AbstractGrid, sides::Tuple; width, time, parent_bottom_height = nothing,
                                zero_net_inflow = false, edge_transports = nothing)
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

    for side in sides
        for name in keys(bands[side])
            band = bands[side][name]
            sample_parent!(band.current, boundary_location(name), sources[side][name], time)
            Base.parent(band.previous) .= Base.parent(band.current)
        end

        sample_depth_mean!(depth_means[side], sources[side][normal_velocity_name(side)], floors[side], side, time)
    end

    face_areas() = NamedTuple{sides}(map(side -> face_area(grid, side, depth_means[side], bands[side][normal_velocity_name(side)]), sides))
    net_inflow = !isnothing(edge_transports) ? ExteriorTransport(face_areas(), edge_transports, grid) :
                 zero_net_inflow ? ExteriorTransport(face_areas(), nothing, grid) : nothing
    adjust_exterior_transport!(depth_means, net_inflow)
    foreach(roll_band!, depth_means)

    return OceanBoundaryExchanger(sources, bands, depth_means, floors, times, net_inflow)
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

    adjust_exterior_transport!(exchanger.depth_means, exchanger.net_inflow)

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
