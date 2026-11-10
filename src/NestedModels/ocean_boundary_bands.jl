#####
##### Boundary bands: the parent ocean sampled onto the child grid along each open side
#####
#
# A band is the outermost `width` rows (or columns) of the child grid along one open side, holding the
# parent state at two times. The open boundary conditions read the band's boundary row, the boundary
# relaxation reads the whole band, and both interpolate linearly in time between the two. The bands are
# refreshed once per child time step by sampling the parent's `FieldTimeSeries` in space and time, so the
# condition and forcing kernels never interpolate the parent in space, and memory scales with the
# boundary, not with the length of the run.

using Oceananigans.Fields: Field, interpolate
using Oceananigans.Grids: Center, Closed, Connected, Face, Open, TwoSided, inactive_cell, new_data, topology,
                          validate_indices, λnode, φnode, rnode, ξnode, ηnode
using Oceananigans.Operators: Δz, rotation_angle
using Oceananigans.OutputReaders: FlavorOfFTS
using Oceananigans.Utils: KernelParameters, launch!
using KernelAbstractions: @kernel, @index

const lateral_sides = (:west, :east, :south, :north)

boundary_dimension(side) = side ∈ (:west, :east) ? 1 : 2

"""
    ParentTimes(previous, current)

The times of the two parent states held by every [`BoundaryBand`](@ref) of a nested ocean. Mutable, so
that refreshing the bands moves them without rebuilding the boundary conditions that read the bands.
"""
mutable struct ParentTimes{FT}
    previous :: FT
    current  :: FT
end

time_bounds(times::ParentTimes) = (times.previous, times.current)
time_bounds(times::Tuple) = times

"""
    BoundaryBand(previous, current, times)

The parent state on one band of the child grid at the two `times` (a [`ParentTimes`](@ref)):
`previous` and `current` are `Field`s restricted to the band. [`band_value`](@ref) interpolates linearly
in time between them, holding the nearer state outside `times`.
"""
struct BoundaryBand{F, T}
    previous :: F
    current  :: F
    times    :: T
end

# Kernels see the times as they are at launch.
Adapt.adapt_structure(to, band::BoundaryBand) =
    BoundaryBand(adapt(to, band.previous), adapt(to, band.current), time_bounds(band.times))

Oceananigans.Architectures.on_architecture(to, band::BoundaryBand) =
    BoundaryBand(on_architecture(to, band.previous), on_architecture(to, band.current), band.times)

"""
    band_value(band, i, j, k, time)

The parent state in `band` at `(i, j, k)` and `time` (a `Time`), interpolated linearly between the band's
two time levels.
"""
@inline function band_value(band::BoundaryBand, i, j, k, time)
    t₀, t₁ = time_bounds(band.times)
    Δt = t₁ - t₀
    w = clamp((time.time - t₀) / Δt, zero(Δt), one(Δt))
    w = ifelse(Δt > 0, w, one(w))
    ψ₀ = @inbounds band.previous[i, j, k]
    ψ₁ = @inbounds band.current[i, j, k]
    return ψ₀ + w * (ψ₁ - ψ₀)
end

# The boundary-adjacent child index of a field with location `ℓ` in the boundary-normal direction:
# the boundary face itself for a boundary-normal velocity, the first interior cell otherwise.
last_index(::Face, N)   = N + 1
last_index(::Center, N) = N

boundary_index(ℓ, side, N) = side ∈ (:west, :south) ? 1 : last_index(ℓ, N)

# Whether the left and right ends of a direction with topology `T` are seams with a neighboring rank.
rank_seams(::Type{TwoSided{L, R}}) where {L, R} = (rank_seam(L), rank_seam(R))
rank_seams(T) = (false, false)

rank_seam(::Type{Connected}) = true
rank_seam(S) = false

# The indices at which a field at location `ℓ` holds the parent along a direction with topology `T`, `N` interior
# cells and data `axis`: the interior points, extended into the halo across a rank seam, where the halo holds
# the neighboring rank's points.
function parent_index_bounds(ℓ, T, N, axis)
    left_seam, right_seam = rank_seams(T)
    return (left_seam ? first(axis) : 1, right_seam ? last(axis) : last_index(ℓ, N))
end

parent_index_bounds(::Nothing, T, N, axis) = (1, 1)

parent_index_bounds(field, loc) =
    map(parent_index_bounds, loc, topology(field.grid), size(field.grid), axes(field.data)[1:length(loc)])

function band_range(ℓ, side, N, width)
    n = last_index(ℓ, N)
    return side ∈ (:west, :south) ? (1:width) : (n - width + 1:n)
end

function band_indices(grid, loc, side, width)
    dim = boundary_dimension(side)
    range = band_range(loc[dim], side, size(grid, dim), width)
    return dim == 1 ? (range, Colon(), Colon()) : (Colon(), range, Colon())
end

"""
    BoundaryBand(grid, loc, side, width, times)

Return a `BoundaryBand` at location `loc` on the outermost `width` rows or columns of `grid` along `side`.
"""
function BoundaryBand(grid, loc, side, width, times)
    indices = validate_indices(band_indices(grid, loc, side, width), loc, grid)
    previous = band_field(grid, loc, indices)
    current  = band_field(grid, loc, indices)
    return BoundaryBand(previous, current, times)
end

# Without boundary conditions, so bands built only on some ranks draw no distributed halo-communication tag
function band_field(grid, loc, indices)
    LX, LY, LZ = typeof.(loc)
    data = new_data(eltype(grid), grid, loc, indices)
    return Field{LX, LY, LZ}(grid, data, nothing, indices, nothing, nothing, nothing)
end

#####
##### Sampling the parent
#####

"""
    ParentVelocity{C}(eastward, northward)

The component `C` (`:u` or `:v`) along the axes of a child grid of a parent velocity whose `eastward` and
`northward` components are `FieldTimeSeries`: both are interpolated onto a child velocity point and rotated there by
the grid's `rotation_angle`.
"""
struct ParentVelocity{C, U, V}
    eastward  :: U
    northward :: V
end

ParentVelocity{C}(eastward::U, northward::V) where {C, U, V} = ParentVelocity{C, U, V}(eastward, northward)

Adapt.adapt_structure(to, v::ParentVelocity{C}) where C = ParentVelocity{C}(adapt(to, v.eastward), adapt(to, v.northward))

parent_grid(source) = source.grid
parent_grid(source::ParentVelocity) = (source.eastward.grid, source.northward.grid)

parent_location(source) = Tuple(L() for L in location(source))
parent_location(source::ParentVelocity) = (parent_location(source.eastward), parent_location(source.northward))

@inline parent_levels_grid(grid) = grid
@inline parent_levels_grid(grids::Tuple) = first(grids)

"""
    child_frame_sources(grid, sources)

Return the parent `sources` as a child on `grid` samples them: unchanged on a `LatitudeLongitudeGrid`, whose
velocity components are eastward and northward like the parent's, and with `u` and `v` replaced by
[`ParentVelocity`](@ref)s on an `OrthogonalSphericalShellGrid`.
"""
child_frame_sources(grid, sources) = sources
child_frame_sources(grid::ImmersedBoundaryGrid, sources) = child_frame_sources(grid.underlying_grid, sources)

child_frame_sources(::OrthogonalSphericalShellGrid, sources) =
    merge(sources, (u = ParentVelocity{:u}(sources.u, sources.v), v = ParentVelocity{:v}(sources.u, sources.v)))

# The parent `source` at the node `X` of the point `(i, j)` of the child `grid` at `(ℓx, ℓy)`
@inline parent_value(source::FlavorOfFTS, X, source_loc, source_grid, time, i, j, grid, ℓx, ℓy) =
    interpolate(X, time, source, source_loc, source_grid)

@inline function parent_value(source::ParentVelocity{C}, X, source_loc, source_grid, time, i, j, grid, ℓx, ℓy) where C
    uₑ = interpolate(X, time, source.eastward,  source_loc[1], source_grid[1])
    vₑ = interpolate(X, time, source.northward, source_loc[2], source_grid[2])
    sinθ, cosθ = sincos(rotation_angle(i, j, grid, ℓx, ℓy))
    return ifelse(C === :u, uₑ * cosθ - vₑ * sinθ, uₑ * sinθ + vₑ * cosθ)
end

# The parent's depths are below its free surface, so a z★ child samples it at its reference depth `rnode`, not its
# `znode`, which rises with η above the parent's top level.
@inline parent_node(i, j, k, grid, ℓx, ℓy, ℓz) = (ξnode(i, j, k, grid, ℓx, ℓy, ℓz), ηnode(i, j, k, grid, ℓx, ℓy, ℓz),
                                                  rnode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline parent_node(i, j, k, grid, ℓx, ℓy, ℓz::Nothing) = (ξnode(i, j, k, grid, ℓx, ℓy, ℓz), ηnode(i, j, k, grid, ℓx, ℓy, ℓz))

"""
    sample_parent!(field, loc, source, time)

Interpolate the parent `source`, a `FieldTimeSeries`, at `time` onto every point of `field`
at location `loc`, halos included. Halo points beyond a wall of the domain, which can lie outside the parent's
region, take the value at the nearest interior point; halo points across a rank seam are sampled where they are.
"""
function sample_parent!(field, loc, source, time)
    grid = field.grid
    ranges = axes(field.data)[1:3]
    params = KernelParameters(length.(ranges), first.(ranges) .- 1)
    bounds = parent_index_bounds(field, loc)
    launch!(architecture(field), grid, params, _sample_parent!,
            field, grid, loc, bounds, source, parent_grid(source), parent_location(source), Time(time))
    return field
end

@kernel function _sample_parent!(field, grid, loc, bounds, source, source_grid, source_loc, time)
    i, j, k = @index(Global, NTuple)
    i′ = clamp(i, bounds[1]...)
    j′ = clamp(j, bounds[2]...)
    k′ = clamp(k, bounds[3]...)
    X = parent_node(i′, j′, k′, grid, loc...)
    @inbounds field[i, j, k] = parent_value(source, X, source_loc, source_grid, time, i′, j′, grid, loc[1], loc[2])
end

"""
    sample_parent_depth_mean!(field, loc, source, floor, time)

Interpolate onto every point of the two-dimensional `field` at horizontal location `loc` the depth mean of the
parent `source` at `time`, over the parent's own levels whose centers lie above the parent's seafloor `floor` (a
bottom height `Field` at the points of `field`), as the reference Mid-Atlantic Bight configuration computes the
barotropic exterior of its Flather condition. Columns where the parent is dry hold zero.
"""
function sample_parent_depth_mean!(field, loc, source, floor, time)
    grid = field.grid
    ranges = axes(field.data)[1:2]
    params = KernelParameters(length.(ranges), first.(ranges) .- 1)
    bounds = parent_index_bounds(field, loc[1:2])
    launch!(architecture(field), grid, params, _sample_parent_depth_mean!,
            field, grid, loc, bounds, source, parent_grid(source), parent_location(source), floor, Time(time))
    return field
end

@kernel function _sample_parent_depth_mean!(field, grid, loc, bounds, source, source_grid, source_loc, floor, time)
    i, j = @index(Global, NTuple)
    i′ = clamp(i, bounds[1]...)
    j′ = clamp(j, bounds[2]...)
    λ = λnode(i′, j′, 1, grid, loc[1], loc[2], Center())
    φ = φnode(i′, j′, 1, grid, loc[1], loc[2], Center())
    zᵇ = @inbounds floor[i′, j′, 1]

    U = zero(grid)
    H = zero(grid)
    levels_grid = parent_levels_grid(source_grid)
    for k in 1:size(levels_grid, 3)
        z = rnode(1, 1, k, levels_grid, Center(), Center(), Center())
        Δzᵏ = Δz(1, 1, k, levels_grid, Center(), Center(), Center())
        u = parent_value(source, (λ, φ, z), source_loc, source_grid, time, i′, j′, grid, loc[1], loc[2])
        wet = z > zᵇ
        U += ifelse(wet, u * Δzᵏ, zero(U))
        H += ifelse(wet, Δzᵏ, zero(H))
    end

    @inbounds field[i, j, 1] = ifelse(H > 0, U / max(H, eps(H)), zero(U))
end

# Whether `side` of a grid with topology `T` in its direction is a wall of the whole domain: a `Closed` or `Open`
# side, not one `Connected` to another rank of a distributed grid.
on_domain_boundary(::Type{TwoSided{L, R}}, side) where {L, R} = wall_side(ifelse(side ∈ (:west, :south), L, R))
on_domain_boundary(T, side) = false

wall_side(::Type{<:Union{Closed, Open}}) = true
wall_side(S) = false

"""
    domain_boundary_sides(grid, sides)

Return the `sides` of `grid` that are walls of the whole domain, `Closed` or `Open`: on a distributed grid only the
sides of the local rank that lie on the edge of the global domain.
"""
function domain_boundary_sides(grid, sides)
    TX, TY, _ = topology(grid)
    side_topology(side) = boundary_dimension(side) == 1 ? TX : TY
    return Tuple(side for side in sides if on_domain_boundary(side_topology(side), side))
end

@kernel function _compute_wet_surface_cells!(wet, grid, Nz)
    i, j = @index(Global, NTuple)
    @inbounds wet[i, j, 1] = !inactive_cell(i, j, Nz, grid)
end

# A CPU matrix of whether each surface cell of `grid` is wet. Moving an immersed grid to the CPU would rebuild its
# immersed boundary, which fails on a rank's local grid.
function compute_wet_surface_cells(grid)
    Nz = size(grid, 3)
    wet = Field{Center, Center, Nothing}(grid)
    launch!(architecture(grid), grid, :xy, _compute_wet_surface_cells!, wet, grid, Nz)
    return on_architecture(CPU(), interior(wet, :, :, 1)) .> 0
end

"""
    open_boundary_sides(grid)

Return the lateral sides of `grid` that are open to the ocean: the walls of the domain (see
[`domain_boundary_sides`](@ref)) with at least one wet surface cell in their boundary-adjacent row or column
anywhere along the domain. Sides that are entirely land stay closed. On a distributed grid, these are the local
rank's sides, the same on every rank along a side, since an `Open` side is one for the whole domain.
"""
function open_boundary_sides(grid)
    Nx, Ny, _ = size(grid)
    wet = compute_wet_surface_cells(grid)
    sides = domain_boundary_sides(grid, lateral_sides)

    touches_water = (west  = any(wet[1,  :]),
                     east  = any(wet[Nx, :]),
                     south = any(wet[:,  1]),
                     north = any(wet[:, Ny]))

    local_sides = Tuple(side for side in sides if touches_water[side])

    return domain_boundary_sides(grid, global_sides(grid, local_sides))
end

# The sides any rank of a distributed grid holds in `sides`; every rank takes part.
function global_sides(grid, sides)
    arch = architecture(grid)
    return Tuple(side for side in lateral_sides if all_reduce(+, Int(side ∈ sides), arch) > 0)
end
