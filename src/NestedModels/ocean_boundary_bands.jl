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

using Oceananigans.Fields: Field, flatten_node, interpolate, validate_indices
using Oceananigans.Grids: Bounded, Center, Face, FullyConnected, LeftConnected, RightConnected, inactive_cell, new_data, topology,
                          _node, λnode, φnode, znode
using Oceananigans.Operators: Δz
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
rank_seams(::Type{FullyConnected}) = (true, true)
rank_seams(::Type{LeftConnected}) = (true, false)
rank_seams(::Type{RightConnected}) = (false, true)
rank_seams(T) = (false, false)

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

@inline parent_value(source::FlavorOfFTS, X, loc, grid, time) = interpolate(X, time, source, loc, grid)

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
    source_loc = Tuple(L() for L in location(source))
    launch!(architecture(field), grid, params, _sample_parent!,
            field, grid, loc, bounds, source, source.grid, source_loc, Time(time))
    return field
end

@kernel function _sample_parent!(field, grid, loc, bounds, source, source_grid, source_loc, time)
    i, j, k = @index(Global, NTuple)
    i′ = clamp(i, bounds[1]...)
    j′ = clamp(j, bounds[2]...)
    k′ = clamp(k, bounds[3]...)
    X = flatten_node(_node(i′, j′, k′, grid, loc...)...)
    @inbounds field[i, j, k] = parent_value(source, X, source_loc, source_grid, time)
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
    source_loc = Tuple(L() for L in location(source))
    launch!(architecture(field), grid, params, _sample_parent_depth_mean!,
            field, grid, loc, bounds, source, source.grid, source_loc, floor, Time(time))
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
    for k in 1:size(source_grid, 3)
        z = znode(1, 1, k, source_grid, Center(), Center(), Center())
        Δzᵏ = Δz(1, 1, k, source_grid, Center(), Center(), Center())
        u = parent_value(source, (λ, φ, z), source_loc, source_grid, time)
        wet = z > zᵇ
        U += ifelse(wet, u * Δzᵏ, zero(U))
        H += ifelse(wet, Δzᵏ, zero(H))
    end

    @inbounds field[i, j, 1] = ifelse(H > 0, U / max(H, eps(H)), zero(U))
end

# Whether `side` of a grid with topology `T` in its direction is a wall of the whole domain: on a distributed grid,
# a `Bounded` direction is `RightConnected` on its first rank, `LeftConnected` on its last, `FullyConnected` between.
on_domain_boundary(::Type{Bounded}, side) = true
on_domain_boundary(::Type{RightConnected}, side) = side ∈ (:west, :south)
on_domain_boundary(::Type{LeftConnected}, side) = side ∈ (:east, :north)
on_domain_boundary(T, side) = false

"""
    domain_boundary_sides(grid, sides)

Return the `sides` of `grid` that are walls of the whole domain: all `Bounded` sides of a serial grid, and on a
distributed grid only the sides of the local rank that lie on the edge of the global domain.
"""
function domain_boundary_sides(grid, sides)
    TX, TY, _ = topology(grid)
    side_topology(side) = boundary_dimension(side) == 1 ? TX : TY
    return Tuple(side for side in sides if on_domain_boundary(side_topology(side), side))
end

"""
    open_boundary_sides(grid)

Return the lateral sides of `grid` that are open to the ocean: the sides on a `Bounded` wall of the domain
(see [`domain_boundary_sides`](@ref)) with at least one wet surface cell in their boundary-adjacent row or
column. Sides that are entirely land stay closed. On a distributed grid, these are the local rank's sides.
"""
function open_boundary_sides(grid)
    Nx, Ny, Nz = size(grid)
    cpu_grid = on_architecture(CPU(), grid)
    wet(i, j) = !inactive_cell(i, j, Nz, cpu_grid)

    touches_water = (west  = any(j -> wet(1,  j), 1:Ny),
                     east  = any(j -> wet(Nx, j), 1:Ny),
                     south = any(i -> wet(i,  1), 1:Nx),
                     north = any(i -> wet(i, Ny), 1:Nx))

    return Tuple(side for side in domain_boundary_sides(grid, lateral_sides) if touches_water[side])
end
