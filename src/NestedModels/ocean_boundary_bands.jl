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

using Oceananigans.Fields: Field, flatten_node, interpolate
using Oceananigans.Grids: Bounded, Center, Face, inactive_cell, topology, _node
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
    LX, LY, LZ = typeof.(loc)
    indices = band_indices(grid, loc, side, width)
    previous = Field{LX, LY, LZ}(grid; indices)
    current  = Field{LX, LY, LZ}(grid; indices)
    return BoundaryBand(previous, current, times)
end

#####
##### Sampling the parent
#####

@inline parent_value(source::FlavorOfFTS, X, loc, grid, time) = interpolate(X, time, source, loc, grid)

"""
    sample_parent!(field, loc, source, time)

Interpolate the parent `source`, a `FieldTimeSeries`, at `time` onto every point of `field`
at location `loc`, halos included. Halo points, which can lie outside the parent's region, take the value
at the nearest interior point.
"""
function sample_parent!(field, loc, source, time)
    grid = field.grid
    ranges = axes(field.data)[1:3]
    params = KernelParameters(length.(ranges), first.(ranges) .- 1)
    interior_size = map((ℓ, N) -> isnothing(ℓ) ? 1 : last_index(ℓ, N), loc, size(grid))
    source_loc = Tuple(L() for L in location(source))
    launch!(architecture(field), grid, params, _sample_parent!,
            field, grid, loc, interior_size, source, source.grid, source_loc, Time(time))
    return field
end

@kernel function _sample_parent!(field, grid, loc, N, source, source_grid, source_loc, time)
    i, j, k = @index(Global, NTuple)
    i′ = clamp(i, 1, N[1])
    j′ = clamp(j, 1, N[2])
    k′ = clamp(k, 1, N[3])
    X = flatten_node(_node(i′, j′, k′, grid, loc...)...)
    @inbounds field[i, j, k] = parent_value(source, X, source_loc, source_grid, time)
end

"""
    open_boundary_sides(grid)

Return the lateral sides of `grid` that are open to the ocean: the `Bounded` sides with at least one
wet surface cell in their boundary-adjacent row or column. Sides that are entirely land stay closed.
"""
function open_boundary_sides(grid)
    TX, TY, _ = topology(grid)
    Nx, Ny, Nz = size(grid)
    cpu_grid = on_architecture(CPU(), grid)
    wet(i, j) = !inactive_cell(i, j, Nz, cpu_grid)

    open = (west  = TX === Bounded && any(j -> wet(1,  j), 1:Ny),
            east  = TX === Bounded && any(j -> wet(Nx, j), 1:Ny),
            south = TY === Bounded && any(i -> wet(i,  1), 1:Nx),
            north = TY === Bounded && any(i -> wet(i, Ny), 1:Nx))

    return Tuple(side for side in lateral_sides if open[side])
end
