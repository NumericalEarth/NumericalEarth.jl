#####
##### Boundary bands: the parent ocean sampled onto the child grid along each open side
#####

using Oceananigans.Fields: Field, interpolate
using Oceananigans.Grids: Center, Closed, Connected, Face, Open, TwoSided, inactive_cell, new_data, topology,
                          validate_indices, λnode, φnode, rnode, ξnode, ηnode
using Oceananigans.Operators: Δz
using Oceananigans.Utils: KernelParameters, launch!
using KernelAbstractions: @kernel, @index

const lateral_sides = (:west, :east, :south, :north)

boundary_dimension(side) = side ∈ (:west, :east) ? 1 : 2

# The (i, j) of the point at index `normal` across and `along` along a side normal to dimension `Dim`
@inline horizontal_indices(::Val{1}, normal, along) = (normal, along)
@inline horizontal_indices(::Val{2}, normal, along) = (along, normal)

# The times of the two parent states the bands hold, mutable so that the boundary conditions follow them
mutable struct ParentTimes{FT}
    previous :: FT
    current  :: FT
end

time_bounds(times::ParentTimes) = (times.previous, times.current)
time_bounds(times::Tuple) = times

# The parent at two times on one band of the child grid
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

# Linear in time between the band's two states, holding the nearer one outside them
@inline function band_value(band::BoundaryBand, i, j, k, time)
    t₀, t₁ = time_bounds(band.times)
    Δt = t₁ - t₀
    w = clamp((time.time - t₀) / Δt, zero(Δt), one(Δt))
    w = ifelse(Δt > 0, w, one(w))
    ψ₀ = @inbounds band.previous[i, j, k]
    ψ₁ = @inbounds band.current[i, j, k]
    return ψ₀ + w * (ψ₁ - ψ₀)
end

# The boundary face for a boundary-normal velocity, the first interior cell otherwise
last_index(::Face, N)   = N + 1
last_index(::Center, N) = N

boundary_index(ℓ, side, N) = side ∈ (:west, :south) ? 1 : last_index(ℓ, N)

# Whether the left and right ends of a direction with topology `T` are seams with a neighboring rank.
rank_seams(::Type{TwoSided{L, R}}) where {L, R} = (L === Connected, R === Connected)
rank_seams(T) = (false, false)

# The interior points along a direction, extended into the halo across a rank seam
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

# A z★ child samples the parent at its reference depth `rnode`, since its `znode` rises with η above the parent's top
@inline parent_node(i, j, k, grid, ℓx, ℓy, ℓz) = (ξnode(i, j, k, grid, ℓx, ℓy, ℓz), ηnode(i, j, k, grid, ℓx, ℓy, ℓz),
                                                  rnode(i, j, k, grid, ℓx, ℓy, ℓz))
@inline parent_node(i, j, k, grid, ℓx, ℓy, ℓz::Nothing) = (ξnode(i, j, k, grid, ℓx, ℓy, ℓz), ηnode(i, j, k, grid, ℓx, ℓy, ℓz))

# Halo points beyond a wall take the nearest interior point's value; across a rank seam they are sampled in place
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
    X = parent_node(i′, j′, k′, grid, loc...)
    @inbounds field[i, j, k] = interpolate(X, time, source, source_loc, source_grid)
end

# The parent's mean over its own levels above its seafloor `floor`, zero where it is dry
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
        z = rnode(1, 1, k, source_grid, Center(), Center(), Center())
        Δzᵏ = Δz(1, 1, k, source_grid, Center(), Center(), Center())
        u = interpolate((λ, φ, z), time, source, source_loc, source_grid)
        wet = z > zᵇ
        U += ifelse(wet, u * Δzᵏ, zero(U))
        H += ifelse(wet, Δzᵏ, zero(H))
    end

    @inbounds field[i, j, 1] = ifelse(H > 0, U / max(H, eps(H)), zero(U))
end

# Whether `side` is `Closed` or `Open`, a wall of the whole domain rather than a seam with another rank
on_domain_boundary(::Type{TwoSided{L, R}}, side) where {L, R} = (side ∈ (:west, :south) ? L : R) <: Union{Closed, Open}
on_domain_boundary(T, side) = false

function domain_boundary_sides(grid, sides)
    TX, TY, _ = topology(grid)
    side_topology(side) = boundary_dimension(side) == 1 ? TX : TY
    return Tuple(side for side in sides if on_domain_boundary(side_topology(side), side))
end

@kernel function _compute_wet_surface_cells!(wet, grid, Nz)
    i, j = @index(Global, NTuple)
    @inbounds wet[i, j, 1] = !inactive_cell(i, j, Nz, grid)
end

# Computed on the grid's own architecture, since moving a rank's immersed grid to the CPU fails
function compute_wet_surface_cells(grid)
    Nz = size(grid, 3)
    wet = Field{Center, Center, Nothing}(grid)
    launch!(architecture(grid), grid, :xy, _compute_wet_surface_cells!, wet, grid, Nz)
    return on_architecture(CPU(), interior(wet, :, :, 1)) .> 0
end

# The walls of the domain next to a wet surface cell anywhere along them
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
