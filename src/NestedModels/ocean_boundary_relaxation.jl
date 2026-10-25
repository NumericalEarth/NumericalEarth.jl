#####
##### Boundary relaxation of a regional ocean toward its boundary bands
#####

using Oceananigans.Forcings: Forcing

"""
    BandRelaxation{Dim}(band, irange, jrange, origin, sign, width)

Relaxation toward one open side's [`BoundaryBand`](@ref) over the cells `irange × jrange` of the band. The
relaxation weight tapers as `cos²(π d / 2 width)` with the distance `d = sign * (n - origin)` in cells from
the boundary, where `n` is the boundary-normal index (`i` for `Dim = 1`, `j` for `Dim = 2`).
"""
struct BandRelaxation{Dim, B, FT}
    band   :: B
    irange :: NTuple{2, Int}
    jrange :: NTuple{2, Int}
    origin :: FT
    sign   :: Int
    width  :: Int
end

BandRelaxation{Dim}(band::B, irange, jrange, origin::FT, sign, width) where {Dim, B, FT} =
    BandRelaxation{Dim, B, FT}(band, irange, jrange, origin, sign, width)

Adapt.adapt_structure(to, r::BandRelaxation{Dim}) where Dim =
    BandRelaxation{Dim}(adapt(to, r.band), r.irange, r.jrange, r.origin, r.sign, r.width)

@inline boundary_normal_index(::BandRelaxation{1}, i, j) = i
@inline boundary_normal_index(::BandRelaxation{2}, i, j) = j

@inline relaxation_taper(d, width) = ifelse(d < width, cos(π * max(0, d) / 2width)^2, zero(d))

@inline function band_relaxation(i, j, k, time, ψ, r::BandRelaxation)
    i₁, i₂ = r.irange
    j₁, j₂ = r.jrange
    inside = (i₁ ≤ i) & (i ≤ i₂) & (j₁ ≤ j) & (j ≤ j₂)
    d = r.sign * (boundary_normal_index(r, i, j) - r.origin)
    μ = relaxation_taper(d, r.width)
    μ = ifelse(inside, μ, zero(μ))
    ψᵖ = band_value(r.band, clamp(i, i₁, i₂), clamp(j, j₁, j₂), k, time)
    ψᶜ = @inbounds ψ[i, j, k]
    return μ * (ψᵖ - ψᶜ)
end

@inline total_band_relaxation(i, j, k, time, ψ, ::Tuple{}) = 0
@inline total_band_relaxation(i, j, k, time, ψ, relaxations::Tuple) =
    band_relaxation(i, j, k, time, ψ, first(relaxations)) + total_band_relaxation(i, j, k, time, ψ, Base.tail(relaxations))

@inline relax_u(i, j, k, grid, clock, fields, p) = p.rate * total_band_relaxation(i, j, k, Time(clock.time), fields.u, p.relaxations)
@inline relax_v(i, j, k, grid, clock, fields, p) = p.rate * total_band_relaxation(i, j, k, Time(clock.time), fields.v, p.relaxations)
@inline relax_T(i, j, k, grid, clock, fields, p) = p.rate * total_band_relaxation(i, j, k, Time(clock.time), fields.T, p.relaxations)
@inline relax_S(i, j, k, grid, clock, fields, p) = p.rate * total_band_relaxation(i, j, k, Time(clock.time), fields.S, p.relaxations)

const relaxation_functions = (u = relax_u, v = relax_v, T = relax_T, S = relax_S)

# Cells of `side`'s band that it relaxes. Along the boundary the band stops short of the bands of the
# perpendicular open sides, so that no corner cell is relaxed twice: the corners belong to west and east.
function band_relaxation_ranges(grid, loc, side, width, sides)
    dim = boundary_dimension(side)
    Nx, Ny, _ = size(grid)
    normal = band_range(loc[dim], side, size(grid, dim), width)
    normal_range = (first(normal), last(normal))

    if dim == 1
        along_range = (1, last_index(loc[2], Ny))
        return normal_range, along_range
    else
        nx = last_index(loc[1], Nx)
        i₁ = :west in sides ? width + 1 : 1
        i₂ = :east in sides ? nx - width : nx
        return (i₁, i₂), normal_range
    end
end

function band_relaxation_geometry(grid, loc, side)
    dim = boundary_dimension(side)
    ℓ = loc[dim]
    N = size(grid, dim)
    FT = eltype(grid)
    origin = side ∈ (:west, :south) ? (ℓ isa Face ? 1 : 1/2) : (ℓ isa Face ? N + 1 : N + 1/2)
    sign = side ∈ (:west, :south) ? 1 : -1
    return convert(FT, origin), sign
end

"""
    ocean_boundary_relaxation(grid, bands; rate, width)

Return a `NamedTuple` of discrete `Forcing`s that relax `u`, `v`, `T` and `S` toward the parent state in
`bands` (see [`ocean_open_boundary_conditions`](@ref)) at `rate` (s⁻¹), over the outermost `width` cells
of each open side with a `cos²` taper from the boundary.
"""
function ocean_boundary_relaxation(grid, bands; rate, width)
    sides = keys(bands)
    names = filter(name -> name ∈ keys(relaxation_functions), keys(first(bands)))

    function relaxation(name)
        loc = boundary_location(name)
        relaxations = map(sides) do side
            irange, jrange = band_relaxation_ranges(grid, loc, side, width, sides)
            origin, sign = band_relaxation_geometry(grid, loc, side)
            BandRelaxation{boundary_dimension(side)}(bands[side][name], irange, jrange, origin, sign, width)
        end
        parameters = (; rate = convert(eltype(grid), rate), relaxations)
        return Forcing(relaxation_functions[name]; discrete_form = true, parameters)
    end

    return NamedTuple(name => relaxation(name) for name in names)
end
