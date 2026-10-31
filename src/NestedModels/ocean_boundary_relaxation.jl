#####
##### Boundary relaxation of a regional ocean toward its boundary bands
#####
#
# A mask falls as cos² from the open sides, two sides combine near a corner as 1 - (1 - m₁)(1 - m₂), and a side's
# band stops at the first land cell in from the boundary, so bays and sounds behind land are left alone. Tracers
# relax as they are; the velocities relax only their baroclinic part, which integrates to zero over the column and
# leaves the barotropic flow to the Flather condition.

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Forcings: Forcing, InflowOutflowRate, evaluate_rate
using Oceananigans.Grids: peripheral_node
using Oceananigans.Operators: Δz

"""
    BandRelaxation{Dim}(band, irange, jrange, rate)

Relaxation toward one open side's [`BoundaryBand`](@ref) over the cells `irange × jrange` of the band, at `rate`: a
number, or an `InflowOutflowRate` of the side. The bands of the open sides partition the relaxation zone, so that
each cell reads the parent from one of them.
"""
struct BandRelaxation{Dim, B, R}
    band   :: B
    irange :: NTuple{2, Int}
    jrange :: NTuple{2, Int}
    rate   :: R
end

BandRelaxation{Dim}(band::B, irange, jrange, rate::R) where {Dim, B, R} = BandRelaxation{Dim, B, R}(band, irange, jrange, rate)

Adapt.adapt_structure(to, r::BandRelaxation{Dim}) where Dim =
    BandRelaxation{Dim}(adapt(to, r.band), r.irange, r.jrange, r.rate)

@inline inside(r::BandRelaxation, i, j) = (r.irange[1] ≤ i) & (i ≤ r.irange[2]) & (r.jrange[1] ≤ j) & (j ≤ r.jrange[2])
@inline band_indices(r::BandRelaxation, i, j) = (clamp(i, r.irange[1], r.irange[2]), clamp(j, r.jrange[1], r.jrange[2]))

@inline column_weight(i, j, k, grid, loc) = ifelse(peripheral_node(i, j, k, grid, loc...), zero(grid), Δz(i, j, k, grid, loc...))

# The wet-column mean of the child field `ψ` and of the parent in band `r`, at (i, j)
@inline function column_means(i, j, grid, loc, ψ, r, time)
    iᵇ, jᵇ = band_indices(r, i, j)
    H = zero(grid)
    ψ̄ = zero(grid)
    ψ̄ᵖ = zero(grid)
    for k in 1:size(grid, 3)
        w = column_weight(i, j, k, grid, loc)
        H += w
        ψ̄ += w * @inbounds ψ[i, j, k]
        ψ̄ᵖ += w * band_value(r.band, iᵇ, jᵇ, k, time)
    end
    H = max(H, eps(H))
    return ψ̄ / H, ψ̄ᵖ / H
end

@inline function band_anomaly(i, j, k, grid, loc, ψ, r, time, ::Val{false})
    iᵇ, jᵇ = band_indices(r, i, j)
    ψᵖ = band_value(r.band, iᵇ, jᵇ, k, time)
    return ψᵖ - @inbounds ψ[i, j, k]
end

@inline function band_anomaly(i, j, k, grid, loc, ψ, r, time, ::Val{true})
    iᵇ, jᵇ = band_indices(r, i, j)
    ψ̄, ψ̄ᵖ = column_means(i, j, grid, loc, ψ, r, time)
    ψᵖ = band_value(r.band, iᵇ, jᵇ, k, time)
    return (ψᵖ - ψ̄ᵖ) - (@inbounds(ψ[i, j, k]) - ψ̄)
end

@inline band_rate(i, j, k, grid, rate::Number, fields, loc) = rate
@inline band_rate(i, j, k, grid, rate::InflowOutflowRate, fields, loc) = evaluate_rate(i, j, k, grid, rate, fields, loc)

@inline function band_relaxation(i, j, k, grid, loc, ψ, fields, r, time, baroclinic)
    δ = band_rate(i, j, k, grid, r.rate, fields, loc) * band_anomaly(i, j, k, grid, loc, ψ, r, time, baroclinic)
    return ifelse(inside(r, i, j), δ, zero(δ))
end

@inline total_band_relaxation(i, j, k, grid, loc, ψ, fields, ::Tuple{}, time, baroclinic) = 0
@inline total_band_relaxation(i, j, k, grid, loc, ψ, fields, relaxations::Tuple, time, baroclinic) =
    band_relaxation(i, j, k, grid, loc, ψ, fields, first(relaxations), time, baroclinic) +
    total_band_relaxation(i, j, k, grid, loc, ψ, fields, Base.tail(relaxations), time, baroclinic)

@inline function relaxation_tendency(i, j, k, grid, clock, ψ, fields, p)
    μ = @inbounds p.mask[i, j, 1]
    return μ * total_band_relaxation(i, j, k, grid, p.location, ψ, fields, p.relaxations, Time(clock.time), p.baroclinic)
end

@inline relax_u(i, j, k, grid, clock, fields, p) = relaxation_tendency(i, j, k, grid, clock, fields.u, fields, p)
@inline relax_v(i, j, k, grid, clock, fields, p) = relaxation_tendency(i, j, k, grid, clock, fields.v, fields, p)
@inline relax_T(i, j, k, grid, clock, fields, p) = relaxation_tendency(i, j, k, grid, clock, fields.T, fields, p)
@inline relax_S(i, j, k, grid, clock, fields, p) = relaxation_tendency(i, j, k, grid, clock, fields.S, fields, p)

const relaxation_functions = (u = relax_u, v = relax_v, T = relax_T, S = relax_S)

# The velocities relax only their baroclinic part.
baroclinic_relaxation(name) = Val(name ∈ (:u, :v))

# An `(; inflow, outflow)` rate is flow dependent for the tracers; the velocities relax at the inflow rate.
side_rate(rate::Number, side, baroclinic, FT) = convert(FT, rate)
side_rate(rate::NamedTuple, side, ::Val{true}, FT) = convert(FT, rate.inflow)
side_rate(rate::NamedTuple, side, ::Val{false}, FT) =
    InflowOutflowRate{side}(inflow_rate = convert(FT, rate.inflow), outflow_rate = convert(FT, rate.outflow))

# Cells of `side`'s band that it relaxes. Along the boundary the band stops short of the bands of the
# perpendicular open sides, so that each corner cell reads the parent once: the corners belong to west and east.
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

"""
    relaxation_masks(grid, sides, width)

The relaxation masks of a regional ocean on `grid` with open `sides`, at cell centers and at `u` and `v` points:
`cos²(π d / 2 width)` at the distance `d` in cells from each open side's boundary face, zero from `width` cells in
and beyond the first land cell in from the boundary, combined across sides as `1 - ∏(1 - mₛ)`. The masks at the
velocity points average the two neighboring cell centers.
"""
function relaxation_masks(grid, sides, width)
    Nx, Ny, _ = size(grid)
    wet = compute_wet_surface_cells(grid)
    taper(d) = d < width ? cos(π * d / 2width)^2 : 0.0

    keep = ones(Nx, Ny)   # ∏(1 - mₛ)
    for j in 1:Ny
        if :west in sides
            for i in 1:min(width, Nx)
                wet[i, j] || break
                keep[i, j] *= 1 - taper(i - 1/2)
            end
        end
        if :east in sides
            for i in Nx:-1:max(Nx - width + 1, 1)
                wet[i, j] || break
                keep[i, j] *= 1 - taper(Nx - i + 1/2)
            end
        end
    end
    for i in 1:Nx
        if :south in sides
            for j in 1:min(width, Ny)
                wet[i, j] || break
                keep[i, j] *= 1 - taper(j - 1/2)
            end
        end
        if :north in sides
            for j in Ny:-1:max(Ny - width + 1, 1)
                wet[i, j] || break
                keep[i, j] *= 1 - taper(Ny - j + 1/2)
            end
        end
    end

    μᶜ = Field{Center, Center, Nothing}(grid)
    interior(μᶜ) .= on_architecture(architecture(grid), 1 .- keep)

    # The halos hold the neighboring rank's mask on a distributed grid, and the boundary cell's mask at a wall.
    fill_halo_regions!(μᶜ)
    μ = on_architecture(CPU(), μᶜ.data)

    μᵘ = Field{Face, Center, Nothing}(grid)
    μᵛ = Field{Center, Face, Nothing}(grid)
    Nxᵘ, Nyᵘ, _ = size(μᵘ)
    Nxᵛ, Nyᵛ, _ = size(μᵛ)
    interior(μᵘ) .= on_architecture(architecture(grid), [(μ[i - 1, j, 1] + μ[i, j, 1]) / 2 for i in 1:Nxᵘ, j in 1:Nyᵘ, _ in 1:1])
    interior(μᵛ) .= on_architecture(architecture(grid), [(μ[i, j - 1, 1] + μ[i, j, 1]) / 2 for i in 1:Nxᵛ, j in 1:Nyᵛ, _ in 1:1])

    return (u = μᵘ, v = μᵛ, c = μᶜ)
end

"""
    ocean_boundary_relaxation(grid, bands; rate, width)

Return a `NamedTuple` of discrete `Forcing`s that relax `T` and `S`, and the baroclinic parts of `u` and `v`, toward
the parent state in `bands` (see [`ocean_open_boundary_conditions`](@ref)) at `rate` (s⁻¹) times the
[`relaxation_masks`](@ref) over the outermost `width` cells of each open side. The baroclinic part of a velocity is
its departure from its mean over the child's wet column, for the child and the parent alike.

`rate` is a number, or `(inflow = ..., outflow = ...)` for a flow-dependent relaxation of the tracers: in each side's
band, `T` and `S` relax at `inflow` where the velocity at the cell flows in through that side and at `outflow` where
it flows out (Oceananigans' `InflowOutflowRate`). The velocities relax at `inflow` everywhere.
"""
function ocean_boundary_relaxation(grid, bands; rate, width)
    sides = keys(bands)
    names = filter(name -> name ∈ keys(relaxation_functions), keys(first(bands)))
    masks = relaxation_masks(grid, sides, width)

    function relaxation_forcing(name)
        loc = boundary_location(name)
        baroclinic = baroclinic_relaxation(name)
        relaxations = map(sides) do side
            irange, jrange = band_relaxation_ranges(grid, loc, side, width, sides)
            BandRelaxation{boundary_dimension(side)}(bands[side][name], irange, jrange,
                                                     side_rate(rate, side, baroclinic, eltype(grid)))
        end
        mask = name === :u ? masks.u : name === :v ? masks.v : masks.c
        parameters = (; mask, location = loc, relaxations, baroclinic)
        return Forcing(relaxation_functions[name]; discrete_form = true, parameters)
    end

    return NamedTuple(name => relaxation_forcing(name) for name in names)
end
