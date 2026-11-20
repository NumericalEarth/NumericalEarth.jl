#####
##### Boundary relaxation of a regional ocean toward its boundary bands
#####

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Forcings: Forcing, InflowOutflowRate, evaluate_rate
using Oceananigans.Grids: peripheral_node
using Oceananigans.Operators: Δz

# Relaxation toward one side's band over its cells `irange × jrange`; the open sides' bands partition the zone
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
@inline nearest_band_indices(r::BandRelaxation, i, j) = (clamp(i, r.irange[1], r.irange[2]), clamp(j, r.jrange[1], r.jrange[2]))

# The wet-column mean of the child field `ψ` and of the parent in band `r`, at (i, j)
@inline function column_means(i, j, grid, loc, ψ, r, time)
    iᵇ, jᵇ = nearest_band_indices(r, i, j)
    H = zero(grid)
    ψ̄ = zero(grid)
    ψ̄ᵖ = zero(grid)
    for k in 1:size(grid, 3)
        w = ifelse(peripheral_node(i, j, k, grid, loc...), zero(grid), Δz(i, j, k, grid, loc...))
        H += w
        ψ̄ += w * @inbounds ψ[i, j, k]
        ψ̄ᵖ += w * band_value(r.band, iᵇ, jᵇ, k, time)
    end
    H = max(H, eps(H))
    return ψ̄ / H, ψ̄ᵖ / H
end

@inline function band_anomaly(i, j, k, grid, loc, ψ, r, time, ::Val{false})
    iᵇ, jᵇ = nearest_band_indices(r, i, j)
    ψᵖ = band_value(r.band, iᵇ, jᵇ, k, time)
    return ψᵖ - @inbounds ψ[i, j, k]
end

# The baroclinic anomaly, which leaves the barotropic flow to Flather
@inline function band_anomaly(i, j, k, grid, loc, ψ, r, time, ::Val{true})
    iᵇ, jᵇ = nearest_band_indices(r, i, j)
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

@inline relaxed_field(fields, ::Val{name}) where name = getproperty(fields, name)

@inline function relaxation_tendency(i, j, k, grid, clock, fields, p)
    μ = @inbounds p.mask[i, j, 1]
    ψ = relaxed_field(fields, p.name)
    return μ * total_band_relaxation(i, j, k, grid, p.location, ψ, fields, p.relaxations, Time(clock.time), p.baroclinic)
end

# An `(; inflow, outflow)` rate is flow dependent for the tracers; the velocities relax at the inflow rate.
side_rate(rate::Number, side, baroclinic, FT) = convert(FT, rate)
side_rate(rate::NamedTuple, side, ::Val{true}, FT) = convert(FT, rate.inflow)
side_rate(rate::NamedTuple, side, ::Val{false}, FT) =
    InflowOutflowRate{side}(inflow_rate = convert(FT, rate.inflow), outflow_rate = convert(FT, rate.outflow))

# Along the boundary a band stops short of the perpendicular open sides' bands, so the corners belong to west and east
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

function relaxation_masks(grid, sides, width)
    Nx, Ny, _ = size(grid)
    wet = compute_wet_surface_cells(grid)
    taper(d) = d < width ? cos(π * d / 2width)^2 : 0.0

    keep = ones(Nx, Ny)   # ∏(1 - mₛ)

    # Taper `cells`, ordered inward from an open side, up to the first land cell
    function taper_inward!(cells)
        for (n, cell) in enumerate(cells)
            wet[cell] || break
            keep[cell] *= 1 - taper(n - 1/2)
        end
    end

    for j in 1:Ny
        :west in sides && taper_inward!(CartesianIndex(i, j) for i in 1:min(width, Nx))
        :east in sides && taper_inward!(CartesianIndex(i, j) for i in Nx:-1:max(Nx - width + 1, 1))
    end
    for i in 1:Nx
        :south in sides && taper_inward!(CartesianIndex(i, j) for j in 1:min(width, Ny))
        :north in sides && taper_inward!(CartesianIndex(i, j) for j in Ny:-1:max(Ny - width + 1, 1))
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

function ocean_boundary_relaxation(grid, bands; rate, width)
    sides = keys(bands)
    names = filter(name -> name ∈ (:u, :v, :T, :S), keys(first(bands)))
    masks = relaxation_masks(grid, sides, width)

    function relaxation_forcing(name)
        loc = boundary_location(name)
        baroclinic = Val(name ∈ (:u, :v))
        relaxations = map(sides) do side
            irange, jrange = band_relaxation_ranges(grid, loc, side, width, sides)
            BandRelaxation{boundary_dimension(side)}(bands[side][name], irange, jrange,
                                                     side_rate(rate, side, baroclinic, eltype(grid)))
        end
        mask = name === :u ? masks.u : name === :v ? masks.v : masks.c
        parameters = (; name = Val(name), mask, location = loc, relaxations, baroclinic)
        return Forcing(relaxation_tendency; discrete_form = true, parameters)
    end

    return NamedTuple(name => relaxation_forcing(name) for name in names)
end
