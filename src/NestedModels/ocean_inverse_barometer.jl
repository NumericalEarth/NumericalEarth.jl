#####
##### The inverse-barometer sea level of a nested ocean's Flather exterior
#####

using Oceananigans: UpdateStateCallsite
using Oceananigans.Operators: Azᶜᶜᶜ
using Oceananigans.Simulations: Callback
using DocStringExtensions: TYPEDEF, TYPEDFIELDS

using ..Oceans: forcing_barotropic_potential

"""
    InverseBarometerSeaLevel{Dim}(sea_level, inverse_barometer)

The Flather exterior sea level of one open side: the parent's `sea_level`, a [`BoundaryTimeSeries`](@ref), plus
the child's inverse-barometer sea level `inverse_barometer` at the same boundary point (see [`InverseBarometer`](@ref)).
"""
struct InverseBarometerSeaLevel{Dim, S, I}
    sea_level         :: S
    inverse_barometer :: I
end

InverseBarometerSeaLevel{Dim}(sea_level::S, inverse_barometer::I) where {Dim, S, I} =
    InverseBarometerSeaLevel{Dim, S, I}(sea_level, inverse_barometer)

Adapt.adapt_structure(to, η::InverseBarometerSeaLevel{Dim}) where Dim =
    InverseBarometerSeaLevel{Dim}(adapt(to, η.sea_level), adapt(to, η.inverse_barometer))

Oceananigans.Architectures.on_architecture(to, η::InverseBarometerSeaLevel{Dim}) where Dim =
    InverseBarometerSeaLevel{Dim}(on_architecture(to, η.sea_level), on_architecture(to, η.inverse_barometer))

@inline function BoundaryConditions.getbc(η::InverseBarometerSeaLevel{1}, j::Integer, k::Integer, grid::AbstractGrid,
                                          clock=nothing, args...)
    b = η.sea_level
    ηᴵᴮ = @inbounds η.inverse_barometer[b.index, clamp(j, b.along...), 1]
    return getbc(b, j, k, grid, clock, args...) + ηᴵᴮ
end

@inline function BoundaryConditions.getbc(η::InverseBarometerSeaLevel{2}, i::Integer, k::Integer, grid::AbstractGrid,
                                          clock=nothing, args...)
    b = η.sea_level
    ηᴵᴮ = @inbounds η.inverse_barometer[clamp(i, b.along...), b.index, 1]
    return getbc(b, i, k, grid, clock, args...) + ηᴵᴮ
end

with_inverse_barometer(sea_level, ::Nothing) = sea_level
with_inverse_barometer(sea_level::BoundaryTimeSeries{Dim}, inverse_barometer::Field) where Dim =
    InverseBarometerSeaLevel{Dim}(sea_level, inverse_barometer)

"""
$(TYPEDEF)

The inverse-barometer sea level `ηᴵᴮ = -(pᵃ / ρ₀ - ⟨pᵃ / ρ₀⟩) / g` of the atmospheric pressure potential `pᵃ / ρ₀`
that a nested ocean is forced with, `⟨⟩` being the mean over the wet surface area of the whole child domain, so that
`ηᴵᴮ` moves no volume through the open sides. A parent without atmospheric pressure (such as an ocean reanalysis)
carries no inverse-barometer response, which the child's open boundaries then add to the parent's sea level.

$(TYPEDFIELDS)
"""
struct InverseBarometer{S, P, A, FT}
    "the inverse-barometer sea level, on the child grid"
    sea_level :: S
    "the child's barotropic pressure potential `pᵃ / ρ₀`"
    potential :: P
    "the wet area of each surface cell of the child grid"
    wet_surface_area :: A
    "the wet surface area of the whole child domain"
    total_wet_surface_area :: FT
    "the gravitational acceleration of the child's free surface"
    gravitational_acceleration :: FT
end

@kernel function _compute_wet_surface_area!(area, grid, Nz)
    i, j = @index(Global, NTuple)
    @inbounds area[i, j, 1] = ifelse(inactive_cell(i, j, Nz, grid), zero(grid), Azᶜᶜᶜ(i, j, Nz, grid))
end

function compute_wet_surface_area!(area, grid)
    launch!(architecture(grid), grid, :xy, _compute_wet_surface_area!, area, grid, size(grid, 3))
    return area
end

"""
$(TYPEDSIGNATURES)

Return the [`InverseBarometer`](@ref) of `child` that writes into `sea_level`, a `Field{Center, Center, Nothing}`
on the child grid.
"""
function InverseBarometer(child, sea_level)
    grid = child.grid
    potential = forcing_barotropic_potential(child)

    isnothing(potential) &&
        throw(ArgumentError("inverse_barometer = true requires a child ocean forced by the atmospheric pressure."))

    area = compute_wet_surface_area!(Field{Center, Center, Nothing}(grid), grid)
    total_area = all_reduce(+, sum(interior(area)), architecture(grid))
    g = convert(eltype(grid), child.free_surface.gravitational_acceleration)

    return InverseBarometer(sea_level, potential, area, convert(eltype(grid), total_area), g)
end

function update_inverse_barometer!(model, ib::InverseBarometer)
    ηᴵᴮ = parent(ib.sea_level)
    ηᴵᴮ .= parent(ib.potential)

    Σ = mapreduce(*, +, interior(ib.sea_level), interior(ib.wet_surface_area))
    mean_potential = all_reduce(+, Σ, architecture(model.grid)) / ib.total_wet_surface_area

    ηᴵᴮ .= (mean_potential .- ηᴵᴮ) ./ ib.gravitational_acceleration

    return nothing
end

inverse_barometer_callbacks(child, ::Nothing) = ()
inverse_barometer_callbacks(child, sea_level) =
    (Callback(update_inverse_barometer!; callsite = UpdateStateCallsite(), parameters = InverseBarometer(child, sea_level)),)
