#####
##### The inverse-barometer sea level of a nested ocean's Flather exterior
#####

using Oceananigans: UpdateStateCallsite
using Oceananigans.TimeSteppers: reconcile_state!
using Oceananigans.Operators: Azᶜᶜᶜ
using Oceananigans.Simulations: Callback

using ..Oceans: forcing_barotropic_potential

# The Flather exterior sea level of one open side: the parent's plus the child's inverse-barometer sea level
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

@inline function BoundaryConditions.getbc(η::InverseBarometerSeaLevel{Dim}, s::Integer, k::Integer, grid::AbstractGrid,
                                          clock=nothing, args...) where Dim
    b = η.sea_level
    i, j = horizontal_indices(Val(Dim), b.index, clamp(s, b.along...))
    ηᴵᴮ = @inbounds η.inverse_barometer[i, j, 1]
    return getbc(b, s, k, grid, clock, args...) + ηᴵᴮ
end

with_inverse_barometer(sea_level, ::Nothing) = sea_level
with_inverse_barometer(sea_level::BoundaryTimeSeries{Dim}, inverse_barometer::Field) where Dim =
    InverseBarometerSeaLevel{Dim}(sea_level, inverse_barometer)

# ηᴵᴮ = -(pᵃ / ρ₀ - ⟨pᵃ / ρ₀⟩) / g, with ⟨⟩ the mean over the child's wet surface, so that ηᴵᴮ moves no volume
struct InverseBarometer{S, P, A, FT}
    sea_level                  :: S
    potential                  :: P
    wet_surface_area           :: A
    initial_sea_level          :: S
    total_wet_surface_area     :: FT
    gravitational_acceleration :: FT
end

@kernel function _compute_wet_surface_area!(area, grid, Nz)
    i, j = @index(Global, NTuple)
    @inbounds area[i, j, 1] = ifelse(inactive_cell(i, j, Nz, grid), zero(grid), Azᶜᶜᶜ(i, j, Nz, grid))
end

function InverseBarometer(child, sea_level)
    grid = child.grid
    potential = forcing_barotropic_potential(child)

    isnothing(potential) &&
        throw(ArgumentError("inverse_barometer = true requires a child ocean forced by the atmospheric pressure."))

    area = Field{Center, Center, Nothing}(grid)
    launch!(architecture(grid), grid, :xy, _compute_wet_surface_area!, area, grid, size(grid, 3))
    total_area = all_reduce(+, sum(interior(area)), architecture(grid))
    g = convert(eltype(grid), child.free_surface.gravitational_acceleration)

    initial_sea_level = Field{Center, Center, Nothing}(grid)

    return InverseBarometer(sea_level, potential, area, initial_sea_level, convert(eltype(grid), total_area), g)
end

function update_inverse_barometer!(model, ib::InverseBarometer)
    ηᴵᴮ = parent(ib.sea_level)
    ηᴵᴮ .= parent(ib.potential)

    Σ = mapreduce(*, +, interior(ib.sea_level), interior(ib.wet_surface_area))
    mean_potential = all_reduce(+, Σ, architecture(model.grid)) / ib.total_wet_surface_area

    ηᴵᴮ .= (mean_potential .- ηᴵᴮ) ./ ib.gravitational_acceleration

    model.clock.iteration == 0 && add_initial_inverse_barometer!(model, ib)

    return nothing
end

# Before the first time step, the child's free surface takes ηᴵᴮ too, so it starts in balance with the pressure
function add_initial_inverse_barometer!(model, ib)
    ηᴵᴮ = interior(ib.sea_level)
    η₀ᴵᴮ = interior(ib.initial_sea_level)

    change = mapreduce((a, b) -> abs(a - b), max, ηᴵᴮ, η₀ᴵᴮ)
    all_reduce(max, change, architecture(model.grid)) > 0 || return nothing

    η = interior(model.free_surface.displacement)
    η .+= (ηᴵᴮ .- η₀ᴵᴮ) .* (interior(ib.wet_surface_area) .> 0)
    η₀ᴵᴮ .= ηᴵᴮ

    reconcile_state!(model)
    update_state!(model)

    return nothing
end

inverse_barometer_callbacks(child, ::Nothing) = ()
inverse_barometer_callbacks(child, sea_level) =
    (Callback(update_inverse_barometer!; callsite = UpdateStateCallsite(), parameters = InverseBarometer(child, sea_level)),)
