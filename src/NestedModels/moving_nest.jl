#####
##### Moving nests: a child that follows a feature across its parent
#####

using Oceananigans.Fields: interior
using Oceananigans.Grids: x_domain, y_domain
using Oceananigans.Utils: TimeInterval, prettytime

using ..EarthSystemModels: EarthSystemModel, relocate!, surface_temperature
using ..EarthSystemModels.InterfaceComputations: initialize_land_interface!

"""
    PrescribedTrack(times, longitude, latitude)

A nest-center track: the position `(λ, φ)` at time `t` interpolates linearly between the
waypoints `(longitude[n], latitude[n])` at `times[n]` (seconds of simulation time) and holds the
end points outside `times`. Callable as `track(nested_model, t)`, the tracker interface of
[`NestMover`](@ref).
"""
struct PrescribedTrack{T, L, P}
    times :: T
    longitude :: L
    latitude :: P
end

function (track::PrescribedTrack)(nested_model, t)
    times = track.times
    n = clamp(searchsortedlast(times, t), 1, length(times) - 1)
    w = clamp((t - times[n]) / (times[n+1] - times[n]), 0, 1)
    λ = (1 - w) * track.longitude[n] + w * track.longitude[n+1]
    φ = (1 - w) * track.latitude[n]  + w * track.latitude[n+1]
    return λ, φ
end

"""
    NestMover(tracker; schedule = TimeInterval(900), tolerance = 1)

Relocate a nested child to follow `tracker`, a callable `(nested_model, t) -> (λ, φ)` giving the
nest center (for example a [`PrescribedTrack`](@ref)). On every actuation of `schedule`, the child
is translated by the whole number of cells that brings its center closest to the tracked position,
when that shift is at least `tolerance` cells. Pass as the `mover` of [`nested_atmosphere_model`](@ref);
an `EarthSystemModel` built on the nest takes the mover over and relocates the surfaces with the child
(see [`translate_domain!`](@ref)). `track` records `(time, λ₀, φ₀)` after every relocation.
"""
struct NestMover{T, S}
    tracker :: T
    schedule :: S
    tolerance :: Int
    track :: Vector{NTuple{3, Float64}}
end

NestMover(tracker; schedule = TimeInterval(900), tolerance = 1) =
    NestMover(tracker, schedule, tolerance, NTuple{3, Float64}[])

Base.summary(mover::NestMover) = string("NestMover following ", summary(mover.tracker), " on ", summary(mover.schedule))

domain_center(grid) = (sum(x_domain(grid)) / 2, sum(y_domain(grid)) / 2)

# The whole-cell shift that brings the center of `grid` closest to `(λ, φ)`.
function cell_shift(grid, λ, φ)
    λ₀, φ₀ = domain_center(grid)
    Nx, Ny, _ = size(grid)
    Δλ = -(reverse(x_domain(grid))...) / Nx
    Δφ = -(reverse(y_domain(grid))...) / Ny
    return (round(Int, (λ - λ₀) / Δλ), round(Int, (φ - φ₀) / Δφ))
end

# new[i, j, k] ← old[i + iₛ, j + jₛ, k] over the cells both grids hold.
function copy_overlap!(new, old, shift)
    iₛ, jₛ = shift
    Nx, Ny, _ = size(interior(new))
    i = max(1, 1 - iₛ):min(Nx, Nx - iₛ)
    j = max(1, 1 - jₛ):min(Ny, Ny - jₛ)
    view(interior(new), i, j, :) .= view(interior(old), i .+ iₛ, j .+ jₛ, :)
    return new
end

"""
$(TYPEDSIGNATURES)

A `NestedModel` whose child is that of `nested_model` rebuilt by `nested_model.rebuild` on the grid
shifted by `shift = (iₛ, jₛ)` whole cells, keeping the state where the old and new grids overlap
and filling the exposed strip from the parent (method in the Breeze extension).
"""
function translated_nest end

"""
$(TYPEDSIGNATURES)

Translate the child of `nested_model` by `shift = (iₛ, jₛ)` whole cells, keeping its state where
the old and new grids overlap and filling the exposed strip from the parent.
"""
function translate_domain!(nested_model::NestedModel, shift)
    translated = translated_nest(nested_model, shift)
    nested_model.child = translated.child
    nested_model.child_callbacks = translated.child_callbacks
    return nested_model
end

"""
$(TYPEDSIGNATURES)

Translate the nested atmosphere of `coupled_model` by `shift` whole cells and rebuild the surface
components on the shifted grid with `coupled_model.surfaces`. The land prognostics and the land
interface temperature carry over where the grids overlap; on the exposed strip they are what the
builder initializes them to. The clock is kept.
"""
function translate_domain!(coupled_model::EarthSystemModel, shift)
    simulation = coupled_model.atmosphere
    nest = translated_nest(component_model(simulation), shift)
    atmosphere = Simulation(nest; Δt = simulation.Δt, verbose = simulation.verbose)
    rebuilt = EarthSystemModel(; atmosphere, clock = coupled_model.clock, coupled_model.surfaces(nest.child.grid)...)

    foreach(prognostic_fields(rebuilt.land), prognostic_fields(coupled_model.land)) do new, old
        copy_overlap!(new, old, shift)
    end
    initialize_land_interface!(rebuilt)
    copy_overlap!(surface_temperature(rebuilt.interfaces.atmosphere_land_interface),
                  surface_temperature(coupled_model.interfaces.atmosphere_land_interface), shift)

    coupled_model.atmosphere = rebuilt.atmosphere
    coupled_model.radiation  = rebuilt.radiation
    coupled_model.land       = rebuilt.land
    coupled_model.sea_ice    = rebuilt.sea_ice
    coupled_model.ocean      = rebuilt.ocean
    coupled_model.interfaces = rebuilt.interfaces
    update_state!(coupled_model)
    return coupled_model
end

nested_model(nested_model::NestedModel) = nested_model
nested_model(coupled_model::EarthSystemModel) = component_model(coupled_model.atmosphere)

function EarthSystemModels.detach_mover(simulation::Simulation{<:NestedModel{<:Any, <:Any, <:Any, <:Any, <:NestMover}})
    nest = simulation.model
    fixed = NestedModel(nest.parent, nest.child, nest.exchanger, nest.child_callbacks, nothing, nest.rebuild)
    return Simulation(fixed; Δt = simulation.Δt, verbose = simulation.verbose), nest.mover
end

function EarthSystemModels.relocate!(model, mover::NestMover)
    mover.schedule(model) || return nothing
    t = model.clock.time
    nest = nested_model(model)
    shift = cell_shift(nest.child.grid, mover.tracker(nest, t)...)
    maximum(abs, shift) < mover.tolerance && return nothing

    elapsed = @elapsed translate_domain!(model, shift)

    λ₀, φ₀ = domain_center(nested_model(model).child.grid)
    push!(mover.track, (t, λ₀, φ₀))
    @info "Relocated the nest by $shift cells to ($(λ₀)°, $(φ₀)°) at t = $(prettytime(t)) in $(prettytime(elapsed))"
    return nothing
end
