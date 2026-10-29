#####
##### Moving nests: a child that follows a feature across its parent
#####

using Oceananigans.Grids: x_domain, y_domain
using Oceananigans.Utils: TimeInterval, prettytime

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
when that shift is at least `tolerance` cells. Pass as the `mover` of [`nested_atmosphere_model`](@ref),
which attaches the child builder that [`translate_domain!`](@ref) uses. `track` records
`(time, λ₀, φ₀)` after every relocation.
"""
struct NestMover{T, S, R}
    tracker :: T
    schedule :: S
    tolerance :: Int
    rebuild :: R
    track :: Vector{NTuple{3, Float64}}
end

NestMover(tracker; schedule = TimeInterval(900), tolerance = 1) =
    NestMover(tracker, schedule, tolerance, nothing, NTuple{3, Float64}[])

NestMover(mover::NestMover, rebuild) =
    NestMover(mover.tracker, mover.schedule, mover.tolerance, rebuild, mover.track)

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

"""
$(TYPEDSIGNATURES)

Translate the child of `nested_model` by `shift = (iₛ, jₛ)` whole cells, keeping its state where
the old and new grids overlap and filling the exposed strip from the parent (method in the Breeze
extension).
"""
function translate_domain! end

relocate!(nested_model, ::Nothing) = nothing

function relocate!(nested_model, mover::NestMover)
    mover.schedule(nested_model) || return nothing
    t = nested_model.clock.time
    shift = cell_shift(nested_model.child.grid, mover.tracker(nested_model, t)...)
    maximum(abs, shift) < mover.tolerance && return nothing

    elapsed = @elapsed translate_domain!(nested_model, shift)

    λ₀, φ₀ = domain_center(nested_model.child.grid)
    push!(mover.track, (t, λ₀, φ₀))
    @info "Relocated the nest by $shift cells to ($(λ₀)°, $(φ₀)°) at t = $(prettytime(t)) in $(prettytime(elapsed))"
    return nothing
end
