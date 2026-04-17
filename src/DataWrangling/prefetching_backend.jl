using Oceananigans.OutputReaders: AbstractInMemoryBackend, FlavorOfFTS, FieldTimeSeries
using Oceananigans.Fields: location, instantiated_location

import Oceananigans.OutputReaders: new_backend
import Oceananigans.Fields: set!

"""
    PrefetchingBackend{B<:DatasetBackend} <: AbstractInMemoryBackend{Int}

Wrapper around a `DatasetBackend` that hides the next window's I/O behind
the current window's compute by reading into a *buffer* `FieldTimeSeries`
on a background `Task`. The next call to `set!` either copies the
already-loaded buffer into the main FTS (hot path) or falls back to a
synchronous read (cold path), then schedules the prefetch for the window
after that.

Fields
------

- `inner`: the wrapped `DatasetBackend`. Carries the `start`/`length`
  window, metadata, inpainting setting — the cold-path read and the
  prefetched read both dispatch through it.
- `task`: the in-flight prefetch `Task`, or `nothing`.
- `buffer_fts`: a clone of the main `FieldTimeSeries` whose `data` array
  is the destination of the prefetch read. Constructed before the task is
  spawned so that the spawn does not race on FTS construction.
- `next_start`: the absolute time index at which `buffer_fts` will start
  once its prefetch task completes — compared against the requested
  `start` on the next `set!` call to decide hot vs. cold path.
"""
mutable struct PrefetchingBackend{B<:DatasetBackend} <: AbstractInMemoryBackend{Int}
    inner :: B
    task :: Union{Task, Nothing}
    buffer_fts :: Any   # erased: a `FieldTimeSeries` whose concrete type
                        # depends on grid/loc/inpainting and is only set
                        # at the first `set!`. Erasing it keeps the
                        # backend's type stable across reloads so that
                        # Oceananigans' `new_backend` round-trip type-checks.
    next_start :: Int
end

PrefetchingBackend(inner::DatasetBackend) = PrefetchingBackend{typeof(inner)}(inner, nothing, nothing, 0)

# Forward properties so `fts.backend.start`, `fts.backend.metadata`, etc.
# continue to work for downstream code that pokes at the inner backend.
function Base.getproperty(p::PrefetchingBackend, name::Symbol)
    if name in (:inner, :task, :buffer_fts, :next_start)
        return getfield(p, name)
    else
        return getproperty(getfield(p, :inner), name)
    end
end

Base.length(p::PrefetchingBackend) = length(p.inner)

Base.summary(p::PrefetchingBackend) =
    string("PrefetchingBackend(", p.inner.start, ", ", p.inner.length,
           "; pending_prefetch=", !isnothing(getfield(p, :task)), ")")

# When Oceananigans rolls the in-memory window forward it calls
# `new_backend(b, start, length)`. Forward the call to the inner backend
# and preserve the prefetch state — the start passed in here is the new
# window's start, which `set!(fts::PrefetchingFTS)` will use to decide if
# the pending prefetch already covers it.
new_backend(p::PrefetchingBackend, start, length) =
    PrefetchingBackend(new_backend(p.inner, start, length),
                       getfield(p, :task),
                       getfield(p, :buffer_fts),
                       getfield(p, :next_start))

# Dropping prefetch state on adapt — `Task` and the buffer FTS aren't
# meaningful on a different architecture, and they can't be cleanly
# serialised across a CPU↔GPU boundary.
Adapt.adapt_structure(to, p::PrefetchingBackend) = Adapt.adapt(to, p.inner)

const PrefetchingFTS = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:PrefetchingBackend}

# Build a clone FTS whose `.backend` is the supplied (inner) DatasetBackend
# and whose `.data` is freshly allocated. The clone shares grid, times,
# location, time_indexing and boundary conditions with `fts`, so writing
# its interior is layout-compatible with the main FTS.
function buffer_field_time_series(fts, inner_backend)
    LX, LY, LZ = location(fts)
    return FieldTimeSeries{LX, LY, LZ}(fts.grid, fts.times;
                                       backend             = inner_backend,
                                       time_indexing       = fts.time_indexing,
                                       boundary_conditions = fts.boundary_conditions)
end

function set!(fts::PrefetchingFTS, backend=fts.backend)
    needed_start = getfield(backend, :inner).start

    pending_task   = getfield(backend, :task)
    pending_fts    = getfield(backend, :buffer_fts)
    pending_start  = getfield(backend, :next_start)

    if !isnothing(pending_task) && pending_start == needed_start && !isnothing(pending_fts)
        # Hot path — wait on the prefetch (likely already done) and copy
        # the buffer FTS into the main FTS.
        wait(pending_task)
        copyto!(parent(fts.data), parent(pending_fts.data))
    else
        # Cold path — drain any stale prefetch, then synchronously load
        # via a one-off buffer FTS. We can't dispatch the JRA55-specific
        # `set!` directly on `fts` because `fts.backend` is the
        # PrefetchingBackend, not the inner one; so we round-trip through
        # a clone FTS whose `.backend` is the inner.
        if !isnothing(pending_task)
            wait(pending_task)
        end

        cold_fts = buffer_field_time_series(fts, getfield(backend, :inner))
        set!(cold_fts)
        copyto!(parent(fts.data), parent(cold_fts.data))
    end

    # Kick off the next prefetch (cyclically wrapping at the end of times).
    Nm = length(getfield(backend, :inner))
    Nt = length(fts.times)
    next_start = mod1(needed_start + Nm, Nt)
    next_inner = new_backend(getfield(backend, :inner), next_start, Nm)
    next_fts   = buffer_field_time_series(fts, next_inner)

    setfield!(backend, :next_start, next_start)
    setfield!(backend, :buffer_fts, next_fts)
    setfield!(backend, :task, Threads.@spawn set!(next_fts))

    return nothing
end
