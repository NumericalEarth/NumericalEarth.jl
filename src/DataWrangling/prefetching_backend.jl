# Asynchronous prefetch wrapper around `DatasetBackend`. Hides the next
# sliding-window's I/O behind the current window's compute by reading into
# a buffer `FieldTimeSeries` on a `Threads.@spawn`-ed task. Every `set!`
# either copies from the prefetched buffer (hot) or loads synchronously
# (cold), then schedules the next window's read. The buffer is allocated
# once at FTS construction and reused for every reload — zero allocation
# per `set!`.
#
# Race invariant: between the spawn at the end of one `set!` and the
# `wait` at the start of the next, the worker is mutating
# `buffer_fts.data`. No code outside `set!(::PrefetchingFTS)` may touch
# `buffer_fts` in that window. Two enforcement points: `:buffer_fts` is
# not forwarded by `getproperty`, and `Adapt.adapt_structure` returns
# only the inner backend. `wait_for_prefetch!` is the safe drain hook.
#
# Requires `JULIA_NUM_THREADS ≥ 2` to actually overlap; one thread makes
# the spawn cooperatively-scheduled and the optimisation a no-op.

using Oceananigans.OutputReaders: AbstractInMemoryBackend, FlavorOfFTS, FieldTimeSeries, time_index
using Oceananigans.Fields: location

import Oceananigans.OutputReaders: new_backend
import Oceananigans.Fields: set!

mutable struct PrefetchingBackend{B<:DatasetBackend, F<:FieldTimeSeries} <: AbstractInMemoryBackend{Int}
    inner :: B
    pending :: Union{Task, Nothing}
    buffer_fts :: F
    next_start :: Int
end

PrefetchingBackend(inner::DatasetBackend, buffer_fts::FieldTimeSeries) =
    PrefetchingBackend{typeof(inner), typeof(buffer_fts)}(inner, nothing, buffer_fts, 0)

# `:buffer_fts` deliberately omitted — see race invariant in preamble.
function Base.getproperty(p::PrefetchingBackend, name::Symbol)
    if name in (:inner, :pending, :next_start)
        return getfield(p, name)
    else
        return getproperty(getfield(p, :inner), name)
    end
end

Base.length(p::PrefetchingBackend) = length(p.inner)

Base.summary(p::PrefetchingBackend) =
    string("PrefetchingBackend(", p.inner.start, ", ", p.inner.length,
           "; pending=", !isnothing(getfield(p, :pending)), ")")

# Mutate in place rather than constructing a fresh wrapper — keeps the
# `pending`/`buffer_fts`/`next_start` mutable state in exactly one object.
function new_backend(p::PrefetchingBackend, start, length)
    setfield!(p, :inner, new_backend(getfield(p, :inner), start, length))
    return p
end

Adapt.adapt_structure(to, p::PrefetchingBackend) = Adapt.adapt(to, getfield(p, :inner))

const PrefetchingFTS = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:PrefetchingBackend}

"""
    wait_for_prefetch!(backend::PrefetchingBackend)

Block until the in-flight prefetch task completes, then clear it.
Required before any code that needs a consistent view of the buffer FTS
(checkpointing, JLD2 serialisation, manual `getfield(..., :buffer_fts)`).
"""
function wait_for_prefetch!(p::PrefetchingBackend)
    pending = getfield(p, :pending)
    if !isnothing(pending)
        wait(pending)
        setfield!(p, :pending, nothing)
    end
    return nothing
end

function set!(fts::PrefetchingFTS, backend::PrefetchingBackend = fts.backend)
    needed_start  = getfield(backend, :inner).start
    pending       = getfield(backend, :pending)
    pending_start = getfield(backend, :next_start)
    buffer_fts    = getfield(backend, :buffer_fts)

    # Cleared up-front so a failed prefetch isn't re-thrown on every later set!.
    setfield!(backend, :pending, nothing)

    if !isnothing(pending) && pending_start == needed_start
        wait(pending)
    else
        !isnothing(pending) && wait(pending)
        Nm = length(getfield(backend, :inner))
        buffer_fts.backend = new_backend(buffer_fts.backend, needed_start, Nm)
        set!(buffer_fts)
    end

    copyto!(parent(fts.data), parent(buffer_fts.data))

    # Time-indexing-aware next-window prediction: `time_index` wraps
    # via mod1 for Cyclical and clamps to Nt for Linear/Clamp.
    Nm = length(getfield(backend, :inner))
    Nt = length(fts.times)
    new_next = time_index(buffer_fts.backend, fts.time_indexing, Nt, Nm + 1)

    if new_next == needed_start
        # Linear/Clamp at end-of-data: window can't advance, no prefetch.
        setfield!(backend, :next_start, 0)
        return nothing
    end

    buffer_fts.backend = new_backend(buffer_fts.backend, new_next, Nm)
    setfield!(backend, :next_start, new_next)
    # Worker-side @error logs context (spawn site is gone by `wait` time); rethrow preserves the original.
    setfield!(backend, :pending, Threads.@spawn begin
        try
            set!(buffer_fts)
        catch e
            m = buffer_fts.backend.metadata
            @error "PrefetchingBackend: prefetch task failed" dataset=typeof(m.dataset) variable=m.name window=(new_next, new_next + Nm - 1) exception=(e, catch_backtrace())
            rethrow()
        end
    end)

    return nothing
end
