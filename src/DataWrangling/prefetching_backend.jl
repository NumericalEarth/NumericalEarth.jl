# Asynchronous prefetch wrapper around `DatasetBackend`. Hides the next sliding-window's I/O behind the current window's compute
# by reading into  a buffer `FieldTimeSeries` on a `Threads.@spawn`-ed task. Every `set!`  either copies from the prefetched buffer 
# (hot) or loads synchronously (cold), then schedules the next window's read. The buffer is allocated once at FTS construction and 
# reused for every reload — zero allocation per `set!`.
#
# Race invariant: between the spawn at the end of one `set!` and the `wait` at the start of the next, the worker is mutating
# `buffer_fts.data`. No code outside `set!(::PrefetchingFTS)` may touch `buffer_fts` in that window. 
# Two enforcement points: `:buffer_fts` is not forwarded by `getproperty`, and `Adapt.adapt_structure` returns
# only the inner backend.
#
# Requires `JULIA_NUM_THREADS ≥ 2` to actually overlap; one thread makes the spawn cooperatively-scheduled and the optimisation a no-op.

using Oceananigans.OutputReaders: AbstractInMemoryBackend, FlavorOfFTS, FieldTimeSeries, time_index
using Oceananigans.Fields: location

import Oceananigans.OutputReaders: new_backend
import Oceananigans.Fields: set!

mutable struct PrefetchingBackend{B<:DatasetBackend, F<:FieldTimeSeries} <: AbstractInMemoryBackend{Int}
    inner_backend :: B
    pending :: Union{Task, Nothing}
    buffer_fts :: F
    next_start :: Int
end

PrefetchingBackend(inner_backend::DatasetBackend, buffer_fts::FieldTimeSeries) = PrefetchingBackend{typeof(inner_backend), typeof(buffer_fts)}(inner_backend, nothing, buffer_fts, 0)

# `:buffer_fts` deliberately warned upon — see race invariant in preamble.
function Base.getproperty(p::PrefetchingBackend, name::Symbol)
    if name in (:inner_backend, :pending, :next_start)
        return getfield(p, name)
    elseif name == :buffer_fts
        @warn "`buffer_fts` is an inner auxiliary field touched in a hot loop by a separate task. " *
              "Mutating it manually might lead to undefined behavior. It is recommended not to modify it."
        return getfield(p, name)
    else
        return getproperty(getfield(p, :inner_backend), name)
    end
end

Base.length(p::PrefetchingBackend) = length(p.inner_backend)
Base.summary(p::PrefetchingBackend) = string("PrefetchingBackend(", p.inner_backend.start, ", ", p.inner_backend.length, "; pending=", !isnothing(getfield(p, :pending)), ")")

# Mutate in place rather than constructing a fresh wrapper — keeps the
# `pending`/`buffer_fts`/`next_start` mutable state in exactly one object.
function new_backend(p::PrefetchingBackend, start, length)
    setfield!(p, :inner_backend, new_backend(getfield(p, :inner_backend), start, length))
    return p
end

Adapt.adapt_structure(to, p::PrefetchingBackend) = Adapt.adapt(to, getfield(p, :inner_backend))

const PrefetchingFTS = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:PrefetchingBackend}

function set!(fts::PrefetchingFTS, backend::PrefetchingBackend = fts.backend)
    needed_start  = getfield(backend, :inner_backend).start
    pending       = getfield(backend, :pending)
    pending_start = getfield(backend, :next_start)
    buffer_fts    = getfield(backend, :buffer_fts)

    # Cleared up-front so a failed prefetch isn't re-thrown on every later set!.
    setfield!(backend, :pending, nothing)

    # Hot path: the pending prefetch already targets `needed_start`. Wait on
    # it (typically a no-op — the background load finished while compute was
    # running). A failed prefetch demotes to the synchronous load below, so a
    # transient I/O error (a brief FS hiccup, a staging race, etc.) doesn't
    # kill the simulation. The spawn site has already logged the exception
    # with full variable/window context, so we just print a short warning
    # here.
    hot = !isnothing(pending) && pending_start == needed_start
    if hot
        try
            wait(pending)
        catch
            m = buffer_fts.backend.metadata
            @warn "PrefetchingBackend: pending prefetch failed; falling back to synchronous load" dataset=typeof(m.dataset) variable=m.name
            hot = false
        end
    elseif !isnothing(pending)
        # Stale prefetch targets a different window; drain it and swallow
        # any failure — we're about to reload from scratch anyway and the
        # spawn site has already logged the failure if there was one.
        try
            wait(pending)
        catch
        end
    end

    if !hot
        Nm = length(getfield(backend, :inner_backend))
        buffer_fts.backend = new_backend(buffer_fts.backend, needed_start, Nm)
        set!(buffer_fts)
    end

    copyto!(parent(fts.data), parent(buffer_fts.data))

    # Time-indexing-aware next-window prediction: `time_index` wraps
    # via mod1 for Cyclical and clamps to Nt for Linear/Clamp. The next
    # reload fires the first time `n₂ = n₁ + 1` falls outside the current
    # window, which happens when `n₁` hits the LAST in-memory index, so
    # the new `start` is `needed_start + Nm - 1`, not `needed_start + Nm`
    # (`update_field_time_series!` sets `start = n₁`). Passing `Nm`
    # instead of `Nm + 1` yields that prediction and keeps every reload
    # on the HOT path.
    Nm = length(getfield(backend, :inner_backend))
    Nt = length(fts.times)
    new_next = time_index(buffer_fts.backend, fts.time_indexing, Nt, Nm)

    # Linear/Clamp at end-of-data: window can't advance, no prefetch.
    if new_next == needed_start
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
