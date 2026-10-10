# Asynchronous prefetch wrapper around `DatasetBackend`. Hides the next sliding-window's I/O behind the current window's compute
# by reading into  a buffer `FieldTimeSeries` on a long-lived worker task. Every `set!`  either copies from the prefetched buffer
# (hot) or loads synchronously (cold), then schedules the next window's read. The buffer is allocated once at FTS construction and
# reused for every reload — zero allocation per `set!`.
#
# Race invariant: between the enqueue at the end of one `set!` and the `take!` at the start of the next, the worker is mutating
# `buffer_fts.data`. No code outside `set!(::PrefetchingFTS)` may touch `buffer_fts` in that window.
# Two enforcement points: `:buffer_fts` is not forwarded by `getproperty`, and `Adapt.adapt_structure` returns
# only the inner backend.
#
# GPU ordering: the worker and the main task run on different CUDA streams, which are not ordered against each other. The worker
# synchronizes its stream before reporting a job done, and the main task synchronizes after copying out of the buffer, before the
# next job may overwrite it.
#
# One worker task per backend, started on first use and kept for the life of the process, rather than a `Threads.@spawn` per
# reload: CUDA gives every task its own stream, so a task per reload created and finalizer-destroyed a stream every few model days
# while that task's temporary device arrays were still being freed against it. The 1degree run segfaulted inside
# `cuMemAllocFromPoolAsync` on a prefetch task after ~870k steps.
#
# Requires `JULIA_NUM_THREADS ≥ 2` to actually overlap; one thread makes the worker cooperatively-scheduled and the optimisation a no-op.

using Oceananigans.OutputReaders: AbstractInMemoryBackend, FlavorOfFTS, FieldTimeSeries, time_index
using Oceananigans.Fields: location
using Oceananigans.Utils: sync_device!

import Oceananigans.OutputReaders: new_backend
import Oceananigans.Fields: set!

mutable struct PrefetchingBackend{B<:DatasetBackend, F<:FieldTimeSeries} <: AbstractInMemoryBackend{Int}
    inner_backend :: B
    pending :: Union{Channel{Any}, Nothing}   # holds `nothing` or the exception once the in-flight job finishes
    buffer_fts :: F
    next_start :: Int
    jobs :: Union{Channel{Any}, Nothing}      # queue of the worker task, `nothing` until the first prefetch
end

PrefetchingBackend(inner_backend::DatasetBackend, buffer_fts::FieldTimeSeries) = PrefetchingBackend{typeof(inner_backend), typeof(buffer_fts)}(inner_backend, nothing, buffer_fts, 0, nothing)

# Runs every prefetch of one backend, so the task (and its CUDA stream) outlives all of them.
function prefetch_worker(jobs)
    for (buffer_fts, done) in jobs
        result = try
            set!(buffer_fts)
            sync_device!(architecture(buffer_fts))
            nothing
        catch e
            # Worker-side @error logs context (the enqueue site is gone by `take!` time); the exception is handed back to `set!`.
            m = buffer_fts.backend.metadata
            start = buffer_fts.backend.start
            @error "PrefetchingBackend: prefetch task failed" dataset=typeof(m.dataset) variable=m.name window=(start, start + length(buffer_fts.backend) - 1) exception=(e, catch_backtrace())
            e
        end
        put!(done, result)
    end
end

# `:buffer_fts` deliberately warned upon — see race invariant in preamble.
function Base.getproperty(p::PrefetchingBackend, name::Symbol)
    if name in (:inner_backend, :pending, :next_start, :jobs)
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

# Diagnostic: name the reloads that block the main loop. A reload on the hot path costs the copy alone; a
# cold one pays the whole read. Off with `OMIP_PROBE=0`.
const report_slow_reloads = get(ENV, "OMIP_PROBE", "1") == "1"
const slow_reload_threshold = 0.25    # seconds

function set!(fts::PrefetchingFTS, backend::PrefetchingBackend = fts.backend)
    entry_time    = time_ns()
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
    # kill the simulation. The worker has already logged the exception
    # with full variable/window context, so we just print a short warning
    # here.
    hot = !isnothing(pending) && pending_start == needed_start
    if hot
        if take!(pending) isa Exception
            m = buffer_fts.backend.metadata
            @warn "PrefetchingBackend: pending prefetch failed; falling back to synchronous load" dataset=typeof(m.dataset) variable=m.name
            hot = false
        end
    elseif !isnothing(pending)
        # Stale prefetch targets a different window; drain it and ignore
        # any failure — we're about to reload from scratch anyway and the
        # worker has already logged the failure if there was one.
        take!(pending)
    end

    waited_time = time_ns()

    if !hot
        Nm = length(getfield(backend, :inner_backend))
        buffer_fts.backend = new_backend(buffer_fts.backend, needed_start, Nm)
        set!(buffer_fts)
    end

    loaded_time = time_ns()

    copyto!(parent(fts.data), parent(buffer_fts.data))
    # The copy is queued on this task's stream; finish it before the worker starts overwriting the buffer.
    sync_device!(architecture(fts))

    if report_slow_reloads
        elapsed = 1e-9 * (time_ns() - entry_time)
        if elapsed > slow_reload_threshold
            m = buffer_fts.backend.metadata
            @info "PROBE_FTS $(hot ? "hot" : "cold") $(m.name) start=$needed_start " *
                  "elapsed=$(round(elapsed, digits=3))s " *
                  "wait=$(round(1e-9 * (waited_time - entry_time), digits=3))s " *
                  "load=$(round(1e-9 * (loaded_time - waited_time), digits=3))s " *
                  "copy=$(round(1e-9 * (time_ns() - loaded_time), digits=3))s"
        end
    end

    # Time-indexing-aware next-window prediction: `time_index` wraps via mod1 for Cyclical and clamps to Nt for Linear/Clamp. 
    # The next reload fires the first time `n₂ = n₁ + 1` falls outside the current window, which happens when `n₁` hits the 
    # LAST in-memory index, so the new `start` is `needed_start + Nm - 1`, not `needed_start + Nm` (`update_field_time_series!` sets `start = n₁`). 
    # Passing `Nm` instead of `Nm + 1` yields that prediction and keeps every reload on the HOT path.
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

    if isnothing(getfield(backend, :jobs))
        jobs = Channel{Any}(1)
        errormonitor(Threads.@spawn prefetch_worker(jobs))
        setfield!(backend, :jobs, jobs)
    end

    done = Channel{Any}(1)
    put!(getfield(backend, :jobs), (buffer_fts, done))
    setfield!(backend, :pending, done)

    return nothing
end
