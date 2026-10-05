#####
##### Asynchronous JLD2 output
#####
#
# `AsyncJLD2Writer` wraps a `JLD2Writer` and takes JLD2 compression and file writes off the time-stepping
# critical path. On the main task a write is only: the schedule checks, `fetch_output`, and a copy of each
# output into a reusable host buffer (on the GPU: converted to the writer's element type on the device, then
# copied into a pinned host buffer). A background task then runs the same `jld2output!` the synchronous writer
# uses, so the files and their contents are unchanged (Zstd compression included). File initialisation and
# splitting stay on the main task, after the background task has drained.
#
# Measured on the 1° model (H100, Zstd, output on NFS): physics 140 → 131 ms/step, BGC 205 → 186 ms/step.
#
# The background task's compression and writes are plain `ccall`s, during which a stop-the-world GC requested
# by the main task has to wait; `collect_before_write` runs a quick GC before each job is queued to make that
# less likely. `gc_safe_compression = true` instead swaps a writer's `ZstdFilter` for `GCSafeZstdFilter`
# (gc_safe_zstd.jl: same bytes, compression in a `gc_safe` ccall), so the GC need not wait for compression;
# the file writes (JLD2's MmapIO `msync`, IOStream `ios_write`) remain plain ccalls. If the `safepoint` time in
# the progress PROBE lines rises, the consumer can be moved to a separate process (the pattern of
# `jra55_data_staging.jl`).
#
# Patterns follow `jra55_data_staging.jl`: background tasks with errors surfaced on the next call, blocking
# only when a result is needed (`flush!`), and a synchronous fallback after a failure.

using Oceananigans: AbstractOutputWriter
using Oceananigans.Architectures: architecture, GPU
using Oceananigans.OutputWriters: JLD2Writer, Checkpointer, fetch_output, jld2output!, iteration_exists,
                                  start_next_file, update_file_splitting_schedule!, has_initial_output, output_time

mutable struct AsyncJLD2Writer{W, S, O} <: AbstractOutputWriter
    writer :: W                     # the wrapped JLD2Writer: owns filepath, part, file splitting, jld2_kw, array_type
    schedule :: S                   # === writer.schedule (the simulation calls `writer.schedule(model)`)
    outputs :: O                    # === writer.outputs (time averages are registered from these)
    host_buffers :: Vector{Any}     # `nbuffers` sets of host buffers, allocated on the first write
    device_staging :: Any           # device arrays for the element-type conversion (GPU only), allocated on the first write
    free :: Channel{Int}            # buffer sets not in flight
    jobs :: Channel{Any}            # (buffer set, filepath, iteration, time) for the background task
    consumer :: Union{Nothing, Task}
    error :: Base.RefValue{Any}
    written :: Set{Int}             # iterations handed to the background task
    async :: Bool                   # false: write synchronously (also the fallback after an error)
    nbuffers :: Int
    collect_before_write :: Bool
end

"""
    AsyncJLD2Writer(writer::JLD2Writer; nbuffers = 2, async = true, collect_before_write = true,
                    gc_safe_compression = false)

Wrap `writer` so that its compression and file writes run on a background task. `nbuffers` sets of host
buffers allow that many writes in flight; a write blocks only when all are. `async = false` writes
synchronously through the same buffers. `collect_before_write` runs `GC.gc(false)` before each write is
queued. `gc_safe_compression = true` replaces a `ZstdFilter` in `writer.jld2_kw[:compress]` by a
`GCSafeZstdFilter` of the same level (in place, on `writer`). The files written are the same as `writer`'s.
"""
function AsyncJLD2Writer(writer::JLD2Writer; nbuffers = 2, async = true, collect_before_write = true,
                         gc_safe_compression = false)
    gc_safe_compression && use_gc_safe_compression!(writer)
    free = Channel{Int}(nbuffers)
    foreach(b -> put!(free, b), 1:nbuffers)
    return AsyncJLD2Writer(writer, writer.schedule, writer.outputs, Any[nothing for _ in 1:nbuffers], nothing,
                           free, Channel{Any}(nbuffers), nothing, Ref{Any}(nothing), Set{Int}(),
                           async, nbuffers, collect_before_write)
end

Base.summary(aw::AsyncJLD2Writer) = string("AsyncJLD2Writer(", summary(aw.writer), ")")

gc_safe_compressor(filter::ZstdFilter) = GCSafeZstdFilter(filter) # same level (the `level::Int32` field)
gc_safe_compressor(filters::AbstractVector) = map(gc_safe_compressor, filters)
gc_safe_compressor(compress) = compress # e.g. `false`, `true` (Deflate) or another filter: unchanged

"""
    use_gc_safe_compression!(writer)

Replace a `ZstdFilter` in the `JLD2Writer` `writer`'s `jld2_kw[:compress]` (or in a vector of filters there)
by a `GCSafeZstdFilter` of the same level. Other compressors are left as they are.
"""
function use_gc_safe_compression!(writer::JLD2Writer)
    haskey(writer.jld2_kw, :compress) && (writer.jld2_kw[:compress] = gc_safe_compressor(writer.jld2_kw[:compress]))
    return writer
end

#####
##### Background task
#####

function consume!(aw::AsyncJLD2Writer)
    for (b, path, iter, t) in aw.jobs
        try
            jld2output!(path, iter, t, aw.host_buffers[b], aw.writer.jld2_kw)
        catch err
            aw.error[] = (err, catch_backtrace())
        finally
            put!(aw.free, b) # the buffer set is reusable only once JLD2 has written it
        end
    end
    return nothing
end

function ensure_consumer!(aw::AsyncJLD2Writer)
    if aw.async && (isnothing(aw.consumer) || istaskdone(aw.consumer))
        aw.consumer = Threads.@spawn consume!(aw)
    end
    return nothing
end

function check_error!(aw::AsyncJLD2Writer)
    if !isnothing(aw.error[])
        err, bt = aw.error[]
        aw.error[] = nothing
        @warn "Asynchronous write to $(aw.writer.filepath) failed; writing synchronously from now on." exception = (err, bt)
        aw.async = false
    end
    return nothing
end

"""
    flush!(aw::AsyncJLD2Writer)

Block until every queued write has been written. Called before file splits, before checkpoints and at the
end of the run.
"""
function flush!(aw::AsyncJLD2Writer)
    taken = [take!(aw.free) for _ in 1:aw.nbuffers]
    foreach(b -> put!(aw.free, b), taken)
    check_error!(aw)
    return nothing
end

#####
##### Copies to host buffers
#####

# Pin host buffers that receive device copies, once (`CUDA.pin` unpins when the buffer is garbage collected).
function pin_host_buffer!(buffer)
    try
        CUDA.pin(buffer)
    catch err
        @warn "Could not pin an output host buffer; device copies will use pageable memory." exception = err maxlog = 1
    end
    return buffer
end

host_buffer(fetched, array_type) = nothing # not an array (e.g. a number): stored as is

function host_buffer(fetched::AbstractArray, array_type)
    buffer = array_type(undef, size(fetched)...)
    architecture(fetched) isa GPU && pin_host_buffer!(buffer)
    return buffer
end

# Device array for converting a device output to the writer's element type before the copy, so the copy is
# a plain device-to-host memcpy (copying across element types goes through a host temporary).
device_staging(fetched, array_type) = nothing

function device_staging(fetched::AbstractArray, array_type)
    T = eltype(array_type)
    return architecture(fetched) isa GPU && eltype(fetched) != T ? similar(fetched, T) : nothing
end

copy_to_host!(::Nothing, staging, fetched) = fetched
copy_to_host!(buffer, ::Nothing, fetched) = copyto!(buffer, fetched)

function copy_to_host!(buffer, staging, fetched)
    staging .= fetched           # element-type conversion on the device (round to nearest, as on the host)
    copyto!(buffer, staging)     # synchronizes the stream and copies: safe before the averages are reset
    return buffer
end

function snapshot!(aw::AsyncJLD2Writer, b, model)
    w = aw.writer
    names = Tuple(keys(w.outputs))
    fetched = Tuple(fetch_output(output, model) for output in values(w.outputs))

    if isnothing(aw.host_buffers[b])
        aw.host_buffers[b] = Dict{Symbol, Any}(name => host_buffer(f, w.array_type) for (name, f) in zip(names, fetched))
    end

    if isnothing(aw.device_staging)
        aw.device_staging = Dict{Symbol, Any}(name => device_staging(f, w.array_type) for (name, f) in zip(names, fetched))
    end

    buffers = aw.host_buffers[b]
    for (name, f) in zip(names, fetched)
        result = copy_to_host!(buffers[name], aw.device_staging[name], f)
        isnothing(buffers[name]) && (buffers[name] = result) # non-array outputs are stored by value
    end

    return buffers
end

#####
##### The OutputWriter interface
#####

Oceananigans.initialize!(aw::AsyncJLD2Writer, model) = Oceananigans.initialize!(aw.writer, model)

function Oceananigans.OutputWriters.write_output!(aw::AsyncJLD2Writer, model)
    w = aw.writer
    model.clock.iteration == 0 && !has_initial_output(w.schedule) && return nothing
    check_error!(aw)

    if !w.initialized
        flush!(aw)
        Oceananigans.initialize!(w, model)
    end

    iter = model.clock.iteration

    # The background task may be appending to w.filepath and JLD2 shares one handle per path between tasks,
    # so don't open the file here: the iterations already queued are tracked in memory instead.
    if iter in aw.written
        @warn "Iteration $iter was already written to $(w.filepath). Skipping output writing."
        return nothing
    end

    # File splitting moves the file and initialises a new one: on the main task, after draining.
    split = w.file_splitting(model)
    if split
        flush!(aw)
        start_next_file(model, w)
    end
    update_file_splitting_schedule!(w.file_splitting, w.filepath)

    # e.g. after a pickup from an older checkpoint the iteration may already be in the file
    if (split || isempty(aw.written)) && isfile(w.filepath)
        flush!(aw)
        iteration_exists(w.filepath, iter) && return nothing
    end

    t = output_time(model.clock, w.schedule)
    aw.collect_before_write && GC.gc(false)

    if aw.async
        ensure_consumer!(aw)
        b = take!(aw.free) # blocks only if every buffer set is still being written
        data = snapshot!(aw, b, model)
        put!(aw.jobs, (b, w.filepath, iter, t))
    else
        b = take!(aw.free)
        try
            data = snapshot!(aw, b, model)
            jld2output!(w.filepath, iter, t, data, w.jld2_kw)
        finally
            put!(aw.free, b)
        end
    end

    push!(aw.written, iter)
    return nothing
end

Oceananigans.OutputWriters.write_output!(aw::AsyncJLD2Writer, sim::Simulation) =
    Oceananigans.OutputWriters.write_output!(aw, sim.model)

# Checkpointing: drain first, then checkpoint the wrapped writer (never the buffers or the task)
Oceananigans.prognostic_state(aw::AsyncJLD2Writer) = (flush!(aw); Oceananigans.prognostic_state(aw.writer))
Oceananigans.restore_prognostic_state!(aw::AsyncJLD2Writer, from) = Oceananigans.restore_prognostic_state!(aw.writer, from)
Oceananigans.restore_prognostic_state!(aw::AsyncJLD2Writer, ::Nothing) = nothing
Oceananigans.OutputWriters.reconcile_restored_output_schedule!(aw::AsyncJLD2Writer, model) =
    Oceananigans.OutputWriters.reconcile_restored_output_schedule!(aw.writer, model)

Base.close(aw::AsyncJLD2Writer) = (flush!(aw); close(aw.jobs); nothing)

#####
##### Simulation-level helpers
#####

"""
    flush_output!(simulation)

Block until every asynchronous output writer of `simulation` has written everything queued.
`run!` does this at its end through the `:flush_async_output` callback; call it also in a `finally`
block around `run!` so that an exception does not lose queued writes.
"""
function flush_output!(simulation)
    for writer in values(simulation.output_writers)
        writer isa AsyncJLD2Writer && flush!(writer)
    end
    return nothing
end

struct FlushAsyncOutput end
(::FlushAsyncOutput)(simulation) = flush_output!(simulation)
Oceananigans.Simulations.finalize!(f::FlushAsyncOutput, simulation) = f(simulation)
Oceananigans.prognostic_state(::FlushAsyncOutput) = nothing # nothing to checkpoint

"""
    use_async_output!(simulation; kwargs...)

Wrap every `JLD2Writer` in `simulation.output_writers` in an `AsyncJLD2Writer(writer; kwargs...)` (e.g.
`gc_safe_compression = true`), add a
callback that flushes them at the end of `run!`, and move the checkpointer(s) after all other writers so that
at an iteration where both write, the outputs are queued (and then drained by the checkpoint) first.
"""
function use_async_output!(simulation; kwargs...)
    writers = simulation.output_writers

    for name in collect(keys(writers))
        writers[name] isa JLD2Writer && (writers[name] = AsyncJLD2Writer(writers[name]; kwargs...))
    end

    for name in [name for (name, writer) in writers if writer isa Checkpointer]
        checkpointer = pop!(writers, name)
        writers[name] = checkpointer
    end

    simulation.callbacks[:flush_async_output] = Callback(FlushAsyncOutput(), IterationInterval(typemax(Int)))

    return simulation
end
