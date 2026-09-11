mutable struct LowPassFilter{FT} <: AbstractSchedule
    interval :: FT
    window :: FT
    cutoff :: FT
    next_frame :: Int # frame k is centered at k * interval; 0 before the first time step
end

"""
$(TYPEDSIGNATURES)

Return a schedule for `JLD2Writer` that writes its outputs low-pass filtered in time, one frame
every `interval`. Each frame averages an output over a `window` centered on the frame time with
the weights of a Lanczos filter [Duchon (1979)](@cite duchon1979lanczos),

    w(τ) = sinc(2τ / cutoff) sinc(2τ / window),    |τ| ≤ window / 2,

which pass periods longer than `cutoff` and remove shorter ones. The defaults remove the diurnal
and semidiurnal tides.

Frames fall on multiples of `interval` and are stamped with their center time; each is written
`window / 2` after it. Frames whose window starts before the first time step are not written, so
a simulation picked up from a checkpoint writes its first frame `window / 2` after the checkpoint.

```jldoctest
using NumericalEarth
using Oceananigans.Units

LowPassFilter(1day)

# output
LowPassFilter(interval=1 day, window=5 days, cutoff=1.667 days)
```
"""
function LowPassFilter(interval; window = 5days, cutoff = 40hours)
    interval, window, cutoff = promote(interval, window, cutoff)
    return LowPassFilter(interval, window, cutoff, 0)
end

Base.summary(filter::LowPassFilter) = string("LowPassFilter(interval=", prettytime(filter.interval),
                                             ", window=", prettytime(filter.window),
                                             ", cutoff=", prettytime(filter.cutoff), ")")

Base.show(io::IO, filter::LowPassFilter) = print(io, summary(filter))

(filter::LowPassFilter)(model) = filter.next_frame > 0 &&
                                 model.clock.time >= filter.next_frame * filter.interval + filter.window / 2

Oceananigans.prognostic_state(::LowPassFilter) = nothing
Oceananigans.restore_prognostic_state!(::LowPassFilter, ::Nothing) = nothing

mutable struct LowPassFilteredOutput{O, A, FT} <: AbstractDiagnostic
    operand :: O
    filter :: LowPassFilter{FT}
    schedule :: IterationInterval
    sums :: Vector{A}     # weighted sum of the operand, one per frame in progress
    weights :: Vector{FT} # sum of the weights in each
    frames :: Vector{Int} # frame each sum belongs to
    previous_time :: FT
end

function LowPassFilteredOutput(operand, filter, model)
    output = fetch_output(operand, model)
    frames_in_progress = floor(Int, filter.window / filter.interval) + 1
    sums = [zero(output) for _ in 1:frames_in_progress]
    FT = typeof(filter.interval)
    return LowPassFilteredOutput(operand, filter, IterationInterval(1), sums,
                                 zeros(FT, frames_in_progress), zeros(Int, frames_in_progress),
                                 convert(FT, model.clock.time))
end

function Oceananigans.run_diagnostic!(output::LowPassFilteredOutput, model)
    filter = output.filter
    t = model.clock.time
    Δt = t - output.previous_time
    output.previous_time = t

    # Frames whose window starts before the first time step seen are never complete.
    if filter.next_frame == 0
        filter.next_frame = ceil(Int, (t + filter.window / 2) / filter.interval)
    end

    φ = fetch_output(output.operand, model)
    first_frame = max(filter.next_frame, ceil(Int, (t - filter.window / 2) / filter.interval))
    last_frame = floor(Int, (t + filter.window / 2) / filter.interval)

    for frame in first_frame:last_frame
        n = mod(frame, length(output.sums)) + 1

        if output.frames[n] != frame
            output.frames[n] = frame
            output.sums[n] .= 0
            output.weights[n] = 0
        end

        τ = t - frame * filter.interval
        w = sinc(2τ / filter.cutoff) * sinc(2τ / filter.window) * Δt
        output.sums[n] .+= w .* φ
        output.weights[n] += w
    end

    return nothing
end

function (output::LowPassFilteredOutput)(model)
    n = mod(output.filter.next_frame, length(output.sums)) + 1
    return output.sums[n] ./ output.weights[n]
end

Oceananigans.Grids.grid(output::LowPassFilteredOutput) = Oceananigans.Grids.grid(output.operand)
Oceananigans.Fields.location(output::LowPassFilteredOutput) = Oceananigans.Fields.location(output.operand)
Oceananigans.Fields.indices(output::LowPassFilteredOutput) = Oceananigans.Fields.indices(output.operand)

function Oceananigans.OutputWriters.time_average_outputs(filter::LowPassFilter, outputs::NamedTuple, model)
    filtered_outputs = NamedTuple(name => LowPassFilteredOutput(outputs[name], filter, model) for name in keys(outputs))
    return filter, filtered_outputs
end

function Oceananigans.Simulations.add_dependency!(diagnostics, output::LowPassFilteredOutput)
    output ∈ values(diagnostics) || (diagnostics[Symbol(:LowPassFilteredOutput, length(diagnostics) + 1)] = output)
    return nothing
end

function Oceananigans.write_output!(writer::JLD2Writer{<:Any, <:LowPassFilter}, model::AbstractModel)
    filter = writer.schedule
    filter(model) || return nothing # only frames whose window is complete
    @invoke Oceananigans.write_output!(writer::JLD2Writer, model::Any)

    jldopen(writer.filepath, "r+"; writer.jld2_kw...) do file
        address = "timeseries/t/$(model.clock.iteration)"
        delete!(file, address)
        file[address] = filter.next_frame * filter.interval
    end

    filter.next_frame += 1
    return nothing
end
