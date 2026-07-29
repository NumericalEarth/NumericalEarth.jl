using Dates: Dates, DateTime, Millisecond

#####
##### Averaging a series onto a coarser cadence
#####
##### TODO: a lazily evaluated, reduced `FieldTimeSeries` is related to upstream
##### `FieldTimeSeriesOperation` work in Oceananigans (CliMA/Oceananigans.jl#5761).
##### Replace this with that when it lands.
#####

const MILLISECONDS_PER_DAY = Dates.value(Millisecond(Dates.Day(1)))

# Days, rather than the exact millisecond count, so the weights stay small enough to
# accumulate in the series' own eltype and the kernel needs no Float64.
@inline function window_overlap(sample_start, sample_stop, window_start, window_stop)
    overlap = min(sample_stop, window_stop) - max(sample_start, window_start)
    return max(0, Dates.value(Millisecond(overlap)) / MILLISECONDS_PER_DAY)
end

function validate_average_bounds(bounds, Nt)
    length(bounds) == Nt + 1 ||
        throw(ArgumentError("time_average needs one more bound than the series has times " *
                            "($(Nt + 1)); got $(length(bounds)). The extra bound closes the " *
                            "last sample's window."))
    return nothing
end

function average_window_edges(bounds, window)
    first_edge, last_edge = DateTime(first(bounds)), DateTime(last(bounds))
    edges = [first_edge]
    while last(edges) < last_edge
        push!(edges, min(last(edges) + window, last_edge))
    end
    return edges
end

# Every cell shares the same weights, so the dates are resolved once on the host and the
# kernel reads scalars out of an `(Nt, Nw)` matrix.
function overlap_weights(FT, bounds, edges)
    Nt = length(bounds) - 1
    Nw = length(edges) - 1
    ω = zeros(FT, Nt, Nw)

    for w in 1:Nw, n in 1:Nt
        ω[n, w] = window_overlap(DateTime(bounds[n]), DateTime(bounds[n + 1]),
                                 edges[w], edges[w + 1])
    end

    return ω
end

@kernel function _time_average!(averaged, data, ω, Nt, Nw)
    i, j, k = @index(Global, NTuple)
    FT = eltype(averaged)

    for w in 1:Nw
        total = zero(FT)
        weight = zero(FT)

        for n in 1:Nt
            value = @inbounds data[i, j, k, n]
            ωₙ = @inbounds ω[n, w]

            # Each cell renormalizes over its own valid samples, so a cloudy pixel drops the
            # sample while its neighbor keeps it.
            counts = (ωₙ > 0) & !isnan(value)
            total += ifelse(counts, ωₙ * value, zero(FT))
            weight += ifelse(counts, ωₙ, zero(FT))
        end

        @inbounds averaged[i, j, k, w] = ifelse(weight > 0, total / weight, convert(FT, NaN))
    end
end

"""
    time_average(fts, metadata, window)
    time_average(fts, bounds, window)

Average the `FieldTimeSeries` `fts` onto windows of length `window`, tiling `bounds` from its
first date.

`bounds` gives the `Nt + 1` dates delimiting the samples: sample `n` covers
`[bounds[n], bounds[n+1])`. Passing the `metadata` the series was read from derives them
with [`sample_bounds`](@ref), which is what a composited product wants: the composite start
dates with the end of the last one appended, where that end comes from the dataset's own
compositing rule rather than a spacing assumption a year-anchored cadence would break.

Samples are weighted by their **days of overlap** with each output window. Two things make
that mandatory rather than tidy. Each value is already a window composite, so integrating a
linear interpolation of them double-counts the compositing — the time interpolation is for
reading an arbitrary date, not for taking a mean. And 8-day periods are year-anchored, so
they do not nest inside a month: an unweighted mean silently mis-weights the samples that
straddle each edge, which is largest exactly across a green-up.

`NaN` samples are skipped and the remaining weights renormalized, so a window with any valid
overlap returns a value and one with none returns `NaN`.

Returns `(; series, edges)`: the averaged `FieldTimeSeries` and the `Nw + 1` window edges. The
result lives on the input's grid, and its times are the window centers on the same origin as
the input's, which holds when `first(bounds)` is the date the input's own time axis is measured
from and is automatic for the `metadata` form, whose first bound is the first stamp that
[`native_times`](@ref) measures from by default. The edges come back because the windows are
not uniform — the last one is whatever the record leaves over — so a series carrying only its
centers cannot say what interval each value covers.

The reduction runs on the series' own architecture, so one on a GPU grid stays on the device;
only the dates resolve to weights on the host.
"""
function time_average(fts::FieldTimeSeries, bounds, window)
    Nt = size(interior(fts))[end]
    validate_average_bounds(bounds, Nt)

    edges = average_window_edges(bounds, window)
    Nw = length(edges) - 1
    grid = fts.grid
    arch = architecture(grid)
    FT = eltype(grid)

    origin = DateTime(first(bounds))
    centers = [edges[w] + (edges[w + 1] - edges[w]) ÷ 2 for w in 1:Nw]
    times = [convert(FT, Dates.value(Millisecond(center - origin)) / 1000) for center in centers]

    LX, LY, LZ = location(fts)
    output = FieldTimeSeries{LX, LY, LZ}(grid, times)

    ω = on_architecture(arch, overlap_weights(FT, bounds, edges))
    launch!(arch, grid, :xyz, _time_average!, interior(output), interior(fts), ω, Nt, Nw)

    return (; series = output, edges)
end

time_average(fts::FieldTimeSeries, metadata::Metadata, window) =
    time_average(fts, sample_bounds(metadata), window)
