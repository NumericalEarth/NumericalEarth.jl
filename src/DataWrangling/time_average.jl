using Dates: Dates, DateTime, Millisecond

#####
##### Averaging a series onto a coarser cadence
#####
##### A model rarely wants a product's native cadence. The reduction is a plain eager
##### function rather than a dataset or a file format: a preprocessed temporal average earns
##### disk when it collapses years of downloads into a handful of files, which is what a
##### multi-year climatology does, and a two-month mean of an already-local series collapses
##### eight arrays into one.
#####
##### TODO: a lazily evaluated, reduced `FieldTimeSeries` is the shape of the upstream
##### `FieldTimeSeriesOperation` work in Oceananigans. Replace this with that when it lands
##### rather than growing bespoke lazy machinery here.
#####

# How much of a sample's window falls inside an output window. The unit is arbitrary — the
# weights are normalized — so milliseconds, which every date difference is exactly.
@inline function window_overlap(sample_start, sample_stop, window_start, window_stop)
    overlap = min(sample_stop, window_stop) - max(sample_start, window_start)
    return max(0, Dates.value(Millisecond(overlap)))
end

function average_window_edges(bounds, window)
    first_edge, last_edge = DateTime(first(bounds)), DateTime(last(bounds))
    edges = [first_edge]
    while last(edges) < last_edge
        push!(edges, min(last(edges) + window, last_edge))
    end
    return edges
end

"""
    time_average(series, bounds, window)

Average `series` — a `FieldTimeSeries`, or an array whose last dimension is time — onto
windows of length `window`, tiling `bounds` from its first date.

`bounds` gives the `Nt + 1` dates delimiting the samples: sample `n` covers
`[bounds[n], bounds[n+1])`. For a composited product those are the composite start dates
with the end of the last one appended, and they are asked for rather than inferred because a
product whose cadence restarts every January has a short final window that no spacing rule
recovers.

Samples are weighted by their **days of overlap** with each output window. Two things make
that mandatory rather than tidy. Each value is already a window composite, so integrating a
linear interpolation of them double-counts the compositing — the time interpolation is for
reading an arbitrary date, not for taking a mean. And 8-day periods are year-anchored, so
they do not nest inside a month: an unweighted mean silently mis-weights the samples that
straddle each edge, which is largest exactly across a green-up.

`NaN` samples are skipped and the remaining weights renormalized, so a window with any valid
overlap returns a value and one with none returns `NaN`.

Returns `(; series, edges)`: the averaged series and the `Nw + 1` window edges. For a
`FieldTimeSeries` input the result is a `FieldTimeSeries` on the same grid whose times are
the window centres, on the same origin as the input's — which holds when `first(bounds)` is
the date the input's own time axis is measured from.
"""
function time_average(data::AbstractArray, bounds, window)
    Nt = size(data)[end]
    length(bounds) == Nt + 1 ||
        throw(ArgumentError("time_average needs one more bound than the series has times " *
                            "($(Nt + 1)); got $(length(bounds)). The extra bound closes the " *
                            "last sample's window."))

    edges = average_window_edges(bounds, window)
    Nw = length(edges) - 1
    spatial = size(data)[1:end-1]

    averaged = fill(convert(eltype(data), NaN), spatial..., Nw)
    total = zeros(Float64, spatial...)
    weight = zeros(Float64, spatial...)
    indices = CartesianIndices(spatial)

    for w in 1:Nw
        fill!(total, 0)
        fill!(weight, 0)

        for n in 1:Nt
            ω = window_overlap(DateTime(bounds[n]), DateTime(bounds[n + 1]),
                               edges[w], edges[w + 1])
            ω > 0 || continue
            sample = selectdim(data, ndims(data), n)

            for I in indices
                value = sample[I]
                isnan(value) && continue
                total[I] += ω * value
                weight[I] += ω
            end
        end

        output = selectdim(averaged, ndims(averaged), w)
        for I in indices
            weight[I] > 0 && (output[I] = total[I] / weight[I])
        end
    end

    return (; series = averaged, edges)
end

function time_average(fts::FieldTimeSeries, bounds, window)
    averaged, edges = time_average(Array(interior(fts)), bounds, window)

    origin = DateTime(first(bounds))
    centres = [edges[w] + (edges[w + 1] - edges[w]) ÷ 2 for w in 1:(length(edges) - 1)]
    times = [convert(eltype(fts.grid), Dates.value(Millisecond(centre - origin)) / 1000)
             for centre in centres]

    LX, LY, LZ = location(fts)
    output = FieldTimeSeries{LX, LY, LZ}(fts.grid, times)
    copyto!(interior(output), averaged)

    return (; series = output, edges)
end
