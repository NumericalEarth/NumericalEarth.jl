using Dates: Dates, DateTime, Period
using Oceananigans.OutputReaders: OutputReaders, time_average

"""
$(TYPEDSIGNATURES)

Average `fts` onto consecutive calendar windows of length `window`, such as `Month(1)`, where
sample `n` covers `[bounds[n], bounds[n+1])`. Each sample is weighted by its overlap with the
window, so windows of unequal length are averaged exactly. The result's times are the window
centers in seconds from `first(bounds)`.
"""
function OutputReaders.time_average(fts::FieldTimeSeries, bounds::AbstractVector{DateTime}, window::Period)
    origin, finish = first(bounds), last(bounds)
    edges = origin:window:(finish + window)

    # Time counted in windows: every window has unit length, and the overlaps within one
    # window keep their proportions.
    function window_time(t)
        w = searchsortedlast(edges, t)
        return w - 1 + (t - edges[w]) / (edges[w + 1] - edges[w])
    end

    averaged = time_average(fts, window_time.(bounds), 1)

    seconds(t) = Dates.value(t - origin) / 1000
    times = [(seconds(edges[w]) + seconds(min(edges[w + 1], finish))) / 2
             for w in eachindex(averaged.times)]

    LX, LY, LZ = location(fts)
    output = FieldTimeSeries{LX, LY, LZ}(fts.grid, times; indices = fts.indices,
                                         time_indexing = fts.time_indexing,
                                         boundary_conditions = averaged.boundary_conditions)
    parent(output) .= parent(averaged)

    return output
end
