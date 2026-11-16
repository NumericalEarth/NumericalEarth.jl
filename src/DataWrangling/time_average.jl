using Dates: Dates, DateTime, Period
using Oceananigans.OutputReaders: OutputReaders, time_average

# TODO: move this method to Oceananigans, which owns `time_average` and every argument type here.
"""
$(TYPEDSIGNATURES)

Average `fts` onto consecutive calendar windows of length `window`, such as `Month(1)`, where
sample `n` covers `[bounds[n], bounds[n+1])`. The windows tile `[first(bounds), last(bounds))`,
the last one truncated at `last(bounds)` with a warning. Each sample is weighted by its overlap
with the window, so windows of unequal length are averaged exactly. The result's times are the
window centers in seconds from `first(bounds)`.
"""
function OutputReaders.time_average(fts::FieldTimeSeries, bounds::AbstractVector{DateTime}, window::Period)
    origin, finish = first(bounds), last(bounds)
    edges = collect(origin:window:finish)
    if last(edges) < finish
        push!(edges, finish)
        @warn "The last window spans $(Dates.canonicalize(finish - edges[end - 1])) rather than $window"
    end

    # Time counted in windows: every window, the truncated last one included, has unit length,
    # and the overlaps within one window keep their proportions.
    function window_time(t)
        w = min(searchsortedlast(edges, t), length(edges) - 1)
        return w - 1 + (t - edges[w]) / (edges[w + 1] - edges[w])
    end

    averaged = time_average(fts, window_time.(bounds), 1)

    seconds(t) = Dates.value(t - origin) / 1000
    times = [(seconds(edges[w]) + seconds(edges[w + 1])) / 2 for w in eachindex(averaged.times)]

    LX, LY, LZ = location(fts)
    return FieldTimeSeries{LX, LY, LZ}(averaged.data, averaged.grid, averaged.backend,
                                       averaged.boundary_conditions, averaged.indices, times,
                                       averaged.path, averaged.name, averaged.time_indexing,
                                       averaged.reader_kw)
end
