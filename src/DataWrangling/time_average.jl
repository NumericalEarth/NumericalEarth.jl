using Dates: Dates, DateTime, Millisecond
using DocStringExtensions: TYPEDSIGNATURES

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

# TODO: stream instead. `overlap_weights` returns the band each window sums over, so only
# `first_sample[w]:last_sample[w]` has to be resident; what pins the whole record in memory is
# the kernel indexing the global sample number rather than the resident buffer, which would
# silently average the wrong slices. The generic fix is the upstream one noted above.
function validate_average_series(fts, Nt)
    Nr = size(interior(fts))[end]
    Nr == Nt ||
        throw(ArgumentError("time_average reduces the whole record at once, so `fts` has to hold " *
                            "all $Nt of its times in memory; it holds $Nr. Read it with " *
                            "`time_indices_in_memory = $Nt`."))
    return nothing
end

function average_window_edges(bounds, window)
    first_edge, last_edge = DateTime(first(bounds)), DateTime(last(bounds))

    first_edge + window > first_edge ||
        throw(ArgumentError("time_average tiles the record with `window`, so it has to advance " *
                            "the clock; got $window."))

    edges = [first_edge]
    while last(edges) < last_edge
        push!(edges, min(last(edges) + window, last_edge))
    end
    return edges
end

# Every cell shares the same weights, so the dates are resolved once on the host and the
# kernel reads scalars out of an `(Nt, Nw)` matrix. That matrix is banded — a window overlaps
# consecutive samples only — so each window also carries the range it has to sum over.
function overlap_weights(FT, bounds, edges)
    Nt = length(bounds) - 1
    Nw = length(edges) - 1
    ω = zeros(FT, Nt, Nw)
    first_sample = zeros(Int, Nw)
    last_sample = zeros(Int, Nw)

    for w in 1:Nw
        for n in 1:Nt
            ω[n, w] = window_overlap(DateTime(bounds[n]), DateTime(bounds[n + 1]),
                                     edges[w], edges[w + 1])
        end

        overlapping = findall(>(0), view(ω, :, w))
        first_sample[w] = isempty(overlapping) ? 1 : first(overlapping)
        last_sample[w] = isempty(overlapping) ? 0 : last(overlapping)
    end

    return ω, first_sample, last_sample
end

@kernel function _time_average!(averaged, data, ω, first_sample, last_sample, Nw)
    i, j, k = @index(Global, NTuple)
    FT = eltype(averaged)

    for w in 1:Nw
        total = zero(FT)
        weight = zero(FT)

        @inbounds for n in first_sample[w]:last_sample[w]
            value = data[i, j, k, n]
            ωₙ = ω[n, w]

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
$(TYPEDSIGNATURES)

Average the `FieldTimeSeries` `fts` onto windows of length `window`, tiling `bounds` from its
first date.

`bounds` gives the `Nt + 1` dates delimiting the samples: sample `n` covers
`[bounds[n], bounds[n+1])`. Samples are weighted by their days of overlap with each window,
which a composited product needs: its values are already window means, so integrating an
interpolation of them double-counts the compositing, and a year-anchored cadence does not nest
inside a month, so an unweighted mean mis-weights every sample straddling an edge.

`NaN` samples are skipped and the remaining weights renormalized, so a window with any valid
overlap returns a value and one with none returns `NaN`.

Returns `(; series, edges)`: the averaged series, carrying the input's grid, indices, and time
indexing, and the `Nw + 1` window edges. Its times are the window centers measured from
`first(bounds)`. The edges come back because the last window is whatever the record leaves
over, so the centers alone cannot say what interval each value covers.

The whole record reduces at once, so `fts` has to hold all of its times in memory: read it with
`time_indices_in_memory = length(fts.times)`. The reduction runs on the series' own architecture;
only the dates resolve to weights on the host.
"""
function time_average(fts::FieldTimeSeries, bounds, window)
    Nt = length(fts.times)
    validate_average_series(fts, Nt)
    validate_average_bounds(bounds, Nt)

    edges = average_window_edges(bounds, window)
    Nw = length(edges) - 1
    grid = fts.grid
    arch = architecture(grid)
    FT = eltype(grid)

    origin = DateTime(first(bounds))
    centers = [edges[w] + (edges[w + 1] - edges[w]) ÷ 2 for w in 1:Nw]
    times = [convert(FT, Dates.value(Millisecond(center - origin)) / 1000) for center in centers]

    # A single window leaves `Cyclical()` no interval to infer its period from.
    cannot_cycle = Nw == 1 && fts.time_indexing isa Cyclical{Nothing}
    time_indexing = cannot_cycle ? Clamp() : fts.time_indexing

    # TODO: window the boundary conditions the way a sliced `Field` does. A sliced
    # `FieldTimeSeries` keeps the unsliced ones, so `fill_halo_regions!` writes a halo the slice
    # does not have — which lands in the next time slice. Fix this in Oceananigans' constructor.
    boundary_conditions = FieldBoundaryConditions(fts.indices, fts.boundary_conditions)

    LX, LY, LZ = location(fts)
    output = FieldTimeSeries{LX, LY, LZ}(grid, times; indices = fts.indices,
                                         time_indexing, boundary_conditions)

    ω, first_sample, last_sample = overlap_weights(FT, bounds, edges)
    averaged = interior(output)

    launch!(arch, grid, size(averaged)[1:3], _time_average!,
            averaged, interior(fts), on_architecture(arch, ω),
            on_architecture(arch, first_sample), on_architecture(arch, last_sample), Nw)

    fill_halo_regions!(output)

    return (; series = output, edges)
end

"""
$(TYPEDSIGNATURES)

Average `fts` onto windows of length `window`, taking the sample bounds from the `metadata` the
series was read from with [`sample_bounds`](@ref) — the composite stamps, closed with the end of
the last composite from the dataset's own compositing rule rather than a spacing assumption.
"""
time_average(fts::FieldTimeSeries, metadata::Metadata, window) =
    time_average(fts, sample_bounds(metadata), window)
