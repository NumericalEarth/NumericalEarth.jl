using Dates: Dates, DateTime
using DocStringExtensions: TYPEDSIGNATURES

#####
##### Averaging a series onto a coarser cadence
#####
##### TODO: a lazily evaluated, reduced `FieldTimeSeries` is related to upstream
##### `FieldTimeSeriesOperation` work in Oceananigans (CliMA/Oceananigans.jl#5761).
##### Replace this with that when it lands.
#####

@inline function window_overlap(sample_start, sample_stop, window_start, window_stop)
    overlap = min(sample_stop, window_stop) - max(sample_start, window_start)
    return max(0, Dates.value(Dates.Second(overlap)))
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

# Each cell renormalizes over its own valid samples, so a cloudy pixel drops the sample
# while its neighbor keeps it.
@kernel function _accumulate_sample!(total, weight, sample, ω)
    i, j, k = @index(Global, NTuple)
    FT = eltype(total)
    @inbounds begin
        value = sample[i, j, k]
        counts = !isnan(value)
        total[i, j, k] += ifelse(counts, ω * value, zero(FT))
        weight[i, j, k] += ifelse(counts, ω, zero(FT))
    end
end

@kernel function _finalize_window!(averaged, total, weight, w)
    i, j, k = @index(Global, NTuple)
    FT = eltype(averaged)
    @inbounds averaged[i, j, k, w] = ifelse(weight[i, j, k] > 0,
                                            total[i, j, k] / weight[i, j, k],
                                            convert(FT, NaN))
end

"""
$(TYPEDSIGNATURES)

Average the `FieldTimeSeries` `fts` onto windows of length `window`, tiling `bounds` from its
first date.

`bounds` gives the `Nt + 1` dates delimiting the samples: sample `n` covers
`[bounds[n], bounds[n+1])`. Samples are weighted by their overlap with each window, which a
composited product needs — its values are already window means, and a year-anchored cadence
does not nest inside a month, so an unweighted mean mis-weights every sample straddling an
edge.

`NaN` samples are skipped and the remaining weights renormalized, so a window with any valid
overlap returns a value and one with none returns `NaN`.

Returns `(; series, edges)`: the averaged series, carrying the input's grid, indices, and time
indexing, and the `Nw + 1` window edges. Its times are the window centers measured from
`first(bounds)`. The edges come back because the last window is whatever the record leaves
over, so the centers alone cannot say what interval each value covers.

The reduction streams through the record one sample at a time, so `fts` may hold any number
of its times in memory; a partly-resident series has its window advanced (and left) at the
end of the record. The reduction runs on the series' own architecture; only the dates resolve
to weights on the host.
"""
function time_average(fts::FieldTimeSeries, bounds, window)
    Nt = length(fts.times)
    validate_average_bounds(bounds, Nt)

    edges = average_window_edges(bounds, window)
    Nw = length(edges) - 1
    grid = fts.grid
    arch = architecture(grid)
    FT = eltype(grid)

    origin = DateTime(first(bounds))
    centers = [edges[w] + (edges[w + 1] - edges[w]) ÷ 2 for w in 1:Nw]
    times = [convert(FT, Dates.value(Dates.Second(center - origin))) for center in centers]

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
    spatial_size = size(averaged)[1:3]
    total = on_architecture(arch, zeros(FT, spatial_size))
    weight = on_architecture(arch, zeros(FT, spatial_size))

    # Samples ascend within a window and consecutive windows share at most the straddling
    # sample, so a partly-resident series only ever advances its window forward, one move per
    # sample at worst — `fts[n]` reads the slice in whatever way its backend provides.
    for w in 1:Nw
        fill!(total, 0)
        fill!(weight, 0)

        for n in first_sample[w]:last_sample[w]
            launch!(arch, grid, spatial_size, _accumulate_sample!,
                    total, weight, interior(fts[n]), ω[n, w])
        end

        launch!(arch, grid, spatial_size, _finalize_window!, averaged, total, weight, w)
    end

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
