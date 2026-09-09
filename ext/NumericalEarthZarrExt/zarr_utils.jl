#####
##### Filters and windowing helpers shared by the Zarr store readers
#####

# The GLO-90 store applies the numcodecs `bitround` filter (lossy mantissa
# rounding done at write time). Decoding is a passthrough: the stored values are
# already the rounded floats, so no inverse is needed. Zarr.jl has no built-in
# bitround filter, so we register a minimal one in its global `filterdict`.
struct BitRoundFilter{T, Tenc} <: Zarr.Filter{T, Tenc}
    keepbits::Int32
end

BitRoundFilter(; keepbits = 14, T = Float32, Tenc = T) = BitRoundFilter{T, Tenc}(Int32(keepbits))

Zarr.zencode(data::AbstractArray, ::BitRoundFilter) = data
Zarr.zdecode(data::AbstractArray, ::BitRoundFilter) = data
Zarr.JSON.lower(filter::BitRoundFilter) = Dict("id" => "bitround", "keepbits" => filter.keepbits)
Zarr.getfilter(::Type{<:BitRoundFilter}, d) = BitRoundFilter(; keepbits = d["keepbits"])

# Register at load time, not precompile time: mutating Zarr's global `filterdict`
# from module top-level runs during precompilation and is discarded, so the entry
# would be missing at runtime.
function __init__()
    Zarr.filterdict["bitround"] = BitRoundFilter
end

# A contiguous block of `count` storage indices into `coordinate` whose values
# bracket the window starting near `target_first`, returned in storage order
# together with whether `coordinate` is ascending. Assumes the store resolution
# matches the native grid, so a contiguous block of length `count` is exact.
function ascending_window(coordinate, target_first, count)
    n = length(coordinate)
    ascending = coordinate[1] < coordinate[end]
    ascending_coordinate = ascending ? coordinate : reverse(coordinate)

    start = searchsortednearest(ascending_coordinate, target_first)
    start = clamp(start, 1, n - count + 1)
    ascending_range = start:(start + count - 1)

    storage_range = ascending ? ascending_range :
                                (n - ascending_range.stop + 1):(n - ascending_range.start + 1)

    return storage_range, ascending
end

function searchsortednearest(sorted, value)
    i = searchsortedfirst(sorted, value)
    i == 1 && return 1
    i > length(sorted) && return length(sorted)
    return abs(sorted[i] - value) < abs(sorted[i-1] - value) ? i : i - 1
end

# Contiguous pieces of `window` split at global multiples of `tile_size`, so tiles share
# no store chunk when `tile_size` is a multiple of the chunk edge.
function tile_ranges(window, tile_size)
    start = fld(first(window) - 1, tile_size) * tile_size + 1
    return (max(first(window), k):min(last(window), k + tile_size - 1)
            for k in start:tile_size:last(window))
end

# Index of the target cell whose faces bracket each coordinate; 0 outside the target grid.
function target_cells(faces, coordinates, N)
    return map(coordinates) do c
        i = searchsortedlast(faces, c)
        ifelse(1 ≤ i ≤ N, i, 0)
    end
end
