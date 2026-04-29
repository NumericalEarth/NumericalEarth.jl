using Oceananigans.OutputReaders: Cyclical, AbstractInMemoryBackend, FlavorOfFTS, time_indices

import Oceananigans.OutputReaders: new_backend
import Oceananigans.Fields: set!

@inline instantiate(T::DataType) = T()
@inline instantiate(T) = T

"""
    DatasetBackend{N, C, I, M} <: AbstractInMemoryBackend{Int}

In-memory backend for a `FieldTimeSeries` backed by a dataset whose metadata
maps each in-memory time index to a file (or subset of a file) on disk. The
backend carries

- `start`, `length`: sliding-window extents into the metadata
- `inpainting`: inpainting algorithm used when reading per-file datasets
  (e.g. `NearestNeighborInpainting`); `nothing` for datasets whose native
  NetCDF already covers the whole target grid (e.g. JRA55).
- `metadata`: the dataset metadata — its dataset type parameterises `set!`
  dispatch, so per-file and chunked (multi-time-per-file) datasets can
  coexist under the same backend.

Type parameters `N` (on-native-grid) and `C` (cache-inpainted-data) are
flags hoisted into the type so that dispatch and `Adapt.adapt_structure`
can act on them without allocating.
"""
struct DatasetBackend{N, C, I, M} <: AbstractInMemoryBackend{Int, false}
    start :: Int
    length :: Int
    inpainting :: I
    metadata :: M

    function DatasetBackend{N, C}(start::Int, length::Int, inpainting, metadata) where {N, C}
        M = typeof(metadata)
        I = typeof(inpainting)
        return new{N, C, I, M}(start, length, inpainting, metadata)
    end
end

Adapt.adapt_structure(to, b::DatasetBackend{N, C}) where {N, C} =
    DatasetBackend{N, C}(b.start, b.length, nothing, nothing)

"""
    DatasetBackend(length, metadata;
                   on_native_grid = false,
                   cache_inpainted_data = false,
                   inpainting = NearestNeighborInpainting(Inf))
    DatasetBackend(start, length, metadata; ...)

Construct a `DatasetBackend` holding `length` in-memory time indices starting
at `start` (default `1`). `inpainting = nothing` selects the dispatch path
for datasets whose files hold multiple time instances (e.g. chunked NetCDF
like JRA55), where per-file inpainting is not applicable.
"""
function DatasetBackend(length, metadata;
                        on_native_grid = false,
                        cache_inpainted_data = false,
                        inpainting = NearestNeighborInpainting(Inf))

    return DatasetBackend{on_native_grid, cache_inpainted_data}(1, length, inpainting, metadata)
end

function DatasetBackend(start::Integer, length::Integer, metadata;
                        on_native_grid = false,
                        cache_inpainted_data = false,
                        inpainting = NearestNeighborInpainting(Inf))

    return DatasetBackend{on_native_grid, cache_inpainted_data}(start, length, inpainting, metadata)
end

Base.length(backend::DatasetBackend)  = backend.length
Base.summary(backend::DatasetBackend) = string("DatasetBackend(", backend.start, ", ", backend.length, ")")

new_backend(b::DatasetBackend{native, cache_data}, start, length) where {native, cache_data} =
    DatasetBackend{native, cache_data}(start, length, b.inpainting, b.metadata)

on_native_grid(::DatasetBackend{native}) where native = native
cache_inpainted_data(::DatasetBackend{native, cache_data}) where {native, cache_data} = cache_data

const DatasetFieldTimeSeries{N} = FlavorOfFTS{<:Any, <:Any, <:Any, <:Any, <:DatasetBackend{N}} where N

# Default per-file set! — each metadata index corresponds to its own file.
# Used by datasets whose native files hold one time instance per file, such
# as ECCO4 / EN4 / WOA monthly climatologies. Datasets whose files hold
# multiple time instances (e.g. JRA55) dispatch on a more specific backend
# signature keyed off the metadata's dataset type.
function set!(fts::DatasetFieldTimeSeries, backend=fts.backend)
    inpainting = backend.inpainting
    cache_data = cache_inpainted_data(backend)

    for t in time_indices(fts)
        metadatum = @inbounds backend.metadata[t]
        set!(fts[t], metadatum; inpainting, cache_inpainted_data=cache_data)
    end

    fill_halo_regions!(fts)

    return nothing
end
