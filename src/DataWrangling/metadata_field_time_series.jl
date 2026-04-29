using Oceananigans.Architectures: AbstractArchitecture, architecture
using Oceananigans.Grids: AbstractGrid
using Oceananigans.Fields: interpolate!

import Oceananigans.OutputReaders: update_field_time_series!, FieldTimeSeries

"""
    FieldTimeSeries(metadata::Metadata [, arch_or_grid=CPU() ];
                    time_indices_in_memory = 2,
                    time_indexing = Cyclical(),
                    inpainting = nothing,
                    cache_inpainted_data = true)

Create a FieldTimeSeries from a dataset that corresponds to `metadata`.

Arguments
=========

- `metadata`: `Metadata` containing information about the dataset.

- `arch_or_grid`: Either a grid to interpolate the data to, or an `arch`itecture
                  to use for the native grid. Default: CPU().

Keyword Arguments
=================

- `time_indices_in_memory`: The number of time indices to keep in memory. Default: 2.

- `time_indexing`: The time indexing scheme to use. Default: `Cyclical()`.

- `inpainting`: The inpainting algorithm to use for the interpolation.
                The only option is `NearestNeighborInpainting(maxiter)`,
                where an average of the valid surrounding values is used `maxiter` times.

- `cache_inpainted_data`: If `true`, the data is cached to disk after inpainting for later retrieving.
                          Default: `true`.
"""
function FieldTimeSeries(metadata::Metadata, arch::AbstractArchitecture=CPU(); kw...)
    download_dataset(metadata)
    grid = native_grid(metadata, arch)
    return FieldTimeSeries(metadata, grid; kw...)
end

function FieldTimeSeries(metadata::Metadata, grid::AbstractGrid;
                         time_indices_in_memory = 2,
                         time_indexing = Cyclical(),
                         inpainting = default_inpainting(metadata),
                         cache_inpainted_data = true,
                         prefetch = false)

    download_dataset(metadata)

    # Detect "the user's grid IS the native grid" structurally 
    on_native_grid = grid == native_grid(metadata, architecture(grid))
    times = native_times(metadata)
    
    # Make sure we do not use more indices then the ones available!
    if length(times) < time_indices_in_memory
        time_indices_in_memory = length(times)
    end

    inpainting isa Int && (inpainting = NearestNeighborInpainting(inpainting))
    inner_backend = DatasetBackend(time_indices_in_memory, metadata; on_native_grid, inpainting, cache_inpainted_data)

    loc = LX, LY, LZ = location(metadata)
    boundary_conditions = FieldBoundaryConditions(grid, instantiate.(loc))

    if prefetch
        Threads.nthreads() < 2 && @warn "prefetch=true is a no-op with JULIA_NUM_THREADS=$(Threads.nthreads()); start Julia with ≥ 2 threads."
        # Buffer FTS is allocated once and reused per reload (see prefetching_backend.jl).
        buffer_inner = new_backend(inner_backend, 1, time_indices_in_memory)
        buffer_fts = FieldTimeSeries{LX, LY, LZ}(grid, times; backend=buffer_inner, time_indexing, boundary_conditions)
        backend = PrefetchingBackend(inner_backend, buffer_fts)
    else
        backend = inner_backend
    end

    fts = FieldTimeSeries{LX, LY, LZ}(grid, times; backend, time_indexing, boundary_conditions)
    set!(fts)

    return fts
end

function FieldTimeSeries(variable_name::Symbol;
                         dataset, dir,
                         architecture = CPU(),
                         start_date = first_date(dataset, variable_name),
                         end_date = last_date(dataset, variable_name),
                         kw...)

    native_dates = all_dates(dataset, variable_name)
    dates = compute_native_date_range(native_dates, start_date, end_date)
    metadata = Metadata(variable_name; dataset, dates, dir)
    return FieldTimeSeries(metadata, architecture; kw...)
end
