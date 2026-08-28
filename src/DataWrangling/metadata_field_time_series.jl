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
function Oceananigans.OutputReaders.FieldTimeSeries(metadata::Metadata, arch::AbstractArchitecture=CPU(); kw...)
    Downloads.download(metadata)
    grid = native_grid(metadata, arch)
    return FieldTimeSeries(metadata, grid; kw...)
end

function Oceananigans.OutputReaders.FieldTimeSeries(metadata::Metadata, grid::AbstractGrid;
                                                    time_indices_in_memory = 2,
                                                    time_indexing = Cyclical(),
                                                    inpainting = default_inpainting(metadata),
                                                    cache_inpainted_data = true,
                                                    prefetch = false)

    Downloads.download(metadata)

    # Keep the time axis in Float64, matching the model clock. `interpolate`'s time weight promotes to the
    # clock's type regardless of the axis, so a narrower axis buys no type stability and only costs
    # resolution: past 2^28 s a Float32 axis rounds nodes by up to 32 s, which lets the bracketing weight
    # exceed 1 and `Cyclical` read that as running off the end of the record, wrapping the in-memory window
    # to the last snapshot for one step out of every few.
    times = native_times(metadata)


    # A window-averaged series repeats over the span its windows tile, not over the span of its
    # nodes, which sit half a window inside it at each end. Oceananigans infers the latter.
    if time_indexing isa Cyclical{Nothing}
        period = window_span(metadata)
        isnothing(period) || (time_indexing = Cyclical(convert(eltype(grid), period)))
    end

    validate_time_coverage(metadata, time_indexing)

    # Make sure we do not use more indices then the ones available!
    if length(times) < time_indices_in_memory
        time_indices_in_memory = length(times)
    end

    inpainting isa Int && (inpainting = NearestNeighborInpainting(inpainting))
    # Grids of different type are never equal; the `typeof` guard short-circuits
    # before the node comparison, which for a `PressureLevelGrid` reduces the whole
    # geopotential to a column-mean profile (`mean_height_profile`) only to discard
    # it whenever — as for any interpolation target — the grid isn't the native one.
    native = native_grid(metadata, architecture(grid))
    on_native_grid = typeof(grid) === typeof(native) && grid == native
    inner_backend = DatasetBackend(time_indices_in_memory, metadata; on_native_grid, inpainting, cache_inpainted_data)

    loc = LX, LY, LZ = location(metadata)
    boundary_conditions = FieldBoundaryConditions(grid, instantiate.(loc))

    if prefetch
        Threads.nthreads() < 2 && @warn "prefetch=true is a no-op with JULIA_NUM_THREADS=$(Threads.nthreads()); start Julia with ≥ 2 threads."
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

"""
    validate_time_coverage(metadata, time_indexing)

Check that `time_indexing` is a valid operation outside the nodes of a window-averaged series.

Nodes sit at [`window_center`](@ref)s, so the first and last half window of the span the data
covers falls outside them — see [`uncovered_time_gaps`](@ref). `Clamp` holds the end values
there; `Cyclical` fills them from the far end of the record; `Linear` holds the first value
before the first node but extrapolates without bound past the last, so it can leave the range
of the data.

A single sample has no far end and nothing to interpolate between (it is constant in time)
so it is left alone whatever the scheme.
"""
validate_time_coverage(metadata, time_indexing) = nothing

# One warning per dataset per session, rather than one for whichever series is built first.
warning_id(metadata) = Symbol(:time_coverage_, nameof(typeof(metadata.dataset)))

function validate_time_coverage(metadata, time_indexing::Cyclical)
    length(metadata) == 1 && return nothing
    head, tail = uncovered_time_gaps(metadata)
    head == 0 && tail == 0 && return nothing

    @warn string(summary(metadata.dataset), " holds window means, so this series interpolates only ",
                 "between ", window_center(first(metadata)), " and ", window_center(last(metadata)), ". ",
                 "`Cyclical()` fills the ", prettytime(head), " before that and the ", prettytime(tail),
                 " after it by wrapping around to the other end of the record, which is the cycle ",
                 "itself only if the dates span whole cycles. Extend the dates by one interval at ",
                 "each end to interpolate across the whole span.") maxlog=1 _id=warning_id(metadata)

    return nothing
end

function validate_time_coverage(metadata, time_indexing::LinearTimeIndexing)
    length(metadata) == 1 && return nothing
    head, tail = uncovered_time_gaps(metadata)
    head == 0 && tail == 0 && return nothing

    throw(ArgumentError(string(summary(metadata.dataset), " holds window means, so this series ",
                               "interpolates only between ", window_center(first(metadata)), " and ",
                               window_center(last(metadata)), ", and `Linear()` extrapolates outside ",
                               "it: it holds the first value for the ", prettytime(head), " before, ",
                               "and extrapolates without bound for the ", prettytime(tail), " after. ",
                               "Use `Clamp()` to hold the end values, `Cyclical()` to wrap around, or ",
                               "extend the dates by one interval at each end.")))
end

function Oceananigans.OutputReaders.FieldTimeSeries(variable_name::Symbol;
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
