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
                                                    cache_inpainted_data = true)

    Downloads.download(metadata)

    # Match the time axis to the grid's float type. `native_times` returns `Float64` seconds, but with a
    # Float32 grid that mismatch makes `interpolate`'s time weight `Float64`, so the interpolated value is
    # `Union{Float32, Float64}` — a type instability that boxes inside GPU tendency/halo kernels.
    times = convert.(eltype(grid), native_times(metadata))

    # A window-averaged series repeats over the span its windows tile, not over the span of its
    # nodes, which sit half a window inside it at each end. Oceananigans infers the latter.
    if time_indexing isa Cyclical{Nothing}
        period = sample_window_span(metadata)
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
    backend = DatasetBackend(time_indices_in_memory, metadata; on_native_grid, inpainting, cache_inpainted_data)

    loc = LX, LY, LZ = location(metadata)
    boundary_conditions = FieldBoundaryConditions(grid, instantiate.(loc))

    fts = FieldTimeSeries{LX, LY, LZ}(grid, times; backend, time_indexing, boundary_conditions)
    set!(fts)

    return fts
end

"""
    validate_time_coverage(metadata, time_indexing)

Check that `time_indexing` has something defensible to say outside the nodes of a
window-averaged series.

Nodes sit at [`window_center`](@ref)s, so the first and last half window of the span the data
covers falls outside them — see [`uncovered_time_gaps`](@ref). `Clamp` holds the end values
there, which is what it was asked to do; `Cyclical` fills them from the far end of the record,
which is the cycle itself only if the dates span whole cycles; `Linear` holds the first value
before the first node but extrapolates without bound past the last, so it alone can leave the
range of the data.

A single sample has no far end and nothing to interpolate between — it is constant in time —
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
