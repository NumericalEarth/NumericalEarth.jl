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
