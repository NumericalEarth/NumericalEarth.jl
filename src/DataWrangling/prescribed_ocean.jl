using ..Oceans: Oceans, PrescribedOcean
using Oceananigans.OutputReaders: Clamp
using Oceananigans.TimeSteppers: Clock

"""
$(TYPEDSIGNATURES)

Return a volumetric [`PrescribedOcean`](@ref NumericalEarth.Oceans.PrescribedOcean) holding `dataset`'s
temperature, salinity, velocities and sea surface height over `region` (a `BoundingBox`) at `dates` — a
range or vector of dates, or a `(start_date, end_date)` tuple expanded to the dataset's native cadence —
on the dataset's native grid. Time `0` of its clock is the start of `first(dates)`.

Each `FieldTimeSeries` holds `time_indices_in_memory` samples and uses `time_indexing`. The default,
`Clamp()`, holds the first sample before its time: a time-averaged product stamps its samples at the
centers of their windows, so the start of a run precedes the first sample. Remaining keyword arguments
`kw` go to `PrescribedOcean(grid, times; kw...)`.

Use it, or a [`BoundaryPrescribedComponent`](@ref NumericalEarth.EarthSystemModels.BoundaryPrescribedComponent)
of them, as the parent of [`nested_ocean_model`](@ref).
"""
function Oceans.PrescribedOcean(region::BoundingBox, dates, dataset;
                                architecture = CPU(),
                                dir = default_download_directory(dataset),
                                time_indices_in_memory = 2,
                                time_indexing = Clamp(),
                                kw...)

    metadata(name) = Metadata(name; dataset, dates, region, dir)
    series(name) = FieldTimeSeries(metadata(name), architecture; time_indices_in_memory, time_indexing)

    T = series(:temperature)
    S = series(:salinity)
    u = series(:u_velocity)
    v = series(:v_velocity)
    η = series(:free_surface)

    FT = eltype(T.grid)

    return PrescribedOcean(T.grid, T.times;
                           clock = Clock{FT}(time = 0),
                           temperature = T,
                           salinity = S,
                           velocities = (; u, v),
                           free_surface = η,
                           kw...)
end
