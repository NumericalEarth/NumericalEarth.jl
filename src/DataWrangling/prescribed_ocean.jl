using ..Oceans: Oceans, PrescribedOcean
using Oceananigans.OutputReaders: Clamp
using Oceananigans.TimeSteppers: Clock

"""
$(TYPEDSIGNATURES)

Return a volumetric [`PrescribedOcean`](@ref NumericalEarth.Oceans.PrescribedOcean) holding `dataset`'s
temperature, salinity, velocities and sea surface height over `region` (a `BoundingBox`) at `dates` — a
range or vector of dates, or a `(start_date, end_date)` tuple expanded to the dataset's native cadence —
on the dataset's native grid. Time `0` of its clock is the start of `first(dates)`.

A time-averaged product places its samples at the centers of their windows, so the start of a run, at the start
of a sample's window, precedes that sample. The series therefore also hold the native samples just before
`first(dates)` and just after `last(dates)` whenever those are needed to interpolate across the whole window,
and keep time `0` at the start of `first(dates)`. Each `FieldTimeSeries` holds `time_indices_in_memory`
samples and uses `time_indexing` beyond the samples it holds. Remaining keyword arguments `kw` go to
`PrescribedOcean(grid, times; kw...)`.

Use it, or a [`BoundaryPrescribedComponent`](@ref NumericalEarth.EarthSystemModels.BoundaryPrescribedComponent)
of them, as the parent of [`nested_ocean_model`](@ref).
"""
function Oceans.PrescribedOcean(region::BoundingBox, dates, dataset;
                                architecture = CPU(),
                                dir = default_download_directory(dataset),
                                time_indices_in_memory = 2,
                                time_indexing = Clamp(),
                                kw...)

    native_dates = bracketing_dates(dataset, :temperature, dates)
    start_time = first(dates)

    metadata(name) = Metadata(name; dataset, dates = native_dates, region, dir)
    series(name) = FieldTimeSeries(metadata(name), architecture; time_indices_in_memory, time_indexing, start_time)

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

# The native dates of `dataset` that cover `dates`, plus the neighboring samples needed to interpolate up to
# `first(dates)` and `last(dates)` when the samples sit at the centers of their averaging windows.
function bracketing_dates(dataset, name, dates)
    available = collect(all_dates(dataset, name))
    native = expand_dates(dataset, name, dates)
    start, stop = comparable_datetime(first(dates)), comparable_datetime(last(dates))
    center(date) = window_center(Metadatum(name; dataset, date))

    n₁ = findfirst(==(first(native)), available)
    n₂ = findfirst(==(last(native)), available)
    n₁ > 1 && center(available[n₁]) > start && (n₁ -= 1)
    n₂ < length(available) && center(available[n₂]) < stop && (n₂ += 1)

    return available[n₁:n₂]
end
