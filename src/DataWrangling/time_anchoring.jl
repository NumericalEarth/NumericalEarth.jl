using Dates: Dates, DateTime, Millisecond, UTInstant, year, isleapyear
using Oceananigans.OutputReaders: Cyclical, PartlyInMemory, find_time_index, memory_index, time_index

const MILLISECONDS_PER_DAY = 86_400_000

"""
    AbstractTimeAnchor

Supertype for the `time_indexing` rules that map a model date onto a dataset's own time stamps.
"""
abstract type AbstractTimeAnchor end

@inline Oceananigans.OutputReaders.interpolating_time_indices(anchor::AbstractTimeAnchor, times, t) =
    find_time_index(times, lookup_date(anchor, t, times))

# An anchor folds a model date back into the record, so a chunked backend wraps the way `Cyclical` does.
@inline Oceananigans.OutputReaders.time_index(backend::PartlyInMemory, ::AbstractTimeAnchor, Nt, m) = time_index(backend, Cyclical(), Nt, m)
@inline Oceananigans.OutputReaders.memory_index(backend::PartlyInMemory, ::AbstractTimeAnchor, Nt, n) = memory_index(backend, Cyclical(), Nt, n)

"""
    CalendarDate()

Read a dataset at the model's own date, folding back by whole calendar years once the model passes the end of
the record.
"""
struct CalendarDate <: AbstractTimeAnchor end

"""
    CalendarPhase()

Read a dataset at the model's month, day and time of day resolved in the dataset's own year. A model 29 February
reads 28 February.
"""
struct CalendarPhase <: AbstractTimeAnchor end

"""
    SimulationStart(start_date)

Read a dataset from its first record at `start_date`, advancing by elapsed time and wrapping by the span of the
record.
"""
struct SimulationStart{D} <: AbstractTimeAnchor
    start_date :: D
end

"""
    lookup_date(anchor, model_date, dates)

The date at which a dataset stamped at `dates` is read for a model at `model_date`.
"""
function lookup_date end

@inline function lookup_date(::CalendarDate, model_date, dates)
    first_stamp, last_stamp = first(dates), last(dates)
    first_stamp <= model_date <= last_stamp && return model_date

    cycle_years = year(last_stamp) - year(first_stamp)
    cycle_years < 1 && return model_date

    folded = model_date
    while folded > last_stamp
        folded = substitute_year(folded, year(folded) - cycle_years)
    end
    while folded < first_stamp
        folded = substitute_year(folded, year(folded) + cycle_years)
    end

    return folded
end

@inline lookup_date(::CalendarPhase, model_date, dates) = substitute_year(model_date, year(first(dates)))

@inline function lookup_date(anchor::SimulationStart, model_date, dates)
    elapsed = Dates.value(Millisecond(model_date - anchor.start_date))
    # The record runs to the end of the interval its final stamp stands for, so a repeat year ending 31 December
    # 21:00 at 3-hourly spacing spans 365 days rather than 364 days 21 hours.
    span = Dates.value(Millisecond(last(dates) - first(dates)) + Millisecond(last(dates) - dates[end - 1]))
    return first(dates) + Millisecond(mod(elapsed, span))
end

"""
    substitute_year(date, target_year)

`date` moved into `target_year`, keeping month, day and time of day. A 29 February moved into a year that is not
a leap year becomes 28 February.
"""
@inline function substitute_year(date, target_year)
    y, m, d = Dates.yearmonthday(date)
    # Day counts, not the validating `DateTime` constructor: its error string does not compile for the GPU.
    milliseconds_in_day = Dates.value(date) - MILLISECONDS_PER_DAY * Dates.totaldays(y, m, d)
    d = ifelse(m == 2 && d == 29 && !isleapyear(target_year), 28, d)

    return DateTime(UTInstant(Millisecond(MILLISECONDS_PER_DAY * Dates.totaldays(target_year, m, d) + milliseconds_in_day)))
end

"""
    default_time_indexing(dataset)

The `time_indexing` `dataset` uses unless the caller overrides it. Climatologies, whose nominal year carries no
meaning, extend this with `CalendarPhase()`.
"""
default_time_indexing(dataset) = CalendarDate()

"""
    validate_time_anchors(anchored_series)

Throw if the `(name, anchor)` pairs in `anchored_series` cannot hold a common phase.
"""
function validate_time_anchors(anchored_series)
    unanchored = [series_name for (series_name, anchor) in anchored_series if anchor isa SimulationStart]

    isempty(unanchored) && return nothing
    length(unanchored) == 1 && length(anchored_series) == 1 && return nothing

    throw(ArgumentError("$(first(unanchored)) uses SimulationStart, which drifts unless it is the only series"))
end
