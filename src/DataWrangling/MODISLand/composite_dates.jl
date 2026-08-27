#####
##### Dates
#####

"""
    modis_composite_dates(start_date, end_date, period_days)

Return the year-anchored composite dates between `start_date` and `end_date` inclusive: in
each year, day-of-year `1, 1 + period_days, 1 + 2 * period_days, …` up to the last period
that begins within that year.

MODIS land products restart their compositing period at day-of-year 1 every January, so
the last period of a year is short (5 days, or 6 in a leap year, for an 8-day product) and
the sequence is *not* a uniform cadence across a year boundary — stepping uniformly from
the first date would drift out of phase after one year.

```jldoctest
julia> using Dates, NumericalEarth.DataWrangling.MODISLand

julia> dates = MODISLand.modis_composite_dates(DateTime(2020), DateTime(2021, 12, 31), 8);

julia> length(dates), dates[46], dates[47]
(92, DateTime("2020-12-26T00:00:00"), DateTime("2021-01-01T00:00:00"))
```
"""
function modis_composite_dates(start_date, end_date, period_days)
    dates = DateTime[]
    for year in Dates.year(start_date):Dates.year(end_date)
        january_first = DateTime(year, 1, 1)
        for day in 1:period_days:Dates.daysinyear(year)
            date = january_first + Dates.Day(day - 1)
            start_date ≤ date ≤ end_date && push!(dates, date)
        end
    end
    return dates
end

DataWrangling.all_dates(dataset::MODISLAIDataset, variable) =
    modis_composite_dates(first_composite_date(dataset), last_composite_date(dataset),
                          composite_period_days(dataset))

"""
    periods_per_year(dataset)

The number of composites a year of `dataset` holds — 46 for an 8-day product, in leap and
common years alike, since the cadence restarts every January.
"""
periods_per_year(dataset) = length(modis_composite_dates(climatology_year_start(),
                                                         climatology_year_end(),
                                                         composite_period_days(dataset)))

# A common (non-leap) placeholder year carries the climatological stamps.
climatology_year_start() = DateTime(2018, 1, 1)
climatology_year_end()   = DateTime(2018, 12, 31)

DataWrangling.all_dates(climatology::MODISLAIClimatology, variable) =
    modis_composite_dates(climatology_year_start(), climatology_year_end(),
                          composite_period_days(climatology))

DataWrangling.is_seasonal_climatology(::MODISLAIClimatology) = true

DataWrangling.all_dates(dataset::MCD12Q1, variable) =
    [DateTime(year) for year in first_landcover_year(dataset):last_landcover_year(dataset)]

"""
    period_index(date, period_days)
    period_index(date, dataset)

The 1-based index of the year-anchored composite period containing `date` — which of a
seasonal climatology's periods a calendar date belongs to, and so the `anchor_periods` a
date-window series needs to map onto a climatology.

```jldoctest
julia> using NumericalEarth, Dates

julia> period_index(DateTime(2019, 7, 4), MCD15A2H())
24
```
"""
period_index(date, period_days::Integer) = (dayofyear(date) - 1) ÷ period_days + 1

period_index(date, dataset::AbstractMODISLandDataset) =
    period_index(date, composite_period_days(dataset))

"""
    composite_window(dataset, date)

The `(start, stop)` dates of the compositing window a file stamped `date` holds. The
cadence restarts on 1 January, so the last window of a year is short — five days, or six in
a leap year, for an 8-day product.
"""
function composite_window(dataset, date)
    period_days = composite_period_days(dataset)
    start = DateTime(date)
    year_start = DateTime(Dates.year(start), 1, 1)
    stop = year_start + Day(period_index(start, period_days) * period_days)
    return start, min(stop, year_start + Dates.Year(1))
end

DataWrangling.averaging_window(metadatum::Union{MODISLAIMetadatum,
                                            MODISLAIClimatologyMetadatum}) =
    composite_window(metadatum.dataset, metadatum.dates)
