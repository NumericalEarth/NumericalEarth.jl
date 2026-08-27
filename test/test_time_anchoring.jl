include("runtests_setup.jl")

using Dates
using NumericalEarth.DataWrangling: CalendarDate, CalendarPhase, SimulationStart,
                                    lookup_date, default_time_indexing, validate_time_anchors

multi_year  = DateTime(1958, 1, 1) : Hour(3) : DateTime(2018, 1, 1)
repeat_year = DateTime(1990, 1, 1) : Hour(3) : DateTime(1990, 12, 31, 21)
climatology = [DateTime(2018, m, 1) for m in 1:12]

@testset "Time anchoring" begin
    @testset "CalendarDate" begin
        @test lookup_date(CalendarDate(), DateTime(1966, 7, 5, 9), multi_year) == DateTime(1966, 7, 5, 9)
        @test lookup_date(CalendarDate(), DateTime(2018, 1, 1),    multi_year) == DateTime(2018, 1, 1)

        # A run continuing past the record folds back by whole calendar years: a repeated OMIP cycle, with the
        # model clock left monotonic.
        @test lookup_date(CalendarDate(), DateTime(2018, 6, 1),     multi_year) == DateTime(1958, 6, 1)
        @test lookup_date(CalendarDate(), DateTime(2043, 3, 15, 9), multi_year) == DateTime(1983, 3, 15, 9)
        @test lookup_date(CalendarDate(), DateTime(2100, 3, 15, 9), multi_year) == DateTime(1980, 3, 15, 9)

        # Every day of a full cycle folds by the same whole-year shift, leap days included.
        mismatches = 0
        date = DateTime(2018, 1, 2)
        while date < DateTime(2078, 1, 1)
            lookup_date(CalendarDate(), date, multi_year) == date - Year(60) || (mismatches += 1)
            date += Day(1)
        end
        @test mismatches == 0
    end

    @testset "CalendarPhase" begin
        # Incompatible nominal years coexist because neither is ever the model's.
        model_date = DateTime(2043, 3, 15, 9)
        @test lookup_date(CalendarPhase(), model_date, repeat_year) == DateTime(1990, 3, 15, 9)
        @test lookup_date(CalendarPhase(), model_date, climatology) == DateTime(2018, 3, 15, 9)

        @test lookup_date(CalendarPhase(), DateTime(2044, 2, 29, 9), repeat_year) == DateTime(1990, 2, 28, 9)
        @test lookup_date(CalendarPhase(), DateTime(2044, 2, 29, 9), climatology) == DateTime(2018, 2, 28, 9)

        leap_nominal = DateTime(1960, 1, 1) : Hour(3) : DateTime(1960, 12, 31, 21)
        @test lookup_date(CalendarPhase(), DateTime(2044, 2, 29, 9), leap_nominal) == DateTime(1960, 2, 29, 9)
    end

    @testset "SimulationStart" begin
        anchor = SimulationStart(DateTime(1958, 1, 1))

        @test lookup_date(anchor, DateTime(1958, 1, 1),    repeat_year) == DateTime(1990, 1, 1)
        @test lookup_date(anchor, DateTime(1958, 1, 2, 9), repeat_year) == DateTime(1990, 1, 2, 9)
        @test lookup_date(anchor, DateTime(1959, 1, 1),    repeat_year) == DateTime(1990, 1, 1)
    end

    @testset "Series of one product resolve to the same instant" begin
        # JRA55-do stamps instantaneous variables on the hour and window means at the window midpoint, so the
        # two sets of stamps sit 90 minutes apart.
        instantaneous = DateTime(1958, 1, 1)        : Hour(3) : DateTime(2018, 1, 1)
        window_mean   = DateTime(1958, 1, 1, 1, 30) : Hour(3) : DateTime(2018, 1, 1)
        model_date    = DateTime(1958, 6, 15, 9)

        @test lookup_date(CalendarDate(), model_date, instantaneous) == model_date
        @test lookup_date(CalendarDate(), model_date, window_mean)   == model_date

        # Counting seconds from each series' own first stamp puts equal offsets at different instants.
        offset = model_date - DateTime(1958, 1, 1)
        @test first(instantaneous) + offset == model_date
        @test first(window_mean)   + offset == model_date + Minute(90)
    end

    @testset "Anchors in one model" begin
        @test default_time_indexing(MultiYearJRA55()) isa CalendarDate
        @test default_time_indexing(RepeatYearJRA55()) isa CalendarPhase
        @test default_time_indexing(WOAMonthly()) isa CalendarPhase

        # A reanalysis with a climatological restoring is the common case and holds phase.
        @test isnothing(validate_time_anchors([(:temperature, default_time_indexing(MultiYearJRA55())),
                                               (:salinity, default_time_indexing(WOAMonthly()))]))

        start = SimulationStart(DateTime(1958, 1, 1))
        @test isnothing(validate_time_anchors([(:temperature, start)]))
        @test_throws ArgumentError validate_time_anchors([(:temperature, start), (:salinity, start)])
        @test_throws ArgumentError validate_time_anchors([(:temperature, start), (:salinity, CalendarPhase())])
    end
end
