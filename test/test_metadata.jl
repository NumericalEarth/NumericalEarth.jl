include("runtests_setup.jl")

using NumericalEarth.DataWrangling: Column, Linear, Nearest,
                                    BoundingBox, dataset_location,
                                    restrict_location, native_grid
using NumericalEarth.DataWrangling: restrict, restrict_longitude, download_cache
using NumericalEarth.DataWrangling: native_times, sample_window, window_center, time_window_offset,
                                    sample_window_span, uncovered_time_gaps, validate_time_coverage
using NumericalEarth.DataWrangling.ERA5: ERA5HourlySingleLevel, ERA5MonthlySingleLevel,
                                         ERA5MonthlyPressureLevels, ERA5MonthlyLand

using Oceananigans: location
using Oceananigans.Grids: topology, Flat, Bounded, Periodic, RectilinearGrid,
                          LatitudeLongitudeGrid, Center, λnodes
using Oceananigans.OutputReaders: Clamp, Cyclical
# `Linear` is a spatial interpolation kind in `DataWrangling` and a time-extrapolation scheme
# in Oceananigans; both are used below.
using Oceananigans.OutputReaders: Linear as LinearTimeIndexing

@testset "Column construction" begin
    col = Column(35.1, 50.1)
    @test col.longitude == 35.1
    @test col.latitude == 50.1
    @test col.z === nothing
    @test col.interpolation isa Linear

    col_nearest = Column(35.1, 50.1; interpolation=Nearest())
    @test col_nearest.interpolation isa Nearest

    col_z = Column(35.1, 50.1; z=(-400, 0))
    @test col_z.z == (-400, 0)
end

@testset "Column isa checks" begin
    @test Column(0, 0) isa Column
    @test !(BoundingBox(longitude=(0, 10), latitude=(0, 10)) isa Column)
    @test !(nothing isa Column)
end

@testset "restrict_location" begin
    # Column reduces horizontal locations to Nothing
    @test restrict_location((Center, Center, Center), Column(0, 0)) == (Nothing, Nothing, Center)
    @test restrict_location((Face, Center, Center), Column(0, 0)) == (Nothing, Nothing, Center)
    @test restrict_location((Center, Face, Center), Column(0, 0)) == (Nothing, Nothing, Center)
    @test restrict_location((Center, Center, Nothing), Column(0, 0)) == (Nothing, Nothing, Nothing)

    # BoundingBox and nothing leave location unchanged
    bbox = BoundingBox(longitude=(0, 10), latitude=(0, 10))
    @test restrict_location((Face, Center, Center), bbox) == (Face, Center, Center)
    @test restrict_location((Center, Center, Center), nothing) == (Center, Center, Center)

    # Restrict location with (0, 360) longitude
    bbox = BoundingBox(longitude=(0, 360), latitude=(0, 10))
    @test restrict_longitude(bbox.longitude, (0, 360), 10) == ((0, 360), 10)
end

@testset "dataset_location fallback" begin
    # Default fallback returns (Center, Center, Center)
    @test dataset_location(ECCO2Monthly(), :temperature) == (Center, Center, Center)
    @test dataset_location(ECCO4Monthly(), :temperature) == (Center, Center, Center)

    # ECCO staggered velocities
    @test dataset_location(ECCO4Monthly(), :u_velocity) == (Face, Center, Center)
    @test dataset_location(ECCO4Monthly(), :v_velocity) == (Center, Face, Center)

    # ECCO 2D fields
    @test dataset_location(ECCO4Monthly(), :free_surface) == (Center, Center, Nothing)

    # Non-ECCO datasets use the generic fallback
    @test dataset_location(JRA55.RepeatYearJRA55(), :temperature) == (Center, Center, Center)
end

@testset "location(metadata) with Column region" begin
    # Column metadata: location is restricted
    col = Column(35.1, 50.1)
    md = Metadatum(:temperature; dataset=ECCO4Monthly(), region=col)
    @test location(md) == (Nothing, Nothing, Center)

    # ECCO velocity + Column: horizontal locations dropped
    md_u = Metadatum(:u_velocity; dataset=ECCO4Monthly(), region=col)
    @test location(md_u) == (Nothing, Nothing, Center)

    # ECCO 2D field + Column
    md_fs = Metadatum(:free_surface; dataset=ECCO4Monthly(), region=col)
    @test location(md_fs) == (Nothing, Nothing, Nothing)

    # No region: full dataset location
    md_full = Metadatum(:u_velocity; dataset=ECCO4Monthly())
    @test location(md_full) == (Face, Center, Center)

    # BoundingBox: full dataset location
    bbox = BoundingBox(longitude=(0, 10), latitude=(0, 10))
    md_bbox = Metadatum(:u_velocity; dataset=ECCO4Monthly(), region=bbox)
    @test location(md_bbox) == (Face, Center, Center)
end

@testset "native_grid with Column region" begin
    col = Column(35.1, 50.1)
    md = Metadatum(:temperature; dataset=ECCO4Monthly(), region=col)
    grid = native_grid(md)

    @test grid isa RectilinearGrid
    @test topology(grid) == (Flat, Flat, Bounded)
    _, _, Nz, _ = size(md)
    @test size(grid) == (1, 1, Nz)
end

@testset "native_grid without region" begin
    md = Metadatum(:temperature; dataset=ECCO4Monthly())
    grid = native_grid(md)

    @test grid isa LatitudeLongitudeGrid
    Nx, Ny, Nz, _ = size(md)
    @test size(grid) == (Nx, Ny, Nz)
end

@testset "native_grid with BoundingBox region" begin
    bbox = BoundingBox(longitude=(0, 10), latitude=(0, 10))
    md = Metadatum(:temperature; dataset=ECCO4Monthly(), region=bbox)
    grid = native_grid(md)

    @test grid isa LatitudeLongitudeGrid
    # Grid should be smaller than the full global grid
    Nx_full, Ny_full, _, _ = size(md)
    Nx, Ny, Nz = size(grid)
    @test Nx < Nx_full
    @test Ny < Ny_full

    # Sub-360° bbox must be Bounded in x (not Periodic) so halos don't wrap.
    @test topology(grid)[1] == Bounded

    # 360°-spanning bbox keeps Periodic in x.
    bbox_full = BoundingBox(longitude=(-180, 180), latitude=(-30, 30))
    md_full = Metadatum(:temperature; dataset=ECCO4Monthly(), region=bbox_full)
    @test topology(native_grid(md_full))[1] == Periodic

    # Latitude-only restriction: longitude is unrestricted, x stays Periodic.
    bbox_lat = BoundingBox(latitude=(-30, 30))
    md_lat = Metadatum(:temperature; dataset=ECCO4Monthly(), region=bbox_lat)
    @test topology(native_grid(md_lat))[1] == Periodic

    # ERA5 uses a 0°..360° native longitude convention. A bbox specified as -110°..30° crosses that
    # seam; `native_grid` restricts to the enclosing native cells (250°..390°) but relabels the grid
    # back into the bbox's own convention (-110°..30°), keeping the full 140° span across the seam.
    seam_bbox = BoundingBox(longitude=(-110, 30), latitude=(-25, 35))
    seam_md = Metadatum(:temperature; dataset=ERA5HourlySingleLevel(),
                        date=DateTime(2004, 12, 27), region=seam_bbox)
    seam_grid = native_grid(seam_md)
    seam_λ = λnodes(seam_grid, Center())

    @test length(seam_λ) == 561
    @test first(seam_λ) == -110.0f0
    @test last(seam_λ) == 30.0f0
    @test topology(seam_grid)[1] == Bounded
end

@testset "Metadata region keyword" begin
    # region keyword replaces BoundingBox
    col = Column(35.1, 50.1)
    md = Metadatum(:temperature; dataset=ECCO4Monthly(), region=col)
    @test md.region isa Column
    @test md.region.longitude == 35.1

    bbox = BoundingBox(longitude=(0, 10), latitude=(0, 10))
    md2 = Metadatum(:temperature; dataset=ECCO4Monthly(), region=bbox)
    @test md2.region isa BoundingBox

    md3 = Metadatum(:temperature; dataset=ECCO4Monthly())
    @test md3.region === nothing
end

@testset "Metadata iteration propagates region" begin
    col = Column(35.1, 50.1)
    md = Metadata(:temperature; dataset=ECCO4Monthly(), region=col,
                  start_date=DateTime(1992, 1, 1), end_date=DateTime(1992, 3, 1))

    for sub_md in md
        @test sub_md.region === col
    end

    @test first(md).region === col
    @test last(md).region === col
    @test md[1].region === col
end

@testset "restrict() center-brackets the bbox on native interfaces" begin
    # Uniform interfaces (cell centers at 0.5, 1.5, …): edges on cell boundaries
    # get one extra cell at each end so a native center brackets them.
    interfaces = collect(0.0:1.0:10.0)
    sliced, rN = restrict((2.0, 6.0), interfaces, 10)
    @test sliced == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    @test rN == 6

    # Uniform interfaces, off-grid bbox: snap outward to the surrounding
    # native cells so the result is a superset of the request.
    sliced, rN = restrict((2.5, 6.5), interfaces, 10)
    @test sliced[1]   ≤ 2.5
    @test sliced[end] ≥ 6.5
    @test rN == length(sliced) - 1

    # Stretched interfaces (cells get wider): snapping must return the
    # actual native interfaces, not a 2-tuple of the user's bbox.
    stretched = [0.0, 0.5, 1.5, 3.0, 5.5, 9.5, 15.0]
    sliced, rN = restrict((1.0, 6.0), stretched, length(stretched) - 1)
    @test sliced == [0.5, 1.5, 3.0, 5.5, 9.5]
    @test rN == 4

    # Out-of-range bbox is clamped, not crashed.
    sliced, rN = restrict((-100.0, 100.0), stretched, length(stretched) - 1)
    @test sliced == stretched
    @test rN == length(stretched) - 1

    # 2-tuple endpoints on a uniform native grid: center-bracketing widens the
    # boundary-aligned endpoints by one cell at each end.
    sliced, rN = restrict((120, 240), (0, 360), 360)
    @test sliced == (119, 241)
    @test rN == 122
end

@testset "download_cache honors NUMERICALEARTH_DATA_DIRECTORY" begin
    saved = get(ENV, "NUMERICALEARTH_DATA_DIRECTORY", nothing)
    try
        # Without the variable, caching falls back to a Scratch.jl space.
        delete!(ENV, "NUMERICALEARTH_DATA_DIRECTORY")
        @test occursin("scratchspaces", download_cache("JRA55"))

        # With the variable, data is cached under a per-key subdirectory of it.
        data_directory = mktempdir()
        ENV["NUMERICALEARTH_DATA_DIRECTORY"] = data_directory
        cache = download_cache("JRA55")
        @test cache == joinpath(data_directory, "JRA55")
        @test isdir(cache)
    finally
        if saved === nothing
            delete!(ENV, "NUMERICALEARTH_DATA_DIRECTORY")
        else
            ENV["NUMERICALEARTH_DATA_DIRECTORY"] = saved
        end
    end
end

@testset "Monthly-mean sample windows" begin
    # The dates these files carry in their own `time` variable: the center of each month,
    # which is a half day later in a 31-day month than in a 30-day one.
    en4_centers = (1 => DateTime(2010, 1, 16, 12), 2 => DateTime(2010, 2, 15),
                   6 => DateTime(2010, 6, 16),     7 => DateTime(2010, 7, 16, 12))

    for (month, center) in en4_centers
        first_of_month = DateTime(2010, month, 1)
        metadatum = Metadatum(:temperature; dataset = EN4Monthly(), date = first_of_month)
        @test sample_window(metadatum) == (first_of_month, first_of_month + Month(1))
        @test time_window_offset(metadatum) == Dates.value(Second(center - first_of_month))
    end

    ecco = Metadatum(:v_velocity; dataset = ECCO4Monthly(), date = DateTime(1993, 1, 1))
    @test sample_window(ecco) == (DateTime(1993, 1, 1), DateTime(1993, 2, 1))
    @test time_window_offset(ecco) == Dates.value(Second(DateTime(1993, 1, 16, 12) - ecco.dates))

    # A twelve-month axis then reproduces the time coordinates the files themselves carry,
    # in days from the first of January.
    metadata = Metadata(:temperature; dataset = EN4Monthly(),
                        dates = [DateTime(2010, m, 1) for m in 1:12])
    @test native_times(metadata) ./ 86400 == [15.5, 45.0, 74.5, 105.0, 135.5, 166.0,
                                              196.5, 227.5, 258.0, 288.5, 319.0, 349.5]

    # An instantaneous product has no window and is left where its stamp puts it.
    hourly = Metadatum(:temperature; dataset = ERA5HourlySingleLevel(), date = DateTime(2020, 4, 1))
    @test sample_window(hourly) == (DateTime(2020, 4, 1), DateTime(2020, 4, 1))
    @test time_window_offset(hourly) == 0
    @test native_times(Metadata(:temperature; dataset = ERA5HourlySingleLevel(),
                                dates = [DateTime(2020, 4, 1, h) for h in 0:2])) == [0, 3600, 7200]

    # Every monthly-mean product spans the calendar month its file is named for, whether the
    # stamp is midnight on the first (most of them) or noon (ECCO Darwin).
    monthly_datasets = (ECCO2Monthly() => :temperature,
                        ECCO4Monthly() => :temperature,
                        ECCO2DarwinMonthly() => :dissolved_inorganic_carbon,
                        ECCO4DarwinMonthly() => :dissolved_inorganic_carbon,
                        EN4Monthly() => :temperature,
                        AVISOMonthly() => :sea_level_anomaly,
                        GLORYSMonthly() => :temperature,
                        WOAMonthly() => :temperature,
                        ERA5MonthlySingleLevel() => :temperature,
                        ERA5MonthlyPressureLevels() => :temperature,
                        ERA5MonthlyLand() => :temperature)

    for (dataset, name) in monthly_datasets, stamp in (DateTime(2010, 7, 1), DateTime(2010, 7, 1, 12))
        metadatum = Metadatum(name; dataset, date = stamp)
        @test sample_window(metadatum) == (DateTime(2010, 7, 1), DateTime(2010, 8, 1))
        @test window_center(metadatum) == DateTime(2010, 7, 16, 12)
    end

    # ERA5-Land monthly nodes sit mid-month, in phase with the single-level product.
    land = Metadatum(:temperature; dataset = ERA5MonthlyLand(), date = DateTime(2010, 7, 1))
    single_level = Metadatum(:temperature; dataset = ERA5MonthlySingleLevel(), date = DateTime(2010, 7, 1))
    @test time_window_offset(land) == 15.5 * 86400
    @test time_window_offset(land) == time_window_offset(single_level)
end

@testset "ERA5 accumulation windows" begin
    # ERA5 accumulations and mean rates cover the hour ending at the stamp, so their windows
    # run backwards from it and their nodes sit half an hour before it.
    for name in (:total_precipitation, :evaporation, :mean_evaporation_rate,
                 :downwelling_shortwave_radiation, :downwelling_longwave_radiation,
                 :mean_surface_momentum_flux_x, :mean_surface_momentum_flux_y,
                 :mean_surface_sensible_heat_flux, :mean_surface_latent_heat_flux)

        metadatum = Metadatum(name; dataset = ERA5HourlySingleLevel(), date = DateTime(2020, 4, 1, 13))
        @test sample_window(metadatum) == (DateTime(2020, 4, 1, 12), DateTime(2020, 4, 1, 13))
        @test time_window_offset(metadatum) == -1800
    end

    for name in (:temperature, :surface_pressure, :eastward_velocity, :significant_wave_height)
        metadatum = Metadatum(name; dataset = ERA5HourlySingleLevel(), date = DateTime(2020, 4, 1, 13))
        @test time_window_offset(metadatum) == 0
    end

    radiation = Metadata(:downwelling_shortwave_radiation; dataset = ERA5HourlySingleLevel(),
                         dates = [DateTime(2020, 4, 1, h) for h in 1:3])
    @test native_times(radiation) == [-1800, 1800, 5400]
end

@testset "JRA55 sample windows" begin
    # The repeat-year files label each mean with the start of its interval: the three-hourly
    # fluxes, radiation, and precipitation run forwards three hours from the stamp, the daily
    # river and iceberg fluxes forwards a day.
    for name in (:rain_freshwater_flux, :snow_freshwater_flux,
                 :downwelling_longwave_radiation, :downwelling_shortwave_radiation)

        metadatum = Metadatum(name; dataset = RepeatYearJRA55(), date = DateTime(1990, 4, 1, 12))
        @test sample_window(metadatum) == (DateTime(1990, 4, 1, 12), DateTime(1990, 4, 1, 15))
        @test time_window_offset(metadatum) == 1.5 * 3600
    end

    for name in (:river_freshwater_flux, :iceberg_freshwater_flux)
        metadatum = Metadatum(name; dataset = RepeatYearJRA55(), date = DateTime(1990, 4, 1))
        @test sample_window(metadatum) == (DateTime(1990, 4, 1), DateTime(1990, 4, 2))
        @test time_window_offset(metadatum) == 12 * 3600
    end

    # The state variables are instantaneous in both products.
    for dataset in (RepeatYearJRA55(), MultiYearJRA55()),
        name in (:temperature, :specific_humidity, :eastward_velocity, :sea_level_pressure)

        metadatum = Metadatum(name; dataset, date = DateTime(1990, 4, 1, 12))
        @test time_window_offset(metadatum) == 0
    end

    # The multi-year files already stamp their means at the window center, so every variable
    # sits at its node as labelled.
    for name in (:rain_freshwater_flux, :downwelling_shortwave_radiation, :river_freshwater_flux)
        metadatum = Metadatum(name; dataset = MultiYearJRA55(), date = DateTime(1990, 4, 1, 1, 30))
        @test time_window_offset(metadatum) == 0
    end
end

@testset "Every dataset with a time axis declares a sample window" begin
    # `sample_window` has no fallback, so a new adapter that forgets it raises a `MethodError`
    # instead of silently placing window averages at their stamps.
    @test !hasmethod(sample_window, Tuple{Any})

    datasets = Dict(AVISODaily() => :sea_level_anomaly,
                    AVISOMonthly() => :sea_level_anomaly,
                    ECCO2Daily() => :temperature,
                    ECCO2Monthly() => :temperature,
                    ECCO4Monthly() => :temperature,
                    EN4Monthly() => :temperature,
                    ERA5HourlySingleLevel() => :temperature,
                    ERA5MonthlySingleLevel() => :temperature,
                    ERA5HourlyPressureLevels() => :temperature,
                    ERA5MonthlyPressureLevels() => :temperature,
                    ERA5HourlyLand() => :temperature,
                    ERA5MonthlyLand() => :temperature,
                    GLORYSDaily() => :temperature,
                    GLORYSMonthly() => :temperature,
                    RepeatYearJRA55() => :temperature,
                    MultiYearJRA55() => :temperature,
                    WOAMonthly() => :temperature)

    for (dataset, name) in datasets
        metadatum = Metadatum(name; dataset, date = DateTime(2010, 7, 1))
        @test hasmethod(sample_window, Tuple{typeof(metadatum)})
    end
end

@testset "Time coverage of window-averaged series" begin
    monthly = Metadata(:temperature; dataset = EN4Monthly(),
                       dates = [DateTime(2010, m, 1) for m in 1:12])

    # Twelve monthly windows tile a year exactly, and the nodes fall half a month inside it.
    @test sample_window_span(monthly) == Dates.value(Second(DateTime(2011, 1, 1) - DateTime(2010, 1, 1)))
    @test uncovered_time_gaps(monthly) == (15.5 * 86400, 15.5 * 86400)

    # An instantaneous product covers exactly its own nodes, and leaves the cyclical period
    # to be inferred from their spacing.
    hourly = Metadata(:temperature; dataset = ERA5HourlySingleLevel(),
                      dates = [DateTime(2020, 4, 1, h) for h in 0:2])
    @test isnothing(sample_window_span(hourly))
    @test uncovered_time_gaps(hourly) == (0, 0)

    # `Cyclical` fills the gaps by wrapping and says so; `Linear` extrapolates and refuses;
    # `Clamp` was asked to hold the end values.
    @test_logs (:warn, r"interpolates only between") validate_time_coverage(monthly, Cyclical(1.0))
    @test_throws ArgumentError validate_time_coverage(monthly, LinearTimeIndexing())
    @test isnothing(validate_time_coverage(monthly, Clamp()))

    # None of it applies to a product without windows, or to a single sample, which is
    # constant in time.
    @test isnothing(validate_time_coverage(hourly, Cyclical(1.0)))
    @test isnothing(validate_time_coverage(hourly, LinearTimeIndexing()))

    single = Metadatum(:temperature; dataset = EN4Monthly(), date = DateTime(2010, 7, 1))
    @test isnothing(validate_time_coverage(single, Cyclical(1.0)))
    @test isnothing(validate_time_coverage(single, LinearTimeIndexing()))

    # A single window still gives a cyclical period, where the node spacing Oceananigans
    # would infer does not exist.
    @test sample_window_span(single) == Dates.value(Second(DateTime(2010, 8, 1) - DateTime(2010, 7, 1)))
end

@testset "nan_convert_missing" begin
    @test isnan(DataWrangling.nan_convert_missing(Float32, missing, missing))
    @test isnan(DataWrangling.nan_convert_missing(Float32, -999, -999))
    @test DataWrangling.nan_convert_missing(Float32, 1.0, missing) === 1f0
    @test DataWrangling.nan_convert_missing(Float32, 1.0, -999) === 1f0
    @test DataWrangling.nan_convert_missing(Float32, Inf, -999) === Inf32
end
