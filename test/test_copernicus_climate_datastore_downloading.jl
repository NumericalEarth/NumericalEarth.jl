include("runtests_setup.jl")
include("download_utils.jl")

using Dates
using NumericalEarth.DataWrangling: metadata_path, BoundingBox
using NumericalEarth.DataWrangling.ERA5: ERA5HourlyLand, ERA5YearlySingleLevel, ERA5MonthlySingleLevel

# Requires CDS API credentials in ~/.cdsapirc or the CDSAPI_URL/CDSAPI_KEY
# environment variables (see https://cds.climate.copernicus.eu/how-to-api).
# Excluded from per-commit CI in runtests.jl; run manually with credentials set,
# or via a separately-triggered downloading CI job (see PR #440 / issue #480).

# A tiny land region (near Rome), kept small to minimize CDS request/processing time.
const era5_region = BoundingBox(longitude=(12, 13), latitude=(41, 42))

# ERA5-Land masks non-land grid cells (ocean, and large lakes) as NaN, so the region above
# (which includes Tyrrhenian coastline) isn't safe for a strict all-isfinite check. Use a
# small landlocked inland region instead (Umbria, Italy — no coast, avoids Lake Trasimeno).
const era5_land_region = BoundingBox(longitude=(12.5, 13.0), latitude=(42.8, 43.2))

@testset "ERA5-Land live download" begin
    dataset = ERA5HourlyLand()
    variable = :skin_temperature
    date = DateTime(2020, 6, 15, 12)

    metadatum = Metadatum(variable; dataset, region=era5_land_region, date)
    filepath = metadata_path(metadatum)
    isfile(filepath) && rm(filepath; force=true)

    download(metadatum)
    @test isfile(filepath)

    ψ = Field(metadatum, CPU())
    @allowscalar begin
        @test all(isfinite, interior(ψ))
        @test !all(iszero, interior(ψ))
    end

    rm(filepath; force=true)
end

@testset "ERA5YearlySingleLevel FieldTimeSeries spans a year boundary" begin
    # Before the fix, build_filename collapsed every requested date to the FIRST
    # date's year file — reading Jan 1's timestamp from the 2020 file would find
    # no matching time and error() loudly. This confirms the real fix reads each
    # date from its own (correct) yearly file with real downloaded data.
    dataset = ERA5YearlySingleLevel()
    variable = :temperature
    dates = [DateTime(2020, 12, 31, 22), DateTime(2020, 12, 31, 23),
             DateTime(2021, 1, 1, 0),    DateTime(2021, 1, 1, 1)]

    mktempdir() do dir
        metadata = Metadata(variable; dataset, dates, region=era5_region, dir)
        fts = FieldTimeSeries(metadata, CPU())

        @test size(fts, 4) == length(dates)
        @allowscalar begin
            for n in eachindex(dates)
                @test all(isfinite, interior(fts[n]))
                @test !all(iszero, interior(fts[n]))
            end
        end
    end
end

@testset "ERA5MonthlySingleLevel FieldTimeSeries spans a year boundary" begin
    # The dangerous variant: each monthly file holds a SINGLE timestep, so the
    # pre-fix collapse silently returned December's data for January too (no
    # error — retrieve_data just reads the only timestep in whatever file it's
    # given). The `v1 != v2` check below is the direct regression guard.
    dataset = ERA5MonthlySingleLevel()
    variable = :temperature
    dates = [DateTime(2020, 12, 1), DateTime(2021, 1, 1)]

    mktempdir() do dir
        metadata = Metadata(variable; dataset, dates, region=era5_region, dir)
        fts = FieldTimeSeries(metadata, CPU())

        @test size(fts, 4) == length(dates)
        @allowscalar begin
            v1 = Array(interior(fts[1]))
            v2 = Array(interior(fts[2]))
            @test all(isfinite, v1) && all(isfinite, v2)
            @test !all(iszero, v1) && !all(iszero, v2)
            @test v1 != v2
        end
    end
end
