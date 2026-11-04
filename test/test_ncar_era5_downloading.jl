include("runtests_setup.jl")

using Dates
using Statistics

@testset "ERA5 from NCARArchive" begin
    dataset = ERA5HourlySingleLevel(archive=NCARArchive())
    region = BoundingBox(longitude=(-10, 10), latitude=(30, 40))
    dir = mktempdir()

    # Crosses the mid-month forecast-file seam at 06 UTC on the 16th
    dates = [DateTime(2024, 6, 16, 0), DateTime(2024, 6, 16, 12)]

    shortwave = Metadata(:downwelling_shortwave_radiation; dataset, dates, region, dir)
    midnight, noon = [mean(interior(Field(metadatum))) for metadatum in shortwave]
    @test midnight == 0
    @test noon > 800

    temperature = Field(Metadatum(:temperature; dataset, date=DateTime(2024, 6, 16, 12), region, dir))
    @test all(250 .< interior(temperature) .< 330)
end
