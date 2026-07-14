include("runtests_setup.jl")

using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, all_dates, dataset_variable_name,
                                    is_three_dimensional
using Oceananigans: location

@testset "AVISO metadata tests" begin
    dataset = AVISOMonthly()
    region = BoundingBox(longitude=(200, 202), latitude=(35, 37))
    dates = DateTime(2020, 1, 1):Month(1):DateTime(2020, 2, 1)
    metadata = Metadata(:free_surface; dates, dataset, region)

    @test first(all_dates(AVISODaily(), :free_surface)) == DateTime(1993, 1, 1)
    @test last(all_dates(AVISODaily(), :free_surface)) == DateTime(2026, 1, 16)
    @test last(all_dates(dataset, :free_surface)) == DateTime(2025, 12, 1)
    @test dataset_variable_name(first(metadata)) == "adt"
    @test size(metadata) == (2880, 1440, 1, length(dates))

    datum = first(metadata)
    @test !is_three_dimensional(datum)
    @test location(datum) === (Center, Center, Nothing)
end
