include("runtests_setup.jl")

@testset "Downloading the NOAA marine boundary layer reference" begin
    dataset = NOAAMarineBoundaryLayer()
    dir = mktempdir()

    for name in (:carbon_dioxide, :methane, :nitrous_oxide, :sulfur_hexafluoride)
        metadatum = Metadatum(name; dataset, date=last(all_dates(dataset, name)), dir)
        download(metadatum)
        @test isfile(metadata_path(metadatum))
    end

    # January carbon dioxide is higher in the north than in the south
    field = Field(Metadatum(:carbon_dioxide; dataset, date=DateTime(2020, 1, 1), dir))
    values = interior(field)[1, :, 1]
    @test all(400 .< values .< 420)
    @test values[end] > values[1]
end
