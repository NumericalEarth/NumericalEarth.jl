include("runtests_setup.jl")

using NCDatasets: NCDataset

# Release 132 of the GEUS solid ice discharge has 266 gates, sampled from 1986 onward.
@testset "Downloading GEUS solid ice discharge" begin
    dir = mktempdir()
    path = download(GEUSSolidIceDischarge(version = 132), dir)
    @test isfile(path)

    number_of_gates, first_sample = NCDataset(ds -> (length(ds["mean_lon"]), first(ds["time"][:])), path)
    @test number_of_gates == 266
    @test year(first_sample) == 1986
end
