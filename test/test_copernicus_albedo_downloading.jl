include("runtests_setup.jl")

using NumericalEarth.DataWrangling: BoundingBox, Metadatum
using Oceananigans.Grids: topology, Bounded
using Dates: DateTime

# Network-gated: downloads two ~global CGLS files (hundreds of MB). Excluded from
# the default suite in runtests.jl, mirroring the other *_downloading tests.
@testset "Copernicus land albedo download and regional read" begin
    dataset = CopernicusAlbedo()
    region = BoundingBox(longitude = (-114, -109), latitude = (33, 38))
    date = DateTime(2019, 7, 10)

    metadatum = Metadatum(:albedo; dataset, region, date)
    α = Field(metadatum)
    values = Array(interior(α))

    @test any(isfinite, values)
    valid = filter(!isnan, vec(values))
    @test !isempty(valid)
    @test all(x -> 0 ≤ x ≤ 1, valid)

    # Sub-360° window must be Bounded in x so halos do not wrap.
    @test topology(α.grid)[1] == Bounded
end
