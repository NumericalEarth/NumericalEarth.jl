include("runtests_setup.jl")

using ArchGDAL  # loads NumericalEarthArchGDALExt (the windowed /vsicurl COG reader)
using NumericalEarth.DataWrangling: BoundingBox, Metadatum, native_grid
using Oceananigans.Grids: topology, Bounded

# Network-gated: reads the 30 m global COGs over /vsicurl (anonymous, no credentials).
# Excluded from the default suite in runtests.jl, mirroring the other *_downloading tests.
@testset "OpenLandMap-soilDB windowed download" begin
    dataset = OpenLandMapSoilDB()
    # Cropland near Ames, Iowa: real soil retrievals, no ice or sand-desert mask.
    region = BoundingBox(longitude = (-93.70, -93.60), latitude = (41.90, 42.00))

    metadatum = Metadatum(:clay_fraction; dataset, region)
    grid = native_grid(metadatum)
    clay_fraction = Field(metadatum)
    values = Array(interior(clay_fraction))

    @test size(values)[1:2] == (size(grid, 1), size(grid, 2))
    @test size(values, 3) == 3  # the three native depth intervals

    valid = filter(!isnan, vec(values))
    @test !isempty(valid)
    @test all(x -> 0 ≤ x ≤ 1, valid)     # mass fraction in kg/kg after WeightPercent
    @test length(unique(valid)) > 1      # a real window, not a constant fill

    # Sub-360° window must be Bounded in x so halos do not wrap.
    @test topology(clay_fraction.grid)[1] == Bounded

    bulk_density = Field(Metadatum(:bulk_density; dataset, region))
    dense = filter(!isnan, vec(Array(interior(bulk_density))))
    @test !isempty(dense)
    @test all(x -> 500 ≤ x ≤ 2200, dense)  # fine-earth bulk density in kg/m³
end
