include("runtests_setup.jl")

using ArchGDAL  # loads NumericalEarthArchGDALExt (the OGR read + rasterize path)
using NumericalEarth.DataWrangling: BoundingBox, Metadatum

# Network-gated: downloads the figshare tile catalog plus one 3D-GloBFP tile archive
# (hundreds of MB). Excluded from the default suite in runtests.jl, mirroring the
# other *_downloading tests.
@testset "3D-GloBFP tile download and morphometry" begin
    dataset = GlobalBuildingFootprints3D()
    # ~1 km box over the City of London: dense enough that every field is populated,
    # heterogeneous enough that σH and λf are nonzero.
    region = BoundingBox(longitude = (-0.09, -0.08), latitude = (51.51, 51.52))

    building_height = Field(Metadatum(:building_height; dataset, region), CPU())
    heights = Array(interior(building_height, :, :, 1))
    @test any(>(0), heights)      # footprints were burned into the raster
    @test all(≥(0), heights)
    @test maximum(heights) < 400  # comfortably above the City's tallest tower, ~278 m

    target_grid = LatitudeLongitudeGrid(CPU(), Float64; size = (4, 4),
                                        longitude = region.longitude, latitude = region.latitude,
                                        topology = (Bounded, Bounded, Flat))
    m = building_morphometry(target_grid; dataset, region)

    λp   = Array(interior(m.built_up_fraction, :, :, 1))
    H    = Array(interior(m.mean_building_height, :, :, 1))
    Hmax = Array(interior(m.maximum_building_height, :, :, 1))
    σH   = Array(interior(m.building_height_std, :, :, 1))
    λf   = Array(interior(m.frontal_area_index, :, :, 1))

    @test all(0 .< λp .≤ 1)
    @test all(Hmax .≥ H .> 0)
    @test any(>(0), σH)
    @test all(λf .≥ 0)
    @test Array(interior(m.gross_building_height, :, :, 1)) ≈ λp .* H
end
