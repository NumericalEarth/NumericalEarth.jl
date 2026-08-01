include("runtests_setup.jl")

using ArchGDAL  # loads NumericalEarthArchGDALExt (the OGR read + rasterize path)
using NumericalEarth.DataWrangling: BoundingBox, Metadatum

# Network-gated: downloads the figshare tile catalog plus one 3D-GloBFP tile archive
# (hundreds of MB). Excluded from the default suite in runtests.jl, mirroring the
# other *_downloading tests.
@testset "3D-GloBFP tile download and morphometry" begin
    dataset = GlobalBuildingFootprints3D()
    # ~1 km box over the City of London: dense enough that every field is populated,
    # heterogeneous enough that σʰ and λᶠ are nonzero.
    region = BoundingBox(longitude = (-0.09, -0.08), latitude = (51.51, 51.52))

    building_height = Field(Metadatum(:building_height; dataset, region), CPU())
    heights = Array(interior(building_height, :, :, 1))
    @test any(>(0), heights)      # footprints were burned into the raster
    @test all(≥(0), heights)
    @test maximum(heights) < 400  # comfortably above the City's tallest tower, ~278 m

    target_grid = LatitudeLongitudeGrid(CPU(), Float64; size = (4, 4),
                                        longitude = region.longitude, latitude = region.latitude,
                                        topology = (Bounded, Bounded, Flat))
    morphometry = building_morphometry(target_grid; dataset, region)

    λᵖ   = Array(interior(morphometry.plan_area_index, :, :, 1))
    h    = Array(interior(morphometry.mean_building_height, :, :, 1))
    hᵐᵃˣ = Array(interior(morphometry.maximum_building_height, :, :, 1))
    σʰ   = Array(interior(morphometry.building_height_deviation, :, :, 1))
    λᶠ   = Array(interior(morphometry.frontal_area_index, :, :, 1))

    @test all(0 .< λᵖ .≤ 1)
    @test all(hᵐᵃˣ .≥ h .> 0)
    @test any(>(0), σʰ)
    @test all(λᶠ .≥ 0)
    @test Array(interior(morphometry.gross_building_height, :, :, 1)) ≈ λᵖ .* h
end
