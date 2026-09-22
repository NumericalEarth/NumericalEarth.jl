include("runtests_setup.jl")

using ArchGDAL  # loads NumericalEarthArchGDALExt (the OGR read + rasterize path)
using NumericalEarth.DataWrangling: BoundingBox, Metadatum, native_region_grid
using NumericalEarth.DataWrangling.GloBFP3D: globfp3d_native_cell_size

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

    # Reducing the same region in latitude bands must reproduce the single pass exactly.
    Δ = globfp3d_native_cell_size(dataset)
    raster = native_region_grid(region, Δ, Δ)
    maximum_raster_cells = raster.Nx * (raster.Ny ÷ 4)
    @test raster.Nx * raster.Ny > maximum_raster_cells    # the limit really forces bands

    banded = building_morphometry(target_grid; dataset, region, maximum_raster_cells)
    for name in keys(morphometry)
        @test Array(interior(banded[name], :, :, 1)) == Array(interior(morphometry[name], :, :, 1))
    end
end

@testset "3D-GloBFP region with no tiles" begin
    dataset = GlobalBuildingFootprints3D()
    # Open ocean south of Hawaii: the tile catalog covers built-up land only, so no tile
    # intersects and `building_morphometry` errors instead of returning empty fields.
    region = BoundingBox(longitude = (-150.0, -149.99), latitude = (10.0, 10.01))
    target_grid = LatitudeLongitudeGrid(CPU(), Float64; size = (2, 2),
                                        longitude = region.longitude, latitude = region.latitude,
                                        topology = (Bounded, Bounded, Flat))

    @test_throws ErrorException building_morphometry(target_grid; dataset, region)
end
