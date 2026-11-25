include("runtests_setup.jl")

using NumericalEarth.DataWrangling: native_grid

# Network-gated and needs NASA Earthdata credentials (EARTHDATA_USERNAME, EARTHDATA_PASSWORD);
# downloads the 1.2 GB BedMachine Antarctica file once.
@testset "BedMachine Antarctica download" begin
    dataset = BedMachineAntarctica()
    # Trunk of Pine Island Glacier
    region = ProjectedBoundingBox(x = (-1.62e6, -1.59e6), y = (-2.8e5, -2.5e5))

    bed, thickness, surface, mask = [Array(interior(Field(Metadatum(name; dataset, region)), :, :, 1))
                                     for name in (:bottom_height, :ice_thickness, :surface_height, :ice_mask)]
    grid = native_grid(Metadatum(:ice_thickness; dataset, region))

    @test size(thickness) == (size(grid, 1), size(grid, 2))
    @test all(isfinite, bed)
    @test all(in(0:4), mask)
    @test any(==(2), mask)
    @test maximum(thickness) > 1000
    @test minimum(bed) < -500
    grounded = (mask .== 2) .& isfinite.(thickness) .& isfinite.(surface)
    @show maximum(abs.(surface[grounded] .- thickness[grounded] .- bed[grounded]))
end
