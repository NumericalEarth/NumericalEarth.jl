include("runtests_setup.jl")
include("download_utils.jl")

using CopernicusMarine

using NumericalEarth.DataWrangling: BoundingBox, is_three_dimensional, z_interfaces
using NumericalEarth.DataWrangling.GLORYS: GLORYSDaily
using Oceananigans.Fields: location

@testset "Downloading GLORYS data" begin
    variables = (:temperature, :salinity, :u_velocity, :v_velocity, :free_surface)
    region = BoundingBox(longitude=(200, 202), latitude=(35, 37))
    dataset = GLORYSDaily()
    for variable in variables
        metadatum = Metadatum(variable; dataset, region)
        filepath = NumericalEarth.DataWrangling.metadata_path(metadatum)
        isfile(filepath) && rm(filepath; force=true)
        NumericalEarth.DataWrangling.download_dataset(metadatum)
        @test isfile(filepath)
    end
end

@testset "Download and set GLORYS free_surface" begin
    for arch in test_architectures
        region = BoundingBox(longitude=(200, 202), latitude=(35, 37))
        dataset = GLORYSDaily()
        md = Metadatum(:free_surface; dataset, region)

        @test !is_three_dimensional(md)
        @test location(md) === (Center, Center, Nothing)
        @test z_interfaces(md) === (-1.0, 0.0)

        source = Field(md, arch; inpainting=nothing)
        @test source isa Field
        @test size(interior(source), 3) == 1

        target_grid = LatitudeLongitudeGrid(arch;
                                            size = (4, 4, 3),
                                            longitude = (200.5, 201.5),
                                            latitude = (35.5, 36.5),
                                            z = (-1000, 0))
        target = CenterField(target_grid)
        set!(target, md; inpainting=nothing)

        interior_target = Array(interior(target))
        @test all(isfinite.(interior_target))
    end
end
