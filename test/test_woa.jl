include("runtests_setup.jl")

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using CUDA: @allowscalar
using NumericalEarth.WOA
using NumericalEarth.DataWrangling: NearestNeighborInpainting, metadata_path, download_dataset
using Oceananigans.Architectures: on_architecture
using WorldOceanAtlasTools

inpainting = NearestNeighborInpainting(10)

for arch in test_architectures
    A = typeof(arch)

    @testset "$A WOA Annual metadata tests" begin
        @info "Running WOA Annual metadata tests on $A..."

        @testset "WOA Annual download and Field creation" begin
            for name in (:temperature, :salinity)
                metadata = Metadatum(name; dataset=WOAAnnual())
                download_dataset(metadata)
                @test isfile(metadata_path(metadata))

                field = Field(metadata, arch; inpainting=NearestNeighborInpainting(2))
                @test field isa Field

                # Check inpainted data was cached
                datapath = NumericalEarth.DataWrangling.inpainted_metadata_path(metadata)
                @test isfile(datapath)
            end
        end

        @testset "Setting a field from WOA Annual" begin
            grid = LatitudeLongitudeGrid(arch;
                                         size = (10, 10, 10),
                                         latitude = (-60, -40),
                                         longitude = (10, 15),
                                         z = (-200, 0))
            field = CenterField(grid)

            for name in (:temperature, :salinity)
                @test begin
                    set!(field, Metadatum(name; dataset=WOAAnnual()); inpainting)
                    true
                end
            end
        end

        @testset "WOA Annual inpainting" begin
            for name in (:temperature, :salinity)
                metadatum = Metadatum(name; dataset=WOAAnnual())

                grid = LatitudeLongitudeGrid(arch,
                                             size = (20, 20, 10),
                                             latitude = (-75, 75),
                                             longitude = (0, 360),
                                             z = (-4000, 0),
                                             halo = (6, 6, 6))

                fully_inpainted = CenterField(grid)
                partially_inpainted = CenterField(grid)

                set!(fully_inpainted, metadatum; inpainting=NearestNeighborInpainting(Inf))
                set!(partially_inpainted, metadatum; inpainting=NearestNeighborInpainting(1))

                fully_interior = on_architecture(CPU(), interior(fully_inpainted))
                partially_interior = on_architecture(CPU(), interior(partially_inpainted))

                @test all(fully_interior .!= 0)
                @test any(partially_interior .== 0)
            end
        end
    end

    @testset "$A WOA Monthly metadata tests" begin
        @info "Running WOA Monthly metadata tests on $A..."

        @testset "WOA Monthly download and Field creation" begin
            for name in (:temperature, :salinity)
                metadata = Metadatum(name; dataset=WOAMonthly())
                download_dataset(metadata)
                @test isfile(metadata_path(metadata))

                field = Field(metadata, arch; inpainting=NearestNeighborInpainting(2))
                @test field isa Field
            end
        end

        @testset "Setting a field from WOA Monthly" begin
            grid = LatitudeLongitudeGrid(arch;
                                         size = (10, 10, 10),
                                         latitude = (-60, -40),
                                         longitude = (10, 15),
                                         z = (-200, 0))
            field = CenterField(grid)

            for name in (:temperature, :salinity)
                @test begin
                    set!(field, Metadatum(name; dataset=WOAMonthly()); inpainting)
                    true
                end
            end
        end
    end
end
