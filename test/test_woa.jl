include("runtests_setup.jl")
include("download_utils.jl")

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using CUDA: @allowscalar
using NumericalEarth.WOA
using NumericalEarth.DataWrangling: NearestNeighborInpainting, metadata_path, download_dataset
using Oceananigans.Architectures: on_architecture
using WorldOceanAtlasTools

inpainting = NearestNeighborInpainting(10)

# Ensure a WOA Metadatum file is on disk, trying NOAA first 
# and falling back to the NumericalEarthArtifacts mirror.
function ensure_woa_file(metadatum; label)
    filepath = metadata_path(metadatum)
    download_dataset_with_fallback(filepath; dataset_name=label) do
        download_dataset(metadatum)
    end
    return filepath
end

for arch in test_architectures
    A = typeof(arch)

    @testset "$A WOA Annual metadata tests" begin
        @info "Running WOA Annual metadata tests on $A..."

        @testset "WOA Annual download and Field creation" begin
            for name in (:temperature, :salinity)
                metadata = Metadatum(name; dataset=WOAAnnual())
                filepath = ensure_woa_file(metadata; label="WOA Annual $name")
                @test isfile(filepath)

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
                metadatum = Metadatum(name; dataset=WOAAnnual())
                ensure_woa_file(metadatum; label="WOA Annual $name")
                @test begin
                    set!(field, metadatum; inpainting)
                    true
                end
            end
        end

        @testset "WOA Annual inpainting" begin
            for name in (:temperature, :salinity)
                metadatum = Metadatum(name; dataset=WOAAnnual())
                ensure_woa_file(metadatum; label="WOA Annual $name")

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
                filepath = ensure_woa_file(metadata; label="WOA Monthly $name")
                @test isfile(filepath)

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
                metadatum = Metadatum(name; dataset=WOAMonthly())
                ensure_woa_file(metadatum; label="WOA Monthly $name")
                @test begin
                    set!(field, metadatum; inpainting)
                    true
                end
            end
        end
    end
end
