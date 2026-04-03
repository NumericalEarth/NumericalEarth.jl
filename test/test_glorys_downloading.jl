include("runtests_setup.jl")

using CopernicusMarine
using NumericalEarth.DataWrangling: metadata_path, download_dataset,
                                    BoundingBox, Column, Nearest, Linear
using NumericalEarth.DataWrangling.GLORYS: GLORYSDaily

@testset "Downloading GLORYS data" begin
    variables = (:temperature, :salinity, :u_velocity, :v_velocity)
    region = BoundingBox(longitude=(200, 202), latitude=(35, 37))
    dataset = GLORYSDaily()

    @testset "BoundingBox download" begin
        for variable in variables
            metadatum = Metadatum(variable; dataset, region)
            filepath = metadata_path(metadatum)
            isfile(filepath) && rm(filepath; force=true)
            download_dataset(metadatum)
            @test isfile(filepath)
        end
    end

    @testset "Column Nearest download" begin
        col = Column(201.0, 36.0; interpolation=Nearest())
        metadatum = Metadatum(:temperature; dataset, region=col)
        filepath = metadata_path(metadatum)
        isfile(filepath) && rm(filepath; force=true)
        download_dataset(metadatum)
        @test isfile(filepath)
        rm(filepath; force=true)
    end

    @testset "Column Linear download" begin
        col = Column(201.0, 36.0; interpolation=Linear())
        metadatum = Metadatum(:temperature; dataset, region=col)
        filepath = metadata_path(metadatum)
        isfile(filepath) && rm(filepath; force=true)
        download_dataset(metadatum)
        @test isfile(filepath)
        rm(filepath; force=true)
    end

    for arch in test_architectures
        A = typeof(arch)
        @testset "GLORYS Field with BoundingBox on $A" begin
            metadatum = Metadatum(:temperature; dataset, region)
            filepath = metadata_path(metadatum)
            isfile(filepath) || download_dataset(metadatum)
            field = Field(metadatum, arch)
            @test field isa Field
            @allowscalar @test any(!=(0), interior(field))
        end
    end
end
