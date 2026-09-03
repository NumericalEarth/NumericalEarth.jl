using NumericalEarth
using NumericalEarth.DataWrangling
using NumericalEarth.SoilGrids
using Oceananigans

include("dataset_status.jl")

import Downloads

@testset "Metadata tests for SoilGrids" begin
    sg_mean = SoilGrids2()
    sg_sand = Metadatum(:sand_fraction, dataset=sg_mean)
    filepath = DataWrangling.metadata_path(sg_sand)

    @dataset_check "SoilGrids2" "sand_fraction" begin
        isfile(filepath) && rm(filepath; force=true)
        Downloads.download(sg_sand)
        isfile(filepath) || error("SoilGrids2 download produced no file at $(filepath)")
        filepath
    end

    @test isfile(filepath)
    for var in keys(SoilGrids.SoilGrids2_dataset_variable_names)
        sgmd = Metadatum(var, dataset=sg_mean)
        field = Field(sgmd)
        @test any(isfinite.(field))
    end
    for var in (:sand_fraction, :silt_fraction, :clay_fraction)
        sgmd = Metadatum(var, dataset=sg_mean)
        field = Field(sgmd)
        # Check that sand/silt/clay fractions are between 0 and 1
        min, max = extrema(filter(isfinite, field.data))
        @test 0 <= min <= 1
    end
end
