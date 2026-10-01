include("runtests_setup.jl")

using NumericalEarth.DataWrangling: masked_regrid!

@testset "masked_regrid!" begin
    topology = (Bounded, Bounded, Flat)
    source_grid = RectilinearGrid(CPU(); size = (4, 4), x = (0, 4), y = (0, 4), topology)
    target_grid = RectilinearGrid(CPU(); size = (2, 2), x = (0, 4), y = (0, 4), topology)

    source = CenterField(source_grid)
    target = CenterField(target_grid)

    interior(source, :, :, 1) .= reshape(1:16, 4, 4)
    interior(source, 1, 1, 1) .= NaN
    interior(source, 3:4, 3:4, 1) .= NaN

    masked_regrid!(target, source)
    result = Array(interior(target, :, :, 1))

    @test result[1, 1] ≈ (2 + 5 + 6) / 3
    @test result[2, 1] ≈ (3 + 4 + 7 + 8) / 4
    @test result[1, 2] ≈ (9 + 10 + 13 + 14) / 4
    @test isnan(result[2, 2])

    set!(source, 3.5)
    interior(source, 2, 2, 1) .= NaN
    masked_regrid!(target, source)
    @test all(Array(interior(target)) .≈ 3.5)
end
