include("runtests_setup.jl")

using ArchGDAL  # loads NumericalEarthArchGDALExt (the windowed /vsicurl COG reader)
using NumericalEarth.DataWrangling: native_grid

# Network-gated: reads the 120 m Antarctic COGs over /vsicurl (anonymous, no credentials).
# Excluded from the default suite in runtests.jl, mirroring the other *_downloading tests.
@testset "ITS_LIVE Antarctica windowed download" begin
    dataset = ITSLIVEAntarctica()
    # Trunk of Pine Island Glacier
    region = ProjectedBoundingBox(x = (-1.62e6, -1.59e6), y = (-2.8e5, -2.5e5))

    u, v, speed = [Array(interior(Field(Metadatum(name; dataset, region)), :, :, 1))
                   for name in (:velocity_x, :velocity_y, :speed)]
    grid = native_grid(Metadatum(:speed; dataset, region))

    @test size(speed) == (size(grid, 1), size(grid, 2))
    observed = isfinite.(u) .& isfinite.(v) .& isfinite.(speed)
    @test count(observed) > length(speed) / 2
    @test maximum(speed[observed]) > 1000           # fast-flowing trunk, m/a
    @test maximum(speed[observed]) < 10000
    @test length(unique(speed[observed])) > 1
    @show maximum(abs.(hypot.(u[observed], v[observed]) .- speed[observed]))
end
