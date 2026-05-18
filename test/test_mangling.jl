include("runtests_setup.jl")

using NumericalEarth.DataWrangling: mangle, mangling_for, ShiftSouth, AverageNorthSouth

using Oceananigans.Fields: location

@testset "mangle dispatch" begin
    data = reshape(Float32[1 2 3; 4 5 6; 7 8 9], 3, 3, 1)

    # Identity: reads (i, j, k) directly.
    @test mangle(2, 2, 1, data, nothing) == 5

    # ShiftSouth: reads (i, j-1, k); j=1 clamps to j=1.
    @test mangle(2, 2, 1, data, ShiftSouth()) == 4
    @test mangle(2, 1, 1, data, ShiftSouth()) == 4

    # AverageNorthSouth: averages (i, j, k) and (i, j+1, k).
    @test mangle(2, 1, 1, data, AverageNorthSouth()) ≈ 4.5f0
    @test mangle(2, 2, 1, data, AverageNorthSouth()) ≈ 5.5f0
end

@testset "mangling_for size dispatch" begin
    md = Metadatum(:v_velocity; dataset=ECCO4Monthly(), date=start_date)
    _, Ny, _, _ = size(md)

    @test mangling_for(md, Ny)     === nothing
    @test mangling_for(md, Ny - 1) isa ShiftSouth
    @test mangling_for(md, Ny + 1) isa AverageNorthSouth
    @test mangling_for(md, Ny - 2) === nothing
    @test mangling_for(md, Ny + 2) === nothing
end

@testset "ECCO v_velocity Field uses ShiftSouth mangling end-to-end" begin
    for arch in test_architectures
        md = Metadatum(:v_velocity; dataset=ECCO4Monthly(), date=start_date)
        field = Field(md, arch)

        # v lives on the latitude Face for ECCO; the file ships Ny-1 lat
        # entries, so mangling_for must select ShiftSouth — anything else
        # would either go out of bounds or read off-by-one everywhere.
        @test location(field) == (Center, Face, Center)

        @allowscalar begin
            @test any(!=(0), interior(field))
            @test any(isfinite, interior(field))
        end
    end
end
