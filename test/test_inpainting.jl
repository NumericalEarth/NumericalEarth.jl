include("runtests_setup.jl")

using NumericalEarth.DataWrangling: inpaint_mask!, remaining_gaps, NearestNeighborInpainting

@testset "Inpainting" begin
    for arch in test_architectures
        @testset "Levels without data are filled from above on $(typeof(arch))" begin
            grid = LatitudeLongitudeGrid(arch; size = (8, 6, 3), longitude = (0, 360), latitude = (-60, 60),
                                         z = (-3, 0), halo = (3, 3, 3))

            field = CenterField(grid)
            set!(field, (λ, φ, z) -> 34 + z)
            mask = CenterField(grid, Bool)

            # The deepest level holds no data, as in reanalyses whose deepest level lies below the
            # deepest ocean point, and one column is land at every depth.
            set!(mask, (λ, φ, z) -> z < -2)
            interior(field)[:, :, 1] .= NaN
            mask_data = Array(interior(mask))
            mask_data[3, 2, :] .= true
            set!(mask, mask_data)
            interior(field)[3, 2, :] .= NaN

            inpaint_mask!(field, mask; inpainting = NearestNeighborInpainting(Inf))
            data = Array(interior(field))

            @test !any(isnan, data)
            @test !any(iszero, data)
            @test data[:, :, 1] == data[:, :, 2]
            @test data[3, 2, 3] ≈ 34 - 0.5
        end
    end

    @testset "Data shallower than the grid is filled from neighbors at the same depth" begin
        grid = LatitudeLongitudeGrid(CPU(); size = (8, 6, 3), longitude = (0, 360), latitude = (-60, 60),
                                     z = (-3, 0), halo = (3, 3, 3))
        field = CenterField(grid)
        set!(field, (λ, φ, z) -> 34 + z)
        mask = CenterField(grid, Bool)

        # An atoll: the data holds only the surface level of one column, whose surface value differs.
        mask_data = zeros(Bool, size(mask))
        mask_data[4, 3, 1:2] .= true
        set!(mask, mask_data)
        interior(field)[4, 3, 1:2] .= NaN
        interior(field)[4, 3, 3] = 30

        inpaint_mask!(field, mask; inpainting = NearestNeighborInpainting(Inf))

        @test interior(field)[4, 3, 1] ≈ 34 - 2.5
        @test interior(field)[4, 3, 2] ≈ 34 - 1.5
    end

    # Summing `isnan` in Float32 stops counting at 2²⁴, which stalled inpainting for large datasets.
    @testset "Gap count is exact beyond 2²⁴ gaps in Float32" begin
        grid = RectilinearGrid(CPU(), Float32; size = (4200, 4000), x = (0, 1), y = (0, 1),
                               topology = (Periodic, Periodic, Flat))
        field = CenterField(grid)
        mask = CenterField(grid, Bool)
        interior(field) .= NaN
        interior(mask) .= true

        @test remaining_gaps(field, mask) == 4200 * 4000
    end
end
