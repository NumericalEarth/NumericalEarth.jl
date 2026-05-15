using NumericalEarth
using Test
using JLD2

using NumericalEarth.Bathymetry: BathymetryPatchLog

@testset "Offline bathymetry conditioning" begin
    @testset "roughness calculation" begin
        h = [10.0 10.0 10.0
             10.0 30.0 10.0
             10.0 10.0 10.0]

        roughness = compute_bathymetry_roughness(h)

        @test roughness[2, 2] == 0.5
        @test roughness[1, 2] == 0.5
        @test roughness[2, 1] == 0.5
        @test roughness[1, 1] == 0.0
    end

    @testset "NaN and Inf diagnostics are unstable" begin
        cfl = [0.1 Inf
               NaN 0.9]
        wetmask = [true true
                   true false]

        mask = flag_unstable_columns(cfl=cfl, wetmask=wetmask, cfl_threshold=0.8)

        @test mask == [false true
                       true false]
    end

    @testset "dilation respects wetmask" begin
        mask = falses(3, 3)
        mask[2, 2] = true

        wetmask = trues(3, 3)
        wetmask[2, 3] = false

        dilated = dilate_mask(mask; radius=1, wetmask)

        @test dilated[2, 2]
        @test dilated[1, 2]
        @test dilated[2, 1]
        @test dilated[3, 2]
        @test !dilated[2, 3]
    end

    @testset "smoothing is local and non-mutating" begin
        h = [30.0 30.0 30.0
             30.0 10.0 30.0
             30.0 30.0 30.0]
        original_h = copy(h)
        mask = falses(3, 3)
        mask[2, 2] = true

        new_h, log = smooth_flagged_bathymetry(h, mask; mode=:deepen_only)

        @test h == original_h
        @test new_h[2, 2] >= h[2, 2]
        @test all(new_h .>= h)
        @test log isa BathymetryPatchLog
    end

    @testset "preserve_mask cells are unchanged" begin
        h = [30.0 30.0 30.0
             30.0 10.0 30.0
             30.0 30.0 30.0]
        mask = falses(3, 3)
        mask[2, 2] = true
        preserve_mask = falses(3, 3)
        preserve_mask[2, 2] = true

        new_h, log = smooth_flagged_bathymetry(h, mask; preserve_mask)

        @test new_h == h
        @test length(log) == 0
    end

    @testset "patch summary counts changed cells" begin
        old_h = [10.0 20.0
                 30.0 40.0]
        new_h = [10.0 25.0
                 30.0 50.0]

        summary = summarize_bathymetry_patch(old_h, new_h)

        @test summary.changed_cells == 2
        @test summary.max_abs_depth_change == 10.0
        @test summary.mean_abs_depth_change == 7.5
        @test summary.total_depth_change == 15.0
    end

    @testset "checkpoint workflow unions instability masks" begin
        dir = mktempdir()
        prefix = joinpath(dir, "checkpoint")

        h = [30.0 30.0 30.0
             30.0 10.0 30.0
             30.0 30.0 30.0]

        for iteration in (1, 2)
            u = zeros(5, 5, 2)
            u[1, 1, 1] = NaN # halo NaN, ignored because h is 3 x 3.
            if iteration == 2
                u[3, 3, 1] = NaN
            end

            jldopen(prefix * "_iteration$(iteration).jld2", "w") do file
                file["simulation/model/ocean/model/velocities/u/data"] = u
            end
        end

        result = condition_bathymetry(prefix;
                                      run = "all",
                                      bathymetry = h,
                                      field_halo = (1, 1),
                                      write = false,
                                      smoothing_iterations = 1)

        @test length(result.checkpoint_files) == 2
        @test result.nan_mask[2, 2]
        @test count(result.nan_mask) == 1
        @test result.halo == (1, 1)
        @test result.repair_mask[2, 2]
        @test result.summary.changed_cells == 1
        @test result.h[2, 2] == 30.0
    end
end
