using NumericalEarth
using Oceananigans
using Statistics: mean

using NumericalEarth.DataWrangling.CanopyRoughness:
    DragPartitionParameters, canopy_roughness, canopy_wind_ratio,
    zero_plane_displacement, canopy_roughness_length,
    semiempirical_roughness, semiempirical_displacement,
    canopy_drag_parameters, drag_partition_group, is_vegetated,
    class_canopy_height, nonvegetated_roughness, snow_adjusted,
    compute_canopy_roughness!, canopy_roughness_climatology,
    fill_temporal_gaps!

using Oceananigans.Fields: interior, set!
using Oceananigans.OutputReaders: FieldTimeSeries

const κ  = 0.4
const ψh = 0.193
const iters = 20

params(g) = canopy_drag_parameters(Float64, g)

@testset "Drag-partition closure (Raupach 1994 / Jasinski 2005)" begin
    # Per-class oracle: the full closure at representative growing-season Λ and per-class
    # height must land between the class-averaged (z0i) and semi-empirical (z0e) values of
    # Borak et al. (2025) Table 5, and reproduce the correct forest ≫ non-forest ordering.
    # (group, height, Λ, z0i, z0e, d0i, d0e)
    oracle = Dict(
        "EBF" => (2, 24.72, 6.0, 1.16, 3.30, 21.18, 16.48),
        "DBF" => (2, 17.43, 5.0, 0.98, 2.32, 14.42, 11.62),
        "ENF" => (1, 16.62, 4.0, 1.36, 2.22, 11.53, 11.08),
        "GRS" => (3,  1.39, 1.5, 0.14, 0.19,  0.97,  0.93),
        "CRP" => (4,  1.32, 3.0, 0.13, 0.18,  0.72,  0.88))

    for (_, (g, h, Λ, z0i, z0e, d0i, d0e)) in oracle
        z0, d0 = canopy_roughness(Λ, h, params(g), κ, ψh, iters)
        @test min(z0i, z0e) - 0.1 ≤ z0 ≤ max(z0i, z0e) + 0.1
        @test min(d0i, d0e) - 1.0 ≤ d0 ≤ max(d0i, d0e) + 1.0
    end

    # Forest roughness/displacement dwarf non-forest.
    z0_forest, d0_forest = canopy_roughness(6.0, 24.72, params(2), κ, ψh, iters)
    z0_crop,   d0_crop   = canopy_roughness(3.0,  1.32, params(4), κ, ψh, iters)
    @test z0_forest > 10z0_crop
    @test d0_forest > 10d0_crop

    # Semi-empirical relations are exact closed forms (d0 = 2h/3, z0 = d0/5).
    @test semiempirical_displacement(24.72) ≈ 2 * 24.72 / 3
    @test semiempirical_roughness(24.72)    ≈ 2 * 24.72 / 3 / 5
    @test semiempirical_roughness(24.72)    ≈ 3.296 atol=1e-3
end

@testset "LAI dependence: monotone d0, skimming z0" begin
    p = params(4)  # cropland, Λmax = 1.5
    Λs = 0.2:0.2:1.4
    d0s = [zero_plane_displacement(Λ, canopy_wind_ratio(Λ, p, iters), 1.32, p) for Λ in Λs]
    @test issorted(d0s)                                    # d0 monotone increasing below Λmax

    # z0 is non-monotonic: it must fall once Λ exceeds Λmax (the skimming effect).
    z0(Λ) = canopy_roughness(Λ, 1.32, p, κ, ψh, iters)[1]
    @test z0(3.0) < z0(1.0)                                # dense canopy skims → lower z0
    @test z0(3.0) ≈ z0(5.0)                                # capped at Λmax beyond the critical value
end

@testset "Class mapping and non-vegetated constants" begin
    @test drag_partition_group(1, 45)  == 1   # ENF
    @test drag_partition_group(2, 45)  == 2   # EBF
    @test drag_partition_group(10, 45) == 3   # grassland
    @test drag_partition_group(12, 45) == 4   # cropland
    @test drag_partition_group(6, 45)  == 5   # closed shrubland
    @test drag_partition_group(8, 30)  == 2   # woody savanna, non-boreal
    @test drag_partition_group(8, 60)  == 1   # woody savanna, boreal
    for nonveg in (13, 15, 16, 17)
        @test drag_partition_group(nonveg, 0) == 0
        @test !is_vegetated(nonveg)
    end
    @test is_vegetated(4)

    @test nonvegetated_roughness(Float64, 13) == (0.8000, 4.83)   # urban
    @test nonvegetated_roughness(Float64, 15) == (0.0024, 0.012)  # snow/ice
    @test nonvegetated_roughness(Float64, 17) == (0.0010, 0.005)  # water

    @test class_canopy_height(Float64, 12) == 1.32
    @test class_canopy_height(Float64, 2)  == 24.72
    @test class_canopy_height(Float64, 17) == 0        # non-veg: no canopy

    # Snow lowers the substrate drag coefficient to 0.0020.
    @test snow_adjusted(params(1)).substrate_drag_coefficient == 0.0020
end

@testset "Kernel safety: finite everywhere, correct eltype" begin
    for FT in (Float32, Float64)
        p = canopy_drag_parameters(FT, 4)
        for (Λ, h) in ((FT(0), FT(1.32)), (FT(1e-6), FT(1.32)), (FT(3), FT(0)), (FT(1e3), FT(1.32)))
            z0, d0 = canopy_roughness(Λ, h, p, FT(0.4), FT(0.193), iters)
            @test isfinite(z0) && isfinite(d0)
            @test z0 isa FT && d0 isa FT
            @test z0 ≥ 0 && d0 ≥ 0
        end
    end
end

@testset "On-grid builder + climatology driver" begin
    grid = LatitudeLongitudeGrid(CPU(), Float64; size = (4, 4),
                                 longitude = (-92, -91), latitude = (37, 38),
                                 topology = (Bounded, Bounded, Flat))
    Λ  = Field{Center, Center, Nothing}(grid)
    lc = Field{Center, Center, Nothing}(grid)
    z0m = Field{Center, Center, Nothing}(grid)
    d0  = Field{Center, Center, Nothing}(grid)

    # Uniform cropland (IGBP 12) at Λ = 3: builder must match the scalar closure exactly.
    set!(Λ, 3); set!(lc, 12)
    compute_canopy_roughness!(z0m, d0, Λ, lc, grid)
    z0ref, d0ref = canopy_roughness(3.0, class_canopy_height(Float64, 12),
                                    canopy_drag_parameters(Float64, 4), κ, ψh, 20)
    @test all(≈(z0ref), interior(z0m))
    @test all(≈(d0ref), interior(d0))

    # Water (IGBP 17): prescribed constants regardless of LAI.
    set!(lc, 17)
    compute_canopy_roughness!(z0m, d0, Λ, lc, grid)
    @test all(≈(0.0010), interior(z0m))
    @test all(≈(0.005),  interior(d0))

    # Invalid LAI over a vegetated cell → honest NaN gap.
    set!(lc, 12); set!(Λ, 1e30)
    compute_canopy_roughness!(z0m, d0, Λ, lc, grid)
    @test all(isnan, interior(z0m))
    @test all(isnan, interior(d0))

    # Climatology driver returns z0m/d0 FieldTimeSeries sharing the LAI grid/times,
    # and the roughness rises with a summer LAI increase over the deciduous forest class.
    set!(lc, 4)  # deciduous broadleaf
    lai = FieldTimeSeries{Center, Center, Nothing}(grid, [0.0, 1.0])
    set!(lai[1], 0.5); set!(lai[2], 5.0)
    z0ts, d0ts = canopy_roughness_climatology(lai, lc)
    @test size(z0ts) == size(lai)
    @test mean(interior(z0ts[2])) > mean(interior(z0ts[1]))
    @test mean(interior(d0ts[2])) > mean(interior(d0ts[1]))
end

@testset "Cyclic temporal gap-fill" begin
    grid = LatitudeLongitudeGrid(CPU(), Float64; size = (1, 1),
                                 longitude = (0, 1), latitude = (0, 1),
                                 topology = (Bounded, Bounded, Flat))
    fts = FieldTimeSeries{Center, Center, Nothing}(grid, [0.0, 1.0, 2.0, 3.0])
    for (n, v) in enumerate((1.0, NaN, 3.0, NaN))
        interior(fts[n]) .= v
    end
    missing_fraction = fill_temporal_gaps!(fts)
    @test missing_fraction == 0.5                        # 2 of 4 cells were missing
    @test interior(fts[2])[1] ≈ 2.0                      # linear between 1 and 3
    @test interior(fts[4])[1] ≈ 2.0                      # cyclic: between 3 and (wrap) 1
    @test all(isfinite, vcat((vec(interior(fts[n])) for n in 1:4)...))

    # An all-missing column is left untouched (reported, not fabricated).
    fts2 = FieldTimeSeries{Center, Center, Nothing}(grid, [0.0, 1.0])
    interior(fts2[1]) .= NaN; interior(fts2[2]) .= NaN
    @test fill_temporal_gaps!(fts2) == 1.0
    @test all(isnan, vec(interior(fts2[1])))
end
