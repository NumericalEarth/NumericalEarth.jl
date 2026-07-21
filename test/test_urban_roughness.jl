using NumericalEarth
using Oceananigans
using Test
using Statistics: mean

using NumericalEarth.DataWrangling.UrbanRoughness:
    UrbanRoughnessParameters, urban_roughness, urban_roughness_point,
    compute_urban_roughness!, frontal_area_index,
    macdonald_displacement_ratio, macdonald_roughness_ratio,
    kanda_displacement_height, kanda_roughness_length,
    MACDONALD, KANDA, LOOKUP, ISOTROPIC, CUBOID

using Oceananigans.Fields: interior, set!

params(FT = Float64) = UrbanRoughnessParameters(FT)

z0m_point(λp, H; method = KANDA, estimator = ISOTROPIC) =
    urban_roughness_point(λp, H, method, estimator, params())[1]
d0_point(λp, H; method = KANDA, estimator = ISOTROPIC) =
    urban_roughness_point(λp, H, method, estimator, params())[2]

@testset "Macdonald morphometric endpoints" begin
    A = params().array_constant
    # Displacement ratio: 0 at λp→0, → cap at λp→1, monotone increasing between.
    @test macdonald_displacement_ratio(0.0, A, 0.95) ≈ 0 atol = 1e-12
    @test macdonald_displacement_ratio(1.0, A, 0.95) == 0.95   # clamped below the singular limit
    ratios = [macdonald_displacement_ratio(λ, A, 0.95) for λ in 0.05:0.05:0.9]
    @test issorted(ratios)

    # Roughness ratio vanishes with the frontal area (no obstacles → no form drag).
    @test macdonald_roughness_ratio(0.0, 0.3, 1.2, 0.4, 1.0) == 0
end

@testset "Displacement is monotone in built fraction" begin
    H = 15.0
    for method in (MACDONALD, KANDA)
        d0s = [d0_point(λ, H; method) for λ in 0.02:0.02:0.9]
        @test issorted(d0s)
        @test all(0 .<= d0s .<= H)          # displacement never exceeds the building height
    end
end

@testset "Roughness peaks at intermediate built fraction (isolated → skimming)" begin
    H = 15.0
    for method in (MACDONALD, KANDA)
        λs = 0.02:0.02:0.95
        z0s = [z0m_point(λ, H; method) for λ in λs]
        peak = argmax(z0s)
        # The peak is interior — z0 rises then falls, not maximal at densest coverage.
        @test 1 < peak < length(z0s)
        @test z0s[end] < z0s[peak]          # skimming regime: dense coverage → lower z0
        @test z0m_point(0.05, H; method) < z0s[peak]   # isolated regime below the peak too
        # Dense-core magnitudes land in the documented physical range.
        @test 0.5 < z0s[peak] < 3.5
    end
end

@testset "Bare-soil and skimming guards" begin
    H = 20.0
    p = params()
    # Below the built-fraction floor the cell reduces to bare soil (prescribed z0, d0 = 0).
    z0b, d0b = urban_roughness_point(0.0, H, KANDA, ISOTROPIC, p)
    @test z0b ≈ p.bare_soil_roughness
    @test d0b == 0

    # Full coverage: displacement is capped strictly below the building height.
    _, d0s = urban_roughness_point(1.0, H, MACDONALD, ISOTROPIC, p)
    @test d0s / H < 1
    @test d0s / H ≈ p.maximum_displacement_ratio

    # Invalid inputs become honest NaN gaps.
    for (λ, h) in ((NaN, H), (0.3, NaN), (0.3, -5.0))
        z0, d0 = urban_roughness_point(λ, h, KANDA, ISOTROPIC, p)
        @test isnan(z0) && isnan(d0)
    end
end

@testset "Frontal-area estimator and Kanda height heterogeneity" begin
    # Isotropic λf = λp; cuboid scales with height / building width.
    @test frontal_area_index(0.3, 15.0, ISOTROPIC, 10.0) == 0.3
    @test frontal_area_index(0.3, 15.0, CUBOID, 10.0)    ≈ 0.3 * 15.0 / 10.0

    # The estimator choice changes the roughness (the dominant Macdonald uncertainty).
    z0_iso  = z0m_point(0.2, 15.0; method = MACDONALD, estimator = ISOTROPIC)
    z0_cub  = z0m_point(0.2, 15.0; method = MACDONALD, estimator = CUBOID)
    @test z0_iso != z0_cub

    # Kanda roughness reduces to a1·z0_Macdonald for a height-homogeneous canopy (σh → 0).
    a1 = params().kanda_roughness[1]
    @test kanda_roughness_length(1.3, 0.3, 15.0, 0.0, a1, 20.21, -0.77) ≈ a1 * 1.3
    # Kanda displacement grows with the assumed height spread.
    d_low  = kanda_displacement_height(0.3, 15.0, 3.0,  37.5, 1.29, 0.36, -0.17)
    d_high = kanda_displacement_height(0.3, 15.0, 10.0, 37.5, 1.29, 0.36, -0.17)
    @test d_high > d_low
end

@testset "Lookup fallback" begin
    H = 12.0
    z0, d0 = urban_roughness_point(0.4, H, LOOKUP, ISOTROPIC, params())
    @test z0 ≈ params().bare_soil_roughness + 0.1H
    @test d0 ≈ 0.7H
end

@testset "Kernel safety: finite, non-negative, correct eltype" begin
    for FT in (Float32, Float64)
        p = params(FT)
        for method in (MACDONALD, KANDA, LOOKUP)
            for (λ, h) in ((FT(0), FT(10)), (FT(1e-6), FT(10)), (FT(0.3), FT(0)),
                           (FT(1), FT(30)), (FT(0.5), FT(1e3)))
                z0, d0 = urban_roughness_point(λ, h, method, ISOTROPIC, p)
                @test isfinite(z0) && isfinite(d0)
                @test z0 isa FT && d0 isa FT
                @test z0 ≥ 0 && d0 ≥ 0
            end
        end
    end
end

@testset "On-grid builder matches the scalar closure" begin
    grid = LatitudeLongitudeGrid(CPU(), Float64; size = (6, 6),
                                 longitude = (-0.1, 0.1), latitude = (51.4, 51.6),
                                 topology = (Bounded, Bounded, Flat))
    λp = Field{Center, Center, Nothing}(grid)
    H  = Field{Center, Center, Nothing}(grid)

    # Uniform urban patch: the field builder reproduces the scalar closure exactly.
    set!(λp, 0.3); set!(H, 15.0)
    z0m, d0 = urban_roughness(H, λp; method = :kanda)
    z0ref, d0ref = urban_roughness_point(0.3, 15.0, KANDA, ISOTROPIC, params())
    @test all(≈(z0ref), interior(z0m))
    @test all(≈(d0ref), interior(d0))

    # Non-built patch reduces to bare soil everywhere.
    set!(λp, 0)
    compute_urban_roughness!(z0m, d0, λp, H, grid; method = :macdonald)
    @test all(≈(params().bare_soil_roughness), interior(z0m))
    @test all(≈(0), interior(d0))

    # Invalid inputs propagate to NaN gaps.
    set!(λp, 0.3); set!(H, NaN)
    compute_urban_roughness!(z0m, d0, λp, H, grid; method = :kanda)
    @test all(isnan, interior(z0m))
    @test all(isnan, interior(d0))
end

@testset "Method / estimator argument validation" begin
    grid = LatitudeLongitudeGrid(CPU(), Float64; size = (2, 2),
                                 longitude = (0, 1), latitude = (0, 1),
                                 topology = (Bounded, Bounded, Flat))
    λp = Field{Center, Center, Nothing}(grid); set!(λp, 0.3)
    H  = Field{Center, Center, Nothing}(grid); set!(H, 10)
    @test_throws ArgumentError urban_roughness(H, λp; method = :bogus)
    @test_throws ArgumentError urban_roughness(H, λp; frontal_area = :bogus)
end
