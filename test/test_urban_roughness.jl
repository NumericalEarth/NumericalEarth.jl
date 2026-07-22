using NumericalEarth
using Oceananigans
using Test

using NumericalEarth.DataWrangling.UrbanRoughness:
    AbstractUrbanRoughness, MacdonaldRoughness, KandaRoughness, LookupRoughness,
    IsotropicFrontalArea, CuboidFrontalArea,
    urban_roughness, roughness_lengths, compute_urban_roughness!, frontal_area_index,
    macdonald_displacement_ratio, macdonald_roughness_ratio,
    kanda_displacement_height, kanda_roughness_length

using Oceananigans.Fields: interior, set!

z0m_point(λp, H; closure = KandaRoughness()) = roughness_lengths(closure, λp, H)[1]
d0_point(λp, H; closure = KandaRoughness()) = roughness_lengths(closure, λp, H)[2]

@testset "Macdonald morphometric endpoints" begin
    A = MacdonaldRoughness().array_constant
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
    for closure in (MacdonaldRoughness(), KandaRoughness())
        d0s = [d0_point(λ, H; closure) for λ in 0.02:0.02:0.9]
        @test issorted(d0s)
        @test all(0 .<= d0s .<= H)          # displacement never exceeds the building height
    end
end

@testset "Roughness peaks at intermediate built fraction (isolated → skimming)" begin
    H = 15.0
    for closure in (MacdonaldRoughness(), KandaRoughness())
        λs = 0.02:0.02:0.95
        z0s = [z0m_point(λ, H; closure) for λ in λs]
        peak = argmax(z0s)
        # The peak is interior — z0 rises then falls, not maximal at densest coverage.
        @test 1 < peak < length(z0s)
        @test z0s[end] < z0s[peak]          # skimming regime: dense coverage → lower z0
        @test z0m_point(0.05, H; closure) < z0s[peak]   # isolated regime below the peak too
        # Dense-core magnitudes land in the documented physical range.
        @test 0.5 < z0s[peak] < 3.5
    end
end

@testset "Bare-soil and skimming guards" begin
    H = 20.0
    kanda = KandaRoughness()
    macd  = MacdonaldRoughness()
    # Below the built-fraction floor the cell reduces to bare soil (prescribed z0, d0 = 0).
    z0b, d0b = roughness_lengths(kanda, 0.0, H)
    @test z0b ≈ macd.bare_soil_roughness   # Kanda inherits the wrapped Macdonald floor
    @test d0b == 0

    # Full coverage: displacement is capped strictly below the building height.
    _, d0s = roughness_lengths(macd, 1.0, H)
    @test d0s / H < 1
    @test d0s / H ≈ macd.maximum_displacement_ratio

    # Invalid inputs become honest NaN gaps.
    for (λ, h) in ((NaN, H), (0.3, NaN), (0.3, -5.0))
        z0, d0 = roughness_lengths(kanda, λ, h)
        @test isnan(z0) && isnan(d0)
    end
end

@testset "Frontal-area estimator and Kanda height heterogeneity" begin
    # Isotropic λf = λp; cuboid scales with height / building width.
    @test frontal_area_index(IsotropicFrontalArea(), 0.3, 15.0) == 0.3
    @test frontal_area_index(CuboidFrontalArea(building_width = 10.0), 0.3, 15.0) ≈ 0.3 * 15.0 / 10.0

    # The estimator choice changes the roughness (the dominant Macdonald uncertainty).
    iso = MacdonaldRoughness(frontal_area = IsotropicFrontalArea())
    cub = MacdonaldRoughness(frontal_area = CuboidFrontalArea(building_width = 10.0))
    @test roughness_lengths(iso, 0.2, 15.0)[1] != roughness_lengths(cub, 0.2, 15.0)[1]

    # Kanda roughness reduces to a1·z0_Macdonald for a height-homogeneous canopy (σh → 0).
    a1 = KandaRoughness().roughness_constants[1]
    @test kanda_roughness_length(1.3, 0.3, 15.0, 0.0, a1, 20.21, -0.77) ≈ a1 * 1.3
    # Kanda displacement grows with the assumed height spread.
    d_low  = kanda_displacement_height(0.3, 15.0, 3.0,  37.5, 1.29, 0.36, -0.17)
    d_high = kanda_displacement_height(0.3, 15.0, 10.0, 37.5, 1.29, 0.36, -0.17)
    @test d_high > d_low
end

@testset "Lookup fallback" begin
    H = 12.0
    lookup = LookupRoughness()
    z0, d0 = roughness_lengths(lookup, 0.4, H)
    @test z0 ≈ lookup.bare_soil_roughness + lookup.roughness_height_fraction * H
    @test d0 ≈ lookup.displacement_height_fraction * H
end

@testset "Kernel safety: finite, non-negative, correct eltype" begin
    for FT in (Float32, Float64)
        for closure in (MacdonaldRoughness(FT), KandaRoughness(FT), LookupRoughness(FT))
            for (λ, h) in ((FT(0), FT(10)), (FT(1e-6), FT(10)), (FT(0.3), FT(0)),
                           (FT(1), FT(30)), (FT(0.5), FT(1e3)))
                z0, d0 = roughness_lengths(closure, λ, h)
                @test isfinite(z0) && isfinite(d0)
                @test z0 isa FT && d0 isa FT
                @test z0 ≥ 0 && d0 ≥ 0
            end
        end
    end
end

@testset "Mixed-FT closure stays Union-free (kernel/GPU safety)" begin
    # A closure whose FT differs from the grid eltype must not make roughness_lengths
    # return a Union — that breaks the launched kernel (dynamic dispatch) on the GPU.
    for (Tgrid, Tclosure) in ((Float64, Float32), (Float32, Float64))
        for closure in (MacdonaldRoughness(Tclosure), KandaRoughness(Tclosure), LookupRoughness(Tclosure))
            z0, d0 = @inferred roughness_lengths(closure, Tgrid(0.3), Tgrid(15))
            @test typeof(z0) == typeof(d0)
            @test isfinite(z0) && isfinite(d0)
        end
    end

    # And through the launched kernel with a closure whose FT ≠ eltype(grid).
    grid = LatitudeLongitudeGrid(CPU(), Float64; size = (3, 3),
                                 longitude = (0, 1), latitude = (0, 1),
                                 topology = (Bounded, Bounded, Flat))
    λp = Field{Center, Center, Nothing}(grid); set!(λp, 0.3)
    H  = Field{Center, Center, Nothing}(grid); set!(H, 15.0)
    z0m, d0 = urban_roughness(H, λp; closure = KandaRoughness(Float32))
    @test all(isfinite, interior(z0m)) && all(isfinite, interior(d0))
end

@testset "On-grid builder matches the scalar closure" begin
    grid = LatitudeLongitudeGrid(CPU(), Float64; size = (6, 6),
                                 longitude = (-0.1, 0.1), latitude = (51.4, 51.6),
                                 topology = (Bounded, Bounded, Flat))
    λp = Field{Center, Center, Nothing}(grid)
    H  = Field{Center, Center, Nothing}(grid)

    # Uniform urban patch: the field builder reproduces the scalar closure exactly.
    set!(λp, 0.3); set!(H, 15.0)
    z0m, d0 = urban_roughness(H, λp; closure = KandaRoughness())
    z0ref, d0ref = roughness_lengths(KandaRoughness(), 0.3, 15.0)
    @test all(≈(z0ref), interior(z0m))
    @test all(≈(d0ref), interior(d0))

    # Non-built patch reduces to bare soil everywhere.
    set!(λp, 0)
    compute_urban_roughness!(z0m, d0, λp, H, grid; closure = MacdonaldRoughness())
    @test all(≈(MacdonaldRoughness().bare_soil_roughness), interior(z0m))
    @test all(≈(0), interior(d0))

    # Invalid inputs propagate to NaN gaps.
    set!(λp, 0.3); set!(H, NaN)
    compute_urban_roughness!(z0m, d0, λp, H, grid; closure = KandaRoughness())
    @test all(isnan, interior(z0m))
    @test all(isnan, interior(d0))
end

@testset "The default closure is Kanda" begin
    H = 15.0
    @test roughness_lengths(KandaRoughness(), 0.3, H) == (z0m_point(0.3, H), d0_point(0.3, H))
    # The callable-struct form matches the function form.
    kanda = KandaRoughness()
    @test kanda(0.3, H) == roughness_lengths(kanda, 0.3, H)
end

@testset "Closure construction and composition" begin
    # A configured Macdonald base propagates into the Kanda closure that wraps it,
    # instead of resetting to the defaults.
    base  = MacdonaldRoughness(array_constant = 3.59, bare_soil_roughness = 0.05,
                               frontal_area = CuboidFrontalArea(building_width = 12.0))
    kanda = KandaRoughness(macdonald = base)
    @test kanda.macdonald.array_constant == 3.59
    @test kanda.macdonald.bare_soil_roughness == 0.05
    @test kanda.macdonald.frontal_area isa CuboidFrontalArea

    grid = LatitudeLongitudeGrid(CPU(), Float64; size = (3, 3),
                                 longitude = (0, 1), latitude = (0, 1),
                                 topology = (Bounded, Bounded, Flat))
    λp = Field{Center, Center, Nothing}(grid); set!(λp, 0)
    H  = Field{Center, Center, Nothing}(grid); set!(H, 15.0)

    # The wrapped Macdonald's bare-soil floor governs the on-grid Kanda result.
    z0m, _ = urban_roughness(H, λp; closure = kanda)
    @test all(≈(0.05), interior(z0m))
end
