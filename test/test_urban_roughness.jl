using NumericalEarth
using Oceananigans
using Test

using NumericalEarth.Lands:
    AbstractUrbanRoughness, MacdonaldRoughness, KandaRoughness, LookupRoughness,
    IsotropicFrontalArea, CuboidFrontalArea,
    urban_roughness, aerodynamic_parameters, compute_aerodynamic_roughness!, frontal_area_index,
    macdonald_displacement_ratio, macdonald_roughness_ratio,
    kanda_displacement_height, kanda_roughness_length

using Oceananigans.Fields: interior, set!

roughness_at_point(λᵖ, h; closure = KandaRoughness()) = aerodynamic_parameters(closure, λᵖ, h)[1]
displacement_at_point(λᵖ, h; closure = KandaRoughness()) = aerodynamic_parameters(closure, λᵖ, h)[2]

@testset "Macdonald morphometric endpoints" begin
    A = MacdonaldRoughness().array_constant
    # Displacement ratio: 0 at λᵖ→0, → cap at λᵖ→1, monotone increasing between.
    @test macdonald_displacement_ratio(0.0, A, 0.95) ≈ 0 atol = 1e-12
    @test macdonald_displacement_ratio(1.0, A, 0.95) == 0.95   # clamped below the singular limit
    ratios = [macdonald_displacement_ratio(λᵖ, A, 0.95) for λᵖ in 0.05:0.05:0.9]
    @test issorted(ratios)

    # Roughness ratio vanishes with the frontal area (no obstacles → no form drag).
    @test macdonald_roughness_ratio(0.0, 0.3, 1.2, 0.4, 1.0) == 0
end

@testset "Displacement is monotone in built fraction" begin
    h = 15.0
    for closure in (MacdonaldRoughness(), KandaRoughness())
        displacements = [displacement_at_point(λᵖ, h; closure) for λᵖ in 0.02:0.02:0.9]
        @test issorted(displacements)
        @test all(0 .<= displacements .<= h)   # displacement never exceeds the building height
    end
end

@testset "Roughness peaks at intermediate built fraction (isolated → skimming)" begin
    h = 15.0
    for closure in (MacdonaldRoughness(), KandaRoughness())
        built_fractions = 0.02:0.02:0.95
        roughnesses = [roughness_at_point(λᵖ, h; closure) for λᵖ in built_fractions]
        peak = argmax(roughnesses)
        # The peak is interior — ℓᵐ rises then falls, not maximal at densest coverage.
        @test 1 < peak < length(roughnesses)
        @test roughnesses[end] < roughnesses[peak]          # skimming regime: dense coverage → lower ℓᵐ
        @test roughness_at_point(0.05, h; closure) < roughnesses[peak]   # isolated regime below the peak too
        # Dense-core magnitudes land in the documented physical range.
        @test 0.5 < roughnesses[peak] < 3.5
    end
end

@testset "Bare-soil and skimming guards" begin
    h = 20.0
    kanda = KandaRoughness()
    macdonald = MacdonaldRoughness()
    # Below the built-fraction floor the cell reduces to bare soil (prescribed ℓᵐ, d = 0).
    ℓᵐ, d = aerodynamic_parameters(kanda, 0.0, h)
    @test ℓᵐ ≈ macdonald.bare_soil_roughness   # Kanda inherits the wrapped Macdonald floor
    @test d == 0

    # Full coverage: displacement is capped strictly below the building height.
    _, dense = aerodynamic_parameters(macdonald, 1.0, h)
    @test dense / h < 1
    @test dense / h ≈ macdonald.maximum_displacement_ratio

    # Invalid inputs become honest NaN gaps.
    for (λᵖ, hᵢ) in ((NaN, h), (0.3, NaN), (0.3, -5.0))
        ℓᵐ, d = aerodynamic_parameters(kanda, λᵖ, hᵢ)
        @test isnan(ℓᵐ) && isnan(d)
    end
end

@testset "Frontal-area estimator and Kanda height heterogeneity" begin
    # Isotropic λᶠ = λᵖ; cuboid scales with height / building width.
    @test frontal_area_index(IsotropicFrontalArea(), 0.3, 15.0) == 0.3
    @test frontal_area_index(CuboidFrontalArea(building_width = 10.0), 0.3, 15.0) ≈ 0.3 * 15.0 / 10.0

    # The estimator choice changes the roughness (the dominant Macdonald uncertainty).
    isotropic = MacdonaldRoughness(frontal_area = IsotropicFrontalArea())
    cuboid = MacdonaldRoughness(frontal_area = CuboidFrontalArea(building_width = 10.0))
    @test aerodynamic_parameters(isotropic, 0.2, 15.0)[1] != aerodynamic_parameters(cuboid, 0.2, 15.0)[1]

    # Kanda roughness reduces to a1·ℓᵐ_Macdonald for a height-homogeneous canopy (σʰ → 0).
    a1 = KandaRoughness().roughness_constants[1]
    @test kanda_roughness_length(1.3, 0.3, 15.0, 0.0, a1, 20.21, -0.77) ≈ a1 * 1.3
    # Kanda displacement grows with the assumed height spread.
    narrow_spread = kanda_displacement_height(0.3, 15.0, 3.0,  37.5, 1.29, 0.36, -0.17)
    wide_spread   = kanda_displacement_height(0.3, 15.0, 10.0, 37.5, 1.29, 0.36, -0.17)
    @test wide_spread > narrow_spread
end

@testset "Lookup fallback" begin
    h = 12.0
    lookup = LookupRoughness()
    ℓᵐ, d = aerodynamic_parameters(lookup, 0.4, h)
    @test ℓᵐ ≈ lookup.bare_soil_roughness + lookup.roughness_height_fraction * h
    @test d ≈ lookup.displacement_height_fraction * h
end

@testset "Kernel safety: finite, non-negative, correct eltype" begin
    for FT in (Float32, Float64)
        for closure in (MacdonaldRoughness(FT), KandaRoughness(FT), LookupRoughness(FT))
            for (λᵖ, h) in ((FT(0), FT(10)), (FT(1e-6), FT(10)), (FT(0.3), FT(0)),
                            (FT(1), FT(30)), (FT(0.5), FT(1e3)))
                ℓᵐ, d = aerodynamic_parameters(closure, λᵖ, h)
                @test isfinite(ℓᵐ) && isfinite(d)
                @test ℓᵐ isa FT && d isa FT
                @test ℓᵐ ≥ 0 && d ≥ 0
            end
        end
    end
end

@testset "Mixed-FT closure stays Union-free (kernel/GPU safety)" begin
    # A closure whose FT differs from the grid eltype must not make aerodynamic_parameters
    # return a Union — that breaks the launched kernel (dynamic dispatch) on the GPU.
    for (Tgrid, Tclosure) in ((Float64, Float32), (Float32, Float64))
        for closure in (MacdonaldRoughness(Tclosure), KandaRoughness(Tclosure), LookupRoughness(Tclosure))
            ℓᵐ, d = @inferred aerodynamic_parameters(closure, Tgrid(0.3), Tgrid(15))
            @test typeof(ℓᵐ) == typeof(d)
            @test isfinite(ℓᵐ) && isfinite(d)
        end
    end

    # And through the launched kernel with a closure whose FT ≠ eltype(grid).
    grid = LatitudeLongitudeGrid(CPU(), Float64; size = (3, 3),
                                 longitude = (0, 1), latitude = (0, 1),
                                 topology = (Bounded, Bounded, Flat))
    λᵖ = Field{Center, Center, Nothing}(grid); set!(λᵖ, 0.3)
    h  = Field{Center, Center, Nothing}(grid); set!(h, 15.0)
    ℓᵐ, d = urban_roughness(h, λᵖ; closure = KandaRoughness(Float32))
    @test all(isfinite, interior(ℓᵐ)) && all(isfinite, interior(d))
end

@testset "On-grid builder matches the scalar closure" begin
    grid = LatitudeLongitudeGrid(CPU(), Float64; size = (6, 6),
                                 longitude = (-0.1, 0.1), latitude = (51.4, 51.6),
                                 topology = (Bounded, Bounded, Flat))
    λᵖ = Field{Center, Center, Nothing}(grid)
    h  = Field{Center, Center, Nothing}(grid)

    # Uniform urban patch: the field builder reproduces the scalar closure exactly.
    set!(λᵖ, 0.3); set!(h, 15.0)
    ℓᵐ, d = urban_roughness(h, λᵖ; closure = KandaRoughness())
    ℓᵐref, dref = aerodynamic_parameters(KandaRoughness(), 0.3, 15.0)
    @test all(≈(ℓᵐref), interior(ℓᵐ))
    @test all(≈(dref), interior(d))

    # Non-built patch reduces to bare soil everywhere.
    set!(λᵖ, 0)
    compute_aerodynamic_roughness!(ℓᵐ, d, MacdonaldRoughness(),
                                   (; plan_area_fraction = λᵖ, building_height = h), grid)
    @test all(≈(MacdonaldRoughness().bare_soil_roughness), interior(ℓᵐ))
    @test all(≈(0), interior(d))

    # Invalid inputs propagate to NaN gaps.
    set!(λᵖ, 0.3); set!(h, NaN)
    compute_aerodynamic_roughness!(ℓᵐ, d, KandaRoughness(),
                                   (; plan_area_fraction = λᵖ, building_height = h), grid)
    @test all(isnan, interior(ℓᵐ))
    @test all(isnan, interior(d))
end

@testset "The default closure is Kanda" begin
    h = 15.0
    @test aerodynamic_parameters(KandaRoughness(), 0.3, h) ==
          (roughness_at_point(0.3, h), displacement_at_point(0.3, h))
    # The callable-struct form matches the function form.
    kanda = KandaRoughness()
    @test kanda(0.3, h) == aerodynamic_parameters(kanda, 0.3, h)
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
    λᵖ = Field{Center, Center, Nothing}(grid); set!(λᵖ, 0)
    h  = Field{Center, Center, Nothing}(grid); set!(h, 15.0)

    # The wrapped Macdonald's bare-soil floor governs the on-grid Kanda result.
    ℓᵐ, _ = urban_roughness(h, λᵖ; closure = kanda)
    @test all(≈(0.05), interior(ℓᵐ))
end

@testset "Cell contract and mixed-type property sampling" begin
    kanda = KandaRoughness()
    # The cell contract reads only the closure's own keys and matches the scalar form.
    cell = (; plan_area_fraction = 0.3, building_height = 15.0, latitude = 51.5)
    @test aerodynamic_parameters(kanda, cell) == aerodynamic_parameters(kanda, 0.3, 15.0)

    # The shared grid builder samples a Field and a uniform scalar property via property_value.
    grid = LatitudeLongitudeGrid(CPU(), Float64; size = (4, 4),
                                 longitude = (-0.1, 0.1), latitude = (51.4, 51.6),
                                 topology = (Bounded, Bounded, Flat))
    ℓᵐ = Field{Center, Center, Nothing}(grid)
    d  = Field{Center, Center, Nothing}(grid)
    λᵖ = Field{Center, Center, Nothing}(grid); set!(λᵖ, 0.3)
    compute_aerodynamic_roughness!(ℓᵐ, d, kanda,
                                   (; plan_area_fraction = λᵖ, building_height = 15.0), grid)
    ℓᵐref, dref = aerodynamic_parameters(kanda, 0.3, 15.0)
    @test all(≈(ℓᵐref), interior(ℓᵐ))
    @test all(≈(dref), interior(d))
end
