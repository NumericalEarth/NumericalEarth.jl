using NumericalEarth
using Oceananigans
using Test

using NumericalEarth.Lands:
    AbstractUrbanRoughness, MorphometricRoughness,
    IsotropicFrontalArea, CuboidFrontalArea, UniformHeight, VariableHeight,
    urban_roughness, aerodynamic_parameters, compute_aerodynamic_roughness!, frontal_area_index,
    packing_displacement_ratio, drag_partition_roughness_ratio,
    maximum_height_displacement, height_spread_roughness

using Oceananigans.Fields: interior, set!

uniform_height_closure(; kw...) = MorphometricRoughness(; height_distribution = UniformHeight(), kw...)

roughness_at_point(λᵖ, h; closure = MorphometricRoughness()) = aerodynamic_parameters(closure, λᵖ, h)[1]
displacement_at_point(λᵖ, h; closure = MorphometricRoughness()) = aerodynamic_parameters(closure, λᵖ, h)[2]

@testset "Obstacle-array morphometric endpoints" begin
    A = MorphometricRoughness().array_constant
    # Displacement ratio: 0 at λᵖ→0, → cap at λᵖ→1, monotone increasing between.
    @test packing_displacement_ratio(0.0, A, 0.95) ≈ 0 atol = 1e-12
    @test packing_displacement_ratio(1.0, A, 0.95) == 0.95   # clamped below the singular limit
    ratios = [packing_displacement_ratio(λᵖ, A, 0.95) for λᵖ in 0.05:0.05:0.9]
    @test issorted(ratios)

    # Roughness ratio vanishes with the frontal area (no obstacles → no form drag).
    @test drag_partition_roughness_ratio(0.0, 0.3, 1.2, 0.4, 1.0) == 0
end

@testset "Displacement is monotone in built fraction" begin
    h = 15.0
    for closure in (uniform_height_closure(), MorphometricRoughness())
        displacements = [displacement_at_point(λᵖ, h; closure) for λᵖ in 0.02:0.02:0.9]
        @test issorted(displacements)
        @test all(0 .<= displacements .<= h)   # displacement never exceeds the building height
    end
end

@testset "Roughness peaks at intermediate built fraction (isolated → skimming)" begin
    h = 15.0
    for closure in (uniform_height_closure(), MorphometricRoughness())
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
    variable = MorphometricRoughness()
    uniform = uniform_height_closure()
    # Below the built-fraction floor the cell reduces to bare soil (prescribed ℓᵐ, d = 0).
    ℓᵐ, d = aerodynamic_parameters(variable, 0.0, h)
    @test ℓᵐ ≈ uniform.bare_soil_roughness   # the floor is a closure-level parameter
    @test d == 0

    # Full coverage: displacement is capped strictly below the building height.
    _, dense = aerodynamic_parameters(uniform, 1.0, h)
    @test dense / h < 1
    @test dense / h ≈ uniform.maximum_displacement_ratio

    # Invalid inputs become honest NaN gaps.
    for (λᵖ, hᵢ) in ((NaN, h), (0.3, NaN), (0.3, -5.0))
        ℓᵐ, d = aerodynamic_parameters(variable, λᵖ, hᵢ)
        @test isnan(ℓᵐ) && isnan(d)
    end
end

@testset "Frontal-area estimator and height-spread correction" begin
    # Isotropic λᶠ = λᵖ; cuboid scales with height / building width.
    @test frontal_area_index(IsotropicFrontalArea(), 0.3, 15.0) == 0.3
    @test frontal_area_index(CuboidFrontalArea(building_width = 10.0), 0.3, 15.0) ≈ 0.3 * 15.0 / 10.0

    # The estimator choice changes the roughness (the dominant drag-partition uncertainty).
    isotropic = uniform_height_closure(frontal_area = IsotropicFrontalArea())
    cuboid = uniform_height_closure(frontal_area = CuboidFrontalArea(building_width = 10.0))
    @test aerodynamic_parameters(isotropic, 0.2, 15.0)[1] != aerodynamic_parameters(cuboid, 0.2, 15.0)[1]

    # The spread correction reduces to a1·(uniform-height ℓᵐ) for σʰ → 0.
    a1 = VariableHeight().roughness_constants[1]
    @test height_spread_roughness(1.3, 0.3, 15.0, 0.0, a1, 20.21, -0.77) ≈ a1 * 1.3
    # Displacement grows with the assumed height spread.
    narrow_spread = maximum_height_displacement(0.3, 15.0, 3.0,  37.5, 1.29, 0.36, -0.17)
    wide_spread   = maximum_height_displacement(0.3, 15.0, 10.0, 37.5, 1.29, 0.36, -0.17)
    @test wide_spread > narrow_spread

    # UniformHeight leaves the drag-partition parameters untouched; VariableHeight does not.
    @test aerodynamic_parameters(uniform_height_closure(), 0.3, 15.0) !=
          aerodynamic_parameters(MorphometricRoughness(), 0.3, 15.0)
end

@testset "Kernel safety: finite, non-negative, correct eltype" begin
    for FT in (Float32, Float64)
        for closure in (MorphometricRoughness(FT; height_distribution = UniformHeight()),
                        MorphometricRoughness(FT))
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
        for closure in (MorphometricRoughness(Tclosure; height_distribution = UniformHeight()),
                        MorphometricRoughness(Tclosure))
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
    ℓᵐ, d = urban_roughness(h, λᵖ; closure = MorphometricRoughness(Float32))
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
    ℓᵐ, d = urban_roughness(h, λᵖ; closure = MorphometricRoughness())
    ℓᵐref, dref = aerodynamic_parameters(MorphometricRoughness(), 0.3, 15.0)
    @test all(≈(ℓᵐref), interior(ℓᵐ))
    @test all(≈(dref), interior(d))

    # Non-built patch reduces to bare soil everywhere.
    set!(λᵖ, 0)
    compute_aerodynamic_roughness!(ℓᵐ, d, uniform_height_closure(),
                                   (; plan_area_fraction = λᵖ, building_height = h), grid)
    @test all(≈(MorphometricRoughness().bare_soil_roughness), interior(ℓᵐ))
    @test all(≈(0), interior(d))

    # Invalid inputs propagate to NaN gaps.
    set!(λᵖ, 0.3); set!(h, NaN)
    compute_aerodynamic_roughness!(ℓᵐ, d, MorphometricRoughness(),
                                   (; plan_area_fraction = λᵖ, building_height = h), grid)
    @test all(isnan, interior(ℓᵐ))
    @test all(isnan, interior(d))
end

@testset "The default height distribution is VariableHeight" begin
    h = 15.0
    @test MorphometricRoughness().height_distribution isa VariableHeight
    @test aerodynamic_parameters(MorphometricRoughness(), 0.3, h) ==
          (roughness_at_point(0.3, h), displacement_at_point(0.3, h))
    # The callable-struct form matches the function form.
    closure = MorphometricRoughness()
    @test closure(0.3, h) == aerodynamic_parameters(closure, 0.3, h)
end

@testset "Closure construction and composition" begin
    # Drag-partition parameters and the height distribution are configured side by side,
    # on one flat closure.
    closure = MorphometricRoughness(array_constant = 3.59, bare_soil_roughness = 0.05,
                                    frontal_area = CuboidFrontalArea(building_width = 12.0),
                                    height_distribution = VariableHeight(height_variability = 0.6))
    @test closure.array_constant == 3.59
    @test closure.bare_soil_roughness == 0.05
    @test closure.frontal_area isa CuboidFrontalArea
    @test closure.height_distribution.height_variability == 0.6

    # A narrower closure FT converts the sub-closures too.
    narrow = MorphometricRoughness(Float32; frontal_area = CuboidFrontalArea(building_width = 12.0))
    @test narrow.frontal_area.building_width isa Float32
    @test narrow.height_distribution.height_variability isa Float32

    grid = LatitudeLongitudeGrid(CPU(), Float64; size = (3, 3),
                                 longitude = (0, 1), latitude = (0, 1),
                                 topology = (Bounded, Bounded, Flat))
    λᵖ = Field{Center, Center, Nothing}(grid); set!(λᵖ, 0)
    h  = Field{Center, Center, Nothing}(grid); set!(h, 15.0)

    # The bare-soil floor governs the on-grid result.
    ℓᵐ, _ = urban_roughness(h, λᵖ; closure)
    @test all(≈(0.05), interior(ℓᵐ))
end

@testset "Cell contract and mixed-type property sampling" begin
    closure = MorphometricRoughness()
    # The cell contract reads only the closure's own keys and matches the scalar form.
    cell = (; plan_area_fraction = 0.3, building_height = 15.0, latitude = 51.5)
    @test aerodynamic_parameters(closure, cell) == aerodynamic_parameters(closure, 0.3, 15.0)

    # The shared grid builder samples a Field and a uniform scalar property via property_value.
    grid = LatitudeLongitudeGrid(CPU(), Float64; size = (4, 4),
                                 longitude = (-0.1, 0.1), latitude = (51.4, 51.6),
                                 topology = (Bounded, Bounded, Flat))
    ℓᵐ = Field{Center, Center, Nothing}(grid)
    d  = Field{Center, Center, Nothing}(grid)
    λᵖ = Field{Center, Center, Nothing}(grid); set!(λᵖ, 0.3)
    compute_aerodynamic_roughness!(ℓᵐ, d, closure,
                                   (; plan_area_fraction = λᵖ, building_height = 15.0), grid)
    ℓᵐref, dref = aerodynamic_parameters(closure, 0.3, 15.0)
    @test all(≈(ℓᵐref), interior(ℓᵐ))
    @test all(≈(dref), interior(d))
end
