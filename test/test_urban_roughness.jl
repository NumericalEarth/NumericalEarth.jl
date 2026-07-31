using NumericalEarth
using Oceananigans
using Test

using NumericalEarth.Lands:
    AbstractUrbanRoughness, MorphometricRoughness,
    IsotropicFrontalArea, EmpiricalFrontalArea,
    UniformHeight, VariableHeight,
    urban_roughness, aerodynamic_parameters, compute_aerodynamic_roughness!, frontal_area_index,
    packing_displacement_ratio, drag_partition_roughness_ratio,
    maximum_height_displacement, height_spread_roughness,
    height_spread, maximum_element_height

using Oceananigans.Fields: interior, set!

uniform_height_closure(; kw...) = MorphometricRoughness(; height_distribution = UniformHeight(), kw...)

roughness_at_point(λᵖ, h; closure = MorphometricRoughness()) = aerodynamic_parameters(closure, λᵖ, h)[1]
displacement_at_point(λᵖ, h; closure = MorphometricRoughness()) = aerodynamic_parameters(closure, λᵖ, h)[2]

@testset "Obstacle-array morphometric endpoints" begin
    A = MorphometricRoughness().array_constant
    # Displacement ratio: 0 at λᵖ→0, → cap at λᵖ→1, monotone increasing between.
    @test packing_displacement_ratio(0.0, A, 1.0) ≈ 0 atol = 1e-12
    @test packing_displacement_ratio(1.0, A, 1.0) ≈ 1     # the array closes into a smooth surface
    ratios = [packing_displacement_ratio(λᵖ, A, 1.0) for λᵖ in 0.05:0.05:0.9]
    @test issorted(ratios)
    @test packing_displacement_ratio(1.0, A, 0.9) == 0.9  # an explicit ceiling still binds

    # Roughness ratio vanishes with the frontal area (no obstacles → no form drag).
    @test drag_partition_roughness_ratio(0.0, 0.3, 1.2, 0.4, 1.0) == 0
end

@testset "Displacement is monotone in built fraction" begin
    h = 15.0
    for closure in (uniform_height_closure(), MorphometricRoughness())
        displacements = [displacement_at_point(λᵖ, h; closure) for λᵖ in 0.02:0.02:0.9]
        @test issorted(displacements)
        @test all(≥(0), displacements)
    end

    # The uniform-height branch keeps d below roof level; the variable-height branch is
    # bounded by the tallest element instead.
    @test all(≤(h), [displacement_at_point(λᵖ, h; closure = uniform_height_closure())
                     for λᵖ in 0.02:0.02:0.9])
    distribution = MorphometricRoughness().height_distribution
    hᵐᵃˣ = maximum_element_height(distribution, h, height_spread(distribution, h))
    @test all(≤(hᵐᵃˣ), [displacement_at_point(λᵖ, h) for λᵖ in 0.02:0.02:0.9])
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

    # Full coverage closes the array back into a smooth surface: the displacement reaches
    # roof level and the roughness collapses onto the bare-soil floor.
    ℓᶜ, dᶜ = aerodynamic_parameters(uniform, 1.0, h)
    @test dᶜ / h ≈ uniform.maximum_displacement_ratio ≈ 1
    @test ℓᶜ ≈ uniform.bare_soil_roughness

    # And ℓᵐ decays monotonically into that limit rather than turning back up.
    skimming = [aerodynamic_parameters(uniform, λᵖ, h)[1] for λᵖ in 0.7:0.05:1.0]
    @test issorted(skimming; rev = true)

    # Invalid inputs become honest NaN gaps.
    for (λᵖ, hᵢ) in ((NaN, h), (0.3, NaN), (0.3, -5.0))
        ℓᵐ, d = aerodynamic_parameters(variable, λᵖ, hᵢ)
        @test isnan(ℓᵐ) && isnan(d)
    end
end

@testset "Empirical frontal area, height spread and tallest element" begin
    # λᶠ = 1.42λᵖ² + 0.4λᵖ, and the λᶠ < 2λᵖ envelope observed across real cities.
    @test frontal_area_index(EmpiricalFrontalArea(), 0.3, 15.0) ≈ 1.42 * 0.09 + 0.4 * 0.3
    for λᵖ in 0.05:0.05:1.0
        λᶠ = frontal_area_index(EmpiricalFrontalArea(), λᵖ, 15.0)
        @test 0 < λᶠ < 2λᵖ
    end
    @test frontal_area_index(EmpiricalFrontalArea(), 0.0, 15.0) == 0

    # σʰ = 1.05h − 3.7, floored at zero below the one-storey zero crossing.
    variable = VariableHeight()
    @test height_spread(variable, 15.0) ≈ 1.05 * 15 - 3.7
    @test height_spread(variable, 2.0) == 0
    @test height_spread(variable, 0.0) == 0

    # hᵐᵃˣ = 12.51·σʰ^0.77, floored at σʰ + h so the displacement parameter stays in [0, 1].
    σʰ = height_spread(variable, 15.0)
    @test maximum_element_height(variable, 15.0, σʰ) ≈ 12.51 * σʰ^0.77
    @test maximum_element_height(variable, 15.0, 0.0) == 15.0
    for h in 1.0:1.0:60.0
        s = height_spread(variable, h)
        @test maximum_element_height(variable, h, s) ≥ s + h   # keeps X ≤ 1
    end
end

@testset "Displacement is bounded by the tallest element, not the mean height" begin
    # Kanda's parametrization exists because d exceeds the mean building height over a
    # height-heterogeneous city; a d ≤ h cap would clip exactly that.
    variable = MorphometricRoughness()
    for h in (10.0, 15.0, 20.0, 25.0)
        _, d = aerodynamic_parameters(variable, 0.3, h)
        @test d > h
        distribution = variable.height_distribution
        σʰ = height_spread(distribution, h)
        @test d ≤ maximum_element_height(distribution, h, σʰ)
    end

    # The uniform-height branch keeps the Macdonald ceiling below roof level.
    uniform = uniform_height_closure()
    for h in (10.0, 15.0, 20.0, 25.0)
        _, d = aerodynamic_parameters(uniform, 0.9, h)
        @test d < h
        @test d ≤ uniform.maximum_displacement_ratio * h
    end
end

@testset "Frontal-area estimator and height-spread correction" begin
    # Isotropic λᶠ = λᵖ, exact for cubes.
    @test frontal_area_index(IsotropicFrontalArea(), 0.3, 15.0) == 0.3

    # The estimator choice changes the roughness (the dominant drag-partition uncertainty).
    isotropic = uniform_height_closure(frontal_area = IsotropicFrontalArea())
    empirical = uniform_height_closure(frontal_area = EmpiricalFrontalArea())
    @test aerodynamic_parameters(isotropic, 0.2, 15.0)[1] != aerodynamic_parameters(empirical, 0.2, 15.0)[1]

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
                                    frontal_area = IsotropicFrontalArea(),
                                    height_distribution = VariableHeight(height_spread_constants = (0.6, 0.0)))
    @test closure.array_constant == 3.59
    @test closure.bare_soil_roughness == 0.05
    @test closure.frontal_area isa IsotropicFrontalArea
    @test closure.height_distribution.height_spread_constants == (0.6, 0.0)

    # A narrower closure FT converts the sub-closures too.
    narrow = MorphometricRoughness(Float32)
    @test narrow.frontal_area.quadratic_coefficient isa Float32
    @test eltype(narrow.height_distribution.height_spread_constants) == Float32

    grid = LatitudeLongitudeGrid(CPU(), Float64; size = (3, 3),
                                 longitude = (0, 1), latitude = (0, 1),
                                 topology = (Bounded, Bounded, Flat))
    λᵖ = Field{Center, Center, Nothing}(grid); set!(λᵖ, 0)
    h  = Field{Center, Center, Nothing}(grid); set!(h, 15.0)

    # The bare-soil floor governs the on-grid result.
    ℓᵐ, _ = urban_roughness(h, λᵖ; closure)
    @test all(≈(0.05), interior(ℓᵐ))
end

@testset "Measured morphometry bypasses the input regressions" begin
    closure = MorphometricRoughness()
    λᵖ, h = 0.3, 15.0
    distribution = closure.height_distribution
    σʰ = height_spread(distribution, h)
    hᵐᵃˣ = maximum_element_height(distribution, h, σʰ)
    λᶠ = frontal_area_index(closure.frontal_area, λᵖ, h)

    # Fed its own regression values, the measured path reproduces the two-argument closure.
    @test aerodynamic_parameters(closure, λᵖ, h, σʰ, hᵐᵃˣ, λᶠ) == aerodynamic_parameters(closure, λᵖ, h)
    @test closure(λᵖ, h, σʰ, hᵐᵃˣ, λᶠ) == closure(λᵖ, h)

    # More frontal area → more form drag → rougher; a taller tallest element → deeper d.
    @test aerodynamic_parameters(closure, λᵖ, h, σʰ, hᵐᵃˣ, 2λᶠ)[1] >
          aerodynamic_parameters(closure, λᵖ, h, σʰ, hᵐᵃˣ, λᶠ)[1]
    @test aerodynamic_parameters(closure, λᵖ, h, σʰ, 2hᵐᵃˣ, λᶠ)[2] >
          aerodynamic_parameters(closure, λᵖ, h, σʰ, hᵐᵃˣ, λᶠ)[2]

    # UniformHeight ignores the height statistics: pure obstacle array with measured λᶠ.
    uniform = uniform_height_closure()
    @test aerodynamic_parameters(uniform, λᵖ, h, 9.0, 90.0, λᶠ) ==
          aerodynamic_parameters(uniform, λᵖ, h, 0.0, h, λᶠ)

    # A low-biased maximum height is floored at the mean, not trusted below it.
    @test aerodynamic_parameters(closure, λᵖ, h, σʰ, h / 2, λᶠ) ==
          aerodynamic_parameters(closure, λᵖ, h, σʰ, h, λᶠ)

    # Bare-soil floor and NaN gaps behave as in the two-argument form.
    ℓᵐ, d = aerodynamic_parameters(closure, 0.0, h, σʰ, hᵐᵃˣ, λᶠ)
    @test ℓᵐ ≈ closure.bare_soil_roughness && d == 0
    for arguments in ((NaN, h, σʰ, hᵐᵃˣ, λᶠ), (λᵖ, h, NaN, hᵐᵃˣ, λᶠ),
                      (λᵖ, h, σʰ, NaN, λᶠ), (λᵖ, h, σʰ, hᵐᵃˣ, NaN))
        ℓᵐ, d = aerodynamic_parameters(closure, arguments...)
        @test isnan(ℓᵐ) && isnan(d)
    end

    # Union-free with a closure FT ≠ input FT (kernel/GPU safety).
    for FT in (Float32, Float64)
        ℓᵐ, d = @inferred aerodynamic_parameters(MorphometricRoughness(FT), λᵖ, h, σʰ, hᵐᵃˣ, λᶠ)
        @test typeof(ℓᵐ) == typeof(d)
        @test isfinite(ℓᵐ) && isfinite(d)
    end
end

@testset "Measured-morphometry grid builder matches the scalar closure" begin
    grid = LatitudeLongitudeGrid(CPU(), Float64; size = (4, 4),
                                 longitude = (-74.0, -73.9), latitude = (40.7, 40.8),
                                 topology = (Bounded, Bounded, Flat))
    λᵖ   = Field{Center, Center, Nothing}(grid); set!(λᵖ, 0.4)
    h    = Field{Center, Center, Nothing}(grid); set!(h, 30.0)
    σʰ   = Field{Center, Center, Nothing}(grid); set!(σʰ, 25.0)
    hᵐᵃˣ = Field{Center, Center, Nothing}(grid); set!(hᵐᵃˣ, 250.0)
    λᶠ   = Field{Center, Center, Nothing}(grid); set!(λᶠ, 0.9)

    ℓᵐ, d = urban_roughness(h, λᵖ, σʰ, hᵐᵃˣ, λᶠ)
    ℓᵐref, dref = aerodynamic_parameters(MorphometricRoughness(), 0.4, 30.0, 25.0, 250.0, 0.9)
    @test all(≈(ℓᵐref), interior(ℓᵐ))
    @test all(≈(dref), interior(d))

    # The measured statistics change the answer relative to the regression-fed builder.
    ℓᵐreg, dreg = urban_roughness(h, λᵖ)
    @test !(interior(ℓᵐ) ≈ interior(ℓᵐreg))
    @test !(interior(d) ≈ interior(dreg))

    # Scalar properties sample through property_value like Fields do.
    compute_aerodynamic_roughness!(ℓᵐ, d, MorphometricRoughness(),
                                   (; plan_area_fraction = λᵖ, building_height = 30.0,
                                      height_deviation = 25.0, maximum_height = 250.0,
                                      frontal_area_index = 0.9), grid)
    @test all(≈(ℓᵐref), interior(ℓᵐ))
    @test all(≈(dref), interior(d))
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
