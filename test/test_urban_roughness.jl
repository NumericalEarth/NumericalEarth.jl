include("runtests_setup.jl")

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

    # σʰ = 1.05h − 3.7, floored at zero below the one-story zero crossing.
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
    isotropic = uniform_height_closure(frontal_area_estimator = IsotropicFrontalArea())
    empirical = uniform_height_closure(frontal_area_estimator = EmpiricalFrontalArea())
    @test aerodynamic_parameters(isotropic, 0.2, 15.0)[1] != aerodynamic_parameters(empirical, 0.2, 15.0)[1]

    # The spread correction reduces to a₁·(uniform-height ℓᵐ) for σʰ → 0.
    a₁ = VariableHeight().roughness_constants[1]
    @test height_spread_roughness(1.3, 0.3, 15.0, 0.0, a₁, 20.21, -0.77) ≈ a₁ * 1.3
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

for arch in test_architectures
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
        grid = LatitudeLongitudeGrid(arch, Float64; size = (3, 3),
                                    longitude = (0, 1), latitude = (0, 1),
                                    topology = (Bounded, Bounded, Flat))
        λᵖ = Field{Center, Center, Nothing}(grid); set!(λᵖ, 0.3)
        h  = Field{Center, Center, Nothing}(grid); set!(h, 15.0)
        ℓᵐ, d = urban_roughness(h, λᵖ; closure = MorphometricRoughness(Float32))
        @test all(isfinite, interior(ℓᵐ)) && all(isfinite, interior(d))
    end
end

for arch in test_architectures
    @testset "On-grid builder matches the scalar closure" begin
        grid = LatitudeLongitudeGrid(arch, Float64; size = (6, 6),
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
                                    (; plan_area_index = λᵖ, mean_building_height = h), grid)
        @test all(≈(MorphometricRoughness().bare_soil_roughness), interior(ℓᵐ))
        @test all(≈(0), interior(d))

        # Invalid inputs propagate to NaN gaps.
        set!(λᵖ, 0.3); set!(h, NaN)
        compute_aerodynamic_roughness!(ℓᵐ, d, MorphometricRoughness(),
                                    (; plan_area_index = λᵖ, mean_building_height = h), grid)
        @test all(isnan, interior(ℓᵐ))
        @test all(isnan, interior(d))
    end
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

for arch in test_architectures
    @testset "Closure construction and composition" begin
        # Drag-partition parameters and the height distribution are configured side by side,
        # on one flat closure.
        closure = MorphometricRoughness(array_constant = 3.59, bare_soil_roughness = 0.05,
                                        frontal_area_estimator = IsotropicFrontalArea(),
                                        height_distribution = VariableHeight(height_spread_constants = (0.6, 0.0)))
        @test closure.array_constant == 3.59
        @test closure.bare_soil_roughness == 0.05
        @test closure.frontal_area_estimator isa IsotropicFrontalArea
        @test closure.height_distribution.height_spread_constants == (0.6, 0.0)

        # A narrower closure FT converts the sub-closures too.
        narrow = MorphometricRoughness(Float32)
        @test narrow.frontal_area_estimator.quadratic_coefficient isa Float32
        @test eltype(narrow.height_distribution.height_spread_constants) == Float32

        grid = LatitudeLongitudeGrid(arch, Float64; size = (3, 3),
                                    longitude = (0, 1), latitude = (0, 1),
                                    topology = (Bounded, Bounded, Flat))
        λᵖ = Field{Center, Center, Nothing}(grid); set!(λᵖ, 0)
        h  = Field{Center, Center, Nothing}(grid); set!(h, 15.0)

        # The bare-soil floor governs the on-grid result.
        ℓᵐ, _ = urban_roughness(h, λᵖ; closure)
        @test all(≈(0.05), interior(ℓᵐ))
    end
end

@testset "Measured morphometry bypasses the input regressions" begin
    closure = MorphometricRoughness()
    λᵖ, h = 0.3, 15.0
    distribution = closure.height_distribution
    σʰ = height_spread(distribution, h)
    hᵐᵃˣ = maximum_element_height(distribution, h, σʰ)
    λᶠ = frontal_area_index(closure.frontal_area_estimator, λᵖ, h)

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

for arch in test_architectures
    @testset "Measured-morphometry grid builder matches the scalar closure" begin
        grid = LatitudeLongitudeGrid(arch, Float64; size = (4, 4),
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

        compute_aerodynamic_roughness!(ℓᵐ, d, MorphometricRoughness(),
                                    (; plan_area_index = λᵖ, mean_building_height = 30.0,
                                        building_height_deviation = 25.0, maximum_building_height = 250.0,
                                        frontal_area_index = 0.9), grid)
        @test all(≈(ℓᵐref), interior(ℓᵐ))
        @test all(≈(dref), interior(d))
    end
end

for arch in test_architectures
    @testset "Cell contract and mixed-type property sampling" begin
        closure = MorphometricRoughness()
        # The cell contract reads only the closure's own keys and matches the scalar form.
        cell = (; plan_area_index = 0.3, mean_building_height = 15.0, latitude = 51.5)
        @test aerodynamic_parameters(closure, cell) == aerodynamic_parameters(closure, 0.3, 15.0)

        grid = LatitudeLongitudeGrid(arch, Float64; size = (4, 4),
                                    longitude = (-0.1, 0.1), latitude = (51.4, 51.6),
                                    topology = (Bounded, Bounded, Flat))
        ℓᵐ = Field{Center, Center, Nothing}(grid)
        d  = Field{Center, Center, Nothing}(grid)
        λᵖ = Field{Center, Center, Nothing}(grid); set!(λᵖ, 0.3)
        compute_aerodynamic_roughness!(ℓᵐ, d, closure,
                                    (; plan_area_index = λᵖ, mean_building_height = 15.0), grid)
        ℓᵐref, dref = aerodynamic_parameters(closure, 0.3, 15.0)
        @test all(≈(ℓᵐref), interior(ℓᵐ))
        @test all(≈(dref), interior(d))
    end
end

#####
##### Dataset-level evaluation: a dataset supplying measured morphometry selects the measured closure.
#####

struct UniformBuildings end

function NumericalEarth.Lands.building_morphometry(grid, ::UniformBuildings)
    uniform(value) = set!(Field{Center, Center, Nothing}(grid), value)
    return (; plan_area_index = uniform(0.4), mean_building_height = uniform(25.0),
              building_height_deviation = uniform(8.0), maximum_building_height = uniform(60.0),
              frontal_area_index = uniform(0.3), gross_building_height = uniform(10.0))
end

@testset "Dataset-level urban roughness from measured morphometry" begin
    grid = LatitudeLongitudeGrid(CPU(), Float64; size = (2, 2),
                                 longitude = (0, 0.2), latitude = (0, 0.2),
                                 topology = (Bounded, Bounded, Flat))
    closure = MorphometricRoughness()
    fields = urban_roughness(grid, UniformBuildings(); closure, surface_layer_height = 100, neighborhood = 5000)
    @test keys(fields) == (:momentum_roughness_length, :zero_plane_displacement, :urban_fraction, :building_height)

    ℓᵐ, d = aerodynamic_parameters(closure, 0.4, 25.0, 8.0, 60.0, 0.3)
    @test all(interior(fields.momentum_roughness_length, :, :, 1) .≈ ℓᵐ)
    @test all(interior(fields.zero_plane_displacement, :, :, 1) .≈ d)
    @test all(interior(fields.urban_fraction, :, :, 1) .== 1)
    @test all(interior(fields.building_height, :, :, 1) .≈ 25)
end

struct TwoNeighborhoods end

function NumericalEarth.Lands.building_morphometry(grid, ::TwoNeighborhoods)
    western(western_value, eastern_value) =
        set!(Field{Center, Center, Nothing}(grid), (λ, φ) -> λ < 0.01 ? western_value : eastern_value)
    return (; plan_area_index = western(0.2, 0.8), mean_building_height = western(10.0, 50.0))
end

@testset "Dataset-level urban roughness preserves the area-mean neutral drag" begin
    # One grid cell holding two equal-area neighborhoods of different density and height.
    grid = LatitudeLongitudeGrid(CPU(), Float64; size = (1, 1),
                                 longitude = (0, 0.02), latitude = (0, 0.01),
                                 topology = (Bounded, Bounded, Flat))
    closure = uniform_height_closure()
    z = 100
    fields = urban_roughness(grid, TwoNeighborhoods(); closure, surface_layer_height = z, neighborhood = 1100)

    drag(ℓᵐ, d) = (closure.von_karman_constant / log((z - d) / ℓᵐ))^2
    ℓ₁, d₁ = aerodynamic_parameters(closure, 0.2, 10.0)
    ℓ₂, d₂ = aerodynamic_parameters(closure, 0.8, 50.0)
    C₁, C₂ = drag(ℓ₁, d₁), drag(ℓ₂, d₂)
    ℓᵐ = interior(fields.momentum_roughness_length)[1]
    d  = interior(fields.zero_plane_displacement)[1]

    @test drag(ℓᵐ, d) ≈ (C₁ + C₂) / 2
    @test d ≈ (C₁ * d₁ + C₂ * d₂) / (C₁ + C₂)
    @test interior(fields.building_height)[1] ≈ (0.2 * 10 + 0.8 * 50) / (0.2 + 0.8)
    @test interior(fields.urban_fraction)[1] == 1
end

struct AlternatingBuildings
    measured :: Bool
end

function NumericalEarth.Lands.building_morphometry(grid, buildings::AlternatingBuildings)
    odd = isodd.(1:size(grid, 1))
    alternating(odd_value, even_value) =
        (field = Field{Center, Center, Nothing}(grid); interior(field) .= ifelse.(odd, odd_value, even_value); field)
    properties = (; plan_area_index = alternating(0.6, 0.2), mean_building_height = alternating(30.0, 10.0))
    buildings.measured || return properties
    return (; properties..., building_height_deviation = alternating(0.0, 0.0),
              maximum_building_height = alternating(30.0, 10.0), frontal_area_index = alternating(0.5, 0.1))
end

@testset "On a grid finer than the neighborhood the closure sees the pooled neighborhood" begin
    # 100 m cells alternating dense-tall and sparse-low; a 500 m neighborhood pools five cells.
    Nx = 8
    grid = LatitudeLongitudeGrid(CPU(), Float64; size = (Nx, 1),
                                 longitude = (0, Nx * 0.0009), latitude = (0, 0.0009),
                                 topology = (Bounded, Bounded, Flat))
    closure = MorphometricRoughness()

    for measured in (false, true)
        fields = urban_roughness(grid, AlternatingBuildings(measured); closure, surface_layer_height = 100, neighborhood = 500)
        roughness    = interior(fields.momentum_roughness_length, :, 1, 1)
        displacement = interior(fields.zero_plane_displacement, :, 1, 1)
        height       = interior(fields.building_height, :, 1, 1)

        # Cell 3 pools three dense cells and two sparse ones; cell 4 the reverse.
        for (i, dense) in ((3, 3), (4, 2))
            sparse = 5 - dense
            built  = 0.6dense + 0.2sparse
            λᵖ = built / 5
            h  = (0.6 * 30 * dense + 0.2 * 10 * sparse) / built
            σʰ = sqrt((0.6 * 30^2 * dense + 0.2 * 10^2 * sparse) / built - h^2)
            λᶠ = (0.5dense + 0.1sparse) / 5
            ℓᵐ, d = measured ? aerodynamic_parameters(closure, λᵖ, h, σʰ, 30.0, λᶠ) : aerodynamic_parameters(closure, λᵖ, h)
            @test roughness[i] ≈ ℓᵐ
            @test displacement[i] ≈ d
            @test height[i] ≈ h
        end
    end
end

struct PatchBuildings end

function NumericalEarth.Lands.building_morphometry(grid, ::PatchBuildings)
    fraction = set!(Field{Center, Center, Nothing}(grid), (λ, φ) -> 0.018 < λ < 0.027 ? 0.6 : 0.0)
    height = set!(Field{Center, Center, Nothing}(grid), (λ, φ) -> 0.018 < λ < 0.027 ? 20.0 : NaN)
    return (; plan_area_index = fraction, mean_building_height = height)
end

for arch in test_architectures
    @testset "Stretched grids resolve the urban patch on $arch" begin
        grid = LatitudeLongitudeGrid(arch; size = (3, 1, 1),
                                     longitude = [0, 0.009, 0.018, 0.918], latitude = (0, 0.009),
                                     z = (0, 10), topology = (Bounded, Bounded, Bounded))
        fields = urban_roughness(grid, PatchBuildings(); surface_layer_height = 100)
        @test Array(interior(fields.urban_fraction))[:, 1, 1] ≈ [0, 0, 0.01]
        @test Array(interior(fields.building_height))[:, 1, 1] ≈ [0, 0, 20]
        expected, _ = aerodynamic_parameters(MorphometricRoughness(), 0.6, 20)
        @test Array(interior(fields.momentum_roughness_length))[end] ≈ expected
    end
end

@testset "Empty cells with missing height do not dilute urban height" begin
    grid = LatitudeLongitudeGrid(CPU(); size = (8, 1),
                                 longitude = (0.0171, 0.0243), latitude = (0, 0.0009),
                                 topology = (Bounded, Bounded, Flat))
    fields = urban_roughness(grid, PatchBuildings(); surface_layer_height = 100, neighborhood = 500)
    @test all(isfinite, interior(fields.momentum_roughness_length))
    @test all(≈(20), interior(fields.building_height))
end

struct LatitudeBandBuildings end

function NumericalEarth.Lands.building_morphometry(grid, ::LatitudeBandBuildings)
    fraction = set!(Field{Center, Center, Nothing}(grid), (λ, φ) -> φ < 60 ? 0.5 : 0.0)
    return (; plan_area_index = fraction, mean_building_height = 20.0)
end

@testset "Urban area uses spherical cell areas" begin
    grid = LatitudeLongitudeGrid(CPU(); size = (1, 1), longitude = (0, 1), latitude = (0, 80),
                                 topology = (Bounded, Bounded, Flat))
    fields = urban_roughness(grid, LatitudeBandBuildings(); surface_layer_height = 100,
                             neighborhood = grid.radius * deg2rad(20))
    @test only(interior(fields.urban_fraction)) ≈ sind(60) / sind(80)
    @test only(interior(fields.building_height)) ≈ 20
end

struct PrescribedBuildings{P}
    properties :: P
end

function NumericalEarth.Lands.building_morphometry(grid, buildings::PrescribedBuildings)
    return map(value -> set!(Field{Center, Center, Nothing}(grid), value), buildings.properties)
end

for arch in test_architectures
    @testset "Invalid urban neighborhoods on $arch" begin
        grid = LatitudeLongitudeGrid(arch; size = (1, 1), longitude = (0, 0.02), latitude = (0, 0.01),
                                     topology = (Bounded, Bounded, Flat))
        properties = (; plan_area_index = 0.4,
                        mean_building_height = (λ, φ) -> ifelse(λ < 0.01, 20.0, NaN))
        for measured in (false, true)
            inputs = measured ? (; properties..., building_height_deviation = 0.0,
                                    maximum_building_height = 20.0, frontal_area_index = 0.3) : properties
            @test_throws ArgumentError urban_roughness(grid, PrescribedBuildings(inputs);
                                                       surface_layer_height = 100, neighborhood = 1100)
        end
    end

    @testset "Missing morphometry in unbuilt cells on $arch" begin
        grid = LatitudeLongitudeGrid(arch; size = (3, 1), longitude = (0, 0.0027), latitude = (0, 0.0009),
                                     topology = (Bounded, Bounded, Flat))
        built(value) = (λ, φ) -> λ < 0.0009 ? NaN : value
        buildings = PrescribedBuildings((; plan_area_index = (λ, φ) -> λ < 0.0009 ? 0.0 : 0.4,
                                           mean_building_height = built(20.0), building_height_deviation = built(0.0),
                                           maximum_building_height = built(20.0), frontal_area_index = built(0.3)))
        closure = uniform_height_closure()
        fields = urban_roughness(grid, buildings; closure, surface_layer_height = 100, neighborhood = 500)
        ℓᵐ, d = aerodynamic_parameters(closure, 0.4 * 2 / 3, 20.0, 0.0, 20.0, 0.3 * 2 / 3)
        @test all(≈(ℓᵐ), Array(interior(fields.momentum_roughness_length)))
        @test all(≈(d), Array(interior(fields.zero_plane_displacement)))
        @test all(≈(20), Array(interior(fields.building_height)))

        # Raster height steps can assign a neighboring wall's frontal area to an unbuilt cell.
        buildings = PrescribedBuildings((; buildings.properties...,
                                           frontal_area_index = (λ, φ) -> λ < 0.0009 ? 0.15 : 0.3))
        fields = urban_roughness(grid, buildings; closure, surface_layer_height = 100, neighborhood = 500)
        ℓᵐ, _ = aerodynamic_parameters(closure, 0.4 * 2 / 3, 20.0, 0.0, 20.0, 0.25)
        @test all(≈(ℓᵐ), Array(interior(fields.momentum_roughness_length)))
    end

    @testset "Float32 thin cells pool by spherical area on $arch" begin
        # 14 m rows at 51.5°N, with faces exact in Float32, alternating dense-tall and sparse-low;
        # a 68 m neighborhood pools five rows.
        Ny = 9
        Δφ = 2.0^-13
        grid = LatitudeLongitudeGrid(arch, Float32; size = (1, Ny), longitude = (0, 0.0001),
                                     latitude = (51.5, 51.5 + Ny * Δφ), topology = (Bounded, Bounded, Flat))
        odd = isodd.(1:Ny)'
        λᵖ = ifelse.(odd, 0.6, 0.2)
        h  = ifelse.(odd, 30.0, 10.0)
        fields = urban_roughness(grid, PrescribedBuildings((; plan_area_index = λᵖ, mean_building_height = h));
                                 surface_layer_height = 100, neighborhood = 68)

        φᶠ = 51.5 .+ (0:Ny) .* Δφ
        area = sind.(φᶠ[2:end]) .- sind.(φᶠ[1:end-1])
        rows(j) = max(1, j - 2):min(Ny, j + 2)
        expected = [sum(λᵖ[b] * h[b] * area[b] for b in rows(j)) / sum(λᵖ[b] * area[b] for b in rows(j)) for j in 1:Ny]
        @test Array(interior(fields.building_height))[1, :, 1] ≈ expected rtol = 1e-5
    end
end

@testset "Gap filling to an unbuilt surface" begin
    closure = MorphometricRoughness()
    grid = LatitudeLongitudeGrid(CPU(), Float64; size = (4, 4),
                                 longitude = (-0.1, 0.1), latitude = (51.4, 51.6),
                                 topology = (Bounded, Bounded, Flat))
    Nx = size(grid, 1)
    built_roughness, built_displacement = aerodynamic_parameters(closure, 0.3, 15.0)
    ℓˢᵒⁱˡ, dˢᵒⁱˡ = aerodynamic_parameters(closure, 0, 0)

    # A building dataset with a missing tile: the closure marks it NaN by design.
    λᵖ = Field{Center, Center, Nothing}(grid)
    h  = Field{Center, Center, Nothing}(grid)
    set!(λᵖ, (λ, φ) -> ifelse(λ < 0, 0.3, NaN))
    set!(h,  (λ, φ) -> ifelse(λ < 0, 15.0, NaN))
    roughness_and_displacement() = urban_roughness(h, λᵖ; closure)

    ℓᵐ, d = roughness_and_displacement()
    @test any(isnan, interior(ℓᵐ))
    fill_aerodynamic_roughness_gaps!(ℓᵐ, d, closure)
    @test all(isfinite, interior(ℓᵐ)) && all(isfinite, interior(d))
    @test ℓᵐ[1, 1, 1] ≈ built_roughness && d[1, 1, 1] ≈ built_displacement   # built cells are untouched
    @test ℓᵐ[Nx, 1, 1] ≈ ℓˢᵒⁱˡ && d[Nx, 1, 1] ≈ dˢᵒⁱˡ                          # the gap becomes unbuilt surface

    # The unbuilt surface can be prescribed, e.g. open water.
    ℓᵐ, d = roughness_and_displacement()
    fill_aerodynamic_roughness_gaps!(ℓᵐ, d, closure; unbuilt = (1e-4, 0))
    @test ℓᵐ[Nx, 1, 1] ≈ 1e-4 && d[Nx, 1, 1] == 0
    @test ℓᵐ[1, 1, 1] ≈ built_roughness

    # A non-positive roughness is a gap too.
    ℓᵐ, d = roughness_and_displacement()
    ℓᵐ[1, 1, 1] = 0
    fill_aerodynamic_roughness_gaps!(ℓᵐ, d, closure)
    @test ℓᵐ[1, 1, 1] ≈ ℓˢᵒⁱˡ && d[1, 1, 1] == 0

    # The unbuilt surface itself must be evaluable.
    @test_throws ArgumentError fill_aerodynamic_roughness_gaps!(ℓᵐ, d, closure; unbuilt = (0, 0))
    @test_throws ArgumentError fill_aerodynamic_roughness_gaps!(ℓᵐ, d, closure; unbuilt = (0.03, -1))
end
