include("runtests_setup.jl")

using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CoefficientBasedFluxes,
    PolynomialNeutralDragCoefficient,
    LargeYeagerTransferCoefficients,
    LinearStableStabilityFunction,
    large_yeager_stability_functions,
    atmosphere_land_stability_functions,
    FreeConvectionMomentumStabilityFunction,
    FreeConvectionScalarStabilityFunction,
    LogarithmicSimilarityProfile,
    similarity_profile,
    SimilarityScales,
    ComponentInterfaces,
    stability_profile,
    inner_stability_profile,
    PaulsonMomentumStabilityFunction,
    PaulsonScalarStabilityFunction,
    FixedIterations

using NumericalEarth.DataWrangling: all_dates

@testset "PolynomialNeutralDragCoefficient" begin
    p = PolynomialNeutralDragCoefficient()
    @test p isa PolynomialNeutralDragCoefficient{Float64}

    # Basic evaluation
    @test p(3.0) > 0
    @test p(3.0) < 5e-3

    # High wind cap
    @test p(40.0) ≈ 2.34e-3

    # Wind floor
    @test p(0.0) == p(0.5)

    # Monotonicity in moderate winds
    @test p(20.0) > p(5.0)

    # Float32
    p32 = PolynomialNeutralDragCoefficient(Float32)
    @test p32 isa PolynomialNeutralDragCoefficient{Float32}
    @test p32(10f0) isa Float32
end

@testset "LinearStableStabilityFunction and large_yeager_stability_functions" begin
    ψ = LinearStableStabilityFunction{Float64}()
    @test stability_profile(ψ, 0.0) ≈ 0.0
    @test stability_profile(ψ, 1.0) ≈ -5.0
    @test stability_profile(ψ, -1.0) ≈ 0.0
    @test stability_profile(ψ, 20.0) ≈ -50.0   # bounded at ζ_max = 10

    sf = large_yeager_stability_functions()
    @test sf isa SimilarityScales

    # Momentum: Paulson unstable + linear stable
    @test stability_profile(sf.momentum, -1.0) > 0
    @test stability_profile(sf.momentum, 1.0) ≈ -5.0
    @test stability_profile(sf.momentum, 0.0) ≈ 0.0 atol=1e-10

    # Scalar: same structure
    @test stability_profile(sf.temperature, -1.0) > 0
    @test stability_profile(sf.temperature, 1.0) ≈ -5.0
end

@testset "Zeng free-convection matched stability functions" begin
    ψu = FreeConvectionMomentumStabilityFunction{Float64}()
    ψc = FreeConvectionScalarStabilityFunction{Float64}()

    # Matched below its matching point to the Businger-Dyer form it extends
    pu = PaulsonMomentumStabilityFunction{Float64}()
    pc = PaulsonScalarStabilityFunction{Float64}()
    for ζ in (-1.0, -0.4, -0.01, 0.0)
        @test stability_profile(ψu, ζ) ≈ stability_profile(pu, ζ)
    end
    @test stability_profile(ψc, -0.4) ≈ stability_profile(pc, -0.4)

    # Continuously differentiable across the match
    for (ψ, ζm) in ((ψu, -1.574), (ψc, -0.465))
        ϵ = 1e-6
        @test abs(stability_profile(ψ, ζm - ϵ) - stability_profile(ψ, ζm + ϵ)) < 10ϵ
        left  = (stability_profile(ψ, ζm - ϵ) - stability_profile(ψ, ζm - 2ϵ)) / ϵ
        right = (stability_profile(ψ, ζm + 2ϵ) - stability_profile(ψ, ζm + ϵ)) / ϵ
        @test left ≈ right rtol=1e-4
    end

    # The roughness-height evaluation keeps the unmatched Businger-Dyer form
    @test inner_stability_profile(ψu, -100.0) == stability_profile(pu, -100.0)
    @test inner_stability_profile(ψc, -100.0) == stability_profile(pc, -100.0)

    # Beyond the match the matched profile turns over (the free-convection term
    # outgrows the log), while the unmatched Businger-Dyer form keeps growing
    @test stability_profile(ψu, -50.0) < stability_profile(pu, -50.0)
    ψu∞ = FreeConvectionMomentumStabilityFunction{Float64}(maximum_stability_parameter = Inf)
    @test stability_profile(ψu∞, -1e3) < stability_profile(ψu∞, -100.0)

    # `maximum_stability_parameter` freezes the profile beyond its bound
    @test stability_profile(ψu, -1e3) == stability_profile(ψu, -100.0)
end

@testset "atmosphere_land_stability_functions" begin
    sf = atmosphere_land_stability_functions()
    @test sf isa SimilarityScales

    # Stable branch bounded at ζ ≤ 2 (ocean large_yeager keeps ζ ≤ 10)
    @test stability_profile(sf.momentum, 1.0) ≈ -5.0
    @test stability_profile(sf.momentum, 20.0) ≈ -10.0
    @test stability_profile(sf.temperature, 20.0) ≈ -10.0

    # The invariant that matters: the similarity profile stays bounded away from
    # zero as ζ → -∞, so the transfer coefficients ϰ / Π stay finite. Unmatched
    # Businger-Dyer (the ocean form) collapses instead.
    form = LogarithmicSimilarityProfile()
    ly = large_yeager_stability_functions()
    for (h, ℓ) in ((10.0, 0.1), (10.0, 0.01), (10.0, 5.0))
        Πⁿ = log(h / ℓ)
        for ζ in (-1e2, -1e4, -1e8)
            L = h / ζ
            Π = similarity_profile(form, sf.momentum, h, ℓ, L)
            @test Π ≥ Πⁿ / 3
            @test Π > similarity_profile(form, ly.momentum, h, ℓ, L)
            @test similarity_profile(form, sf.temperature, h, ℓ, L) ≥ Πⁿ / 25
        end
    end

    # Ocean defaults untouched by the land form
    @test stability_profile(ly.momentum, -100.0) ≈ stability_profile(PaulsonMomentumStabilityFunction{Float64}(), -100.0)
    @test stability_profile(ly.momentum, 20.0) ≈ -50.0

    sf32 = atmosphere_land_stability_functions(Float32)
    @test stability_profile(sf32.momentum, 20f0) isa Float32
    @test stability_profile(sf32.momentum, -100f0) isa Float32
    @test similarity_profile(LogarithmicSimilarityProfile(), sf32.momentum, 10f0, 0.1f0, -1f0) isa Float32
end

@testset "LargeYeagerTransferCoefficients constructor" begin
    ly = LargeYeagerTransferCoefficients()
    @test ly isa LargeYeagerTransferCoefficients{Float64}
    @test ly.reference_height ≈ 10.0
    @test ly.stable_heat_transfer_coefficient ≈ 18.0
    @test ly.unstable_heat_transfer_coefficient ≈ 32.7
    @test ly.moisture_transfer_coefficient ≈ 34.6
    @test ly.neutral_drag_coefficient isa PolynomialNeutralDragCoefficient{Float64}

    ly32 = LargeYeagerTransferCoefficients(Float32)
    @test ly32 isa LargeYeagerTransferCoefficients{Float32}
end

@testset "CoefficientBasedFluxes with constant coefficients" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch;
                                     size = 1,
                                     latitude = 10,
                                     longitude = 10,
                                     z = (-1, 0),
                                     topology = (Flat, Flat, Bounded))

        ocean = ocean_simulation(grid;
                                 momentum_advection = nothing,
                                 tracer_advection = nothing,
                                 closure = nothing,
                                 bottom_drag_coefficient = 0)

        dates = all_dates(RepeatYearJRA55(), :temperature)
        atmosphere = JRA55PrescribedAtmosphere(arch; end_date=dates[2])

        constant_fluxes = CoefficientBasedFluxes(transfer_coefficients = SimilarityScales(2e-3, 2e-3, 2e-3))
        interfaces = ComponentInterfaces(atmosphere, ocean; atmosphere_ocean_fluxes=constant_fluxes)

        set!(ocean.model, T=15, S=35)
        coupled_model = OceanOnlyModel(ocean; atmosphere, interfaces)
        fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes

        CUDA.@allowscalar begin
            @test isfinite(fluxes.sensible_heat[1, 1, 1])
            @test isfinite(fluxes.latent_heat[1, 1, 1])
            @test isfinite(fluxes.water_vapor[1, 1, 1])
        end
    end
end

@testset "CoefficientBasedFluxes with PolynomialNeutralDragCoefficient" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch;
                                     size = 1,
                                     latitude = 10,
                                     longitude = 10,
                                     z = (-1, 0),
                                     topology = (Flat, Flat, Bounded))

        ocean = ocean_simulation(grid;
                                 momentum_advection = nothing,
                                 tracer_advection = nothing,
                                 closure = nothing,
                                 bottom_drag_coefficient = 0)

        dates = all_dates(RepeatYearJRA55(), :temperature)
        atmosphere = JRA55PrescribedAtmosphere(arch; end_date=dates[2])

        poly_drag = PolynomialNeutralDragCoefficient()
        poly_fluxes = CoefficientBasedFluxes(transfer_coefficients = SimilarityScales(poly_drag, 1e-3, 1e-3))

        interfaces = ComponentInterfaces(atmosphere, ocean;
                                         atmosphere_ocean_fluxes=poly_fluxes)

        set!(ocean.model, T=15, S=35)
        coupled_model = OceanOnlyModel(ocean; atmosphere, interfaces)
        fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes

        CUDA.@allowscalar begin
            @test isfinite(fluxes.sensible_heat[1, 1, 1])
            @test isfinite(fluxes.latent_heat[1, 1, 1])
            @test isfinite(fluxes.water_vapor[1, 1, 1])
            @test fluxes.friction_velocity[1, 1, 1] > 0
        end
    end
end

@testset "CoefficientBasedFluxes with LargeYeagerTransferCoefficients" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch;
                                     size = 1,
                                     latitude = 10,
                                     longitude = 10,
                                     z = (-1, 0),
                                     topology = (Flat, Flat, Bounded))

        ocean = ocean_simulation(grid;
                                 momentum_advection = nothing,
                                 tracer_advection = nothing,
                                 closure = nothing,
                                 bottom_drag_coefficient = 0)

        dates = all_dates(RepeatYearJRA55(), :temperature)
        atmosphere = JRA55PrescribedAtmosphere(arch; end_date=dates[2])

        ly = LargeYeagerTransferCoefficients()
        ly_fluxes = CoefficientBasedFluxes(transfer_coefficients = ly,
                                             solver_stop_criteria = FixedIterations(5))

        interfaces = ComponentInterfaces(atmosphere, ocean;
                                         atmosphere_ocean_fluxes=ly_fluxes)

        set!(ocean.model, T=15, S=35)
        coupled_model = OceanOnlyModel(ocean; atmosphere, interfaces)
        fluxes = coupled_model.interfaces.atmosphere_ocean_interface.fluxes

        CUDA.@allowscalar begin
            @test isfinite(fluxes.sensible_heat[1, 1, 1])
            @test isfinite(fluxes.latent_heat[1, 1, 1])
            @test isfinite(fluxes.water_vapor[1, 1, 1])
            @test fluxes.friction_velocity[1, 1, 1] > 0
        end
    end
end
