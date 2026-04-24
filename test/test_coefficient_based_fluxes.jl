include("runtests_setup.jl")

using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CoefficientBasedFluxes,
    PolynomialNeutralDragCoefficient,
    LargeYeagerTransferCoefficients,
    LinearStableStabilityFunction,
    large_yeager_stability_functions,
    SimilarityScales,
    ComponentInterfaces,
    stability_profile,
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

@testset "LargeYeagerTransferCoefficients constructor" begin
    ly = LargeYeagerTransferCoefficients()
    @test ly isa LargeYeagerTransferCoefficients{Float64}
    @test ly.reference_height ≈ 10.0
    @test ly.stanton_stable ≈ 18.0
    @test ly.stanton_unstable ≈ 32.7
    @test ly.dalton ≈ 34.6
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
        atmosphere = JRA55PrescribedAtmosphere(arch, Float64; end_date=dates[2], backend=InMemory())

        constant_fluxes = CoefficientBasedFluxes(transfer_coefficients = SimilarityScales(2e-3, 2e-3, 2e-3))
        interfaces = ComponentInterfaces(atmosphere, ocean;
                                         atmosphere_ocean_fluxes=constant_fluxes)

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
        atmosphere = JRA55PrescribedAtmosphere(arch, Float64; end_date=dates[2], backend=InMemory())

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
        atmosphere = JRA55PrescribedAtmosphere(arch, Float64; end_date=dates[2], backend=InMemory())

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
