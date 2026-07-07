include("runtests_setup.jl")

using NumericalEarth.EarthSystemModels.InterfaceComputations:
    FairallMomentumStabilityFunction,
    FairallScalarStabilityFunction,
    ChengBrutsaertMomentumStabilityFunction,
    ChengBrutsaertScalarStabilityFunction,
    LinearStableStabilityFunction,
    SplitStabilityFunction,
    SimilarityScales,
    jimenez_stability_functions,
    stability_profile

# Independent reference implementations of the Jiménez et al. (2012) formulas,
# written directly from the paper (not from the source under test).
ref_kansas_momentum(ζ) = (x = (1 - 16ζ)^(1/4);
                          2log((1 + x) / 2) + log((1 + x^2) / 2) - 2atan(x) + π/2)
ref_kansas_scalar(ζ)   = (x = (1 - 16ζ)^(1/4); 2log((1 + x^2) / 2))
ref_convective(ζ, a)   = (y = (1 - a * ζ)^(1/3);
                          3/2 * log((y^2 + y + 1) / 3) - sqrt(3) * atan((2y + 1) / sqrt(3)) + π/sqrt(3))
ref_blend(ζ, ψᵏ, ψᶜ)   = (f = ζ^2 / (1 + ζ^2); (1 - f) * ψᵏ + f * ψᶜ)

ref_fairall_momentum(ζ) = ref_blend(ζ, ref_kansas_momentum(ζ), ref_convective(ζ, 10))
ref_fairall_scalar(ζ)   = ref_blend(ζ, ref_kansas_scalar(ζ), ref_convective(ζ, 34))

ref_cheng_brutsaert(ζ, a, b) = -a * log(ζ + (1 + ζ^b)^(1/b))

@testset "Fairall (F96) unstable stability functions" begin
    ψm = FairallMomentumStabilityFunction{Float64}()
    ψh = FairallScalarStabilityFunction{Float64}()

    for ζ in (-5.0, -1.0, -0.1)
        @test stability_profile(ψm, ζ) ≈ ref_fairall_momentum(ζ) rtol=1e-6
        @test stability_profile(ψh, ζ) ≈ ref_fairall_scalar(ζ) rtol=1e-6

        # Unstable conditions enhance mixing: ψ > 0
        @test stability_profile(ψm, ζ) > 0
        @test stability_profile(ψh, ζ) > 0
    end

    # Self-guard: zero at neutral and across the stable branch, so that
    # `SplitStabilityFunction` can safely evaluate both branches.
    for ζ in (0.0, 0.1, 1.0, 10.0)
        @test abs(stability_profile(ψm, ζ)) < 1e-10
        @test abs(stability_profile(ψh, ζ)) < 1e-10
    end
end

@testset "Cheng-Brutsaert (CB05) stable stability functions" begin
    ψm = ChengBrutsaertMomentumStabilityFunction{Float64}()
    ψh = ChengBrutsaertScalarStabilityFunction{Float64}()

    for ζ in (0.1, 1.0, 10.0)
        @test stability_profile(ψm, ζ) ≈ ref_cheng_brutsaert(ζ, 6.1, 2.5) rtol=1e-6
        @test stability_profile(ψh, ζ) ≈ ref_cheng_brutsaert(ζ, 5.3, 1.1) rtol=1e-6

        # Stable conditions suppress mixing: ψ < 0
        @test stability_profile(ψm, ζ) < 0
        @test stability_profile(ψh, ζ) < 0
    end

    # Self-guard: zero at neutral and across the unstable branch (no NaN from
    # fractional powers of negative arguments).
    for ζ in (-10.0, -1.0, -0.1, 0.0)
        @test abs(stability_profile(ψm, ζ)) < 1e-10
        @test abs(stability_profile(ψh, ζ)) < 1e-10
    end

    # CB05 suppresses fluxes more mildly than the linear Large-Yeager form
    # at strong stability -- the documented asset of the Jiménez scheme.
    linear = LinearStableStabilityFunction{Float64}()
    @test stability_profile(ψm, 10.0) > stability_profile(linear, 10.0)
end

@testset "jimenez_stability_functions factory" begin
    sf = jimenez_stability_functions()
    @test sf isa SimilarityScales
    @test sf.momentum isa SplitStabilityFunction
    @test sf.temperature isa SplitStabilityFunction
    @test sf.water_vapor isa SplitStabilityFunction

    # Each regime dispatches to the corresponding component function
    @test stability_profile(sf.momentum, -1.0) ≈ ref_fairall_momentum(-1.0)
    @test stability_profile(sf.momentum,  1.0) ≈ ref_cheng_brutsaert(1.0, 6.1, 2.5)
    @test stability_profile(sf.temperature, -1.0) ≈ ref_fairall_scalar(-1.0)
    @test stability_profile(sf.temperature,  1.0) ≈ ref_cheng_brutsaert(1.0, 5.3, 1.1)

    # No NaN/Inf across the neutral point, where both branches are evaluated
    for ψ in (sf.momentum, sf.temperature, sf.water_vapor)
        for ζ in range(-5, 5, length=101)
            @test isfinite(stability_profile(ψ, ζ))
        end
        @test abs(stability_profile(ψ, 0.0)) < 1e-10
    end

    # Float32 factory produces Float32 results
    sf32 = jimenez_stability_functions(Float32)
    @test stability_profile(sf32.momentum, -1f0) isa Float32
    @test stability_profile(sf32.momentum, 1f0) isa Float32
end
