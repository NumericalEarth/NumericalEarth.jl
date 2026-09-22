include("runtests_setup.jl")

using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters
using NumericalEarth.EarthSystemModels.InterfaceComputations: RelativeVelocity,
                                                              SimilarityScales,
                                                              iterate_interface_fluxes,
                                                              monin_obukhov_length

@testset "Monin--Obukhov length" begin
    for FT in (Float32, Float64)
        u★ = FT(0.3)
        b★ = FT(2e-3)
        ϰ  = FT(0.4)

        @test monin_obukhov_length(u★, b★, ϰ) isa FT
        @test monin_obukhov_length(u★, b★, ϰ) ≈ u★^2 / (ϰ * b★)

        # A neutral interface has an infinite length scale, in the working precision.
        @test monin_obukhov_length(u★, zero(FT), ϰ) === FT(Inf)
        @test monin_obukhov_length(zero(FT), zero(FT), ϰ) === FT(Inf)

        # ... so the solver stays in the working precision too.
        zero_ψ(ζ) = zero(ζ)
        fluxes = SimilarityTheoryFluxes(FT;
                                        momentum_roughness_length    = FT(0.1),
                                        temperature_roughness_length = FT(0.01),
                                        water_vapor_roughness_length = FT(0.01),
                                        subgrid_velocities = nothing,
                                        stability_functions = SimilarityScales(zero_ψ, zero_ψ, zero_ψ))

        approximate_state     = (; fluxes = (; u★, θ★ = zero(FT), q★ = zero(FT)), u = zero(FT), v = zero(FT))
        atmosphere_state      = (; u = FT(5), v = zero(FT), p = FT(101325), h_bℓ = FT(512))
        interface_properties  = (; velocity_formulation = RelativeVelocity())
        atmosphere_properties = (; thermodynamics_parameters = AtmosphereThermodynamicsParameters(FT),
                                   gravitational_acceleration = FT(9.81))

        argument_types = (typeof(fluxes), FT, FT, FT, FT, FT,
                          typeof(approximate_state), typeof(atmosphere_state),
                          typeof(interface_properties), typeof(atmosphere_properties))

        @test Base.return_types(iterate_interface_fluxes, argument_types) == [Tuple{FT, FT, FT}]
    end
end
