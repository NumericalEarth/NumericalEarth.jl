include("runtests_setup.jl")

using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.Lands: DryLand
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters
using Thermodynamics

# `Adapt` is not a test dependency; reuse the binding loaded by InterfaceComputations
using NumericalEarth.EarthSystemModels.InterfaceComputations: adapt
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    ConvectiveGustiness,
    SubgridVelocityCorrection,
    mahrt_sun_subgrid_velocity,
    subgrid_velocity²,
    atmosphere_land_stability_functions

@testset "ConvectiveGustiness" begin
    g = ConvectiveGustiness{Float64}()
    @test g.gustiness_parameter == 1.2
    @test g.minimum_gustiness == 0.01

    # Bit-identity with the formerly-inline kernel formula: exact `==`, not `≈`.
    for (u★, b★, h_bℓ) in ((0.3, -0.005, 600.0),   # unstable: Jᵇ > 0
                           (0.3,  0.005, 600.0),   # stable: floor applies
                           (1e-4, 0.0,   512.0))   # neutral: floor applies
        Jᵇ = - u★ * b★
        Uᴳ = max(g.minimum_gustiness, g.gustiness_parameter * cbrt(max(zero(Jᵇ), Jᵇ) * h_bℓ))
        @test subgrid_velocity²(g, u★, b★, h_bℓ) == Uᴳ^2
    end

    # Unstable enhancement exceeds the floor; stable falls back to it
    @test subgrid_velocity²(g, 0.3, -0.005, 600.0) > g.minimum_gustiness^2
    @test subgrid_velocity²(g, 0.3, 0.005, 600.0) == g.minimum_gustiness^2

    # Zeroed gustiness contributes exactly nothing (analytic log-law tests rely on this)
    g0 = ConvectiveGustiness{Float64}(gustiness_parameter = 0, minimum_gustiness = 0)
    @test subgrid_velocity²(g0, 0.3, -0.005, 600.0) == 0.0

    # Float32 purity
    g32 = ConvectiveGustiness{Float32}()
    @test subgrid_velocity²(g32, 0.1f0, -1f-3, 512f0) isa Float32
end

@testset "Mahrt-Sun mesoscale velocity" begin
    # No enhancement at or below the 5 km threshold; WRF form above it
    @test mahrt_sun_subgrid_velocity(1e3) == 0
    @test mahrt_sun_subgrid_velocity(5e3) == 0
    @test mahrt_sun_subgrid_velocity(25e3) ≈ 0.32 * 4^0.33
    @test mahrt_sun_subgrid_velocity(25e3) > mahrt_sun_subgrid_velocity(10e3) > 0
end

@testset "SubgridVelocityCorrection composition" begin
    # Components add in quadrature; `nothing` slots contribute exactly zero
    u★, b★, h_bℓ = 0.3, -0.005, 600.0
    g = ConvectiveGustiness{Float64}()
    vsg = mahrt_sun_subgrid_velocity(25e3)

    sv = SubgridVelocityCorrection(Float64; convective = g, mesoscale = vsg)
    @test subgrid_velocity²(sv, u★, b★, h_bℓ) ==
          subgrid_velocity²(g, u★, b★, h_bℓ) + vsg^2

    sv_nomeso = SubgridVelocityCorrection(Float64; convective = g)
    @test subgrid_velocity²(sv_nomeso, u★, b★, h_bℓ) == subgrid_velocity²(g, u★, b★, h_bℓ)

    sv_none = SubgridVelocityCorrection(Float64; convective = nothing)
    @test subgrid_velocity²(sv_none, u★, b★, h_bℓ) == 0

    # FT-aware constructor converts a Float64 mesoscale scalar (no Float64 leak
    # into Float32 kernels)
    sv32 = SubgridVelocityCorrection(Float32; mesoscale = mahrt_sun_subgrid_velocity(25e3))
    @test sv32.mesoscale isa Float32
    @test sv32.convective isa ConvectiveGustiness{Float32}
    @test isbits(sv32)
end

@testset "SimilarityTheoryFluxes construction and adapt" begin
    # Default: composite built from the legacy gustiness kwargs, no mesoscale
    fluxes = SimilarityTheoryFluxes()
    @test fluxes.subgrid_velocities isa SubgridVelocityCorrection
    @test fluxes.subgrid_velocities.convective isa ConvectiveGustiness{Float64}
    @test isnothing(fluxes.subgrid_velocities.mesoscale)

    # Legacy kwargs still parameterize the convective component
    z = SimilarityTheoryFluxes(gustiness_parameter = 0, minimum_gustiness = 0)
    @test z.subgrid_velocities.convective.gustiness_parameter == 0
    @test z.subgrid_velocities.convective.minimum_gustiness == 0

    # Explicit composite is used as-is
    sv = SubgridVelocityCorrection(Float64; mesoscale = mahrt_sun_subgrid_velocity(25e3))
    fluxes = SimilarityTheoryFluxes(subgrid_velocities = sv)
    @test fluxes.subgrid_velocities === sv

    # Float32 constructor produces a Float32 composite
    f32 = SimilarityTheoryFluxes(Float32)
    @test f32.subgrid_velocities.convective isa ConvectiveGustiness{Float32}
    @test isbits(f32.subgrid_velocities)

    # Explicitly supplied composites are converted to FT (no Float64 leak into
    # Float32 kernels)
    f32sv = SimilarityTheoryFluxes(Float32; subgrid_velocities = SubgridVelocityCorrection(Float64))
    @test f32sv.subgrid_velocities.convective isa ConvectiveGustiness{Float32}
    f32meso = SimilarityTheoryFluxes(Float32;
                                     subgrid_velocities = SubgridVelocityCorrection(Float64; mesoscale = mahrt_sun_subgrid_velocity(25e3)))
    @test f32meso.subgrid_velocities.mesoscale isa Float32

    # The composite survives an `adapt` roundtrip (GPU struct integrity)
    adapted = adapt(Array, fluxes)
    @test adapted.subgrid_velocities === fluxes.subgrid_velocities
end

@testset "Coupled single-column: mesoscale velocity increases calm-wind u★" begin
    for arch in test_architectures
        Tᵃᵗ = 288
        qᵃᵗ = 0.003
        cᵖᵐ = Thermodynamics.cp_m(AtmosphereThermodynamicsParameters(Float64), qᵃᵗ)
        g   = 9.80665
        neutral_skin = Tᵃᵗ + 10 / cᵖᵐ * g

        function calm_flux_response(mesoscale)
            grid = LatitudeLongitudeGrid(arch, Float64;
                                         size = 1, latitude = 10, longitude = 10,
                                         z = (-1, 0), topology = (Flat, Flat, Bounded))
            h = 10.0
            atmosphere = PrescribedAtmosphere(grid; surface_layer_height = h, boundary_layer_height = 512)
            @allowscalar begin
                fill!(parent(atmosphere.temperature),       Tᵃᵗ)
                fill!(parent(atmosphere.specific_humidity), qᵃᵗ)
                fill!(parent(atmosphere.velocities.u), 0.5)
                fill!(parent(atmosphere.velocities.v), 0)
                fill!(parent(atmosphere.pressure),     101325)
            end
            subgrid_velocities = SubgridVelocityCorrection(Float64; mesoscale)
            fluxes = SimilarityTheoryFluxes(; momentum_roughness_length    = 0.1,
                                              temperature_roughness_length = 0.01,
                                              water_vapor_roughness_length = 0.01,
                                              stability_functions = atmosphere_land_stability_functions(Float64),
                                              subgrid_velocities)
            land = SlabLand(grid; hydrology = DryLand(), energy = SlabEnergy(eltype(grid)))
            set!(land; T = neutral_skin)
            model = AtmosphereLandModel(atmosphere, land; atmosphere_land_fluxes = fluxes, radiation = nothing)
            update_state!(model)
            f = model.interfaces.atmosphere_land_interface.fluxes
            return (u★ = @allowscalar(f.friction_velocity[1, 1, 1]),
                    Q  = @allowscalar(f.sensible_heat[1, 1, 1]))
        end

        calm     = calm_flux_response(nothing)
        enhanced = calm_flux_response(mahrt_sun_subgrid_velocity(25e3))

        @test isfinite(calm.u★) && isfinite(enhanced.u★)
        @test isfinite(calm.Q) && isfinite(enhanced.Q)
        @test enhanced.u★ > calm.u★
    end
end
