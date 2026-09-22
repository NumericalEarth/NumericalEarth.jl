include("runtests_setup.jl")

using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters, PrescribedAtmosphere
using NumericalEarth.Lands: DryLand
using NumericalEarth.EarthSystemModels.InterfaceComputations: RelativeVelocity,
                                                              SimilarityScales,
                                                              default_atmosphere_land_fluxes,
                                                              iterate_interface_fluxes,
                                                              monin_obukhov_length,
                                                              saturation_specific_humidity
using Thermodynamics

# Single land column forced by a uniform prescribed atmosphere, after `test_slab_land.jl`.
function single_column_land_model(arch, FT = Float64;
                                  surface_layer_height = 10,
                                  uᵃᵗ = 5, vᵃᵗ = 0, Tᵃᵗ = 288, qᵃᵗ = 0.003, pᵃᵗ = 101325,
                                  Tₛ = 293,
                                  hydrology = DryLand(),
                                  water_storage = 0,
                                  fluxes = nothing)

    grid = LatitudeLongitudeGrid(arch, FT; size = 1, latitude = 10, longitude = 10,
                                 z = (-1, 0), topology = (Flat, Flat, Bounded))

    atmosphere = PrescribedAtmosphere(grid; surface_layer_height, boundary_layer_height = 512)
    fill!(parent(atmosphere.temperature),       Tᵃᵗ)
    fill!(parent(atmosphere.specific_humidity), qᵃᵗ)
    fill!(parent(atmosphere.velocities.u), uᵃᵗ)
    fill!(parent(atmosphere.velocities.v), vᵃᵗ)
    fill!(parent(atmosphere.pressure),     pᵃᵗ)

    land = SlabLand(grid; hydrology, energy = SlabEnergy(FT))
    set!(land; T = Tₛ)
    fill!(land.water_storage, water_storage)

    atmosphere_land_fluxes = isnothing(fluxes) ? default_atmosphere_land_fluxes(land, FT) : fluxes
    model = AtmosphereLandModel(atmosphere, land; atmosphere_land_fluxes, radiation = nothing)
    update_state!(model.land)
    update_state!(model)

    return model
end

column(field) = Array(interior(field))[1, 1, 1]

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

@testset "Interface specific humidity" begin
    for arch in test_architectures
        Tₛ = 293.0
        pᵃᵗ = 101325.0

        dry = single_column_land_model(arch; Tₛ, pᵃᵗ, hydrology = DryLand())
        @test column(dry.interfaces.atmosphere_land_interface.specific_humidity) == 0

        wet = single_column_land_model(arch; Tₛ, pᵃᵗ,
                                       hydrology = BucketHydrology(Float64; maximum_water_storage = 10),
                                       water_storage = 5)

        ℂᵃᵗ = wet.atmosphere.thermodynamics_parameters
        qᵛ⁺ = saturation_specific_humidity(ℂᵃᵗ, Tₛ, pᵃᵗ, Thermodynamics.Liquid())
        @test column(wet.interfaces.atmosphere_land_interface.specific_humidity) ≈ qᵛ⁺
    end
end
