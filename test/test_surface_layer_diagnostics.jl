include("runtests_setup.jl")

using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters, PrescribedAtmosphere
using NumericalEarth.Lands: DryLand
using NumericalEarth.EarthSystemModels.InterfaceComputations: RelativeVelocity,
                                                              SimilarityScales,
                                                              WindDependentWaveFormulation,
                                                              atmosphere_land_stability_functions,
                                                              default_atmosphere_land_fluxes,
                                                              iterate_interface_fluxes,
                                                              monin_obukhov_length,
                                                              saturation_specific_humidity
using Thermodynamics

single_column_grid(arch, FT = Float64) =
    LatitudeLongitudeGrid(arch, FT; size = 1, latitude = 10, longitude = 10,
                          z = (-1, 0), topology = (Flat, Flat, Bounded))

# Single land column forced by a uniform prescribed atmosphere, after `test_slab_land.jl`.
function single_column_land_model(grid;
                                  surface_layer_height = 10,
                                  uᵃᵗ = 5, vᵃᵗ = 0, Tᵃᵗ = 288, qᵃᵗ = 0.003, pᵃᵗ = 101325,
                                  Tₛ = 293,
                                  hydrology = DryLand(),
                                  water_storage = 0,
                                  fluxes = nothing)

    FT = eltype(grid)
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

# The atmosphere--land flux defaults, with the pieces each testset varies left open.
similarity_fluxes(FT = Float64; kw...) =
    SimilarityTheoryFluxes(FT;
                           stability_functions          = atmosphere_land_stability_functions(FT),
                           momentum_roughness_length    = FT(0.1),
                           temperature_roughness_length = FT(0.01),
                           water_vapor_roughness_length = FT(0.01),
                           subgrid_velocities           = nothing,
                           kw...)

zero_stability_function(ζ) = zero(ζ)

neutral_stability_functions() = SimilarityScales(zero_stability_function,
                                                 zero_stability_function,
                                                 zero_stability_function)

column(field) = Array(interior(field))[1, 1, 1]
value(operation) = column(compute!(Field(operation)))

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
        fluxes = similarity_fluxes(FT; stability_functions = neutral_stability_functions())

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

        dry = single_column_land_model(single_column_grid(arch); Tₛ, pᵃᵗ, hydrology = DryLand())
        @test column(dry.interfaces.atmosphere_land_interface.specific_humidity) == 0

        wet = single_column_land_model(single_column_grid(arch); Tₛ, pᵃᵗ,
                                       hydrology = BucketHydrology(Float64; maximum_water_storage = 10),
                                       water_storage = 5)

        ℂᵃᵗ = wet.atmosphere.thermodynamics_parameters
        qᵛ⁺ = saturation_specific_humidity(ℂᵃᵗ, Tₛ, pᵃᵗ, Thermodynamics.Liquid())
        @test column(wet.interfaces.atmosphere_land_interface.specific_humidity) ≈ qᵛ⁺
    end
end

@testset "Surface-layer diagnostics at the reference height" begin
    for arch in test_architectures
        h = 10.0
        model = single_column_land_model(single_column_grid(arch);
                                         surface_layer_height = h, fluxes = similarity_fluxes())

        diagnostics = surface_layer_diagnostics(model; velocity_height = h,
                                                       temperature_height = h,
                                                       specific_humidity_height = h)

        @test value(diagnostics.u) ≈ 5
        @test abs(value(diagnostics.v)) < 1e-12
        @test value(diagnostics.T) ≈ 288
        @test value(diagnostics.q) ≈ 0.003
    end
end

@testset "Surface-layer diagnostics at the observation heights" begin
    for arch in test_architectures
        h = 10.0
        model = single_column_land_model(single_column_grid(arch);
                                         surface_layer_height = h, fluxes = similarity_fluxes())

        reference = surface_layer_diagnostics(model; velocity_height = h,
                                                     temperature_height = h,
                                                     specific_humidity_height = h)
        diagnostics = surface_layer_diagnostics(model)

        # The column is unstable over a dry surface, so the air cools and moistens upward.
        @test value(diagnostics.T) > value(reference.T)
        @test value(diagnostics.q) < value(reference.q)

        @test value(sqrt(diagnostics.u^2 + diagnostics.v^2)) ≈ value(diagnostics.u)
    end
end

@testset "Surface-layer diagnostics follow the neutral log law" begin
    for arch in test_architectures
        h   = 10.0
        uᵃᵗ = 5.0
        ℓ   = 1e-4
        fluxes = similarity_fluxes(; momentum_roughness_length    = ℓ,
                                     temperature_roughness_length = ℓ,
                                     water_vapor_roughness_length = ℓ,
                                     stability_functions = neutral_stability_functions())

        model = single_column_land_model(single_column_grid(arch); surface_layer_height = h, fluxes)

        for z in (2.0, 5.0, 10.0)
            @test value(surface_layer_diagnostics(model; velocity_height = z).u) ≈
                  uᵃᵗ * log(z / ℓ) / log(h / ℓ)
        end
    end
end

@testset "Surface-layer diagnostics with a zero-plane displacement" begin
    for arch in test_architectures
        h   = 10.0
        uᵃᵗ = 5.0
        ℓ   = 0.1
        d   = 4.0

        displaced_fluxes(zero_plane_displacement) =
            similarity_fluxes(; momentum_roughness_length    = ℓ,
                                temperature_roughness_length = ℓ,
                                water_vapor_roughness_length = ℓ,
                                zero_plane_displacement,
                                stability_functions = neutral_stability_functions())

        # The 2 m defaults sit below this canopy, so every height is named.
        diagnostics_at(model, z) = surface_layer_diagnostics(model; velocity_height = z,
                                                                    temperature_height = z,
                                                                    specific_humidity_height = z)

        grid = single_column_grid(arch)
        model = single_column_land_model(grid; surface_layer_height = h, fluxes = displaced_fluxes(d))

        # The similarity profile is displaced by `d`; the adiabatic term is not.
        for z in (5.0, 10.0)
            @test value(diagnostics_at(model, z).u) ≈ uᵃᵗ * log((z - d) / ℓ) / log((h - d) / ℓ)
        end

        @test value(diagnostics_at(model, h).T) ≈ 288

        @test_throws ArgumentError diagnostics_at(model, d)
        @test isnan(value(diagnostics_at(model, d + ℓ / 2).u))

        # A per-cell displacement gives the same profile.
        zero_plane_displacement = Field{Center, Center, Nothing}(grid)
        fill!(zero_plane_displacement, d)
        field_model = single_column_land_model(single_column_grid(arch); surface_layer_height = h,
                                               fluxes = displaced_fluxes(zero_plane_displacement))

        @test value(diagnostics_at(field_model, 5.0).u) ≈ uᵃᵗ * log((5 - d) / ℓ) / log((h - d) / ℓ)
    end
end

@testset "Surface-layer diagnostics with convective gustiness" begin
    for arch in test_architectures
        h = 10.0
        model = single_column_land_model(single_column_grid(arch); surface_layer_height = h)

        diagnostics = surface_layer_diagnostics(model; velocity_height = h,
                                                       temperature_height = h,
                                                       specific_humidity_height = h)

        # The gustiness enhancement is carried by u★ and is not undone here, while the
        # temperature and humidity scales are built from the resolved increments.
        @test value(diagnostics.u) > 5
        @test value(diagnostics.T) ≈ 288
        @test value(diagnostics.q) ≈ 0.003
    end
end

@testset "Surface-layer diagnostics in Float32" begin
    for arch in test_architectures
        h = 10f0
        model = single_column_land_model(single_column_grid(arch, Float32);
                                         surface_layer_height = h,
                                         fluxes = similarity_fluxes(Float32))

        diagnostics = surface_layer_diagnostics(model; velocity_height = h,
                                                       temperature_height = h,
                                                       specific_humidity_height = h)

        for (name, reference) in zip((:u, :v, :T, :q), (5f0, 0f0, 288f0, 0.003f0))
            field = compute!(Field(diagnostics[name]))
            @test eltype(field) == Float32
            @test all(isfinite, Array(interior(field)))
            @test isapprox(column(field), reference; atol = 1f-4, rtol = 1f-4)
        end
    end
end

@testset "Surface-layer diagnostics over the ocean" begin
    for arch in test_architectures
        h   = 10.0
        uᵃᵗ = 5.0
        uᵒᶜ = 0.5

        function ocean_column_model(fluxes)
            grid = single_column_grid(arch)
            ocean = ocean_simulation(grid; momentum_advection = nothing,
                                           tracer_advection = nothing,
                                           closure = nothing,
                                           bottom_drag_coefficient = 0)
            set!(ocean.model, u = uᵒᶜ, v = 0, T = 15, S = 30)

            atmosphere = PrescribedAtmosphere(grid; surface_layer_height = h, boundary_layer_height = 512)
            fill!(parent(atmosphere.temperature),       288)
            fill!(parent(atmosphere.specific_humidity), 0.003)
            fill!(parent(atmosphere.velocities.u), uᵃᵗ)
            fill!(parent(atmosphere.velocities.v), 0)
            fill!(parent(atmosphere.pressure),     101325)

            interfaces = ComponentInterfaces(atmosphere, ocean; atmosphere_ocean_fluxes = fluxes)
            return OceanOnlyModel(ocean; atmosphere, interfaces)
        end

        # The air-sea defaults resolve the roughness lengths from `u★`.
        model = ocean_column_model(SimilarityTheoryFluxes(Float64; subgrid_velocities = nothing))
        diagnostics = surface_layer_diagnostics(model, model.interfaces.atmosphere_ocean_interface;
                                                velocity_height = h,
                                                temperature_height = h,
                                                specific_humidity_height = h)

        # The surface current is added back, and the Celsius interface temperature converted.
        @test value(diagnostics.u) ≈ uᵃᵗ
        @test value(diagnostics.T) ≈ 288

        # A wind-dependent wave formulation needs the bulk velocity, which is not stored.
        wave_roughness = MomentumRoughnessLength(Float64; wave_formulation = WindDependentWaveFormulation(Float64))
        wave_model = ocean_column_model(SimilarityTheoryFluxes(Float64; momentum_roughness_length = wave_roughness))
        @test_throws ArgumentError surface_layer_diagnostics(wave_model,
                                                             wave_model.interfaces.atmosphere_ocean_interface)
    end
end

@testset "Surface-layer diagnostics reject unsupported interfaces" begin
    for arch in test_architectures
        model = single_column_land_model(single_column_grid(arch);
                                         fluxes = CoefficientBasedFluxes(Float64))

        @test_throws ArgumentError surface_layer_diagnostics(model)
        @test_throws ArgumentError surface_layer_diagnostics(model, nothing)
    end
end
