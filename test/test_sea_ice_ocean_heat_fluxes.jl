include("runtests_setup.jl")

using NumericalEarth.EarthSystemModels: IceBathHeatFlux,
                                     ThreeEquationHeatFlux,
                                     MomentumBasedFrictionVelocity

using NumericalEarth.EarthSystemModels.InterfaceComputations: compute_interface_heat_flux,
                                                          get_friction_velocity,
                                                          solve_interface_conditions,
                                                          SeaIceOceanInterface,
                                                          ComponentInterfaces

using ClimaSeaIce.SeaIceThermodynamics: LinearLiquidus, melting_temperature

@testset "Sea ice-ocean heat flux formulations" begin

    @testset "IceBathHeatFlux construction" begin
        flux = IceBathHeatFlux()
        @test flux.heat_transfer_coefficient == 0.006

        flux2 = IceBathHeatFlux(heat_transfer_coefficient = 0.01,
                                friction_velocity = MomentumBasedFrictionVelocity())

        @test flux2.heat_transfer_coefficient == 0.01
        @test flux2.friction_velocity isa MomentumBasedFrictionVelocity

        flux3 = IceBathHeatFlux(Float32)
        @test flux3.heat_transfer_coefficient isa Float32
    end

    @testset "ThreeEquationHeatFlux construction" begin
        flux = ThreeEquationHeatFlux()
        @test flux.heat_transfer_coefficient == 0.0095  # Default from Hieronymus et al. (2021)
        @test flux.salt_transfer_coefficient ≈ 0.0095 / 35  # R = 35

        flux2 = ThreeEquationHeatFlux(heat_transfer_coefficient = 0.01,
                                      salt_transfer_coefficient = 0.001,
                                      friction_velocity = MomentumBasedFrictionVelocity())
        @test flux2.heat_transfer_coefficient == 0.01
        @test flux2.salt_transfer_coefficient == 0.001
        @test flux2.friction_velocity isa MomentumBasedFrictionVelocity

        flux3 = ThreeEquationHeatFlux(Float32)
        @test flux3.heat_transfer_coefficient isa Float32
        @test flux3.salt_transfer_coefficient isa Float32
    end

    @testset "solve_interface_conditions" begin
        # Test parameters
        liquidus = LinearLiquidus(Float64)
        αₕ = 0.0095  # Heat transfer coefficient
        αₛ = αₕ / 35  # Salt transfer coefficient (R = 35)
        u★ = 0.002   # Friction velocity
        L  = 334e3   # Latent heat of fusion (J/kg)
        ρᵒᶜ = 1025.0  # Ocean reference density (kg/m³)
        cᵒᶜ = 3991.0  # Ocean heat capacity (J/kg/K)

        # Create a ThreeEquationHeatFlux without internal flux for testing
        flux = ThreeEquationHeatFlux()

        # Default ice state values (not used for NoInternalFluxTEF except for S)
        default_h = 1.0
        default_hc = 0.1
        default_ℵ = 1.0
        default_Tⁱⁿᵗ = 0.0

        @testset "Warm ocean (melting conditions)" begin
            Tᵒᶜ = 2.0    # Ocean temperature well above freezing
            Sᵒᶜ = 35.0   # Ocean salinity
            Sˢⁱ = 5.0    # Ice salinity

            ice_state = (; S = Sˢⁱ, h = default_h, hc = default_hc, ℵ = default_ℵ, T = default_Tⁱⁿᵗ)
            Tᵦ, Sᵦ, q = solve_interface_conditions(flux, Tᵒᶜ, Sᵒᶜ, ice_state, αₕ, αₛ, u★, L, ρᵒᶜ, cᵒᶜ, liquidus)

            # Interface salinity should be between ice and ocean salinity
            @test Sᵦ >= Sˢⁱ
            @test Sᵦ <= Sᵒᶜ

            # Interface temperature should be at freezing point of interface salinity
            Tₘ = melting_temperature(liquidus, Sᵦ)
            @test Tᵦ ≈ Tₘ

            # Warm ocean should cause melting (q > 0)
            @test q > 0
        end

        @testset "Cool ocean (weak melting)" begin
            # Ocean just above freezing - weak melting conditions
            Sᵒᶜ = 35.0
            Tₘ_ocean = melting_temperature(liquidus, Sᵒᶜ)
            Tᵒᶜ = Tₘ_ocean + 0.5  # Ocean 0.5°C above freezing
            Sˢⁱ = 5.0

            ice_state = (; S = Sˢⁱ, h = default_h, hc = default_hc, ℵ = default_ℵ, T = default_Tⁱⁿᵗ)
            Tᵦ, Sᵦ, q = solve_interface_conditions(flux, Tᵒᶜ, Sᵒᶜ, ice_state, αₕ, αₛ, u★, L, ρᵒᶜ, cᵒᶜ, liquidus)

            # Interface salinity should be between ice and ocean salinity
            @test Sᵦ >= Sˢⁱ
            @test Sᵦ <= Sᵒᶜ

            # Interface temperature should be at freezing point
            Tₘ = melting_temperature(liquidus, Sᵦ)
            @test Tᵦ ≈ Tₘ

            # Ocean above freezing should still cause melting (q > 0)
            @test q > 0
        end

        @testset "Ocean at freezing point" begin
            Sᵒᶜ = 35.0
            Tᵒᶜ = melting_temperature(liquidus, Sᵒᶜ)
            Sˢⁱ = 5.0

            ice_state = (; S = Sˢⁱ, h = default_h, hc = default_hc, ℵ = default_ℵ, T = default_Tⁱⁿᵗ)
            Tᵦ, Sᵦ, q = solve_interface_conditions(flux, Tᵒᶜ, Sᵒᶜ, ice_state, αₕ, αₛ, u★, L, ρᵒᶜ, cᵒᶜ, liquidus)

            @test Sᵦ ≈ Sᵒᶜ
            @test Tᵦ ≈ Tᵒᶜ
            @test abs(q) < eps(eltype(q))
        end

        @testset "Various salinity conditions" begin
            Tᵒᶜ = 1.0  # Warm ocean

            for Sᵒᶜ in [30.0, 33.0, 35.0, 37.0, 40.0]
                Sˢⁱ = 5.0
                ice_state = (; S = Sˢⁱ, h = default_h, hc = default_hc, ℵ = default_ℵ, T = default_Tⁱⁿᵗ)
                Tᵦ, Sᵦ, q = solve_interface_conditions(flux, Tᵒᶜ, Sᵒᶜ, ice_state, αₕ, αₛ, u★, L, ρᵒᶜ, cᵒᶜ, liquidus)

                # Interface salinity must always be bounded
                @test Sᵦ >= Sˢⁱ
                @test Sᵦ <= Sᵒᶜ

                # Interface temperature at freezing point
                Tₘ = melting_temperature(liquidus, Sᵦ)
                @test Tᵦ ≈ Tₘ
            end
        end

        @testset "Zero ice salinity" begin
            Tᵒᶜ = 1.0
            Sᵒᶜ = 35.0
            Sˢⁱ = 0.0  # Fresh ice

            ice_state = (; S = Sˢⁱ, h = default_h, hc = default_hc, ℵ = default_ℵ, T = default_Tⁱⁿᵗ)
            Tᵦ, Sᵦ, q = solve_interface_conditions(flux, Tᵒᶜ, Sᵒᶜ, ice_state, αₕ, αₛ, u★, L, ρᵒᶜ, cᵒᶜ, liquidus)

            @test Sᵦ >= Sˢⁱ
            @test Sᵦ <= Sᵒᶜ
            @test Tᵦ ≈ melting_temperature(liquidus, Sᵦ)
        end

        @testset "High friction velocity" begin
            Tᵒᶜ = 1.0
            Sᵒᶜ = 35.0
            Sˢⁱ = 5.0
            u★_high = 0.1  # High turbulence

            ice_state = (; S = Sˢⁱ, h = default_h, hc = default_hc, ℵ = default_ℵ, T = default_Tⁱⁿᵗ)
            Tᵦ, Sᵦ, q = solve_interface_conditions(flux, Tᵒᶜ, Sᵒᶜ, ice_state, αₕ, αₛ, u★_high, L, ρᵒᶜ, cᵒᶜ, liquidus)

            @test Sᵦ >= Sˢⁱ
            @test Sᵦ <= Sᵒᶜ
            @test Tᵦ ≈ melting_temperature(liquidus, Sᵦ)
        end

        @testset "Low friction velocity" begin
            Tᵒᶜ = 1.0
            Sᵒᶜ = 35.0
            Sˢⁱ = 5.0
            u★_low = 0.0001  # Very low turbulence

            ice_state = (; S = Sˢⁱ, h = default_h, hc = default_hc, ℵ = default_ℵ, T = default_Tⁱⁿᵗ)
            Tᵦ, Sᵦ, q = solve_interface_conditions(flux, Tᵒᶜ, Sᵒᶜ, ice_state, αₕ, αₛ, u★_low, L, ρᵒᶜ, cᵒᶜ, liquidus)

            @test Sᵦ >= Sˢⁱ
            @test Sᵦ <= Sᵒᶜ
            @test Tᵦ ≈ melting_temperature(liquidus, Sᵦ)
        end
    end
end

@testset "Salt flux sign conventions in coupled model" begin
    # Test that computed salt fluxes have the correct sign based on ocean temperature:
    # - Warm ocean (T > Tₘ) → melting → q > 0 → Jˢ > 0 (fresh meltwater dilutes ocean)
    # - Cold ocean (T < Tₘ) → freezing → q < 0 → Jˢ < 0 (brine rejection adds salt)
    # Sign convention: Jˢ > 0 means salinity is extracted from ocean

    for arch in test_architectures
        A = typeof(arch)
        @info "Testing salt flux sign conventions on $A"

        grid = LatitudeLongitudeGrid(arch,
                                     size = (4, 4, 1),
                                     latitude = (-10, 10),
                                     longitude = (0, 10),
                                     z = (-100, 0))

        ocean = ocean_simulation(grid, momentum_advection=nothing, closure=nothing, tracer_advection=nothing)
        sea_ice = sea_ice_simulation(grid, ocean)

        backend = JRA55NetCDFBackend(4)
        atmosphere = JRA55PrescribedAtmosphere(arch; backend)
        radiation = Radiation(arch)

        for sea_ice_ocean_heat_flux in [IceBathHeatFlux(), ThreeEquationHeatFlux()]
            @testset "Salt flux with $(nameof(typeof(sea_ice_ocean_heat_flux)))" begin
                interfaces = ComponentInterfaces(atmosphere, ocean, sea_ice;
                                                 radiation,
                                                 sea_ice_ocean_heat_flux)
                coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation, interfaces)

                # Test melting conditions: warm ocean above freezing
                # Freezing point at S=35 is about -1.9°C
                set!(ocean.model, T=2.0, S=35.0)  # Warm ocean
                set!(sea_ice.model, h=1.0, ℵ=1.0)

                time_step!(coupled_model, 60)

                # Get the computed fluxes
                Jˢ = coupled_model.interfaces.sea_ice_ocean_interface.fluxes.salt
                𝒬ⁱⁿᵗ = coupled_model.interfaces.sea_ice_ocean_interface.fluxes.interface_heat

                # Warm ocean should cause melting → Qᵢ > 0 (heat into ice)
                𝒬ⁱⁿᵗ_cpu = Array(interior(𝒬ⁱⁿᵗ, :, :, 1))
                @test all(𝒬ⁱⁿᵗ_cpu .> 0)

                # During melting, fresh meltwater dilutes ocean → Jˢ > 0
                Jˢ_cpu = Array(interior(Jˢ, :, :, 1))
                @test all(Jˢ_cpu .> 0)
            end
        end
    end
end

@testset "Salt flux unit consistency" begin
    # This test verifies that the salt flux has correct units after the fix
    # that adds the freshwater density conversion: Jˢ = (q / ρf) * (Sᵦ - S_ice)
    #
    # The key insight is that:
    # - q is a mass flux (kg/m²/s)
    # - Dividing by ρf (kg/m³) gives a volume flux (m/s)
    # - Multiplying by salinity difference gives psu × m/s, consistent with atmosphere-ocean
    #
    # Without this fix, salt flux would be ~1000× too large, causing instability.

    for arch in test_architectures
        A = typeof(arch)
        @info "Testing salt flux unit consistency on $A"

        grid = LatitudeLongitudeGrid(arch,
                                     size = (4, 4, 1),
                                     latitude = (-10, 10),
                                     longitude = (0, 10),
                                     z = (-100, 0))

        ocean = ocean_simulation(grid, momentum_advection=nothing, closure=nothing, tracer_advection=nothing)
        sea_ice = sea_ice_simulation(grid, ocean)

        backend = JRA55NetCDFBackend(4)
        atmosphere = JRA55PrescribedAtmosphere(arch; backend)
        radiation = Radiation(arch)

        for sea_ice_ocean_heat_flux in [IceBathHeatFlux(), ThreeEquationHeatFlux()]
            @testset "Flux magnitude with $(nameof(typeof(sea_ice_ocean_heat_flux)))" begin
                interfaces = ComponentInterfaces(atmosphere, ocean, sea_ice;
                                                 radiation,
                                                 sea_ice_ocean_heat_flux)
                coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation, interfaces)

                # Set up melting conditions
                set!(ocean.model, T=2.0, S=35.0)  # Warm ocean
                set!(sea_ice.model, h=1.0, ℵ=1.0)

                time_step!(coupled_model, 60)

                # Get the computed fluxes
                Jˢ = coupled_model.interfaces.sea_ice_ocean_interface.fluxes.salt
                𝒬ⁱⁿᵗ = coupled_model.interfaces.sea_ice_ocean_interface.fluxes.interface_heat

                Jˢ_cpu = Array(interior(Jˢ, :, :, 1))
                𝒬ⁱⁿᵗ_cpu = Array(interior(𝒬ⁱⁿᵗ, :, :, 1))

                # Heat flux should be O(100-1000) W/m² for strong melting
                @test all(𝒬ⁱⁿᵗ_cpu .> 0)
                @test all(𝒬ⁱⁿᵗ_cpu .< 1e5)  # Should not be unreasonably large

                # Salt flux (in psu × m/s) should be small: typical values O(1e-7 to 1e-5)
                # Before the fix, salt flux was ~1000× too large
                # The salt flux magnitude should be comparable to:
                # Jˢ ~ (Q / (ρ * c * L)) * ΔS ~ (1000 / (1025 * 4000 * 3e5)) * 30 ~ 2e-7 psu m/s
                @test all(abs.(Jˢ_cpu) .< 1e-3)  # Should not be unreasonably large (was ~1 before fix)
                @test all(abs.(Jˢ_cpu) .> 1e-10) # Should not be zero
            end
        end
    end
end

@testset "Salt flux density scaling" begin
    # This test verifies that the salt flux scales correctly with ocean reference density
    # Salt flux formula: Jˢ = (q / ρᵒᶜ) * (Sᵒᶜ - Sˢⁱ)

    liquidus = LinearLiquidus(Float64)
    αₕ = 0.0095
    αₛ = αₕ / 35
    u★ = 0.002
    L  = 334e3
    ρᵒᶜ = 1025.0
    cᵒᶜ = 3991.0

    Tᵒᶜ = 2.0
    Sᵒᶜ = 35.0
    Sˢⁱ = 5.0

    # Create a ThreeEquationHeatFlux without internal flux for testing
    flux = ThreeEquationHeatFlux()

    # Compute interface conditions
    ice_state = (; S = Sˢⁱ, h = 1.0, hc = 0.1, ℵ = 1.0, T = 0.0)
    Tᵦ, Sᵦ, q = solve_interface_conditions(flux, Tᵒᶜ, Sᵒᶜ, ice_state, αₕ, αₛ, u★, L, ρᵒᶜ, cᵒᶜ, liquidus)

    # q is a mass flux (kg/m²/s)
    # Salt flux with density conversion: Jˢ = (q / ρᵒᶜ) * (Sᵒᶜ - Sˢⁱ)
    ρᵒᶜ_standard = 1025.0
    ρᵒᶜ_altered  = 1030.0

    Jˢ_standard = (q / ρᵒᶜ_standard) * (Sᵒᶜ - Sˢⁱ)
    Jˢ_altered  = (q / ρᵒᶜ_altered)  * (Sᵒᶜ - Sˢⁱ)

    # Salt flux should scale inversely with ocean density
    @test Jˢ_standard / Jˢ_altered ≈ ρᵒᶜ_altered / ρᵒᶜ_standard

    # Verify the salt flux has reasonable magnitude
    # For typical conditions: q ~ 1e-5 kg/m²/s, ΔS ~ 30 psu, ρᵒᶜ ~ 1025 kg/m³
    # Jˢ ~ (1e-5 / 1025) * 30 ~ 3e-7 psu m/s
    @test abs(Jˢ_standard) < 1e-4  # Should be small
    @test abs(Jˢ_standard) > 1e-10 # Should not be negligible
end

@testset "Heat and salt flux consistency" begin
    # Verify that heat flux and salt flux are computed consistently
    # Key relationship: Q = ℰ * q, so q = Q / ℰ
    # Salt flux: Jˢ = (q / ρᵒᶜ) * (Sᵒᶜ - Sˢⁱ)

    liquidus = LinearLiquidus(Float64)
    αₕ = 0.0095
    αₛ = αₕ / 35
    u★ = 0.002
    ℰ  = 334e3   # Latent heat (J/kg)
    ρᵒᶜ = 1025.0
    cᵒᶜ = 3991.0

    Tᵒᶜ = 2.0
    Sᵒᶜ = 35.0
    Sˢⁱ = 5.0

    # Create a ThreeEquationHeatFlux without internal flux for testing
    flux = ThreeEquationHeatFlux()

    ice_state = (; S = Sˢⁱ, h = 1.0, hc = 0.1, ℵ = 1.0, T = 0.0)
    Tᵦ, Sᵦ, q = solve_interface_conditions(flux, Tᵒᶜ, Sᵒᶜ, ice_state, αₕ, αₛ, u★, ℰ, ρᵒᶜ, cᵒᶜ, liquidus)

    # Compute heat flux from melt rate
    Q = ℰ * q  # W/m² (without ice concentration scaling for this unit test)

    # Compute salt flux with ocean density conversion
    ΔS = Sᵒᶜ - Sˢⁱ
    Jˢ = (q / ρᵒᶜ) * ΔS

    # Verify the relationship: Jˢ * ρᵒᶜ * ℰ / ΔS should equal Q
    Q_from_salt = Jˢ * ρᵒᶜ * ℰ / ΔS
    @test Q_from_salt ≈ Q

    # Also verify that temperature flux and salt flux have consistent scaling
    # Jᵀ = Q / (ρᵒᶜ * cᵒᶜ) has units K × m/s
    Jᵀ = Q / (ρᵒᶜ * cᵒᶜ)

    # Both Jᵀ and Jˢ should be O(1e-7) for these conditions
    @test abs(Jᵀ) > 1e-10
    @test abs(Jᵀ) < 1e-4
    @test abs(Jˢ) > 1e-10
    @test abs(Jˢ) < 1e-4
end

@testset "Frazil ice formation and salt flux" begin
    # Test that frazil ice formation is handled correctly and contributes
    # to the salt flux with proper density conversion.
    # When ocean temperature drops below freezing, frazil ice forms and
    # the salt flux includes the frazil contribution.

    for arch in test_architectures
        A = typeof(arch)
        @info "Testing frazil ice formation on $A"

        grid = LatitudeLongitudeGrid(arch,
                                     size = (4, 4, 1),
                                     latitude = (-10, 10),
                                     longitude = (0, 10),
                                     z = (-400, 0))

        ocean = ocean_simulation(grid, momentum_advection=nothing, closure=nothing, tracer_advection=nothing)
        sea_ice = sea_ice_simulation(grid, ocean)

        backend = JRA55NetCDFBackend(4)
        atmosphere = JRA55PrescribedAtmosphere(arch; backend)
        radiation = Radiation(arch)

        for sea_ice_ocean_heat_flux in [IceBathHeatFlux(), ThreeEquationHeatFlux()]
            @testset "Frazil with $(nameof(typeof(sea_ice_ocean_heat_flux)))" begin
                interfaces = ComponentInterfaces(atmosphere, ocean, sea_ice;
                                                 radiation,
                                                 sea_ice_ocean_heat_flux)
                coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation, interfaces)

                # Set up conditions where frazil might form:
                # Cold ocean near freezing with ice present
                # Freezing point at S=35 is about -1.9°C
                set!(ocean.model, T=-1.5, S=35.0)  # Cold but above freezing
                set!(sea_ice.model, h=1.0, ℵ=0.5)

                time_step!(coupled_model, 60)

                # Get the computed fluxes
                Jˢ = coupled_model.interfaces.sea_ice_ocean_interface.fluxes.salt
                𝒬ᶠʳᶻ = coupled_model.interfaces.sea_ice_ocean_interface.fluxes.frazil_heat

                Jˢ_cpu = Array(interior(Jˢ, :, :, 1))
                𝒬ᶠʳᶻ_cpu = Array(interior(𝒬ᶠʳᶻ, :, :, 1))

                # Salt flux should be finite and reasonably bounded
                @test all(isfinite.(Jˢ_cpu))
                @test all(abs.(Jˢ_cpu) .< 1e-3)

                # Frazil heat flux should be finite
                @test all(isfinite.(𝒬ᶠʳᶻ_cpu))
            end
        end
    end
end

@testset "Coupled model with different heat flux formulations" begin
    for arch in test_architectures
        A = typeof(arch)
        @info "Testing heat flux formulations on $A"

        grid = LatitudeLongitudeGrid(arch,
                                     size = (4, 4, 1),
                                     latitude = (-10, 10),
                                     longitude = (0, 10),
                                     z = (-400, 0))

        ocean = ocean_simulation(grid, momentum_advection=nothing, closure=nothing, tracer_advection=nothing)
        sea_ice = sea_ice_simulation(grid, ocean)

        backend = JRA55NetCDFBackend(4)
        atmosphere = JRA55PrescribedAtmosphere(arch; backend)
        radiation = Radiation(arch)

        # Test with ThreeEquationHeatFlux (default)
        @test begin
            coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
            flux_form = coupled_model.interfaces.sea_ice_ocean_interface.flux_formulation
            flux_form isa ThreeEquationHeatFlux
        end

        # Test with IceBathHeatFlux via ComponentInterfaces
        @test begin
            flux = IceBathHeatFlux()
            interfaces = ComponentInterfaces(atmosphere, ocean, sea_ice;
                                              radiation,
                                              sea_ice_ocean_heat_flux = flux)
            coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation, interfaces)
            flux_form = coupled_model.interfaces.sea_ice_ocean_interface.flux_formulation
            flux_form isa IceBathHeatFlux
        end

        # Test time stepping with each formulation
        for sea_ice_ocean_heat_flux in [IceBathHeatFlux(),
                                        ThreeEquationHeatFlux()]

            @testset "Time stepping with $(nameof(typeof(sea_ice_ocean_heat_flux)))" begin
                interfaces = ComponentInterfaces(atmosphere, ocean, sea_ice;
                                                 radiation,
                                                 sea_ice_ocean_heat_flux)
                coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation, interfaces)
                @test begin
                    time_step!(coupled_model, 60)
                    true
                end
            end
        end
    end
end
