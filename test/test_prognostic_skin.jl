include("runtests_setup.jl")

using Oceananigans
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    SoilSkinTemperature, EnergyBalanceTemperature, SoilConductiveFlux
using NumericalEarth.Radiations: default_stefan_boltzmann_constant

bare_soil_model(arch, temperature; wind = 5.0, water = 90.0, kw...) =
    coupled_land_model(arch; wind, water, kw..., atmosphere_land_interface_state = (; temperature, specific_humidity = bare_soil_humidity(Float64)))

skin_temperature(model) = model.interfaces.atmosphere_land_interface.temperature

# Calm, moist transition (the issue-549 bare-soil exemplar).
calm = (; shortwave = 50.0, wind = 0.2, air_temperature = 304.0, land_temperature = 310.0, water = 135.0)

@testset "Prognostic energy-balance skin" begin
    for arch in test_architectures
        windy = [bare_soil_model(arch, SoilSkinTemperature(1.5, 0.05); surface_storage=storage)
                 for storage in (PrognosticSkin(heat_capacity = 1e5), nothing)]
        for _ in 1:36, model in windy
            time_step!(model, 300.0)
        end
        prognostic, diagnostic = (model.interfaces.atmosphere_land_interface for model in windy)
        @test scalar(prognostic.temperature) ≈ scalar(diagnostic.temperature) atol = 0.5
        @test scalar(prognostic.fluxes.latent_heat) ≈ scalar(diagnostic.fluxes.latent_heat) atol = 10

        # The storage tendency closes the surface energy balance up to linearization error.
        C, Δt, Λ, σ = 1e5, 300.0, 30.0, default_stefan_boltzmann_constant
        model = bare_soil_model(arch, SoilSkinTemperature(1.5, 0.05); surface_storage=PrognosticSkin(heat_capacity=C), calm...)
        interface = model.interfaces.atmosphere_land_interface
        Tⁱⁿ⁻, worst = scalar(interface.temperature), 0.0
        for _ in 1:48
            Tˡᵃ = scalar(model.land.temperature)
            time_step!(model, Δt)
            Tⁱⁿ   = scalar(interface.temperature)
            𝒬ᵍʳⁿᵈ = Λ * (Tⁱⁿ - Tˡᵃ)
            𝒬ᵛ   = scalar(interface.fluxes.latent_heat)
            @test scalar(model.land.fluxes.surface_energy_flux) ≈ -𝒬ᵍʳⁿᵈ atol = 1e-9
            @test abs(𝒬ᵛ) < 2000
            net_radiation = (1 - 0.2) * 50 + 0.95 * (350 - σ * Tⁱⁿ^4)
            residual = net_radiation - 𝒬ᵍʳⁿᵈ - scalar(interface.fluxes.sensible_heat) - 𝒬ᵛ - C * (Tⁱⁿ - Tⁱⁿ⁻) / Δt
            worst = max(worst, abs(residual))
            Tⁱⁿ⁻ = Tⁱⁿ
        end
        @test worst < 15
    end
end

@testset "Explicit prognostic skin initialization" begin
    for FT in (Float32, Float64)
        @test PrognosticSkin(FT).heat_capacity isa FT
    end

    # The first update_state! lands near, but not exactly on, the massless energy-balance root.
    for arch in test_architectures
        Tp = scalar(skin_temperature(bare_soil_model(arch, SoilSkinTemperature(1.5, 0.05); surface_storage=PrognosticSkin())))
        Td = scalar(skin_temperature(bare_soil_model(arch, SoilSkinTemperature(1.5, 0.05))))
        @test Tp ≈ Td atol = 1
        @test abs(Tp - Td) < abs(298 - Td)
    end
end

@testset "Field-valued skin heat capacity" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch; size = (2, 1, 1), longitude = (10, 12), latitude = (10, 11),
                                     z = (-1, 0), topology = (Bounded, Bounded, Bounded))
        Cfield = Field{Center, Center, Nothing}(grid)
        set!(Cfield, reshape([1e4, 1e6], 2, 1, 1))
        models = [bare_soil_model(arch, SoilSkinTemperature(1.5, 0.05); surface_storage=PrognosticSkin(heat_capacity=C), grid, calm...) for C in (Cfield, 1e4, 1e6)]
        for _ in 1:12, model in models
            time_step!(model, 300.0)
        end
        Tm, Tl, Th = (Array(interior(skin_temperature(model)))[:, 1, 1] for model in models)
        @test all(isfinite, Tm)
        @test Tm[1] ≈ Tl[1] atol = 1e-10
        @test Tm[2] ≈ Th[2] atol = 1e-10
        @test !isapprox(Tl[1], Th[1]; atol = 1e-3)
    end
end

# Λᵍʳⁿᵈ = 0, ϵ = 0 and calm air leave no conductance, so the skin must be held rather than divided by zero.
@testset "Fully decoupled skin stays finite" begin
    for arch in test_architectures, storage in (nothing, PrognosticSkin(heat_capacity = 1e5))
        model = bare_soil_model(arch, SoilSkinTemperature(0, 0.05); surface_storage=storage,
                                shortwave = 0.0, longwave = 0.0, wind = 0.0, ϵ = 0)
        interface = model.interfaces.atmosphere_land_interface
        @test isfinite(scalar(interface.temperature))
        for _ in 1:3
            time_step!(model, 300.0)
        end
        @test all(isfinite, scalar.((interface.temperature, interface.fluxes.latent_heat, interface.fluxes.sensible_heat)))
    end
end
