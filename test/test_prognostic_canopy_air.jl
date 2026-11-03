include("runtests_setup.jl")

using Oceananigans
using Oceananigans: prognostic_state, restore_prognostic_state!
using Oceananigans.TimeSteppers: time_step!, update_state!, reset!
using NumericalEarth.EarthSystemModels: step_interface!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyInterface, CanopyConductanceHumidity, CriticalSaturation, AbsorbedPhotosyntheticallyActiveRadiation,
    AtmosphericThermodynamics
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters

canopy_model(arch; canopy_air_depth = nothing, kw...) = coupled_land_model(arch; kw...,
    atmosphere_land_interface_state = CanopyInterface(Float64; canopy_air_depth, soil_humidity = bare_soil_humidity(Float64),
        canopy_humidity = CanopyConductanceHumidity(Float64; leaf_area_index = 4.0, moisture_stress = CriticalSaturation(0.5),
                                                    absorbed_par = AbsorbedPhotosyntheticallyActiveRadiation(Float64))))

@testset "Prognostic canopy air space" begin
    for arch in test_architectures
        mp = canopy_model(arch; canopy_air_depth = 10)
        Ts = mp.interfaces.atmosphere_land_interface.state
        T₀, q₀ = scalar(Ts.temperature), scalar(Ts.specific_humidity)
        @test 285 < T₀ < 320
        @test 0 < q₀ < 0.05

        # The default diagnostic node carries no state, and at windy daytime it matches the prognostic node.
        md = canopy_model(arch)
        @test isnothing(md.interfaces.atmosphere_land_interface.formulation.state.canopy_air_depth)
        for _ in 1:24
            time_step!(md, 300.0)
            time_step!(mp, 300.0)
        end
        fd = md.interfaces.atmosphere_land_interface.fluxes
        fp = mp.interfaces.atmosphere_land_interface.fluxes
        @test scalar(fp.latent_heat) ≈ scalar(fd.latent_heat) atol = 2
        @test scalar(fp.sensible_heat) ≈ scalar(fd.sensible_heat) atol = 2

        # Over an interface step, flux to the atmosphere = supply − storage tendency,
        # with Cᵀ = ρ cᵖᵐ hᵃᶜᵃⁿ and Cᵛ = ρ hᵃᶜᵃⁿ (hᵃᶜᵃⁿ = 10 m).
        T⁻, q⁻ = scalar(Ts.temperature), scalar(Ts.specific_humidity)
        Δt = 300.0
        step_interface!(mp.interfaces, mp, Δt)
        ℂ = AtmosphereThermodynamicsParameters(Float64)
        ρhᵃᶜᵃⁿ = AtmosphericThermodynamics.air_density(ℂ, 300.0, 101325.0, 0.008) * 10.0
        Sᵀ = ρhᵃᶜᵃⁿ * AtmosphericThermodynamics.cp_m(ℂ, 0.008) * (scalar(Ts.temperature) - T⁻) / Δt
        Sᵛ = AtmosphericThermodynamics.latent_heat_vapor(ℂ, 300.0) * ρhᵃᶜᵃⁿ * (scalar(Ts.specific_humidity) - q⁻) / Δt
        𝒬ᵀˡᵉᵃᶠ, 𝒬ᵛˡᵉᵃᶠ = scalar(Ts.canopy_sensible_heat), scalar(Ts.canopy_latent_heat)
        @test 𝒬ᵀˡᵉᵃᶠ + scalar(Ts.ground_sensible_heat) - scalar(fp.sensible_heat) ≈ Sᵀ atol = 1e-6
        @test 𝒬ᵛˡᵉᵃᶠ + scalar(Ts.ground_latent_heat) - scalar(fp.latent_heat) ≈ Sᵛ atol = 1e-6

        # Leaf net radiation balances the leaf fluxes at the stored leaf temperature.
        Tˡ, Tᵍʳⁿᵈ = scalar(Ts.canopy), scalar(Ts.ground)
        σ, λˡᵉᵃᶠ, ϵᵍʳⁿᵈ = 5.670374419e-8, 4.0, 0.96
        ϵˡ = 0.98 * (1 - exp(-λˡᵉᵃᶠ))
        ℐꜛˡʷᵍʳⁿᵈ = ϵᵍʳⁿᵈ * σ * Tᵍʳⁿᵈ^4 + (1 - ϵᵍʳⁿᵈ) * ((1 - ϵˡ) * 350.0 + ϵˡ * σ * Tˡ^4)
        Rˡ = (1 - 0.15) * (1 - exp(-0.5 * λˡᵉᵃᶠ)) * 600.0 + ϵˡ * (350.0 + ℐꜛˡʷᵍʳⁿᵈ) - 2 * ϵˡ * σ * Tˡ^4
        @test Rˡ ≈ 𝒬ᵀˡᵉᵃᶠ + 𝒬ᵛˡᵉᵃᶠ atol = 1e-3

        @test min(Tˡ, Tᵍʳⁿᵈ, 300) - 1 ≤ scalar(Ts.temperature) ≤ max(Tˡ, Tᵍʳⁿᵈ, 300) + 1   # 300 K is the air temperature
    end

    # Hot wet soil under cooler dry air at near-calm dusk.
    for arch in test_architectures, layer_depth in (0.1, 10.0), Δt in (300.0, 3600.0)
        mp = canopy_model(arch; canopy_air_depth = layer_depth, shortwave = 50.0, wind = 0.2,
                          air_temperature = 304.0, land_temperature = 310.0, water = 135.0)
        time_step!(mp, 1.0)
        state = mp.interfaces.atmosphere_land_interface.state
        set!(state.temperature, 300)
        set!(state.specific_humidity, 0.008)
        interface = mp.interfaces.atmosphere_land_interface
        Ts, fluxes = interface.state, interface.fluxes
        ℂ = AtmosphereThermodynamicsParameters(Float64)
        ρh = AtmosphericThermodynamics.air_density(ℂ, 304.0, 101325.0, 0.008) * layer_depth
        Cᵀ = ρh * AtmosphericThermodynamics.cp_m(ℂ, 0.008)
        ℒCᵛ = ρh * AtmosphericThermodynamics.latent_heat_vapor(ℂ, 304.0)
        for _ in 1:12
            T⁻, q⁻ = scalar(state.temperature), scalar(state.specific_humidity)
            step_interface!(mp.interfaces, mp, Δt)
            T, q = scalar(state.temperature), scalar(state.specific_humidity)
            @test 280 < T < 330
            @test 0 < q < 0.06
            @test Cᵀ * (T - T⁻) / Δt ≈ scalar(Ts.canopy_sensible_heat) + scalar(Ts.ground_sensible_heat) - scalar(fluxes.sensible_heat) atol=1e-6
            @test ℒCᵛ * (q - q⁻) / Δt ≈ scalar(Ts.canopy_latent_heat) + scalar(Ts.ground_latent_heat) - scalar(fluxes.latent_heat) atol=1e-6
        end
    end
end

@testset "Field-valued canopy-air depth" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch; size = (2, 1, 1), latitude = (10, 11), longitude = (10, 12),
                                     z = (-1, 0), topology = (Bounded, Bounded, Bounded))
        depth = Field{Center, Center, Nothing}(grid)
        set!(depth, reshape([1.0, 100.0], 2, 1, 1))
        models = [canopy_model(arch; grid, canopy_air_depth) for canopy_air_depth in (depth, 1.0, 100.0)]
        for model in models
            fill!(parent(model.atmosphere.temperature), 305)
            time_step!(model, 30.0)
        end
        for name in (:temperature, :specific_humidity)
            values = map(model -> Array(interior(getproperty(model.interfaces.atmosphere_land_interface.state, name))), models)
            @test values[1][1, 1, 1] ≈ values[2][1, 1, 1]
            @test values[1][2, 1, 1] ≈ values[3][2, 1, 1]
            @test values[1][1, 1, 1] != values[1][2, 1, 1]
        end
    end
end

@testset "Single-precision prognostic canopy" begin
    for arch in test_architectures
        canopy = CanopyInterface(Float32; soil_humidity=bare_soil_humidity(Float32), canopy_air_depth=10)
        model = coupled_land_model(arch, Float32; atmosphere_land_interface_state=canopy)
        time_step!(model, 300.0)
        @test 280 < scalar(model.interfaces.atmosphere_land_interface.state.temperature) < 330
        @test 0 < scalar(model.interfaces.atmosphere_land_interface.state.specific_humidity) < 0.05
    end
end

@testset "Setting the canopy-air state" begin
    for arch in test_architectures
        model = canopy_model(arch; canopy_air_depth = 10)
        interface = model.interfaces.atmosphere_land_interface
        Ts = interface.state
        T₀ = scalar(Ts.temperature)

        # The set node survives update_state! at iteration 0, and the fluxes follow it.
        sensible_heat = map((290, 310)) do T
            set!(model.interfaces; atmosphere_land_interface = (; T, q = 0.012))
            update_state!(model)
            @test scalar(Ts.temperature) == T
            @test scalar(Ts.specific_humidity) == 0.012
            scalar(interface.fluxes.sensible_heat)
        end
        @test sensible_heat[1] < sensible_heat[2]

        # Every prognostic variable, and only those, must be given.
        for values in ((; T = 295), (; T = 295, q = 0.012, canopy = 295))
            @test_throws ArgumentError set!(model.interfaces; atmosphere_land_interface = values)
        end
        diagnostic = canopy_model(arch)
        @test_throws ArgumentError set!(diagnostic.interfaces; atmosphere_land_interface = (; T = 295, q = 0.012))

        # A checkpoint taken at iteration 0 restores the set state.
        restored = canopy_model(arch; canopy_air_depth = 10)
        restore_prognostic_state!(restored, deepcopy(prognostic_state(model)))
        @test scalar(restored.interfaces.atmosphere_land_interface.state.temperature) == 310

        # The first step starts from the set node.
        other = canopy_model(arch; canopy_air_depth = 10)
        set!(other.interfaces; atmosphere_land_interface = (; T = 290, q = 0.012))
        time_step!(model, 60.0)
        time_step!(other, 60.0)
        @test scalar(Ts.temperature) > scalar(other.interfaces.atmosphere_land_interface.state.temperature)

        # A reset model starts again from the massless node.
        reset!(other)
        update_state!(other)
        @test scalar(other.interfaces.atmosphere_land_interface.state.temperature) != 290
    end
end
