include("runtests_setup.jl")

using Oceananigans
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyInterface, CanopyConductanceHumidity, CriticalSaturation, InteractiveAbsorbedPAR,
    AtmosphericThermodynamics
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters

canopy_model(arch; kw...) = coupled_land_model(arch; kw...,
    atmosphere_land_interface_state = CanopyInterface(Float64; soil_humidity = bare_soil_humidity(Float64),
        canopy_humidity = CanopyConductanceHumidity(Float64; leaf_area_index = 4.0, moisture_stress = CriticalSaturation(0.5),
                                                    absorbed_par = InteractiveAbsorbedPAR(Float64))))

@testset "Prognostic canopy air space" begin
    for arch in test_architectures
        mp = canopy_model(arch; surface_storage=PrognosticCanopyAir())
        Ts = mp.interfaces.atmosphere_land_interface.temperature
        prognostic = mp.land.surface_storage.prognostic
        T₀, q₀ = scalar(prognostic.temperature), scalar(prognostic.specific_humidity)
        @test 285 < T₀ < 320
        @test 0 < q₀ < 0.05
        @test scalar(Ts.interface) == T₀

        # The default diagnostic node carries no state, and at windy daytime it matches the prognostic node.
        md = canopy_model(arch)
        @test isnothing(md.land.surface_storage)
        for _ in 1:24
            time_step!(md, 300.0)
            time_step!(mp, 300.0)
        end
        fd = md.interfaces.atmosphere_land_interface.fluxes
        fp = mp.interfaces.atmosphere_land_interface.fluxes
        @test scalar(fp.latent_heat) ≈ scalar(fd.latent_heat) atol = 2
        @test scalar(fp.sensible_heat) ≈ scalar(fd.sensible_heat) atol = 2

        # Flux to the atmosphere = supply − storage tendency, with Cᵀ = ρ cᵖᵐ hᵃᶜᵃⁿ and Cᵛ = ρ hᵃᶜᵃⁿ (hᵃᶜᵃⁿ = 10 m).
        T⁻, q⁻ = scalar(prognostic.temperature), scalar(prognostic.specific_humidity)
        Δt = 300.0
        time_step!(mp.land, Δt)
        heat_supply = scalar(Ts.canopy_sensible_heat) + scalar(Ts.ground_sensible_heat) - scalar(fp.sensible_heat)
        vapor_supply = scalar(Ts.canopy_latent_heat) + scalar(Ts.ground_latent_heat) - scalar(fp.latent_heat)
        ℂ = AtmosphereThermodynamicsParameters(Float64)
        ρhᵃᶜᵃⁿ = AtmosphericThermodynamics.air_density(ℂ, 300.0, 101325.0, 0.008) * 10.0
        Sᵀ = ρhᵃᶜᵃⁿ * AtmosphericThermodynamics.cp_m(ℂ, 0.008) * (scalar(prognostic.temperature) - T⁻) / Δt
        Sᵛ = AtmosphericThermodynamics.latent_heat_vapor(ℂ, 300.0) * ρhᵃᶜᵃⁿ * (scalar(prognostic.specific_humidity) - q⁻) / Δt
        𝒬ᵀˡᵉᵃ, 𝒬ᵛˡᵉᵃ = scalar(Ts.canopy_sensible_heat), scalar(Ts.canopy_latent_heat)
        @test heat_supply ≈ Sᵀ atol = 1e-6
        @test vapor_supply ≈ Sᵛ atol = 1e-6

        # Leaf net radiation balances the leaf fluxes at the stored leaf temperature.
        Tˡ, Tᵍʳᵒ = scalar(Ts.canopy), scalar(Ts.ground)
        σ, λˡᵉᵃᶠ, ϵᵍʳᵒ = 5.670374419e-8, 4.0, 0.96
        ϵˡ = 0.98 * (1 - exp(-λˡᵉᵃᶠ))
        ℐꜛˡʷᵍʳᵒ = ϵᵍʳᵒ * σ * Tᵍʳᵒ^4 + (1 - ϵᵍʳᵒ) * ((1 - ϵˡ) * 350.0 + ϵˡ * σ * Tˡ^4)
        Rˡ = (1 - 0.15) * (1 - exp(-0.5 * λˡᵉᵃᶠ)) * 600.0 + ϵˡ * (350.0 + ℐꜛˡʷᵍʳᵒ) - 2 * ϵˡ * σ * Tˡ^4
        @test Rˡ ≈ 𝒬ᵀˡᵉᵃ + 𝒬ᵛˡᵉᵃ atol = 1e-3

        @test min(Tˡ, Tᵍʳᵒ, 300) - 1 ≤ scalar(Ts.interface) ≤ max(Tˡ, Tᵍʳᵒ, 300) + 1   # 300 K is the air temperature
    end

    # Hot wet soil under cooler dry air at near-calm dusk, where the diagnostic node has no steady state.
    for arch in test_architectures
        mp = canopy_model(arch; surface_storage=PrognosticCanopyAir(), shortwave = 50.0, wind = 0.2,
                          air_temperature = 304.0, land_temperature = 310.0, water = 135.0)
        fp = mp.interfaces.atmosphere_land_interface.fluxes
        for _ in 1:48
            time_step!(mp, 300.0)
            @test abs(scalar(fp.latent_heat)) < 2000
        end
        @test isfinite(scalar(mp.land.surface_storage.prognostic.temperature))
        @test 0 < scalar(mp.land.surface_storage.prognostic.specific_humidity) < 0.06
    end
end

@testset "Field-valued canopy-air depth" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch; size = (2, 1, 1), latitude = (10, 11), longitude = (10, 12),
                                     z = (-1, 0), topology = (Bounded, Bounded, Bounded))
        depth = Field{Center, Center, Nothing}(grid)
        set!(depth, reshape([1.0, 100.0], 2, 1, 1))
        models = [canopy_model(arch; grid, surface_storage=PrognosticCanopyAir(; layer_depth))
                  for layer_depth in (depth, 1.0, 100.0)]
        for model in models
            fill!(parent(model.atmosphere.temperature), 305)
            time_step!(model, 30.0)
        end
        for name in (:temperature, :specific_humidity)
            values = map(model -> Array(interior(model.land.surface_storage.prognostic[name])), models)
            @test values[1][1, 1, 1] ≈ values[2][1, 1, 1]
            @test values[1][2, 1, 1] ≈ values[3][2, 1, 1]
            @test values[1][1, 1, 1] != values[1][2, 1, 1]
        end
    end
end

@testset "Single-precision implicit canopy" begin
    for arch in test_architectures
        canopy = CanopyInterface(; soil_humidity=bare_soil_humidity(Float32))
        model = coupled_land_model(arch, Float32; atmosphere_land_interface_state=canopy,
                                    surface_storage=PrognosticCanopyAir())
        time_step!(model, 300.0)
        @test 280 < scalar(model.land.surface_storage.prognostic.temperature) < 330
        @test 0 < scalar(model.land.surface_storage.prognostic.specific_humidity) < 0.05
    end
end
