include("runtests_setup.jl")

using Oceananigans
using Oceananigans: set!, interior, CenterField
using Oceananigans.TimeSteppers: update_state!, time_step!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyInterface, CanopyConductanceHumidity, CriticalSaturation, AbsorbedPhotosyntheticallyActiveRadiation,
    SoilConductiveFlux,
    SimilarityTheoryFluxes, atmosphere_land_stability_functions

tiled_canopy(FT) = CanopyInterface(FT;
    soil_humidity = bare_soil_humidity(FT),
    canopy_humidity = CanopyConductanceHumidity(FT; leaf_area_index = 3.0, moisture_stress = CriticalSaturation(0.5),
                                                absorbed_par = AbsorbedPhotosyntheticallyActiveRadiation(FT)),
    ground_heat_flux = SoilConductiveFlux(1.5, 0.05))

land_roughness(ℓᵐ, ℓˢ) = SimilarityTheoryFluxes(Float64; stability_functions = atmosphere_land_stability_functions(Float64),
    momentum_roughness_length = ℓᵐ, temperature_roughness_length = ℓˢ, water_vapor_roughness_length = ℓˢ)

tiled_canopy_model(arch, FT = Float64; cover_fraction, wind = 3.0, interface_kw...) =
    tiled_land_model(arch, FT; vegetated = tiled_canopy(FT), cover_fraction, wind, interface_kw = (; interface_kw...))

@testset "Tiled surface-layer diagnostics" begin
    for arch in test_architectures
        grid, _, _, _ = coupled_land_components(arch)
        cover_fraction = Field{Center, Center, Nothing}(grid)
        set!(cover_fraction, 0.6)
        model = tiled_canopy_model(arch; cover_fraction)
        interface = model.interfaces.atmosphere_land_interface

        for include_subgrid_velocities in (false, true)
            heights = (; velocity_height = 2, temperature_height = 5, specific_humidity_height = 5, include_subgrid_velocities)
            diagnostics = (surface_layer_diagnostics(model; heights...),
                           surface_layer_diagnostics(model, interface.vegetated; heights...),
                           surface_layer_diagnostics(model, interface.bare; heights...))
            for f in (0.0, 0.6, 1.0)
                set!(cover_fraction, f)
                for name in keys(diagnostics[1])
                    blended, vegetated, bare = (Array(interior(compute!(Field(d[name])))) for d in diagnostics)
                    @test blended ≈ f .* vegetated .+ (1 - f) .* bare
                end
            end
        end
    end
end

@testset "Zero-area tiles do not contaminate surface-layer diagnostics" begin
    for arch in test_architectures, f in (0.0, 1.0)
        rough, smooth = land_roughness(3.0, 3.0), land_roughness(0.01, 0.001)
        model = tiled_canopy_model(arch; cover_fraction = f,
                                 vegetated_fluxes = f == 0 ? rough : smooth,
                                 bare_fluxes = f == 0 ? smooth : rough)
        interface = model.interfaces.atmosphere_land_interface
        active = f == 0 ? interface.bare : interface.vegetated
        absent = f == 0 ? interface.vegetated : interface.bare
        for include_subgrid_velocities in (false, true)
            heights = (; velocity_height = 2, temperature_height = 2, specific_humidity_height = 2, include_subgrid_velocities)
            blended = surface_layer_diagnostics(model; heights...)
            expected = surface_layer_diagnostics(model, active; heights...)
            undefined = surface_layer_diagnostics(model, absent; heights...)
            @test isnan(scalar(compute!(Field(undefined.T))))
            for name in keys(blended)
                @test scalar(compute!(Field(blended[name]))) == scalar(compute!(Field(expected[name])))
            end
        end
    end
end

@testset "TiledLandInterface" begin
    flux_names = (:sensible_heat, :latent_heat, :water_vapor, :x_momentum, :y_momentum)
    for arch in test_architectures
        f, Λ = 0.6, 1.5 / 0.05
        model = tiled_canopy_model(arch; cover_fraction = f)
        tiled = model.interfaces.atmosphere_land_interface
        T, veg, bare = tiled.temperature, tiled.vegetated, tiled.bare
        @test all(isfinite, scalar.((tiled.fluxes.sensible_heat, tiled.fluxes.latent_heat, T.ground_heat_flux, T.effective)))
        @test 285 < scalar(T.interface) < 320
        # Radiation is internalized per tile, so the slab receives only the conduction.
        @test scalar(model.land.fluxes.surface_energy_flux) ≈ -scalar(T.ground_heat_flux) atol = 1e-10

        for name in flux_names
            @test scalar(getproperty(tiled.fluxes, name)) ≈
                  f * scalar(getproperty(veg.fluxes, name)) + (1 - f) * scalar(getproperty(bare.fluxes, name)) rtol = 1e-12
        end
        Tᵇ = scalar(bare.temperature)
        blend(name, bare_value) = f * scalar(getproperty(veg.temperature, name)) + (1 - f) * bare_value
        @test scalar(T.interface) ≈ blend(:interface, Tᵇ) rtol = 1e-12
        @test scalar(T.ground) ≈ blend(:ground, Tᵇ) rtol = 1e-12
        @test scalar(T.ground_heat_flux) ≈ blend(:ground_heat_flux, Λ * (Tᵇ - scalar(model.land.temperature))) rtol = 1e-12
        @test scalar(T.ground_latent_heat) ≈ blend(:ground_latent_heat, scalar(bare.fluxes.latent_heat)) rtol = 1e-12
        @test scalar(T.canopy_latent_heat) == scalar(veg.temperature.canopy_latent_heat)
        @test scalar(T.canopy) == scalar(veg.temperature.canopy)
        σ = model.radiation.stefan_boltzmann_constant
        ϵ = model.radiation.surface_properties.land.emissivity
        ℐꜜˡʷ = scalar(model.interfaces.exchanger.radiation.state.ℐꜜˡʷ)
        @test σ * scalar(T.effective)^4 ≈
              f * σ * scalar(veg.temperature.effective)^4 + (1 - f) * (ϵ * σ * Tᵇ^4 - (1 - ϵ) * ℐꜜˡʷ) rtol = 1e-12
        @test scalar(T.effective_albedo) ≈ blend(:effective_albedo, model.radiation.surface_properties.land.albedo)

        @test scalar(veg.temperature.ground) < scalar(veg.temperature.canopy)
        @test scalar(veg.temperature.canopy_latent_heat) > scalar(veg.temperature.ground_latent_heat)
        @test scalar(veg.temperature.ground_latent_heat) < scalar(bare.fluxes.latent_heat)

        # Endpoints reduce bit-for-bit to one tile, and f = 1 to a standalone CanopyInterface.
        m1, m0 = tiled_canopy_model(arch; cover_fraction = 1.0), tiled_canopy_model(arch; cover_fraction = 0.0)
        t1, t0 = m1.interfaces.atmosphere_land_interface, m0.interfaces.atmosphere_land_interface
        for name in flux_names
            @test scalar(getproperty(t1.fluxes, name)) == scalar(getproperty(t1.vegetated.fluxes, name))
            @test scalar(getproperty(t0.fluxes, name)) == scalar(getproperty(t0.bare.fluxes, name))
        end
        @test scalar(t1.temperature.ground_heat_flux) == scalar(t1.vegetated.temperature.ground_heat_flux)
        @test scalar(t0.temperature.ground) == scalar(t0.bare.temperature)
        @test scalar(m0.land.fluxes.surface_energy_flux) ≈ -Λ * (scalar(t0.bare.temperature) - scalar(m0.land.temperature)) atol = 1e-10

        standalone = coupled_land_model(arch; atmosphere_land_interface_state = tiled_canopy(Float64))
        s = standalone.interfaces.atmosphere_land_interface
        @test scalar(t1.fluxes.sensible_heat) == scalar(s.fluxes.sensible_heat)
        @test scalar(t1.fluxes.latent_heat) == scalar(s.fluxes.latent_heat)
        @test scalar(t1.temperature.ground_heat_flux) == scalar(s.temperature.ground_heat_flux)

        rough = tiled_canopy_model(arch; cover_fraction = 0.5, wind = 6.0,
                                 vegetated_fluxes = land_roughness(1.0, 0.1), bare_fluxes = land_roughness(1e-3, 1e-4))
        tr = rough.interfaces.atmosphere_land_interface
        @test scalar(tr.vegetated.fluxes.friction_velocity) > scalar(tr.bare.fluxes.friction_velocity)

        cover_field = CenterField(model.land.grid)
        set!(cover_field, f)
        tf = tiled_canopy_model(arch; cover_fraction = cover_field).interfaces.atmosphere_land_interface
        @test scalar(tf.fluxes.sensible_heat) ≈ scalar(tiled.fluxes.sensible_heat) rtol = 1e-12
        @test scalar(tf.fluxes.latent_heat) ≈ scalar(tiled.fluxes.latent_heat) rtol = 1e-12

        M₀ = scalar(model.land.water_storage)
        for _ in 1:20
            time_step!(model, 600.0)
        end
        @test scalar(model.land.water_storage) < M₀
        @test all(isfinite, scalar.((tiled.fluxes.sensible_heat, tiled.fluxes.latent_heat, model.land.temperature)))
    end

    for FT in (Float32, Float64)
        model = tiled_canopy_model(CPU(), FT; cover_fraction = FT(0.5))
        𝒬ᵀ = scalar(model.interfaces.atmosphere_land_interface.fluxes.sensible_heat)
        @test 𝒬ᵀ isa FT && isfinite(𝒬ᵀ)
    end
end
