include("runtests_setup.jl")

using Oceananigans
using Oceananigans: prognostic_state, restore_prognostic_state!
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyGroundSurface, CanopyConductanceHumidity, CriticalSaturation, InteractiveAbsorbedPAR,
    SoilConductiveFlux, SoilSkinTemperature, DiagnosticSkin, PrognosticSkin,
    DiagnosticCanopyAir, PrognosticCanopyAir, compute_atmosphere_land_fluxes!

checkpoint_test_canopy_surface(FT; storage) = CanopyGroundSurface(FT;
    soil_humidity = bare_soil_humidity(FT),
    canopy_humidity = CanopyConductanceHumidity(FT; leaf_area_index = 4.0, moisture_stress = CriticalSaturation(0.5),
                                       absorbed_par = InteractiveAbsorbedPAR(FT)),
    ground_heat_flux = SoilConductiveFlux(1.5, 0.05), storage)

checkpoint_test_model(arch; temperature, humidity = temperature) =
    coupled_land_model(arch; atmosphere_land_interface_temperature = temperature,
                       atmosphere_land_interface_specific_humidity = humidity)

# Step `m1` forward, checkpoint it, restore into the freshly built `m2`, then step
# both together and require identical fluxes — the restored run must be
# indistinguishable from the uninterrupted one.
function roundtrip!(m1, m2; spinup = 6, continuation = 6, Δt = 300.0)
    for _ in 1:spinup
        time_step!(m1, Δt)
    end
    state = deepcopy(prognostic_state(m1))
    restore_prognostic_state!(m2, state)
    @test prognostic_state(m2.interfaces) == state.interfaces
    @test prognostic_state(m2.land) == state.land
    for _ in 1:continuation
        time_step!(m1, Δt)
        time_step!(m2, Δt)
    end
    f1 = m1.interfaces.atmosphere_land_interface.fluxes
    f2 = m2.interfaces.atmosphere_land_interface.fluxes
    return (scalar(f1.latent_heat), scalar(f1.sensible_heat)),
           (scalar(f2.latent_heat), scalar(f2.sensible_heat))
end

@testset "Checkpoint continuation with changing forcing" begin
    for arch in test_architectures
        models = ntuple(2) do _
            grid, _, land, _ = coupled_land_components(arch)
            times = [0.0, 1000.0]
            atmosphere = PrescribedAtmosphere(grid, times)
            radiation = PrescribedRadiation(grid, times)
            fill!(parent(atmosphere.velocities.u), 3)
            fill!(parent(atmosphere.specific_humidity), 0.008)
            for (n, time) in enumerate(times)
                set!(atmosphere.temperature[n], 280 + 0.04time)
                set!(atmosphere.precipitation_flux.rain[n], 1e-6time)
                set!(radiation.downwelling_shortwave[n], 200 + 0.6time)
                set!(radiation.downwelling_longwave[n], 300 + 0.1time)
            end
            return AtmosphereLandModel(atmosphere, land; radiation)
        end
        m1, m2 = models
        fluxes1, fluxes2 = roundtrip!(m1, m2; spinup = 5, continuation = 1, Δt = 100.0)
        @test fluxes1 == fluxes2
        @test Array(interior(m1.land.temperature)) == Array(interior(m2.land.temperature))
        @test Array(interior(m1.land.water_storage)) == Array(interior(m2.land.water_storage))
    end
end

@testset "Prognostic land surface checkpointing" begin
    for arch in test_architectures
        # --- prognostic canopy air: the node state must survive the round trip.
        canopy_surface() = checkpoint_test_canopy_surface(Float64; storage = PrognosticCanopyAir())
        m1 = checkpoint_test_model(arch; temperature = canopy_surface())
        m2 = checkpoint_test_model(arch; temperature = canopy_surface())
        state = prognostic_state(m1)
        @test !isempty(state.land.surface)
        (LE1, H1), (LE2, H2) = roundtrip!(m1, m2)
        @test scalar(m2.land.surface.prognostic.temperature) ≈ scalar(m1.land.surface.prognostic.temperature)
        @test scalar(m2.land.surface.prognostic.specific_humidity) ≈ scalar(m1.land.surface.prognostic.specific_humidity)
        @test LE2 ≈ LE1
        @test H2 ≈ H1

        # --- prognostic skin temperature survives the round trip.
        skin() = SoilSkinTemperature(1.5, 0.05; storage = PrognosticSkin(heat_capacity = 1e5))
        m1 = checkpoint_test_model(arch; temperature = skin(), humidity = bare_soil_humidity(Float64))
        m2 = checkpoint_test_model(arch; temperature = skin(), humidity = bare_soil_humidity(Float64))
        state = prognostic_state(m1)
        @test !isempty(state.land.surface)
        (LE1, H1), (LE2, H2) = roundtrip!(m1, m2)
        @test scalar(m2.interfaces.atmosphere_land_interface.temperature) ≈
              scalar(m1.interfaces.atmosphere_land_interface.temperature)
        @test LE2 ≈ LE1
        @test H2 ≈ H1

        # --- diagnostic surface temperatures survive the round trip.
        cd() = checkpoint_test_canopy_surface(Float64; storage = DiagnosticCanopyAir())
        m1 = checkpoint_test_model(arch; temperature = cd())
        m2 = checkpoint_test_model(arch; temperature = cd())
        @test isempty(prognostic_state(m1).land.surface)
        (LE1, H1), (LE2, H2) = roundtrip!(m1, m2)
        @test LE2 ≈ LE1
        @test H2 ≈ H1
    end
end

@testset "Flux evaluation preserves land surface storage" begin
    for arch in test_architectures
        canopy = checkpoint_test_canopy_surface(Float64; storage = PrognosticCanopyAir())
        skin = SoilSkinTemperature(1.5, 0.05; storage = PrognosticSkin(heat_capacity = 1e5))
        for surface in (LandSurface(canopy), LandSurface(skin; specific_humidity = bare_soil_humidity(Float64)))
            model = coupled_land_model(arch; surface)
            time_step!(model, 300.0)
            state = model.land.surface.prognostic
            before = map(f -> copy(Array(interior(f))), state)
            compute_atmosphere_land_fluxes!(model, 300.0)
            candidate = map(f -> copy(Array(interior(f))), model.land.surface.next_state)
            fluxes = model.interfaces.atmosphere_land_interface.fluxes
            heat = scalar(fluxes.latent_heat)
            compute_atmosphere_land_fluxes!(model, 300.0)
            @test map(f -> Array(interior(f)), state) == before
            @test map(f -> Array(interior(f)), model.land.surface.next_state) == candidate
            @test scalar(fluxes.latent_heat) == heat
            time_step!(model, 300.0)
            @test map(f -> Array(interior(f)), state) == candidate
        end
    end
end

@testset "Tiled canopy and skin storage restart together" begin
    for arch in test_architectures
        canopy = checkpoint_test_canopy_surface(Float64; storage = PrognosticCanopyAir())
        skin = SoilSkinTemperature(1.5, 0.05; storage = PrognosticSkin(heat_capacity = 1e5))
        surface = TiledLandSurface(LandSurface(canopy); cover_fraction = 0.6,
                                    bare = LandSurface(skin; specific_humidity = canopy.soil_humidity))
        m1 = coupled_land_model(arch; surface)
        m2 = coupled_land_model(arch; surface)
        fluxes1, fluxes2 = roundtrip!(m1, m2)
        @test fluxes1 == fluxes2
        for tile in (:vegetated, :bare)
            state1 = getproperty(m1.land.surface, tile).prognostic
            state2 = getproperty(m2.land.surface, tile).prognostic
            @test map(f -> Array(interior(f)), state1) == map(f -> Array(interior(f)), state2)
        end
    end
end
