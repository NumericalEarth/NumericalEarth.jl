include("runtests_setup.jl")

using Oceananigans
using Oceananigans: prognostic_state, restore_prognostic_state!
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyAirSpace, CanopyConductanceHumidity, CriticalSaturation, InteractiveAbsorbedPAR,
    SoilConductiveFlux, SoilSkinTemperature, DiagnosticSkin, PrognosticSkin,
    DiagnosticCanopyAir, PrognosticCanopyAir, compute_atmosphere_land_fluxes!

checkpoint_test_cas(FT; storage) = CanopyAirSpace(FT;
    soil = bare_soil_humidity(FT),
    canopy = CanopyConductanceHumidity(FT; leaf_area_index = 4.0, moisture_stress = CriticalSaturation(0.5),
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
    for _ in 1:continuation
        time_step!(m1, Δt)
        time_step!(m2, Δt)
    end
    f1 = m1.interfaces.atmosphere_land_interface.fluxes
    f2 = m2.interfaces.atmosphere_land_interface.fluxes
    return (scalar(f1.latent_heat), scalar(f1.sensible_heat)),
           (scalar(f2.latent_heat), scalar(f2.sensible_heat))
end

@testset "Prognostic land surface checkpointing" begin
    for arch in test_architectures
        # --- prognostic canopy air: the node state must survive the round trip.
        cas() = checkpoint_test_cas(Float64; storage = PrognosticCanopyAir())
        m1 = checkpoint_test_model(arch; temperature = cas())
        m2 = checkpoint_test_model(arch; temperature = cas())
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

        # --- diagnostic surface temperatures are reconstructed after restart.
        cd() = checkpoint_test_cas(Float64; storage = DiagnosticCanopyAir())
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
        canopy = checkpoint_test_cas(Float64; storage = PrognosticCanopyAir())
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
        canopy = checkpoint_test_cas(Float64; storage = PrognosticCanopyAir())
        skin = SoilSkinTemperature(1.5, 0.05; storage = PrognosticSkin(heat_capacity = 1e5))
        surface = TiledLandSurface(LandSurface(canopy); fraction = 0.6,
                                    bare = LandSurface(skin; specific_humidity = canopy.soil))
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
