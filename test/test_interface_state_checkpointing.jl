include("runtests_setup.jl")

using Oceananigans
using Oceananigans: prognostic_state, restore_prognostic_state!
using Oceananigans.TimeSteppers: time_step!, update_state!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyInterface, CanopyConductanceHumidity, CriticalSaturation, AbsorbedPhotosyntheticallyActiveRadiation,
    SoilConductiveFlux, SoilSkinTemperature

checkpoint_canopy() = CanopyInterface(Float64; soil_humidity = bare_soil_humidity(Float64),
    canopy_humidity = CanopyConductanceHumidity(Float64; leaf_area_index = 4.0, moisture_stress = CriticalSaturation(0.5),
                                                absorbed_par = AbsorbedPhotosyntheticallyActiveRadiation(Float64)),
    ground_heat_flux = SoilConductiveFlux(1.5, 0.05))

checkpoint_skin() = SoilSkinTemperature(1.5, 0.05)
arrays(fields) = map(f -> Array(interior(f)), fields)

# Step `m1`, restore its checkpoint into the fresh `m2`, step both, and return each model's (latent, sensible) heat flux.
function roundtrip!(m1, m2; spinup = 6, continuation = 6, Δt = 300.0)
    foreach(_ -> time_step!(m1, Δt), 1:spinup)
    state = deepcopy(prognostic_state(m1))
    restore_prognostic_state!(m2, state)
    @test prognostic_state(m2.interfaces) == state.interfaces
    @test prognostic_state(m2.land) == state.land
    foreach(_ -> (time_step!(m1, Δt); time_step!(m2, Δt)), 1:continuation)
    fluxes(m) = m.interfaces.atmosphere_land_interface.fluxes
    return map(m -> (scalar(fluxes(m).latent_heat), scalar(fluxes(m).sensible_heat)), (m1, m2))
end

@testset "Checkpoint continuation with changing forcing" begin
    for arch in test_architectures
        m1, m2 = ntuple(2) do _
            grid, _, land, _ = coupled_land_components(arch)
            times = [0.0, 1000.0]
            atmosphere, radiation = PrescribedAtmosphere(grid, times), PrescribedRadiation(grid, times)
            fill!(parent(atmosphere.velocities.u), 3)
            fill!(parent(atmosphere.specific_humidity), 0.008)
            for (n, time) in enumerate(times)
                set!(atmosphere.temperature[n], 280 + 0.04time)
                set!(atmosphere.precipitation_flux.rain[n], 1e-6time)
                set!(radiation.downwelling_shortwave[n], 200 + 0.6time)
                set!(radiation.downwelling_longwave[n], 300 + 0.1time)
            end
            AtmosphereLandModel(atmosphere, land; radiation)
        end
        fluxes1, fluxes2 = roundtrip!(m1, m2; spinup = 5, continuation = 1, Δt = 100.0)
        @test fluxes1 == fluxes2
        @test Array(interior(m1.land.temperature)) == Array(interior(m2.land.temperature))
        @test Array(interior(m1.land.water_storage)) == Array(interior(m2.land.water_storage))
    end
end

@testset "Prognostic land surface checkpointing" begin
    cases = ((checkpoint_canopy(), PrognosticCanopyAir()),
             ((temperature=checkpoint_skin(), specific_humidity=bare_soil_humidity(Float64)), PrognosticSkin()),
             (checkpoint_canopy(), nothing))
    for arch in test_architectures, (state, storage) in cases
        m1, m2 = ntuple(_ -> coupled_land_model(arch; atmosphere_land_interface_state=state,
                                                atmosphere_land_interface_storage=storage), 2)
        (LE1, H1), (LE2, H2) = roundtrip!(m1, m2)
        @test LE2 ≈ LE1
        @test H2 ≈ H1
        storage_state(m) = prognostic_state(m.interfaces.atmosphere_land_interface.storage)
        @test storage_state(m2) == storage_state(m1)
    end
end

@testset "Updating the state preserves interface storage" begin
    cases = ((checkpoint_canopy(), PrognosticCanopyAir()),
             ((temperature=checkpoint_skin(), specific_humidity=bare_soil_humidity(Float64)), PrognosticSkin()))
    for arch in test_architectures, (formulation, storage) in cases
        model = coupled_land_model(arch; atmosphere_land_interface_state=formulation,
                                   atmosphere_land_interface_storage=storage)
        time_step!(model, 300.0)
        state = model.interfaces.atmosphere_land_interface.storage.prognostic
        before = arrays(state)
        latent_heat = model.interfaces.atmosphere_land_interface.fluxes.latent_heat
        heat = scalar(latent_heat)
        update_state!(model)
        update_state!(model)
        @test arrays(state) == before
        @test scalar(latent_heat) == heat
        time_step!(model, 300.0)
        @test arrays(state) != before
    end
end

@testset "Tiled canopy storage restarts" begin
    for arch in test_architectures
        m1, m2 = ntuple(_ -> tiled_land_model(arch; vegetated = checkpoint_canopy(), cover_fraction = 0.6,
                                              interface_kw = (; storage = PrognosticCanopyAir())), 2)
        fluxes1, fluxes2 = roundtrip!(m1, m2)
        @test fluxes1 == fluxes2
        storage_state(m) = prognostic_state(m.interfaces.atmosphere_land_interface.vegetated.storage)
        @test storage_state(m2) == storage_state(m1)
    end
end
