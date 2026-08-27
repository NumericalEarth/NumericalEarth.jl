include("runtests_setup.jl")

using Oceananigans
using Oceananigans: set!, interior
using Oceananigans: prognostic_state, restore_prognostic_state!
using Oceananigans.TimeSteppers: update_state!, time_step!
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    CanopyAirSpace, CanopyConductanceHumidity, DryLayerHumidity, StorageBasedDryLayerDepth,
    DryLayerVaporPistonVelocity, ConstantTortuosity, CriticalSaturation, InteractiveAbsorbedPAR,
    SoilConductiveFlux, SoilSkinTemperature, DiagnosticSkin, PrognosticSkin,
    DiagnosticCanopyAir, PrognosticCanopyAir
using NumericalEarth.Atmospheres: PrescribedAtmosphere
using NumericalEarth.Lands: SlabLand, SlabEnergy, BucketHydrology
using NumericalEarth.Radiations: PrescribedRadiation, SurfaceRadiationProperties

checkpoint_test_cas(FT; storage) = CanopyAirSpace(FT;
    soil = DryLayerHumidity(FT;
        dry_layer_depth = StorageBasedDryLayerDepth(FT; maximum_dry_layer_depth = 0.015,
                                                    dry_layer_onset_saturation = 0.5, dry_layer_exponent = 2),
        vapor_exchange  = DryLayerVaporPistonVelocity(FT; minimum_dry_layer_depth = 1e-3,
                                                      molecular_diffusivity = 2.4e-5, tortuosity = ConstantTortuosity()),
        thermal_exchange_depth = 0.05, porosity = 0.4),
    canopy = CanopyConductanceHumidity(FT; leaf_area_index = 4.0, moisture_stress = CriticalSaturation(0.5),
                                       absorbed_par = InteractiveAbsorbedPAR(FT)),
    soil_skin_flux = SoilConductiveFlux(1.5, 0.05), storage)

function checkpoint_test_model(arch; temperature, humidity = temperature)
    FT = Float64
    grid = LatitudeLongitudeGrid(arch, FT; size = 1, latitude = 10, longitude = 10,
                                 z = (-1, 0), topology = (Flat, Flat, Bounded))
    atmosphere = PrescribedAtmosphere(grid; surface_layer_height = 10, boundary_layer_height = 512)
    fill!(parent(atmosphere.temperature), 300.0)
    fill!(parent(atmosphere.specific_humidity), 0.008)
    fill!(parent(atmosphere.velocities.u), 3.0)
    fill!(parent(atmosphere.pressure), 101325.0)
    land = SlabLand(grid; hydrology = BucketHydrology(FT; maximum_water_storage = 150.0), energy = SlabEnergy(FT))
    set!(land; T = 298.0)
    fill!(parent(land.water_storage), 45.0)
    radiation = PrescribedRadiation(grid; ocean_surface = nothing, sea_ice_surface = nothing,
                                    land_surface = SurfaceRadiationProperties(0.2, 0.95))
    fill!(parent(radiation.downwelling_shortwave), 600.0)
    fill!(parent(radiation.downwelling_longwave), 350.0)
    update_state!(radiation)
    model = AtmosphereLandModel(atmosphere, land; radiation,
                atmosphere_land_interface_temperature = temperature,
                atmosphere_land_interface_specific_humidity = humidity)
    update_state!(model.land)
    update_state!(model)
    return model
end

@inline value1(f) = Array(interior(f))[1, 1, 1]

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
    return (value1(f1.latent_heat), value1(f1.sensible_heat)),
           (value1(f2.latent_heat), value1(f2.sensible_heat))
end

@testset "Prognostic interface state checkpointing" begin
    for arch in test_architectures
        # --- prognostic canopy air: the node state must survive the round trip.
        cas() = checkpoint_test_cas(Float64; storage = PrognosticCanopyAir())
        m1 = checkpoint_test_model(arch; temperature = cas())
        m2 = checkpoint_test_model(arch; temperature = cas())
        state = prognostic_state(m1)
        @test state.interfaces.atmosphere_land_interface !== nothing
        (LE1, H1), (LE2, H2) = roundtrip!(m1, m2)
        Ts1 = m1.interfaces.atmosphere_land_interface.temperature
        Ts2 = m2.interfaces.atmosphere_land_interface.temperature
        @test value1(Ts2.state.temperature) ≈ value1(Ts1.state.temperature)
        @test value1(Ts2.state.specific_humidity) ≈ value1(Ts1.state.specific_humidity)
        @test LE2 ≈ LE1
        @test H2 ≈ H1

        # --- prognostic skin: the interface-temperature field is the state.
        skin() = SoilSkinTemperature(1.5, 0.05; storage = PrognosticSkin(heat_capacity = 1e5))
        soil() = DryLayerHumidity(Float64;
            dry_layer_depth = StorageBasedDryLayerDepth(Float64; maximum_dry_layer_depth = 0.015,
                                                        dry_layer_onset_saturation = 0.5, dry_layer_exponent = 2),
            vapor_exchange  = DryLayerVaporPistonVelocity(Float64; minimum_dry_layer_depth = 1e-3,
                                                          molecular_diffusivity = 2.4e-5, tortuosity = ConstantTortuosity()),
            thermal_exchange_depth = 0.05, porosity = 0.4)
        m1 = checkpoint_test_model(arch; temperature = skin(), humidity = soil())
        m2 = checkpoint_test_model(arch; temperature = skin(), humidity = soil())
        state = prognostic_state(m1)
        @test state.interfaces.atmosphere_land_interface !== nothing
        (LE1, H1), (LE2, H2) = roundtrip!(m1, m2)
        @test value1(m2.interfaces.atmosphere_land_interface.temperature) ≈
              value1(m1.interfaces.atmosphere_land_interface.temperature)
        @test LE2 ≈ LE1
        @test H2 ≈ H1

        # --- diagnostic formulations contribute no interface state (everything
        # they need is regenerated by the next update), yet still round-trip
        # exactly given the checkpointed slab forcing; and restoring an old
        # checkpoint without an interface entry is tolerated.
        cd() = checkpoint_test_cas(Float64; storage = DiagnosticCanopyAir())
        m1 = checkpoint_test_model(arch; temperature = cd())
        m2 = checkpoint_test_model(arch; temperature = cd())
        @test prognostic_state(m1).interfaces.atmosphere_land_interface === nothing
        (LE1, H1), (LE2, H2) = roundtrip!(m1, m2)
        @test LE2 ≈ LE1
        @test H2 ≈ H1
        restore_prognostic_state!(m2.interfaces, nothing)
        restore_prognostic_state!(m2.interfaces, (;))
        @test true
    end
end
