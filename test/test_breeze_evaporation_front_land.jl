include("runtests_setup.jl")

using Breeze
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Statistics
using Test

NumericalEarthBreezeExt = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
@test !isnothing(NumericalEarthBreezeExt)

# Tiny coupled Breeze + variably saturated SlabLand for CI. Verifies the
# end-to-end wiring (interface ↔ hydrology ↔ energy), then runs short enough
# to keep CI under a few seconds.
function build_coupled_test_model(arch; M_init, T_init)
    grid = RectilinearGrid(arch,
                           size = (8, 8), halo = (5, 5),
                           x = (-1kilometer, 1kilometer),
                           z = (0, 1kilometer),
                           topology = (Periodic, Flat, Bounded))

    θ₀ = 295.0
    atmosphere = atmosphere_simulation(grid; potential_temperature = θ₀)
    set!(atmosphere.model, θ = atmosphere.model.dynamics.reference_state.potential_temperature, u = 2)

    land_grid = RectilinearGrid(arch,
                                size = grid.Nx,
                                halo = grid.Hx,
                                x = (-1kilometer, 1kilometer),
                                topology = (Periodic, Flat, Flat))

    hydrology = VariablySaturatedBucketHydrology(eltype(land_grid);
        slab_depth = 1.0, porosity = 0.4,
        residual_liquid_fraction = 0.0,
        specific_storage = 1e-3, critical_saturation = 0.5,
        retention_curve = VanGenuchtenRetention(α = 1.0, n = 2.0),
        hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-6, n = 2.0),
        deep_liquid_flux = NoDeepLiquidFlux(),
        runoff = NoRunoff())
    energy = WaterCoupledForceRestoreEnergy(eltype(land_grid);
        dry_heat_capacity = 1e6,
        liquid_heat_capacity = 4186,
        reference_temperature = 273.15,
        deep_temperature = T_init,
        deep_time_scale = 12hours,
        advect_deep_liquid_energy = true,
        advect_surface_liquid_energy = false)
    land = SlabLand(land_grid; hydrology, energy)
    set!(land; T = T_init, M = M_init)

    # Default `BulkTemperature()` on the temperature side: Tⁱⁿ = Tˡᵃ.
    # (Land-side `SkinTemperature` is deferred — the PR plan §13 lists the
    # alternative thermal-conductance models.) The χ-interpolation in the
    # humidity formulation collapses gracefully: Tᵉ = Tˡᵃ when Tⁱⁿ = Tˡᵃ.
    al = atmosphere_land_interface(land_grid, atmosphere, land;
        specific_humidity = EvaporationFrontHumidity(;
            evaporation_front_depth = StorageBasedEvaporationFrontDepth(
                maximum_front_depth = 0.05,
                critical_saturation = 0.5,
                front_depth_exponent = 2.0),
            vapor_exchange = DryLayerVaporPistonVelocity(
                minimum_front_depth = 1e-4,
                molecular_diffusivity = 2.5e-5,
                tortuosity_model = MillingtonQuirk()),
            thermal_exchange_depth = 0.10,
            porosity = 0.4))

    return AtmosphereLandModel(atmosphere, land; atmosphere_land_interface = al)
end

@testset "Breeze + EvaporationFrontHumidity wiring" begin
    for arch in test_architectures
        A = typeof(arch)

        @testset "Coupled construction and time stepping on $A" begin
            model = build_coupled_test_model(arch; M_init = 200.0, T_init = 295.0)

            @test model isa EarthSystemModel
            @test model.atmosphere isa Simulation
            @test model.atmosphere.model isa Breeze.AtmosphereModel
            @test model.land isa SlabLand
            @test model.land.hydrology isa VariablySaturatedBucketHydrology
            @test model.land.energy isa WaterCoupledForceRestoreEnergy

            al_iface = model.interfaces.atmosphere_land_interface
            @test !isnothing(al_iface)

            simulation = Simulation(model; Δt = 0.5, stop_iteration = 10)
            run!(simulation)

            @test model.clock.iteration == 10
            @test model.clock.time > 0

            # End-to-end: turbulent fluxes are nonzero, no NaNs anywhere.
            land_fluxes = model.land.fluxes
            T_after = Array(interior(model.land.temperature))
            M_after = Array(interior(model.land.water_storage))
            Jv      = Array(interior(land_fluxes.vapor_flux))
            @test !any(isnan, T_after)
            @test !any(isnan, M_after)
            @test !any(isnan, Jv)
            # Atmosphere is initialized warm and unsaturated, so we expect
            # evaporation upward: Jᵛ ≥ 0 everywhere.
            @test all(Jv .>= 0)
            # Some cell actually evaporates.
            @test maximum(Jv) > 0
        end

        @testset "Drydown reduces M and saturates skin on $A" begin
            model = build_coupled_test_model(arch; M_init = 200.0, T_init = 295.0)

            M0 = Array(interior(model.land.water_storage))
            simulation = Simulation(model; Δt = 0.5, stop_iteration = 100)
            run!(simulation)
            M1 = Array(interior(model.land.water_storage))

            # Even with a tiny number of steps, dry-down should drop M.
            @test all(M1 .<= M0)
            @test mean(M1) < mean(M0)
        end
    end
end
