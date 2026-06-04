include("runtests_setup.jl")

using Oceananigans
using Oceananigans.TimeSteppers: time_step!

@testset "VariablySaturatedHydrology diagnostics" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch;
                               size = 1,
                               x = (0, 1),
                               y = (0, 1),
                               z = (-1, 0),
                               topology = (Flat, Flat, Bounded))

        # d = 1 m, ρˡ = 1000, ν = 0.4, Sₛ = 1e-3, θʳ = 0 ⇒ M⁺ = 400 kg/m².
        hydrology = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0,
            porosity = 0.4,
            residual_liquid_fraction = 0.0,
            specific_storage = 1e-3,
            critical_saturation = 0.5,
            retention_curve = VanGenuchtenRetention(α = 1.0, n = 2.0),
            hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-6, n = 2.0),
            deep_liquid_flux = NoDeepLiquidFlux(),
            runoff = NoRunoff(),
        )
        land = SlabLand(grid; hydrology)

        # Test cases: M = 200, 400, 401.
        set!(land; M = 200.0)
        @test @allowscalar(interior(land.saturation)[1, 1, 1]) ≈ 0.5

        set!(land; M = 400.0)
        @test @allowscalar(interior(land.saturation)[1, 1, 1]) ≈ 1.0

        set!(land; M = 401.0)   # over-saturated; saturation clamps at 1
        @test @allowscalar(interior(land.saturation)[1, 1, 1]) ≈ 1.0
    end
end

@testset "VariablySaturatedHydrology conservation" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch;
                               size = 1,
                               x = (0, 1),
                               y = (0, 1),
                               z = (-1, 0),
                               topology = (Flat, Flat, Bounded))

        # No-flux: M(t) = M₀.
        hydrology = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0,
            porosity = 0.4,
            specific_storage = 1e-3,
            critical_saturation = 0.5,
            retention_curve = VanGenuchtenRetention(α = 1.0, n = 2.0),
            hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-6, n = 2.0),
            deep_liquid_flux = NoDeepLiquidFlux(),
            runoff = NoRunoff(),
        )
        land = SlabLand(grid; hydrology)
        set!(land; M = 200.0)
        fill!(land.fluxes.vapor_flux, 0)
        fill!(land.fluxes.liquid_precipitation_flux, 0)

        Δt = 100.0
        for _ in 1:10
            time_step!(land, Δt)
        end
        @test @allowscalar(interior(land.water_storage)[1, 1, 1]) ≈ 200.0

        # Constant evaporation: dM/dt = -Jᵛ = -0.01.
        set!(land; M = 200.0)
        fill!(land.fluxes.vapor_flux, 0.01)
        time_step!(land, 1000.0)
        @test @allowscalar(interior(land.water_storage)[1, 1, 1]) ≈ 190.0

        # Constant precip below infiltration capacity: dM/dt = +Pˡ.
        hydrology_capped = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0,
            porosity = 0.4,
            specific_storage = 1e-3,
            critical_saturation = 0.5,
            retention_curve = VanGenuchtenRetention(α = 1.0, n = 2.0),
            hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-6, n = 2.0),
            deep_liquid_flux = NoDeepLiquidFlux(),
            runoff = InfiltrationCapacityRunoff(infiltration_capacity = 7.0),
        )
        land_capped = SlabLand(grid; hydrology = hydrology_capped)
        set!(land_capped; M = 0.0)
        fill!(land_capped.fluxes.vapor_flux, 0.0)
        fill!(land_capped.fluxes.liquid_precipitation_flux, 5.0)  # below capacity 7
        time_step!(land_capped, 1.0)
        @test @allowscalar(interior(land_capped.water_storage)[1, 1, 1]) ≈ 5.0
        @test @allowscalar(interior(land_capped.diagnostics.surface_runoff)[1, 1, 1]) ≈ 0.0

        # Precip above capacity: surface runoff = Pˡ - capacity.
        set!(land_capped; M = 0.0)
        fill!(land_capped.fluxes.liquid_precipitation_flux, 10.0)  # above capacity 7
        time_step!(land_capped, 1.0)
        @test @allowscalar(interior(land_capped.water_storage)[1, 1, 1]) ≈ 7.0
        @test @allowscalar(interior(land_capped.diagnostics.surface_runoff)[1, 1, 1]) ≈ 3.0

        # Free drainage: dM/dt = -ρˡ K_b. With K_sat = 1e-6, ρˡ = 1000:
        # at full saturation, K = K_sat, so Jˡ_b = -1e-3 kg/m²/s.
        hydrology_drain = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0,
            porosity = 0.4,
            specific_storage = 1e-3,
            critical_saturation = 0.5,
            retention_curve = VanGenuchtenRetention(α = 1.0, n = 2.0),
            hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-6, n = 2.0),
            deep_liquid_flux = FreeDrainageFlux(),
            runoff = NoRunoff(),
        )
        land_drain = SlabLand(grid; hydrology = hydrology_drain)
        set!(land_drain; M = 400.0)  # fully saturated
        fill!(land_drain.fluxes.vapor_flux, 0)
        fill!(land_drain.fluxes.liquid_precipitation_flux, 0)
        time_step!(land_drain, 100.0)
        # Expect M to decrease by 100 * 1e-3 = 0.1
        @test @allowscalar(interior(land_drain.water_storage)[1, 1, 1]) ≈ 399.9 atol = 1e-3
    end
end
