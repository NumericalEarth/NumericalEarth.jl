include("runtests_setup.jl")

using Oceananigans
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth.Lands: viscosity_correction

@testset "VariablySaturatedHydrology diagnostics" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch;
                               size = 1,
                               x = (0, 1),
                               y = (0, 1),
                               z = (-1, 0),
                               topology = (Flat, Flat, Bounded))

        # hˡᵃ = 1 m, ρˡ = 1000, ν = 0.4, hˢˢ = 1000, θʳ = 0 ⇒ M⁺ = 400 kg/m².
        hydrology = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0,
            porosity = 0.4,
            residual_liquid_fraction = 0.0,
            storage_height = 1000,
            retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 1.0, pore_size_uniformity = 2.0),
            hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 1e-6, pore_size_uniformity = 2.0),
            deep_liquid_flux = NoDeepLiquidFlux(),
            runoff = NoRunoff(),
        )
        land = SlabLand(grid; hydrology)

        # Test cases: M = 200, 400, 401.
        set!(land; M = 200.0)
        @test only(Array(interior(land.saturation))) ≈ 0.5

        set!(land; M = 400.0)
        @test only(Array(interior(land.saturation))) ≈ 1.0

        set!(land; M = 401.0)   # over-saturated; saturation clamps at 1
        @test only(Array(interior(land.saturation))) ≈ 1.0
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
            storage_height = 1000,
            retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 1.0, pore_size_uniformity = 2.0),
            hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 1e-6, pore_size_uniformity = 2.0),
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
        @test only(Array(interior(land.water_storage))) ≈ 200.0

        # Constant evaporation: dM/dt = -Jᵛ = -0.01.
        set!(land; M = 200.0)
        fill!(land.fluxes.vapor_flux, 0.01)
        time_step!(land, 1000.0)
        @test only(Array(interior(land.water_storage))) ≈ 190.0

        # Constant precip below infiltration capacity: dM/dt = +Pˡ.
        hydrology_capped = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0,
            porosity = 0.4,
            storage_height = 1000,
            retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 1.0, pore_size_uniformity = 2.0),
            hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 1e-6, pore_size_uniformity = 2.0),
            deep_liquid_flux = NoDeepLiquidFlux(),
            runoff = InfiltrationCapacityRunoff(infiltration_capacity = 7.0),
        )
        land_capped = SlabLand(grid; hydrology = hydrology_capped)
        set!(land_capped; M = 0.0)
        fill!(land_capped.fluxes.vapor_flux, 0.0)
        fill!(land_capped.fluxes.liquid_precipitation_flux, 5.0)  # below capacity 7
        time_step!(land_capped, 1.0)
        @test only(Array(interior(land_capped.water_storage))) ≈ 5.0
        @test only(Array(interior(land_capped.diagnostics.surface_runoff))) ≈ 0.0

        # Precip above capacity: surface runoff = Pˡ - capacity.
        set!(land_capped; M = 0.0)
        fill!(land_capped.fluxes.liquid_precipitation_flux, 10.0)  # above capacity 7
        time_step!(land_capped, 1.0)
        @test only(Array(interior(land_capped.water_storage))) ≈ 7.0
        @test only(Array(interior(land_capped.diagnostics.surface_runoff))) ≈ 3.0

        # Free drainage: dM/dt = -ρˡ K_b. At full saturation K = K_sat Θ(T), where Θ is
        # the viscosity correction, so the rate carries the slab temperature.
        hydrology_drain = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0,
            porosity = 0.4,
            storage_height = 1000,
            retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 1.0, pore_size_uniformity = 2.0),
            hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 1e-6, pore_size_uniformity = 2.0),
            deep_liquid_flux = FreeDrainageFlux(),
            runoff = NoRunoff(),
        )
        land_drain = SlabLand(grid; hydrology = hydrology_drain)
        set!(land_drain; T = 293.0, M = 400.0)  # fully saturated, warmer than the reference
        fill!(land_drain.fluxes.vapor_flux, 0)
        fill!(land_drain.fluxes.liquid_precipitation_flux, 0)
        viscosity = hydrology_drain.hydraulic_conductivity.water_viscosity
        Θ = viscosity_correction(viscosity, 293.0)
        time_step!(land_drain, 100.0)
        expected = 400.0 - 100 * 1000 * 1e-6 * Θ
        @test only(Array(interior(land_drain.water_storage))) ≈ expected atol = 1e-5
        # 293 K is 13 % less viscous than the reference, so the isothermal 0.1 kg drop is
        # outside the tolerance above — the temperature really is being read.
        @test !isapprox(only(Array(interior(land_drain.water_storage))), 399.9, atol = 1e-3)
        # The correction is unity at the reference temperature, and only there.
        @test viscosity_correction(viscosity, viscosity.reference_temperature) == 1
    end
end
