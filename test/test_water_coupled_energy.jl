include("runtests_setup.jl")

using Oceananigans
using Oceananigans.TimeSteppers: time_step!

@testset "WaterCoupledEnergy force-restore" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch;
                               size = 1,
                               x = (0, 1),
                               y = (0, 1),
                               z = (-1, 0),
                               topology = (Flat, Flat, Bounded))

        # Dry slab (M = 0) so C reduces to C_dry. Pure restoration:
        #     T(t) − Tᵈ = (T₀ − Tᵈ) exp(−(Λ/C) t).
        Cdry  = 1.0e6
        Λ     = 5.0          # W m⁻² K⁻¹
        Tᵈ    = 280.0
        T₀    = 290.0
        Δt    = 100.0
        steps = 100          # ~10000 s ≈ 0.05 * τ; small enough for the linear truncation

        hydrology = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0, porosity = 0.4, storage_height = 1000,
            retention_curve = VanGenuchtenRetention(α = 1.0, n = 2.0),
            hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-6, n = 2.0),
            deep_liquid_flux = NoDeepLiquidFlux(), runoff = NoRunoff())
        energy = WaterCoupledEnergy(eltype(grid);
            dry_heat_capacity = Cdry,
            liquid_heat_capacity = 4186,
            reference_temperature = 273.15,
            deep_temperature = Tᵈ,
            deep_conductance = Λ,
            advect_deep_liquid_energy = false,
            advect_surface_liquid_energy = false)

        land = SlabLand(grid; hydrology, energy)
        set!(land; T = T₀, M = 0.0)
        fill!(land.fluxes.surface_energy_flux, 0)
        fill!(land.fluxes.vapor_flux, 0)
        fill!(land.fluxes.liquid_precipitation_flux, 0)
        fill!(land.fluxes.liquid_precipitation_temperature, T₀)

        t = 0.0
        for _ in 1:steps
            time_step!(land, Δt)
            t += Δt
        end

        expected_temperature = Tᵈ + (T₀ - Tᵈ) * exp(-Λ / Cdry * t)
        actual_temperature = only(Array(interior(land.temperature)))
        # Forward Euler converges to the analytic solution as Δt → 0; with
        # ΛΔt/C = 5e-4 the truncation error per step is O(Δt²) ⇒ test tolerance 1e-3 K.
        @test isapprox(actual_temperature, expected_temperature; atol = 1e-3)
    end
end

@testset "WaterCoupledEnergy advective drainage at slab T" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch;
                               size = 1,
                               x = (0, 1),
                               y = (0, 1),
                               z = (-1, 0),
                               topology = (Flat, Flat, Bounded))

        # Drain water out the bottom at slab temperature; with Λᵈ = 0 and no surface
        # fluxes, dE/dt = eˡ(T) Jˡ_b cancels cˡ(T−Tᵣ) dM/dt exactly ⇒ dT/dt = 0.
        hydrology = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0, porosity = 0.4, storage_height = 1000,
            retention_curve = VanGenuchtenRetention(α = 1.0, n = 2.0),
            hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-6, n = 2.0),
            deep_liquid_flux = LinearReservoirDrainage(drainage_time_scale = 1e6),
            runoff = NoRunoff())
        energy = WaterCoupledEnergy(eltype(grid);
            dry_heat_capacity = 1e6,
            liquid_heat_capacity = 4186,
            reference_temperature = 273.15,
            deep_temperature = 290.0,
            deep_conductance = 0.0,
            advect_deep_liquid_energy = true,
            advect_surface_liquid_energy = false)

        land = SlabLand(grid; hydrology, energy)
        set!(land; T = 290.0, M = 200.0)
        fill!(land.fluxes.surface_energy_flux, 0)
        fill!(land.fluxes.vapor_flux, 0)
        fill!(land.fluxes.liquid_precipitation_flux, 0)
        fill!(land.fluxes.liquid_precipitation_temperature, 290.0)

        for _ in 1:50
            time_step!(land, 1000.0)
        end

        actual_temperature = only(Array(interior(land.temperature)))
        actual_water_storage = only(Array(interior(land.water_storage)))
        @test isapprox(actual_temperature, 290.0; atol = 1e-10)
        @test actual_water_storage < 200  # water did leave
    end
end

@testset "WaterCoupledEnergy evaporation is reference-temperature invariant" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch; size = 1, x = (0, 1), y = (0, 1),
                               z = (-1, 0), topology = (Flat, Flat, Bounded))

        hydrology = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0, porosity = 0.4, storage_height = 1000,
            retention_curve = VanGenuchtenRetention(α = 1.0, n = 2.0),
            hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-6, n = 2.0),
            deep_liquid_flux = NoDeepLiquidFlux(), runoff = NoRunoff())

        # Identical forcing, two arbitrary internal-energy references. The physical
        # temperature evolution cannot depend on Tᵣ; it does so iff the vapor mass
        # flux is unpaired in the energy budget.
        function evaporate_with_reference(Tᵣ)
            energy = WaterCoupledEnergy(eltype(grid);
                dry_heat_capacity = 1e6, liquid_heat_capacity = 4186,
                reference_temperature = Tᵣ,
                deep_temperature = 290.0, deep_conductance = 0.0,
                advect_deep_liquid_energy = true,
                advect_surface_liquid_energy = false)
            land = SlabLand(grid; hydrology, energy)
            set!(land; T = 300.0, M = 200.0)
            fill!(land.fluxes.surface_energy_flux, 50.0)  # latent cooling, positive upward
            fill!(land.fluxes.vapor_flux, 2e-5)           # evaporation, positive upward
            fill!(land.fluxes.liquid_precipitation_flux, 0)
            for _ in 1:100
                time_step!(land, 100.0)
            end
            return only(Array(interior(land.temperature)))
        end

        @test isapprox(evaporate_with_reference(0.0),
                       evaporate_with_reference(273.15); atol = 1e-8)
    end
end

@testset "WaterCoupledEnergy rain with surface advection off is a pure mass addition" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch; size = 1, x = (0, 1), y = (0, 1),
                               z = (-1, 0), topology = (Flat, Flat, Bounded))

        hydrology = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0, porosity = 0.4, storage_height = 1000,
            retention_curve = VanGenuchtenRetention(α = 1.0, n = 2.0),
            hydraulic_conductivity = VanGenuchtenConductivity(K_saturated = 1e-6, n = 2.0),
            deep_liquid_flux = NoDeepLiquidFlux(), runoff = NoRunoff())

        # With `advect_surface_liquid_energy = false`, rain enters at the slab
        # temperature: it changes M but not T, for any internal-energy reference Tᵣ.
        function rain_with_reference(Tᵣ)
            energy = WaterCoupledEnergy(eltype(grid);
                dry_heat_capacity = 1e6, liquid_heat_capacity = 4186,
                reference_temperature = Tᵣ,
                deep_temperature = 290.0, deep_conductance = 0.0,
                advect_deep_liquid_energy = true,
                advect_surface_liquid_energy = false)
            land = SlabLand(grid; hydrology, energy)
            set!(land; T = 300.0, M = 100.0)
            fill!(land.fluxes.surface_energy_flux, 0)
            fill!(land.fluxes.vapor_flux, 0)
            fill!(land.fluxes.liquid_precipitation_flux, 1e-4)  # rain, positive downward
            fill!(land.fluxes.liquid_precipitation_temperature, 275.0)
            for _ in 1:100
                time_step!(land, 100.0)
            end
            return (only(Array(interior(land.temperature))),
                    only(Array(interior(land.water_storage))))
        end

        T₁, M₁ = rain_with_reference(0.0)
        T₂, M₂ = rain_with_reference(273.15)
        @test T₁ == T₂
        @test T₁ == 300.0
        @test M₁ == M₂
        @test isapprox(M₁, 100.0 + 1e-4 * 100 * 100.0; rtol = 1e-12)
    end
end
