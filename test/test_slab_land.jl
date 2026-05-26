include("runtests_setup.jl")

using CUDA
using Oceananigans.Fields: AbstractField
using Oceananigans
using Oceananigans.Utils: launch!
using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.Lands: update_diagnostics!

@testset "SlabLand energy and hydrology" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch;
                               size = 1,
                               x = (0, 1),
                               y = (0, 1),
                               z = (-1, 0),
                               topology = (Flat, Flat, Bounded))

        energy = SlabEnergy(eltype(grid); dry_heat_capacity = 2.0,
                                         liquid_heat_capacity = 4.0)
        hydrology = BucketHydrology(eltype(grid);
                                   maximum_water_storage = 10.0,
                                   critical_wetness_ratio = 0.5)
        surface = ConstantSurfaceProperties(eltype(grid);
                                           momentum_roughness_length = 0.1,
                                           scalar_roughness_length = 0.01)

        land = SlabLand(grid; energy, hydrology, surface)

        @test land.energy.dry_heat_capacity isa Number
        @test land.energy.liquid_heat_capacity isa Number
        @test land.hydrology.maximum_water_storage isa Number
        @test land.hydrology.critical_wetness_ratio isa Number
        @test land.surface.momentum_roughness_length isa Number
        @test land.surface.scalar_roughness_length isa Number

        # With `state.water_storage = 0`, the slab responds to `fluxes.net_energy_flux`
        # using only `dry_heat_capacity`.
        fill!(land.water_storage, 0)
        fill!(land.temperature, 10.0)
        fill!(land.fluxes.net_energy_flux, 10.0)
        NumericalEarth.Lands.step!(land.energy, land, 1.0)
        @test isapprox(CUDA.@allowscalar(land.temperature[1, 1, 1]), 15.0; atol=1e-12)

        # With `state.water_storage = 5`, the liquid-water contribution reduces the
        # temperature tendency.
        fill!(land.water_storage, 5.0)
        fill!(land.temperature, 10.0)
        fill!(land.fluxes.net_energy_flux, 10.0)
        NumericalEarth.Lands.step!(land.energy, land, 1.0)
        @test isapprox(CUDA.@allowscalar(land.temperature[1, 1, 1]), 10.4545454545; rtol=1e-7)

        # Hydrology clamps water_storage at maximum_water_storage; excess is shed.
        fill!(land.water_storage, 8.0)
        fill!(land.fluxes.precipitation, 4.0)
        fill!(land.fluxes.evaporation, 1.0)
        fill!(land.fluxes.net_energy_flux, 0.0)
        time_step!(land, 1.0)

        # The bucket receives `3 kg m⁻²` net water over the step, capping at 10.
        @test isapprox(CUDA.@allowscalar(land.water_storage[1, 1, 1]), 10.0; atol=1e-12)
        @test isapprox(CUDA.@allowscalar(land.moisture_availability[1, 1, 1]), 1.0; atol=1e-12)

        # Exposer for atmosphere-facing roughness fields.
        ex = NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(land, land.grid)
        @test hasproperty(ex.state, :momentum_roughness_length)
        @test hasproperty(ex.state, :scalar_roughness_length)
    end
end

@testset "SlabLand property providers" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch;
                               size = 1,
                               x = (0, 1),
                               y = (0, 1),
                               z = (-1, 0),
                               topology = (Flat, Flat, Bounded))

        Cdry = CenterField(grid)
        Cl   = CenterField(grid)
        Wmax = CenterField(grid)
        Wcrit = CenterField(grid)
        ℓᵐ = CenterField(grid)
        ℓˢ = CenterField(grid)

        fill!(Cdry, 8.0)
        fill!(Cl, 2.0)
        fill!(Wmax, 12.0)
        fill!(Wcrit, 0.5)
        fill!(ℓᵐ, 0.2)
        fill!(ℓˢ, 0.02)

        energy = SlabEnergy(eltype(grid);
                            dry_heat_capacity = Cdry,
                            liquid_heat_capacity = Cl)
        hydrology = BucketHydrology(eltype(grid);
                                    maximum_water_storage = Wmax,
                                    critical_wetness_ratio = Wcrit)
        surface = ConstantSurfaceProperties(eltype(grid);
                                            momentum_roughness_length = ℓᵐ,
                                            scalar_roughness_length = ℓˢ)

        land = SlabLand(grid; energy, hydrology, surface)

        @test land.energy.dry_heat_capacity isa AbstractField
        @test land.energy.liquid_heat_capacity isa AbstractField
        @test land.hydrology.maximum_water_storage isa AbstractField
        @test land.hydrology.critical_wetness_ratio isa AbstractField
        @test land.surface.momentum_roughness_length isa AbstractField
        @test land.surface.scalar_roughness_length isa AbstractField

        fill!(land.water_storage, 4.0)
        fill!(land.temperature, 10.0)
        fill!(land.fluxes.net_energy_flux, 10.0)
        NumericalEarth.Lands.step!(land.energy, land, 1.0)

        # The field-valued land properties are read pointwise from the grid.
        @test isapprox(CUDA.@allowscalar(land.temperature[1, 1, 1]), 10.625; atol=1e-12)

        fill!(land.water_storage, 10.0)
        fill!(land.fluxes.precipitation, 4.0)
        fill!(land.fluxes.evaporation, 0.0)
        time_step!(land, 1.0)

        @test isapprox(CUDA.@allowscalar(land.water_storage[1, 1, 1]), 12.0; atol=1e-12)

        ex = NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(land, land.grid)
        @test isapprox(CUDA.@allowscalar(ex.state.momentum_roughness_length[1, 1, 1]), 0.2; atol=1e-12)
        @test isapprox(CUDA.@allowscalar(ex.state.scalar_roughness_length[1, 1, 1]), 0.02; atol=1e-12)
    end
end

@testset "BucketHydrology root depth and LAI modifiers" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch;
                               size = 1,
                               x = (0, 1),
                               y = (0, 1),
                               z = (-1, 0),
                               topology = (Flat, Flat, Bounded))

        Wmax = CenterField(grid)
        Wcrit = CenterField(grid)
        zʳ = CenterField(grid)
        Λ = CenterField(grid)

        fill!(Wmax, 10.0)
        fill!(Wcrit, 0.5)
        fill!(zʳ, 2.0)
        fill!(Λ, 1.0)

        energy = SlabEnergy(eltype(grid); dry_heat_capacity = 1000.0, liquid_heat_capacity = 4000.0)
        hydrology = BucketHydrology(eltype(grid);
                                    maximum_water_storage = Wmax,
                                    critical_wetness_ratio = Wcrit,
                                    root_depth = zʳ,
                                    leaf_area_index = Λ,
                                    lai_stress = 0.2)
        surface = ConstantSurfaceProperties(eltype(grid);
                                            momentum_roughness_length = 0.1,
                                            scalar_roughness_length = 0.01)

        land = SlabLand(grid; energy, hydrology, surface)

        fill!(land.water_storage, 5.0)
        fill!(land.moisture_availability, 0.0)
        update_diagnostics!(land.hydrology, land)

        # Effective capacity is Wmax * zʳ = 20 kg m⁻², so β_before = 5/(0.5*20)=0.5.
        # LAI stress with kᴸ=0.2 and Λ=1.0 gives β = 0.5 * exp(-0.2).
        expected_β = 0.5 * exp(-0.2)
        @test isapprox(CUDA.@allowscalar(land.moisture_availability[1, 1, 1]), expected_β; atol=1e-12)

        fill!(land.water_storage, 18.0)
        fill!(land.fluxes.precipitation, 7.0)
        fill!(land.fluxes.evaporation, 0.0)
        time_step!(land, 1.0)

        # Effective capacity is Wmax * zʳ = 20 kg m⁻²; water_storage caps there.
        @test isapprox(CUDA.@allowscalar(land.water_storage[1, 1, 1]), 20.0; atol=1e-12)
        # At saturation β_raw = 1; the canopy LAI stress (kᴸ Λ = 0.2)
        # still multiplies through, giving β = exp(-0.2).
        @test isapprox(CUDA.@allowscalar(land.moisture_availability[1, 1, 1]), exp(-0.2); atol=1e-12)
    end
end

@testset "Force restore energy closure" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch;
                               size = 1,
                               x = (0, 1),
                               y = (0, 1),
                               z = (-1, 0),
                               topology = (Flat, Flat, Bounded))

        # Surface flux + relaxation toward the prescribed deep temperature.
        # With τ = 1 s and Δt = 1 s, the restoring term moves T all the way
        # from 280 to Tᵈᵉᵉᵖ = 260, plus Q/C = 10/2 = 5 K of flux heating.
        energy = ForceRestoreEnergy(eltype(grid);
                                   dry_heat_capacity = 2.0,
                                   liquid_heat_capacity = 4.0,
                                   deep_temperature = 260.0,
                                   deep_time_scale = 1.0)
        surface = ConstantSurfaceProperties(eltype(grid);
                                           momentum_roughness_length = 0.1,
                                           scalar_roughness_length = 0.01)
        land = SlabLand(grid; energy, hydrology = DryLand(), surface)

        fill!(land.temperature, 280.0)
        fill!(land.fluxes.net_energy_flux, 10.0)

        NumericalEarth.Lands.step!(land.energy, land, 1.0, 0.0)
        # T_new = 280 + (10/2 + (260 − 280)/1)·1 = 280 + 5 − 20 = 265.
        @test isapprox(CUDA.@allowscalar(land.temperature[1, 1, 1]), 265.0; atol=1e-12)

        # Time-dependent prescribed deep temperature (stateindex path).
        deep_by_time = (λ, φ, z, t) -> t > 0 ? 265.0 : 260.0
        energy_time = ForceRestoreEnergy(eltype(grid);
                                        dry_heat_capacity = 2.0,
                                        liquid_heat_capacity = 4.0,
                                        deep_temperature = deep_by_time,
                                        deep_time_scale = 1.0)
        land_time = SlabLand(grid; energy = energy_time, hydrology = DryLand(), surface)

        fill!(land_time.temperature, 280.0)
        fill!(land_time.fluxes.net_energy_flux, 0.0)
        time_step!(land_time, 1.0)
        # Post-step time is 1 s ⇒ Tᵈᵉᵉᵖ = 265; T relaxes fully to it in one step.
        @test isapprox(CUDA.@allowscalar(land_time.temperature[1, 1, 1]), 265.0; atol=1e-12)
    end
end

@testset "SlabLand flux assembly sign conventions" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch;
                               size = 1,
                               x = (0, 1),
                               y = (0, 1),
                               z = (-1, 0),
                               topology = (Flat, Flat, Bounded))

        Q = CenterField(grid)
        P = CenterField(grid)
        E = CenterField(grid)
        Jʳⁿ = CenterField(grid)  # atmospheric rainfall flux on exchange grid
        interface_fluxes = (sensible_heat = CenterField(grid),
                           latent_heat   = CenterField(grid),
                           water_vapor  = CenterField(grid))

        # Evaporation in model convention is positive upward => water-vapor flux > 0
        fill!(parent(interface_fluxes.sensible_heat), 0.0)
        fill!(parent(interface_fluxes.latent_heat),   0.0)
        fill!(parent(interface_fluxes.water_vapor),   0.5)
        fill!(parent(Jʳⁿ), 0.0)
        fill!(parent(Q), 0.0)
        fill!(parent(P), 0.0)
        fill!(parent(E), 0.0)

        launch!(arch, grid, :xy,
                NumericalEarth.Lands._assemble_slab_land_fluxes!,
                Q, P, E, interface_fluxes, Jʳⁿ)

        @test isapprox(CUDA.@allowscalar(P[1, 1, 1]), 0.0; atol=1e-12)
        @test isapprox(CUDA.@allowscalar(E[1, 1, 1]), 0.5; atol=1e-12)

        # Condensation (dew) is captured by negative atmospheric mass flux.
        fill!(parent(interface_fluxes.water_vapor), -0.5)
        fill!(parent(P), 0.0)
        fill!(parent(E), 0.0)

        launch!(arch, grid, :xy,
                NumericalEarth.Lands._assemble_slab_land_fluxes!,
                Q, P, E, interface_fluxes, Jʳⁿ)

        @test isapprox(CUDA.@allowscalar(P[1, 1, 1]), 0.5; atol=1e-12)
        @test isapprox(CUDA.@allowscalar(E[1, 1, 1]), 0.0; atol=1e-12)

        # Prescribed atmospheric rainfall is accumulated into precipitation
        # alongside any condensation.
        fill!(parent(interface_fluxes.water_vapor), -0.2)
        fill!(parent(Jʳⁿ), 1.5)
        fill!(parent(P), 0.0)
        fill!(parent(E), 0.0)

        launch!(arch, grid, :xy,
                NumericalEarth.Lands._assemble_slab_land_fluxes!,
                Q, P, E, interface_fluxes, Jʳⁿ)

        @test isapprox(CUDA.@allowscalar(P[1, 1, 1]), 1.7; atol=1e-12)
        @test isapprox(CUDA.@allowscalar(E[1, 1, 1]), 0.0; atol=1e-12)

        # Net energy adds turbulent sensible + latent, positive into the land slab.
        fill!(parent(interface_fluxes.sensible_heat),  2.0)
        fill!(parent(interface_fluxes.latent_heat),   -5.0)
        fill!(parent(interface_fluxes.water_vapor),    0.0)
        fill!(parent(Jʳⁿ), 0.0)
        fill!(parent(Q), 0.0)

        launch!(arch, grid, :xy,
                NumericalEarth.Lands._assemble_slab_land_fluxes!,
                Q, P, E, interface_fluxes, Jʳⁿ)

        @test isapprox(CUDA.@allowscalar(Q[1, 1, 1]), 3.0; atol=1e-12)
    end
end

@testset "Atmosphere-Land radiation coupling" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch;
                                    size = 1,
                                    latitude = 10,
                                    longitude = 10,
                                    z = (-1, 0),
                                    topology = (Flat, Flat, Bounded))

        function make_atmosphere()
            return PrescribedAtmosphere(grid; surface_layer_height = 10,
                                             boundary_layer_height = 512)
        end

        make_land() = SlabLand(grid;
                               hydrology = DryLand(),
                               energy = SlabEnergy(eltype(grid); dry_heat_capacity = 10_000.0),
                               surface = ConstantSurfaceProperties(eltype(grid);
                                                                  momentum_roughness_length = 0.1,
                                                                  scalar_roughness_length = 0.01))

        radiation_none = PrescribedRadiation(grid; ocean_surface = SurfaceRadiationProperties(0.0, 1.0),
                                            sea_ice_surface = SurfaceRadiationProperties(0.0, 1.0))
        radiation_land = PrescribedRadiation(grid; land_surface = SurfaceRadiationProperties(0.0, 1.0),
                                            ocean_surface = SurfaceRadiationProperties(0.0, 1.0),
                                            sea_ice_surface = SurfaceRadiationProperties(0.0, 1.0))

        model_no_land = AtmosphereLandModel(make_atmosphere(), make_land(); radiation = radiation_none)
        model_with_land = AtmosphereLandModel(make_atmosphere(), make_land(); radiation = radiation_land)

        for m in (model_no_land, model_with_land)
            fill!(parent(m.atmosphere.velocities.u), 1.0)
            fill!(parent(m.atmosphere.velocities.v), 0.0)
            fill!(parent(m.atmosphere.tracers.T), 300.0)
            fill!(parent(m.atmosphere.tracers.q), 0.005)
            fill!(parent(m.atmosphere.pressure), 101_325.0)
            fill!(m.land.temperature, 300.0)
            fill!(parent(m.land.fluxes.net_energy_flux), 0.0)
            fill!(m.land.water_storage, 0)
        end

        update_state!(model_no_land)
        update_state!(model_with_land)

        # Land radiative fluxes are only populated when land surface properties are provided.
        @test all(parent(model_no_land.radiation.interface_fluxes.land.upwelling_longwave) .== 0)
        @test all(parent(model_with_land.radiation.interface_fluxes.land.upwelling_longwave) .> 0)

        # Net land heating includes the emitted longwave radiative cooling term when land
        # properties are available.
        Q_land_no_props = CUDA.@allowscalar(model_no_land.land.fluxes.net_energy_flux[1, 1, 1])
        Q_land_with_props = CUDA.@allowscalar(model_with_land.land.fluxes.net_energy_flux[1, 1, 1])
        @test Q_land_with_props < Q_land_no_props
    end
end
