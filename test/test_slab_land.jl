include("runtests_setup.jl")

using CUDA
using Oceananigans.Fields: AbstractField
using Oceananigans
using Oceananigans.Utils: launch!
using Oceananigans.TimeSteppers: update_state!
using Oceananigans: prognostic_fields, prognostic_state, restore_prognostic_state!
using NumericalEarth.Lands: update_diagnostics!, ForceRestoreEnergy, DryLand
using NumericalEarth.EarthSystemModels.InterfaceComputations: default_atmosphere_land_fluxes,
                                                              atmosphere_land_stability_functions,
                                                              atmosphere_ocean_stability_functions,
                                                              EdsonMomentumStabilityFunction,
                                                              SimilarityScales,
                                                              celsius_to_kelvin
using NumericalEarth.Atmospheres: AtmosphereThermodynamicsParameters
using Thermodynamics

@testset "SlabLand energy and hydrology" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch;
                               size = 1,
                               x = (0, 1),
                               y = (0, 1),
                               z = (-1, 0),
                               topology = (Flat, Flat, Bounded))

        energy = SlabEnergy(eltype(grid); dry_heat_capacity = 2,
                                         liquid_heat_capacity = 4)
        hydrology = BucketHydrology(eltype(grid); maximum_water_storage = 10)

        land = SlabLand(grid; energy, hydrology)

        @test land.energy.dry_heat_capacity isa Number
        @test land.energy.liquid_heat_capacity isa Number
        @test land.hydrology.maximum_water_storage isa Number

        # With `state.water_storage = 0`, the slab responds to
        # `fluxes.surface_energy_flux` (positive upward, so a negative value warms)
        # using only `dry_heat_capacity`.
        fill!(land.water_storage, 0)
        fill!(land.temperature, 10)
        fill!(land.fluxes.surface_energy_flux, -10)
        NumericalEarth.Lands.time_step!(land.energy, land, 1)
        @test isapprox(CUDA.@allowscalar(land.temperature[1, 1, 1]), 15; atol=1e-12)

        # With `state.water_storage = 5`, the liquid-water contribution reduces the
        # temperature tendency.
        fill!(land.water_storage, 5)
        fill!(land.temperature, 10)
        fill!(land.fluxes.surface_energy_flux, -10)
        NumericalEarth.Lands.time_step!(land.energy, land, 1)
        @test isapprox(CUDA.@allowscalar(land.temperature[1, 1, 1]), 10.4545454545; rtol=1e-7)

        # Hydrology clamps water_storage at maximum_water_storage; excess is shed.
        fill!(land.water_storage, 8)
        fill!(land.fluxes.precipitation, 4)
        fill!(land.fluxes.evaporation, 1)
        fill!(land.fluxes.surface_energy_flux, 0)
        time_step!(land, 1)

        # The bucket receives `3 kg m⁻²` net water over the step, capping at 10.
        @test isapprox(CUDA.@allowscalar(land.water_storage[1, 1, 1]), 10; atol=1e-12)
        @test isapprox(CUDA.@allowscalar(land.saturation[1, 1, 1]), 1; atol=1e-12)

        # Sign-convention guard: `surface_energy_flux` is positive *upward* (out of
        # the slab). Upward flux removes energy ⇒ cooling; downward (negative) ⇒ heating.
        fill!(land.water_storage, 0)

        fill!(land.temperature, 10)
        fill!(land.fluxes.surface_energy_flux, 10)   # energy leaving the slab ⇒ cools
        NumericalEarth.Lands.time_step!(land.energy, land, 1)
        @test only(Array(interior(land.temperature))) < 10

        fill!(land.temperature, 10)
        fill!(land.fluxes.surface_energy_flux, -10)  # energy entering the slab ⇒ warms
        NumericalEarth.Lands.time_step!(land.energy, land, 1)
        @test only(Array(interior(land.temperature))) > 10

        # The atmosphere-facing land state exposes skin temperature and saturation;
        # roughness lengths belong to the flux closure, not the land.
        ex = NumericalEarth.EarthSystemModels.InterfaceComputations.ComponentExchanger(land, land.grid)
        @test hasproperty(ex.state, :T)
        @test hasproperty(ex.state, :saturation)
        @test !hasproperty(ex.state, :momentum_roughness_length)
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

        fill!(Cdry, 8)
        fill!(Cl, 2)
        fill!(Wmax, 12)

        energy = SlabEnergy(eltype(grid);
                            dry_heat_capacity = Cdry,
                            liquid_heat_capacity = Cl)
        hydrology = BucketHydrology(eltype(grid); maximum_water_storage = Wmax)

        land = SlabLand(grid; energy, hydrology)

        @test land.energy.dry_heat_capacity isa AbstractField
        @test land.energy.liquid_heat_capacity isa AbstractField
        @test land.hydrology.maximum_water_storage isa AbstractField

        fill!(land.water_storage, 4)
        fill!(land.temperature, 10)
        fill!(land.fluxes.surface_energy_flux, -10)
        NumericalEarth.Lands.time_step!(land.energy, land, 1)

        # The field-valued land properties are read pointwise from the grid.
        @test isapprox(CUDA.@allowscalar(land.temperature[1, 1, 1]), 10.625; atol=1e-12)

        fill!(land.water_storage, 10)
        fill!(land.fluxes.precipitation, 4)
        fill!(land.fluxes.evaporation, 0)
        time_step!(land, 1)

        @test isapprox(CUDA.@allowscalar(land.water_storage[1, 1, 1]), 12; atol=1e-12)
    end
end

@testset "BucketHydrology continuous saturation" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch;
                               size = 1,
                               x = (0, 1),
                               y = (0, 1),
                               z = (-1, 0),
                               topology = (Flat, Flat, Bounded))

        Wmax = CenterField(grid)
        fill!(Wmax, 20)

        energy = SlabEnergy(eltype(grid); dry_heat_capacity = 1000, liquid_heat_capacity = 4000)
        hydrology = BucketHydrology(eltype(grid); maximum_water_storage = Wmax)

        land = SlabLand(grid; energy, hydrology)

        # Saturation is continuous: 𝒮 = M / Mˡᵃ⁺, with Mˡᵃ⁺ = Wmax = 20.
        fill!(land.water_storage, 5)
        fill!(land.saturation, 0)
        update_diagnostics!(land.hydrology, land)
        @test isapprox(CUDA.@allowscalar(land.saturation[1, 1, 1]), 0.25; atol=1e-12)

        # No water ⇒ 0.
        fill!(land.water_storage, 0)
        update_diagnostics!(land.hydrology, land)
        @test isapprox(CUDA.@allowscalar(land.saturation[1, 1, 1]), 0; atol=1e-12)

        # The bucket caps at Mˡᵃ⁺ = 20: excess water is shed and saturation tops
        # out at 1 when the bucket is full.
        fill!(land.water_storage, 18)
        fill!(land.fluxes.precipitation, 7)
        fill!(land.fluxes.evaporation, 0)
        time_step!(land, 1)
        @test isapprox(CUDA.@allowscalar(land.water_storage[1, 1, 1]), 20; atol=1e-12)
        @test isapprox(CUDA.@allowscalar(land.saturation[1, 1, 1]), 1; atol=1e-12)
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
        # from 280 to Tᵈᵉᵉᵖ = 260, plus −Jᴱs/C = −(−10)/2 = 5 K of flux heating.
        energy = ForceRestoreEnergy(eltype(grid);
                                   dry_heat_capacity = 2,
                                   liquid_heat_capacity = 4,
                                   deep_temperature = 260,
                                   deep_time_scale = 1)
        land = SlabLand(grid; energy, hydrology = DryLand())

        fill!(land.temperature, 280)
        fill!(land.fluxes.surface_energy_flux, -10)

        NumericalEarth.Lands.time_step!(land.energy, land, 1, 0)
        # T_new = 280 + (−(−10)/2 + (260 − 280)/1)·1 = 280 + 5 − 20 = 265.
        @test isapprox(CUDA.@allowscalar(land.temperature[1, 1, 1]), 265; atol=1e-12)

        # Time-dependent prescribed deep temperature (stateindex path).
        deep_by_time = (λ, φ, z, t) -> t > 0 ? 265 : 260
        energy_time = ForceRestoreEnergy(eltype(grid);
                                        dry_heat_capacity = 2,
                                        liquid_heat_capacity = 4,
                                        deep_temperature = deep_by_time,
                                        deep_time_scale = 1)
        land_time = SlabLand(grid; energy = energy_time, hydrology = DryLand())

        fill!(land_time.temperature, 280)
        fill!(land_time.fluxes.surface_energy_flux, 0)
        time_step!(land_time, 1)
        # Post-step time is 1 s ⇒ Tᵈᵉᵉᵖ = 265; T relaxes fully to it in one step.
        @test isapprox(CUDA.@allowscalar(land_time.temperature[1, 1, 1]), 265; atol=1e-12)
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

        Es = CenterField(grid)
        P = CenterField(grid)
        E = CenterField(grid)
        Jʳⁿ = CenterField(grid)  # atmospheric rainfall flux on exchange grid
        interface_fluxes = (sensible_heat = CenterField(grid),
                           latent_heat   = CenterField(grid),
                           water_vapor  = CenterField(grid))

        # Evaporation in model convention is positive upward => water-vapor flux > 0
        fill!(parent(interface_fluxes.sensible_heat), 0)
        fill!(parent(interface_fluxes.latent_heat),   0)
        fill!(parent(interface_fluxes.water_vapor),   0.5)
        fill!(parent(Jʳⁿ), 0)
        fill!(parent(Es), 0)
        fill!(parent(P), 0)
        fill!(parent(E), 0)

        launch!(arch, grid, :xy,
                NumericalEarth.Lands._assemble_slab_land_fluxes!,
                P, E, nothing, nothing, nothing, nothing, interface_fluxes, Jʳⁿ, nothing, nothing)

        @test isapprox(CUDA.@allowscalar(P[1, 1, 1]), 0; atol=1e-12)
        @test isapprox(CUDA.@allowscalar(E[1, 1, 1]), 0.5; atol=1e-12)

        # Condensation (dew) is captured by negative atmospheric mass flux.
        fill!(parent(interface_fluxes.water_vapor), -0.5)
        fill!(parent(P), 0)
        fill!(parent(E), 0)

        launch!(arch, grid, :xy,
                NumericalEarth.Lands._assemble_slab_land_fluxes!,
                P, E, nothing, nothing, nothing, nothing, interface_fluxes, Jʳⁿ, nothing, nothing)

        @test isapprox(CUDA.@allowscalar(P[1, 1, 1]), 0.5; atol=1e-12)
        @test isapprox(CUDA.@allowscalar(E[1, 1, 1]), 0; atol=1e-12)

        # Prescribed atmospheric rainfall is accumulated into precipitation
        # alongside any condensation.
        fill!(parent(interface_fluxes.water_vapor), -0.2)
        fill!(parent(Jʳⁿ), 1.5)
        fill!(parent(P), 0)
        fill!(parent(E), 0)

        launch!(arch, grid, :xy,
                NumericalEarth.Lands._assemble_slab_land_fluxes!,
                P, E, nothing, nothing, nothing, nothing, interface_fluxes, Jʳⁿ, nothing, nothing)

        @test isapprox(CUDA.@allowscalar(P[1, 1, 1]), 1.7; atol=1e-12)
        @test isapprox(CUDA.@allowscalar(E[1, 1, 1]), 0; atol=1e-12)

        # `surface_energy_flux` adds turbulent sensible + latent, positive upward
        # (out of the slab): Es = 𝒬ᵀ + 𝒬ᵛ = 2 + (−5) = −3.
        fill!(parent(interface_fluxes.sensible_heat),  2)
        fill!(parent(interface_fluxes.latent_heat),   -5)
        fill!(parent(interface_fluxes.water_vapor),    0)
        fill!(parent(Jʳⁿ), 0)
        fill!(parent(Es), 0)

        launch!(arch, grid, :xy,
                NumericalEarth.Lands._assemble_slab_land_fluxes!,
                P, E, nothing, Es, nothing, nothing, interface_fluxes, Jʳⁿ, nothing, nothing)

        @test isapprox(only(Array(interior(Es))), -3; atol=1e-12)
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
                               energy = SlabEnergy(eltype(grid); dry_heat_capacity = 10_000))

        radiation_none = PrescribedRadiation(grid; ocean_surface = SurfaceRadiationProperties(0, 1),
                                            sea_ice_surface = SurfaceRadiationProperties(0, 1))
        radiation_land = PrescribedRadiation(grid; land_surface = SurfaceRadiationProperties(0, 1),
                                            ocean_surface = SurfaceRadiationProperties(0, 1),
                                            sea_ice_surface = SurfaceRadiationProperties(0, 1))

        model_no_land = AtmosphereLandModel(make_atmosphere(), make_land(); radiation = radiation_none)
        model_with_land = AtmosphereLandModel(make_atmosphere(), make_land(); radiation = radiation_land)

        for m in (model_no_land, model_with_land)
            fill!(parent(m.atmosphere.velocities.u), 1)
            fill!(parent(m.atmosphere.velocities.v), 0)
            fill!(parent(m.atmosphere.temperature), 300)
            fill!(parent(m.atmosphere.specific_humidity), 0.005)
            fill!(parent(m.atmosphere.pressure), 101_325)
            fill!(m.land.temperature, 300)
            fill!(parent(m.land.fluxes.surface_energy_flux), 0)
            fill!(m.land.water_storage, 0)
        end

        update_state!(model_no_land)
        update_state!(model_with_land)

        # Land radiative fluxes are only populated when land surface properties are provided.
        @test all(parent(model_no_land.radiation.interface_fluxes.land.upwelling_longwave) .== 0)
        @test all(parent(model_with_land.radiation.interface_fluxes.land.upwelling_longwave) .> 0)

        # `surface_energy_flux` is positive upward, so the emitted longwave radiative
        # cooling term *raises* it (more energy leaving the slab) when land properties
        # are available.
        surface_energy_without_properties = only(Array(interior(model_no_land.land.fluxes.surface_energy_flux)))
        surface_energy_with_properties = only(Array(interior(model_with_land.land.fluxes.surface_energy_flux)))
        @test surface_energy_with_properties > surface_energy_without_properties
    end
end

@testset "Land default stability functions (footgun guard)" begin
    FT = Float64

    land_ψ  = atmosphere_land_stability_functions(FT)
    ocean_ψ = atmosphere_ocean_stability_functions(FT)

    # The named land seam must differ from the Edson ocean functions.
    @test !(land_ψ.momentum isa EdsonMomentumStabilityFunction)
    @test typeof(land_ψ.momentum) != typeof(ocean_ψ.momentum)

    # `default_atmosphere_land_fluxes` must wire the land functions explicitly
    # rather than silently inheriting the `SimilarityTheoryFluxes` ocean default.
    grid = RectilinearGrid(size = 1, x = (0, 1), y = (0, 1), z = (-1, 0),
                           topology = (Flat, Flat, Bounded))
    land = SlabLand(grid)
    fluxes = default_atmosphere_land_fluxes(land, FT)
    @test typeof(fluxes.stability_functions.momentum) == typeof(land_ψ.momentum)
    @test !(fluxes.stability_functions.momentum isa EdsonMomentumStabilityFunction)
end

@testset "SlabLand prognostic_fields and checkpointing" begin
    for arch in test_architectures
        A = typeof(arch)
        @info "Testing SlabLand prognostic_fields + checkpointing on $A"

        grid = RectilinearGrid(arch; size = 1, x = (0, 1), y = (0, 1), z = (-1, 0),
                               topology = (Flat, Flat, Bounded))
        land = SlabLand(grid)
        set!(land; T = 300.0, M = 150.0)

        # `prognostic_fields` is the math-named NamedTuple of the prognostics
        # (saturation is diagnostic), aliasing the underlying fields.
        pf = prognostic_fields(land)
        @test keys(pf) == (:T, :M)
        @test pf.T === land.temperature
        @test pf.M === land.water_storage

        # Checkpoint round-trip. `deepcopy` decouples the snapshot from the live
        # fields the way on-disk serialization does in a real checkpoint.
        snapshot = deepcopy(prognostic_state(land))
        @test :temperature in keys(snapshot)
        @test :water_storage in keys(snapshot)

        set!(land; T = 250.0, M = 50.0)
        restore_prognostic_state!(land, snapshot)
        @test Array(interior(land.temperature))[1]   == 300
        @test Array(interior(land.water_storage))[1] == 150

        # Restoring from `nothing` is a no-op that returns the land.
        @test restore_prognostic_state!(land, nothing) === land
    end
end

@testset "Atmosphere-Land turbulent fluxes (analytic neutral)" begin
    for arch in test_architectures
        grid = LatitudeLongitudeGrid(arch, Float64;
                                     size = 1, latitude = 10, longitude = 10,
                                     z = (-1, 0), topology = (Flat, Flat, Bounded))

        h   = 10.0
        uᵃᵗ = 5.0
        vᵃᵗ = 0.0
        Tᵃᵗ = 288.0
        qᵃᵗ = 0.003
        pᵃᵗ = 101325.0

        atmosphere = PrescribedAtmosphere(grid; surface_layer_height = h, boundary_layer_height = 512)
        @allowscalar begin
            fill!(parent(atmosphere.temperature),       Tᵃᵗ)
            fill!(parent(atmosphere.specific_humidity), qᵃᵗ)
            fill!(parent(atmosphere.velocities.u), uᵃᵗ)
            fill!(parent(atmosphere.velocities.v), vᵃᵗ)
            fill!(parent(atmosphere.pressure),     pᵃᵗ)
        end

        ℂᵃᵗ = atmosphere.thermodynamics_parameters
        cᵖᵐ = Thermodynamics.cp_m(ℂᵃᵗ, qᵃᵗ)
        ρᵃᵗ = Thermodynamics.air_density(ℂᵃᵗ, Tᵃᵗ, pᵃᵗ, qᵃᵗ)

        # Neutralize stability and gustiness with a constant roughness length so the
        # land fluxes reduce to the analytic log-law form (cf. the atmosphere-ocean test).
        ℓ = 1e-4
        zero_ψ(ζ) = zero(ζ)
        stability_functions = SimilarityScales(zero_ψ, zero_ψ, zero_ψ)
        fluxes = SimilarityTheoryFluxes(; momentum_roughness_length    = ℓ,
                                          temperature_roughness_length = ℓ,
                                          water_vapor_roughness_length = ℓ,
                                          subgrid_velocities = nothing,
                                          stability_functions)

        land = SlabLand(grid; hydrology = DryLand(), energy = SlabEnergy(eltype(grid)))
        T_skin = Tᵃᵗ + 2
        set!(land; T = T_skin)

        model = AtmosphereLandModel(atmosphere, land; atmosphere_land_fluxes = fluxes, radiation = nothing)
        update_state!(model)

        g = model.interfaces.properties.gravitational_acceleration
        interface_fluxes = model.interfaces.atmosphere_land_interface.fluxes

        # Surface velocities are zero for land, so ΔU is the full near-surface wind.
        ϰ  = fluxes.von_karman_constant
        ΔU = sqrt(uᵃᵗ^2 + vᵃᵗ^2)
        Δθ = Tᵃᵗ - T_skin + h / cᵖᵐ * g
        u★ = ϰ / log(h / ℓ) * ΔU
        θ★ = ϰ / log(h / ℓ) * Δθ
        τˣ = - ρᵃᵗ * u★^2 * uᵃᵗ / ΔU
        𝒬ᵀ = - ρᵃᵗ * cᵖᵐ * u★ * θ★

        @test @allowscalar(interface_fluxes.x_momentum[1, 1, 1])        ≈ τˣ
        @test @allowscalar(interface_fluxes.friction_velocity[1, 1, 1]) ≈ u★
        @test @allowscalar(interface_fluxes.sensible_heat[1, 1, 1])     ≈ 𝒬ᵀ
        @test @allowscalar(abs(interface_fluxes.y_momentum[1, 1, 1]))   < eps(Float64)
    end
end

@testset "Atmosphere-Land flux stability and roughness response" begin
    for arch in test_architectures
        Tᵃᵗ = 288
        qᵃᵗ = 0.003
        cᵖᵐ = Thermodynamics.cp_m(AtmosphereThermodynamicsParameters(Float64), qᵃᵗ)
        g   = 9.80665
        neutral_skin = Tᵃᵗ + 10 / cᵖᵐ * g
        # Single-column coupled model using the real land stability functions; returns
        # the friction velocity and sensible heat for a given skin temperature / roughness.
        function land_flux_response(T_skin; ℓ = 0.1)
            grid = LatitudeLongitudeGrid(arch, Float64;
                                         size = 1, latitude = 10, longitude = 10,
                                         z = (-1, 0), topology = (Flat, Flat, Bounded))
            h = 10.0
            atmosphere = PrescribedAtmosphere(grid; surface_layer_height = h, boundary_layer_height = 512)
            @allowscalar begin
                fill!(parent(atmosphere.temperature),       Tᵃᵗ)
                fill!(parent(atmosphere.specific_humidity), qᵃᵗ)
                fill!(parent(atmosphere.velocities.u), 5)
                fill!(parent(atmosphere.velocities.v), 0)
                fill!(parent(atmosphere.pressure),     101325)
            end
            fluxes = SimilarityTheoryFluxes(; momentum_roughness_length    = ℓ,
                                              temperature_roughness_length = 0.01,
                                              water_vapor_roughness_length = 0.01,
                                              stability_functions = atmosphere_land_stability_functions(Float64))
            land = SlabLand(grid; hydrology = DryLand(), energy = SlabEnergy(eltype(grid)))
            set!(land; T = T_skin)
            model = AtmosphereLandModel(atmosphere, land; atmosphere_land_fluxes = fluxes, radiation = nothing)
            update_state!(model)
            f = model.interfaces.atmosphere_land_interface.fluxes
            return (u★ = @allowscalar(f.friction_velocity[1, 1, 1]),
                    Q  = @allowscalar(f.sensible_heat[1, 1, 1]))
        end

        warm    = land_flux_response(neutral_skin + 6)
        neutral = land_flux_response(neutral_skin)
        cold    = land_flux_response(neutral_skin - 6)

        # Sensible-heat sign tracks the surface-air temperature gradient.
        @test warm.Q > 0
        @test abs(neutral.Q) < 1e-6
        @test cold.Q < 0

        # Friction velocity responds to stability: unstable enhances drag, stable suppresses it.
        @test warm.u★ > neutral.u★ > cold.u★

        # Larger roughness length increases the surface drag for a fixed wind.
        @test land_flux_response(neutral_skin; ℓ = 0.5).u★ > land_flux_response(neutral_skin; ℓ = 0.05).u★
    end
end
