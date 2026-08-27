include("runtests_setup.jl")

using Breeze
using Breeze.Microphysics: DCMIP2016KesslerMicrophysics
using Breeze.Thermodynamics: TetensFormula
using Breeze.TerrainFollowingDiscretization: TerrainFollowingVerticalDiscretization, LinearDecay, materialize_terrain!
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: time_step!, update_state!
using Oceananigans.Utils: launch!
using NumericalEarth.Radiations: default_stefan_boltzmann_constant
using Test

NumericalEarthBreezeExt = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
@test !isnothing(NumericalEarthBreezeExt)

# Helper to build a fresh atmosphere + slab ocean model
function build_test_model(arch)
    grid = RectilinearGrid(arch,
                           size = (16, 16), halo = (5, 5),
                           x = (-10kilometers, 10kilometers),
                           z = (0, 10kilometers),
                           topology = (Periodic, Flat, Bounded))

    θ₀ = 285

    atmosphere = atmosphere_simulation(grid; potential_temperature=θ₀)
    set!(atmosphere.model, θ=atmosphere.model.dynamics.reference_state.surface_potential_temperature, u=1)

    sst_grid = RectilinearGrid(arch,
                               size = grid.Nx,
                               halo = grid.Hx,
                               x = (-10kilometers, 10kilometers),
                               topology = (Periodic, Flat, Flat))

    ocean = SlabOcean(sst_grid, depth=50, density=1025, heat_capacity=4000)
    set!(ocean, T=θ₀)

    model = AtmosphereOceanModel(atmosphere, ocean)

    return model
end

# Helper to build a fresh atmosphere + slab land model with a warm surface and a
# nonzero near-surface wind, so the MOST coupling produces a nonzero surface stress.
function build_land_test_model(arch)
    grid = RectilinearGrid(arch,
                           size = (16, 16), halo = (5, 5),
                           x = (-10kilometers, 10kilometers),
                           z = (0, 10kilometers),
                           topology = (Periodic, Flat, Bounded))

    θ₀ = 285

    atmosphere = atmosphere_simulation(grid; potential_temperature=θ₀)
    set!(atmosphere.model, θ=atmosphere.model.dynamics.reference_state.surface_potential_temperature, u=5)

    land_grid = RectilinearGrid(arch,
                                size = grid.Nx,
                                halo = grid.Hx,
                                x = (-10kilometers, 10kilometers),
                                topology = (Periodic, Flat, Flat))

    land = SlabLand(land_grid)
    set!(land; T=θ₀ + 5)

    model = AtmosphereLandModel(atmosphere, land)

    return model
end

# The land setup with the child wrapped in a NestedModel (inert prescribed parent) inside
# a Simulation — the shape a parent-driven LAM presents to AtmosphereLandModel.
function build_nested_land_test_model(arch;
                                      microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()),
                                      thermodynamic_constants = ThermodynamicConstants())
    grid = RectilinearGrid(arch,
                           size = (16, 16), halo = (5, 5),
                           x = (-10kilometers, 10kilometers),
                           z = (0, 10kilometers),
                           topology = (Periodic, Flat, Bounded))

    θ₀ = 285

    child = atmosphere_model(grid; potential_temperature=θ₀, microphysics, thermodynamic_constants)
    set!(child, θ=child.dynamics.reference_state.surface_potential_temperature, u=5)

    parent_grid = RectilinearGrid(arch,
                                  size = (8, 8),
                                  x = (-10kilometers, 10kilometers),
                                  z = (0, 10kilometers),
                                  topology = (Periodic, Flat, Bounded))

    parent = PrescribedAtmosphere(parent_grid, [0.0, 1days])
    atmosphere = Simulation(NestedModel(parent, child); Δt=10)

    land_grid = RectilinearGrid(arch,
                                size = grid.Nx,
                                halo = grid.Hx,
                                x = (-10kilometers, 10kilometers),
                                topology = (Periodic, Flat, Flat))

    land = SlabLand(land_grid)
    set!(land; T=θ₀ + 5)

    return AtmosphereLandModel(atmosphere, land)
end

@testset "AtmosphereOceanModel with Breeze" begin
    for arch in test_architectures
        A = typeof(arch)

        @testset "atmosphere_simulation on $A" begin
            grid = RectilinearGrid(arch,
                                   size = (16, 16), halo = (5, 5),
                                   x = (-10kilometers, 10kilometers),
                                   z = (0, 10kilometers),
                                   topology = (Periodic, Flat, Bounded))

            atmos = atmosphere_simulation(grid)
            @test atmos isa Simulation
            @test atmos.model isa Breeze.AtmosphereModel
        end

        @testset "Construction on $A" begin
            model = build_test_model(arch)

            @test model isa EarthSystemModel
            @test model.ocean isa SlabOcean
            @test model.atmosphere isa Simulation
            @test model.atmosphere.model isa Breeze.AtmosphereModel
            @test model.architecture isa typeof(arch)

            # Check that interfaces were created
            @test !isnothing(model.interfaces)
            @test !isnothing(model.interfaces.atmosphere_ocean_interface)
        end

        @testset "Time stepping on $A" begin
            # KNOWN BROKEN on Breeze 0.7 (NumericalEarth/Breeze.jl#827): the coupled solver is
            # unstable at Δt = 10 s — the stable Δt dropped below 5 s vs ≥10 s on Breeze 0.6. The
            # failure mode is arch-dependent: CPU throws a θ^γ DomainError, GPU integrates to NaN.
            # Fold the would-be success into ONE @test_broken so it records broken on either arch
            # (throw or false) and flips to an error (prompting re-enable) once the run is clean again.
            @test_broken begin
                model = build_test_model(arch)
                SST_before = Array(interior(model.ocean.temperature))
                run!(Simulation(model, Δt=10, stop_iteration=10))
                SST_after = Array(interior(model.ocean.temperature))
                model.clock.iteration == 10 && !any(isnan, SST_after) && SST_after ≉ SST_before
            end
        end

        @testset "SST responds to fluxes on $A" begin
            # KNOWN BROKEN on Breeze 0.7 (same coupled-solver instability at Δt = 10 s as above;
            # NumericalEarth/Breeze.jl#827). CPU throws, GPU yields NaN fluxes — both record broken.
            @test_broken begin
                model = build_test_model(arch)
                run!(Simulation(model, Δt=10, stop_iteration=50))
                fluxes = model.interfaces.atmosphere_ocean_interface.fluxes
                !any(isnan, fluxes.sensible_heat) && maximum(abs, fluxes.sensible_heat) > 0
            end
        end
    end
end

@testset "AtmosphereLandModel with Breeze: surface stress feedback" begin
    for arch in test_architectures
        A = typeof(arch)

        @testset "MOST stress reaches the atmosphere bottom BC on $A" begin
            model = build_land_test_model(arch)
            update_state!(model)

            atmos = model.atmosphere.model
            al = model.interfaces.atmosphere_land_interface
            @test !isnothing(al)

            ρu_bc = atmos.momentum.ρu.boundary_conditions.bottom.condition
            ρv_bc = atmos.momentum.ρv.boundary_conditions.bottom.condition

            # The atmosphere's stored net momentum-flux fields ARE the bottom-flux BC fields.
            @test model.interfaces.net_fluxes.atmosphere.ρu === ρu_bc
            @test model.interfaces.net_fluxes.atmosphere.ρv === ρv_bc

            # After update_state!, those BC fields receive the computed MOST land surface
            # stress (the atmosphere-feels-drag link), and the stress is nonzero.
            @test Array(interior(ρu_bc)) ≈ Array(interior(al.fluxes.x_momentum))
            @test Array(interior(ρv_bc)) ≈ Array(interior(al.fluxes.y_momentum))
            @test maximum(abs, interior(ρu_bc)) > 0
        end

        @testset "Cloud condensate is excluded from interface specific humidity on $A" begin
            model = build_land_test_model(arch)
            atmosphere = model.atmosphere.model

            # At 280 K, qᵗ = 0.02 produces cloud liquid under the default warm-phase
            # saturation adjustment. The surface solver must receive vapor only.
            set!(atmosphere; T = 280, qᵗ = 0.02)
            update_state!(model)

            qᵛ = Array(interior(specific_humidity(atmosphere), :, :, 1))
            qˡ = Array(interior(atmosphere.microphysical_fields.qˡ, :, :, 1))
            q_interface = Array(interior(model.interfaces.exchanger.atmosphere.state.q, :, :, 1))

            @test minimum(qˡ) > 0
            @test q_interface ≈ qᵛ
        end
    end
end

@testset "Breeze RTM surface longwave follows live temperature" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch;
                               size = 1,
                               x = (0, 1),
                               y = (0, 1),
                               z = (-1, 0),
                               topology = (Flat, Flat, Bounded))

        surface_energy_flux = CenterField(grid)
        surface_temperature = CenterField(grid)
        surface_emissivity = CenterField(grid)
        downwelling_longwave_flux = CenterField(grid)
        downwelling_shortwave_flux = CenterField(grid)
        surface_albedo = CenterField(grid)

        initial_surface_energy_flux = 25
        first_surface_temperature = 280
        second_surface_temperature = 300
        emissivity = 0.95
        downwelling_longwave = 300
        downwelling_shortwave = 600
        albedo = 0.2
        stefan_boltzmann_constant = convert(eltype(grid), default_stefan_boltzmann_constant)

        fill!(parent(surface_emissivity), emissivity)
        fill!(parent(downwelling_longwave_flux), downwelling_longwave)
        fill!(parent(downwelling_shortwave_flux), downwelling_shortwave)
        fill!(parent(surface_albedo), albedo)

        function surface_energy_at(temperature)
            fill!(parent(surface_energy_flux), initial_surface_energy_flux)
            fill!(parent(surface_temperature), temperature)
            launch!(arch, grid, :xy,
                    NumericalEarthBreezeExt._apply_breeze_air_land_radiative_fluxes!,
                    surface_energy_flux,
                    surface_temperature,
                    surface_emissivity,
                    stefan_boltzmann_constant,
                    downwelling_longwave_flux,
                    downwelling_shortwave_flux,
                    surface_albedo)
            return only(Array(interior(surface_energy_flux)))
        end

        first_surface_energy = surface_energy_at(first_surface_temperature)
        second_surface_energy = surface_energy_at(second_surface_temperature)

        expected_surface_energy(temperature) =
            initial_surface_energy_flux +
            emissivity * stefan_boltzmann_constant * temperature^4 -
            emissivity * downwelling_longwave -
            (1 - albedo) * downwelling_shortwave

        @test first_surface_energy ≈ expected_surface_energy(first_surface_temperature)
        @test second_surface_energy ≈ expected_surface_energy(second_surface_temperature)
        @test second_surface_energy - first_surface_energy ≈
              emissivity * stefan_boltzmann_constant *
              (second_surface_temperature^4 - first_surface_temperature^4)
    end
end

@testset "AtmosphereLandModel with a nested Breeze atmosphere" begin
    for arch in test_architectures
        A = typeof(arch)

        @testset "Simulation(NestedModel) couples like its child on $A" begin
            model = build_nested_land_test_model(arch)

            @test model isa EarthSystemModel
            @test model.atmosphere isa Simulation
            @test model.atmosphere.model isa NestedModel
            @test !isnothing(model.interfaces.atmosphere_land_interface)

            update_state!(model)

            child = model.atmosphere.model.child
            al = model.interfaces.atmosphere_land_interface

            ρu_bc = child.momentum.ρu.boundary_conditions.bottom.condition
            ρv_bc = child.momentum.ρv.boundary_conditions.bottom.condition

            # The net fluxes extracted through the nest ARE the child's bottom-flux BC fields.
            @test model.interfaces.net_fluxes.atmosphere.ρu === ρu_bc
            @test model.interfaces.net_fluxes.atmosphere.ρv === ρv_bc

            # MOST land surface stress reaches the nested child's bottom BC.
            @test Array(interior(ρu_bc)) ≈ Array(interior(al.fluxes.x_momentum))
            @test Array(interior(ρv_bc)) ≈ Array(interior(al.fluxes.y_momentum))
            @test maximum(abs, interior(ρu_bc)) > 0
        end

        @testset "Coupled step keeps parent and child clocks synchronized on $A" begin
            model = build_nested_land_test_model(arch)
            time_step!(model, 1)

            nest = model.atmosphere.model
            @test nest.child.clock.time ≈ 1
            @test nest.parent.clock.time ≈ nest.child.clock.time
            @test model.land.clock.time ≈ nest.child.clock.time
        end

        @testset "Child rain reaches the exchanger and the land bucket on $A" begin
            # After one coupled step the Kessler child's surface rain flux must be positive
            # and assembled into the land's precipitation accumulator. Kessler requires
            # Tetens saturation constants (Breeze.jl#858).
            model = build_nested_land_test_model(arch;
                microphysics = DCMIP2016KesslerMicrophysics(),
                thermodynamic_constants = ThermodynamicConstants(saturation_vapor_pressure = TetensFormula(liquid_temperature_offset = 36)))
            child = model.atmosphere.model.child
            set!(child.microphysical_fields.ρqʳ, 1e-3)
            time_step!(model, 1)

            Jʳⁿ = model.interfaces.exchanger.atmosphere.state.Jʳⁿ
            @test maximum(Array(interior(Jʳⁿ))) > 0
            @test maximum(Array(interior(model.land.fluxes.precipitation))) > 0

            dry = build_nested_land_test_model(arch)
            update_state!(dry)
            Jʳⁿ_dry = dry.interfaces.exchanger.atmosphere.state.Jʳⁿ
            @test all(iszero, Array(interior(Jʳⁿ_dry)))
        end
    end
end

#####
##### Surface-layer reference height: per-column and GPU-safe (issue #379)
#####

@testset "state2dindex accessor" begin
    IC = NumericalEarth.EarthSystemModels.InterfaceComputations
    @test IC.state2dindex(3.0, 1, 1) === 3.0            # scalar broadcasts to every column
    @test IC.state2dindex(600, 5, 2) === 600            # Int (boundary-layer-height fallback)
    f = CenterField(RectilinearGrid(CPU(), size = (3, 3, 1), extent = (1, 1, 1)))
    set!(f, (x, y, z) -> 10x + y)
    @test IC.state2dindex(f, 2, 3) == f[2, 3, 1]        # field read at column (i, j)
end

@testset "surface_layer_height on a stretched vertical grid (issue #379)" begin
    for arch in test_architectures
        A = typeof(arch)

        @testset "builds + steps without host scalar indexing on $A" begin
            Nz = 24
            zfaces(k) = 10000 * (1 - cos(π/2 * (k - 1) / Nz))   # near-surface refinement
            grid = RectilinearGrid(arch, size = (8, 8, Nz), halo = (5, 5, 5),
                                   x = (-10kilometers, 10kilometers),
                                   y = (-10kilometers, 10kilometers),
                                   z = zfaces, topology = (Periodic, Periodic, Bounded))

            θ₀ = 285
            atmosphere = atmosphere_simulation(grid; potential_temperature = θ₀)
            set!(atmosphere.model, θ = θ₀, u = 5)   # uniform background θ; the u = 5 shear drives a nonzero surface stress

            land_grid = RectilinearGrid(arch, size = (grid.Nx, grid.Ny), halo = (grid.Hx, grid.Hy),
                                        x = (-10kilometers, 10kilometers),
                                        y = (-10kilometers, 10kilometers),
                                        topology = (Periodic, Periodic, Flat))
            land = SlabLand(land_grid)
            set!(land; T = θ₀ + 5)

            model = AtmosphereLandModel(atmosphere, land)

            # A non-terrain grid has a horizontally uniform first-cell height, so the cached
            # reference height is a scalar equal to ½·Δz(1).
            slh = model.interfaces.properties.surface_layer_height
            @test slh isa Number
            z1 = @allowscalar Oceananigans.zspacing(1, 1, 1, grid, Center(), Center(), Center()) / 2
            @test slh ≈ z1

            # On GPU the old host read of a device Δz array threw here; now it steps clean.
            update_state!(model)
            al = model.interfaces.atmosphere_land_interface
            @test !any(isnan, Array(interior(al.fluxes.friction_velocity)))
            @test maximum(abs, interior(al.fluxes.x_momentum)) > 0
        end
    end
end

@testset "surface_layer_height on a terrain-following grid" begin
    for arch in test_architectures
        A = typeof(arch)

        @testset "per-column reference height varies over terrain on $A" begin
            Nx, Nz = 24, 16
            Lx, Lz = 40kilometers, 5000.0
            zfaces = TerrainFollowingVerticalDiscretization(collect(range(0, Lz, length = Nz + 1));
                                                            formulation = LinearDecay())
            grid = RectilinearGrid(arch; size = (Nx, Nz), halo = (5, 5),
                                   x = (-Lx/2, Lx/2), z = zfaces,
                                   topology = (Periodic, Flat, Bounded))
            h₀, a = 800.0, 4000.0
            materialize_terrain!(grid, x -> h₀ * exp(-x^2 / a^2))

            θ₀ = 285
            atmosphere = atmosphere_simulation(grid; potential_temperature = θ₀)
            set!(atmosphere.model, θ = θ₀, u = 5)   # uniform background θ; the u = 5 shear drives a nonzero surface stress

            land_grid = RectilinearGrid(arch; size = Nx, halo = 5,
                                        x = (-Lx/2, Lx/2), topology = (Periodic, Flat, Flat))
            land = SlabLand(land_grid)
            set!(land; T = θ₀ + 5)

            model = AtmosphereLandModel(atmosphere, land)

            # Terrain makes the first-cell height vary per column → a 2-D field, each
            # column equal to ½·Δz(i, 1).
            slh = model.interfaces.properties.surface_layer_height
            @test slh isa Field
            slh_h = Array(interior(slh))[:]
            z_expected = @allowscalar [Oceananigans.zspacing(i, 1, 1, grid, Center(), Center(), Center()) / 2 for i in 1:Nx]
            @test slh_h ≈ z_expected
            @test maximum(slh_h) - minimum(slh_h) > 0     # genuinely varies over the hill

            update_state!(model)
            al = model.interfaces.atmosphere_land_interface
            @test !any(isnan, Array(interior(al.fluxes.friction_velocity)))
        end
    end
end
