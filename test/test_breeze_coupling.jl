include("runtests_setup.jl")

using Breeze
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: time_step!, update_state!
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
function build_nested_land_test_model(arch; microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium()))
    grid = RectilinearGrid(arch,
                           size = (16, 16), halo = (5, 5),
                           x = (-10kilometers, 10kilometers),
                           z = (0, 10kilometers),
                           topology = (Periodic, Flat, Bounded))

    θ₀ = 285

    child = atmosphere_model(grid; potential_temperature=θ₀, microphysics)
    set!(child, θ=child.dynamics.reference_state.surface_potential_temperature, u=5)

    parent_grid = RectilinearGrid(arch,
                                  size = (8, 8),
                                  x = (-10kilometers, 10kilometers),
                                  z = (0, 10kilometers),
                                  topology = (Periodic, Flat, Bounded))

    parent = PrescribedAtmosphere(parent_grid, [0.0, 1day])
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
            # Kessler child with rain in the column: after one coupled step the surface rain
            # flux must be positive and assembled into the land's precipitation accumulator;
            # the non-precipitating default carries the zero-flux fallback.
            model = build_nested_land_test_model(arch; microphysics = DCMIP2016KesslerMicrophysics())
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
