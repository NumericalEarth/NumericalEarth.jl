include("runtests_setup.jl")

using Breeze
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.TimeSteppers: update_state!
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
    set!(atmosphere.model, θ=atmosphere.model.dynamics.reference_state.potential_temperature, u=1)

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
    set!(atmosphere.model, θ=atmosphere.model.dynamics.reference_state.potential_temperature, u=5)

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
            model = build_test_model(arch)
            SST = model.ocean.temperature
            SST_before = Array(interior(SST))

            simulation = Simulation(model, Δt=10, stop_iteration=10)
            run!(simulation)

            @test model.clock.iteration == 10
            @test model.clock.time > 0

            SST_after = Array(interior(SST))
            # SST should have changed and contain no NaN
            @test !any(isnan, SST_after)
            @test SST_after ≉ SST_before
        end

        @testset "SST responds to fluxes on $A" begin
            model = build_test_model(arch)

            simulation = Simulation(model, Δt=10, stop_iteration=50)
            run!(simulation)

            # Check that the ESM interface fluxes are nonzero
            ao_fluxes = model.interfaces.atmosphere_ocean_interface.fluxes
            @test maximum(abs, ao_fluxes.sensible_heat) > 0
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
