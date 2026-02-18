using Breeze
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Test

NumericalEarthBreezeExt = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
@test !isnothing(NumericalEarthBreezeExt)

# Helper to build a fresh atmosphere + slab ocean model
function build_test_model()
    grid = RectilinearGrid(size = (16, 16), halo = (5, 5),
                           x = (-10kilometers, 10kilometers),
                           z = (0, 10kilometers),
                           topology = (Periodic, Flat, Bounded))

    θ₀ = 285

    atmosphere = atmosphere_simulation(grid; potential_temperature=θ₀)
    set!(atmosphere, θ=atmosphere.dynamics.reference_state.potential_temperature, u=1)

    sst_grid = RectilinearGrid(grid.architecture,
                               size = grid.Nx,
                               halo = grid.Hx,
                               x = (-10kilometers, 10kilometers),
                               topology = (Periodic, Flat, Flat))

    ocean = SlabOcean(sst_grid, depth=50, density=1025, heat_capacity=4000)
    set!(ocean, T=θ₀)

    model = AtmosphereOceanModel(atmosphere, ocean)

    return model
end

@testset "AtmosphereOceanModel with Breeze" begin
    @testset "atmosphere_simulation" begin
        grid = RectilinearGrid(size = (16, 16), halo = (5, 5),
                               x = (-10kilometers, 10kilometers),
                               z = (0, 10kilometers),
                               topology = (Periodic, Flat, Bounded))

        atmos = atmosphere_simulation(grid)
        @test atmos isa Breeze.AtmosphereModel
    end

    @testset "Construction" begin
        model = build_test_model()

        @test model isa EarthSystemModel
        @test model.ocean isa SlabOcean
        @test model.atmosphere isa Breeze.AtmosphereModel
        @test model.architecture isa CPU

        # Check that interfaces were created
        @test !isnothing(model.interfaces)
        @test !isnothing(model.interfaces.atmosphere_ocean_interface)
    end

    @testset "Time stepping" begin
        model = build_test_model()
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

    @testset "SST responds to fluxes" begin
        model = build_test_model()

        simulation = Simulation(model, Δt=10, stop_iteration=50)
        run!(simulation)

        # Check that the ESM interface fluxes are nonzero
        ao_fluxes = model.interfaces.atmosphere_ocean_interface.fluxes
        @test maximum(abs, ao_fluxes.sensible_heat) > 0
    end
end
