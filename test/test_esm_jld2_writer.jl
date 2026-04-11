include("runtests_setup.jl")

@testset "EarthSystemModel JLD2 outputwriting" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch, size=10, z=(-100, 0), topology=(Flat, Flat, Bounded))
        ocean = ocean_simulation(grid, timestepper=:QuasiAdamsBashforth2)
        model = OceanOnlyModel(ocean)

        # model.grid should return the exchange grid
        @test model.grid === model.interfaces.exchanger.grid

        # default_included_properties should include :grid
        @test :grid in Oceananigans.Models.default_included_properties(model)

        T = ocean.model.tracers.T

        filepath = "esm_roundtrip_test.jld2"

        sim = Simulation(model; Δt=1, stop_iteration=5)
        sim.output_writers[:test] = JLD2Writer(sim.model, (; T);
                                               schedule = IterationInterval(1),
                                               filename = filepath,
                                               overwrite_existing = true)
        run!(sim)

        # Load with FieldTimeSeries and verify the grid roundtrips
        fts = FieldTimeSeries(filepath, "T")
        @test size(fts.grid) == size(grid)
        @test length(fts.times) == 6

        rm(filepath, force=true)
    end
end
