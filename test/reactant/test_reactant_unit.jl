using Test
using Reactant
using Oceananigans: Oceananigans, CPU, interior, set!
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: Bounded, Flat, LatitudeLongitudeGrid, Periodic, RectilinearGrid
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ExplicitFreeSurface, HydrostaticFreeSurfaceModel
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth
using Breeze: CompressibleDynamics, SplitExplicitTimeDiscretization

include("setup.jl")

@testset "Reactant land energy flux assembly" begin
    for (temperature, expected_flux) in ((BulkTemperature(), 200),
                                        (SoilSkinTemperature(1.5, 0.05), -150))
        grid = LatitudeLongitudeGrid(ReactantState(); size = 1, latitude = 10, longitude = 10,
                                     z = (-1, 0), topology = (Flat, Flat, Bounded))
        atmosphere = PrescribedAtmosphere(grid, [0.0, 1.0e8])
        land = SlabLand(grid)
        fluxes = SimilarityTheoryFluxes(; solver_stop_criteria=FixedIterations(8))
        model = AtmosphereLandModel(atmosphere, land; atmosphere_land_fluxes=fluxes, atmosphere_land_interface_state=(; temperature, specific_humidity=BulkHumidity()))
        interface = model.interfaces.atmosphere_land_interface
        set!(land; T=290)
        set!(interface.temperature, 295)
        set!(interface.fluxes.sensible_heat, 80)
        set!(interface.fluxes.latent_heat, 120)

        Reactant.@jit NumericalEarth.EarthSystemModels.update_net_fluxes!(model, land)
        @test only(Array(interior(land.fluxes.surface_energy_flux))) ≈ expected_flux
    end
end

# A compiled step of a NestedModel advances a prescribed parent with the child: the parent takes the
# child's (device) clock instead of branching on the traced gap between them (issue #816).
@testset "NestedModel with a prescribed parent steps under compilation" begin
    grid = RectilinearGrid(ReactantState(); size = (4, 4, 4), extent = (1, 1, 1), topology = (Periodic, Periodic, Bounded))
    parent = PrescribedAtmosphere(grid, [0.0, 10.0])
    # An explicit free surface: the implicit one's FFT solver works on complex arrays, which Reactant cannot raise
    child = HydrostaticFreeSurfaceModel(grid; free_surface = ExplicitFreeSurface())
    nested = NumericalEarth.NestedModels.NestedModel(parent, child)
    @test parent.clock.time isa Reactant.ConcreteRNumber

    step! = @compile raise_first = true raise = true time_step!(nested, 1.0)
    step!(nested, 1.0)
    step!(nested, 1.0)
    @test Reactant.to_number(nested.child.clock.time) == 2
    @test Reactant.to_number(parent.clock.time) == 2

    clock = NumericalEarth.EarthSystemModels.default_clock(ReactantState(), Float32)
    @test clock.time isa Reactant.ConcreteRNumber{Float64}
end

# `atmosphere_model` defaults to `Clock(grid)`, a device clock on ReactantState, so a compiled step
# advances it; a host clock stays where tracing left it (issue #810).
@testset "atmosphere_model's clock advances through compiled steps" begin
    grid = RectilinearGrid(ReactantState(); size = (8, 8, 8), x = (0, 8e3), y = (0, 8e3), z = (0, 8e3),
                           topology = (Periodic, Periodic, Bounded), halo = (5, 5, 5))
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(substeps = 4); reference_potential_temperature = 300)
    model = atmosphere_model(grid; dynamics, microphysics = nothing)
    @test model.clock.time isa Reactant.ConcreteRNumber

    step! = @compile raise_first = true raise = true time_step!(model, 1.0)
    step!(model, 1.0)
    step!(model, 1.0)
    @test Reactant.to_number(model.clock.time) == 2
end

# Broadcasting into `interior` of a ReactantState field reads device memory element by element, which
# Reactant refuses, so the level pressure goes through `set!`.
@testset "ERA5 pressure_level_field on ReactantState" begin
    levels = Float32[1000, 850, 500] .* 100
    for arch in (CPU(), ReactantState())
        grid = LatitudeLongitudeGrid(arch, Float32; size = (4, 4, 3), longitude = (0, 1), latitude = (0, 1), z = (0, 1))
        pressure = NumericalEarth.DataWrangling.ERA5.pressure_level_field(grid, (; pressure_levels = levels), arch)
        @test all(Array(interior(pressure))[i, j, :] == levels for i in 1:4, j in 1:4)
    end
end

# A NetCDF slice (`Array{Union{Missing, T}}`) reaches the device with `missing` as NaN: Reactant cannot
# hold the `Missing` element type (EnzymeAD/Reactant.jl#3569).
@testset "A slice with missing values moved to ReactantState" begin
    slice = Union{Missing, Float32}[1, missing, 1]
    device_slice = Oceananigans.Architectures.on_architecture(ReactantState(), slice)
    @test device_slice isa Reactant.AbstractConcreteArray{Float32}
    @test isequal(Array(device_slice), Float32[1, NaN, 1])

    reshaped_slice = Oceananigans.Architectures.on_architecture(ReactantState(), reshape(slice, 1, 3))
    @test size(reshaped_slice) == (1, 3)
    @test isequal(Array(reshaped_slice), Float32[1 NaN 1])
end

# The coupled model's default clock around an atmosphere `Simulation` on ReactantState is a device clock in every
# field, so a compiled step can tick it: `Clock{TT}(time = 0)` left the iteration a host `Int`.
@testset "EarthSystemModel default clock ticks under compilation" begin
    grid = Oceananigans.RectilinearGrid(ReactantState(); size = (2, 2, 2), extent = (1, 1, 1))
    model = Oceananigans.HydrostaticFreeSurfaceModel(grid; free_surface = Oceananigans.ExplicitFreeSurface())
    atmosphere = Oceananigans.Simulation(model; Δt = 1, verbose = false)
    clock = NumericalEarth.EarthSystemModels.default_earth_system_clock(atmosphere)
    @test clock.iteration isa Reactant.ConcreteRNumber{Int}

    @jit Oceananigans.TimeSteppers.tick!(clock, 1.0)
    @test Reactant.to_number(clock.time) == 1
    @test Reactant.to_number(clock.iteration) == 1
end
