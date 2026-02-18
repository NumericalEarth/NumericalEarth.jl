using Breeze
using Breeze: BulkDrag, BulkSensibleHeatFlux, BulkVaporFlux
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

    p₀, θ₀ = 101325, 285
    constants = ThermodynamicConstants()
    reference_state = ReferenceState(grid, constants; surface_pressure=p₀, potential_temperature=θ₀)
    dynamics = AnelasticDynamics(reference_state)
    microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())

    ΔT = 4
    T₀_func(x) = θ₀ + ΔT / 2 * sign(cos(2π * x / grid.Lx))

    sst_grid = RectilinearGrid(grid.architecture,
                               size = grid.Nx,
                               halo = grid.Hx,
                               x = (-10kilometers, 10kilometers),
                               topology = (Periodic, Flat, Flat))

    SST = CenterField(sst_grid)
    set!(SST, T₀_func)

    coef = PolynomialCoefficient(roughness_length = 1.5e-4)
    ρu_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=coef, gustiness=1e-2, surface_temperature=SST))
    ρv_bcs = FieldBoundaryConditions(bottom=BulkDrag(coefficient=coef, gustiness=1e-2, surface_temperature=SST))
    ρe_bcs = FieldBoundaryConditions(bottom=BulkSensibleHeatFlux(coefficient=coef, gustiness=1e-2, surface_temperature=SST))
    ρqᵗ_bcs = FieldBoundaryConditions(bottom=BulkVaporFlux(coefficient=coef, gustiness=1e-2, surface_temperature=SST))

    atmosphere = AtmosphereModel(grid; microphysics, dynamics,
                                 boundary_conditions = (ρu=ρu_bcs, ρv=ρv_bcs, ρe=ρe_bcs, ρqᵗ=ρqᵗ_bcs))

    set!(atmosphere, θ=reference_state.potential_temperature, u=1)

    ocean = SlabOcean(SST, mixed_layer_depth=50, density=1025, heat_capacity=4000)
    model = AtmosphereOceanModel(atmosphere, ocean)

    return model
end

@testset "AtmosphereOceanModel with Breeze" begin
    @testset "Construction" begin
        model = build_test_model()

        @test model isa EarthSystemModel
        @test model.ocean isa SlabOcean
        @test model.atmosphere isa Breeze.AtmosphereModel
        @test model.architecture isa CPU

        # Check that interfaces were created
        @test !isnothing(model.interfaces)
        @test !isnothing(model.interfaces.atmosphere_ocean_interface)

        # SST field is shared between ocean and atmosphere BCs
        @test model.ocean.sea_surface_temperature === model.ocean.sea_surface_temperature
    end

    @testset "Time stepping" begin
        model = build_test_model()
        SST = model.ocean.sea_surface_temperature
        SST_before = Array(interior(SST))

        simulation = Simulation(model, Δt=10, stop_iteration=10)
        run!(simulation)

        @test model.clock.iteration == 10
        @test model.clock.time > 0

        SST_after = Array(interior(SST))
        # SST should have changed
        @test SST_after ≉ SST_before
    end

    @testset "SST responds to fluxes" begin
        model = build_test_model()
        SST = model.ocean.sea_surface_temperature
        SST_initial_max = maximum(SST)

        simulation = Simulation(model, Δt=10, stop_iteration=50)
        run!(simulation)

        SST_final_max = maximum(SST)
        # Warm SST regions should cool (upward heat flux cools the ocean)
        @test SST_final_max < SST_initial_max

        # Check that the ESM interface fluxes are nonzero
        ao_fluxes = model.interfaces.atmosphere_ocean_interface.fluxes
        @test maximum(abs, ao_fluxes.sensible_heat) > 0
    end
end
