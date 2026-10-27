include("runtests_setup.jl")

using CUDA
using Oceananigans.OrthogonalSphericalShellGrids
using Oceananigans.Grids: znode
using Oceananigans.Fields: interior
using NumericalEarth.EarthSystemModels: above_freezing_ocean_temperature!
using ClimaSeaIce.SeaIceDynamics
using ClimaSeaIce.SeaIceThermodynamics: melting_temperature
using ClimaSeaIce.SeaIceThermodynamics: SlabThermodynamics, ColumnEnergyThermodynamics,
                                        SeaIceColumnDiscretization, FixedDrainedIceSalinityProfile
using NumericalEarth.EarthSystemModels.InterfaceComputations: BulkTemperature, SkinTemperature,
                                                              default_ai_temperature, ThreeEquationHeatFlux
using ClimaSeaIce.Rheologies

@inline kernel_melting_temperature(i, j, k, grid, liquidus, S) =
    @inbounds melting_temperature(liquidus, S[i, j, k], znode(i, j, k, grid, Center(), Center(), Center()))

@testset "Deep ocean initialization uses in situ freezing" begin
    grid = RectilinearGrid(size = 2, z = (-1000, 0), topology = (Flat, Flat, Bounded))
    ocean = ocean_simulation(grid; momentum_advection = nothing, tracer_advection = nothing,
                             closure = nothing, radiative_forcing = nothing)
    sea_ice = sea_ice_simulation(grid, ocean; dynamics = nothing)
    liquidus = sea_ice.model.phase_transitions.liquidus
    salinity = 35.0
    surface_freezing = melting_temperature(liquidus, salinity)
    bottom_depth = znode(1, 1, 1, grid, Center(), Center(), Center())
    bottom_freezing = melting_temperature(liquidus, salinity, bottom_depth)
    initial_temperature = (surface_freezing + bottom_freezing) / 2

    set!(ocean.model; T = initial_temperature, S = salinity)
    above_freezing_ocean_temperature!(ocean, grid, sea_ice)

    @test bottom_freezing < initial_temperature < surface_freezing
    @test Array(interior(ocean.model.tracers.T))[1, 1, 1] ≈ initial_temperature
end

@testset "Time stepping test" begin
    for arch in test_architectures
        A = typeof(arch)

        grid = TripolarGrid(arch;
                            size = (50, 50, 10),
                            halo = (7, 7, 7),
                            z = (-5000, 0))

        bottom_height = synthetic_bottom_height(grid;
                                                minimum_depth = 10,
                                                interpolation_passes = 5,
                                                major_basins = 1)

        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)
        free_surface = SplitExplicitFreeSurface(grid; substeps=20)

        #####
        ##### Coupled ocean-sea ice and prescribed atmosphere
        #####

        initial_state = MetadataSet(:temperature, :salinity; dataset=SyntheticOcean(), date=DateTime(2000, 1, 1))

        ocean = ocean_simulation(grid; free_surface)

        sea_ice  = sea_ice_simulation(grid, ocean; advection=WENO(order=7))
        liquidus = sea_ice.model.phase_transitions.liquidus

        # Set the ocean temperature and salinity
        set!(ocean.model, initial_state)
        above_freezing_ocean_temperature!(ocean, grid, sea_ice)

        # Test that ocean temperatures are above freezing
        T = on_architecture(CPU(), ocean.model.tracers.T)
        S = on_architecture(CPU(), ocean.model.tracers.S)

        Tm = KernelFunctionOperation{Center, Center, Center}(kernel_melting_temperature, grid, liquidus, S)
        @test all(T .≥ Tm)

        atmosphere = synthetic_prescribed_atmosphere(arch)
        radiation = synthetic_prescribed_radiation(arch)

        # Fluxes are computed when the model is constructed, so we just test that this works.
        # And that we can time step with sea ice
        @test begin
            coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
            time_step!(coupled_model, 1)
            iteration(sea_ice) == 1 && sea_ice.model.clock.time == ocean.model.clock.time
        end

        @info "Testing OceanSeaIceModel with land on $A..."
        land = synthetic_prescribed_land(arch)

        @test begin
            ocean_with_land = ocean_simulation(grid; free_surface)
            set!(ocean_with_land.model, initial_state)
            sea_ice_with_land = sea_ice_simulation(grid, ocean_with_land; advection=WENO(order=7))
            above_freezing_ocean_temperature!(ocean_with_land, grid, sea_ice_with_land)

            coupled_model = OceanSeaIceModel(ocean_with_land, sea_ice_with_land; atmosphere, land, radiation)
            @test !isnothing(coupled_model.interfaces.exchanger.land)
            time_step!(coupled_model, 1)
            true
        end
    end
end

@testset "Selectable sea-ice thermodynamics" begin
    horizontal = (x = (0, 4), y = (0, 4))
    slab_grid = RectilinearGrid(CPU(); size = (4, 4, 1), halo = (3, 3, 3),
                                z = (-1, 0), topology = (Periodic, Periodic, Bounded), horizontal...)
    column_grid = RectilinearGrid(CPU(); size = (4, 4, 8), halo = (3, 3, 3),
                                  z = SeaIceColumnDiscretization((0, 1)),
                                  topology = (Periodic, Periodic, Bounded), horizontal...)

    slab = sea_ice_simulation(slab_grid; dynamics = nothing)
    column = sea_ice_simulation(column_grid; thermodynamics = :column, dynamics = nothing)

    @test slab.model.ice_thermodynamics isa SlabThermodynamics
    @test default_ai_temperature(slab) isa SkinTemperature
    @test !isnothing(ThreeEquationHeatFlux(slab).conductive_flux)

    @test column.model.ice_thermodynamics isa ColumnEnergyThermodynamics
    @test default_ai_temperature(column) isa BulkTemperature
    @test isnothing(ThreeEquationHeatFlux(column).conductive_flux)

    profile = FixedDrainedIceSalinityProfile(Float64)
    profiled_column = sea_ice_simulation(column_grid; thermodynamics = :column,
                                         column_bulk_salinity = profile, dynamics = nothing)
    salinity = Array(interior(profiled_column.model.ice_thermodynamics.fields.bulk_salinity))
    @test salinity[1, 1, 1] > salinity[1, 1, end]
    @test isapprox(salinity[1, 1, end], profile(7.5 / 8); atol = 1e-14)

    set!(profiled_column.model; h = 1, ℵ = 0.8, hs = 0.1)
    set!(profiled_column.model.ice_thermodynamics; bulk_salinity = profile, temperature = -5)
    time_step!(profiled_column.model, 60)
    initialized_salinity = Array(interior(profiled_column.model.ice_thermodynamics.fields.bulk_salinity))
    initialized_energy = Array(interior(profiled_column.model.ice_thermodynamics.fields.internal_energy))
    @test initialized_salinity == salinity
    @test all(isfinite, initialized_energy)
    @test initialized_energy[1, 1, 1] != initialized_energy[1, 1, end]

    set!(column.model; h = 1, ℵ = 0.8, hs = 0.1)
    set!(column.model.ice_thermodynamics; temperature = -5)
    time_step!(column.model, 60)
    @test all(isfinite, Array(interior(column.model.ice_thermodynamics.fields.temperature)))
end
