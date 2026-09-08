include("runtests_setup.jl")

using CUDA
using Oceananigans.OrthogonalSphericalShellGrids
using NumericalEarth.EarthSystemModels: above_freezing_ocean_temperature!
using ClimaSeaIce.SeaIceDynamics
using ClimaSeaIce.SeaIceThermodynamics: melting_temperature
using ClimaSeaIce.Rheologies

@inline kernel_melting_temperature(i, j, k, grid, liquidus, S) = @inbounds melting_temperature(liquidus, S[i, j, k])

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
            true
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
