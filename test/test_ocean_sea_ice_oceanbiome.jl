include("runtests_setup.jl")

using OceanBioME

@testset "Time stepping test" begin
    for arch in test_architectures
        grid = TripolarGrid(arch;
                            size = (50, 50, 10),
                            halo = (7, 7, 7),
                            z = (-5000, 0))

        bottom_height = regrid_bathymetry(grid;
                                          minimum_depth = 10,
                                          interpolation_passes = 5,
                                          major_basins = 1)

        grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height); active_cells_map=true)
        free_surface = SplitExplicitFreeSurface(grid; substeps=20)

        surface_PAR = PARFromShortwave(grid)

        biogeochemistry = Biogeochemistry(nothing;
                                          light_attenuation = PrescribedAttenuationPAR(grid, surface_PAR))

        @info "Testing timestepping on $arch"

        ocean = ocean_simulation(grid; free_surface, biogeochemistry)
        sea_ice  = sea_ice_simulation(grid, ocean; advection=nothing)
        atmosphere = JRA55PrescribedAtmosphere(arch; time_indices_in_memory=2)
        radiation = JRA55PrescribedRadiation(arch; time_indices_in_memory=2)
        
        coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

        Oceananigans.TimeSteppers.update_state!(coupled_model)

        # its getting set
        @test maximum(biogeochemistry.light_attenuation.surface_PAR.surface_shortwave) > 0

        set!(sea_ice.model, ℵ = 1, h = 10)

        # for some reason we have to call this twice for it to work?
        Oceananigans.TimeSteppers.update_state!(coupled_model)
        Oceananigans.TimeSteppers.update_state!(coupled_model)

        # its post-ice masking
        @test maximum(biogeochemistry.light_attenuation.surface_PAR.surface_shortwave) == 0
    end
end
