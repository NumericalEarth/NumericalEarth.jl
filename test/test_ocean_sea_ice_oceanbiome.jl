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

        biogeochemistry = ImplicitBiology(grid;
                                          light_attenuation = PrescribedAttenuationPAR(grid, surface_PAR), 
                                          oxygen = Oxygen(), 
                                          inorganic_carbon=CarbonateSystem())

        @info "Testing timestepping on $arch"

        O₂ = Oceananigans.Fields.ConstantField(100)
        pCO₂ = FieldTimeSeries((nothing, nothing, nothing), grid, 0:10800:3.15252f7, time_indexing = Oceananigans.OutputReaders.Cyclical())
        pCO₂ .= reshape(250 .+ 200 .* [1:2920;] ./ 2920, 1, 1, 1, 2920)

        river_alkalinity = Field{Center, Center, Nothing}(grid)
        set!(river_alkalinity, (λ, φ) -> 1000 * (sqrt((λ - 310)^2 + (φ - 1)^2) < 10)) # set the amazon mouth to high concentration

        ocean = ocean_simulation(grid; free_surface, biogeochemistry, freshwater_tracer_content = (; Alk = river_alkalinity))
        sea_ice  = sea_ice_simulation(grid, ocean; advection=nothing)
        atmosphere = JRA55PrescribedAtmosphere(arch; 
                                               time_indices_in_memory=2, 
                                               tracers = (; O₂, pCO₂))
        radiation = JRA55PrescribedRadiation(arch; time_indices_in_memory=2)
        land = JRA55PrescribedLand(arch, time_indices_in_memory=2)

        coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation, land)

        @test maximum(biogeochemistry.light_attenuation.surface_PAR.surface_shortwave) > 0
        @test maximum(abs, ocean.model.tracers.DIC.boundary_conditions.top.condition.func.flux_field) > 0

        set!(sea_ice.model, ℵ = 1, h = 10)
        Oceananigans.TimeSteppers.update_state!(coupled_model)

        @test maximum(biogeochemistry.light_attenuation.surface_PAR.surface_shortwave) == 0
        @test maximum(abs, ocean.model.tracers.DIC.boundary_conditions.top.condition.func.flux_field) == 0

        set!(sea_ice.model, ℵ = (λ, φ) -> φ > 0, h = 10)
        Oceananigans.TimeSteppers.update_state!(coupled_model)

        @test maximum(view(biogeochemistry.light_attenuation.surface_PAR.surface_shortwave, :, 1:24, 1)) > 0
        @test maximum(abs, view(ocean.model.tracers.DIC.boundary_conditions.top.condition.func.flux_field, :, 1:24, 1)) > 0
        @test maximum(view(biogeochemistry.light_attenuation.surface_PAR.surface_shortwave, :, 25:50, 1)) == 0
        @test maximum(abs, view(ocean.model.tracers.DIC.boundary_conditions.top.condition.func.flux_field, :, 25:50, 1)) == 0

        time_step!(coupled_model, 1)

        @test maximum(ocean.model.tracers.Alk) > 0 # the amazon is doing some flux
    end
end
