include("runtests_setup.jl")

using CUDA: @allowscalar
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.Units
using NumericalEarth.EarthSystemModels: above_freezing_ocean_temperature!

@testset "Tracer conservation under surface fluxes" begin
    for arch in test_architectures
        for fold_topology in (RightFaceFolded, RightCenterFolded)
            fold_topology = RightFaceFolded
            underlying_grid = TripolarGrid(arch;
                                           size = (20, 20, 20),
                                           z = (-100, 0),
                                           halo = (7, 7, 4),
                                           fold_topology)

            bottom_height = regrid_bathymetry(underlying_grid, Metadatum(:bottom_height, dataset=ETOPO2022());
                                              minimum_depth=15,
                                              interpolation_passes=1,
                                              major_basins=1)

            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

            ocean = ocean_simulation(grid, free_surface=SplitExplicitFreeSurface(substeps=20))
            sea_ice = sea_ice_simulation(grid, ocean)

            time_indices_in_memory = 2
            radiation = JRA55PrescribedRadiation(arch; time_indices_in_memory)
            atmosphere = JRA55PrescribedAtmosphere(arch; time_indices_in_memory)

            # initialize ocean and sea ice models
            date = DateTime(1993, 1, 1)
            dataset = ECCO4Monthly()

            set!(ocean.model,
                 T=Metadatum(:temperature; dataset, date),
                 S=Metadatum(:salinity; dataset, date))

            set!(sea_ice.model,
                 h=Metadatum(:sea_ice_thickness; dataset, date),
                 ℵ=Metadatum(:sea_ice_concentration; dataset, date))
            above_freezing_ocean_temperature!(ocean, grid, sea_ice)

            coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

            Δt = 65seconds
            simulation = Simulation(coupled_model; Δt, stop_iteration=3)

            ocean_properties = simulation.model.interfaces.ocean_properties
            ρᵒᶜ = ocean_properties.reference_density
            cᵒᶜ = ocean_properties.heat_capacity
            Sᵒᶜ = 35

            surface_forcing = (; heat_flux = net_ocean_heat_flux(simulation.model),
                                 freshwater_flux = net_ocean_freshwater_flux(simulation.model, reference_salinity=Sᵒᶜ))

            T = simulation.model.ocean.model.tracers.T
            S = simulation.model.ocean.model.tracers.S

            ocean_heat_content = Integral( ρᵒᶜ * cᵒᶜ * T, dims=(1, 2, 3))
            freshwater_content = Integral(-ρᵒᶜ / Sᵒᶜ * S, dims=(1, 2, 3))
            heat_rate = Integral(surface_forcing.heat_flux, dims=(1, 2))
            freshwater_rate = Integral(surface_forcing.freshwater_flux, dims=(1, 2))

            previous_ocean_heat_content = @allowscalar first(Field(ocean_heat_content))
            previous_freshwater_content = @allowscalar first(Field(freshwater_content))
            previous_heat_flux = @allowscalar first(Field(heat_rate))
            previous_freshwater_flux= @allowscalar first(Field(freshwater_rate))

            previous_time = Float64(simulation.model.clock.time)

            @test abs(previous_heat_flux) > 0
            @test abs(previous_freshwater_flux) > 0

            @show current_ocean_heat_content = @allowscalar first(Field(ocean_heat_content))
            @show current_freshwater_content = @allowscalar first(Field(freshwater_content))

            for _ = 1:simulation.stop_iteration
                time_step!(coupled_model, Δt)

                @show current_ocean_heat_content = @allowscalar first(Field(ocean_heat_content))
                @show current_freshwater_content = @allowscalar first(Field(freshwater_content))
                current_time = Float64(simulation.model.clock.time)
                last_Δt = simulation.model.clock.last_Δt


                @show current_ocean_heat_content - previous_ocean_heat_content
                @show -previous_heat_flux * last_Δt

                @show current_freshwater_content - previous_freshwater_content
                @show -previous_freshwater_flux * last_Δt

                @test current_ocean_heat_content - previous_ocean_heat_content ≈ -previous_heat_flux * last_Δt rtol=1e-4 atol=1e-2
                @test current_freshwater_content - previous_freshwater_content ≈ -previous_freshwater_flux * last_Δt rtol=1e-4 atol=1e-2

                previous_ocean_heat_content = current_ocean_heat_content
                previous_freshwater_content = current_freshwater_content
                previous_heat_flux = @allowscalar first(heat_rate)
                previous_freshwater_flux = @allowscalar first(freshwater_rate)
                previous_time = current_time
            end
        end
    end
end
