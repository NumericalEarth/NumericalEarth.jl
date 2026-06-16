include("runtests_setup.jl")

using CUDA: @allowscalar
using Oceananigans.Units
using NumericalEarth.EarthSystemModels: above_freezing_ocean_temperature!

@testset "Tracer conservation under surface fluxes" begin
    for arch in test_architectures
        for fold_topology in (RightFaceFolded, RightCenterFolded)

            underlying_grid = TripolarGrid(arch;
                                           size = (20, 20, 20),
                                           z = (-100, 0),
                                           halo = (7, 7, 4),
                                           fold_topology=RightFaceFolded)

            bottom_height = regrid_bathymetry(underlying_grid,
                                              Metadatum(:bottom_height, dataset=ETOPO2022());
                                              minimum_depth=15,
                                              interpolation_passes=1,
                                              major_basins=1)

            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height);
                                        active_cells_map=true)

            ocean = ocean_simulation(grid, free_surface=SplitExplicitFreeSurface(substeps=20))
            sea_ice = sea_ice_simulation(grid, ocean)

            time_indices_in_memory = 4
            radiation = JRA55PrescribedRadiation(arch; time_indices_in_memory)
            atmosphere = JRA55PrescribedAtmosphere(arch; time_indices_in_memory)

            date = DateTime(1993, 1, 1)
            dataset = ECCO4Monthly()

            set!(ocean.model,
                 T = Metadatum(:temperature; dataset, date),
                 S = Metadatum(:salinity; dataset, date))

            coupled_model = OceanOnlyModel(ocean; atmosphere, radiation)

            Δt = 65seconds
            simulation = Simulation(coupled_model; Δt, stop_iteration=4)

            ocean_properties = simulation.model.interfaces.ocean_properties
            ρᵒᶜ = ocean_properties.reference_density
            cᵒᶜ = ocean_properties.heat_capacity
            Sᵒᶜ = 35

            surface_forcing = (;
                heat_flux = net_ocean_heat_flux(simulation.model),
                freshwater_flux = net_ocean_freshwater_flux(simulation.model, reference_salinity=Sᵒᶜ),
            )

            T = simulation.model.ocean.model.tracers.T
            S = simulation.model.ocean.model.tracers.S

            T⁻ = Field(T)
            S⁻ = Field(S)

            ΔT = Field(T - T⁻)
            ΔS = Field(S - S⁻)

            Δocean_heat_content = Integral(ρᵒᶜ * cᵒᶜ * ΔT, dims=(1, 2, 3))
            Δfreshwater_content = Integral(-ρᵒᶜ / Sᵒᶜ * ΔS, dims=(1, 2, 3))

            heat_rate = Integral(surface_forcing.heat_flux, dims=(1, 2))
            freshwater_rate = Integral(surface_forcing.freshwater_flux, dims=(1, 2))

            for _ = 1:simulation.stop_iteration

                set!(T⁻, T)
                set!(S⁻, S)

                previous_heat_flux = @allowscalar first(Field(heat_rate))
                previous_freshwater_flux = @allowscalar first(Field(freshwater_rate))

                time_step!(coupled_model, Δt)

                last_Δt = simulation.model.clock.last_Δt

                heat_content_tendency = @allowscalar first(Field(Δocean_heat_content))
                freshwater_content_tendency = @allowscalar first(Field(Δfreshwater_content))

                expected_heat_content_tendency = -previous_heat_flux * last_Δt
                expected_freshwater_content_tendency = -previous_freshwater_flux * last_Δt

                @info "Iteration $(simulation.model.clock.iteration): time = $(Float64(simulation.model.clock.time)) s, Δt = $(last_Δt) s"

                @show heat_content_tendency
                @show expected_heat_content_tendency
                @show heat_content_tendency - expected_heat_content_tendency

                @test heat_content_tendency ≈ expected_heat_content_tendency
                @test freshwater_content_tendency ≈ expected_freshwater_content_tendency
            end
        end
    end
end