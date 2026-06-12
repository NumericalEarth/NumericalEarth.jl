include("runtests_setup.jl")

using CUDA: @allowscalar
using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.EarthSystemModels: above_freezing_ocean_temperature!

scalar_integral(field) = begin
    compute!(field)
    @allowscalar interior(field)[1, 1, 1]
end

@testset "Tracer conservation under surface fluxes" begin
    for arch in test_architectures
        for topology in (RightFaceFolded, RightCenterFolded)
            underlying_grid = TripolarGrid(arch;
                                size = (10, 10, 10),
                                z = (-100, 0),
                                halo = (7, 7, 4),
                                fold_topology=topology)

            bottom_height = regrid_bathymetry(underlying_grid, Metadatum(:bottom_height, dataset=ETOPO2022());
                                minimum_depth=15,
                                interpolation_passes=25,
                                major_basins=1)

            grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map=true)

            ocean = ocean_simulation(grid)
            dates = collect(DateTime(1993, 1, 1):Month(1):DateTime(1993, 12, 1))

            dataset = ECCO4Monthly()
            set!(ocean.model,
            T=Metadata(:temperature; dates=first(dates), dataset),
            S=Metadata(:salinity; dates=first(dates), dataset))

            @info "Creating sea ice model"
            sea_ice = sea_ice_simulation(grid, ocean)
            time_indices_in_memory = 2

            set!(sea_ice.model,
            h=Metadatum(:sea_ice_thickness; dataset),
            ℵ=Metadatum(:sea_ice_concentration; dataset))
            above_freezing_ocean_temperature!(ocean, grid, sea_ice)

            @info "Defining Atmospheric state"
            radiation = JRA55PrescribedRadiation(arch; time_indices_in_memory)
            atmosphere = JRA55PrescribedAtmosphere(arch; time_indices_in_memory)

            @info "Defining coupled model"
            @time coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

            simulation = Simulation(coupled_model; Δt=10minutes)
            simulation.stop_iteration = 10

            surface_forcing = (; heat_flux=Field(net_ocean_heat_flux(simulation.model)),
                               fw_flux=Field(net_ocean_freshwater_flux(simulation.model)))

            ocean_properties = simulation.model.interfaces.ocean_properties
            ρ₀ = ocean_properties.reference_density
            cₚ = ocean_properties.heat_capacity

            ohc_integral = Field(Integral(simulation.model.ocean.model.tracers.T, dims=(1, 2, 3)))
            fw_content_integral = Field(Integral(simulation.model.ocean.model.tracers.S, dims=(1, 2, 3)))
            heat_flux_integral = Field(Integral(surface_forcing.heat_flux, dims=(1, 2, 3)))
            fw_flux_integral = Field(Integral(surface_forcing.fw_flux, dims=(1, 2, 3)))

            update_state!(simulation.model)

            previous_ohc = ρ₀ * cₚ * scalar_integral(ohc_integral)
            previous_fw_content = -ρ₀ / 35 * scalar_integral(fw_content_integral)
            previous_heat_flux = scalar_integral(heat_flux_integral)
            previous_fw_flux = scalar_integral(fw_flux_integral)
            previous_time = Float64(simulation.model.clock.time)

            @test abs(previous_heat_flux) > 0
            @test abs(previous_fw_flux) > 0

            for _ = 1:simulation.stop_iteration
                time_step!(simulation.model, simulation.Δt)

                current_ohc = ρ₀ * cₚ * scalar_integral(ohc_integral)
                current_fw_content = -ρ₀ / 35 * scalar_integral(fw_content_integral)
                current_time = Float64(simulation.model.clock.time)
                Δt_model = current_time - previous_time

                @test current_ohc - previous_ohc ≈ -previous_heat_flux * Δt_model rtol=1e-4 atol=1e-2
                @test current_fw_content - previous_fw_content ≈ -previous_fw_flux * Δt_model rtol=1e-4 atol=1e-2

                update_state!(simulation.model)

                previous_ohc = current_ohc
                previous_fw_content = current_fw_content
                previous_heat_flux = scalar_integral(heat_flux_integral)
                previous_fw_flux = scalar_integral(fw_flux_integral)
                previous_time = current_time
            end
        end
    end
end

