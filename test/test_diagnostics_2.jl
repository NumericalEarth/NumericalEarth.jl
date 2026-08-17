include("runtests_setup.jl")

using NumericalEarth.Diagnostics: Diagnostics

temperature_budgets(simulation) =
    [callback.func for callback in values(simulation.callbacks)
     if callback.func isa BudgetComputation && callback.func.tracer_name === :temperature]

struct ConstantAdditionalTemperatureFlux{T}
    value :: T
end

@inline (flux::ConstantAdditionalTemperatureFlux)(i, j, grid, clock, fields) = flux.value

function analytical_immersed_grid(underlying_grid::TripolarGrid;
                                           radius = 5,
                                           active_cells_map = false)
    λp = underlying_grid.conformal_mapping.first_pole_longitude
    φp = underlying_grid.conformal_mapping.north_poles_latitude
    φm = underlying_grid.conformal_mapping.southernmost_latitude
    Lz = underlying_grid.Lz

    # Mask the two northern tripolar singularities and the South Pole.
    bottom_height(λ, φ) = ((abs(λ - λp) < radius)       & (abs(φp - φ) < radius)) |
                          ((abs(λ - λp - 180) < radius) & (abs(φp - φ) < radius)) |
                          (φ < φm) ? 0 : -Lz

    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map)
end


function analytical_immersed_grid(underlying_grid::LatitudeLongitudeGrid;
                                           radius = 5,
                                           active_cells_map = false)
    Lz = underlying_grid.Lz

    # Mask the polar caps so regridded integrals ignore cells that are not
    # part of the active ocean area.
    bottom_height(λ, φ) = abs(φ) > 90 - radius ? 0 : -Lz

    return ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height); active_cells_map)
end

for arch in test_architectures
    A = typeof(arch)
    @info "Testing InterfaceFluxOutputs [$A]..."

    grid = RectilinearGrid(arch;
                            size = (4, 5, 2),
                            extent = (1, 1, 1),
                            topology = (Periodic, Bounded, Bounded))

    T_flux_value = 2.0
    additional_T_flux_value = 0.4
    S_flux_value = 5.0
    frazil_heat_flux_value = 0.2
    interface_heat_flux_value = 0.3
    sea_ice_ocean_salt_flux_value = 0.9

    model_configurations = ((name = "OceanOnlyModel", with_sea_ice = false),
                            (name = "OceanSeaIceModel", with_sea_ice = true))

    for config in model_configurations
        @testset "InterfaceFluxOutputs on $(config.name) [$A]" begin
            ocean = ocean_simulation(grid;
                                     momentum_advection = nothing,
                                     tracer_advection = nothing,
                                     closure = nothing,
                                     coriolis = nothing,
                                     additional_surface_fluxes = (; T = ConstantAdditionalTemperatureFlux(additional_T_flux_value)))
            atmosphere = PrescribedAtmosphere(grid, [0.0])
            esm = if config.with_sea_ice
                sea_ice = sea_ice_simulation(grid, ocean)
                OceanSeaIceModel(ocean, sea_ice; atmosphere)
            else
                OceanOnlyModel(ocean; atmosphere)
            end

            T_flux = Diagnostics.flux_field(ocean.model.tracers.T.boundary_conditions.top.condition)
            S_flux = Diagnostics.flux_field(ocean.model.tracers.S.boundary_conditions.top.condition)

            fill!(T_flux, T_flux_value)
            fill!(S_flux, S_flux_value)

            if config.with_sea_ice
                sea_ice_ocean_fluxes = esm.interfaces.sea_ice_ocean_interface.fluxes
                fill!(sea_ice_ocean_fluxes.frazil_heat, frazil_heat_flux_value)
                fill!(sea_ice_ocean_fluxes.interface_heat, interface_heat_flux_value)
                fill!(sea_ice_ocean_fluxes.salt, sea_ice_ocean_salt_flux_value)
            end

            expected_frazil_heat = config.with_sea_ice ? frazil_heat_flux_value : 0
            expected_interface_heat = config.with_sea_ice ? interface_heat_flux_value : 0
            expected_sea_ice_ocean_salt = config.with_sea_ice ? sea_ice_ocean_salt_flux_value : 0

            ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
            cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
            Sᵒᶜ = 35.0

            # exported diagnostics
            net_ocean_heat = net_ocean_heat_flux(esm)
            sea_ice_ocean_heat = sea_ice_ocean_heat_flux(esm)
            atmosphere_ocean_heat = atmosphere_ocean_heat_flux(esm)
            net_ocean_freshwater = net_ocean_freshwater_flux(esm; reference_salinity = Sᵒᶜ)
            sea_ice_ocean_freshwater = sea_ice_ocean_freshwater_flux(esm; reference_salinity = Sᵒᶜ)
            atmosphere_ocean_freshwater = atmosphere_ocean_freshwater_flux(esm; reference_salinity = Sᵒᶜ)

            # internal diagnostics
            frazil_temperature = Diagnostics.frazil_temperature_flux(esm)
            net_ocean_temperature = Diagnostics.net_ocean_temperature_flux(esm)
            sea_ice_ocean_temperature = Diagnostics.sea_ice_ocean_temperature_flux(esm)
            atmosphere_ocean_temperature = Diagnostics.atmosphere_ocean_temperature_flux(esm)
            frazil_heat = Diagnostics.frazil_heat_flux(esm)
            net_ocean_salinity = Diagnostics.net_ocean_salinity_flux(esm)
            sea_ice_ocean_salinity = Diagnostics.sea_ice_ocean_salinity_flux(esm)
            atmosphere_ocean_salinity = Diagnostics.atmosphere_ocean_salinity_flux(esm)

            for f in (frazil_temperature, net_ocean_temperature, sea_ice_ocean_temperature,
                      atmosphere_ocean_temperature, frazil_heat, net_ocean_heat, sea_ice_ocean_heat,
                      atmosphere_ocean_heat, net_ocean_salinity, sea_ice_ocean_salinity,
                      atmosphere_ocean_salinity, net_ocean_freshwater, sea_ice_ocean_freshwater,
                      atmosphere_ocean_freshwater)

                @test f isa Oceananigans.Fields.AbstractField

                if !(f isa Oceananigans.Fields.ZeroField)
                    @test Oceananigans.location(f) == (Center, Center, Nothing)
                end
                compute!(f)
            end

            @allowscalar begin
                @test frazil_heat[1, 1, 1] ≈ expected_frazil_heat
                @test net_ocean_heat[1, 1, 1] ≈ ρᵒᶜ * cᵒᶜ * (T_flux_value + additional_T_flux_value) + expected_frazil_heat
                @test sea_ice_ocean_heat[1, 1, 1] ≈ expected_frazil_heat + expected_interface_heat
                @test atmosphere_ocean_heat[1, 1, 1] ≈ ρᵒᶜ * cᵒᶜ * (T_flux_value + additional_T_flux_value) - expected_interface_heat
                @test net_ocean_heat[1, 1, 1] ≈ atmosphere_ocean_heat[1, 1, 1] + sea_ice_ocean_heat[1, 1, 1]

                @test frazil_temperature[1, 1, 1] ≈ 1 / (ρᵒᶜ * cᵒᶜ) * expected_frazil_heat
                @test net_ocean_temperature[1, 1, 1] ≈ T_flux_value + additional_T_flux_value + 1 / (ρᵒᶜ * cᵒᶜ) * expected_frazil_heat
                @test sea_ice_ocean_temperature[1, 1, 1] ≈ 1 / (ρᵒᶜ * cᵒᶜ) * (expected_frazil_heat + expected_interface_heat)
                @test atmosphere_ocean_temperature[1, 1, 1] ≈ T_flux_value + additional_T_flux_value - 1 / (ρᵒᶜ * cᵒᶜ) * expected_interface_heat
                @test net_ocean_temperature[1, 1, 1] ≈ atmosphere_ocean_temperature[1, 1, 1] + sea_ice_ocean_temperature[1, 1, 1]

                @test net_ocean_salinity[1, 1, 1] ≈ S_flux_value
                @test sea_ice_ocean_salinity[1, 1, 1] ≈ expected_sea_ice_ocean_salt
                @test atmosphere_ocean_salinity[1, 1, 1] ≈ S_flux_value - expected_sea_ice_ocean_salt
                @test net_ocean_salinity[1, 1, 1] ≈ atmosphere_ocean_salinity[1, 1, 1] + sea_ice_ocean_salinity[1, 1, 1]

                @test net_ocean_freshwater[1, 1, 1] ≈ - ρᵒᶜ / Sᵒᶜ * S_flux_value
                @test sea_ice_ocean_freshwater[1, 1, 1] ≈ - ρᵒᶜ / Sᵒᶜ * expected_sea_ice_ocean_salt
                @test atmosphere_ocean_freshwater[1, 1, 1] ≈ - ρᵒᶜ / Sᵒᶜ * (S_flux_value - expected_sea_ice_ocean_salt)
                @test net_ocean_freshwater[1, 1, 1] ≈ atmosphere_ocean_freshwater[1, 1, 1] + sea_ice_ocean_freshwater[1, 1, 1]
            end
        end
    end

    @testset "Global meridional heat transport closure [$A]" begin
        Nx, Ny, Nz = 4, 5, 2
        grid = LatitudeLongitudeGrid(arch;
                                     size = (Nx, Ny, Nz),
                                     longitude = (0, 360),
                                     latitude = (-90, 90),
                                     z = (-1, 0))

        ocean = ocean_simulation(grid;
                                 momentum_advection = nothing,
                                 tracer_advection = nothing,
                                 closure = nothing,
                                 coriolis = nothing)

        atmosphere = PrescribedAtmosphere(grid, [0.0])
        esm = OceanOnlyModel(ocean; atmosphere)
        simulation = Simulation(esm; Δt = 10)

        set!(ocean.model, v=1, T=1)

        mht = Field(meridional_heat_transport(simulation, MeridionalFluxMethod()))
        compute!(mht)

        callback_count = length(simulation.callbacks)
        @test isempty(temperature_budgets(simulation))

        meridional_heat_transport(simulation, TendencyMethod())
        @test length(simulation.callbacks) == callback_count + 1

        temperature_budget = only(temperature_budgets(simulation))
        @test simulation.callbacks[:temperature_budget].func === temperature_budget

        meridional_heat_transport(simulation)
        @test length(simulation.callbacks) == callback_count + 1
        @test only(temperature_budgets(simulation)) === temperature_budget

        simulation_with_budget = Simulation(esm; Δt = 10, stop_iteration = 1)
        registered_budget = BudgetComputation(:temperature, esm)
        add_callback!(simulation_with_budget, registered_budget; name = :custom_temperature_budget)
        callback_count = length(simulation_with_budget.callbacks)

        meridional_heat_transport(simulation_with_budget, TendencyMethod())
        @test length(simulation_with_budget.callbacks) == callback_count
        @test only(temperature_budgets(simulation_with_budget)) === registered_budget

        simulation_with_collision = Simulation(esm; Δt = 10, stop_iteration = 1)
        add_callback!(simulation_with_collision, simulation -> nothing; name = :temperature_budget)
        existing_callback = simulation_with_collision.callbacks[:temperature_budget]
        callback_count = length(simulation_with_collision.callbacks)

        meridional_heat_transport(simulation_with_collision, TendencyMethod())
        @test simulation_with_collision.callbacks[:temperature_budget] === existing_callback
        @test length(simulation_with_collision.callbacks) == callback_count + 1
        @test haskey(simulation_with_collision.callbacks, :temperature_budget_2)
        @test only(temperature_budgets(simulation_with_collision)) ===
              simulation_with_collision.callbacks[:temperature_budget_2].func

        simulation_with_duplicate_budgets = Simulation(esm; Δt = 10, stop_iteration = 1)
        add_callback!(simulation_with_duplicate_budgets, BudgetComputation(:temperature, esm);
                      name = :first_temperature_budget)
        add_callback!(simulation_with_duplicate_budgets, BudgetComputation(:temperature, esm);
                      name = :second_temperature_budget)
        callback_count = length(simulation_with_duplicate_budgets.callbacks)

        @test_throws ArgumentError meridional_heat_transport(simulation_with_duplicate_budgets,
                                                              TendencyMethod())
        @test length(simulation_with_duplicate_budgets.callbacks) == callback_count

        initialized_simulation = Simulation(esm; Δt = 10, stop_iteration = 1, verbose = false)
        Oceananigans.initialize!(initialized_simulation)
        meridional_heat_transport(initialized_simulation, TendencyMethod())
        initialized_budget = only(temperature_budgets(initialized_simulation))

        @test Array(interior(initialized_budget.previous_heat_content)) ==
              Array(interior(initialized_budget.heat_content))

        @allowscalar begin
            @test abs(mht[1, 1,      1]) < 1e8
            @test abs(mht[1, Ny + 1, 1]) < 1e8
        end
    end

    @testset "Tripolar tendency-based meridional heat transport [$A]" begin
        underlying_grid = TripolarGrid(arch;
                                       size = (32, 16, 2),
                                       z = MutableVerticalDiscretization((-100, 0)))

        tripolar_grid = analytical_immersed_grid(underlying_grid;
                                                          radius = 15,
                                                          active_cells_map = true)

        destination_grid = LatitudeLongitudeGrid(arch;
                                                  size = (8, 8, 1),
                                                  longitude = (0, 360),
                                                  latitude = (-90, 90),
                                                  z = (-1, 0))

        latlon_grid = analytical_immersed_grid(destination_grid;
                                                radius = 15,
                                                active_cells_map = true)

        # Use the same smooth initial state as the tracer-budget tests.
        Tᵢ(λ, φ, z) = 2 + 26 * cosd(φ)^2 * exp(z / 30)
        Sᵢ(λ, φ, z) = 35 - 1//2 * exp(z / 30)

        polar_ice_fraction(φ) = clamp((abs(φ) - 70) / 20, 0, 1)
        hᵢ(λ, φ) = 2 * polar_ice_fraction(φ)
        ℵᵢ(λ, φ) = polar_ice_fraction(φ)

        for timestepper in (:SplitRungeKutta3, :QuasiAdamsBashforth2)
            @testset "$(timestepper)" begin
                ocean = ocean_simulation(tripolar_grid;
                                         momentum_advection = nothing,
                                         tracer_advection = Centered(),
                                         closure = nothing,
                                         coriolis = nothing,
                                         timestepper)

                # Pending merging of PR#138 in ClimaSeaIce.jl: [CliMA/ClimaSeaIce.jl#138](https://github.com/CliMA/ClimaSeaIce.jl/pull/138)
                sea_ice = sea_ice_simulation(tripolar_grid, ocean; dynamics = nothing)
                set!(ocean.model, T=Tᵢ, S=Sᵢ)
                set!(sea_ice.model, h=hᵢ, ℵ=ℵᵢ)
                esm = OceanSeaIceModel(ocean, sea_ice)

                simulation = Simulation(esm; Δt = 10, stop_iteration = 4)
                callback_count = length(simulation.callbacks)
                @test isempty(temperature_budgets(simulation))
                @test_throws ArgumentError meridional_heat_transport(simulation, TendencyMethod())
                @test length(simulation.callbacks) == callback_count
                @test isempty(temperature_budgets(simulation))

                mht = meridional_heat_transport(simulation; destination_grid = latlon_grid)
                @test length(simulation.callbacks) == callback_count + 1

                budget = only(temperature_budgets(simulation))

                ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
                cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
                raw_temperature_flux = Diagnostics.flux_field(ocean.model.tracers.T.boundary_conditions.top.condition)
                sea_ice_ocean_fluxes = esm.interfaces.sea_ice_ocean_interface.fluxes

                applied_surface_flux = budget.residual - budget.tendency + budget.applied_radiative_heat_flux
                regridded_residual = RegriddedOperation(budget.residual, latlon_grid)

                meridional_heat_transport(simulation; destination_grid = latlon_grid)
                @test length(simulation.callbacks) == callback_count + 1

                mktempdir() do dir
                    iteration_filename = joinpath(dir, "iteration_mht.jld2")
                    averaged_filename = joinpath(dir, "averaged_mht.jld2")
                    outputs = (; mht)

                    simulation.output_writers[:iteration_mht] = JLD2Writer(simulation.model, outputs;
                                                                           schedule = IterationInterval(1),
                                                                           filename = iteration_filename,
                                                                           overwrite_existing = true)

                    simulation.output_writers[:averaged_mht] = JLD2Writer(simulation.model, outputs;
                                                                          schedule = AveragedTimeInterval(40),
                                                                          filename = averaged_filename,
                                                                          overwrite_existing = true)

                    run!(simulation)

                    iteration_mht = FieldTimeSeries(iteration_filename, "mht")
                    averaged_mht = FieldTimeSeries(averaged_filename, "mht")

                    completed_iterations = findall(t -> 0 < t <= 40, iteration_mht.times)
                    @test iteration_mht.times[completed_iterations] == [10, 20, 30, 40]
                    iteration_values = [Array(interior(iteration_mht[n])) for n in completed_iterations]

                    # The final latitude contains the cumulative global heat budget.
                    # It should be tiny compared with the strongest interior transport.
                    @test all(values -> all(isfinite, values), iteration_values)
                    peak_transport = maximum(maximum(abs, values) for values in iteration_values)
                    @test peak_transport > 1e8

                    closure_tolerance = max(5e9, 5e-6 * peak_transport)
                    for values in iteration_values
                        southern_boundary = values[1, 1, 1]
                        northern_boundary = values[1, size(values, 2), 1]
                        @test abs(southern_boundary) < closure_tolerance
                        @test abs(northern_boundary) < closure_tolerance
                    end

                    # Time-dependent forcing and tracer evolution should produce
                    # new public MHT fields rather than one repeated cached value.
                    changes = [maximum(abs, iteration_values[n] - iteration_values[n-1])
                               for n in 2:length(iteration_values)]
                    @test any(>(1e8), changes)

                    expected_average = sum(iteration_values) / length(iteration_values)
                    full_average = findfirst(==(40), averaged_mht.times)
                    @test full_average !== nothing
                    averaged_values = Array(interior(averaged_mht[full_average]))
                    averaging_rtol = 10sqrt(eps(eltype(averaged_values)))
                    @test all(isapprox.(averaged_values, expected_average; rtol = averaging_rtol))
                end
            end
        end
    end
end
