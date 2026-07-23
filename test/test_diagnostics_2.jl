include("runtests_setup.jl")

using NumericalEarth.Diagnostics: Diagnostics

struct ConstantAdditionalTemperatureFlux{T}
    value :: T
end

@inline (flux::ConstantAdditionalTemperatureFlux)(i, j, grid, clock, fields) = flux.value

# TEMPORARY DEBUGGING: remove these helpers and the callback below after the
# fixed-grid top-flux NaN has been identified and fixed.
@inline function debug_top_area(i, j, k, grid)
    return Oceananigans.Operators.Azᶜᶜᶠ(i, j, grid.Nz + 1, grid)
end

@inline function debug_raw_top_temperature_flux(i, j, k, grid, advection, fields)
    return Oceananigans.Advection._advective_tracer_flux_z(i, j, grid.Nz + 1,
                                                            grid, advection,
                                                            fields.w, fields.T)
end

function report_nonfinite_top_flux!(simulation, debug)
    compute!(debug.area)
    compute!(debug.raw_flux)
    compute!(debug.top_flux)

    area = Array(interior(debug.area))
    raw_flux = Array(interior(debug.raw_flux))
    top_flux = Array(interior(debug.top_flux))
    w_top = Array(interior(debug.w))[:, :, end]
    T_top = Array(interior(debug.T))[:, :, end]

    bad_cells = findall(x -> !isfinite(x), top_flux)
    isempty(bad_cells) && return nothing

    samples = map(first(bad_cells, min(8, length(bad_cells)))) do index
        i, j, _ = Tuple(index)
        return (; index=(i, j),
                area=area[index],
                raw_flux=raw_flux[index],
                w=w_top[i, j],
                T=T_top[i, j],
                top_flux=top_flux[index])
    end

    @info "TEMPORARY top-flux NaN diagnostics" iteration=simulation.model.clock.iteration nonfinite_count=length(bad_cells) samples
    return nothing
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

        @allowscalar begin
            @test abs(mht[1, 1,      1]) < 1e8
            @test abs(mht[1, Ny + 1, 1]) < 1e8
        end
    end

    @testset "Tripolar tendency-based meridional heat transport [$A]" begin
        tripolar_grid = TripolarGrid(arch;
                                     size = (32, 16, 2),
                                     z = (-1, 0))

        destination_grid = LatitudeLongitudeGrid(arch;
                                                  size = (8, 8, 1),
                                                  longitude = (0, 360),
                                                  latitude = (-90, 90),
                                                  z = (-1, 0))

        for timestepper in (:SplitRungeKutta3, :QuasiAdamsBashforth2)
            @testset "$(timestepper) flux timing" begin
                ocean = ocean_simulation(tripolar_grid;
                                         momentum_advection = nothing,
                                         tracer_advection = Centered(),
                                         closure = nothing,
                                         coriolis = nothing,
                                         timestepper)

                Tᵢ(λ, φ, z) = 10 + φ / 90
                set!(ocean.model, v = 1, T = Tᵢ, S = 35)
                esm = OceanOnlyModel(ocean)

                # Change the surface heat flux after each completed timestep. This
                # makes an AB2 cache that is one step early or late visible in the
                # global MHT closure.
                temperature_flux = Diagnostics.flux_field(ocean.model.tracers.T.boundary_conditions.top.condition)

                function update_temperature_flux!(simulation, flux)
                    next_iteration = simulation.model.clock.iteration + 1
                    fill!(flux, next_iteration * 1e-6)
                    return nothing
                end

                simulation = Simulation(esm; Δt = 10, stop_iteration = 4)
                budget = BudgetComputation(:temperature, esm)
                @test_throws ArgumentError meridional_heat_transport(simulation; destination_grid)
                add_callback!(simulation, budget)
                @test_throws MethodError meridional_heat_transport(budget, TendencyMethod(); destination_grid)

                debug_grid = ocean.model.grid
                debug_fields = (; w=ocean.model.transport_velocities.w,
                                T=budget.stage_temperature)
                area_operation = Oceananigans.AbstractOperations.KernelFunctionOperation{Center, Center, Nothing}(debug_top_area,
                                                                                                                  debug_grid)
                raw_flux_operation = Oceananigans.AbstractOperations.KernelFunctionOperation{Center, Center, Nothing}(debug_raw_top_temperature_flux,
                                                                                                                      debug_grid,
                                                                                                                      ocean.model.advection.T,
                                                                                                                      debug_fields)
                debug = (; area=Field(area_operation),
                         raw_flux=Field(raw_flux_operation),
                         top_flux=Field(Diagnostics.ocean_top_advective_temperature_flux(esm, budget.stage_temperature)),
                         debug_fields...)
                debug_callback = Callback(report_nonfinite_top_flux!, IterationInterval(1);
                                          parameters=debug)
                add_callback!(simulation, debug_callback; name=:temporary_top_flux_debug)

                flux_callback = Callback(update_temperature_flux!, IterationInterval(1);
                                         parameters=temperature_flux)
                add_callback!(simulation, flux_callback; name=:update_test_temperature_flux)

                mht = Field(meridional_heat_transport(simulation; destination_grid))

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

                    closure_tolerance = max(1e8, 1e-6 * peak_transport)
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

                late_budget = BudgetComputation(:temperature, esm)
                @test_logs (:warn, r"cannot be attached after the simulation has already started") begin
                    @test_throws ArgumentError add_callback!(simulation, late_budget; name=:late_budget)
                end
            end
        end
    end
end
