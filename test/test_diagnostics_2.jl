include("runtests_setup.jl")

using NumericalEarth.Diagnostics: Diagnostics

for arch in test_architectures
    A = typeof(arch)
    @info "Testing InterfaceFluxOutputs [$A]..."

    grid = RectilinearGrid(arch;
                            size = (4, 5, 2),
                            extent = (1, 1, 1),
                            topology = (Periodic, Bounded, Bounded))

    T_flux_value = 2.0
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
                                     coriolis = nothing)
            atmosphere = PrescribedAtmosphere(grid, [0.0])
            esm = if config.with_sea_ice
                sea_ice = sea_ice_simulation(grid, ocean)
                OceanSeaIceModel(ocean, sea_ice; atmosphere)
            else
                OceanOnlyModel(ocean; atmosphere)
            end

            T_flux = ocean.model.tracers.T.boundary_conditions.top.condition
            S_flux = ocean.model.tracers.S.boundary_conditions.top.condition

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
                @test net_ocean_heat[1, 1, 1] ≈ ρᵒᶜ * cᵒᶜ * T_flux_value + expected_frazil_heat
                @test sea_ice_ocean_heat[1, 1, 1] ≈ expected_frazil_heat + expected_interface_heat
                @test atmosphere_ocean_heat[1, 1, 1] ≈ ρᵒᶜ * cᵒᶜ * T_flux_value - expected_interface_heat
                @test net_ocean_heat[1, 1, 1] ≈ atmosphere_ocean_heat[1, 1, 1] + sea_ice_ocean_heat[1, 1, 1]

                @test frazil_temperature[1, 1, 1] ≈ 1 / (ρᵒᶜ * cᵒᶜ) * expected_frazil_heat
                @test net_ocean_temperature[1, 1, 1] ≈ T_flux_value + 1 / (ρᵒᶜ * cᵒᶜ) * expected_frazil_heat
                @test sea_ice_ocean_temperature[1, 1, 1] ≈ 1 / (ρᵒᶜ * cᵒᶜ) * (expected_frazil_heat + expected_interface_heat)
                @test atmosphere_ocean_temperature[1, 1, 1] ≈ T_flux_value - 1 / (ρᵒᶜ * cᵒᶜ) * expected_interface_heat
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

        set!(ocean.model, v=1, T=1)

        mht = Field(meridional_heat_transport(esm))
        compute!(mht)

        @allowscalar begin
            @show mht[1, 1,      1], mht[1, Ny + 1, 1]
            @test iszero(mht[1, 1,      1])
            @test iszero(mht[1, Ny + 1, 1])
        end
    end

    @testset "Tripolar tendency-based meridional heat transport [$A]" begin
        tripolar_grid = TripolarGrid(arch;
                                     size = (8, 4, 2),
                                     z = (-1, 0))

        destination_grid = LatitudeLongitudeGrid(arch;
                                                  size = (8, 4, 1),
                                                  longitude = (0, 360),
                                                  latitude = (-90, 90),
                                                  z = (-1, 0))

        ocean = ocean_simulation(tripolar_grid;
                                 momentum_advection = nothing,
                                 tracer_advection = nothing,
                                 closure = nothing,
                                 coriolis = nothing)

        atmosphere = PrescribedAtmosphere(tripolar_grid, [0.0])
        esm = OceanOnlyModel(ocean; atmosphere)

        mht = Field(meridional_heat_transport(esm, TendencyMethod(); destination_grid))
        @test all(iszero, interior(mht))

        budget = esm.interfaces.budgets.ocean_heat
        @test budget !== nothing

        time_step!(esm, 1)
        compute!(mht)
        @test all(iszero, interior(mht))

        # The same operation must conservatively remap newly completed budgets;
        # reconstructing the diagnostic between output samples is not required.
        set!(budget.residual, 1)
        compute!(mht)
        unit_budget_mht = Array(interior(mht))

        set!(budget.residual, 2)
        compute!(mht)
        double_budget_mht = Array(interior(mht))

        @test any(x -> !iszero(x), unit_budget_mht)
        @test all(isapprox.(double_budget_mht, 2 .* unit_budget_mht))
    end
end
