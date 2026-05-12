include("runtests_setup.jl")

using Oceananigans: location
using Oceananigans.Models: buoyancy_operation
using NumericalEarth.Diagnostics: MixedLayerDepthField, MixedLayerDepthOperand
using SeawaterPolynomials: TEOS10EquationOfState

for arch in test_architectures
    A = typeof(arch)
    @info "Testing InterfaceFluxOutputs on $A"

    @testset "InterfaceFluxOutputs on $A" begin
        grid = RectilinearGrid(arch;
                               size = (4, 5, 2),
                               extent = (1, 1, 1),
                               topology = (Periodic, Bounded, Bounded))

        ocean = ocean_simulation(grid;
                                 momentum_advection = nothing,
                                 tracer_advection = nothing,
                                 closure = nothing,
                                 coriolis = nothing)

        sea_ice = sea_ice_simulation(grid, ocean)
        atmosphere = PrescribedAtmosphere(grid, [0.0])
        esm = OceanSeaIceModel(sea_ice, ocean; atmosphere)

        T_flux = ocean.model.tracers.T.boundary_conditions.top.condition
        S_flux = ocean.model.tracers.S.boundary_conditions.top.condition
        sea_ice_ocean_fluxes = esm.interfaces.sea_ice_ocean_interface.fluxes

        T_flux_value = 2.0
        S_flux_value = 5.0
        frazil_heat_flux_value = 0.2
        interface_heat_flux_value = 0.3
        sea_ice_ocean_salt_flux_value = 0.9

        fill!(T_flux, T_flux_value)
        fill!(S_flux, S_flux_value)
        fill!(sea_ice_ocean_fluxes.frazil_heat, frazil_heat_flux_value)
        fill!(sea_ice_ocean_fluxes.interface_heat, interface_heat_flux_value)
        fill!(sea_ice_ocean_fluxes.salt, sea_ice_ocean_salt_flux_value)

        ρᵒᶜ = esm.interfaces.ocean_properties.reference_density
        cᵒᶜ = esm.interfaces.ocean_properties.heat_capacity
        S₀ = 35.0

        frazil_temperature = frazil_temperature_flux(esm)
        net_ocean_temperature = net_ocean_temperature_flux(esm)
        sea_ice_ocean_temperature = sea_ice_ocean_temperature_flux(esm)
        atmosphere_ocean_temperature = atmosphere_ocean_temperature_flux(esm)
        frazil_heat = frazil_heat_flux(esm)
        net_ocean_heat = net_ocean_heat_flux(esm)
        sea_ice_ocean_heat = sea_ice_ocean_heat_flux(esm)
        atmosphere_ocean_heat = atmosphere_ocean_heat_flux(esm)
        net_ocean_salinity = net_ocean_salinity_flux(esm)
        sea_ice_ocean_salinity = sea_ice_ocean_salinity_flux(esm)
        atmosphere_ocean_salinity = atmosphere_ocean_salinity_flux(esm)
        net_ocean_freshwater = net_ocean_freshwater_flux(esm; reference_salinity = 35)
        sea_ice_ocean_freshwater = sea_ice_ocean_freshwater_flux(esm; reference_salinity = 35)
        atmosphere_ocean_freshwater = atmosphere_ocean_freshwater_flux(esm; reference_salinity = 35)

        for f in (frazil_temperature, net_ocean_temperature, sea_ice_ocean_temperature,
                  atmosphere_ocean_temperature, frazil_heat, net_ocean_heat, sea_ice_ocean_heat,
                  atmosphere_ocean_heat, net_ocean_salinity, sea_ice_ocean_salinity,
                  atmosphere_ocean_salinity, net_ocean_freshwater, sea_ice_ocean_freshwater,
                  atmosphere_ocean_freshwater)

            @test f isa Field
            @test location(f) == (Center, Center, Nothing)
            compute!(f)
        end

        @allowscalar begin
            @test net_ocean_heat[1, 1, 1] ≈ ρᵒᶜ * cᵒᶜ * T_flux_value + frazil_heat_flux_value
            @test atmosphere_ocean_heat[1, 1, 1] ≈ ρᵒᶜ * cᵒᶜ * T_flux_value - interface_heat_flux_value
            @test sea_ice_ocean_heat[1, 1, 1] ≈ frazil_heat_flux_value + interface_heat_flux_value
            @test net_ocean_heat[1, 1, 1] ≈ atmosphere_ocean_heat[1, 1, 1] + sea_ice_ocean_heat[1, 1, 1]

            @test net_ocean_freshwater[1, 1, 1] ≈ - ρᵒᶜ / S₀ * S_flux_value
            @test sea_ice_ocean_freshwater[1, 1, 1] ≈ - ρᵒᶜ / S₀ * sea_ice_ocean_salt_flux_value
            @test atmosphere_ocean_freshwater[1, 1, 1] ≈ - ρᵒᶜ / S₀ * (S_flux_value - sea_ice_ocean_salt_flux_value)
            @test net_ocean_freshwater[1, 1, 1] ≈ atmosphere_ocean_freshwater[1, 1, 1] + sea_ice_ocean_freshwater[1, 1, 1]

            @test net_ocean_temperature[1, 1, 1] ≈ T_flux_value + 1 / (ρᵒᶜ * cᵒᶜ) * frazil_heat_flux_value
            @test atmosphere_ocean_temperature[1, 1, 1] ≈ T_flux_value - 1 / (ρᵒᶜ * cᵒᶜ) * interface_heat_flux_value
            @test sea_ice_ocean_temperature[1, 1, 1] ≈ 1 / (ρᵒᶜ * cᵒᶜ) * (frazil_heat_flux_value + interface_heat_flux_value)
            @test net_ocean_temperature[1, 1, 1] ≈ atmosphere_ocean_temperature[1, 1, 1] + sea_ice_ocean_temperature[1, 1, 1]

            @test net_ocean_freshwater[1, 1, 1] ≈ - ρᵒᶜ / S₀ * S_flux_value
            @test sea_ice_ocean_freshwater[1, 1, 1] ≈ - ρᵒᶜ / S₀ * sea_ice_ocean_salt_flux_value
            @test atmosphere_ocean_freshwater[1, 1, 1] ≈ - ρᵒᶜ / S₀ * (S_flux_value - sea_ice_ocean_salt_flux_value)
            @test net_ocean_freshwater[1, 1, 1] ≈ atmosphere_ocean_freshwater[1, 1, 1] + sea_ice_ocean_freshwater[1, 1, 1]
        end

        ocean_only_ocean = ocean_simulation(grid;
                                            momentum_advection = nothing,
                                            tracer_advection = nothing,
                                            closure = nothing,
                                            coriolis = nothing)

        ocean_only_atmosphere = PrescribedAtmosphere(grid, [0.0])
        ocean_only_esm = OceanOnlyModel(ocean_only_ocean; atmosphere = ocean_only_atmosphere)

        ocean_only_T_flux = ocean_only_ocean.model.tracers.T.boundary_conditions.top.condition
        ocean_only_S_flux = ocean_only_ocean.model.tracers.S.boundary_conditions.top.condition

        fill!(ocean_only_T_flux, T_flux_value)
        fill!(ocean_only_S_flux, S_flux_value)

        ρᵒᶜ = ocean_only_esm.interfaces.ocean_properties.reference_density
        cᵒᶜ = ocean_only_esm.interfaces.ocean_properties.heat_capacity

        ocean_only_frazil_temperature = frazil_temperature_flux(ocean_only_esm)
        ocean_only_net_ocean_temperature = net_ocean_temperature_flux(ocean_only_esm)
        ocean_only_sea_ice_ocean_temperature = sea_ice_ocean_temperature_flux(ocean_only_esm)
        ocean_only_atmosphere_ocean_temperature = atmosphere_ocean_temperature_flux(ocean_only_esm)
        ocean_only_frazil_heat = frazil_heat_flux(ocean_only_esm)
        ocean_only_net_ocean_heat = net_ocean_heat_flux(ocean_only_esm)
        ocean_only_sea_ice_ocean_heat = sea_ice_ocean_heat_flux(ocean_only_esm)
        ocean_only_atmosphere_ocean_heat = atmosphere_ocean_heat_flux(ocean_only_esm)
        ocean_only_net_ocean_salinity = net_ocean_salinity_flux(ocean_only_esm)
        ocean_only_sea_ice_ocean_salinity = sea_ice_ocean_salinity_flux(ocean_only_esm)
        ocean_only_atmosphere_ocean_salinity = atmosphere_ocean_salinity_flux(ocean_only_esm)
        ocean_only_net_ocean_freshwater = net_ocean_freshwater_flux(ocean_only_esm; reference_salinity = 35)
        ocean_only_sea_ice_ocean_freshwater = sea_ice_ocean_freshwater_flux(ocean_only_esm; reference_salinity = 35)
        ocean_only_atmosphere_ocean_freshwater = atmosphere_ocean_freshwater_flux(ocean_only_esm; reference_salinity = 35)

        for f in (ocean_only_frazil_temperature, ocean_only_net_ocean_temperature,
                  ocean_only_sea_ice_ocean_temperature, ocean_only_atmosphere_ocean_temperature,
                  ocean_only_frazil_heat, ocean_only_net_ocean_heat, ocean_only_sea_ice_ocean_heat,
                  ocean_only_atmosphere_ocean_heat, ocean_only_net_ocean_salinity,
                  ocean_only_sea_ice_ocean_salinity, ocean_only_atmosphere_ocean_salinity,
                  ocean_only_net_ocean_freshwater, ocean_only_sea_ice_ocean_freshwater,
                  ocean_only_atmosphere_ocean_freshwater)

            @test f isa Field
            @test location(f) == (Center, Center, Nothing)
            compute!(f)
        end

        @allowscalar begin
            @test ocean_only_frazil_heat[1, 1, 1] ≈ 0
            @test ocean_only_sea_ice_ocean_heat[1, 1, 1] ≈ 0
            @test ocean_only_net_ocean_heat[1, 1, 1] ≈ ρᵒᶜ * cᵒᶜ * T_flux_value
            @test ocean_only_atmosphere_ocean_heat[1, 1, 1] ≈ ocean_only_net_ocean_heat[1, 1, 1]

            @test ocean_only_frazil_temperature[1, 1, 1] ≈ 0
            @test ocean_only_sea_ice_ocean_temperature[1, 1, 1] ≈ 0
            @test ocean_only_net_ocean_temperature[1, 1, 1] ≈ T_flux_value
            @test ocean_only_atmosphere_ocean_temperature[1, 1, 1] ≈ ocean_only_net_ocean_temperature[1, 1, 1]

            @test ocean_only_sea_ice_ocean_salinity[1, 1, 1] ≈ 0
            @test ocean_only_net_ocean_salinity[1, 1, 1] ≈ S_flux_value
            @test ocean_only_atmosphere_ocean_salinity[1, 1, 1] ≈ ocean_only_net_ocean_salinity[1, 1, 1]

            @test ocean_only_sea_ice_ocean_freshwater[1, 1, 1] ≈ 0
            @test ocean_only_net_ocean_freshwater[1, 1, 1] ≈ - ρᵒᶜ / S₀ * S_flux_value
            @test ocean_only_atmosphere_ocean_freshwater[1, 1, 1] ≈ ocean_only_net_ocean_freshwater[1, 1, 1]
        end
    end
end
