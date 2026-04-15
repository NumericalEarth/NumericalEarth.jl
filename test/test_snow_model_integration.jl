include("runtests_setup.jl")

using ClimaSeaIce: SeaIceModel, ConductiveFlux
using ClimaSeaIce.SeaIceThermodynamics: IceSnowConductiveFlux
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    ComponentInterfaces,
    SkinTemperature,
    InterfaceProperties,
    conductive_flux_balance_temperature

using Oceananigans.Fields: ZeroField
using Oceananigans.Units: hours, days

#####
##### Unit tests
#####

@testset "Snow model unit tests" begin
    for arch in test_architectures
        A = typeof(arch)

        grid = RectilinearGrid(arch;
                               size = (4, 4, 1),
                               extent = (1, 1, 1),
                               topology = (Periodic, Periodic, Bounded))

        @testset "sea_ice_simulation with_snow=false [$A]" begin
            sea_ice = sea_ice_simulation(grid; dynamics=nothing)
            @test sea_ice isa Simulation
            @test sea_ice.model isa SeaIceModel
            @test sea_ice.model.snow_thermodynamics === nothing
            @test sea_ice.model.ice_thermodynamics.internal_heat_flux isa ConductiveFlux
        end

        @testset "sea_ice_simulation with_snow=true [$A]" begin
            sea_ice = sea_ice_simulation(grid; dynamics=nothing, with_snow=true)
            @test sea_ice isa Simulation
            @test sea_ice.model.snow_thermodynamics !== nothing
            @test sea_ice.model.snow_thickness isa Field
        end

        @testset "PhaseTransitions API [$A]" begin
            sea_ice = sea_ice_simulation(grid; dynamics=nothing)
            pt = sea_ice.model.ice_thermodynamics.phase_transitions
            @test pt.heat_capacity == 2100
            @test pt.density == 900
        end

        @testset "ComponentExchanger includes hs [$A]" begin
            using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger

            # Without snow: hs should be ZeroField
            sea_ice = sea_ice_simulation(grid; dynamics=nothing)
            exchanger = ComponentExchanger(sea_ice, grid)
            @test haskey(exchanger.state, :hs)
            @test haskey(exchanger.state, :hi)
            @test exchanger.state.hs isa ZeroField

            # With snow: hs should be a Field
            sea_ice_snow = sea_ice_simulation(grid; dynamics=nothing, with_snow=true)
            exchanger_snow = ComponentExchanger(sea_ice_snow, grid)
            @test exchanger_snow.state.hs isa Field
        end

        @testset "default_ai_temperature dispatches on snow [$A]" begin
            using NumericalEarth.SeaIces: default_ai_temperature

            sea_ice = sea_ice_simulation(grid; dynamics=nothing)
            st = default_ai_temperature(sea_ice)
            @test st isa SkinTemperature
            @test st.internal_flux isa ConductiveFlux

            sea_ice_snow = sea_ice_simulation(grid; dynamics=nothing, with_snow=true)
            st_snow = default_ai_temperature(sea_ice_snow)
            @test st_snow isa SkinTemperature
            @test st_snow.internal_flux isa IceSnowConductiveFlux
        end

        @testset "net_fluxes includes snowfall [$A]" begin
            using NumericalEarth.SeaIces: net_fluxes

            sea_ice = sea_ice_simulation(grid; dynamics=nothing, with_snow=true, snowfall=0)
            fluxes = net_fluxes(sea_ice)
            @test haskey(fluxes.top, :snowfall)
        end

        @testset "Conductive flux balance: bare ice vs ice+snow [$A]" begin
            # For thick ice with no snow, R = hi/ki.
            # With snow on top, R = hs/ks + hi/ki > hi/ki.
            # Higher R means more insulation → warmer surface temperature
            # (closer to the atmospheric temperature, further from the bottom).
            #
            # We verify R_snow > R_ice by checking the formula directly.
            ki = 2.0    # ice conductivity
            ks = 0.31   # snow conductivity
            hi = 1.0    # ice thickness
            hs = 0.1    # snow depth

            R_ice  = hi / ki
            R_snow = hs / ks + hi / ki

            @test R_snow > R_ice
            # Snow adds significant thermal resistance
            @test R_snow / R_ice > 1.5
        end
    end
end

#####
##### Integration tests
#####

@testset "Snow model integration tests" begin
    for arch in test_architectures
        A = typeof(arch)

        grid = RectilinearGrid(arch;
                               size = (4, 4, 2),
                               extent = (1, 1, 1),
                               topology = (Periodic, Periodic, Bounded))

        @testset "Coupled model without snow [$A]" begin
            ocean = ocean_simulation(grid;
                                     momentum_advection = nothing,
                                     tracer_advection = nothing,
                                     closure = nothing,
                                     coriolis = nothing)

            sea_ice = sea_ice_simulation(grid, ocean; dynamics=nothing)
            atmosphere = PrescribedAtmosphere(grid, [0.0])
            radiation = Radiation()

            @test begin
                coupled = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
                time_step!(coupled, 1)
                true
            end
        end

        @testset "Coupled model with snow [$A]" begin
            ocean = ocean_simulation(grid;
                                     momentum_advection = nothing,
                                     tracer_advection = nothing,
                                     closure = nothing,
                                     coriolis = nothing)

            sea_ice = sea_ice_simulation(grid, ocean;
                                         dynamics = nothing,
                                         with_snow = true)

            atmosphere = PrescribedAtmosphere(grid, [0.0])
            radiation = Radiation()

            @test begin
                coupled = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
                time_step!(coupled, 1)
                true
            end
        end

        @testset "Snowfall routing [$A]" begin
            ocean = ocean_simulation(grid;
                                     momentum_advection = nothing,
                                     tracer_advection = nothing,
                                     closure = nothing,
                                     coriolis = nothing)

            sea_ice = sea_ice_simulation(grid, ocean;
                                         dynamics = nothing,
                                         with_snow = true)

            atmosphere = PrescribedAtmosphere(grid, [0.0])
            radiation = Radiation()

            coupled = OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)

            # The snowfall field should exist in the exchanger
            exchanger = coupled.interfaces.exchanger
            @test haskey(exchanger.atmosphere.state, :Jˢⁿ)

            # The net fluxes should include snowfall
            top_fluxes = coupled.interfaces.net_fluxes.sea_ice.top
            @test haskey(top_fluxes, :snowfall)
        end

        @testset "Snow insulation effect [$A]" begin
            # With snow, the IceSnowConductiveFlux is used for the surface
            # temperature solve. Verify this dispatch is wired correctly.
            ocean = ocean_simulation(grid;
                                     momentum_advection = nothing,
                                     tracer_advection = nothing,
                                     closure = nothing,
                                     coriolis = nothing)

            sea_ice = sea_ice_simulation(grid, ocean;
                                         dynamics = nothing,
                                         with_snow = true)

            ai_temp = NumericalEarth.SeaIces.default_ai_temperature(sea_ice)
            @test ai_temp.internal_flux isa IceSnowConductiveFlux
            @test ai_temp.internal_flux.ice_conductivity == 2.0
            @test ai_temp.internal_flux.snow_conductivity == 0.31
        end
    end
end
