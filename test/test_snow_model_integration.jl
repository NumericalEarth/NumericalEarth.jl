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
            sea_ice = sea_ice_simulation(grid; dynamics=nothing, snow_thermodynamics=nothing)
            @test sea_ice isa Simulation
            @test sea_ice.model isa SeaIceModel
            @test sea_ice.model.snow_thermodynamics === nothing
            @test sea_ice.model.ice_thermodynamics.internal_heat_flux isa ConductiveFlux
        end

        @testset "sea_ice_simulation with_snow=true [$A]" begin
            sea_ice = sea_ice_simulation(grid; dynamics=nothing)
            @test sea_ice isa Simulation
            @test sea_ice.model.snow_thermodynamics !== nothing
            @test sea_ice.model.snow_thickness isa Field
        end

        @testset "PhaseTransitions API [$A]" begin
            sea_ice = sea_ice_simulation(grid; dynamics=nothing)
            pt = sea_ice.model.phase_transitions
            @test pt.heat_capacity == 2100
            @test pt.density == 900
        end

        @testset "ComponentExchanger includes hs [$A]" begin
            using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentExchanger

            # Without snow: hs should be ZeroField
            sea_ice = sea_ice_simulation(grid; dynamics=nothing, snow_thermodynamics=nothing)
            exchanger = ComponentExchanger(sea_ice, grid)
            @test haskey(exchanger.state, :hs)
            @test haskey(exchanger.state, :hi)
            @test exchanger.state.hs isa ZeroField

            # With snow: hs should be a Field
            sea_ice_snow = sea_ice_simulation(grid; dynamics=nothing)
            exchanger_snow = ComponentExchanger(sea_ice_snow, grid)
            @test exchanger_snow.state.hs isa Field
        end

        @testset "default_ai_temperature dispatches on snow [$A]" begin
            using NumericalEarth.SeaIces: default_ai_temperature

            sea_ice = sea_ice_simulation(grid; dynamics=nothing, snow_thermodynamics=nothing)
            st = default_ai_temperature(sea_ice)
            @test st isa SkinTemperature
            @test st.internal_flux isa ConductiveFlux

            sea_ice_snow = sea_ice_simulation(grid; dynamics=nothing)
            st_snow = default_ai_temperature(sea_ice_snow)
            @test st_snow isa SkinTemperature
            @test st_snow.internal_flux isa IceSnowConductiveFlux
        end

        @testset "net_fluxes includes snowfall [$A]" begin
            using NumericalEarth.SeaIces: net_fluxes

            sea_ice = sea_ice_simulation(grid; dynamics=nothing)
            fluxes = net_fluxes(sea_ice)
            @test haskey(fluxes.top, :snowfall)
        end

        @testset "Snow insulates: warmer surface temperature [$A]" begin
            # Build two coupled models — one without snow, one with snow and
            # nonzero snow thickness — then compare surface temperatures after
            # one coupled time step. Snow adds thermal resistance, so the
            # surface should be warmer (closer to the warmer atmosphere).
            #
            # Radiation is disabled (ε=0) so the surface energy balance
            # reduces to conductive + turbulent fluxes; otherwise the
            # Stefan–Boltzmann loss swamps the small snow-insulation signal.
            ocean_grid = RectilinearGrid(arch;
                                         size = (1, 1, 2),
                                         extent = (1, 1, 1),
                                         topology = (Periodic, Periodic, Bounded))

            function build_coupled(; with_snow)
                ocean = ocean_simulation(ocean_grid;
                                         momentum_advection = nothing,
                                         tracer_advection = nothing,
                                         closure = nothing,
                                         coriolis = nothing)
                set!(ocean.model, T = -1.8, S = 34)

                snow_thermodynamics = with_snow ? default_snow_thermodynamics(ocean_grid) : nothing
                sea_ice = sea_ice_simulation(ocean_grid, ocean;
                                             dynamics = nothing,
                                             snow_thermodynamics)
                set!(sea_ice.model, h = 1.0, ℵ = 1.0)
                if with_snow
                    set!(sea_ice.model, hs = 0.2)
                end

                atmosphere = PrescribedAtmosphere(ocean_grid, [0.0])
                parent(atmosphere.velocities.u) .= 2.0
                radiation = Radiation(ocean_emissivity = 0, sea_ice_emissivity = 0)
                return OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation)
            end

            bare  = build_coupled(with_snow = false)
            snowy = build_coupled(with_snow = true)

            time_step!(bare,  1)
            time_step!(snowy, 1)

            Ts_bare  = bare.interfaces.atmosphere_sea_ice_interface.temperature
            Ts_snowy = snowy.interfaces.atmosphere_sea_ice_interface.temperature

            @allowscalar begin
                # Snow insulation → warmer (or equal) surface temperature
                @test Ts_snowy[1, 1, 1] ≥ Ts_bare[1, 1, 1]
            end
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

        @testset "Coupled model with snow [$A]" begin
            ocean = ocean_simulation(grid;
                                     momentum_advection = nothing,
                                     tracer_advection = nothing,
                                     closure = nothing,
                                     coriolis = nothing)

            sea_ice = sea_ice_simulation(grid, ocean;
                                         dynamics = nothing)

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
                                         snow_thermodynamics = nothing)

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
                                         dynamics = nothing)

            ai_temp = NumericalEarth.SeaIces.default_ai_temperature(sea_ice)
            @test ai_temp.internal_flux isa IceSnowConductiveFlux
            @test ai_temp.internal_flux.ice_conductivity == 2.0
            @test ai_temp.internal_flux.snow_conductivity == 0.31
        end
    end
end
