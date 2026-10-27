include("runtests_setup.jl")

using ClimaSeaIce: SeaIceModel, ConductiveFlux
using ClimaSeaIce.SeaIceThermodynamics: IceSnowConductiveFlux, UniformThicknessDistribution, GammaThicknessDistribution,
                                        thermal_resistance, effective_conductivity
using Oceananigans.Fields: ZeroField
using Oceananigans.Units: hours, days
using NumericalEarth.EarthSystemModels.InterfaceComputations:
    default_ai_temperature,
    net_fluxes,
    ComponentExchanger,
    ComponentInterfaces,
    SkinTemperature,
    InterfaceProperties,
    conductive_flux_balance_temperature
using NumericalEarth.SeaIces: default_snow_thermodynamics, ThicknessDependentConductivity

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
            sea_ice = sea_ice_simulation(grid; dynamics=nothing)
            fluxes = net_fluxes(sea_ice)
            @test haskey(fluxes.top, :snowfall)
        end

        @testset "Snow insulates: warmer surface temperature [$A]" begin
            # Snow adds thermal resistance, so the surface ends up warmer after one coupled step.
            # Radiation is disabled (ε=0) so the energy balance reduces to conductive + turbulent
            # fluxes; otherwise the Stefan-Boltzmann loss swamps the snow-insulation signal.
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
                return OceanSeaIceModel(ocean, sea_ice; atmosphere, radiation=nothing)
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
            radiation = nothing

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
            radiation = nothing

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

            ai_temp = default_ai_temperature(sea_ice)
            @test ai_temp.internal_flux isa IceSnowConductiveFlux
            @test ai_temp.internal_flux.ice.conductivity == 2.0
            @test ai_temp.internal_flux.snow.conductivity == 0.31
        end
    end
end

@testset "Sub-grid thickness distribution of the ice and snow conductivity" begin
    grid = RectilinearGrid(size = (2, 2), x = (0, 1), y = (0, 1), topology = (Periodic, Periodic, Flat))

    # `ThicknessDependentConductivity` is the former name of `GammaThicknessDistribution`
    @test ThicknessDependentConductivity === GammaThicknessDistribution

    # `thickness_categories = N` is a `UniformThicknessDistribution` on the ice and the snow, and the
    # conductivities stay the raw ones: the factor ∑ 1/(2i-1) enters through `effective_conductivity`
    N = 4
    F = sum(1 / (2i - 1) for i in 1:N)
    sea_ice = sea_ice_simulation(grid; dynamics = nothing, thickness_categories = N)
    flux = default_ai_temperature(sea_ice).internal_flux

    @test flux.ice.itd_shape == UniformThicknessDistribution(Float64; categories = N)
    @test flux.snow.itd_shape == UniformThicknessDistribution(Float64; categories = N)
    @test flux.ice.conductivity == 2 && flux.snow.conductivity == 0.31
    @test effective_conductivity(flux.ice, 1.0) ≈ 2F

    # The series resistance of the column is the old one, with both conductivities premultiplied by F
    for (hi, hs) in ((0.3, 0.0), (1.0, 0.1), (2.5, 0.5))
        @test thermal_resistance(flux, hi, hs) ≈ (hs / 0.31 + hi / 2) / F
    end

    # One category, and an explicit distribution for the ice and for the snow
    @test sea_ice_simulation(grid; dynamics = nothing).model.ice_thermodynamics.internal_heat_flux.itd_shape === nothing
    gamma = sea_ice_simulation(grid; dynamics = nothing, itd_shape = GammaThicknessDistribution())
    @test gamma.model.ice_thermodynamics.internal_heat_flux.itd_shape isa GammaThicknessDistribution
    @test default_snow_thermodynamics(grid; itd_shape = UniformThicknessDistribution(categories = 3)).internal_heat_flux.itd_shape ==
          UniformThicknessDistribution(Float64; categories = 3)
    @test default_snow_thermodynamics(grid; thickness_categories = 3).internal_heat_flux.itd_shape ==
          UniformThicknessDistribution(Float64; categories = 3)

    # ClimaSeaIce has one distribution per layer
    @test_throws ArgumentError sea_ice_simulation(grid; dynamics = nothing, thickness_categories = 4, itd_shape = GammaThicknessDistribution())
end
