include("runtests_setup.jl")

using Breeze
using NumericalEarth
using Oceananigans
using Oceananigans.Fields: interior
using Oceananigans.Grids: Periodic, Bounded, Flat
using Oceananigans.Units
using Oceananigans.TimeSteppers: time_step!
using Test

# An eastward wind in air warmer and moister than the surface below it
function warm_eastward_atmosphere(arch)
    grid = LatitudeLongitudeGrid(arch; size = (8, 4), longitude = (0, 360), latitude = (-40, 40),
                                 topology = (Periodic, Bounded, Flat), halo = (3, 3))

    atmosphere = PrescribedAtmosphere(grid, [0.0])
    set!(atmosphere.velocities.u, 10)
    set!(atmosphere.temperature, 318.15)
    set!(atmosphere.specific_humidity, 0.06)
    set!(atmosphere.pressure, 101325)

    return atmosphere
end

mean_interior(field) = sum(Array(interior(field))) / length(interior(field))

@testset "Surface fluxes push every component the way the air does" begin
    for arch in test_architectures
        A = typeof(arch)

        @testset "Oceananigans ocean under warm eastward air on $A" begin
            grid = LatitudeLongitudeGrid(arch; size = (8, 4, 1), longitude = (0, 360), latitude = (-40, 40), z = (-10, 0))
            ocean = ocean_simulation(grid; momentum_advection = nothing, tracer_advection = nothing, closure = nothing)
            set!(ocean.model, T = 20, S = 35)

            model = OceanOnlyModel(ocean; atmosphere = warm_eastward_atmosphere(arch))
            time_step!(model, 10minutes)

            @test mean_interior(ocean.model.tracers.T) > 20
            @test mean_interior(ocean.model.velocities.u) > 0
        end

        @testset "Slab ocean under warm air on $A" begin
            grid = LatitudeLongitudeGrid(arch; size = (8, 4), longitude = (0, 360), latitude = (-40, 40),
                                         topology = (Periodic, Bounded, Flat), halo = (3, 3))
            ocean = SlabOcean(grid, depth = 50, density = 1025, heat_capacity = 4000)
            set!(ocean, T = 293)

            model = AtmosphereOceanModel(warm_eastward_atmosphere(arch), ocean)
            time_step!(model, 10minutes)

            @test mean_interior(ocean.temperature) > 293
        end

        @testset "Breeze atmosphere over a warm ocean on $A" begin
            grid = RectilinearGrid(arch; size = (8, 8, 8), halo = (5, 5, 5),
                                   x = (0, 20kilometers), y = (0, 20kilometers), z = (0, 8kilometers),
                                   topology = (Periodic, Periodic, Bounded))

            atmosphere = atmosphere_simulation(grid; potential_temperature = 285)
            set!(atmosphere.model, θ = 285, u = 5)

            ocean_grid = RectilinearGrid(arch; size = (8, 8), halo = (5, 5),
                                         x = (0, 20kilometers), y = (0, 20kilometers),
                                         topology = (Periodic, Periodic, Flat))

            ocean = SlabOcean(ocean_grid, depth = 50, density = 1025, heat_capacity = 4000)
            set!(ocean, T = 295)

            model = AtmosphereOceanModel(atmosphere, ocean)

            # Upward heat and moisture fluxes, and a drag against the eastward wind
            net = model.interfaces.net_fluxes.atmosphere
            @test mean_interior(net.ρE) > 0
            @test mean_interior(net.ρqᵛᵉ) > 0
            @test mean_interior(net.ρu) < 0
        end
    end
end
