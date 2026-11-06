include("runtests_setup.jl")

using Breeze
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.Architectures: on_architecture
using Oceananigans.Grids: λnode, φnode, λnodes, φnodes
using Oceananigans.MultiRegion: YinYangGrid
using Oceananigans.Utils: getregion
using Test

# A Breeze atmosphere over a land and a prescribed ocean, split half and half, at rest but for an
# eastward wind `U`; land and ocean warmer than the air
function uniform_coupled_model(grid, surface_grid; U = 8)
    dynamics = CompressibleDynamics(SplitExplicitTimeDiscretization(); reference_potential_temperature = 285)
    atmosphere = atmosphere_simulation(grid; dynamics, momentum_advection = WENO(order=5),
                                       scalar_advection = WENO(order=5), Δt = 10)
    set!(atmosphere.model; θ = 285, u = U, ρ = HydrostaticallyBalancedDensity())

    land = SlabLand(surface_grid)
    set!(land.temperature, 288)
    set!(land; M = 20)

    ocean = PrescribedOcean(surface_grid, [0.0])
    set!(ocean; T = 291)

    return EarthSystemModel(; atmosphere, land, ocean, radiation = nothing, ocean_fraction = 1/2)
end

interior_value(field, i, j) = Array(interior(field))[i, j, 1]

@testset "Breeze coupling on a Yin-Yang grid" begin
    for arch in test_architectures
        yin_yang = uniform_coupled_model(YinYangGrid(arch; size = (48, 16, 16), z = (0, 10kilometers), halo = (3, 3, 3)),
                                         YinYangGrid(arch; size = (48, 16, 1), z = (0, 1), halo = (3, 3, 3)))

        latitude_longitude = uniform_coupled_model(LatitudeLongitudeGrid(arch; size = (36, 18, 16), longitude = (0, 360),
                                                                         latitude = (-60, 60), z = (0, 10kilometers), halo = (3, 3, 3)),
                                                   LatitudeLongitudeGrid(arch; size = (36, 18), longitude = (0, 360),
                                                                         latitude = (-60, 60), halo = (3, 3),
                                                                         topology = (Periodic, Bounded, Flat)))

        # A Yin cell, and the latitude-longitude cell holding its center
        i, j = size(yin_yang.interfaces.exchanger.grid, 1) ÷ 2, size(yin_yang.interfaces.exchanger.grid, 2) ÷ 2
        yin = on_architecture(CPU(), getregion(yin_yang.interfaces.exchanger.grid, 1))
        λ = λnode(i, j, 1, yin, Center(), Center(), Center())
        φ = φnode(i, j, 1, yin, Center(), Center(), Center())
        grid = on_architecture(CPU(), latitude_longitude.interfaces.exchanger.grid)
        iˡˡ = searchsortedlast(λnodes(grid, Face()), mod(λ, 360))
        jˡˡ = searchsortedlast(φnodes(grid, Face()), φ)

        for interface in (:atmosphere_ocean_interface, :atmosphere_land_interface)
            yin_yang_fluxes = getproperty(yin_yang.interfaces, interface).fluxes
            latitude_longitude_fluxes = getproperty(latitude_longitude.interfaces, interface).fluxes

            for name in (:sensible_heat, :latent_heat, :friction_velocity)
                a = interior_value(getregion(getproperty(yin_yang_fluxes, name), 1), i, j)
                b = interior_value(getproperty(latitude_longitude_fluxes, name), iˡˡ, jˡˡ)
                @test a ≈ b
            end

            # The stress magnitude does not depend on the basis of the panel
            τ(fluxes, i, j) = hypot(interior_value(fluxes.x_momentum, i, j), interior_value(fluxes.y_momentum, i, j))
            @test τ(getregion(yin_yang_fluxes, 1), i, j) ≈ τ(latitude_longitude_fluxes, iˡˡ, jˡˡ)
        end

        for n in 1:3
            time_step!(yin_yang, 10)
        end

        for r in 1:2, field in (yin_yang.atmosphere.model.velocities.u, yin_yang.land.temperature)
            @test all(isfinite, Array(interior(getregion(field, r))))
        end
    end
end
