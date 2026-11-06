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

using NumericalEarth.EarthSystemModels: interface_state_hooks, interface_flux_hooks
using NumericalEarth.EarthSystemModels.InterfaceComputations: computed_fluxes
using NumericalEarth.Grids: yin_yang_bilinear_weights, yin_yang_area_average_weights, panel_centers
using Oceananigans.Operators: Azᶜᶜᶜ
using Oceananigans.TimeSteppers: update_state!

# The atmosphere state, bilinearly interpolated, sets the ocean's exchanged air state; the ocean-grid air-sea
# heat flux, area-averaged, replaces the atmosphere's
@testset "Yin-Yang atmosphere coupled to an ocean and sea ice" begin
    surface_grid = YinYangGrid(CPU(); size = (48, 16, 1), z = (0, 1), halo = (3, 3, 3))
    atmosphere_model = uniform_coupled_model(YinYangGrid(CPU(); size = (48, 16, 16), z = (0, 10kilometers), halo = (3, 3, 3)),
                                             surface_grid)

    ocean_grid = LatitudeLongitudeGrid(CPU(); size = (36, 15, 4), longitude = (0, 360), latitude = (-75, 75), z = (-100, 0), halo = (7, 7, 4))
    ocean = ocean_simulation(ocean_grid; momentum_advection = nothing, tracer_advection = nothing, closure = nothing)
    set!(ocean.model, T = 18, S = 35)
    sea_ice = sea_ice_simulation(ocean_grid, ocean)
    exchange_atmosphere_grid = LatitudeLongitudeGrid(CPU(); size = (4, 4), longitude = (0, 360), latitude = (-90, 90),
                                                     topology = (Periodic, Bounded, Flat))
    ocean_model = OceanSeaIceModel(ocean, sea_ice; atmosphere = PrescribedAtmosphere(exchange_atmosphere_grid))

    Nx, Ny = size(ocean_grid, 1), size(ocean_grid, 2)
    λ = vec([λnode(i, j, 1, ocean_grid, Center(), Center(), Center()) for i in 1:Nx, j in 1:Ny])
    φ = vec([φnode(i, j, 1, ocean_grid, Center(), Center(), Center()) for i in 1:Nx, j in 1:Ny])
    area = vec([Azᶜᶜᶜ(i, j, 1, ocean_grid) for i in 1:Nx, j in 1:Ny])
    indices, weights = yin_yang_bilinear_weights(surface_grid, λ, φ)
    averages = yin_yang_area_average_weights(surface_grid, λ, φ, area, trues(length(λ)))

    panel_values(field) = vcat((vec(Array(interior(getregion(field, r), :, :, 1))) for r in 1:2)...)
    atmosphere_state = atmosphere_model.interfaces.exchanger.atmosphere.state
    ocean_air_state = ocean_model.interfaces.exchanger.atmosphere.state

    interface_state_hooks[ocean_model] = model -> for name in (:T, :q, :p)
        values = panel_values(getproperty(atmosphere_state, name))
        interior(getproperty(ocean_air_state, name), :, :, 1) .= reshape(sum(weights .* values[indices]; dims = 1), Nx, Ny)
    end

    ocean_heat_flux() = vec(Array(interior(computed_fluxes(ocean_model.interfaces.atmosphere_ocean_interface).sensible_heat, :, :, 1)))
    function averaged_heat_flux(r)
        Q = ocean_heat_flux()
        A = zeros(size(panel_centers(surface_grid, r)[1]))
        foreach((n, m, w) -> A[n] += w * Q[m], averages[r].rows, averages[r].columns, averages[r].weights)
        return A, unique(averages[r].rows)
    end

    interface_flux_hooks[atmosphere_model] = model -> for r in 1:2
        A, covered = averaged_heat_flux(r)
        Q = interior(getregion(computed_fluxes(model.interfaces.atmosphere_ocean_interface).sensible_heat, r), :, :, 1)
        view(Q, covered) .= A[covered]
    end

    for n in 1:3
        time_step!(atmosphere_model, 10)
        time_step!(ocean_model, 10)
    end
    update_state!(atmosphere_model)

    @test all(isfinite, ocean_heat_flux())
    @test interior(ocean_air_state.T, 1, Ny ÷ 2, 1) ≈ interior(getregion(atmosphere_state.T, 1), 1, 1, 1) rtol = 0.01
    for r in 1:2
        A, covered = averaged_heat_flux(r)
        Q = Array(interior(getregion(computed_fluxes(atmosphere_model.interfaces.atmosphere_ocean_interface).sensible_heat, r), :, :, 1))
        @test Q[covered] ≈ A[covered]
        @test all(isfinite, Array(interior(getregion(atmosphere_model.atmosphere.model.velocities.u, r))))
    end

    delete!(interface_state_hooks, ocean_model)
    delete!(interface_flux_hooks, atmosphere_model)
end
