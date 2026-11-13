include("runtests_setup.jl")

using Breeze
using ConservativeRegridding
using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Oceananigans.Fields: interior
using Oceananigans.Operators: Azᶜᶜᶜ
using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.EarthSystemModels.InterfaceComputations: conservative_exchange, regrid_surface_field!
using Test


function area_integral(field)
    grid = field.grid
    Nx, Ny, _ = size(grid)
    return sum(interior(field)[i, j, 1] * Azᶜᶜᶜ(i, j, 1, grid) for i in 1:Nx, j in 1:Ny)
end

function breeze_atmosphere_over_slab_ocean(arch, atmosphere_size, ocean_size)
    atmosphere_grid = RectilinearGrid(arch; size = (atmosphere_size..., 8), halo = (5, 5, 5),
                                      x = (0, 40kilometers), y = (0, 40kilometers), z = (0, 10kilometers),
                                      topology = (Periodic, Periodic, Bounded))

    atmosphere = atmosphere_simulation(atmosphere_grid; potential_temperature=285)
    set!(atmosphere.model, θ=285, u=5, v=-2)

    ocean_grid = RectilinearGrid(arch; size = ocean_size, halo = (5, 5),
                                 x = (0, 40kilometers), y = (0, 40kilometers),
                                 topology = (Periodic, Periodic, Flat))

    ocean = SlabOcean(ocean_grid, depth=50, density=1025, heat_capacity=4000)
    set!(ocean, T = (x, y) -> 288 + 5 * sinpi(2x / 40kilometers) * cospi(2y / 40kilometers))

    return AtmosphereOceanModel(atmosphere, ocean)
end

@testset "Breeze atmosphere on another horizontal grid" begin
    for arch in test_architectures
        model = breeze_atmosphere_over_slab_ocean(arch, (8, 8), (32, 32))
        update_state!(model)

        interfaces = model.interfaces
        fluxes = interfaces.surface_partition.fluxes
        net = interfaces.net_fluxes.atmosphere

        @test maximum(abs, interior(fluxes.sensible_heat)) > 0

        # First-order conservative remapping returns every surface flux to the atmosphere without loss
        @test area_integral(net.ρE)   ≈ area_integral(fluxes.sensible_heat)
        @test area_integral(net.ρqᵛᵉ) ≈ area_integral(fluxes.water_vapor)
        @test area_integral(net.ρu)   ≈ area_integral(fluxes.x_momentum)
        @test area_integral(net.ρv)   ≈ area_integral(fluxes.y_momentum)

        # The atmosphere's uniform state reaches the exchange grid unchanged
        state = interfaces.exchanger.atmosphere.state
        @test all(interior(state.u) .≈ interior(state.u)[1, 1, 1])

        time_step!(model, 1)
        @test isfinite(maximum(abs, interior(model.atmosphere.model.momentum.ρu)))
    end
end

@testset "Breeze atmosphere over land on another horizontal grid" begin
    for arch in test_architectures
        atmosphere_grid = RectilinearGrid(arch; size = (8, 8, 8), halo = (5, 5, 5),
                                          x = (0, 40kilometers), y = (0, 40kilometers), z = (0, 10kilometers),
                                          topology = (Periodic, Periodic, Bounded))

        atmosphere = atmosphere_simulation(atmosphere_grid; potential_temperature=285)
        set!(atmosphere.model, θ=285, u=5)

        land_grid = RectilinearGrid(arch; size = (32, 32), halo = (5, 5),
                                    x = (0, 40kilometers), y = (0, 40kilometers),
                                    topology = (Periodic, Periodic, Flat))

        land = SlabLand(land_grid)
        set!(land; T = (x, y) -> 290 + 5 * sinpi(2x / 40kilometers))

        model = AtmosphereLandModel(atmosphere, land)
        update_state!(model)

        fluxes = model.interfaces.surface_partition.fluxes

        @test maximum(abs, interior(fluxes.sensible_heat)) > 0
        @test area_integral(model.interfaces.net_fluxes.atmosphere.ρE) ≈ area_integral(fluxes.sensible_heat)
    end
end

@testset "Both copies of a tripolar fold row receive the regridded value" begin
    bottom_height(λ, φ) = ifelse(φ < -70, 0, -1000)

    for arch in test_architectures, z in ((-100, 0), ExponentialDiscretization(2, -100, 0))
        underlying_grid = TripolarGrid(arch; size = (40, 20, 2), z, halo = (5, 5, 4))
        component_grid = LatitudeLongitudeGrid(arch; size = (30, 15, 1), z = (0, 1),
                                               longitude = (0, 360), latitude = (-90, 90))

        for grid in (underlying_grid, ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height)))
            exchange = conservative_exchange(on_architecture(CPU(), component_grid), grid)
            source = Field{Center, Center, Nothing}(component_grid)
            destination = Field{Center, Center, Nothing}(grid)

            set!(source, 1)
            regrid_surface_field!(destination, exchange.from_component, source)
            @test all(Array(interior(destination)) .≈ 1)

            set!(source, (λ, φ) -> φ + cosd(λ))
            regrid_surface_field!(destination, exchange.from_component, source)
            fold_row = Array(interior(destination, :, 20, 1))
            @test fold_row ≈ reverse(fold_row)
        end
    end
end

@testset "Breeze atmosphere on a horizontally identical grid needs no regridder" begin
    for arch in test_architectures
        model = breeze_atmosphere_over_slab_ocean(arch, (8, 8), (8, 8))
        @test isnothing(model.interfaces.exchanger.atmosphere.regridder)
    end
end
