include("runtests_setup.jl")
include("synthetic_datasets.jl")

using KernelAbstractions: @kernel, @index
using Oceananigans.Grids: φnode
using Oceananigans.Utils: launch!
using Oceananigans.ImmersedBoundaries: bottom_height_field
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.TimeSteppers: update_state!
using NumericalEarth.EarthSystemModels.InterfaceComputations: compute_sea_ice_ocean_fluxes!

@kernel function _immerse_latitude_band!(bottom_height, grid, south, north)
    i, j = @index(Global, NTuple)
    φ = φnode(i, j, 1, grid, Center(), Center(), Center())
    @inbounds z = bottom_height[i, j, 1]
    @inbounds bottom_height[i, j, 1] = ifelse((φ > south) & (φ < north), oftype(z, 100), z)
end

function latitude_band_grid(grid, south, north)
    underlying = grid.underlying_grid
    bottom = Field{Center, Center, Nothing}(underlying)
    parent(bottom) .= parent(bottom_height_field(grid))
    launch!(architecture(grid), underlying, :xy, _immerse_latitude_band!, bottom, underlying, south, north)
    fill_halo_regions!(bottom)
    return ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom); active_cells_map = true)
end

const ice_ocean_flux_names = (:interface_heat, :frazil_heat, :salt, :freshwater, :freshwater_heat_content,
                              :x_momentum, :y_momentum, :x_momentum_coefficient, :y_momentum_coefficient)

ice_ocean_flux(model, name) = getproperty(model.interfaces.sea_ice_ocean_interface.fluxes, name)

@testset "Sea ice on a grid with an extra immersed latitude band" begin
    for arch in test_architectures
        A = typeof(arch)
        @info "Testing the ice-ocean fluxes of sea ice on an immersed latitude band on $A..."

        underlying = TripolarGrid(arch; size = (40, 30, 6), halo = (7, 7, 7), z = (-1000, 0))
        bottom(λ, φ) = (20 < λ < 60) & (-30 < φ < 30) ? 100.0 : -800.0
        grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom); active_cells_map = true)

        function coupled_model(sea_ice_grid)
            ocean = ocean_simulation(grid; Δt = 600, closure = nothing)
            set!(ocean.model, T = (λ, φ, z) -> -1 + 0.02 * abs(φ), S = 34.5,
                              u = (λ, φ, z) -> 0.1 * sind(3λ) * cosd(φ), v = (λ, φ, z) -> 0.05 * cosd(2λ))
            sea_ice = sea_ice_simulation(sea_ice_grid, ocean; Δt = 600, advection = nothing)
            set!(sea_ice.model, h = (λ, φ) -> abs(φ) > 40 ? 1.5 : 0.0, ℵ = (λ, φ) -> abs(φ) > 40 ? 0.9 : 0.0)
            return OceanSeaIceModel(ocean, sea_ice; atmosphere = synthetic_prescribed_atmosphere(arch),
                                                    radiation = synthetic_prescribed_radiation(arch))
        end

        a = coupled_model(grid)
        b = coupled_model(latitude_band_grid(grid, -10, 10))

        identical_fluxes(a, b) = all(isequal(Array(interior(ice_ocean_flux(a, name))),
                                             Array(interior(ice_ocean_flux(b, name)))) for name in ice_ocean_flux_names)

        @test identical_fluxes(a, b)

        foreach(name -> fill!(parent(ice_ocean_flux(b, name)), NaN), ice_ocean_flux_names)
        compute_sea_ice_ocean_fluxes!(b)
        @test all(!any(isnan, interior(ice_ocean_flux(b, name))) for name in ice_ocean_flux_names)

        for n in 1:3
            for model in (a, b)
                u, v = model.ocean.model.velocities
                interior(u) .= interior(u) .* 0.9 .+ 0.02n
                interior(v) .= interior(v) .* 0.9 .- 0.01n
                fill_halo_regions!(u)
                fill_halo_regions!(v)
                time_step!(model.sea_ice, 600)
                update_state!(model)
            end

            @test identical_fluxes(a, b)
        end
    end
end
