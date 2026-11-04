include("runtests_setup.jl")

using Breeze
using Oceananigans.Units: kilometers

using NumericalEarth.Atmospheres: PrescribedAtmosphere

surface_grid(arch) = RectilinearGrid(arch, size = (4, 4), extent = (100kilometers, 100kilometers),
                                     topology = (Periodic, Periodic, Flat))

function cold_prescribed_atmosphere(grid)
    atmosphere = PrescribedAtmosphere(grid)
    parent(atmosphere.temperature) .= 250
    parent(atmosphere.velocities.u) .= 5
    return atmosphere
end

function prescribed_surface_model(arch, concentration)
    grid = surface_grid(arch)
    atmosphere = cold_prescribed_atmosphere(grid)
    radiation = PrescribedRadiation(grid)
    parent(radiation.downwelling_longwave) .= 200
    ocean = PrescribedOcean(grid)
    set!(ocean, T = 271.3, S = 34)
    sea_ice = PrescribedSeaIce(grid; concentration)
    return EarthSystemModel(; atmosphere, radiation, ocean, sea_ice)
end

@testset "PrescribedSeaIce" begin
    for arch in test_architectures
        A = typeof(arch)

        @testset "Construction and time interpolation on $A" begin
            grid = surface_grid(arch)
            sea_ice = PrescribedSeaIce(grid, [0, 3600])
            parent(sea_ice.concentration_time_series[2]) .= 1
            @test sea_ice isa PrescribedSeaIce
            @test summary(sea_ice) == "PrescribedSeaIce{Float64}(Nt = 2)"

            time_step!(sea_ice, 900)
            @test all(Array(interior(sea_ice.concentration)) .≈ 0.25)

            time_step!(sea_ice, 900)
            @test all(Array(interior(sea_ice.concentration)) .≈ 0.5)
        end

        @testset "Atmosphere–sea ice fluxes with a PrescribedAtmosphere on $A" begin
            open_water = prescribed_surface_model(arch, 0)
            ice_covered = prescribed_surface_model(arch, 1)

            for _ in 1:3
                time_step!(open_water, 60)
                time_step!(ice_covered, 60)
            end

            ai_open = open_water.interfaces.atmosphere_sea_ice_interface
            ai_ice = ice_covered.interfaces.atmosphere_sea_ice_interface
            @test all(Array(interior(ai_open.fluxes.sensible_heat)) .== 0)

            # Ice cooled by longwave emission under cold air sits below the ocean surface temperature,
            # so it gives up less sensible heat than the open water does.
            Qᵃⁱ = Array(interior(ai_ice.fluxes.sensible_heat))
            Qᵃᵒ = Array(interior(ice_covered.interfaces.atmosphere_ocean_interface.fluxes.sensible_heat))
            Tⁱ = Array(interior(ai_ice.temperature))
            @test all(isfinite, Qᵃⁱ)
            @test all(Tⁱ .< -1.8)
            @test all(Tⁱ .> 250 - 273.15)
            @test all(0 .< Qᵃⁱ .< Qᵃᵒ)
        end

        @testset "Breeze surface fluxes split by concentration on $A" begin
            for ℵ in (0, 0.5, 1)
                grid = RectilinearGrid(arch, size = (4, 4, 8), halo = (5, 5, 5),
                                       x = (-10kilometers, 10kilometers), y = (-10kilometers, 10kilometers),
                                       z = (0, 10kilometers), topology = (Periodic, Periodic, Bounded))

                atmosphere = atmosphere_simulation(grid; potential_temperature = 260)
                set!(atmosphere.model, θ = 260, u = 5)

                sgrid = RectilinearGrid(arch, size = (4, 4), halo = (5, 5),
                                        x = (-10kilometers, 10kilometers), y = (-10kilometers, 10kilometers),
                                        topology = (Periodic, Periodic, Flat))

                ocean = PrescribedOcean(sgrid)
                set!(ocean, T = 271.3, S = 34)
                sea_ice = PrescribedSeaIce(sgrid; concentration = ℵ)
                model = EarthSystemModel(; atmosphere, ocean, sea_ice)

                for _ in 1:3
                    time_step!(model, 1)
                end

                interfaces = model.interfaces
                Qᵃᵒ = Array(interior(interfaces.atmosphere_ocean_interface.fluxes.sensible_heat))
                Qᵃⁱ = Array(interior(interfaces.atmosphere_sea_ice_interface.fluxes.sensible_heat))
                Q = Array(interior(interfaces.net_fluxes.atmosphere.ρE))

                @test all(isfinite, Q)
                @test Q ≈ (1 - ℵ) .* Qᵃᵒ .+ ℵ .* Qᵃⁱ
                ℵ == 0 && @test Q ≈ Qᵃᵒ
                ℵ == 1 && @test Q ≈ Qᵃⁱ
            end
        end
    end
end
