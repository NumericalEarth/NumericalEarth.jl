include("runtests_setup.jl")

using NumericalEarth: IceShelfOceanInterface, compute_ice_shelf_fluxes!, ice_shelf_tracer_forcing
using NumericalEarth.EarthSystemModels: ThreeEquationHeatFlux

using Oceananigans.Units

const Lx = 10kilometers
const Ly = 4kilometers
const Nx = 8
const Ny = 4

# The ice draft rises from -75 m at x = 0 to -15 m at x = 6 km; the ocean is open beyond.
freshwater_test_top_height(x, y) = x < 6kilometers ? -75 + 60x / 6kilometers : 0

function freshwater_test_grid(arch, VerticalBoundaries; z = MutableVerticalDiscretization(collect(-100:10:0)))
    underlying_grid = RectilinearGrid(arch; size = (Nx, Ny, 10), x = (0, Lx), y = (0, Ly), z,
                                      topology = (Bounded, Periodic, Bounded))
    boundaries = VerticalBoundaries(; bottom_height = -100, top_height = freshwater_test_top_height)
    return ImmersedBoundaryGrid(underlying_grid, boundaries)
end

volume_integral(c) = Array(interior(compute!(Field(Integral(c)))))[1]
area_integral(J) = sum(Array(interior(J))) * (Lx / Nx) * (Ly / Ny)

function freshwater_test_model(grid, free_surface, timestepper; ice_salinity = 0)
    interface = IceShelfOceanInterface(grid; free_surface, ice_salinity,
                                       flux_formulation = ThreeEquationHeatFlux())

    model = HydrostaticFreeSurfaceModel(grid; free_surface, timestepper,
                                        tracers = (:T, :S),
                                        buoyancy = SeawaterBuoyancy(),
                                        forcing = ice_shelf_tracer_forcing(interface),
                                        vertical_coordinate = ZStarCoordinate())

    set!(model, T = 1, S = 34.5)
    compute_ice_shelf_fluxes!(interface, model)

    return model, interface
end

@testset "Ice shelf real freshwater flux" begin
    for arch in test_architectures
        @testset "Volume forcing only on a mutable grid [$(typeof(arch))]" begin
            grid = freshwater_test_grid(arch, GridFittedVerticalBoundaries)
            @test keys(ice_shelf_tracer_forcing(IceShelfOceanInterface(grid))) == (:T, :S, :η)

            static_grid = freshwater_test_grid(arch, GridFittedVerticalBoundaries; z = (-100, 0))
            @test keys(ice_shelf_tracer_forcing(IceShelfOceanInterface(static_grid))) == (:T, :S)
        end

        for VerticalBoundaries in (GridFittedVerticalBoundaries, PartialCellVerticalBoundaries),
            free_surface_name in (:split_explicit, :implicit),
            timestepper in (:QuasiAdamsBashforth2, :SplitRungeKutta3)

            @testset "Frozen-flux budgets, $VerticalBoundaries, $free_surface_name, $timestepper [$(typeof(arch))]" begin
                grid = freshwater_test_grid(arch, VerticalBoundaries)
                free_surface = free_surface_name == :implicit ? ImplicitFreeSurface() :
                                                                SplitExplicitFreeSurface(grid; substeps = 10)

                Sˢⁱ = 4
                model, interface = freshwater_test_model(grid, free_surface, timestepper; ice_salinity = Sˢⁱ)
                fluxes = interface.fluxes
                ρc = interface.properties.reference_density * interface.properties.heat_capacity

                volume = CenterField(grid)
                set!(volume, 1)

                V₀ = volume_integral(volume)
                ΣT₀ = volume_integral(model.tracers.T)
                ΣS₀ = volume_integral(model.tracers.S)

                Nt, Δt = 10, 60
                for _ in 1:Nt
                    time_step!(model, Δt)
                end
                t = Nt * Δt

                @test area_integral(fluxes.freshwater) > 0
                @test volume_integral(volume) - V₀ ≈ t * area_integral(fluxes.freshwater) rtol = 1e-8
                @test volume_integral(model.tracers.S) - ΣS₀ ≈ t * Sˢⁱ * area_integral(fluxes.freshwater) rtol = 1e-8
                @test volume_integral(model.tracers.T) - ΣT₀ ≈ t * (area_integral(fluxes.freshwater_heat_content) -
                                                                    area_integral(fluxes.interface_heat) / ρc) rtol = 1e-8
            end
        end

        for free_surface_name in (:split_explicit, :implicit)
            @testset "Salt conservation with live fluxes, $free_surface_name [$(typeof(arch))]" begin
                grid = freshwater_test_grid(arch, PartialCellVerticalBoundaries)
                free_surface = free_surface_name == :implicit ? ImplicitFreeSurface() :
                                                                SplitExplicitFreeSurface(grid; substeps = 10)

                model, interface = freshwater_test_model(grid, free_surface, :SplitRungeKutta3)
                ΣS₀ = volume_integral(model.tracers.S)

                simulation = Simulation(model; Δt = 60, stop_iteration = 20, verbose = false)
                add_callback!(simulation, sim -> compute_ice_shelf_fluxes!(interface, sim), IterationInterval(1))
                run!(simulation)

                @test all(isfinite, Array(interior(model.tracers.S)))
                @test volume_integral(model.tracers.S) ≈ ΣS₀ rtol = 1e-13
            end
        end
    end
end
