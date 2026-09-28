include("runtests_setup.jl")

using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.Models.HydrostaticFreeSurfaceModels: update_grid_scaling!
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.TurbulenceClosures: VerticalScalarDiffusivity
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: surface_radiative_buoyancy_flux, transmitted_fraction,
                                                                     transmitted_fraction_derivative, transmitted_thickness
using Oceananigans.Units: Time
using NumericalEarth.Oceans: TwoColorRadiation, absorption_coefficient, compute_absorption_coefficient!,
                             shortwave_radiative_forcing, get_radiative_forcing, default_ocean_closure
using SeawaterPolynomials: thermal_expansion

ocean_properties = (reference_density=1020.0, heat_capacity=3991.0)
Iˢʷ = -200.0 # W m⁻², negative because it enters the ocean

# Temperature flux that the forcing deposits between the surface and the bottom face of cell k
absorbed_above(i, j, k, grid, radiation) = sum(radiation(i, j, n, grid, nothing, nothing) * Δzᶜᶜᶜ(i, j, n, grid) for n in k:size(grid, 3))

@testset "TwoColorRadiation" begin
    # The whole shortwave is absorbed in the interior, the bottom cell taking what reaches the bottom
    for (Nz, depth) in ((1, 2), (100, 400))
        grid = RectilinearGrid(size=Nz, z=(-depth, 0), topology=(Flat, Flat, Bounded))
        radiation = TwoColorRadiation(grid)
        @test shortwave_radiative_forcing(1, 1, grid, radiation, Iˢʷ, ocean_properties) == 0
        @test absorbed_above(1, 1, 1, grid, radiation) ≈ radiation.surface_flux[1, 1, 1]
    end

    # The transmitted fraction is what the forcing leaves below depth d, and its derivative and integral are consistent with it
    grid = RectilinearGrid(size=100, z=(-400, 0), topology=(Flat, Flat, Bounded))
    radiation = TwoColorRadiation(grid)
    shortwave_radiative_forcing(1, 1, grid, radiation, Iˢʷ, ocean_properties)
    J₀ = radiation.surface_flux[1, 1, 1]
    zᶠ = znodes(grid, Face())

    for k in (100, 99, 90, 60)
        d = - zᶠ[k]
        @test absorbed_above(1, 1, k, grid, radiation) ≈ J₀ * (1 - transmitted_fraction(radiation, 1, 1, grid, d))
    end

    T(d) = transmitted_fraction(radiation, 1, 1, grid, d)
    ℒ(h) = transmitted_thickness(radiation, 1, 1, grid, h)
    @test T(0) == 1
    @test ℒ(0) == 0

    for d in (0.1, 1.0, 10.0, 100.0)
        δ = 1e-5 * d
        @test transmitted_fraction_derivative(radiation, 1, 1, grid, d) ≈ (T(d + δ) - T(d - δ)) / 2δ rtol=1e-6
        @test (ℒ(d + δ) - ℒ(d - δ)) / 2δ ≈ T(d) rtol=1e-6
    end

    # Chlorophyll that varies horizontally makes κ₂ a field, read at the same column by the forcing and by CATKE
    grid = RectilinearGrid(size=(2, 1, 20), x=(0, 1), y=(0, 1), z=(-100, 0),
                           topology=(Periodic, Periodic, Bounded))

    chlorophyll = Field{Center, Center, Nothing}(grid)
    set!(chlorophyll, (λ, φ) -> ifelse(λ < 0.5, 0.05, 1.5))
    radiation = TwoColorRadiation(grid; chlorophyll)
    compute_absorption_coefficient!(radiation, Time(0))

    κ₂ = radiation.second_absorption_coefficient
    @test κ₂[1, 1, 1] ≈ absorption_coefficient(radiation.chlorophyll_optics, 0.05)
    @test κ₂[2, 1, 1] ≈ absorption_coefficient(radiation.chlorophyll_optics, 1.5)

    d = - znodes(grid, Face())[15]
    for i in 1:2
        shortwave_radiative_forcing(i, 1, grid, radiation, Iˢʷ, ocean_properties)
        @test absorbed_above(i, 1, 1, grid, radiation) ≈ radiation.surface_flux[i, 1, 1]
        @test absorbed_above(i, 1, 15, grid, radiation) ≈ radiation.surface_flux[i, 1, 1] * (1 - transmitted_fraction(radiation, i, 1, grid, d))
    end

    # The greener column lets less light through
    @test transmitted_fraction(radiation, 2, 1, grid, 10) < transmitted_fraction(radiation, 1, 1, grid, 10)

    # A moving vertical coordinate stretches the column, which still absorbs the whole shortwave
    grid = RectilinearGrid(size=(1, 1, 20), x=(0, 1), y=(0, 1),
                           z=MutableVerticalDiscretization((-100, 0)),
                           topology=(Periodic, Periodic, Bounded))

    radiation = TwoColorRadiation(grid)
    shortwave_radiative_forcing(1, 1, grid, radiation, Iˢʷ, ocean_properties)

    for η in (0.01, 0.1, 1.0)
        grid.z.ηⁿ[1, 1, 1] = η
        update_grid_scaling!(grid.z, 1, 1, grid)
        @test isapprox(absorbed_above(1, 1, 1, grid, radiation), radiation.surface_flux[1, 1, 1], rtol=1e-12)
    end
end

@testset "CATKE sees the penetrating shortwave" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch, size=20, z=(-100, 0), topology=(Flat, Flat, Bounded))
        closure = (default_ocean_closure(), VerticalScalarDiffusivity(κ=1e-5))
        ocean = ocean_simulation(grid; closure, momentum_advection=nothing, tracer_advection=nothing, warn=false)
        model = ocean.model
        radiation = get_radiative_forcing(model)

        # ocean_simulation hands its radiative forcing to the CATKE in the closure tuple
        @test radiation isa TwoColorRadiation
        @test model.closure[1].penetrative_radiation === radiation

        # Fluxes positive upward: the shortwave enters with a negative buoyancy flux, -g α J₀
        set!(model, T=15, S=35)
        @allowscalar shortwave_radiative_forcing(1, 1, grid, radiation, Iˢʷ, ocean_properties)
        b = model.buoyancy.formulation
        Jʳ = @allowscalar surface_radiative_buoyancy_flux(1, 1, grid, radiation, model.buoyancy, merge(model.velocities, model.tracers))
        α = thermal_expansion(15.0, 35.0, 0.0, b.equation_of_state)
        @test Jʳ < 0
        @test Jʳ ≈ - b.gravitational_acceleration * α * (@allowscalar radiation.surface_flux[1, 1, 1])

        for _ in 1:3
            time_step!(model, 600)
        end

        @test all(isfinite, Array(interior(model.tracers.e)))
    end
end
