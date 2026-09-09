include("runtests_setup.jl")

using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.Models.HydrostaticFreeSurfaceModels: update_grid_scaling!
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.Units: Time
using NumericalEarth.Oceans: TwoColorRadiation, absorption_coefficient, compute_absorption_coefficient!,
                             shortwave_radiative_forcing, surface_absorbed_fraction

ocean_properties = (reference_density=1020.0, heat_capacity=3991.0)
Iˢʷ = -200.0 # W m⁻², negative because it enters the ocean

function absorbed_shortwave(i, j, grid, radiation)
    Qss = shortwave_radiative_forcing(i, j, grid, radiation, Iˢʷ, ocean_properties)
    surface = - Qss / (ocean_properties.reference_density * ocean_properties.heat_capacity)
    interior = sum(radiation(i, j, k, grid, nothing, nothing) * Δzᶜᶜᶜ(i, j, k, grid) for k in 1:size(grid, 3))
    return surface, interior
end

@testset "TwoColorRadiation shortwave conservation" begin
    # The single shallow cell is the column whose surface cell is also the bottom cell, where the
    # radiation that Beer's law leaves unabsorbed has nowhere to go but back into the surface cell.
    for (Nz, depth) in ((1, 2), (100, 400))
        grid = RectilinearGrid(size=Nz, z=(-depth, 0), topology=(Flat, Flat, Bounded))
        radiation = TwoColorRadiation(grid)

        surface, interior = absorbed_shortwave(1, 1, grid, radiation)
        J₀ = radiation.surface_flux[1, 1, 1]

        @test surface + interior ≈ J₀
        @test surface ≈ surface_absorbed_fraction(1, 1, grid, radiation) * J₀
    end

    grid = RectilinearGrid(size=100, z=(-400, 0), topology=(Flat, Flat, Bounded))
    radiation = TwoColorRadiation(grid)
    shortwave_radiative_forcing(1, 1, grid, radiation, Iˢʷ, ocean_properties)

    # A column deep enough to have a cell below the surface leaves nothing in the surface cell
    @test radiation(1, 1, 100, grid, nothing, nothing) == 0
    @test radiation(1, 1, 99, grid, nothing, nothing) > 0

    # Chlorophyll that varies horizontally makes κ₂ a field, which the surface fraction and the
    # flux divergence have to read at the same column for the two to cancel.
    grid = RectilinearGrid(size=(2, 1, 20), x=(0, 1), y=(0, 1), z=(-100, 0),
                           topology=(Periodic, Periodic, Bounded))

    chlorophyll = Field{Center, Center, Nothing}(grid)
    set!(chlorophyll, (λ, φ) -> ifelse(λ < 0.5, 0.05, 1.5))
    radiation = TwoColorRadiation(grid; chlorophyll)
    compute_absorption_coefficient!(radiation, Time(0))

    κ₂ = radiation.second_absorption_coefficient
    @test κ₂[1, 1, 1] ≈ absorption_coefficient(radiation.chlorophyll_optics, 0.05)
    @test κ₂[2, 1, 1] ≈ absorption_coefficient(radiation.chlorophyll_optics, 1.5)

    for i in 1:2
        surface, interior = absorbed_shortwave(i, 1, grid, radiation)
        @test surface + interior ≈ radiation.surface_flux[i, 1, 1]
    end

    # The greener column absorbs more of the blue-green band within the surface cell
    @test surface_absorbed_fraction(2, 1, grid, radiation) > surface_absorbed_fraction(1, 1, grid, radiation)

    # A moving vertical coordinate stretches the surface cell between the coupled step that sets the
    # boundary condition and the substeps that evaluate the interior source, so the two have to close
    # against a surface fraction that the free surface leaves alone.
    grid = RectilinearGrid(size=(1, 1, 20), x=(0, 1), y=(0, 1),
                           z=MutableVerticalDiscretization((-100, 0)),
                           topology=(Periodic, Periodic, Bounded))

    radiation = TwoColorRadiation(grid)
    surface, _ = absorbed_shortwave(1, 1, grid, radiation)
    J₀ = radiation.surface_flux[1, 1, 1]

    for η in (0.01, 0.1, 1.0)
        grid.z.ηⁿ[1, 1, 1] = η
        update_grid_scaling!(grid.z, 1, 1, grid)
        interior = sum(radiation(1, 1, k, grid, nothing, nothing) * Δzᶜᶜᶜ(1, 1, k, grid) for k in 1:size(grid, 3))
        @test isapprox(surface + interior, J₀, rtol=1e-12)
    end
end
