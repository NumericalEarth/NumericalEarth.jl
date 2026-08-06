using OMIPSimulations
using OMIPSimulations: ConservativeSurfaceFluxRestoring, update_restoring_flux!

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.Operators: volume
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.TimeSteppers: time_step!
using Oceananigans.Fields: interior
using NumericalEarth.Oceans: ocean_simulation
using Test

# A minimal surface-salinity restoring: a virtual salt flux `Vp (S - S★(i))` toward a
# zonally-varying target whose global mean is offset from the ocean's uniform initial salinity,
# so the raw flux carries a nonzero net salt input.
struct ZonalSalinityRestoring{FT} <: Function
    piston_velocity :: FT
    S★ :: FT
    δS :: FT
end

@inline function Oceananigans.BoundaryConditions.getbc(r::ZonalSalinityRestoring, i, j, grid, clock, fields)
    Nx = size(grid, 1)
    Nz = size(grid, 3)
    S = @inbounds fields.S[i, j, Nz]
    S★ = r.S★ + r.δS * sinpi(2 * (i - 1) / Nx)
    return r.piston_velocity * (S - S★)
end

@testset "Conservative salinity restoring" begin
    arch = CPU()

    Lx = Ly = 1e5
    Nx = Ny = 8
    S₀ = 35.0
    piston_velocity = 1 / 86400 # m s⁻¹ (≈ 1 m day⁻¹)

    make_grid() = RectilinearGrid(arch;
                                  size = (Nx, Ny, 4),
                                  halo = (4, 4, 4),
                                  x = (0, Lx), y = (0, Ly),
                                  z = MutableVerticalDiscretization((-100, 0)),
                                  topology = (Periodic, Periodic, Bounded))

    # Integrate an ocean whose only salt flux is the restoring `make_flux(grid)`, returning the
    # initial and final total salt content and the final surface-salinity spread.
    function integrate_salt(make_flux, conservative)
        grid = make_grid()
        flux_S = make_flux(grid)
        ocean = ocean_simulation(grid;
                                 momentum_advection = nothing,
                                 closure = nothing,
                                 free_surface = SplitExplicitFreeSurface(substeps=30),
                                 radiative_forcing = nothing,
                                 bottom_drag_coefficient = 0,
                                 additional_surface_fluxes = (; S = flux_S))
        set!(ocean.model, T=20, S=S₀)

        cell_volume = KernelFunctionOperation{Center, Center, Center}(volume, grid, Center(), Center(), Center())
        total_salt() = sum(ocean.model.tracers.S * cell_volume)
        refresh!() = conservative && update_restoring_flux!(flux_S, ocean.model)

        refresh!()
        ∫S⁻ = total_salt()

        Δt = 2minutes
        for _ in 1:60
            refresh!()
            time_step!(ocean.model, Δt)
        end

        S_surface = Array(interior(ocean.model.tracers.S, :, :, size(grid, 3)))
        return ∫S⁻, total_salt(), maximum(S_surface) - minimum(S_surface)
    end

    # Restore toward a zonally-varying target whose global mean is offset from S₀, so the raw
    # restoring flux carries a nonzero net salt input.
    inner = ZonalSalinityRestoring(piston_velocity, S₀ + 2, 0.5)

    # Uncorrected, the net salt flux drifts the total salt content measurably.
    ∫S⁻_bare, ∫S_bare, _ = integrate_salt(g -> inner, false)
    @test abs(∫S_bare - ∫S⁻_bare) > 1e-9 * ∫S⁻_bare

    # The zero-global-mean correction redistributes salt but injects none globally, so total salt
    # content is conserved to machine precision...
    ∫S⁻, ∫S, ΔS_surface = integrate_salt(g -> ConservativeSurfaceFluxRestoring(inner, g), true)
    @test abs(∫S - ∫S⁻) < 1e-10 * ∫S⁻

    # ... while the target's zonal structure still drives a nonzero surface-salinity spread, so the
    # correction is not a trivial no-op that disables restoring altogether.
    @test ΔS_surface > 0
end
