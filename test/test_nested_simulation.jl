include("runtests_setup.jl")

using NumericalEarth
using NumericalEarth.EarthSystemModels.NestedSimulations: parent_boundary_conditions
using Oceananigans
using Test

@testset "PrescribedAtmosphere defaults switch on grid dimensionality" begin
    # 2D (single z-layer) — existing surface-atmosphere defaults
    g2 = LatitudeLongitudeGrid(longitude = (-180, 180),
                               latitude  = (-80,  80),
                               z         = (0, 1),
                               size      = (8, 8, 1),
                               topology  = (Periodic, Bounded, Bounded))
    pa2 = PrescribedAtmosphere(g2, [0.0, 1.0])
    @test keys(pa2.velocities) == (:u, :v)
    @test keys(pa2.tracers) == (:T, :q)
    @test pa2.freshwater_flux isa NumericalEarth.Atmospheres.PrescribedPrecipitationFlux

    # 3D (multi-layer) — volumetric defaults
    g3 = RectilinearGrid(size     = (8, 8, 4),
                         x        = (-1, 1),
                         y        = (-1, 1),
                         z        = (0, 1),
                         topology = (Bounded, Bounded, Bounded))
    pa3 = PrescribedAtmosphere(g3, [0.0, 1.0])
    @test keys(pa3.velocities) == (:u, :v, :w)
    @test keys(pa3.tracers) == (:T, :q)
    @test pa3.freshwater_flux === nothing
    @test size(pa3.velocities.u) == (8, 8, 4, 2)
    @test size(pa3.pressure)     == (8, 8, 4, 2)
end

# A translating Lamb-Oseen vortex: a 2D vortex with closed-form velocity
# `u_θ(r) = Γ/(2πr) (1 - exp(-r²/a²))`, advected by a uniform background U in x.
const Γ_LO  = 1.0
const a_LO  = 0.1
const U_LO  = 0.5
const x₀_LO = -0.5
const y₀_LO = 0.0

@inline function lamb_oseen_uv(x, y, t)
    dx = x - x₀_LO - U_LO * t
    dy = y - y₀_LO
    r² = dx*dx + dy*dy
    uθ_over_r = r² < eps() ? zero(r²) : (Γ_LO / (2π * r²)) * (1 - exp(-r² / a_LO^2))
    return (U_LO - uθ_over_r * dy, uθ_over_r * dx)
end

@testset "NestedSimulation: Lamb-Oseen vortex through a child NonhydrostaticModel" begin
    # Parent atmosphere holds the analytic Lamb-Oseen state on a 3D PrescribedAtmosphere,
    # populated by set! at a few coarse time snapshots; interpolation handles the rest.
    # Nz > 1 so the volumetric defaults (CCC velocities/tracers/pressure) kick in.
    parent_grid = RectilinearGrid(size     = (32, 32, 2),
                                  x        = (-1, 1),
                                  y        = (-1, 1),
                                  z        = (0, 1),
                                  topology = (Bounded, Bounded, Bounded))

    times  = collect(0.0:0.1:1.0)
    parent = PrescribedAtmosphere(parent_grid, times)

    set!(parent.velocities.u, (x, y, z, t) -> lamb_oseen_uv(x, y, t)[1])
    set!(parent.velocities.v, (x, y, z, t) -> lamb_oseen_uv(x, y, t)[2])
    set!(parent.tracers.T,    (x, y, z, t) -> 288.15)        # isothermal
    set!(parent.tracers.q,    (x, y, z, t) -> 0.0)            # dry

    # Child runs Oceananigans NonhydrostaticModel on a same-resolution grid.
    # A finer child would exercise spatial-interpolation in the BCs more, but
    # adds runtime; bumping resolution is a follow-up.
    child_grid = RectilinearGrid(size     = (32, 32, 2),
                                 x        = (-1, 1),
                                 y        = (-1, 1),
                                 z        = (0, 1),
                                 topology = (Bounded, Bounded, Bounded))

    bcs = parent_boundary_conditions(parent;
                                     variables = (u=:u, v=:v),
                                     sides     = (:west, :east, :south, :north),
                                     grid      = child_grid)

    model = NonhydrostaticModel(child_grid; boundary_conditions = bcs)

    # Initial condition from the parent at t=0.
    set!(model, u = (x, y, z) -> lamb_oseen_uv(x, y, 0.0)[1],
                v = (x, y, z) -> lamb_oseen_uv(x, y, 0.0)[2])

    sim    = Simulation(model; Δt = 0.001, stop_iteration = 5, verbose = false)
    nested = NestedSimulation(parent, sim)

    run!(nested)

    @test model.clock.iteration == 5
    @test parent.clock.time ≈ model.clock.time
    @test all(isfinite, interior(model.velocities.u))
    @test all(isfinite, interior(model.velocities.v))

    # The vortex centre at t=0 sits at (x₀_LO, y₀_LO) = (-0.5, 0). The interior
    # near the centre should retain a recognisable vortex signature after a
    # few short timesteps — i.e. max |u| stays well above the background U.
    u_interior = Array(interior(model.velocities.u))
    @test maximum(abs, u_interior) > 1.5 * U_LO
end
