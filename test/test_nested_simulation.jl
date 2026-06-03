include("runtests_setup.jl")

using NumericalEarth
using NumericalEarth.EarthSystemModels.NestedSimulations: parent_boundary_conditions
using Oceananigans
using Oceananigans.BoundaryConditions: ValueBoundaryCondition, fill_halo_regions!
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
    # Domain extends strictly beyond the child so the FTS brackets every child
    # boundary node (required by InterpolatedFTSBoundary's validation).
    parent_grid = RectilinearGrid(size     = (16, 16, 4),
                                  x        = (-1.5, 1.5),
                                  y        = (-1.5, 1.5),
                                  z        = (-0.2, 1.2),
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

    bcs = parent_boundary_conditions(child_grid;
                                     variables = (u = parent.velocities.u,
                                                  v = parent.velocities.v),
                                     sides     = (:west, :east, :south, :north))

    model = NonhydrostaticModel(child_grid; boundary_conditions = bcs)

    # Initial condition from the parent at t=0.
    set!(model, u = (x, y, z) -> lamb_oseen_uv(x, y, 0.0)[1],
                v = (x, y, z) -> lamb_oseen_uv(x, y, 0.0)[2])

    nested = NestedSimulation(parent, model; Δt = 0.001, stop_iteration = 5, verbose = false)

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

@testset "parent_boundary_conditions: bc_types selects BC kind per field" begin
    parent_grid = RectilinearGrid(size     = (8, 8, 4),
                                  x        = (0, 100), y = (0, 100), z = (0, 100),
                                  topology = (Bounded, Bounded, Bounded))

    times = collect(0.0:50.0:200.0)
    u_fts = FieldTimeSeries{Face,   Center, Center}(parent_grid, times)
    T_fts = FieldTimeSeries{Center, Center, Center}(parent_grid, times)
    fill!(parent(u_fts.data), 1.0)
    fill!(parent(T_fts.data), 42.0)

    child_grid = RectilinearGrid(size     = (4, 4, 4),
                                 x        = (20, 80), y = (20, 80), z = (10, 90),
                                 topology = (Bounded, Bounded, Bounded))

    bcs = parent_boundary_conditions(child_grid;
                                     variables = (u = u_fts, T = T_fts),
                                     sides     = (:west, :east, :south, :north),
                                     bc_types  = (T = ValueBoundaryCondition,))

    # u falls through to the OpenBoundaryCondition default.
    for side in (:west, :east, :south, :north)
        @test getproperty(bcs.u, side).classification isa Oceananigans.BoundaryConditions.Open
        @test getproperty(bcs.T, side).classification isa Oceananigans.BoundaryConditions.Value
    end

    # Passing `schemes` for a non-OpenBC field must error.
    @test_throws ArgumentError parent_boundary_conditions(
        child_grid;
        variables = (T = T_fts,),
        sides     = (:west, :east),
        schemes   = (T = nothing,),
        bc_types  = (T = ValueBoundaryCondition,))
end

# Regression for the GPU `InvalidIRError` in the prognostic-parent path: a LIVE model
# parent (not a PrescribedAtmosphere/FTS) drives the child through
# `Interpolated{<:AbstractField}` BCs. On GPU the source field `Adapt`s to a bare data
# array inside the halo-fill kernel, so `getbc`/`_query_source` must stay generically
# typed and take the location explicitly (rather than dispatching on `::AbstractField`
# and calling `instantiated_location(source)` in-kernel). Includes a Center
# `ValueBoundaryCondition` — the exact BC kind that failed. Runs on every
# `test_architecture` (GPU CI is where the regression bites).
@testset "NestedSimulation: prognostic (live AbstractField) parent on $(arch)" for arch in test_architectures
    parent_grid = RectilinearGrid(arch; size = (16, 16, 4),
                                  x = (-1.5, 1.5), y = (-1.5, 1.5), z = (-0.2, 1.2),
                                  topology = (Bounded, Bounded, Bounded))
    parent = NonhydrostaticModel(parent_grid; tracers = :c)
    set!(parent, u = (x, y, z) -> 0.1, v = (x, y, z) -> 0.05, c = (x, y, z) -> x + y)

    child_grid = RectilinearGrid(arch; size = (8, 8, 4),
                                 x = (-1, 1), y = (-1, 1), z = (0, 1),
                                 topology = (Bounded, Bounded, Bounded))
    bcs = parent_boundary_conditions(child_grid;
                                     variables = (u = parent.velocities.u,   # Open (Face)
                                                  v = parent.velocities.v,
                                                  c = parent.tracers.c),      # Value (Center) — the kind that broke
                                     sides     = (:west, :east, :south, :north),
                                     bc_types  = (c = ValueBoundaryCondition,))
    child = NonhydrostaticModel(child_grid; tracers = :c, boundary_conditions = bcs)
    set!(child, u = (x, y, z) -> 0.1, v = (x, y, z) -> 0.05, c = (x, y, z) -> x + y)

    nested = NestedSimulation(parent, child; Δt = 0.001, stop_iteration = 3, verbose = false)
    run!(nested)   # pre-fix: InvalidIRError on GPU during the first child halo fill

    @test child.clock.iteration == 3
    @test parent.clock.time ≈ child.clock.time
    @test all(isfinite, Array(interior(child.velocities.u)))
    @test all(isfinite, Array(interior(child.tracers.c)))
end

# An `Interpolated` Value BC must sample the source at the boundary FACE, not the child
# field's center node — otherwise a Center field's halo is reconstructed from a value a
# half-cell *inside* the boundary (a half-cell-gradient bias). With a source exactly
# linear in the boundary-normal coordinate, the reconstructed boundary-face value
# ½(halo + first-interior) must equal the source AT the face — i.e. = 0 to roundoff with
# the face fix, but off by ½Δ·(slope) with the (buggy) center placement. Covers BOTH
# normal directions, so the Dim-1 (west/east) and Dim-2 (south/north) `node` edits are
# each exercised.
@testset "Interpolated Value BC samples at the boundary face on $(arch)" for arch in test_architectures
    src_grid = RectilinearGrid(arch; size = (16, 16, 4), x = (-1, 3), y = (-1, 3), z = (0, 1),
                               topology = (Bounded, Bounded, Bounded))
    cg = RectilinearGrid(arch; size = (8, 8, 4), x = (0, 2), y = (0, 2), z = (0, 1),
                         topology = (Bounded, Bounded, Bounded))

    # child tracer c driven by a Value BC from a source equal to `f`, with c's IC = f too.
    function reconstructed_c(f, sides)
        src = CenterField(src_grid); set!(src, f)
        bcs = parent_boundary_conditions(cg; variables = (c = src,),
                                         sides = sides, bc_types = (c = ValueBoundaryCondition,))
        model = NonhydrostaticModel(cg; tracers = :c, boundary_conditions = bcs)
        set!(model, c = f)
        fill_halo_regions!(model.tracers.c)
        return model.tracers.c
    end

    # x-normal: source linear in x ⇒ west face at x=0, east face at x=2.
    cx = reconstructed_c((x, y, z) -> x, (:west, :east))
    @test isapprox(CUDA.@allowscalar((cx[0, 4, 2] + cx[1, 4, 2]) / 2), 0.0; atol = 1e-4)
    @test isapprox(CUDA.@allowscalar((cx[8, 4, 2] + cx[9, 4, 2]) / 2), 2.0; atol = 1e-4)

    # y-normal: source linear in y ⇒ south face at y=0, north face at y=2 (Dim-2 edit).
    cy = reconstructed_c((x, y, z) -> y, (:south, :north))
    @test isapprox(CUDA.@allowscalar((cy[4, 0, 2] + cy[4, 1, 2]) / 2), 0.0; atol = 1e-4)
    @test isapprox(CUDA.@allowscalar((cy[4, 8, 2] + cy[4, 9, 2]) / 2), 2.0; atol = 1e-4)
end
