include("runtests_setup.jl")

using NumericalEarth
using NumericalEarth.NestedModels: parent_boundary_conditions, ParentStateBoundary
using Oceananigans
using Oceananigans.Units: Time
using Oceananigans.Fields: location
using Oceananigans.Grids: znode
using Oceananigans.BoundaryConditions: ValueBoundaryCondition, FieldBoundaryConditions, fill_halo_regions!,
                                       regularize_boundary_condition, getbc, LeftBoundary
using Breeze
using Breeze: ThermodynamicConstants, dry_air_gas_constant, vapor_gas_constant, CompressibleDynamics
using Test

@testset "PrescribedAtmosphere: grid vertical topology selects surface vs volumetric fields" begin
    # A `Flat` vertical builds a surface atmosphere (u, v; 2D temperature / specific_humidity /
    # pressure) for ocean / sea-ice coupling. Temperature & humidity are direct properties now;
    # `tracers` is reserved for gas species (empty by default), and `microphysical_variables` for
    # cloud / precip species (also empty by default).
    gs = RectilinearGrid(size = (8, 8), x = (-1, 1), y = (-1, 1), topology = (Bounded, Bounded, Flat))
    pas = PrescribedAtmosphere(gs, [0.0, 1.0])
    @test keys(pas.velocities) == (:u, :v)
    @test location(pas.temperature) == (Center, Center, Nothing)
    @test location(pas.specific_humidity) == (Center, Center, Nothing)
    @test pas.microphysical_variables == NamedTuple()
    @test pas.tracers == NamedTuple()
    @test pas.precipitation_flux isa NumericalEarth.Atmospheres.PrescribedPrecipitationFlux

    # A resolved vertical builds a 3D atmosphere (adds w) — e.g. a `NestedSimulation` parent.
    gv = RectilinearGrid(size = (8, 8, 4), x = (-1, 1), y = (-1, 1), z = (0, 1),
                         topology = (Bounded, Bounded, Bounded))
    pav = PrescribedAtmosphere(gv, [0.0, 1.0])
    @test keys(pav.velocities) == (:u, :v, :w)
    @test location(pav.temperature) == (Center, Center, Center)
    @test size(pav.velocities.u) == (8, 8, 4, 2)
    @test size(pav.pressure)     == (8, 8, 4, 2)
    @test isnothing(pav.precipitation_flux)   # a 3D atmosphere carries precip in microphysical_variables

    # A surface atmosphere defaults to a precipitation flux; opt out via the keyword.
    pa0 = PrescribedAtmosphere(gs, [0.0, 1.0]; precipitation_flux = nothing)
    @test pa0.precipitation_flux === nothing

    # Cloud / precip species ride through `microphysical_variables`.
    qcl = FieldTimeSeries{Center, Center, Center}(gv, [0.0, 1.0])
    pam = PrescribedAtmosphere(gv, [0.0, 1.0]; microphysical_variables = (; qᶜˡ = qcl))
    @test keys(pam.microphysical_variables) == (:qᶜˡ,)
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
    # The resolved-vertical (3D) grid gives CCC velocities/tracers/pressure so the FTS can be
    # interpolated at the child's interior z-nodes.
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
    set!(parent.temperature,       (x, y, z, t) -> 288.15)   # isothermal
    set!(parent.specific_humidity, (x, y, z, t) -> 0.0)      # dry

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

    # u falls through to the NormalFlowBoundaryCondition default.
    for side in (:west, :east, :south, :north)
        @test getproperty(bcs.u, side).classification isa Oceananigans.BoundaryConditions.NormalFlow
        @test getproperty(bcs.T, side).classification isa Oceananigans.BoundaryConditions.Value
    end

    # Passing `schemes` for a non-NormalFlowBC field must error.
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
                                     variables = (u = parent.velocities.u,   # NormalFlow (Face)
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

# The era5_breeze telescoping nest can't use a live AbstractField BC source (it fails GPU
# codegen — see the prognostic-parent regression above), so it drives the inner child from a
# "rolling FieldTimeSeries": each boundary variable is a 2-slot FTS on the parent grid whose both
# slots are overwritten from the live parent field every step. An FTS survives Adapt as a
# FlavorOfFTS (GPU-clean), and the wide [0, 1e9] time bracket makes the time-interpolation return
# the current state at any clock time. This guards the rolling idiom: a refresh propagates the
# live field into the FTS, and time-interpolation returns the refreshed state.
@testset "Rolling FieldTimeSeries tracks a live parent field" begin
    grid = RectilinearGrid(size = (4, 4, 4), x = (0, 1), y = (0, 1), z = (0, 1),
                           topology = (Bounded, Bounded, Bounded))
    src  = CenterField(grid)
    set!(src, (x, y, z) -> x + 2y + 3z)

    times = [0.0, 1.0e9]                         # wide bracket ⇒ interpolation returns the slot value
    fts = FieldTimeSeries{Center, Center, Center}(grid, times)
    for n in 1:2
        interior(fts[n]) .= interior(src)
    end
    @test interior(fts[1]) == interior(src)
    @test interior(fts[2]) == interior(src)

    # Mutate the live field and "roll" the FTS forward; both slots follow.
    set!(src, (x, y, z) -> 10 + x)
    for n in 1:2
        interior(fts[n]) .= interior(src)
    end
    @test interior(fts[1]) == interior(src)

    # Time-interpolating anywhere inside the bracket returns the refreshed state.
    @test interior(fts[Time(123.4)]) ≈ interior(src)
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

# Unit test for the Breeze-ext helper that converts a moist thermodynamic state
# (T, qᵛ, qᶜ, qⁱ, p) into the prognostic fields Breeze's `CompressibleDynamics`
# integrates (ρ, θˡⁱ, qᵗ). Checks the dry/saturation-pressure limit exactly and
# the moist + condensate case against the documented formulas.
@testset "breeze_prognostic_state derives (ρ, θˡⁱ, qᵗ)" begin
    constants = ThermodynamicConstants()
    Rᵈ   = dry_air_gas_constant(constants)
    Rᵛ   = vapor_gas_constant(constants)
    cₚᵈ  = constants.dry_air.heat_capacity
    Lᵥ   = constants.liquid.reference_latent_heat
    Lₛ   = constants.ice.reference_latent_heat
    κ    = Rᵈ / cₚᵈ
    εfac = Rᵛ / Rᵈ - 1
    pˢᵗ  = 1e5

    grid = RectilinearGrid(size = (2, 2, 2), x = (0, 1), y = (0, 1), z = (0, 1),
                           topology = (Periodic, Periodic, Bounded))
    T  = CenterField(grid); qᵛ = CenterField(grid); qᶜ = CenterField(grid)
    qⁱ = CenterField(grid); p  = CenterField(grid)

    # Dry, p = pˢᵗ ⇒ θ = T, no latent correction ⇒ θˡⁱ = T; ρ = p/(Rᵈ T); qᵗ = 0.
    set!(T, 300.0); set!(qᵛ, 0); set!(qᶜ, 0); set!(qⁱ, 0); set!(p, pˢᵗ)
    s = breeze_prognostic_state(constants, T, qᵛ, qᶜ, qⁱ, p)
    @test all(interior(s.qᵗ) .== 0)
    @test all(isapprox.(interior(s.θˡⁱ), 300.0; rtol = 1e-12))
    @test all(isapprox.(interior(s.ρ), pˢᵗ / (Rᵈ * 300.0); rtol = 1e-12))

    # Moist + condensate, p ≠ pˢᵗ ⇒ check against the documented formulas.
    set!(T, 290.0); set!(qᵛ, 0.01); set!(qᶜ, 1e-3); set!(qⁱ, 5e-4); set!(p, 9e4)
    s2 = breeze_prognostic_state(constants, T, qᵛ, qᶜ, qⁱ, p)
    Tᵛ = 290.0 * (1 + εfac * 0.01)
    θ  = 290.0 * (pˢᵗ / 9e4)^κ
    @test all(isapprox.(interior(s2.qᵗ), 0.01 + 1e-3 + 5e-4; rtol = 1e-12))
    @test all(isapprox.(interior(s2.ρ), 9e4 / (Rᵈ * Tᵛ); rtol = 1e-10))
    @test all(isapprox.(interior(s2.θˡⁱ), θ * (1 - (Lᵥ * 1e-3 + Lₛ * 5e-4) / (cₚᵈ * 290.0)); rtol = 1e-10))
    @test all(interior(s2.θˡⁱ) .< θ)   # condensate loading lowers θˡⁱ below the dry θ
end

# Integration test for the example's production path: a Breeze `AtmosphereModel`
# driven as a `NestedSimulation` child via the direct-wiring primitives
# (`atmosphere_simulation(…).model` + `parent_boundary_conditions`). Exercises the
# #220 contract that `atmosphere_simulation` returns a `Simulation` whose `.model`
# is an `AbstractModel` suitable for `NestedModel`, and steps it a few iterations.
@testset "Breeze AtmosphereModel as a NestedSimulation child on $(arch)" for arch in test_architectures
    # Parent: a 3D PrescribedAtmosphere strictly bracketing the child,
    # holding a uniform state. Velocity slots carry momentum (ρu, ρv) per the
    # Breeze nesting convention; density-weighted scalar FTSs drive the rest.
    parent_grid = RectilinearGrid(arch; size = (12, 12, 8),
                                  x = (-3000, 3000), y = (-3000, 3000), z = (-200, 2200),
                                  topology = (Bounded, Bounded, Bounded))
    times  = [0.0, 100.0]
    parent = PrescribedAtmosphere(parent_grid, times)
    set!(parent.velocities.u, (x, y, z, t) -> 1.0)
    set!(parent.velocities.v, (x, y, z, t) -> 0.0)

    ρ̄, θ̄ = 1.0, 288.0
    ρ_fts   = FieldTimeSeries{Center, Center, Center}(parent_grid, times); fill!(ρ_fts.data,   ρ̄)
    ρθ_fts  = FieldTimeSeries{Center, Center, Center}(parent_grid, times); fill!(ρθ_fts.data,  ρ̄ * θ̄)
    ρqᵉ_fts = FieldTimeSeries{Center, Center, Center}(parent_grid, times); fill!(ρqᵉ_fts.data, 0.0)

    child_grid = RectilinearGrid(arch; size = (8, 8, 8),
                                 x = (-2000, 2000), y = (-2000, 2000), z = (0, 2000),
                                 halo = (5, 5, 5), topology = (Bounded, Bounded, Bounded))

    bcs = parent_boundary_conditions(child_grid;
              variables = (ρu = parent.velocities.u, ρv = parent.velocities.v,
                           ρ = ρ_fts, ρe = ρθ_fts, ρqᵉ = ρqᵉ_fts),
              sides     = (:west, :east, :south, :north),
              bc_types  = (ρ = ValueBoundaryCondition, ρe = ValueBoundaryCondition, ρqᵉ = ValueBoundaryCondition))

    # No ESM coupling here, so override the coupling bottom-flux BCs with Dirichlet placeholders.
    bcs = merge(bcs, (; ρe  = FieldBoundaryConditions(west = bcs.ρe.west,  east = bcs.ρe.east,
                                                      south = bcs.ρe.south, north = bcs.ρe.north,
                                                      bottom = ValueBoundaryCondition(ρ̄ * θ̄)),
                        ρqᵉ = FieldBoundaryConditions(west = bcs.ρqᵉ.west,  east = bcs.ρqᵉ.east,
                                                      south = bcs.ρqᵉ.south, north = bcs.ρqᵉ.north,
                                                      bottom = ValueBoundaryCondition(0.0))))

    # #220: `atmosphere_simulation` returns a `Simulation`; its `.model` is the child model.
    child_sim = atmosphere_simulation(child_grid; boundary_conditions = bcs,
                                      dynamics = CompressibleDynamics(surface_pressure = 1e5))
    @test child_sim isa Simulation
    child = child_sim.model
    @test child isa Breeze.AtmosphereModel

    set!(child; ρ = ρ̄, u = 1.0, v = 0.0, qᵗ = 0.0, θˡⁱ = θ̄)

    nested = NestedSimulation(parent, child; Δt = 0.1, stop_iteration = 2, verbose = false)
    run!(nested)

    @test child.clock.iteration == 2
    @test parent.clock.time ≈ child.clock.time
    @test all(isfinite, Array(interior(child.velocities.u)))
    @test all(isfinite, Array(interior(child.dynamics.density)))
end

# Coupling a Breeze `CompressibleDynamics` atmosphere to a `SlabLand` via
# `AtmosphereLandModel`, then wrapping it as a `NestedSimulation` child. The coupled
# `update_state!` runs `interpolate_state!`, which must handle compressible dynamics
# (prognostic density, no anelastic reference state) via the Breeze ext's
# `dynamics_density`/`surface_pressure` accessors. Construction-level: stepping the
# coupled child awaits a Breeze energy-flux/qᵛ fix for the compressible path.
@testset "AtmosphereLandModel (compressible Breeze) as a NestedSimulation child on $(arch)" for arch in test_architectures
    atmos_grid = RectilinearGrid(arch; size = (8, 8, 16),
                                 x = (0, 8000), y = (0, 8000), z = (0, 8000),
                                 halo = (5, 5, 5), topology = (Periodic, Periodic, Bounded))
    land_grid = RectilinearGrid(arch; size = (8, 8), x = (0, 8000), y = (0, 8000),
                                halo = (atmos_grid.Hx, atmos_grid.Hy), topology = (Periodic, Periodic, Flat))

    atmos = atmosphere_simulation(atmos_grid; dynamics = CompressibleDynamics(surface_pressure = 1e5))
    set!(atmos.model; ρ = 1.2, θˡⁱ = 288.0, qᵗ = 0.0)

    land = SlabLand(land_grid)
    set!(land.temperature, 288.0)
    set!(land.water_storage, 50.0)
    Oceananigans.TimeSteppers.update_state!(land)

    alm = AtmosphereLandModel(atmos, land)            # radiation = nothing (radiatively decoupled)

    parent = PrescribedAtmosphere(atmos_grid, [0.0, 100.0];
                                  thermodynamics_parameters = nothing)
    nested = NestedSimulation(parent, alm; Δt = 0.05, stop_iteration = 2)
    @test nested isa Simulation                       # NestedModel accepted the coupled child
    @test all(isfinite, Array(interior(atmos.model.dynamics.density)))
end

# The on-the-fly nesting BCs (`ParentStateBoundary`) interpolate the parent's raw state at the
# child's boundary nodes. A child grid legitimately extends *below* the parent's lowest level (ERA5
# pressure levels don't reach the surface), so (1) the source-bracket check must NOT reject the
# vertical extent and (2) the vertical interpolation must clamp to the parent's edge value there.
# With a parent temperature linear in z, in-range nodes recover it exactly and out-of-range nodes
# return the nearest edge value. CPU-only: `getbc` is queried on the host (GPU is exercised by the
# model-attach integration tests above).
@testset "ParentStateBoundary: vertical interpolation clamps past the parent's z-extent" begin
    parent_grid = RectilinearGrid(CPU(); size = (8, 8, 4), x = (-1.5, 1.5), y = (-1.5, 1.5),
                                  z = (0.1, 0.9), topology = (Bounded, Bounded, Bounded))   # centers 0.2…0.8
    Tfts = FieldTimeSeries{Center, Center, Center}(parent_grid, [0.0, 1.0])
    set!(Tfts, (x, y, z, t) -> 100 + 10z)
    fill_halo_regions!(Tfts)

    # Single-source carrier whose transform just returns the interpolated temperature.
    psb = ParentStateBoundary((; T = Tfts), (T = identity,), s -> s.T)

    # Child spans z ∈ [-0.2, 1.2] — below AND above the parent's center range [0.2, 0.8].
    child_grid = RectilinearGrid(CPU(); size = (4, 4, 4), x = (-1, 1), y = (-1, 1),
                                 z = (-0.2, 1.2), topology = (Bounded, Bounded, Bounded))
    loc = (Center(), Center(), Center())

    # Regularization must NOT throw despite the child z-extent exceeding the parent's.
    west = regularize_boundary_condition(psb, child_grid, loc, 1, LeftBoundary)

    zc = [znode(1, 2, k, child_grid, Center(), Center(), Center()) for k in 1:4]
    expected = [100 + 10 * clamp(z, 0.2, 0.8) for z in zc]   # exact in-range; clamped to edges outside
    got = [getbc(west, 2, k, child_grid, nothing) for k in 1:4]

    @test got ≈ expected rtol = 1e-6
    @test got[1] ≈ 102      # below the lowest level (z = -0.025) → clamped to T(0.2)
    @test got[4] ≈ 108      # above the highest level (z = 1.025) → clamped to T(0.8)
end

# The Breeze-ext `StateExchanger` holds the child prognostics as a 2-level FieldTimeSeries bracketing
# the child clock (memory-O(1) in time); `exchange_state!` advances that window as the clock crosses a
# parent time interval, recomputing the two resident levels from the parent. This guards the cycling:
# the window `start` advances 1→2 crossing into the 2nd interval and back to 1, with finite + physical
# prognostics throughout. Uses a synthetic `PrescribedAtmosphere` parent (no ERA5 download); here its
# `pressure` is an FTS (the ERA5 parent uses a static `Field`) — the exchanger handles both.
@testset "StateExchanger: 2-level window cycles across parent intervals on $(arch)" for arch in test_architectures
    ext = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)

    parent_grid = RectilinearGrid(arch; size = (8, 8, 4), x = (-1.5, 1.5), y = (-1.5, 1.5),
                                  z = (0, 1), topology = (Bounded, Bounded, Bounded))
    times  = [0.0, 1.0, 2.0]                                   # 3 levels ⇒ 2 intervals
    parent = PrescribedAtmosphere(parent_grid, times)
    set!(parent.temperature,       (x, y, z, t) -> 280 + t)    # time-varying ⇒ cycling changes values
    set!(parent.specific_humidity, (x, y, z, t) -> 0.005)
    set!(parent.velocities.u,      (x, y, z, t) -> 1.0)
    set!(parent.velocities.v,      (x, y, z, t) -> 0.0)
    set!(parent.pressure,          (x, y, z, t) -> 9.0e4)

    constants = ThermodynamicConstants()
    exchanger = ext.state_exchanger(parent, 1.0e5, constants; condensates = (qᶜˡ = nothing, qᶜⁱ = nothing))
    prog      = exchanger.prognostic
    exchange  = NumericalEarth.NestedModels.exchange_state!

    # Initial window brackets t = times[1] ⇒ start = 1.
    @test prog.ρᵈ.backend.start == 1
    @test all(isfinite, Array(interior(prog.ρθ[1])))

    # Cross into the 2nd interval [1, 2] ⇒ the derived window cycles forward to start = 2.
    exchange(exchanger, 1.5)
    @test prog.ρᵈ.backend.start == 2
    @test all(isfinite, Array(interior(prog.ρθ[2])))
    θ = Array(interior(prog.ρθ[2])) ./ Array(interior(prog.ρᵈ[2]))
    @test all(250 .< θ .< 400)                                 # physical potential temperature

    # Cycle back to the 1st interval.
    exchange(exchanger, 0.0)
    @test prog.ρᵈ.backend.start == 1
end
