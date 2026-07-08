include("runtests_setup.jl")

using NumericalEarth
using NumericalEarth.NestedModels: parent_boundary_conditions, nested_atmosphere_model
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.OutputReaders: interpolating_time_indices, memory_index
using Oceananigans.Units: Time
using Oceananigans.Fields: location
using Oceananigans.BoundaryConditions: ValueBoundaryCondition, FieldBoundaryConditions, fill_halo_regions!
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
    s = breeze_prognostic_state(constants, pˢᵗ, T, qᵛ, qᶜ, qⁱ, p)
    @test all(interior(s.qᵗ) .== 0)
    @test all(isapprox.(interior(s.θˡⁱ), 300.0; rtol = 1e-12))
    @test all(isapprox.(interior(s.ρ), pˢᵗ / (Rᵈ * 300.0); rtol = 1e-12))

    # Moist + condensate, p ≠ pˢᵗ ⇒ check against the documented formulas.
    set!(T, 290.0); set!(qᵛ, 0.01); set!(qᶜ, 1e-3); set!(qⁱ, 5e-4); set!(p, 9e4)
    s2 = breeze_prognostic_state(constants, pˢᵗ, T, qᵛ, qᶜ, qⁱ, p)
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
    @test all(isfinite, Array(interior(child.dynamics.total_density)))
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
    @test all(isfinite, Array(interior(atmos.model.dynamics.total_density)))
end

@testset "Breeze nested_atmosphere_model defaults to a moving derived window on CPU()" begin
    parent_grid = LatitudeLongitudeGrid(CPU(); size = (8, 8, 4),
                                         longitude = (-1.5, 1.5), latitude = (-1.5, 1.5),
                                         z = (0, 1), topology = (Bounded, Bounded, Bounded))
    times = collect(0.0:1.0:4.0)
    parent = PrescribedAtmosphere(parent_grid, times)
    set!(parent.temperature,       (x, y, z, t) -> 280 + t)
    set!(parent.specific_humidity, (x, y, z, t) -> 0.005)
    set!(parent.velocities.u,      (x, y, z, t) -> 1.0)
    set!(parent.velocities.v,      (x, y, z, t) -> 0.0)
    set!(parent.pressure,          (x, y, z, t) -> 9.0e4)

    child_grid = LatitudeLongitudeGrid(CPU(); size = (8, 8, 8),
                                        longitude = (-1, 1), latitude = (-1, 1), z = (0, 1000),
                                        halo = (5, 5, 5), topology = (Bounded, Bounded, Bounded))

    nested = nested_atmosphere_model(parent, child_grid; parent_condensates = (qᶜˡ = nothing, qᶜⁱ = nothing))
    @test length(parent.temperature.times) == length(times)
    @test length(times) > 3
    @test length(nested.exchanger.prognostic.ρᵈ.backend) == 3
end

# The Breeze-ext `StateExchanger` holds the child prognostics as a 3-level FieldTimeSeries bracketing
# the child clock (memory-O(1) in time); `exchange_state!` positions that window one level below the
# bracket of `t + Δt` so it spans a node-crossing step, recomputing the resident levels from the parent.
# This guards the cycling: the window `start` advances as the clock crosses parent intervals and back,
# with finite + physical prognostics throughout. Uses a synthetic `PrescribedAtmosphere` parent (no ERA5
# download); here `pressure` is an FTS (the ERA5 parent uses a static `Field`) — the exchanger handles both.
@testset "StateExchanger: 3-level window cycles across parent intervals on $(arch)" for arch in test_architectures
    ext = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)

    parent_grid = RectilinearGrid(arch; size = (8, 8, 4), x = (-1.5, 1.5), y = (-1.5, 1.5),
                                  z = (0, 1), topology = (Bounded, Bounded, Bounded))
    times  = [0.0, 1.0, 2.0, 3.0, 4.0]                         # 5 levels ⇒ the 3-level window can cycle
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

    # Initial fill at t = times[1] = 0 ⇒ window start = 1 (resident levels 1, 2, 3).
    @test prog.ρᵈ.backend.start == 1
    @test all(isfinite, Array(interior(prog.ρθ[1])))

    # Cross to a later interval ⇒ the window cycles forward (start = clamp(n₁-1, 1, N-2)).
    exchange(exchanger, 2.5)                                    # bracket n₁ = 3 ⇒ start = 2 (levels 2, 3, 4)
    @test prog.ρᵈ.backend.start == 2
    @test all(isfinite, Array(interior(prog.ρθ[3])))
    θ = Array(interior(prog.ρθ[3])) ./ Array(interior(prog.ρᵈ[3]))
    @test all(250 .< θ .< 400)                                 # physical potential temperature

    # Cycle back toward the 1st interval.
    exchange(exchanger, 0.5)                                    # bracket n₁ = 1 ⇒ start = 1
    @test prog.ρᵈ.backend.start == 1

    # reconstruct_parent_state reads the parent's FULL-memory fields, not the windowed levels: with the
    # window parked forward, a reconstruction at t = 0 still recovers the parent's t = 0 state
    # (θˡⁱ = T (pˢᵗ/p)^κ with T = 280 + t, condensate-free), proving no residency aliasing.
    reconstruct = NumericalEarth.NestedModels.reconstruct_parent_state
    κ = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    exchange(exchanger, 2.5)                                    # park the window forward
    θ₀ = Array(interior(reconstruct(exchanger, 0.0).θˡⁱ))
    θ₃ = Array(interior(reconstruct(exchanger, 3.0).θˡⁱ))
    @test all(θ₀ .≈ 280 * (1e5 / 9e4)^κ)
    @test all(θ₃ .≈ 283 * (1e5 / 9e4)^κ)
end

@testset "StateExchanger: moving-window interpolation never aliases nonresident time slots on $(arch)" for arch in test_architectures
    ext = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)

    parent_grid = RectilinearGrid(arch; size = (8, 8, 4), x = (-1.5, 1.5), y = (-1.5, 1.5),
                                  z = (0, 1), topology = (Bounded, Bounded, Bounded))
    times  = [0.0, 1.0, 2.0, 3.0, 4.0]
    parent = PrescribedAtmosphere(parent_grid, times)
    set!(parent.temperature,       (x, y, z, t) -> 280 + 5t)
    set!(parent.specific_humidity, (x, y, z, t) -> 0.005)
    set!(parent.velocities.u,      (x, y, z, t) -> 1.0)
    set!(parent.velocities.v,      (x, y, z, t) -> 0.0)
    set!(parent.pressure,          (x, y, z, t) -> 9.0e4)

    constants = ThermodynamicConstants()
    exchanger = ext.state_exchanger(parent, 1.0e5, constants; condensates = (qᶜˡ = nothing, qᶜⁱ = nothing))
    prog      = exchanger.prognostic
    exchange  = NumericalEarth.NestedModels.exchange_state!

    exchange(exchanger, 2.1) # start = 2, resident time indices are 2, 3, 4.
    fts = prog.ρθ
    @test fts.backend.start == 2
    @test length(fts.backend) == 3

    # A runtime boundary/relaxation kernel uses elementwise interpolation, which computes memory slots
    # directly under @inbounds. A query whose time bracket touches global index 5 must not map to slot 4
    # of this 3-slot moving window; that is an out-of-bounds GPU read, unlike the full-memory path.
    _, n₁, n₂ = interpolating_time_indices(fts.time_indexing, fts.times, 3.5)
    m₁ = memory_index(fts.backend, fts.time_indexing, length(fts.times), n₁)
    m₂ = memory_index(fts.backend, fts.time_indexing, length(fts.times), n₂)
    @test 1 ≤ m₁ ≤ length(fts.backend)
    @test 1 ≤ m₂ ≤ length(fts.backend)
end

@testset "StateExchanger: moving-window runtime queries match full-memory queries on CPU()" begin
    ext = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)

    function parent_for_exchanger_equivalence()
        parent_grid = RectilinearGrid(CPU(); size = (8, 8, 4), x = (-1.5, 1.5), y = (-1.5, 1.5),
                                      z = (0, 1), topology = (Bounded, Bounded, Bounded))
        times = collect(0.0:1.0:6.0)
        parent = PrescribedAtmosphere(parent_grid, times)
        set!(parent.temperature,       (x, y, z, t) -> 280 + 5t + 3x - 2y + z)
        set!(parent.specific_humidity, (x, y, z, t) -> 0.005 + 1e-4*t + 1e-5*x)
        set!(parent.velocities.u,      (x, y, z, t) -> 1 + 0.1t + x)
        set!(parent.velocities.v,      (x, y, z, t) -> -0.2 + 0.05t + y)
        set!(parent.pressure,          (x, y, z, t) -> 9.0e4 + 10t + 100x - 50y)
        return parent
    end

    constants = ThermodynamicConstants()
    condensates = (qᶜˡ = nothing, qᶜⁱ = nothing)
    moving = ext.state_exchanger(parent_for_exchanger_equivalence(), 1.0e5, constants; condensates,
                                 time_indices_in_memory = 3)
    full = ext.state_exchanger(parent_for_exchanger_equivalence(), 1.0e5, constants; condensates,
                               time_indices_in_memory = 7)
    exchange = NumericalEarth.NestedModels.exchange_state!
    field_names = (:ρᵈ, :ρθ, :ρqᵛ, :ρu, :ρv)
    indices = ((1, 1, 1), (2, 2, 2), (8, 8, 4))

    for step_end_time in (0.5, 1.1, 2.1, 3.1, 4.1, 5.1)
        exchange(moving, step_end_time)
        exchange(full, step_end_time)

        for query_time in (max(0, step_end_time - 0.2), step_end_time)
            for name in field_names
                moving_fts = getproperty(moving.prognostic, name)
                full_fts = getproperty(full.prognostic, name)

                for I in indices
                    moving_value = moving_fts[I..., Time(query_time)]
                    full_value = full_fts[I..., Time(query_time)]
                    @test isapprox(moving_value, full_value; rtol=1e-5, atol=1e-6)
                end
            end
        end
    end
end

# Temporal-seam correctness of the ON-THE-FLY (windowed FTS) interpolation the child's boundary
# conditions + Davies relaxation actually query. `NestedModel.time_step!` refreshes the exchanger at the
# step END (`t + Δt`), so on a step that CROSSES a parent node the derived 2-level window sits at
# `[node, node+1]` while the child's start-of-step sub-stages sample at `t < node` — a start-side query
# one interval BELOW the resident window. That query must return FINITE, PHYSICAL values (a clean
# extrapolation/clamp toward the window edge), NOT window-eviction/wrap garbage. This is the exact
# hourly-seam the ERA5 nest crosses every hour; the reconstruct_parent_state test above covers the
# full-memory diagnostic path, but the RUNTIME path is the windowed `fts[Time(t)]` probed here.
@testset "StateExchanger: windowed query across a parent node is finite + physical on $(arch)" for arch in test_architectures
    ext = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)

    parent_grid = RectilinearGrid(arch; size = (8, 8, 4), x = (-1.5, 1.5), y = (-1.5, 1.5),
                                  z = (0, 1), topology = (Bounded, Bounded, Bounded))
    times  = [0.0, 1.0, 2.0, 3.0, 4.0]                         # 5 levels ⇒ the 3-level window is a strict subset
    parent = PrescribedAtmosphere(parent_grid, times)
    set!(parent.temperature,       (x, y, z, t) -> 280 + 5t)   # linear in t ⇒ a correct interpolation is exact
    set!(parent.specific_humidity, (x, y, z, t) -> 0.005)
    set!(parent.velocities.u,      (x, y, z, t) -> 1.0)
    set!(parent.velocities.v,      (x, y, z, t) -> 0.0)
    set!(parent.pressure,          (x, y, z, t) -> 9.0e4)

    constants = ThermodynamicConstants()
    exchanger = ext.state_exchanger(parent, 1.0e5, constants; condensates = (qᶜˡ = nothing, qᶜⁱ = nothing))
    prog      = exchanger.prognostic
    exchange  = NumericalEarth.NestedModels.exchange_state!
    κ = dry_air_gas_constant(constants) / constants.dry_air.heat_capacity
    θtrue(t) = (280 + 5t) * (1e5 / 9e4)^κ

    # Baseline: an in-window query is correct.
    exchange(exchanger, 0.5)                                    # bracket n₁ = 1 ⇒ start = 1 (levels 1,2,3)
    @test prog.ρᵈ.backend.start == 1
    θin = Array(interior(prog.ρθ[Time(0.5)])) ./ Array(interior(prog.ρᵈ[Time(0.5)]))
    @test all(isapprox.(θin, θtrue(0.5); rtol = 1e-4))

    # CROSSING STEP over the node at t = 2.0: bracket t+Δt = 2.1 ⇒ window advances to start = 2 (resident
    # levels 2,3,4 = times [1,2,3]). The child's start-of-step query at t = 1.9 (< node) sits one interval
    # BELOW where a 2-level window would sit — with a 2-level window that read a stale/wrong target (the
    # hourly-seam kick); the 3-level window keeps [1.9's bracket] resident so the target stays correct.
    exchange(exchanger, 2.1)
    @test prog.ρᵈ.backend.start == 2

    ρθq = Array(interior(prog.ρθ[Time(1.9)]))
    ρdq = Array(interior(prog.ρᵈ[Time(1.9)]))
    @test all(isfinite, ρθq)                    # a 2-level window would read an evicted level here
    @test all(isfinite, ρdq)
    @test all(0.05 .< ρdq .< 2.0)               # physical dry density (not a stale/aliased value)
    θq = ρθq ./ ρdq
    @test all(250 .< θq .< 400)                 # physical θ
    @test all(isapprox.(θq, θtrue(1.9); rtol = 1e-4))   # linear-in-t ⇒ exact; a stale target would miss by ~5 K
end

# End-to-end guard for the moving-window fix: step the FULL coupled Breeze child across a derived-window
# MOVE and assert its prognostics stay finite. The exchanger/`memory_index` @testsets above check the
# derived FTS in isolation; this exercises the RUNTIME path that actually broke — the child's BC + Davies
# forcing kernels reading the derived FTS elementwise while the window rolls `start` 1→2 mid-run. A uniform
# synthetic parent with short (4 s) node spacing reaches the move (crossing node #3 at t = 8 s) in a few
# acoustically-stable steps; the move is what triggered the blow-up, not the parent's data, so the minimal
# parent reproduces it. Pre-fix (generic cyclic `memory_index` indexing past the window's storage) the
# child NaNs the step `start` advances 1→2 (`InexactError: Int64(NaN)`).
@testset "Nested child survives a derived-window move (moving-window regression) on $(arch)" for arch in test_architectures
    ext   = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
    times = [0.0, 4.0, 8.0, 12.0, 16.0]        # 5 levels ⇒ the 3-level window moves when crossing node #3

    parent_grid = LatitudeLongitudeGrid(arch; size = (12, 12, 8),
                                        longitude = (-2, 2), latitude = (34.6, 38.6),
                                        z = (0, 16000), halo = (5, 5, 5),
                                        topology = (Bounded, Bounded, Bounded))
    parent = PrescribedAtmosphere(parent_grid, times)
    set!(parent.temperature,       (λ, φ, z, t) -> 288 - 6.5e-3 * z + 1e-3 * t)
    set!(parent.specific_humidity, (λ, φ, z, t) -> 0.006)
    set!(parent.velocities.u,      (λ, φ, z, t) -> 8)
    set!(parent.velocities.v,      (λ, φ, z, t) -> 0)
    set!(parent.pressure,          (λ, φ, z, t) -> 1e5 * exp(-z / 8000))

    child_grid = LatitudeLongitudeGrid(arch; size = (8, 8, 8),
                                       longitude = (-1, 1), latitude = (35.6, 37.6),
                                       z = (0, 16000), halo = (5, 5, 5),
                                       topology = (Bounded, Bounded, Bounded))

    # Default microphysics: with CloudMicrophysics unloaded (as in the test env) this is the Breeze-native
    # `SaturationAdjustment(WarmPhaseEquilibrium())` — no extra dependency needed to exercise the seam.
    model = nested_atmosphere_model(parent, child_grid;
                relaxation_rate = 1/300, relaxation_width = 3, surface_pressure = 1e5,
                coriolis = nothing, terrain = nothing, parent_condensates = nothing)
    ext.initialize_nested_child!(model, nothing, first(times), ""; balancer = false)

    sim = Simulation(model; Δt = 0.5, stop_time = 12.0)
    conjure_time_step_wizard!(sim, IterationInterval(1); cfl = 0.3, max_Δt = 2.0)
    run!(sim)

    @test model.exchanger.prognostic.ρᵈ.backend.start > 1        # actually crossed the window move
    @test model.clock.time ≥ 12.0 - 2.0                          # reached the end (no NaN Δt stall)
    for (name, field) in pairs(prognostic_fields(model.child))
        @test all(isfinite, Array(interior(field)))
    end
end
