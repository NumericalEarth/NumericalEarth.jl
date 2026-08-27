include("runtests_setup.jl")

using NumericalEarth
using NumericalEarth.NestedModels: parent_boundary_conditions, nested_atmosphere_model
using Oceananigans
using Oceananigans: prognostic_fields
using Oceananigans.OutputReaders: interpolating_time_indices, memory_index
using Oceananigans.Units: Time
using Oceananigans.Fields: location
using Oceananigans.BoundaryConditions: ValueBoundaryCondition, FieldBoundaryConditions, fill_halo_regions!
using Oceananigans.Forcings: MultipleForcings
using Breeze
using Breeze: ThermodynamicConstants, dry_air_gas_constant, vapor_gas_constant, CompressibleDynamics,
              SpecificForcing, SaturationAdjustment, WarmPhaseEquilibrium, moisture_prognostic_name,
              adjustment_saturation_specific_humidity
using Breeze.Thermodynamics: MoistureMassFractions, LiquidIcePotentialTemperatureState, temperature,
                             with_temperature
using Breeze.Microphysics: compute_temperature
using Test

@testset "PrescribedAtmosphere: grid vertical topology selects surface vs volumetric fields" begin
    # A `Flat` vertical gives a surface atmosphere: u, v and 2D temperature, specific humidity
    # and pressure, with no gas or microphysical species.
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
    # The parent holds the analytic Lamb-Oseen state at a few coarse time snapshots on a 3D grid,
    # and extends strictly beyond the child so it brackets every child boundary node.
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

    # The vortex center at t=0 sits at (x₀_LO, y₀_LO) = (-0.5, 0). The interior
    # near the center should retain a recognizable vortex signature after a
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

# A live model parent (rather than a PrescribedAtmosphere or FieldTimeSeries) drives the child
# through `Interpolated` BCs, including a Center `ValueBoundaryCondition`. On GPU the source
# field `Adapt`s to a bare data array inside the halo-fill kernel.
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
    run!(nested)

    @test child.clock.iteration == 3
    @test parent.clock.time ≈ child.clock.time
    @test all(isfinite, Array(interior(child.velocities.u)))
    @test all(isfinite, Array(interior(child.tracers.c)))
end

# A "rolling FieldTimeSeries" drives a child from a live parent field: a 2-slot FTS whose slots
# are both overwritten from the field every step, with a time bracket wide enough that
# interpolation returns the refreshed state at any clock time.
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

# An `Interpolated` Value BC samples the source at the boundary face, not at the child field's
# center node. With a source exactly linear in the boundary-normal coordinate, the reconstructed
# face value ½(halo + first interior) equals the source at the face. Covers both normal directions.
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

@testset "breeze_prognostic_state derives (ρ, θˡⁱ, qᵗ)" begin
    constants = ThermodynamicConstants()
    Rᵈ   = dry_air_gas_constant(constants)
    Rᵛ   = vapor_gas_constant(constants)
    cₚᵈ  = constants.dry_air.heat_capacity
    Lᵥ   = constants.liquid.reference_latent_heat
    Lₛ   = constants.ice.reference_latent_heat
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

    # Moist + condensate, p ≠ pˢᵗ: `temperature` must return T exactly.
    set!(T, 290.0); set!(qᵛ, 0.01); set!(qᶜ, 1e-3); set!(qⁱ, 5e-4); set!(p, 9e4)
    s2 = breeze_prognostic_state(constants, pˢᵗ, T, qᵛ, qᶜ, qⁱ, p)
    Rᵐ = (1 - 0.01 - 1e-3 - 5e-4) * Rᵈ + 0.01 * Rᵛ   # mixture gas constant: condensate loads the mixture
    @test all(isapprox.(interior(s2.qᵗ), 0.01 + 1e-3 + 5e-4; rtol = 1e-12))
    @test all(isapprox.(interior(s2.ρ), 9e4 / (Rᵐ * 290.0); rtol = 1e-10))

    q  = MoistureMassFractions(0.01, 1e-3, 5e-4)
    θˡⁱ = Array(interior(s2.θˡⁱ))[1, 1, 1]
    @test temperature(LiquidIcePotentialTemperatureState(θˡⁱ, q, pˢᵗ, 9e4), constants) ≈ 290.0 rtol = 1e-12
    @test all(interior(s2.θˡⁱ) .< 290.0 * (pˢᵗ / 9e4)^(Rᵈ / cₚᵈ))   # condensate lowers θˡⁱ below the dry θ

    # A dry-κ formula does not invert here; guards against a revert to a hand-rolled definition.
    θdry = 290.0 * (pˢᵗ / 9e4)^(Rᵈ / cₚᵈ) * (1 - (Lᵥ * 1e-3 + Lₛ * 5e-4) / (cₚᵈ * 290.0))
    @test !isapprox(temperature(LiquidIcePotentialTemperatureState(θdry, q, pˢᵗ, 9e4), constants), 290.0;
                    atol = 1e-3)
end

# The specific members (`θ`, `u`, `v`) are the intensive partners of the density-weighted ones.
@testset "state exchanger: specific θ/u/v are the intensive partners of ρθ/ρu/ρv" begin
    ext = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
    grid = RectilinearGrid(size = (8, 8, 4), x = (-1, 1), y = (-1, 1), z = (0, 1),
                           topology = (Bounded, Bounded, Bounded))
    parent = PrescribedAtmosphere(grid, [0.0, 1.0, 2.0])   # ≥3 times for the exchanger's moving window
    set!(parent.temperature,       (x, y, z, t) -> 290 + 5z)
    set!(parent.specific_humidity, (x, y, z, t) -> 0.005)
    set!(parent.pressure,          (x, y, z, t) -> 9e4)
    set!(parent.velocities.u,      (x, y, z, t) -> 10x)
    set!(parent.velocities.v,      (x, y, z, t) -> -5y)

    ex = ext.state_exchanger(parent, 1e5, ThermodynamicConstants();
                             condensates = (qᶜˡ = nothing, qʳ = nothing, qᶜⁱ = nothing, qˢ = nothing))
    es = ex.prognostic
    @test es.ρθ[1] ≈ es.ρᵈ[1] .* es.θ[1]   # ρθ = ρᵈ·θ
    @test es.ρu[1] ≈ es.ρᵈ[1] .* es.u[1]   # ρu = ρᵈ·u
    @test es.ρv[1] ≈ es.ρᵈ[1] .* es.v[1]
    @test es.u[1]  ≈ parent.velocities.u[1]          # u/v are verbatim parent copies
    @test es.v[1]  ≈ parent.velocities.v[1]
end

# The exchanger's moisture slot holds what `moisture_prognostic_name` binds: `:ρqᵛ` is true vapor,
# `:ρqᵉ` is vapor + cloud condensate (qᵗ less precipitation).
@testset "state exchanger: the moisture slot matches the child's moisture_prognostic_name" begin
    ext = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
    constants = ThermodynamicConstants()
    pˢᵗ, T₀, p₀ = 1e5, 283.15, 9e4
    qᵛ₀, qᶜˡ₀, qᶜⁱ₀, qʳ₀, qˢ₀ = 8.0e-3, 5.0e-4, 1.0e-4, 2.0e-4, 5.0e-5

    grid = RectilinearGrid(size = (4, 4, 4), x = (-1, 1), y = (-1, 1), z = (0, 1),
                           topology = (Bounded, Bounded, Bounded))
    parent = PrescribedAtmosphere(grid, [0.0, 1.0, 2.0])
    set!(parent.temperature,       (x, y, z, t) -> T₀)
    set!(parent.specific_humidity, (x, y, z, t) -> qᵛ₀)
    set!(parent.pressure,          (x, y, z, t) -> p₀)
    set!(parent.velocities.u,      (x, y, z, t) -> 1.0)
    set!(parent.velocities.v,      (x, y, z, t) -> 0.0)
    uniform(v) = (f = CenterField(grid); set!(f, (x, y, z) -> v); f)
    condensates = (qᶜˡ = uniform(qᶜˡ₀), qʳ = uniform(qʳ₀), qᶜⁱ = uniform(qᶜⁱ₀), qˢ = uniform(qˢ₀))

    # Every hydrometeor loads the mixture.
    Rᵈ, Rᵛ = dry_air_gas_constant(constants), vapor_gas_constant(constants)
    qᵗ = qᵛ₀ + qᶜˡ₀ + qʳ₀ + qᶜⁱ₀ + qˢ₀
    ρ  = p₀ / (((1 - qᵗ) * Rᵈ + qᵛ₀ * Rᵛ) * T₀)

    # `:ρqᵛ` ⇒ true vapor only.
    exᵛ = ext.state_exchanger(parent, pˢᵗ, constants; condensates, moisture_name = :ρqᵛ)
    @test all(isapprox.(interior(exᵛ.prognostic.ρqᵛᵉ[1]), ρ * qᵛ₀; rtol = 1e-12))

    # `:ρqᵉ` ⇒ vapor + CLOUD condensate; precipitation stays out (qᵉ = qᵗ − qʳ − qˢ).
    exᵉ = ext.state_exchanger(parent, pˢᵗ, constants; condensates, moisture_name = :ρqᵉ)
    @test all(isapprox.(interior(exᵉ.prognostic.ρqᵛᵉ[1]), ρ * (qᵛ₀ + qᶜˡ₀ + qᶜⁱ₀); rtol = 1e-12))

    # The two differ by exactly the cloud condensate the vapor-only write drops.
    @test all(isapprox.(interior(exᵉ.prognostic.ρqᵛᵉ[1]) .- interior(exᵛ.prognostic.ρqᵛᵉ[1]),
                        ρ * (qᶜˡ₀ + qᶜⁱ₀); rtol = 1e-12))

    # Saturation adjustment is the default nesting path, so it must land on `:ρqᵉ`.
    @test moisture_prognostic_name(SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())) == :ρqᵉ
end

# End-to-end closure of the handoff: invert the emitted (θˡⁱ, qᵛᵉ) pair with the child's own saturation
# adjustment and recover the temperature the parent sent.
@testset "state exchanger: the child recovers the parent temperature it was sent" begin
    ext = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
    constants = ThermodynamicConstants()
    microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
    pˢᵗ, T₀, p₀, qᶜˡ₀ = 1e5, 283.15, 9e4, 5.0e-4

    # Saturate with the adjustment's own qsat, so the state is one it leaves alone.
    qᵛ₀ = 8.0e-3
    for _ in 1:80
        qᵛ₀ = adjustment_saturation_specific_humidity(T₀, p₀, qᵛ₀ + qᶜˡ₀, constants, WarmPhaseEquilibrium())
    end

    grid = RectilinearGrid(size = (4, 4, 4), x = (-1, 1), y = (-1, 1), z = (0, 1),
                           topology = (Bounded, Bounded, Bounded))
    parent = PrescribedAtmosphere(grid, [0.0, 1.0, 2.0])
    set!(parent.temperature,       (x, y, z, t) -> T₀)
    set!(parent.specific_humidity, (x, y, z, t) -> qᵛ₀)
    set!(parent.pressure,          (x, y, z, t) -> p₀)
    set!(parent.velocities.u,      (x, y, z, t) -> 1.0)
    set!(parent.velocities.v,      (x, y, z, t) -> 0.0)
    uniform(v) = (f = CenterField(grid); set!(f, (x, y, z) -> v); f)
    # No precipitation: qʳ/qˢ carry latent heat into θˡⁱ that `qᵉ` cannot give back, since the child
    # has no rain/snow prognostic. That is a separate defect.
    condensates = (qᶜˡ = uniform(qᶜˡ₀), qʳ = nothing, qᶜⁱ = nothing, qˢ = nothing)

    Rᵈ, Rᵛ = dry_air_gas_constant(constants), vapor_gas_constant(constants)
    ρ = p₀ / (((1 - qᵛ₀ - qᶜˡ₀) * Rᵈ + qᵛ₀ * Rᵛ) * T₀)

    recovered(name) = begin
        ex = ext.state_exchanger(parent, pˢᵗ, constants; condensates, moisture_name = name)
        at(f) = Array(interior(f))[1, 1, 1]
        θˡⁱ = at(ex.prognostic.ρθ[1]) / at(ex.prognostic.ρᵈ[1])
        qᵛᵉ = at(ex.prognostic.ρqᵛᵉ[1]) / ρ
        𝒰   = LiquidIcePotentialTemperatureState(θˡⁱ, MoistureMassFractions(qᵛᵉ), pˢᵗ, p₀)
        compute_temperature(𝒰, microphysics, constants)
    end

    # With the pair the child's scheme expects, the handoff closes to solver tolerance.
    @test recovered(:ρqᵉ) ≈ T₀ atol = 1e-3

    # Vapor-only into an equilibrium slot loses a damped ℒqᶜˡ/cᵖ, ≈ 0.5 K here.
    @test T₀ - recovered(:ρqᵛ) > 0.4
    @test T₀ - recovered(:ρqᵛ) < 0.7
end

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

# Construction only: stepping the coupled child awaits a Breeze energy-flux/qᵛ fix for the
# compressible path.
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

# The Davies relaxation is keyed by the density-weighted prognostic, so a caller's own specific-key
# forcing combines with it rather than replacing it.
@testset "Davies relaxation survives a caller-supplied specific forcing" begin
    parent_grid = LatitudeLongitudeGrid(CPU(); size = (8, 8, 4),
                                        longitude = (-1.5, 1.5), latitude = (-1.5, 1.5),
                                        z = (0, 1), topology = (Bounded, Bounded, Bounded))
    parent = PrescribedAtmosphere(parent_grid, collect(0.0:1.0:2.0))
    set!(parent.temperature,       (x, y, z, t) -> 280.0)
    set!(parent.specific_humidity, (x, y, z, t) -> 0.005)
    set!(parent.velocities.u,      (x, y, z, t) -> 1.0)
    set!(parent.velocities.v,      (x, y, z, t) -> 0.0)
    set!(parent.pressure,          (x, y, z, t) -> 9.0e4)

    child_grid = LatitudeLongitudeGrid(CPU(); size = (8, 8, 8),
                                       longitude = (-1, 1), latitude = (-1, 1), z = (0, 1000),
                                       halo = (5, 5, 5), topology = (Bounded, Bounded, Bounded))

    nested = nested_atmosphere_model(parent, child_grid;
                                     relaxation_rate = 1/300,
                                     parent_condensates = (qᶜˡ = nothing, qᶜⁱ = nothing),
                                     forcing = (θ = Relaxation(rate = 1/600, target = 300.0),))

    ρθ_forcing = nested.child.forcing.ρθ
    @test ρθ_forcing isa MultipleForcings
    @test length(ρθ_forcing.forcings) == 2                       # Davies relaxation + the caller's θ forcing
    @test all(f -> f isa SpecificForcing, ρθ_forcing.forcings)   # both ρᵈ-weighted at kernel time
end

# The exchanger's 3-level window advances as the clock crosses parent intervals and back, with
# finite and physical prognostics throughout.
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
    # (T = 280 + t, condensate-free), proving no residency aliasing.
    #
    # Condensate-free is not dry: qᵛ = 0.005 makes Rᵐ/cᵖᵐ differ from Rᵈ/cᵖᵈ by ≈0.02 K here.
    reconstruct = NumericalEarth.NestedModels.reconstruct_parent_state
    θˡⁱ_of(T) = with_temperature(LiquidIcePotentialTemperatureState(0.0, MoistureMassFractions(0.005),
                                                                    1e5, 9.0e4), T, constants).potential_temperature
    exchange(exchanger, 2.5)                                    # park the window forward
    θ₀ = Array(interior(reconstruct(exchanger, 0.0).θˡⁱ))
    θ₃ = Array(interior(reconstruct(exchanger, 3.0).θˡⁱ))
    @test all(θ₀ .≈ θˡⁱ_of(280.0))
    @test all(θ₃ .≈ θˡⁱ_of(283.0))
end

# Every liquid and ice hydrometeor the parent carries — cloud liquid and rain, cloud ice and snow —
# is mass that is not dry gas, so it loads the density through the mixture gas constant
# (ρ = p / (Rᵐ T), Rᵐ = (1 − qᵗ) Rᵈ + qᵛ Rᵛ) and enters qᵗ.
@testset "StateExchanger: cloud + precipitation load the density on $(arch)" for arch in test_architectures
    ext = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
    reconstruct = NumericalEarth.NestedModels.reconstruct_parent_state

    parent_grid = RectilinearGrid(arch; size = (4, 4, 2), x = (-1, 1), y = (-1, 1),
                                  z = (0, 1), topology = (Bounded, Bounded, Bounded))
    times = [0.0, 1.0, 2.0]
    hydrometeor() = FieldTimeSeries{Center, Center, Center}(parent_grid, times)
    qᶜˡ, qʳ, qᶜⁱ, qˢ = hydrometeor(), hydrometeor(), hydrometeor(), hydrometeor()
    set!(qᶜˡ, (x, y, z, t) -> 1.0e-3)   # cloud liquid
    set!(qʳ,  (x, y, z, t) -> 1.5e-3)   # rain
    set!(qᶜⁱ, (x, y, z, t) -> 4.0e-4)   # cloud ice
    set!(qˢ,  (x, y, z, t) -> 2.0e-3)   # snow
    parent = PrescribedAtmosphere(parent_grid, times; microphysical_variables = (; qᶜˡ, qʳ, qᶜⁱ, qˢ))
    set!(parent.temperature,       (x, y, z, t) -> 280.0)
    set!(parent.specific_humidity, (x, y, z, t) -> 0.005)
    set!(parent.velocities.u,      (x, y, z, t) -> 1.0)
    set!(parent.velocities.v,      (x, y, z, t) -> 0.0)
    set!(parent.pressure,          (x, y, z, t) -> 9.0e4)

    constants = ThermodynamicConstants()
    exchanger = ext.state_exchanger(parent, 1.0e5, constants)   # default condensates ⇒ all four species
    state     = reconstruct(exchanger, 0.0)

    Rᵈ = dry_air_gas_constant(constants)
    Rᵛ = vapor_gas_constant(constants)
    qᵛ, T, p = 0.005, 280.0, 9.0e4
    qˡ = 1.0e-3 + 1.5e-3            # total liquid: cloud + rain
    qⁱ = 4.0e-4 + 2.0e-3            # total ice: cloud ice + snow
    Rᵐ = (1 - qᵛ - qˡ - qⁱ) * Rᵈ + qᵛ * Rᵛ
    ρ  = p / (Rᵐ * T)

    @test all(Array(interior(state.qᵗ))            .≈ qᵛ + qˡ + qⁱ)       # qᵗ counts every hydrometeor
    @test all(Array(interior(state.ρ))             .≈ ρ)                   # ρ = Breeze's mixture-gas EOS
    @test all(Array(interior(exchanger.prognostic.ρᵈ[1])) .≈ ρ * (1 - (qᵛ + qˡ + qⁱ)))  # kernel path

    # Dropping rain + snow (cloud-only) gives a measurably different, biased density.
    ρ_cloud_only = p / (((1 - qᵛ - 1.0e-3 - 4.0e-4) * Rᵈ + qᵛ * Rᵛ) * T)
    @test !isapprox(ρ, ρ_cloud_only; rtol = 1e-6)
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
    field_names = (:ρᵈ, :ρθ, :ρqᵛᵉ, :ρu, :ρv)
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

# The exchanger refreshes at the end of a step, so on a step that crosses a parent node the child's
# start-of-step sub-stages query the window one interval below its resident levels. Such a query must
# return finite, physical values.
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
    # Condensate-free but moist (qᵛ = 0.005): θˡⁱ carries the mixture exponent Rᵐ/cᵖᵐ.
    θtrue(t) = with_temperature(LiquidIcePotentialTemperatureState(0.0, MoistureMassFractions(0.005),
                                                                   1e5, 9.0e4),
                                280 + 5t, constants).potential_temperature

    # Baseline: an in-window query is correct.
    exchange(exchanger, 0.5)                                    # bracket n₁ = 1 ⇒ start = 1 (levels 1,2,3)
    @test prog.ρᵈ.backend.start == 1
    θin = Array(interior(prog.ρθ[Time(0.5)])) ./ Array(interior(prog.ρᵈ[Time(0.5)]))
    @test all(isapprox.(θin, θtrue(0.5); rtol = 1e-4))

    # Crossing the node at t = 2: the window advances to start = 2 (times [1, 2, 3]), so the child's
    # start-of-step query at t = 1.9 still falls inside the resident levels.
    exchange(exchanger, 2.1)
    @test prog.ρᵈ.backend.start == 2

    ρθq = Array(interior(prog.ρθ[Time(1.9)]))
    ρdq = Array(interior(prog.ρᵈ[Time(1.9)]))
    @test all(isfinite, ρθq)                    # a 2-level window would read an evicted level here
    @test all(isfinite, ρdq)
    @test all(0.05 .< ρdq .< 2.0)               # physical dry density (not a stale/aliased value)
    θq = ρθq ./ ρdq
    @test all(250 .< θq .< 400)                 # physical θ
    @test all(isapprox.(θq, θtrue(1.9); rtol = 1e-4))   # linear in t ⇒ exact
end

# The full coupled Breeze child steps across a window move and its prognostics stay finite. A uniform
# synthetic parent with 4 s node spacing reaches the move (crossing node 3 at t = 8 s) in a few
# acoustically stable steps.
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

    sim = Simulation(model; Δt = 0.5, stop_time = 10.0)
    conjure_time_step_wizard!(sim, IterationInterval(1); cfl = 0.3, max_Δt = 2.0)
    run!(sim)

    @test model.exchanger.prognostic.ρᵈ.backend.start > 1        # actually crossed the window move
    @test model.clock.time ≥ 10.0 - 2.0                          # reached the end (no NaN Δt stall)
    for (name, field) in pairs(prognostic_fields(model.child))
        @test all(isfinite, Array(interior(field)))
    end
end

# `terrain_blend_length` is a physical length converted to a cell count per grid, so the blend slope is
# resolution-invariant: a 4×-finer grid gets ~4× the cells.
@testset "default_terrain_blend_width: physical length gives a resolution-invariant slope" begin
    ext = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)
    lon, lat = (-98.8, -96.2), (35.4, 37.8)
    coarse = LatitudeLongitudeGrid(size = (24, 22, 4); longitude = lon, latitude = lat, z = (0, 1e4),
                                   topology = (Bounded, Bounded, Bounded))
    fine   = LatitudeLongitudeGrid(size = (96, 88, 4); longitude = lon, latitude = lat, z = (0, 1e4),
                                   topology = (Bounded, Bounded, Bounded))
    wc = ext.default_terrain_blend_width(coarse, 60_000)
    wf = ext.default_terrain_blend_width(fine, 60_000)
    @test wc ≥ 1
    @test wf ≥ 3 * wc          # 4×-finer grid ⇒ ~4× cells (fixed physical width, not fixed cells)
    # physical blend width ≈ the requested length at both resolutions
    Δxc = minimum_xspacing(coarse, Center(), Center(), Center())
    Δxf = minimum_xspacing(fine,   Center(), Center(), Center())
    @test isapprox(wc * Δxc, 60_000; rtol = 0.2)
    @test isapprox(wf * Δxf, 60_000; rtol = 0.2)
end
