using OMIPSimulations
using OMIPSimulations: ConservativeSurfaceFluxRestoring, update_restoring_flux!, BoundaryValueTransport

using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.Operators: volume
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.TimeSteppers: time_step!, update_state!
using Oceananigans.Grids: znodes, Face
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

@testset "Boundary-value eddy transport" begin
    arch = CPU()

    # The nonlinear Eady problem of Ferrari et al. (2010), Section 5.1: constant N² and a constant
    # horizontal buoyancy gradient give a uniform neutral slope, for which the boundary-value
    # problem has the closed-form solution Υ = -κ S μ̄ with
    #
    #     μ̄(z) = 1 - cosh[(z + H/2) / λ] / cosh[H / (2λ)] ,     λ = c/N
    #
    # (their Eqs. 67-69). With mode_number M = 1 the speed is c = N H / π and μ̄ reduces to their
    # Eq. 39, the curve that overlies the Fox-Kemper et al. (2008) structure function in Fig. 1.

    H  = 1000.0
    N² = 1e-5
    M² = 1e-8     # ∂x b, so the neutral slope is S = -M²/N² = -10⁻³, well under the taper threshold
    κ  = 1000.0

    S = - M² / N²
    c = sqrt(N²) * H / π
    λ = c / sqrt(N²)
    μ̄(z) = 1 - cosh((z + H/2) / λ) / cosh(H / (2λ))
    Υ_analytic(z) = - κ * S * μ̄(z)

    eady_transport(Nz) = begin
        grid = RectilinearGrid(arch; size = (4, 4, Nz), x = (0, 1e6), y = (0, 1e6), z = (-H, 0),
                               topology = (Bounded, Bounded, Bounded))

        closure = BoundaryValueTransport(; κ_skew = κ, mode_number = 1, minimum_speed = 1e-8)

        model = HydrostaticFreeSurfaceModel(grid; closure, buoyancy = BuoyancyTracer(), tracers = :b)
        set!(model, b = (x, y, z) -> N² * z + M² * x)
        update_state!(model)

        zf = znodes(grid, Face())
        Υˣ = model.closure_fields.Υˣ

        return maximum(abs(Υˣ[2, 2, k] - Υ_analytic(zf[k])) for k in 1:Nz+1), Υˣ
    end

    coarse_error, Υ_coarse = eady_transport(32)
    fine_error,   _        = eady_transport(64)

    # Second-order convergence to the analytic solution.
    @test coarse_error / fine_error > 3.8
    @test fine_error < 1e-4

    # Homogeneous Dirichlet conditions hold exactly, with no tapering applied to reach them.
    @test all(interior(Υ_coarse, :, :, 1)  .== 0)
    @test all(interior(Υ_coarse, :, :, 33) .== 0)

    # The eddy-induced velocity must be discretely non-divergent, or the transport would be a
    # spurious tracer source.
    grid = RectilinearGrid(arch; size = (4, 4, 32), x = (0, 1e6), y = (0, 1e6), z = (-H, 0),
                           topology = (Bounded, Bounded, Bounded))
    closure = BoundaryValueTransport(; κ_skew = κ, mode_number = 1, minimum_speed = 1e-8)
    model = HydrostaticFreeSurfaceModel(grid; closure, buoyancy = BuoyancyTracer(), tracers = :b)
    set!(model, b = (x, y, z) -> N² * z + M² * x + 1e-7 * sinpi(2y / 1e6))
    update_state!(model)

    K = model.closure_fields
    divergence = Field(∂x(K.u) + ∂y(K.v) + ∂z(K.w))
    compute!(divergence)
    @test maximum(abs, interior(divergence, 2:3, 2:3, :)) < 1e-18

    # A depth-varying κ_skew is inconsistent with the scheme and must be rejected up front.
    @test_throws ArgumentError OMIPSimulations.check_depth_independent_skew_coefficient(:cesm, :boundary_value)
    @test_throws ArgumentError OMIPSimulations.check_depth_independent_skew_coefficient(:hybrid, :boundary_value)
    @test isnothing(OMIPSimulations.check_depth_independent_skew_coefficient(:nemo, :boundary_value))
    @test isnothing(OMIPSimulations.check_depth_independent_skew_coefficient(:cesm, :diffusive))

    # The closure lands in the tuple `omip_closure` builds, alongside a Redi-only companion.
    closure_tuple = OMIPSimulations.omip_closure(:catke; κ_skew = 1000, κ_symmetric = 1000,
                                                 biharmonic_timescale = nothing,
                                                 skew_flux_formulation = :boundary_value)
    @test any(c -> c isa BoundaryValueTransport, closure_tuple)
end

@testset "Strait freshwater transports" begin
    # A uniform channel, so every flux has a closed form and sign, units and the salinity
    # weighting can each be checked against it independently.
    Nx, Ny, Nz = 8, 8, 4
    Lx, Ly, Lz = 8e3, 8e3, 400.0
    grid = RectilinearGrid(CPU(); size = (Nx, Ny, Nz), x = (0, Lx), y = (0, Ly), z = (-Lz, 0),
                           topology = (Bounded, Bounded, Bounded))

    Δx = Lx / Nx
    Δz = Lz / Nz
    section = OMIPSimulations.StraitSection(3:5, 4:4, :v)   # three cells wide
    area = 3 * Δx * Nz * Δz

    u = zeros(Nx, Ny, Nz)
    v = fill(0.1, Nx, Ny, Nz)
    S★ = 34.8

    volume_flux = OMIPSimulations.section_volume_flux(grid, u, v, section)
    @test volume_flux ≈ 0.1 * area

    liquid(S) = OMIPSimulations.section_liquid_freshwater_flux(grid, u, v, fill(S, Nx, Ny, Nz),
                                                               section, S★)

    # Water at the reference salinity carries no freshwater anomaly at all.
    @test liquid(S★) ≈ 0 atol = 1e-12

    # Pure freshwater carries the whole volume flux; half the reference salinity, half of it.
    @test liquid(0.0)    ≈ volume_flux
    @test liquid(S★ / 2) ≈ volume_flux / 2

    # Fresher than reference is a positive (northward) freshwater flux; saltier is negative.
    @test liquid(30.0) > 0
    @test liquid(36.0) < 0

    # Southward flow reverses the sign — Arctic export through Fram and Davis is negative.
    @test OMIPSimulations.section_liquid_freshwater_flux(grid, u, -v, fill(30.0, Nx, Ny, Nz),
                                                        section, S★) ≈ -liquid(30.0)

    # Solid flux: 1 m of ice at full concentration moving at 0.1 m/s across the same width.
    ice_salinity, ice_density = 4.0, 900.0
    fraction = (ice_density / 1000) * (S★ - ice_salinity) / S★
    ui = zeros(Nx, Ny, 1)
    vi = fill(0.1, Nx, Ny, 1)
    ℵ  = ones(Nx, Ny, 1)
    h  = ones(Nx, Ny, 1)

    solid = OMIPSimulations.section_ice_freshwater_flux(grid, ui, vi, ℵ, h, section, fraction)
    @test solid ≈ 0.1 * 1.0 * 3Δx * fraction

    # Half the concentration carries half the ice, hence half the freshwater.
    @test OMIPSimulations.section_ice_freshwater_flux(grid, ui, vi, 0.5ℵ, h, section, fraction) ≈ solid / 2

    # km³ yr⁻¹ conversion: 1 m³ s⁻¹ is a bit over 0.03 km³ yr⁻¹.
    @test OMIPSimulations.cubic_kilometers_per_year ≈ 0.0315360

    # Both Arctic gateways exist on the ORCA mesh and are zonal sections.
    orca = OMIPSimulations.strait_sections(:orca)
    @test haskey(orca, :fram) && haskey(orca, :davis)
    @test orca.fram.axis == :v && orca.davis.axis == :v
end
