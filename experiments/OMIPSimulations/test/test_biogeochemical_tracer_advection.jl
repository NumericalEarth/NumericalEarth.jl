using Test
using OMIPSimulations
using OMIPSimulations: omip_tracer_advection
using Oceananigans
using Oceananigans.Advection: AdaptiveImplicitVerticalAdvection, weno_order
using Oceananigans.Biogeochemistry: AbstractContinuousFormBiogeochemistry
using Oceananigans.Fields: ZeroField, interior
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth.Oceans: ocean_simulation

import Oceananigans.Biogeochemistry: required_biogeochemical_tracers, biogeochemical_drift_velocity

# A passive tracer `P` sinking at a prescribed speed.
struct SinkingTracer{W} <: AbstractContinuousFormBiogeochemistry
    sinking_velocity :: W
end

required_biogeochemical_tracers(::SinkingTracer) = (:P,)
biogeochemical_drift_velocity(bgc::SinkingTracer, ::Val{:P}) = (u = ZeroField(), v = ZeroField(), w = bgc.sinking_velocity)

# Some OceanBioME models list `T` and `S` among their required tracers.
struct TemperatureDependentTracer <: AbstractContinuousFormBiogeochemistry end
required_biogeochemical_tracers(::TemperatureDependentTracer) = (:P, :T)

@testset "Biogeochemical tracer advection" begin
    grid = RectilinearGrid(CPU(); size = (8, 8, 16), halo = (4, 4, 4), x = (0, 1e5), y = (0, 1e5), z = (-400, 0),
                           topology = (Periodic, Periodic, Bounded))

    w_sink = ZFaceField(grid)
    set!(w_sink, -100 / 86400)                        # 100 m day⁻¹ downward (negative = sinking)
    bgc = SinkingTracer(w_sink)

    implicit = AdaptiveVerticallyImplicitDiscretization(cfl = 0.5)
    advection = omip_tracer_advection(bgc, 7, 5, implicit, Val(:default))

    @test keys(advection) == (:T, :S, :P)
    @test advection.P isa AdaptiveImplicitVerticalAdvection
    @test weno_order(advection.P.z) == 5 && weno_order(advection.P.x) == 5
    @test weno_order(advection.T.z) == 7

    # A required `T` keeps the temperature-salinity scheme.
    with_temperature = omip_tracer_advection(TemperatureDependentTracer(), 7, 5, implicit, Val(:default))
    @test keys(with_temperature) == (:T, :S, :P)
    @test weno_order(with_temperature.T.z) == 7

    # No biogeochemistry: only T and S, as before.
    @test keys(omip_tracer_advection(nothing, 7, 5, implicit, Val(:default))) == (:T, :S)

    explicit = omip_tracer_advection(bgc, 7, 5, ExplicitTimeDiscretization(), Val(:default))
    @test !(explicit.P isa AdaptiveImplicitVerticalAdvection)

    # The scheme reaches the model, and sinking at a vertical Courant number |w| Δt / Δz = 100 m day⁻¹ × 12 h / 25 m = 2,
    # well above the explicit limit, stays finite and non-negative and conserves the tracer content. The drift
    # velocity vanishes on the top and bottom faces, so the column is closed and `P` piles up in the bottom cell.
    set!(w_sink, (x, y, z) -> -400 < z < 0 ? -100 / 86400 : 0.0)
    ocean = ocean_simulation(grid; biogeochemistry = bgc,
                             tracer_advection = advection,
                             momentum_advection = nothing,
                             closure = nothing,
                             free_surface = SplitExplicitFreeSurface(substeps = 10),
                             radiative_forcing = nothing,
                             bottom_drag_coefficient = 0)

    @test ocean.model.advection.P isa AdaptiveImplicitVerticalAdvection
    @test !isnothing(ocean.model.timestepper.implicit_solver)

    set!(ocean.model, T = 10, S = 35, P = (x, y, z) -> z > -50 ? 1.0 : 0.0)
    P₀ = sum(interior(ocean.model.tracers.P))

    for _ in 1:10
        time_step!(ocean.model, 12 * 3600)
    end

    P = interior(ocean.model.tracers.P)
    @test all(isfinite, P)
    @test minimum(P) > -1e-3                          # WENO undershoots slightly (≈ -2e-5 here), no more
    @test sum(P) ≈ P₀ rtol = 1e-8
    @test P[1, 1, 16] < 1e-3                          # 5 days at 100 m day⁻¹ empties the upper 50 m...
    @test P[1, 1, 1] > 1                              # ... into the bottom cell
end
