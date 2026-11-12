include("runtests_setup.jl")

using NumericalEarth
using NumericalEarth.NestedModels: nested_atmosphere_model
using Oceananigans
using Oceananigans: prognostic_fields
using Breeze
using Breeze: ThermodynamicConstants, dry_air_gas_constant, vapor_gas_constant, SaturationAdjustment,
              WarmPhaseEquilibrium, moisture_prognostic_name
using Breeze.Thermodynamics: MoistureMassFractions, LiquidIcePotentialTemperatureState, temperature
using Breeze.Microphysics.PredictedParticleProperties: PredictedParticlePropertiesMicrophysics
using Test

const ext = Base.get_extension(NumericalEarth, :NumericalEarthBreezeExt)

# Parent hydrometeors: one warm column (λ < 0) and one cold column (λ > 0), uniform condensate.
const seeding_qᶜˡ, seeding_qʳ, seeding_qᶜⁱ, seeding_qˢ = 5.0e-4, 2.0e-4, 1.0e-4, 5.0e-5
seeding_temperature(λ) = λ < 0 ? 290 : 250
seeding_humidity(λ)    = λ < 0 ? 8.0e-3 : 1.0e-3
const seeding_pressure = 8.0e4

function condensate_parent(FT, times = [0.0, 1.0, 2.0])
    grid = LatitudeLongitudeGrid(CPU(), FT; size = (4, 4, 2), longitude = (-1, 1), latitude = (-1, 1),
                                 z = (0, 1000), topology = (Bounded, Bounded, Bounded))
    hydrometeor(q) = (f = FieldTimeSeries{Center, Center, Center}(grid, times); set!(f, (λ, φ, z, t) -> q); f)
    microphysical_variables = (qᶜˡ = hydrometeor(seeding_qᶜˡ), qʳ = hydrometeor(seeding_qʳ),
                               qᶜⁱ = hydrometeor(seeding_qᶜⁱ), qˢ = hydrometeor(seeding_qˢ))
    parent = PrescribedAtmosphere(grid, times; microphysical_variables)
    set!(parent.temperature,       (λ, φ, z, t) -> seeding_temperature(λ))
    set!(parent.specific_humidity, (λ, φ, z, t) -> seeding_humidity(λ))
    set!(parent.pressure,          (λ, φ, z, t) -> seeding_pressure)
    set!(parent.velocities.u,      (λ, φ, z, t) -> 0)
    set!(parent.velocities.v,      (λ, φ, z, t) -> 0)
    return parent
end

at(f, i) = Array(interior(f))[i, 1, 1]

@testset "Condensate seeding: equilibrium and unknown schemes seed nothing" begin
    @test isnothing(ext.condensate_seeding(SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())))
    @test isnothing(ext.condensate_seeding(nothing))
    @test ext.seeded_prognostic_names(nothing) == ()
    @test ext.parent_condensate_prognostics(nothing, (qᶜˡ = 1.0, qʳ = 1.0, qᶜⁱ = 1.0, qˢ = 1.0), 1.0) == NamedTuple()
end

for FT in (Float64, Float32)
    p3 = PredictedParticlePropertiesMicrophysics(FT)
    seeding = ext.condensate_seeding(p3)
    constants = ThermodynamicConstants(FT)
    pˢᵗ = FT(1e5)
    rtol = FT == Float64 ? 1e-10 : 1e-5

    @testset "P3 condensate seeding names are P3 prognostics [$FT]" begin
        @test seeding isa ext.P3CondensateSeeding{FT}
        names = ext.seeded_prognostic_names(seeding)
        @test all(name -> name ∈ prognostic_field_names(p3), names)
        @test :ρqᶠ ∉ names && :ρbᶠ ∉ names && :ρqʷⁱ ∉ names   # unrimed, dry ice: left at zero
        @test moisture_prognostic_name(p3) == :ρqᵛ
    end

    @testset "P3 condensate seeding on the parent grid [$FT]" begin
        parent = condensate_parent(FT)
        ex = ext.state_exchanger(parent, pˢᵗ, constants; moisture_name = :ρqᵛ, seeding)
        P = ex.prognostic
        @test ex.seeding isa ext.P3CondensateSeeding{FT}

        Rᵈ, Rᵛ = dry_air_gas_constant(constants), vapor_gas_constant(constants)
        qˡ = seeding_qᶜˡ + seeding_qʳ
        qⁱ = seeding_qᶜⁱ + seeding_qˢ
        mᵢ_min, mᵢ_max = seeding.minimum_ice_mean_mass, seeding.maximum_ice_mean_mass
        λʳ_min, λʳ_max = seeding.minimum_rain_slope, seeding.maximum_rain_slope
        ρʷ = seeding.liquid_water_density

        for (i, λ) in ((1, -0.75), (4, 0.75))   # warm column, cold column
            T, qᵛ, p = seeding_temperature(λ), seeding_humidity(λ), seeding_pressure
            ρ = p / (((1 - qᵛ - qˡ - qⁱ) * Rᵈ + qᵛ * Rᵛ) * T)

            # Masses: ρ·q, with cloud ice and snow merged into P3's single ice category.
            @test at(P.ρqᶜˡ[1], i) ≈ ρ * seeding_qᶜˡ rtol = rtol
            @test at(P.ρqʳ[1], i)  ≈ ρ * seeding_qʳ  rtol = rtol
            @test at(P.ρqⁱ[1], i)  ≈ ρ * qⁱ          rtol = rtol
            @test at(P.ρqᵛᵉ[1], i) ≈ ρ * qᵛ          rtol = rtol

            # Total water and total density survive the mapping.
            ρqᵗ = at(P.ρqᵛᵉ[1], i) + at(P.ρqᶜˡ[1], i) + at(P.ρqʳ[1], i) + at(P.ρqⁱ[1], i)
            @test ρqᵗ ≈ ρ * (qᵛ + qˡ + qⁱ) rtol = rtol
            @test at(Field(ext.total_density(ex)), i) ≈ ρ rtol = rtol

            # Numbers: positive, rain slope and mean ice mass inside P3's bounds.
            ρnʳ, ρnⁱ = at(P.ρnʳ[1], i), at(P.ρnⁱ[1], i)
            @test ρnʳ > 0 && isfinite(ρnʳ)
            @test ρnⁱ > 0 && isfinite(ρnⁱ)
            λʳ = cbrt(π * ρʷ * ρnʳ / at(P.ρqʳ[1], i))
            @test λʳ_min * (1 - rtol) ≤ λʳ ≤ λʳ_max * (1 + rtol)
            m̄ⁱ = at(P.ρqⁱ[1], i) / ρnⁱ
            @test mᵢ_min ≤ m̄ⁱ ≤ mᵢ_max

            # Marshall–Palmer: n = N₀^¾ (ρqʳ / π ρʷ)^¼ when the slope is unclamped.
            @test ρnʳ ≈ seeding.rain_intercept^(3/4) * (ρ * seeding_qʳ / (π * ρʷ))^(1/4) rtol = 10rtol

            # θˡⁱ removes latent heat for exactly the seeded condensate: the child recovers T.
            θˡⁱ = at(P.ρθ[1], i) / at(P.ρᵈ[1], i)
            q = MoistureMassFractions(FT(qᵛ), FT(qˡ), FT(qⁱ))
            T̃ = temperature(LiquidIcePotentialTemperatureState(θˡⁱ, q, pˢᵗ, FT(p)), constants)
            @test T̃ ≈ T atol = (FT == Float64 ? 1e-4 : 2e-2)

            # Pairing the same θˡⁱ with vapor alone loses the seeded latent heat: ≳ 1 K here.
            T̃ᵛ = temperature(LiquidIcePotentialTemperatureState(θˡⁱ, MoistureMassFractions(FT(qᵛ)), pˢᵗ, FT(p)), constants)
            @test T - T̃ᵛ > 1
        end
    end

    @testset "P3 seeding floors vanishing masses instead of diagnosing a number [$FT]" begin
        tiny = (qᶜˡ = FT(1e-20), qʳ = FT(1e-20), qᶜⁱ = FT(1e-20), qˢ = FT(1e-20))
        seeded = ext.parent_condensate_prognostics(seeding, tiny, one(FT))
        @test all(iszero, values(seeded))
        @test ext.seeded_condensate(seeding, tiny) == (zero(FT), zero(FT))

        none = (qᶜˡ = zero(FT), qʳ = zero(FT), qᶜⁱ = zero(FT), qˢ = zero(FT))
        @test all(iszero, values(ext.parent_condensate_prognostics(seeding, none, one(FT))))

        # Just above the floor every number is finite.
        small = (qᶜˡ = FT(1e-12), qʳ = FT(1e-12), qᶜⁱ = FT(1e-12), qˢ = FT(1e-12))
        @test all(isfinite, values(ext.parent_condensate_prognostics(seeding, small, one(FT))))
        @test all(≥(0), values(ext.parent_condensate_prognostics(seeding, small, one(FT))))
    end

    @testset "Unseeded vapor-only exchange is unchanged by the seeding hook [$FT]" begin
        parent = condensate_parent(FT)
        unseeded = ext.state_exchanger(parent, pˢᵗ, constants; moisture_name = :ρqᵛ)
        seeded   = ext.state_exchanger(parent, pˢᵗ, constants; moisture_name = :ρqᵛ, seeding)
        @test keys(unseeded.prognostic) == (:ρᵈ, :ρu, :ρv, :ρθ, :ρqᵛᵉ, :θ, :u, :v)
        for name in (:ρᵈ, :ρqᵛᵉ, :ρu, :ρv)   # dry density and vapor do not depend on seeding
            @test interior(unseeded.prognostic[name][1]) == interior(seeded.prognostic[name][1])
        end
        @test interior(unseeded.prognostic.θ[1]) != interior(seeded.prognostic.θ[1])
        # The unseeded θˡⁱ keeps no condensate at all.
        θ₀ = ext.liquid_ice_potential_temperature(FT(290), FT(8e-3), zero(FT), zero(FT), FT(seeding_pressure), pˢᵗ, constants)
        @test at(unseeded.prognostic.θ[1], 1) ≈ θ₀ rtol = rtol
    end

    @testset "P3 child is seeded and boundary-relaxed from the parent [$FT]" begin
        times = [0.0, 3600.0, 7200.0]
        parent_grid = LatitudeLongitudeGrid(CPU(), FT; size = (12, 12, 8), longitude = (-2, 2), latitude = (34.6, 38.6),
                                            z = (0, 12000), halo = (5, 5, 5), topology = (Bounded, Bounded, Bounded))
        hydrometeor(q) = (f = FieldTimeSeries{Center, Center, Center}(parent_grid, times); set!(f, (λ, φ, z, t) -> q); f)
        microphysical_variables = (qᶜˡ = hydrometeor(seeding_qᶜˡ), qʳ = hydrometeor(seeding_qʳ),
                                   qᶜⁱ = hydrometeor(seeding_qᶜⁱ), qˢ = hydrometeor(seeding_qˢ))
        parent = PrescribedAtmosphere(parent_grid, times; microphysical_variables)
        set!(parent.temperature,       (λ, φ, z, t) -> 288 - 6.5e-3 * z)
        set!(parent.specific_humidity, (λ, φ, z, t) -> 4e-3 * exp(-z / 1500))   # subsaturated: no new cloud
        set!(parent.velocities.u,      (λ, φ, z, t) -> 0)
        set!(parent.velocities.v,      (λ, φ, z, t) -> 0)
        set!(parent.pressure,          (λ, φ, z, t) -> 1e5 * exp(-z / 8000))

        child_grid = LatitudeLongitudeGrid(CPU(), FT; size = (12, 12, 6), longitude = (-1, 1), latitude = (35.6, 37.6),
                                           z = (0, 12000), halo = (5, 5, 5), topology = (Bounded, Bounded, Bounded))

        build(; kw...) = nested_atmosphere_model(parent, child_grid; microphysics = p3,
                                                 relaxation_rate = 1/60, relaxation_width = 3, base_pressure = 1e5,
                                                 coriolis = nothing, closure = nothing, kw...)

        # Opt-out: vapor only, as before.
        unseeded = build(; seed_parent_condensate = false)
        @test isnothing(unseeded.exchanger.seeding)
        @test !haskey(unseeded.exchanger.prognostic, :ρqᶜˡ)

        model = build()
        @test model.exchanger.seeding isa ext.P3CondensateSeeding{FT}
        ext.initialize_nested_child!(model, nothing, first(times), ""; balancer = false)
        μ = model.child.microphysical_fields
        ρ = model.child.dynamics.total_density   # total, not dry

        # Initial condition: specific condensate equals the parent's, rime and liquid-on-ice are zero.
        interior_of(f) = Array(interior(f))
        tol = FT == Float64 ? 1e-6 : 1e-3
        @test all(isapprox.(interior_of(μ.ρqᶜˡ) ./ interior_of(ρ), seeding_qᶜˡ; rtol = tol))
        @test all(isapprox.(interior_of(μ.ρqʳ)  ./ interior_of(ρ), seeding_qʳ;  rtol = tol))
        @test all(isapprox.(interior_of(μ.ρqⁱ)  ./ interior_of(ρ), seeding_qᶜⁱ + seeding_qˢ; rtol = tol))
        @test all(>(0), interior_of(μ.ρnʳ))
        @test all(>(0), interior_of(μ.ρnⁱ))
        @test all(iszero, interior_of(μ.ρqᶠ))
        @test all(iszero, interior_of(μ.ρbᶠ))
        @test all(iszero, interior_of(μ.ρqʷⁱ))

        # Boundary relaxation: clear the child's cloud liquid and step. The relaxation zone refills
        # toward the parent; the (motionless, subsaturated) interior does not grow cloud liquid.
        fill!(interior(μ.ρqᶜˡ), 0)
        sim = Simulation(model; Δt = 2, stop_iteration = 10, verbose = false)
        run!(sim)

        ρqᶜˡ = interior_of(μ.ρqᶜˡ)
        Nx, Ny, _ = size(child_grid)
        @test all(isfinite, ρqᶜˡ)
        for (name, field) in pairs(prognostic_fields(model.child))
            @test all(isfinite, interior_of(field))
        end
        @test minimum(ρqᶜˡ[1, :, 1:3]) > 0                 # the west wall moves toward the parent
        @test minimum(ρqᶜˡ[Nx, :, 1:3]) > 0                # and the east wall
        @test maximum(abs, ρqᶜˡ[5:Nx-4, 5:Ny-4, :]) ≤ 1e-8 * seeding_qᶜˡ   # interior stays clear
    end
end

@testset "Equilibrium child: seeding switch is a no-op" begin
    parent = condensate_parent(Float64)
    child_grid = LatitudeLongitudeGrid(CPU(); size = (8, 8, 4), longitude = (-0.5, 0.5), latitude = (-0.5, 0.5),
                                       z = (0, 1000), halo = (5, 5, 5), topology = (Bounded, Bounded, Bounded))
    microphysics = SaturationAdjustment(equilibrium = WarmPhaseEquilibrium())
    on  = nested_atmosphere_model(parent, child_grid; microphysics, relaxation_rate = 1/60, coriolis = nothing)
    off = nested_atmosphere_model(parent, child_grid; microphysics, relaxation_rate = 1/60, coriolis = nothing,
                                  seed_parent_condensate = false)
    @test isnothing(on.exchanger.seeding)
    @test keys(on.exchanger.prognostic) == keys(off.exchanger.prognostic)
    for name in keys(on.exchanger.prognostic)
        @test interior(on.exchanger.prognostic[name][1]) == interior(off.exchanger.prognostic[name][1])
    end
end
