include("runtests_setup.jl")

using Oceananigans
using Oceananigans: set!, interior
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth.Lands: SlabLand, SlabEnergy, VariablySaturatedHydrology,
    InterceptingHydrology, SurfaceWaterStore,
    VanGenuchtenRetention, VanGenuchtenConductivity,
    NoDeepLiquidFlux, InfiltrationCapacityRunoff

scalar(field) = Array(interior(field))[1, 1, 1]

pond_test_grid(arch, FT) = RectilinearGrid(arch, FT; size = 1, x = (0, 1), y = (0, 1),
                                           z = (-1, 0), topology = (Flat, Flat, Bounded))

soil_hydrology(FT; infiltration_capacity = 1e-3) = VariablySaturatedHydrology(FT;
    slab_depth = 1.0, porosity = 0.4, storage_height = 1000,
    retention_curve = VanGenuchtenRetention(FT; inverse_air_entry_head = 1.0, pore_size_uniformity = 2.0),
    hydraulic_conductivity = VanGenuchtenConductivity(FT; matching_point_conductivity = 1e-6, pore_size_uniformity = 2.0),
    deep_liquid_flux = NoDeepLiquidFlux(),
    runoff = InfiltrationCapacityRunoff(FT; infiltration_capacity))

function pond_land(arch, FT; drainage_timescale = 3600, infiltration_capacity = 1e-3, pond = true)
    grid = pond_test_grid(arch, FT)
    soil = soil_hydrology(FT; infiltration_capacity)
    hydrology = pond ? SurfaceWaterStore(FT; soil, drainage_timescale) : soil
    return SlabLand(grid; energy = SlabEnergy(FT), hydrology)
end

# The coupler rewrites the flux accumulators every step; stand-alone stepping must do
# the same, since the drain kernel adds the pond's re-offer onto the rain in place.
set_rain!(land, rain) = fill!(parent(land.fluxes.liquid_precipitation_flux), rain)

@testset "SurfaceWaterStore declarations & container" begin
    for arch in test_architectures
        FT = Float64
        land = pond_land(arch, FT)
        @test :surface_water_storage in keys(land.prognostic)
        @test :liquid_precipitation_flux in keys(land.fluxes)
        @test :vapor_flux in keys(land.fluxes)                          # from the wrapped soil
        @test :surface_water_runoff in keys(land.diagnostics)
        @test :surface_water_storage_tendency in keys(land.diagnostics)
        @test :surface_runoff in keys(land.diagnostics)                 # from the wrapped soil
    end
end

@testset "Closed water budget under an over-cap rain pulse" begin
    for arch in test_architectures
        FT = Float64
        land = pond_land(arch, FT)   # infiltration cap 1e-3 kg m⁻² s⁻¹
        set!(land; T = 290.0, M = 100.0)
        Δt   = 60.0
        rain = 5e-3                  # well above the cap; the excess ponds
        Jᵛ   = 1e-5
        fill!(parent(land.fluxes.vapor_flux), Jᵛ)

        M₀ = scalar(land.water_storage)
        S₀ = scalar(land.prognostic.surface_water_storage)
        ∫P = 0.0; ∫Jᵛ = 0.0; ∫R = 0.0; ∫Rˡᵃᵗ = 0.0
        for n in 1:600
            Pⁿ = n ≤ 200 ? rain : 0.0
            set_rain!(land, Pⁿ)
            time_step!(land, Δt)
            ∫P    += Pⁿ * Δt
            ∫Jᵛ   += Jᵛ * Δt
            ∫R    += scalar(land.diagnostics.surface_water_runoff) * Δt
            ∫Rˡᵃᵗ += scalar(land.diagnostics.subsurface_runoff) * Δt
            # The soil positivity floor `max(M + Δt dM/dt, 0)` destroys water when it
            # binds; M staying strictly positive proves it never did.
            @test scalar(land.water_storage) > 0
        end

        ΔM = scalar(land.water_storage) - M₀
        ΔS = scalar(land.prognostic.surface_water_storage) - S₀
        @test ∫P ≈ ΔM + ΔS + ∫Jᵛ + ∫R + ∫Rˡᵃᵗ rtol=1e-12
    end
end

@testset "Pond drain e-folds exactly" begin
    for arch in test_architectures
        FT = Float64
        τ  = 1800.0
        S₀ = 10.0
        # Cap 0: the soil rejects everything, so drainage is the only sink.
        land = pond_land(arch, FT; drainage_timescale = τ, infiltration_capacity = 0.0)
        set!(land; T = 290.0, M = 100.0, surface_water_storage = S₀)

        Δt = τ / 12
        for _ in 1:12   # integrate to t = τ
            set_rain!(land, 0.0)
            time_step!(land, Δt)
        end
        @test scalar(land.prognostic.surface_water_storage) ≈ S₀ * exp(-1) rtol=1e-13
        @test scalar(land.water_storage) == 100.0   # nothing infiltrated
    end
end

@testset "Rejected rain reinfiltrates" begin
    for arch in test_architectures
        FT = Float64
        with    = pond_land(arch, FT; pond = true)
        without = pond_land(arch, FT; pond = false)
        Δt = 60.0
        for land in (with, without)
            set!(land; T = 290.0, M = 100.0)
            for n in 1:400
                set_rain!(land, n ≤ 50 ? 5e-3 : 0.0)
                time_step!(land, Δt)
            end
        end
        # The pond re-offers the rejected pulse after the rain stops.
        @test scalar(with.water_storage) > scalar(without.water_storage)
    end
end

@testset "Store stays non-negative and finite" begin
    for arch in test_architectures
        for FT in (Float64, Float32)
            land = pond_land(arch, FT)
            set!(land; T = 290.0, M = 50.0)
            fill!(parent(land.fluxes.vapor_flux), 1e-5)
            Δt = 600.0
            for n in 1:300
                set_rain!(land, n ≤ 100 ? 5e-3 : 0.0)
                time_step!(land, Δt)
                S = scalar(land.prognostic.surface_water_storage)
                M = scalar(land.water_storage)
                @test isfinite(S) && S ≥ 0
                @test isfinite(M) && M ≥ 0
            end
            @test scalar(land.prognostic.surface_water_storage) isa FT
        end
    end
end

@testset "Checkpoint round-trip of the surface water store" begin
    for arch in test_architectures
        FT = Float64
        land = pond_land(arch, FT)
        set!(land; T = 290.0, M = 120.0, surface_water_storage = 3.5)

        state = Oceananigans.prognostic_state(land)
        @test :surface_water_storage in keys(state.prognostic)

        land2 = pond_land(arch, FT)
        set!(land2; T = 300.0, M = 50.0, surface_water_storage = 9.9)   # perturb
        Oceananigans.restore_prognostic_state!(land2, state)
        @test scalar(land2.prognostic.surface_water_storage) == 3.5
        @test scalar(land2.water_storage) == 120.0
        @test scalar(land2.temperature) == 290.0
    end
end

@testset "Retention curve forwards through the pond" begin
    FT = Float64
    grid = pond_test_grid(CPU(), FT)
    ponded  = SurfaceWaterStore(FT; soil = soil_hydrology(FT))
    wrapped = InterceptingHydrology(FT; soil = ponded, leaf_area_index = 3.0)
    for hydrology in (ponded, wrapped)
        land = SlabLand(grid; energy = SlabEnergy(FT), hydrology)
        @test !isnothing(NumericalEarth.EarthSystemModels.surface_retention_curve(land))
    end
end

@testset "Composition with InterceptingHydrology" begin
    for arch in test_architectures
        FT = Float64
        grid = pond_test_grid(arch, FT)
        hydrology = InterceptingHydrology(FT;
            soil = SurfaceWaterStore(FT; soil = soil_hydrology(FT)),
            leaf_area_index = 3.0)
        land = SlabLand(grid; energy = SlabEnergy(FT), hydrology)

        @test :canopy_water_storage in keys(land.prognostic)
        @test :surface_water_storage in keys(land.prognostic)
        @test :surface_water_runoff in keys(land.diagnostics)
        @test :throughfall in keys(land.diagnostics)

        set!(land; T = 290.0, M = 100.0)
        Δt = 60.0
        for _ in 1:20
            set_rain!(land, 5e-3)
            time_step!(land, Δt)
        end
        S = scalar(land.prognostic.surface_water_storage)
        W = scalar(land.prognostic.canopy_water_storage)
        @test isfinite(S) && S ≥ 0
        @test isfinite(W) && W ≥ 0
        @test S > 0   # over-cap throughfall ponded
    end
end
