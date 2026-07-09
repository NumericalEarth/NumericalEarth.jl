include("runtests_setup.jl")

using Oceananigans
using Oceananigans.Fields: interior
using Oceananigans.TimeSteppers: time_step!

# Texture as mass fractions (kg/kg), bulk density in kg/m³ — the units delivered by
# the DataWrangling soil datasets. (sand, silt, clay, bulk_density).
sandy_loam = (0.55, 0.25, 0.20, 1500.0)
silt_loam  = (0.20, 0.65, 0.15, 1300.0)
clay_soil  = (0.25, 0.25, 0.50, 1400.0)
sand_soil  = (0.92, 0.05, 0.03, 1600.0)

@testset "ContinuousPedotransfer pedotransfer function" begin
    ptf = ContinuousPedotransfer()
    for texture in (sandy_loam, silt_loam, clay_soil, sand_soil)
        p = soil_hydraulic_parameters(ptf, texture...)
        @test 0.3 < p.ν < 0.6              # porosity in a physical range
        @test p.θʳ < p.ν                   # residual below saturation
        @test p.n > 1                      # van Genuchten n
        @test p.α > 0                      # m⁻¹
        @test p.K_saturated > 0            # m s⁻¹
    end

    # Sand drains far faster than clay.
    @test soil_hydraulic_parameters(ptf, sand_soil...).K_saturated >
          soil_hydraulic_parameters(ptf, clay_soil...).K_saturated

    # Pure sand (clay = silt = 0) must not blow up (1/x, ln x terms are floored).
    p_puresand = soil_hydraulic_parameters(ptf, 1.0, 0.0, 0.0, 1600.0)
    @test isfinite(p_puresand.ν) && isfinite(p_puresand.α) &&
          isfinite(p_puresand.n) && isfinite(p_puresand.K_saturated)

    # Type stability: Float32 inputs / Float32 ptf → Float32 outputs.
    p32 = soil_hydraulic_parameters(ContinuousPedotransfer(Float32), 0.4f0, 0.4f0, 0.2f0, 1400.0f0)
    @test p32.ν isa Float32
    @test p32.K_saturated isa Float32
end

@testset "layer_weights" begin
    zi = [-1.0, -0.6, -0.3, 0.0]   # OpenLandMap faces: 60-100, 30-60, 0-30 cm

    @test layer_weights(zi, 1.0) ≈ [0.4, 0.3, 0.3]     # full column
    @test layer_weights(zi, 0.3) ≈ [0.0, 0.0, 0.3]     # thin skin → 0-30 cm only
    @test layer_weights(zi, 0.5) ≈ [0.0, 0.2, 0.3]     # deepest included layer clipped
    @test sum(layer_weights(zi, 0.5)) ≈ 0.5            # weights sum to slab_depth
    @test sum(layer_weights(zi, 2.0)) ≈ 1.0            # clipped to the column depth
end

@testset "soil_hydraulic_properties reduction" begin
    zi = [-1.0, -0.6, -0.3, 0.0]

    for arch in test_architectures
        grid = RectilinearGrid(arch; size = (2, 1, 3), x = (0, 2), y = (0, 1), z = zi,
                               topology = (Bounded, Bounded, Bounded))
        sand = CenterField(grid); silt = CenterField(grid)
        clay = CenterField(grid); bulk_density = CenterField(grid)

        # Column 1: uniform loam. Column 2: sand (top) over clay (deep) — strong contrast.
        set!(sand, (x, y, z) -> x < 1 ? 0.40 : (z > -0.3 ? 0.90 : 0.20))
        set!(silt, (x, y, z) -> x < 1 ? 0.40 : (z > -0.3 ? 0.07 : 0.30))
        set!(clay, (x, y, z) -> x < 1 ? 0.20 : (z > -0.3 ? 0.03 : 0.50))
        set!(bulk_density, (x, y, z) -> 1400.0)

        props = soil_hydraulic_properties(sand, silt, clay, bulk_density;
                                          slab_depth = 1.0, z_interfaces = zi)

        # Outputs are 2-D (Center, Center, Nothing) fields the slab reads at [i, j].
        @test location(props.porosity) == (Center, Center, Nothing)
        @test size(props.K_saturated) == (2, 1, 1)

        ν  = Array(interior(props.porosity))[:, 1, 1]
        θʳ = Array(interior(props.residual_liquid_fraction))[:, 1, 1]
        α  = Array(interior(props.α))[:, 1, 1]
        n  = Array(interior(props.n))[:, 1, 1]
        Ks = Array(interior(props.K_saturated))[:, 1, 1]

        @test all(0.3 .< ν .< 0.6)
        @test all(θʳ .< ν)
        @test all(n .> 1)
        @test all(Ks .> 0)

        # Per-layer values of the sand-over-clay column (deepest-first).
        ptf = ContinuousPedotransfer()
        w = layer_weights(zi, 1.0); W = sum(w)
        layers = ((0.20, 0.30, 0.50, 1400.0),   # clay
                  (0.20, 0.30, 0.50, 1400.0),   # clay
                  (0.90, 0.07, 0.03, 1400.0))   # sand
        Ks_layers = [soil_hydraulic_parameters(ptf, l...).K_saturated for l in layers]
        α_layers  = [soil_hydraulic_parameters(ptf, l...).α for l in layers]

        Ks_harmonic   = W / sum(w ./ Ks_layers)
        Ks_arithmetic = sum(w .* Ks_layers) / W
        α_geometric   = exp(sum(w .* log.(α_layers)) / W)

        # Kₛ upscales harmonically (clay-limited), strictly below the arithmetic mean.
        @test Ks[2] ≈ Ks_harmonic
        @test Ks_harmonic < Ks_arithmetic
        # α upscales geometrically.
        @test α[2] ≈ α_geometric
    end
end

@testset "reduction degenerates to a single layer for a thin slab" begin
    zi = [-1.0, -0.6, -0.3, 0.0]
    for arch in test_architectures
        grid = RectilinearGrid(arch; size = (1, 1, 3), x = (0, 1), y = (0, 1), z = zi,
                               topology = (Bounded, Bounded, Bounded))
        sand = CenterField(grid); silt = CenterField(grid)
        clay = CenterField(grid); bulk_density = CenterField(grid)
        set!(sand, (x, y, z) -> z > -0.3 ? 0.90 : 0.20)   # top layer = sand
        set!(silt, (x, y, z) -> z > -0.3 ? 0.07 : 0.30)
        set!(clay, (x, y, z) -> z > -0.3 ? 0.03 : 0.50)
        set!(bulk_density, (x, y, z) -> 1400.0)

        # slab_depth = 0.3 uses only the 0-30 cm (top) layer.
        props = soil_hydraulic_properties(sand, silt, clay, bulk_density;
                                          slab_depth = 0.3, z_interfaces = zi)
        top = soil_hydraulic_parameters(ContinuousPedotransfer(), 0.90, 0.07, 0.03, 1400.0)
        @test Array(interior(props.porosity))[1, 1, 1]    ≈ top.ν
        @test Array(interior(props.K_saturated))[1, 1, 1] ≈ top.K_saturated
        @test Array(interior(props.α))[1, 1, 1]           ≈ top.α
    end
end

@testset "Field-backed van Genuchten closures" begin
    # Scalar path unchanged: matches the closed-form van Genuchten pressure head.
    r = VanGenuchtenRetention(α = 2.0, n = 1.4)
    𝒮 = 0.5
    m = 1 - 1/1.4
    Π_ref = -(𝒮^(-1/m) - 1)^(1/1.4) / 2.0
    @test NumericalEarth.Lands.pressure_head(r, 𝒮, 1, 1) ≈ Π_ref

    for arch in test_architectures
        grid = RectilinearGrid(arch; size = (2, 1), x = (0, 2), y = (0, 1),
                               topology = (Bounded, Bounded, Flat))

        # Two columns with different hydraulic parameters (Field-backed α, n, Kₛ, ν).
        makefield(v1, v2) = (f = Field{Center, Center, Nothing}(grid);
                             set!(f, (x, y) -> x < 1 ? v1 : v2); f)
        ν  = makefield(0.45, 0.35)
        α  = makefield(1.0, 4.0)
        n  = makefield(1.6, 1.2)
        Ks = makefield(1e-5, 1e-7)

        hydrology = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0, porosity = ν, storage_height = 1000,
            retention_curve = VanGenuchtenRetention(; α, n),
            hydraulic_conductivity = VanGenuchtenConductivity(; K_saturated = Ks, n),
            deep_liquid_flux = FreeDrainageFlux(), runoff = NoRunoff())

        land = SlabLand(grid; hydrology)
        set!(land; M = 150.0)
        fill!(land.fluxes.vapor_flux, 0)
        fill!(land.fluxes.liquid_precipitation_flux, 0)

        for _ in 1:20
            time_step!(land, 3600.0)
        end

        M = Array(interior(land.water_storage))[:, 1, 1]
        # Different Kₛ per column ⇒ different drainage ⇒ storage diverges.
        @test M[1] != M[2]
        @test all(M .< 150.0)   # both columns drained
    end
end
