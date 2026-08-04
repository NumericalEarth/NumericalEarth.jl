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
        @test 0.3 < p.porosity < 0.6                        # physical range
        @test p.residual_liquid_fraction < p.porosity       # residual below saturation
        @test p.pore_size_uniformity > 1                    # van Genuchten n
        @test p.inverse_air_entry_head > 0                  # m⁻¹
        @test p.K_saturated > 0                             # m s⁻¹
    end

    # Sand drains far faster than clay.
    @test soil_hydraulic_parameters(ptf, sand_soil...).K_saturated >
          soil_hydraulic_parameters(ptf, clay_soil...).K_saturated

    # Pure sand (clay = silt = 0) must not blow up (1/x, ln x terms are floored).
    p_puresand = soil_hydraulic_parameters(ptf, 1.0, 0.0, 0.0, 1600.0)
    @test isfinite(p_puresand.porosity) && isfinite(p_puresand.inverse_air_entry_head) &&
          isfinite(p_puresand.pore_size_uniformity) && isfinite(p_puresand.K_saturated)

    # Extrapolating the regression is what breaks it, so the predictors are held
    # inside the range it behaves in: without that, texture → 0 pushes θs past 1 and
    # a peat-like bulk density drives n to exactly 1, where m = 1 - 1/n vanishes and
    # the retention curve is singular.
    for (sand, silt, clay, ρᵇ) in ((1.0, 0.0, 0.0, 1500.0),    # texture floor
                                   (0.5, 0.0, 0.5, 1400.0),    # silt floor
                                   (0.4, 0.4, 0.2, 100.0),     # peat-like ρᵇ
                                   (0.4, 0.4, 0.2, 2600.0))    # rock-like ρᵇ
        p = soil_hydraulic_parameters(ptf, sand, silt, clay, ρᵇ)
        @test 0 < p.porosity < 1
        @test p.pore_size_uniformity > 1.01          # clear of the m = 0 singularity
        @test 1 - 1/p.pore_size_uniformity > 0.01
        @test 0 < p.inverse_air_entry_head < 1e3
        @test 0 < p.K_saturated < 1e-2
    end

    # HYPRES fits its retention curves with θʳ = 0; α and n describe that curve.
    @test soil_hydraulic_parameters(ptf, sandy_loam...).residual_liquid_fraction == 0

    # Topsoil drains through its aggregate structure, subsoil does not. The flag
    # comes off the layer depth, and for clay it is worth a factor of several in Kₛ.
    clay_top  = soil_hydraulic_parameters(ptf, clay_soil..., 0.15)
    clay_deep = soil_hydraulic_parameters(ptf, clay_soil..., 0.80)
    @test clay_top.K_saturated > 3 * clay_deep.K_saturated
    # Omitting the depth reports the surface value.
    @test soil_hydraulic_parameters(ptf, clay_soil...).K_saturated == clay_top.K_saturated

    # Type stability: Float32 inputs / Float32 ptf → Float32 outputs.
    p32 = soil_hydraulic_parameters(ContinuousPedotransfer(Float32), 0.4f0, 0.4f0, 0.2f0, 1400.0f0)
    @test p32.porosity isa Float32
    @test p32.K_saturated isa Float32
end

@testset "pedotransfer float type follows the grid" begin
    # Devices reject unsupported float types outright (Metal has no Float64), so no
    # Float64 may survive in the pedotransfer function handed to the kernel.
    ptf32 = NumericalEarth.Lands.on_float_type(Float32, ContinuousPedotransfer(Float64))
    @test ptf32 isa ContinuousPedotransfer{Float32}
    coefficients = ptf32.regression_coefficients
    @test all(name -> eltype(getfield(coefficients, name)) === Float32,
              fieldnames(HYPRESRegression))
    @test ptf32.pore_connectivity_exponent isa Float32

    # Same regression either way, to Float32 precision.
    p64 = soil_hydraulic_parameters(ContinuousPedotransfer(Float64), 0.4, 0.4, 0.2, 1400.0)
    p32 = soil_hydraulic_parameters(ptf32, 0.4f0, 0.4f0, 0.2f0, 1400.0f0)
    @test p32.porosity ≈ p64.porosity rtol=1e-5
    @test p32.K_saturated ≈ p64.K_saturated rtol=1e-5

    # A Float32 grid must produce Float32 property fields from the default (Float64) ptf.
    grid = RectilinearGrid(Float32; size = (1, 1, 3), x = (0, 1), y = (0, 1),
                           z = [-1.0, -0.6, -0.3, 0.0], topology = (Bounded, Bounded, Bounded))
    texture = map(_ -> CenterField(grid), (1, 2, 3, 4))
    set!(texture[1], 0.4); set!(texture[2], 0.4); set!(texture[3], 0.2); set!(texture[4], 1400)
    props = soil_hydraulic_properties(texture...; slab_depth = 1.0,
                                      z_interfaces = [-1.0, -0.6, -0.3, 0.0])
    @test eltype(props.porosity) === Float32
    @test eltype(props.K_saturated) === Float32
end

@testset "layer_weights and layer_depths" begin
    zi = [-1.0, -0.6, -0.3, 0.0]   # OpenLandMap faces: 60-100, 30-60, 0-30 cm

    @test layer_weights(zi, 1.0) ≈ [0.4, 0.3, 0.3]     # full column
    @test layer_weights(zi, 0.3) ≈ [0.0, 0.0, 0.3]     # thin skin → 0-30 cm only
    @test layer_weights(zi, 0.5) ≈ [0.0, 0.2, 0.3]     # deepest included layer clipped
    @test sum(layer_weights(zi, 0.5)) ≈ 0.5            # weights sum to slab_depth
    @test sum(layer_weights(zi, 2.0)) ≈ 1.0            # clipped to the column depth

    # Faces given downward would otherwise silently produce all-zero weights.
    @test_throws ArgumentError layer_weights(reverse(zi), 1.0)

    @test layer_depths(zi) ≈ [0.8, 0.45, 0.15]         # midpoints, positive down
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
        α  = Array(interior(props.inverse_air_entry_head))[:, 1, 1]
        n  = Array(interior(props.pore_size_uniformity))[:, 1, 1]
        Ks = Array(interior(props.K_saturated))[:, 1, 1]

        @test all(0.3 .< ν .< 0.6)
        @test all(θʳ .< ν)
        @test all(n .> 1)
        @test all(Ks .> 0)

        # Per-layer values of the sand-over-clay column (deepest-first), each at its
        # own depth so the deep layers are read as subsoil.
        ptf = ContinuousPedotransfer()
        w = layer_weights(zi, 1.0); W = sum(w); d = layer_depths(zi)
        layers = ((0.20, 0.30, 0.50, 1400.0),   # clay
                  (0.20, 0.30, 0.50, 1400.0),   # clay
                  (0.90, 0.07, 0.03, 1400.0))   # sand
        per_layer = [soil_hydraulic_parameters(ptf, l..., dk) for (l, dk) in zip(layers, d)]
        Ks_layers = [p.K_saturated for p in per_layer]
        α_layers  = [p.inverse_air_entry_head for p in per_layer]
        n_layers  = [p.pore_size_uniformity for p in per_layer]

        Ks_harmonic   = W / sum(w ./ Ks_layers)
        Ks_arithmetic = sum(w .* Ks_layers) / W
        n_geometric   = 1 + exp(sum(w .* log.(n_layers .- 1)) / W)
        n_arithmetic  = sum(w .* n_layers) / W
        # Each layer's α weighted by the resistance it contributes, with the column's
        # own n as the exponent.
        α_resistance  = (sum(w .* α_layers .^ n_geometric ./ Ks_layers) /
                         sum(w ./ Ks_layers))^(1/n_geometric)
        α_geometric   = exp(sum(w .* log.(α_layers)) / W)

        # Kₛ upscales harmonically (clay-limited), strictly below the arithmetic mean.
        @test Ks[2] ≈ Ks_harmonic
        @test Ks_harmonic < Ks_arithmetic
        # n upscales geometrically in n - 1: contrasting layers flatten the column's
        # retention curve, so the effective n falls toward the smaller value.
        @test n[2] ≈ n_geometric
        @test n_geometric < n_arithmetic
        # α is resistance-weighted. The thin conductive layer carries little
        # resistance, so it moves α far less than a thickness-weighted mean would.
        @test α[2] ≈ α_resistance
        @test α_resistance < α_geometric
    end
end

@testset "layers outside the slab cannot contaminate a column" begin
    # Missing data below `slab_depth` arrives as NaN. Those layers carry zero weight,
    # and `0 * NaN` is NaN, so the mask that drops them is what keeps the column finite.
    zi = [-1.0, -0.6, -0.3, 0.0]
    for arch in test_architectures
        grid = RectilinearGrid(arch; size = (1, 1, 3), x = (0, 1), y = (0, 1), z = zi,
                               topology = (Bounded, Bounded, Bounded))
        sand = CenterField(grid); silt = CenterField(grid)
        clay = CenterField(grid); bulk_density = CenterField(grid)
        set!(sand, (x, y, z) -> z > -0.3 ? 0.90 : NaN)
        set!(silt, (x, y, z) -> z > -0.3 ? 0.07 : NaN)
        set!(clay, (x, y, z) -> z > -0.3 ? 0.03 : NaN)
        set!(bulk_density, (x, y, z) -> z > -0.3 ? 1400.0 : NaN)

        props = soil_hydraulic_properties(sand, silt, clay, bulk_density;
                                          slab_depth = 0.3, z_interfaces = zi)
        top = soil_hydraulic_parameters(ContinuousPedotransfer(), 0.90, 0.07, 0.03, 1400.0,
                                        layer_depths(zi)[3])
        @test Array(interior(props.porosity))[1, 1, 1]             ≈ top.porosity
        @test Array(interior(props.K_saturated))[1, 1, 1]          ≈ top.K_saturated
        @test Array(interior(props.pore_size_uniformity))[1, 1, 1]  ≈ top.pore_size_uniformity

        # Missing data *inside* the slab must still propagate — it is a real gap.
        holed = soil_hydraulic_properties(sand, silt, clay, bulk_density;
                                          slab_depth = 1.0, z_interfaces = zi)
        @test isnan(Array(interior(holed.porosity))[1, 1, 1])
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
        top = soil_hydraulic_parameters(ContinuousPedotransfer(), 0.90, 0.07, 0.03, 1400.0,
                                        layer_depths(zi)[3])
        @test Array(interior(props.porosity))[1, 1, 1]    ≈ top.porosity
        @test Array(interior(props.K_saturated))[1, 1, 1] ≈ top.K_saturated
        @test Array(interior(props.inverse_air_entry_head))[1, 1, 1] ≈ top.inverse_air_entry_head
    end
end

@testset "Field-backed van Genuchten closures" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch; size = (2, 1), x = (0, 2), y = (0, 1),
                               topology = (Bounded, Bounded, Flat))

        # Scalar path unchanged: matches the closed-form van Genuchten pressure head.
        r = VanGenuchtenRetention(inverse_air_entry_head = 2.0, pore_size_uniformity = 1.4)
        𝒮 = 0.5
        m = 1 - 1/1.4
        Π_ref = -(𝒮^(-1/m) - 1)^(1/1.4) / 2.0
        @test NumericalEarth.Lands.pressure_head(1, 1, grid, r, 𝒮) ≈ Π_ref

        # Two columns with different hydraulic parameters (Field-backed α, n, Kₛ, ν).
        makefield(v1, v2) = (f = Field{Center, Center, Nothing}(grid);
                             set!(f, (x, y) -> x < 1 ? v1 : v2); f)
        ν  = makefield(0.45, 0.35)
        α  = makefield(1.0, 4.0)
        n  = makefield(1.6, 1.2)
        Ks = makefield(1e-5, 1e-7)

        hydrology = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0, porosity = ν, storage_height = 1000,
            retention_curve = VanGenuchtenRetention(; inverse_air_entry_head = α, pore_size_uniformity = n),
            hydraulic_conductivity = VanGenuchtenConductivity(; K_saturated = Ks, pore_size_uniformity = n),
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
