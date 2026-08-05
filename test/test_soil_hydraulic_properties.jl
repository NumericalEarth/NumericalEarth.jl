include("runtests_setup.jl")

using Oceananigans
using Oceananigans.Fields: interior
using Oceananigans.TimeSteppers: time_step!

# (sand, silt, clay, bulk_density) in kg/kg and kg/m³.
sandy_loam = (0.55, 0.25, 0.20, 1500.0)
silt_loam  = (0.20, 0.65, 0.15, 1300.0)
clay_soil  = (0.25, 0.25, 0.50, 1400.0)
sand_soil  = (0.92, 0.05, 0.03, 1600.0)
textures   = (sandy_loam, silt_loam, clay_soil, sand_soil)

θ_van_genuchten(ψ, θʳ, ν, α, n) = θʳ + (ν - θʳ) * (1 + (α * ψ)^n)^(-(1 - 1/n))

@testset "pedotransfer functions" begin
    for ptf in (WeynantsPedotransfer(), HYPRESPedotransfer())
        for texture in textures
            p = soil_hydraulic_parameters(ptf, texture...)
            @test 0.3 < p.porosity < 0.6                        # physical range
            @test 0 <= p.residual_liquid_fraction < p.porosity
            @test p.pore_size_uniformity > 1                     # van Genuchten n
            @test p.inverse_air_entry_head > 0                   # m⁻¹
            @test p.matching_point_conductivity > 0                              # m s⁻¹
            @test isfinite(p.pore_connectivity_exponent)
        end

        # Sand drains far faster than clay.
        @test soil_hydraulic_parameters(ptf, sand_soil...).matching_point_conductivity >
              soil_hydraulic_parameters(ptf, clay_soil...).matching_point_conductivity

        # Bounded predictors: nothing outside the fit can push `θs` past 1 or `n` to 1,
        # where `m = 1 - 1/n` vanishes.
        for (sand, silt, clay, ρᵇ) in ((1.0, 0.0, 0.0, 1500.0),    # no clay, no silt
                                       (0.0, 0.0, 1.0, 1400.0),    # pure clay
                                       (0.4, 0.4, 0.2, 100.0),     # peat-like ρᵇ
                                       (0.4, 0.4, 0.2, 2600.0))    # rock-like ρᵇ
            p = soil_hydraulic_parameters(ptf, sand, silt, clay, ρᵇ)
            @test 0 < p.porosity < 1
            @test p.pore_size_uniformity > 1.01          # clear of the m = 0 singularity
            @test 1 - 1/p.pore_size_uniformity > 0.01
            @test 0 < p.inverse_air_entry_head < 1e3
            @test 0 < p.matching_point_conductivity < 1e-2
        end

        # Type stability: Float32 ptf → Float32 outputs, no allocation (kernels).
        ptf32 = NumericalEarth.Lands.convert_eltype(Float32, ptf)
        p32 = soil_hydraulic_parameters(ptf32, 0.4f0, 0.4f0, 0.2f0, 1400.0f0)
        @test all(v -> v isa Float32, values(p32))
        @test 0 == @allocated soil_hydraulic_parameters(ptf32, 0.4f0, 0.4f0, 0.2f0, 1400.0f0, 0.15f0)

        # Same regression either way, to Float32 precision.
        p64 = soil_hydraulic_parameters(NumericalEarth.Lands.convert_eltype(Float64, ptf),
                                        0.4, 0.4, 0.2, 1400.0)
        for name in keys(p64)
            @test p32[name] ≈ p64[name] rtol=1e-5 atol=1e-8
        end
    end
end

@testset "Weynants pedotransfer function" begin
    ptf = WeynantsPedotransfer(organic_carbon = 1.0)

    # Weihermüller et al. (2017) Fig. 1 gives K₀ ≈ 48 cm/day for a sand and ≈ 4 for a clay,
    # at ρᵇ = 1.4 g/cm³ and OC = 1 % *by weight*, at the USDA class centroids. Organic
    # carbon in g/kg — the mistake the erratum corrects — would scale K₀ by 0.24.
    sand_centroid = soil_hydraulic_parameters(ptf, 0.95, 0.02, 0.03, 1400.0)
    clay_centroid = soil_hydraulic_parameters(ptf, 0.20, 0.20, 0.60, 1400.0)
    @test sand_centroid.matching_point_conductivity * 8.64e6 ≈ 48 atol=1.5                # cm/day
    @test clay_centroid.matching_point_conductivity * 8.64e6 ≈ 4.5 atol=1.0
    @test clay_centroid.inverse_air_entry_head / 100 ≈ 0.0095 atol=0.0005  # cm⁻¹
    # Bulk density is g/cm³ and texture is %, so a kg/m³ or fraction slip is caught too.
    @test soil_hydraulic_parameters(ptf, 0.20, 0.20, 0.60, 1770.0).porosity <
          soil_hydraulic_parameters(ptf, 0.20, 0.20, 0.60,  890.0).porosity

    # θʳ is dropped, not fitted: Weynants found it indistinguishable from zero.
    @test all(soil_hydraulic_parameters(ptf, t...).residual_liquid_fraction == 0
              for t in textures)

    # ℓ is strongly negative and steepens with clay — nowhere near Mualem's 1/2.
    ℓ(t) = soil_hydraulic_parameters(ptf, t...).pore_connectivity_exponent
    @test ℓ(sand_soil) > ℓ(sandy_loam) > ℓ(clay_soil)
    @test -1 > ℓ(sand_soil) > -3        # -1.64 for a sand
    @test -6 > ℓ(clay_soil) > -12       # -8.28 for a 50 % clay

    # K₀ is a matrix matching point, so it must sit below a macropore-inclusive Kₛ.
    for t in textures
        @test soil_hydraulic_parameters(ptf, t...).matching_point_conductivity <
              saturated_conductivity(CosbyConductivity(), t[1])
    end

    # Organic carbon enters α and Kₛ only, and lowers both.
    lean = soil_hydraulic_parameters(WeynantsPedotransfer(organic_carbon = 0.58), silt_loam...)
    rich = soil_hydraulic_parameters(WeynantsPedotransfer(organic_carbon = 3.0), silt_loam...)
    @test rich.matching_point_conductivity < lean.matching_point_conductivity
    @test rich.inverse_air_entry_head < lean.inverse_air_entry_head
    @test rich.pore_size_uniformity == lean.pore_size_uniformity
    @test rich.porosity == lean.porosity
end

@testset "HYPRES pedotransfer function" begin
    ptf = HYPRESPedotransfer()

    # HYPRES publishes no θʳ and no usable ℓ, so both are constants: Wösten's Table 4
    # class fit, and the Mualem exponent an unreduced matching point pairs with.
    @test all(soil_hydraulic_parameters(ptf, t...).residual_liquid_fraction == 0.01
              for t in textures)
    @test all(soil_hydraulic_parameters(ptf, t...).pore_connectivity_exponent == 0.5
              for t in textures)
    custom = HYPRESPedotransfer(residual_liquid_fraction = 0.025,
                                pore_connectivity_exponent = -0.31)
    @test soil_hydraulic_parameters(custom, sand_soil...).residual_liquid_fraction == 0.025
    @test soil_hydraulic_parameters(custom, sand_soil...).pore_connectivity_exponent == -0.31

    # The topsoil flag comes off the layer depth, and for clay it is worth a factor of
    # several in Kₛ: a clay topsoil drains through its aggregate structure, a subsoil does not.
    clay_top  = soil_hydraulic_parameters(ptf, clay_soil..., 0.15)
    clay_deep = soil_hydraulic_parameters(ptf, clay_soil..., 0.80)
    @test clay_top.matching_point_conductivity > 3 * clay_deep.matching_point_conductivity
    # Omitting the depth reports the surface value.
    @test soil_hydraulic_parameters(ptf, clay_soil...).matching_point_conductivity ==
          clay_top.matching_point_conductivity
end

@testset "pedotransfer float type follows the grid" begin
    # Devices reject unsupported float types outright (Metal has no Float64), so no
    # Float64 may survive in the pedotransfer function handed to the kernel.
    for (ptf, R) in ((WeynantsPedotransfer(Float64), WeynantsRegression),
                     (HYPRESPedotransfer(Float64), HYPRESRegression))
        ptf32 = NumericalEarth.Lands.convert_eltype(Float32, ptf)
        @test ptf32 isa typeof(ptf).name.wrapper{Float32}
        coefficients = ptf32.regression_coefficients
        @test all(name -> eltype(getfield(coefficients, name)) === Float32, fieldnames(R))
    end

    # A Float32 grid must produce Float32 property fields from the default (Float64) ptf.
    grid = RectilinearGrid(Float32; size = (1, 1, 3), x = (0, 1), y = (0, 1),
                           z = [-1.0, -0.6, -0.3, 0.0], topology = (Bounded, Bounded, Bounded))
    texture = map(_ -> CenterField(grid), (1, 2, 3, 4))
    set!(texture[1], 0.4); set!(texture[2], 0.4); set!(texture[3], 0.2); set!(texture[4], 1400)
    props = soil_hydraulic_properties(texture...; slab_depth = 1.0,
                                      z_interfaces = [-1.0, -0.6, -0.3, 0.0])
    @test all(f -> eltype(f) === Float32, values(props))
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

@testset "matched_retention_parameters inverts the retention curve" begin
    matched = NumericalEarth.Lands.matched_retention_parameters
    θʳ, ν, ψ¹, ψ² = 0.03, 0.45, 1.0, 150.0
    recover(α, n) = matched(θ_van_genuchten(ψ¹, θʳ, ν, α, n),
                            θ_van_genuchten(ψ², θʳ, ν, α, n), θʳ, ν, ψ¹, ψ²)

    # Exact wherever both heads carry information, which covers the whole range either
    # pedotransfer function can produce (α ≤ 8.7 m⁻¹, n ≤ 2.3).
    for (α, n) in ((1.0, 1.05), (2.0, 1.3), (0.5, 2.0), (3.0, 1.6), (8.7, 2.3), (0.8, 1.08))
        αᶜ, nᶜ = recover(α, n)
        @test nᶜ ≈ n rtol=1e-6
        @test αᶜ ≈ α rtol=1e-6
    end

    # The boundary: drained past ~2 % saturation at the wetter head, both water contents
    # sit within rounding of θʳ and no (α, n) is determined by them. That needs n ≳ 3 with
    # α ≳ 3 m⁻¹, which no shipped pedotransfer function reaches.
    @test !isapprox(first(recover(8.0, 4.5)), 8.0, rtol=1e-2)
    for (α, n) in ((8.0, 4.5), (20.0, 4.5))          # still bounded, just meaningless
        αᶜ, nᶜ = recover(α, n)
        @test isfinite(αᶜ) && 1 < nᶜ < 12
    end

    # A degenerate pair, no water lost between the two heads, must stay bounded.
    αᶜ, nᶜ = matched(0.30, 0.30, 0.0, 0.45, 1.0, 150.0)
    @test isfinite(αᶜ) && 1 < nᶜ < 12

    # Float32 reaches the same answer: the bisection tests only its midpoint, so the
    # overflow of `𝒮^(-1/m)` at the brackets cannot poison it.
    α32, n32 = matched(0.30f0, 0.12f0, 0.0f0, 0.45f0, 1.0f0, 150.0f0)
    α64, n64 = matched(0.30, 0.12, 0.0, 0.45, 1.0, 150.0)
    @test n32 ≈ n64 rtol=1e-4
    @test α32 ≈ α64 rtol=1e-4
end

@testset "soil_hydraulic_properties reduction" begin
    zi = [-1.0, -0.6, -0.3, 0.0]
    heads = (1, 150)

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
                                          slab_depth = 1.0, z_interfaces = zi,
                                          matching_heads = heads)

        # Outputs are 2-D (Center, Center, Nothing) fields the slab reads at [i, j].
        @test location(props.porosity) == (Center, Center, Nothing)
        @test size(props.matching_point_conductivity) == (2, 1, 1)
        @test keys(props) == (:porosity, :residual_liquid_fraction, :inverse_air_entry_head,
                              :pore_size_uniformity, :matching_point_conductivity,
                              :pore_connectivity_exponent)

        column(name) = Array(interior(props[name]))[:, 1, 1]
        ν, θʳ = column(:porosity), column(:residual_liquid_fraction)
        α, n  = column(:inverse_air_entry_head), column(:pore_size_uniformity)
        Ks, ℓ = column(:matching_point_conductivity), column(:pore_connectivity_exponent)

        @test all(0.3 .< ν .< 0.6)
        @test all(θʳ .< ν)
        @test all(n .> 1)
        @test all(Ks .> 0)

        # Per-layer values of the sand-over-clay column (deepest-first), each at its
        # own depth so the deep layers are read as subsoil.
        ptf = WeynantsPedotransfer()
        w = layer_weights(zi, 1.0); W = sum(w); d = layer_depths(zi)
        layers = ((0.20, 0.30, 0.50, 1400.0),   # clay
                  (0.20, 0.30, 0.50, 1400.0),   # clay
                  (0.90, 0.07, 0.03, 1400.0))   # sand
        per_layer = [soil_hydraulic_parameters(ptf, l..., dk) for (l, dk) in zip(layers, d)]

        # ν and θʳ are arithmetic means, exact at ψ = 0 and ψ → ∞.
        @test ν[2]  ≈ sum(w .* [p.porosity for p in per_layer]) / W
        @test θʳ[2] ≈ sum(w .* [p.residual_liquid_fraction for p in per_layer]) / W

        # Kₛ upscales harmonically (clay-limited), strictly below the arithmetic mean.
        Ks_layers   = [p.matching_point_conductivity for p in per_layer]
        Ks_harmonic = W / sum(w ./ Ks_layers)
        @test Ks[2] ≈ Ks_harmonic
        @test Ks_harmonic < sum(w .* Ks_layers) / W

        @test ℓ[2] ≈ sum(w .* [p.pore_connectivity_exponent for p in per_layer]) / W

        # α and n are *defined* by passing through the thickness-weighted mean retention
        # curve at both matching heads. That identity is the whole reduction.
        for (k, ψ) in Iterators.product(1:2, heads)
            layer_set = k == 1 ? fill((0.40, 0.40, 0.20, 1400.0), 3) : layers
            ps = [soil_hydraulic_parameters(ptf, l..., dk) for (l, dk) in zip(layer_set, d)]
            θ̄ = sum(w[j] * θ_van_genuchten(ψ, ps[j].residual_liquid_fraction, ps[j].porosity,
                                           ps[j].inverse_air_entry_head,
                                           ps[j].pore_size_uniformity)
                    for j in eachindex(ps)) / W
            @test θ_van_genuchten(ψ, θʳ[k], ν[k], α[k], n[k]) ≈ θ̄ rtol=1e-6
        end

        # A uniform column must reduce to its own layer parameters, exactly.
        uniform = soil_hydraulic_parameters(ptf, 0.40, 0.40, 0.20, 1400.0, d[1])
        @test α[1] ≈ uniform.inverse_air_entry_head rtol=1e-8
        @test n[1] ≈ uniform.pore_size_uniformity   rtol=1e-8
        @test ℓ[1] ≈ uniform.pore_connectivity_exponent
    end

    # Two increasing positive heads, or nothing.
    grid = RectilinearGrid(size = (1, 1, 3), x = (0, 1), y = (0, 1), z = zi,
                           topology = (Bounded, Bounded, Bounded))
    f = map(_ -> CenterField(grid), (1, 2, 3, 4))
    set!(f[1], 0.4); set!(f[2], 0.4); set!(f[3], 0.2); set!(f[4], 1400)
    @test_throws ArgumentError soil_hydraulic_properties(f...; slab_depth = 1.0,
                                  z_interfaces = zi, matching_heads = (150, 1))
    @test_throws ArgumentError soil_hydraulic_properties(f...; slab_depth = 1.0,
                                  z_interfaces = zi, matching_heads = (0, 150))
end

@testset "layers outside the slab cannot contaminate a column" begin
    # Missing data below `slab_depth` arrives as NaN. Zero weight is not enough to drop
    # those layers, since `0 * NaN` is NaN.
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
        top = soil_hydraulic_parameters(WeynantsPedotransfer(), 0.90, 0.07, 0.03, 1400.0,
                                        layer_depths(zi)[3])
        for name in keys(props)
            @test Array(interior(props[name]))[1, 1, 1] ≈ top[name] rtol=1e-8
        end

        # Missing data *inside* the slab must propagate to every *predicted* output. The
        # bisection for (α, n) is where this is easy to lose: `NaN > 0` is false, so
        # unguarded it converges on its own bracket and reports a plausible n = 12.
        holed = soil_hydraulic_properties(sand, silt, clay, bulk_density;
                                          slab_depth = 1.0, z_interfaces = zi)
        for name in (:porosity, :inverse_air_entry_head, :pore_size_uniformity,
                     :matching_point_conductivity, :pore_connectivity_exponent)
            @test isnan(Array(interior(holed[name]))[1, 1, 1])
        end
        # Constants never read the data, so they stay constant.
        @test Array(interior(holed.residual_liquid_fraction))[1, 1, 1] == 0
        hypres_holed = soil_hydraulic_properties(sand, silt, clay, bulk_density;
                                                 slab_depth = 1.0, z_interfaces = zi,
                                                 ptf = HYPRESPedotransfer())
        @test Array(interior(hypres_holed.residual_liquid_fraction))[1, 1, 1] == 0.01
        @test Array(interior(hypres_holed.pore_connectivity_exponent))[1, 1, 1] == 0.5
        @test isnan(Array(interior(hypres_holed.porosity))[1, 1, 1])
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
        for ptf in (WeynantsPedotransfer(), HYPRESPedotransfer())
            props = soil_hydraulic_properties(sand, silt, clay, bulk_density;
                                              slab_depth = 0.3, z_interfaces = zi, ptf)
            top = soil_hydraulic_parameters(ptf, 0.90, 0.07, 0.03, 1400.0, layer_depths(zi)[3])
            for name in keys(props)
                @test Array(interior(props[name]))[1, 1, 1] ≈ top[name] rtol=1e-8
            end
        end
    end
end

@testset "Field-backed van Genuchten closures" begin
    for arch in test_architectures
        grid = RectilinearGrid(arch; size = (2, 1), x = (0, 2), y = (0, 1),
                               topology = (Bounded, Bounded, Flat))

        r = VanGenuchtenRetention(inverse_air_entry_head = 2.0, pore_size_uniformity = 1.4)
        𝒮 = 0.5
        m = 1 - 1/1.4
        Π_ref = -(𝒮^(-1/m) - 1)^(1/1.4) / 2.0
        @test NumericalEarth.Lands.pressure_head(1, 1, grid, r, 𝒮) ≈ Π_ref
        # `van_genuchten_saturation` is its inverse, and the reduction relies on that.
        @test NumericalEarth.Lands.van_genuchten_saturation(2.0 * abs(Π_ref), 1.4) ≈ 𝒮

        makefield(v1, v2) = (f = Field{Center, Center, Nothing}(grid);
                             set!(f, (x, y) -> x < 1 ? v1 : v2); f)
        ν  = makefield(0.45, 0.35)
        α  = makefield(1.0, 4.0)
        n  = makefield(1.6, 1.2)
        Ks = makefield(1e-5, 1e-7)
        ℓ  = makefield(-1.6, -8.9)

        hydrology = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0, porosity = ν, storage_height = 1000,
            retention_curve = VanGenuchtenRetention(; inverse_air_entry_head = α, pore_size_uniformity = n),
            hydraulic_conductivity = VanGenuchtenConductivity(; matching_point_conductivity = Ks,
                                        pore_size_uniformity = n, pore_connectivity_exponent = ℓ),
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

@testset "conductivity closure defaults and the Kₛ / K₀ split" begin
    # ℓ defaults to Mualem's 1/2, the value an unreduced matching point pairs with.
    c = VanGenuchtenConductivity(matching_point_conductivity = 1e-6, pore_size_uniformity = 1.5)
    @test c.pore_connectivity_exponent == 0.5

    # Cosby's Kₛ spans the texture triangle far more widely than HYPRES's regression and
    # rises monotonically with sand, which is why it is here at all.
    Ks = [saturated_conductivity(CosbyConductivity(), s) for s in (0.05, 0.25, 0.45, 0.65, 0.85, 0.95)]
    @test issorted(Ks)
    @test Ks[end] / Ks[1] > 15
    @test saturated_conductivity(CosbyConductivity(), 0.92) * 3.6e6 ≈ 84.8 atol=0.5    # mm/hr
    # The within-class spread is half a decade or more, which bounds any calibration.
    @test conductivity_spread(CosbyConductivity(), 0.05) ≈ 0.475 atol=0.001
    @test conductivity_spread(CosbyConductivity(), 0.65) > conductivity_spread(CosbyConductivity(), 0.05)

    # Float type comes from the constructor, as for every other parameterized closure.
    c32 = CosbyConductivity(Float32)
    @test c32 isa CosbyConductivity{Float32}
    @test saturated_conductivity(c32, 0.2f0) isa Float32
    @test conductivity_spread(c32, 0.2f0) isa Float32
    @test saturated_conductivity(c32, 0.92f0) ≈ saturated_conductivity(CosbyConductivity(), 0.92) rtol=1e-5

    # Recalibrating means constructing a different one, not editing the source.
    steeper = CosbyConductivity(sand_coefficient = 0.02)
    @test saturated_conductivity(steeper, 0.92) > saturated_conductivity(CosbyConductivity(), 0.92)
    @test saturated_conductivity(steeper, 0.0) == saturated_conductivity(CosbyConductivity(), 0.0)
end

@testset "critical saturation and the evaporation shutoff point" begin
    # 𝒮ᶜ is a function of n alone (α cancels out of α hᶜ), and falls as the pore-size
    # distribution narrows, so coarse soils sustain capillary supply to lower saturation.
    @test critical_saturation(1.2) > critical_saturation(1.5) > critical_saturation(3.0)
    @test all(0 .< critical_saturation.([1.05, 1.2, 1.5, 3.0, 7.0]) .< 1)

    # θ½ ≈ θʳ + (ν - θʳ)𝒮ᶜ is the water content at which bare-soil evaporation falls to
    # half its potential rate. Merlin et al. (2016) fitted θ½ = 0.20 + 0.28fᶜˡᵃʸ -
    # 0.16fˢᵃⁿᵈ to 34 flux-tower sites. Both pedotransfer functions run wet against it by
    # a similar amount, so this bounds the offset rather than targeting it.
    θhalf_observed(fsand, fclay) = 0.20 + 0.28fclay - 0.16fsand
    for ptf in (WeynantsPedotransfer(), HYPRESPedotransfer())
        errors = Float64[]
        for (sand, silt, clay, ρᵇ) in textures
            p = soil_hydraulic_parameters(ptf, sand, silt, clay, ρᵇ)
            θr, ν = p.residual_liquid_fraction, p.porosity
            θhalf = θr + (ν - θr) * critical_saturation(p.pore_size_uniformity)
            @test θr <= θhalf < ν
            push!(errors, θhalf - θhalf_observed(sand, clay))
        end
        @test all(>(0), errors)                               # shutoff is predicted too wet
        @test sqrt(sum(errors.^2) / length(errors)) < 0.18    # by this much, and no more
    end
end

@testset "viscosity correction on hydraulic conductivity" begin
    v = WaterViscosity()
    Θ(T) = viscosity_correction(v, T)
    # Darcy conductivity tracks the inverse dynamic viscosity of water, so K rises with T.
    @test Θ(v.reference_temperature) == 1
    @test Θ(298.0) > 1 > Θ(275.0)
    # Against tabulated μ: μ(15 °C)/μ(20 °C) = 1.1375/1.0016, and K ∝ 1/μ.
    @test Θ(288.15) / Θ(293.15) ≈ 1.0016 / 1.1375 rtol=5e-3
    # About 30 % per 10 K, and a factor of 2.4 across a realistic soil cycle.
    @test Θ(293.0) / Θ(283.0) ≈ 1.30 atol=0.02
    @test Θ(310.0) / Θ(275.0) ≈ 2.41 atol=0.05
    @test viscosity_correction(WaterViscosity(Float32), 298.0f0) isa Float32
    # The pole of the viscosity law sits at 149.3 K; the floor keeps it unreachable.
    @test 0 < Θ(100.0) == Θ(150.0) < 0.01

    # Shifting the reference moves where Θ = 1, which is how a K₀ measured at another
    # temperature is reconciled.
    warm = WaterViscosity(reference_temperature = 298)
    @test viscosity_correction(warm, 298.0) == 1
    @test viscosity_correction(warm, 288.0) < 1
    @test viscosity_correction(warm, 288.0) ≈ Θ(288.0) / Θ(298.0)

    # The corrected conductivity is the isothermal one scaled by that factor.
    grid = RectilinearGrid(size = (1, 1, 1), x = (0, 1), y = (0, 1), z = (-1, 0),
                           topology = (Bounded, Bounded, Bounded))
    c = VanGenuchtenConductivity(matching_point_conductivity = 1e-6, pore_size_uniformity = 1.5)
    K₀ = NumericalEarth.Lands.hydraulic_conductivity(1, 1, grid, c, 0.6)
    Kᵀ = NumericalEarth.Lands.hydraulic_conductivity(1, 1, grid, c, 0.6, 298.0)
    @test Kᵀ ≈ K₀ * Θ(298.0)
    @test NumericalEarth.Lands.hydraulic_conductivity(1, 1, grid, c, 0.6, 288.0) ≈ K₀
end
