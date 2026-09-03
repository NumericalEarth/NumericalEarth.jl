include("runtests_setup.jl")

using Oceananigans
using Oceananigans.Fields: interior
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth.Lands: hydraulic_conductivity, pressure_head,
                            matched_retention_parameters, van_genuchten_saturation

# (sand, silt, clay, bulk_density) in kg/kg and kg/m³.
sandy_loam = (0.55, 0.25, 0.20, 1500.0)
silt_loam  = (0.20, 0.65, 0.15, 1300.0)
clay_soil  = (0.25, 0.25, 0.50, 1400.0)
sand_soil  = (0.92, 0.05, 0.03, 1600.0)
textures   = (sandy_loam, silt_loam, clay_soil, sand_soil)

van_genuchten_water_content(ψ, θʳ, ν, α, 𝓃) = θʳ + (ν - θʳ) * (1 + (α * ψ)^𝓃)^(-(1 - 1/𝓃))

# OpenLandMap depth faces: 60–100, 30–60 and 0–30 cm.
z_interfaces = [-1.0, -0.6, -0.3, 0.0]

soil_column_grid(arch, Nx = 1) =
    RectilinearGrid(arch; size = (Nx, 1, 3), x = (0, Nx), y = (0, 1), z = z_interfaces,
                    topology = (Bounded, Bounded, Bounded))

function texture_fields(grid; sand, silt, clay, bulk_density = 1400)
    fields = map(_ -> CenterField(grid), (1, 2, 3, 4))
    set!(fields[1], sand); set!(fields[2], silt); set!(fields[3], clay); set!(fields[4], bulk_density)
    return fields
end

column(field) = Array(interior(field))[:, 1, 1]

@testset "pedotransfer functions" begin
    for (ptf, ptf32) in ((WeynantsPedotransfer(), WeynantsPedotransfer(Float32)),
                         (HYPRESPedotransfer(), HYPRESPedotransfer(Float32)))
        for texture in textures
            p = soil_hydraulic_parameters(ptf, texture...)
            @test 0.3 < p.porosity < 0.6
            @test 0 <= p.residual_liquid_fraction < p.porosity
            @test p.pore_size_uniformity > 1
            @test p.inverse_air_entry_head > 0
            @test p.matching_point_conductivity > 0
            @test isfinite(p.pore_connectivity_exponent)
        end

        @test soil_hydraulic_parameters(ptf, sand_soil...).matching_point_conductivity >
              soil_hydraulic_parameters(ptf, clay_soil...).matching_point_conductivity

        # Predictors are held inside the fit's range, so extreme inputs give boundary soils.
        for (sand, silt, clay, ρᵇ) in ((1.0, 0.0, 0.0, 1500.0), (0.0, 0.0, 1.0, 1400.0),
                                       (0.4, 0.4, 0.2, 100.0), (0.4, 0.4, 0.2, 2600.0))
            p = soil_hydraulic_parameters(ptf, sand, silt, clay, ρᵇ)
            @test 0 < p.porosity < 1
            @test 1 - 1/p.pore_size_uniformity > 0.01
            @test 0 < p.inverse_air_entry_head < 1e3
            @test 0 < p.matching_point_conductivity < 1e-2
        end

        p32 = soil_hydraulic_parameters(ptf32, 0.4f0, 0.4f0, 0.2f0, 1400.0f0)
        p64 = soil_hydraulic_parameters(ptf, 0.4, 0.4, 0.2, 1400.0)
        @test all(v -> v isa Float32, values(p32))
        @test 0 == @allocated soil_hydraulic_parameters(ptf32, 0.4f0, 0.4f0, 0.2f0, 1400.0f0, 0.15f0)
        for name in keys(p64)
            @test p32[name] ≈ p64[name] rtol=1e-5
        end
    end
end

@testset "Weynants pedotransfer function" begin
    ptf = WeynantsPedotransfer(organic_carbon = 1.0)

    # Weihermüller et al. (2017) Fig. 1: K₀ ≈ 48 cm/day for a sand and ≈ 4 for a clay at
    # ρᵇ = 1.4 g/cm³ and OC = 1 % by weight, at the USDA class centroids.
    sand_centroid = soil_hydraulic_parameters(ptf, 0.95, 0.02, 0.03, 1400.0)
    clay_centroid = soil_hydraulic_parameters(ptf, 0.20, 0.20, 0.60, 1400.0)
    @test sand_centroid.matching_point_conductivity * 8.64e6 ≈ 48 atol=1.5
    @test clay_centroid.matching_point_conductivity * 8.64e6 ≈ 4.5 atol=1.0
    @test clay_centroid.inverse_air_entry_head / 100 ≈ 0.0095 atol=0.0005
    @test soil_hydraulic_parameters(ptf, 0.20, 0.20, 0.60, 1770.0).porosity <
          soil_hydraulic_parameters(ptf, 0.20, 0.20, 0.60,  890.0).porosity

    @test all(soil_hydraulic_parameters(ptf, t...).residual_liquid_fraction == 0 for t in textures)

    # ηᴷ is strongly negative and steepens with clay.
    ηᴷ(t) = soil_hydraulic_parameters(ptf, t...).pore_connectivity_exponent
    @test ηᴷ(sand_soil) > ηᴷ(sandy_loam) > ηᴷ(clay_soil)
    @test -1 > ηᴷ(sand_soil) > -3
    @test -6 > ηᴷ(clay_soil) > -12

    # Organic carbon enters α and K₀ only, and lowers both.
    lean = soil_hydraulic_parameters(WeynantsPedotransfer(organic_carbon = 0.58), silt_loam...)
    rich = soil_hydraulic_parameters(WeynantsPedotransfer(organic_carbon = 3.0), silt_loam...)
    @test rich.matching_point_conductivity < lean.matching_point_conductivity
    @test rich.inverse_air_entry_head < lean.inverse_air_entry_head
    @test rich.pore_size_uniformity == lean.pore_size_uniformity
    @test rich.porosity == lean.porosity
end

@testset "HYPRES pedotransfer function" begin
    ptf = HYPRESPedotransfer()

    @test all(soil_hydraulic_parameters(ptf, t...).residual_liquid_fraction == 0.01 for t in textures)
    @test all(soil_hydraulic_parameters(ptf, t...).pore_connectivity_exponent == 0.5 for t in textures)
    custom = HYPRESPedotransfer(residual_liquid_fraction = 0.025, pore_connectivity_exponent = -0.31)
    @test soil_hydraulic_parameters(custom, sand_soil...).residual_liquid_fraction == 0.025
    @test soil_hydraulic_parameters(custom, sand_soil...).pore_connectivity_exponent == -0.31

    # The topsoil flag comes off the layer depth and is worth a factor of several in K₀ for clay.
    clay_top  = soil_hydraulic_parameters(ptf, clay_soil..., 0.15)
    clay_deep = soil_hydraulic_parameters(ptf, clay_soil..., 0.80)
    @test clay_top.matching_point_conductivity > 3 * clay_deep.matching_point_conductivity
    @test soil_hydraulic_parameters(ptf, clay_soil...).matching_point_conductivity ==
          clay_top.matching_point_conductivity
end

@testset "property float type follows the grid" begin
    grid = RectilinearGrid(Float32; size = (1, 1, 3), x = (0, 1), y = (0, 1), z = z_interfaces,
                           topology = (Bounded, Bounded, Bounded))
    texture = texture_fields(grid; sand = 0.4, silt = 0.4, clay = 0.2)
    for ptf in (WeynantsPedotransfer(), HYPRESPedotransfer())
        props = soil_hydraulic_properties(texture...; slab_depth = 1.0, ptf)
        @test all(f -> eltype(f) === Float32, values(props))
    end
end

@testset "layer_weights and layer_depths" begin
    @test layer_weights(z_interfaces, 1.0) ≈ [0.4, 0.3, 0.3]
    @test layer_weights(z_interfaces, 0.3) ≈ [0.0, 0.0, 0.3]
    @test layer_weights(z_interfaces, 0.5) ≈ [0.0, 0.2, 0.3]
    @test sum(layer_weights(z_interfaces, 0.5)) ≈ 0.5
    @test sum(layer_weights(z_interfaces, 2.0)) ≈ 1.0
    @test layer_depths(z_interfaces) ≈ [0.8, 0.45, 0.15]
end

@testset "matched_retention_parameters inverts the retention curve" begin
    θʳ, ν, ψ¹, ψ² = 0.03, 0.45, 1.0, 150.0
    recover(α, 𝓃) = matched_retention_parameters(van_genuchten_water_content(ψ¹, θʳ, ν, α, 𝓃),
                                                 van_genuchten_water_content(ψ², θʳ, ν, α, 𝓃),
                                                 θʳ, ν, ψ¹, ψ²)

    for (α, 𝓃) in ((1.0, 1.05), (2.0, 1.3), (0.5, 2.0), (3.0, 1.6), (8.7, 2.3), (0.8, 1.08),
                   (8.0, 4.5), (20.0, 4.5))
        αᵉ, 𝓃ᵉ = recover(α, 𝓃)
        @test 𝓃ᵉ ≈ 𝓃 rtol=1e-6
        @test αᵉ ≈ α rtol=1e-6
    end

    matched_retention_parameters(0.30, 0.12, 0.0, 0.45, 1.0, 150.0)
    @test 0 == @allocated matched_retention_parameters(0.30, 0.12, 0.0, 0.45, 1.0, 150.0)

    α32, 𝓃32 = matched_retention_parameters(0.30f0, 0.12f0, 0.0f0, 0.45f0, 1.0f0, 150.0f0)
    α64, 𝓃64 = matched_retention_parameters(0.30, 0.12, 0.0, 0.45, 1.0, 150.0)
    @test 𝓃32 ≈ 𝓃64 rtol=1e-4
    @test α32 ≈ α64 rtol=1e-4
end

@testset "soil_hydraulic_properties reduction" begin
    heads = (1, 150)
    ptf = WeynantsPedotransfer()
    Δz = layer_weights(z_interfaces, 1.0); ΣΔz = sum(Δz); depths = layer_depths(z_interfaces)

    for arch in test_architectures
        grid = soil_column_grid(arch, 2)
        # Column 1: uniform loam. Column 2: sand (top) over clay (deep).
        texture = texture_fields(grid;
                                 sand = (x, y, z) -> x < 1 ? 0.40 : (z > -0.3 ? 0.90 : 0.20),
                                 silt = (x, y, z) -> x < 1 ? 0.40 : (z > -0.3 ? 0.07 : 0.30),
                                 clay = (x, y, z) -> x < 1 ? 0.20 : (z > -0.3 ? 0.03 : 0.50))

        props = soil_hydraulic_properties(texture...; slab_depth = 1.0, matching_heads = heads)
        @test location(props.porosity) == (Center, Center, Nothing)
        @test size(props.matching_point_conductivity) == (2, 1, 1)

        ν, θʳ  = column(props.porosity), column(props.residual_liquid_fraction)
        α, 𝓃   = column(props.inverse_air_entry_head), column(props.pore_size_uniformity)
        K₀, ηᴷ = column(props.matching_point_conductivity), column(props.pore_connectivity_exponent)

        loam   = fill((0.40, 0.40, 0.20, 1400.0), 3)
        layers = ((0.20, 0.30, 0.50, 1400.0), (0.20, 0.30, 0.50, 1400.0), (0.90, 0.07, 0.03, 1400.0))
        per_layer = [soil_hydraulic_parameters(ptf, l..., d) for (l, d) in zip(layers, depths)]

        # ν, θʳ and ηᴷ are thickness-weighted arithmetic means, K₀ the harmonic mean.
        @test ν[2]  ≈ sum(Δz .* [p.porosity for p in per_layer]) / ΣΔz
        @test θʳ[2] ≈ sum(Δz .* [p.residual_liquid_fraction for p in per_layer]) / ΣΔz
        @test ηᴷ[2] ≈ sum(Δz .* [p.pore_connectivity_exponent for p in per_layer]) / ΣΔz
        Kᵏ = [p.matching_point_conductivity for p in per_layer]
        @test K₀[2] ≈ ΣΔz / sum(Δz ./ Kᵏ)
        @test K₀[2] < sum(Δz .* Kᵏ) / ΣΔz

        # α and 𝓃 put the effective curve through the thickness-weighted mean curve at both heads.
        for (k, ψ) in Iterators.product(1:2, heads)
            ps = [soil_hydraulic_parameters(ptf, l..., d) for (l, d) in zip(k == 1 ? loam : layers, depths)]
            θ̄ = sum(Δz[j] * van_genuchten_water_content(ψ, ps[j].residual_liquid_fraction, ps[j].porosity,
                                                       ps[j].inverse_air_entry_head, ps[j].pore_size_uniformity)
                    for j in eachindex(ps)) / ΣΔz
            @test van_genuchten_water_content(ψ, θʳ[k], ν[k], α[k], 𝓃[k]) ≈ θ̄ rtol=1e-6
        end

        # A uniform column reduces to its own layer parameters.
        uniform = soil_hydraulic_parameters(ptf, loam[1]..., depths[1])
        @test α[1] ≈ uniform.inverse_air_entry_head rtol=1e-8
        @test 𝓃[1] ≈ uniform.pore_size_uniformity   rtol=1e-8
        @test ηᴷ[1] ≈ uniform.pore_connectivity_exponent
    end
end

@testset "layers outside the slab cannot contaminate a column" begin
    for arch in test_architectures
        grid = soil_column_grid(arch)
        texture = texture_fields(grid; sand = (x, y, z) -> z > -0.3 ? 0.90 : NaN,
                                       silt = (x, y, z) -> z > -0.3 ? 0.07 : NaN,
                                       clay = (x, y, z) -> z > -0.3 ? 0.03 : NaN,
                                       bulk_density = (x, y, z) -> z > -0.3 ? 1400.0 : NaN)

        props = soil_hydraulic_properties(texture...; slab_depth = 0.3)
        top = soil_hydraulic_parameters(WeynantsPedotransfer(), 0.90, 0.07, 0.03, 1400.0, layer_depths(z_interfaces)[3])
        for name in keys(props)
            @test only(column(props[name])) ≈ top[name] rtol=1e-8
        end

        # Missing data inside the slab propagates to every predicted output; constants stay constant.
        holed = soil_hydraulic_properties(texture...; slab_depth = 1.0)
        for name in (:porosity, :inverse_air_entry_head, :pore_size_uniformity,
                     :matching_point_conductivity, :pore_connectivity_exponent)
            @test isnan(only(column(holed[name])))
        end
        @test only(column(holed.residual_liquid_fraction)) == 0
        hypres_holed = soil_hydraulic_properties(texture...; slab_depth = 1.0, ptf = HYPRESPedotransfer())
        @test only(column(hypres_holed.residual_liquid_fraction)) == 0.01
        @test only(column(hypres_holed.pore_connectivity_exponent)) == 0.5
        @test isnan(only(column(hypres_holed.porosity)))
    end
end

@testset "reduction degenerates to a single layer for a thin slab" begin
    for arch in test_architectures
        grid = soil_column_grid(arch)
        texture = texture_fields(grid; sand = (x, y, z) -> z > -0.3 ? 0.90 : 0.20,
                                       silt = (x, y, z) -> z > -0.3 ? 0.07 : 0.30,
                                       clay = (x, y, z) -> z > -0.3 ? 0.03 : 0.50)

        for ptf in (WeynantsPedotransfer(), HYPRESPedotransfer())
            props = soil_hydraulic_properties(texture...; slab_depth = 0.3, ptf)
            top = soil_hydraulic_parameters(ptf, 0.90, 0.07, 0.03, 1400.0, layer_depths(z_interfaces)[3])
            for name in keys(props)
                @test only(column(props[name])) ≈ top[name] rtol=1e-8
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
        𝓂 = 1 - 1/1.4
        Π = -(𝒮^(-1/𝓂) - 1)^(1/1.4) / 2.0
        @test pressure_head(1, 1, grid, r, 𝒮) ≈ Π
        @test van_genuchten_saturation(2.0 * abs(Π), 1.4) ≈ 𝒮

        makefield(v1, v2) = (f = Field{Center, Center, Nothing}(grid);
                             set!(f, (x, y) -> x < 1 ? v1 : v2); f)
        ν  = makefield(0.45, 0.35)
        α  = makefield(1.0, 4.0)
        𝓃  = makefield(1.6, 1.2)
        K₀ = makefield(1e-5, 1e-7)
        ηᴷ = makefield(-1.6, -8.9)

        hydrology = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0, porosity = ν, storage_height = 1000,
            retention_curve = VanGenuchtenRetention(; inverse_air_entry_head = α, pore_size_uniformity = 𝓃),
            hydraulic_conductivity = VanGenuchtenConductivity(; matching_point_conductivity = K₀,
                                        pore_size_uniformity = 𝓃, pore_connectivity_exponent = ηᴷ),
            deep_liquid_flux = FreeDrainageFlux(), runoff = NoRunoff())

        land = SlabLand(grid; hydrology)
        set!(land; T = 290.0, M = 150.0)
        fill!(land.fluxes.vapor_flux, 0)
        fill!(land.fluxes.liquid_precipitation_flux, 0)

        for _ in 1:20
            time_step!(land, 3600.0)
        end

        M = column(land.water_storage)
        @test M[1] != M[2]
        @test all(M .< 150.0)
    end
end

@testset "Cosby saturated conductivity" begin
    c = VanGenuchtenConductivity(matching_point_conductivity = 1e-6, pore_size_uniformity = 1.5)
    @test c.pore_connectivity_exponent == 0.5

    K⁺ = [saturated_conductivity(CosbyConductivity(), s) for s in (0.05, 0.25, 0.45, 0.65, 0.85, 0.95)]
    @test issorted(K⁺)
    @test K⁺[end] / K⁺[1] > 15
    @test saturated_conductivity(CosbyConductivity(), 0.92) * 3.6e6 ≈ 84.8 atol=0.5    # mm/hr
    @test conductivity_spread(CosbyConductivity(), 0.05) ≈ 0.475 atol=0.001
    @test conductivity_spread(CosbyConductivity(), 0.65) > conductivity_spread(CosbyConductivity(), 0.05)

    c32 = CosbyConductivity(Float32)
    @test saturated_conductivity(c32, 0.2f0) isa Float32
    @test saturated_conductivity(c32, 0.92f0) ≈ saturated_conductivity(CosbyConductivity(), 0.92) rtol=1e-5

    steeper = CosbyConductivity(sand_coefficient = 0.02)
    @test saturated_conductivity(steeper, 0.92) > saturated_conductivity(CosbyConductivity(), 0.92)
    @test saturated_conductivity(steeper, 0.0) == saturated_conductivity(CosbyConductivity(), 0.0)
end

@testset "capillary disconnect saturation" begin
    @test capillary_disconnect_saturation(1.2) > capillary_disconnect_saturation(1.5) >
          capillary_disconnect_saturation(3.0)
    @test all(0 .< capillary_disconnect_saturation.([1.05, 1.2, 1.5, 3.0, 7.0]) .< 1)

    # Merlin et al. (2016) fitted θ½ = 0.20 + 0.28 clay − 0.16 sand to 34 flux-tower sites;
    # both pedotransfer functions predict the shutoff wetter than that, by a bounded amount.
    θ½ᵒᵇˢ(sand, clay) = 0.20 + 0.28clay - 0.16sand
    for ptf in (WeynantsPedotransfer(), HYPRESPedotransfer())
        errors = Float64[]
        for (sand, silt, clay, ρᵇ) in textures
            p = soil_hydraulic_parameters(ptf, sand, silt, clay, ρᵇ)
            θʳ, ν = p.residual_liquid_fraction, p.porosity
            θ½ = θʳ + (ν - θʳ) * capillary_disconnect_saturation(p.pore_size_uniformity)
            @test θʳ <= θ½ < ν
            push!(errors, θ½ - θ½ᵒᵇˢ(sand, clay))
        end
        @test all(>(0), errors)
        @test sqrt(sum(errors.^2) / length(errors)) < 0.18
    end
end

@testset "viscosity correction on hydraulic conductivity" begin
    v = WaterViscosity()
    Θ(T) = viscosity_correction(v, T)
    @test Θ(v.reference_temperature) == 1
    @test Θ(298.0) > 1 > Θ(275.0)
    # Against tabulated μ: μ(15 °C)/μ(20 °C) = 1.1375/1.0016, and K ∝ 1/μ.
    @test Θ(288.15) / Θ(293.15) ≈ 1.0016 / 1.1375 rtol=5e-3
    @test Θ(293.0) / Θ(283.0) ≈ 1.30 atol=0.02
    @test Θ(310.0) / Θ(275.0) ≈ 2.41 atol=0.05
    @test viscosity_correction(WaterViscosity(Float32), 298.0f0) isa Float32

    warm = WaterViscosity(reference_temperature = 298)
    @test viscosity_correction(warm, 298.0) == 1
    @test viscosity_correction(warm, 288.0) < 1
    @test viscosity_correction(warm, 288.0) ≈ Θ(288.0) / Θ(298.0)

    grid = RectilinearGrid(size = (1, 1, 1), x = (0, 1), y = (0, 1), z = (-1, 0),
                           topology = (Bounded, Bounded, Bounded))
    c = VanGenuchtenConductivity(matching_point_conductivity = 1e-6, pore_size_uniformity = 1.5)
    K₀ = hydraulic_conductivity(1, 1, grid, c, 0.6, 288.0)
    @test hydraulic_conductivity(1, 1, grid, c, 0.6, 298.0) ≈ K₀ * Θ(298.0)

    # The arithmetic follows the state, so a Float64 closure runs in a Float32 kernel.
    grid32 = RectilinearGrid(Float32; size = (1, 1, 1), x = (0, 1), y = (0, 1), z = (-1, 0),
                             topology = (Bounded, Bounded, Bounded))
    @test viscosity_correction(c.water_viscosity, 298.0f0) isa Float32
    @test hydraulic_conductivity(1, 1, grid32, c, 0.6f0, 298.0f0) isa Float32

    isothermal = VanGenuchtenConductivity(matching_point_conductivity = 1e-6,
                                          pore_size_uniformity = 1.5,
                                          water_viscosity = nothing)
    @test hydraulic_conductivity(1, 1, grid, isothermal, 0.6, 310.0) == K₀
end

@testset "conductivity and pressure head stay finite as the soil dries" begin
    K(c, 𝒮, grid) = hydraulic_conductivity(1, 1, grid, c, 𝒮, convert(eltype(grid), 288))
    Π(r, 𝒮, grid) = pressure_head(1, 1, grid, r, 𝒮)

    for FT in (Float64, Float32)
        grid = RectilinearGrid(FT; size = (1, 1, 1), x = (0, 1), y = (0, 1), z = (-1, 0),
                               topology = (Bounded, Bounded, Bounded))
        saturations = FT[0, 1e-12, 1e-6, 1e-3, 0.1, 0.5, 1]

        # Weynants' (𝓃, ηᴷ) pairs, from a sand to a clay.
        for (𝓃, ηᴷ) in ((1.6, -1.6), (1.082, -8.3), (1.2, -8.9))
            c = VanGenuchtenConductivity(FT; matching_point_conductivity = 1e-6,
                                             pore_size_uniformity = 𝓃,
                                             pore_connectivity_exponent = ηᴷ)
            r = VanGenuchtenRetention(FT; inverse_air_entry_head = 0.95, pore_size_uniformity = 𝓃)

            @test K(c, zero(FT), grid) == 0
            @test all(isfinite, [K(c, 𝒮, grid) for 𝒮 in saturations])
            @test issorted([K(c, 𝒮, grid) for 𝒮 in saturations])
            @test K(c, one(FT), grid) ≈ 1e-6

            @test isfinite(Π(r, zero(FT), grid)) && Π(r, zero(FT), grid) < -1e18
            @test all(𝒮 -> isfinite(Π(r, 𝒮, grid)) && Π(r, 𝒮, grid) <= 0, saturations)
            @test issorted([Π(r, 𝒮, grid) for 𝒮 in saturations])
        end

        # Where both are well behaved, the closed forms are reproduced.
        rtol = FT === Float64 ? 1e-12 : 1e-4
        c = VanGenuchtenConductivity(FT; matching_point_conductivity = 1e-6, pore_size_uniformity = 1.5)
        r = VanGenuchtenRetention(FT; inverse_air_entry_head = 2, pore_size_uniformity = 1.4)
        for 𝒮 in FT[0.2, 0.6, 0.95]
            𝓂 = 1 - 1/FT(1.5)
            @test K(c, 𝒮, grid) ≈ 1e-6 * 𝒮^0.5 * (1 - (1 - 𝒮^(1/𝓂))^𝓂)^2 rtol=rtol
            𝓂ᵣ = 1 - 1/FT(1.4)
            @test Π(r, 𝒮, grid) ≈ -(𝒮^(-1/𝓂ᵣ) - 1)^(1/FT(1.4)) / 2 rtol=rtol
        end
    end

    # Against BigFloat: the direct product loses the Mualem bracket to cancellation for a clay 𝓃.
    let 𝓃 = 1.082, ηᴷ = -8.3
        grid = RectilinearGrid(size = (1, 1, 1), x = (0, 1), y = (0, 1), z = (-1, 0),
                               topology = (Bounded, Bounded, Bounded))
        c = VanGenuchtenConductivity(matching_point_conductivity = 1e-6,
                                     pore_size_uniformity = 𝓃, pore_connectivity_exponent = ηᴷ)
        for 𝒮 in (0.3, 0.1, 0.08, 0.065, 0.05)
            𝓂 = 1 - 1/BigFloat(𝓃)
            S = BigFloat(𝒮)
            Kᵉ = BigFloat(1e-6) * S^BigFloat(ηᴷ) * (1 - (1 - S^(1/𝓂))^𝓂)^2
            @test K(c, 𝒮, grid) ≈ Float64(Kᵉ) rtol=1e-12
        end
    end

    # A column evaporated to the storage floor with a Weynants ηᴷ stays finite.
    for arch in test_architectures
        grid = RectilinearGrid(arch; size = (1, 1), x = (0, 1), y = (0, 1),
                               topology = (Bounded, Bounded, Flat))
        hydrology = VariablySaturatedHydrology(eltype(grid);
            slab_depth = 1.0, porosity = 0.45, storage_height = 1000,
            retention_curve = VanGenuchtenRetention(inverse_air_entry_head = 0.95,
                                                    pore_size_uniformity = 1.082),
            hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = 1e-4,
                                                              pore_size_uniformity = 1.082,
                                                              pore_connectivity_exponent = -8.3),
            deep_liquid_flux = DarcyDeepLiquidFlux(exchange_length = 1.0),
            runoff = NoRunoff())

        land = SlabLand(grid; hydrology)
        set!(land; T = 290.0, M = 10.0)
        fill!(land.fluxes.liquid_precipitation_flux, 0)
        fill!(land.fluxes.vapor_flux, 1e-4)
        for _ in 1:50
            time_step!(land, 3600.0)
        end
        @test only(column(land.water_storage)) == 0
        @test isfinite(only(column(land.saturation)))
        @test isfinite(only(column(land.diagnostics.deep_liquid_flux)))
    end
end
