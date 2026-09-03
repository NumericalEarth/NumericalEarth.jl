include("runtests_setup.jl")

using NumericalEarth.Lands
using NumericalEarth.Lands:
    DragPartitionParameters, DragPartitionRoughness, aerodynamic_parameters,
    canopy_roughness, canopy_wind_ratio, zero_plane_displacement,
    semiempirical_roughness, semiempirical_displacement,
    canopy_drag_parameters, drag_partition_group, representative_canopy_height,
    is_vegetated, nonvegetated_roughness,
    compute_aerodynamic_roughness!, canopy_roughness_climatology

using Oceananigans.OutputReaders: FieldTimeSeries
using Statistics: mean

const ϰ = 0.4
const ψ = 0.193
const iters = 20

params(class) = canopy_drag_parameters(Float64, class)
dp = DragPartitionRoughness(Float64)   # default :evergreen_broadleaf_forest, (ϰ, ψ, iters) = (0.4, 0.193, 20)

#####
##### Raupach (1994) / Jasinski (2005) drag-partition closure: per-class oracle + ordering.
#####

@testset "Drag-partition closure" begin
    # Per-class oracle: at a representative growing-season leaf area index and per-class height
    # the full closure reproduces the class-integrated satellite estimates ℓᵐi, di of Borak et al.
    # (2025) Table 5 (their class-averaged product), to within the spread of its
    # seasonal/spatial averaging, and stays at or below the semi-empirical ℓᵐe.
    # (IGBP class, height, leaf area index, ℓᵐi, ℓᵐe, di, de)
    oracle = Dict(
        "EBF" => (:evergreen_broadleaf_forest,  24.72, 6.0, 1.16, 3.30, 21.18, 16.48),
        "DBF" => (:deciduous_broadleaf_forest,  17.43, 5.0, 0.98, 2.32, 14.42, 11.62),
        "ENF" => (:evergreen_needleleaf_forest, 16.62, 4.0, 1.36, 2.22, 11.53, 11.08),
        "GRS" => (:grassland,                    1.39, 1.5, 0.14, 0.19,  0.97,  0.93),
        "CRP" => (:cropland,                     1.32, 3.0, 0.13, 0.18,  0.72,  0.88))

    for (_, (class, h, 𝒜, ℓᵐi, ℓᵐe, di, de)) in oracle
        ℓᵐ, d = canopy_roughness(𝒜, h, params(class), ϰ, ψ, iters)
        @test abs(ℓᵐ - ℓᵐi) ≤ 0.25          # reproduces the class-integrated satellite ℓᵐ
        @test abs(d - di) ≤ 1.0             # reproduces the class-integrated satellite d
        @test ℓᵐ ≤ ℓᵐe + 0.05               # drag partition skims below the height-only rule
    end

    # Forest roughness/displacement dwarf non-forest.
    ℓᵐ_forest, d_forest = canopy_roughness(6.0, 24.72, params(:evergreen_broadleaf_forest), ϰ, ψ, iters)
    ℓᵐ_crop,   d_crop   = canopy_roughness(3.0,  1.32, params(:cropland), ϰ, ψ, iters)
    @test ℓᵐ_forest > 10ℓᵐ_crop
    @test d_forest > 10d_crop

    # Semi-empirical relations are exact closed forms (d = 2h/3, ℓᵐ = d/5).
    @test semiempirical_displacement(24.72) ≈ 2 * 24.72 / 3
    @test semiempirical_roughness(24.72)    ≈ 2 * 24.72 / 3 / 5

    # DragPartitionRoughness exposes the shared `aerodynamic_parameters(closure, cell)`
    # contract: over a cell it reproduces the raw scalar closure for its vegetation class.
    cell = (; leaf_area_index = 6.0, canopy_height = 24.72)
    @test aerodynamic_parameters(dp, cell) == canopy_roughness(6.0, 24.72, params(:evergreen_broadleaf_forest), ϰ, ψ, iters)
    @test dp(cell) == aerodynamic_parameters(dp, cell)

    # A non-default vegetation class selects different parameters.
    grass = DragPartitionRoughness(Float64; vegetation_type = :grassland)
    @test aerodynamic_parameters(grass, (; leaf_area_index = 1.5, canopy_height = 1.39)) ==
          canopy_roughness(1.5, 1.39, params(:grassland), ϰ, ψ, iters)
end

@testset "Leaf-area-index dependence: monotone d, skimming ℓᵐ" begin
    p = params(:cropland)  # 𝒜ᶜ = 1.5
    indices = 0.2:0.2:1.4
    d(𝒜) = zero_plane_displacement(𝒜, canopy_wind_ratio(𝒜, p, iters), 1.32, p)
    @test issorted(d.(indices))                            # d monotone increasing
    # 𝒜ᶜ caps only the wind ratio; displacement keeps rising past it toward the canopy top.
    @test d(5.0) > d(1.5)

    # ℓᵐ is non-monotonic and keeps skimming: it falls as density grows past 𝒜ᶜ.
    ℓᵐ(𝒜) = canopy_roughness(𝒜, 1.32, p, ϰ, ψ, iters)[1]
    @test ℓᵐ(3.0) < ℓᵐ(1.0)                                # dense canopy skims → lower ℓᵐ
    @test ℓᵐ(5.0) < ℓᵐ(3.0)                                # skimming continues past 𝒜ᶜ
end

@testset "IGBP class taxonomy" begin
    # class → Borak drag group
    @test drag_partition_group(:evergreen_needleleaf_forest) == :boreal
    @test drag_partition_group(:evergreen_broadleaf_forest)  == :broadleaf
    @test drag_partition_group(:grassland)                   == :grassland
    @test drag_partition_group(:cropland)                    == :cropland
    @test drag_partition_group(:closed_shrubland)            == :shrubland
    @test drag_partition_group(:woody_savanna)               == :broadleaf   # non-boreal
    @test drag_partition_group(:boreal_woody_savanna)        == :boreal      # boreal split

    # classes sharing a group share drag parameters
    @test canopy_drag_parameters(Float64, :evergreen_broadleaf_forest) ==
          canopy_drag_parameters(Float64, :deciduous_broadleaf_forest)

    # per-class representative heights (Borak Table 4)
    @test representative_canopy_height(Float64, :evergreen_broadleaf_forest) == 24.72
    @test representative_canopy_height(Float64, :deciduous_broadleaf_forest) == 17.43
    @test representative_canopy_height(Float64, :cropland)                   == 1.32
    @test representative_canopy_height(Float32, :grassland)                  isa Float32

    # vegetated vs non-vegetated
    @test is_vegetated(:evergreen_broadleaf_forest)
    for nonveg in (:urban, :snow_and_ice, :barren, :water)
        @test !is_vegetated(nonveg)
    end

    # prescribed non-vegetated roughness (Borak Table 3)
    @test nonvegetated_roughness(Float64, :urban)        == (0.8000, 4.83)
    @test nonvegetated_roughness(Float64, :snow_and_ice) == (0.0024, 0.012)
    @test nonvegetated_roughness(Float64, :water)        == (0.0010, 0.005)
    @test nonvegetated_roughness(Float64, :barren)       == (0.0100, 0.05)
end

@testset "Kernel safety: finite everywhere, correct eltype" begin
    for FT in (Float32, Float64)
        p = canopy_drag_parameters(FT, :cropland)
        for (𝒜, h) in ((FT(0), FT(1.32)), (FT(1e-6), FT(1.32)), (FT(3), FT(0)), (FT(1e3), FT(1.32)))
            ℓᵐ, d = canopy_roughness(𝒜, h, p, FT(0.4), FT(0.193), iters)
            @test isfinite(ℓᵐ) && isfinite(d)
            @test ℓᵐ isa FT && d isa FT
            @test ℓᵐ ≥ 0 && d ≥ 0
        end
    end
end

#####
##### On-grid builder with canopy height as a native field input.
#####

grid2(FT=Float64) = LatitudeLongitudeGrid(CPU(), FT; size = (2, 2),
                                          longitude = (-92, -91), latitude = (37, 38),
                                          topology = (Bounded, Bounded, Flat))
scalarfield(grid) = Field{Center, Center, Nothing}(grid)

@testset "On-grid builder matches the scalar closure" begin
    grid = grid2()
    leaf_area_index, h = scalarfield(grid), scalarfield(grid)
    ℓᵐ, d = scalarfield(grid), scalarfield(grid)

    # Uniform evergreen-broadleaf canopy (𝒜 = 5, 24.72 m) → the builder matches the scalar closure.
    set!(leaf_area_index, 5); set!(h, 24.72)
    compute_aerodynamic_roughness!(ℓᵐ, d, dp, (; leaf_area_index, canopy_height = h), grid)
    ℓᵐref, dref = canopy_roughness(5.0, 24.72, params(:evergreen_broadleaf_forest), ϰ, ψ, iters)
    @test all(≈(ℓᵐref), interior(ℓᵐ))
    @test all(≈(dref), interior(d))

    # A scalar height broadcasts identically to a uniform field.
    ℓᵐs, ds = scalarfield(grid), scalarfield(grid)
    compute_aerodynamic_roughness!(ℓᵐs, ds, dp, (; leaf_area_index, canopy_height = 24.72), grid)
    @test interior(ℓᵐs) == interior(ℓᵐ)
    @test interior(ds) == interior(d)

    # Omitting canopy_height → the class representative height (EBF 24.72) fills every cell;
    # since h was also 24.72, the result matches the explicit-field build.
    ℓᵐf, df = scalarfield(grid), scalarfield(grid)
    compute_aerodynamic_roughness!(ℓᵐf, df, dp, (; leaf_area_index), grid)
    @test interior(ℓᵐf) == interior(ℓᵐ)
    @test interior(df) == interior(d)

    # Invalid leaf area index → honest NaN gap.
    set!(leaf_area_index, 1e30)
    compute_aerodynamic_roughness!(ℓᵐ, d, dp, (; leaf_area_index, canopy_height = h), grid)
    @test all(isnan, interior(ℓᵐ))
    @test all(isnan, interior(d))
end

@testset "Per-cell canopy height and gaps" begin
    grid = grid2()
    leaf_area_index, h = scalarfield(grid), scalarfield(grid)
    ℓᵐ, d = scalarfield(grid), scalarfield(grid)
    set!(leaf_area_index, 5)

    H = interior(h)
    H[1, 1, 1] = 30.0     # tall measured canopy
    H[2, 1, 1] = 0.0      # non-forest reading → ℓᵐ = 0
    H[1, 2, 1] = NaN      # no observation → NaN gap
    H[2, 2, 1] = 10.0     # short measured canopy
    compute_aerodynamic_roughness!(ℓᵐ, d, dp, (; leaf_area_index, canopy_height = h), grid)
    L = interior(ℓᵐ)
    @test L[1, 1, 1] ≈ canopy_roughness(5.0, 30.0, params(:evergreen_broadleaf_forest), ϰ, ψ, iters)[1]
    @test L[2, 1, 1] == 0                                  # h = 0 → ℓᵐ = 0
    @test isnan(L[1, 2, 1])                                # NaN height → gap
    @test L[2, 2, 1] ≈ canopy_roughness(5.0, 10.0, params(:evergreen_broadleaf_forest), ϰ, ψ, iters)[1]
    @test L[1, 1, 1] > L[2, 2, 1]                          # taller canopy → rougher
end

@testset "Climatology driver" begin
    grid = grid2()
    h = scalarfield(grid)
    set!(h, 25.0)

    leaf_area_index = FieldTimeSeries{Center, Center, Nothing}(grid, [0.0, 1.0])
    set!(leaf_area_index[1], 0.5); set!(leaf_area_index[2], 5.0)

    ℓᵐts, dts = canopy_roughness_climatology(leaf_area_index, h)
    @test size(ℓᵐts) == size(leaf_area_index)
    # Displacement rises with the summer leaf-area increase; roughness skims down as the dense
    # summer canopy closes (past the ℓᵐ peak).
    @test mean(interior(dts[2])) > mean(interior(dts[1]))
    @test mean(interior(ℓᵐts[2])) < mean(interior(ℓᵐts[1]))

    # The static height feeds through: the driver matches the per-slice builder.
    ℓᵐcheck, dcheck = scalarfield(grid), scalarfield(grid)
    compute_aerodynamic_roughness!(ℓᵐcheck, dcheck, dp,
                                   (; leaf_area_index = leaf_area_index[2], canopy_height = h), grid)
    @test interior(ℓᵐts[2]) == interior(ℓᵐcheck)

    # Without a canopy height, the climatology uses the class representative height.
    ℓᵐrep, drep = canopy_roughness_climatology(leaf_area_index)
    ℓᵐrepcheck = scalarfield(grid); drepcheck = scalarfield(grid)
    compute_aerodynamic_roughness!(ℓᵐrepcheck, drepcheck, dp,
                                   (; leaf_area_index = leaf_area_index[2]), grid)
    @test interior(ℓᵐrep[2]) == interior(ℓᵐrepcheck)
end

@testset "Tunable leaf-area ceiling, Field and FieldTimeSeries canopy height" begin
    grid = grid2()

    # `maximum_valid_leaf_area_index` is a closure field the user can raise: a finite index above
    # the default ceiling (10) gaps, and lifting the ceiling admits it.
    hi = DragPartitionRoughness(Float64; maximum_valid_leaf_area_index = 15)
    @test all(isnan,    canopy_roughness(dp, 12.0, 24.72))    # 12 > default ceiling → NaN gap
    @test all(isfinite, canopy_roughness(hi, 12.0, 24.72))    # raised ceiling admits it

    # A spatially-varying canopy-height Field flows through the unified `canopy_roughness`.
    leaf_area_index = scalarfield(grid); set!(leaf_area_index, 5)
    h = scalarfield(grid)
    H = interior(h); H[1, 1, 1] = 30.0; H[2, 2, 1] = 10.0
    ℓᵐ, d = canopy_roughness(dp, leaf_area_index, h)
    @test interior(ℓᵐ)[1, 1, 1] ≈ canopy_roughness(5.0, 30.0, params(:evergreen_broadleaf_forest), ϰ, ψ, iters)[1]
    @test interior(ℓᵐ)[2, 2, 1] ≈ canopy_roughness(5.0, 10.0, params(:evergreen_broadleaf_forest), ϰ, ψ, iters)[1]

    # A FieldTimeSeries canopy height is indexed per period alongside the leaf area index.
    𝒜ts = FieldTimeSeries{Center, Center, Nothing}(grid, [0.0, 1.0])
    set!(𝒜ts[1], 5.0); set!(𝒜ts[2], 5.0)
    hts = FieldTimeSeries{Center, Center, Nothing}(grid, [0.0, 1.0])
    set!(hts[1], 20.0); set!(hts[2], 30.0)
    ℓᵐts, dts = canopy_roughness(dp, 𝒜ts, hts)
    @test all(interior(ℓᵐts[1]) .≈ canopy_roughness(5.0, 20.0, params(:evergreen_broadleaf_forest), ϰ, ψ, iters)[1])
    @test all(interior(ℓᵐts[2]) .≈ canopy_roughness(5.0, 30.0, params(:evergreen_broadleaf_forest), ϰ, ψ, iters)[1])
    @test interior(ℓᵐts[2]) != interior(ℓᵐts[1])             # height varies in time → ℓᵐ varies
end
