include("runtests_setup.jl")

using NumericalEarth.Lands
using NumericalEarth.Lands:
    DragPartitionParameters, DragPartitionRoughness, aerodynamic_parameters,
    canopy_roughness, canopy_wind_ratio, zero_plane_displacement,
    semiempirical_roughness, semiempirical_displacement,
    canopy_drag_parameters, drag_partition_group, representative_canopy_height,
    is_vegetated, nonvegetated_roughness,
    compute_aerodynamic_roughness!, canopy_roughness_climatology

using Oceananigans.Fields: interior, set!
using Oceananigans.OutputReaders: FieldTimeSeries
using Statistics: mean

const κ  = 0.4
const ψh = 0.193
const iters = 20

params(class) = canopy_drag_parameters(Float64, class)
dp = DragPartitionRoughness(Float64)   # default :evergreen_broadleaf_forest, (κ, ψₕ, iters) = (0.4, 0.193, 20)

#####
##### Raupach (1994) / Jasinski (2005) drag-partition closure: per-class oracle + ordering.
#####

@testset "Drag-partition closure" begin
    # Per-class oracle: at a representative growing-season Λ and per-class height the full
    # closure reproduces the class-integrated satellite estimates z0i, d0i of Borak et al.
    # (2025) Table 5 (their class-averaged product), to within the spread of its
    # seasonal/spatial averaging, and stays at or below the semi-empirical z0e.
    # (IGBP class, height, Λ, z0i, z0e, d0i, d0e)
    oracle = Dict(
        "EBF" => (:evergreen_broadleaf_forest,  24.72, 6.0, 1.16, 3.30, 21.18, 16.48),
        "DBF" => (:deciduous_broadleaf_forest,  17.43, 5.0, 0.98, 2.32, 14.42, 11.62),
        "ENF" => (:evergreen_needleleaf_forest, 16.62, 4.0, 1.36, 2.22, 11.53, 11.08),
        "GRS" => (:grassland,                    1.39, 1.5, 0.14, 0.19,  0.97,  0.93),
        "CRP" => (:cropland,                     1.32, 3.0, 0.13, 0.18,  0.72,  0.88))

    for (_, (class, h, Λ, z0i, z0e, d0i, d0e)) in oracle
        z0, d0 = canopy_roughness(Λ, h, params(class), κ, ψh, iters)
        @test abs(z0 - z0i) ≤ 0.25          # reproduces the class-integrated satellite z0
        @test abs(d0 - d0i) ≤ 1.0           # reproduces the class-integrated satellite d0
        @test z0 ≤ z0e + 0.05               # drag partition skims below the height-only rule
    end

    # Forest roughness/displacement dwarf non-forest.
    z0_forest, d0_forest = canopy_roughness(6.0, 24.72, params(:evergreen_broadleaf_forest), κ, ψh, iters)
    z0_crop,   d0_crop   = canopy_roughness(3.0,  1.32, params(:cropland), κ, ψh, iters)
    @test z0_forest > 10z0_crop
    @test d0_forest > 10d0_crop

    # Semi-empirical relations are exact closed forms (d0 = 2h/3, z0 = d0/5).
    @test semiempirical_displacement(24.72) ≈ 2 * 24.72 / 3
    @test semiempirical_roughness(24.72)    ≈ 2 * 24.72 / 3 / 5

    # DragPartitionRoughness exposes the shared `aerodynamic_parameters(closure, cell)`
    # contract: over a cell it reproduces the raw scalar closure for its vegetation class.
    cell = (; lai = 6.0, canopy_height = 24.72)
    @test aerodynamic_parameters(dp, cell) == canopy_roughness(6.0, 24.72, params(:evergreen_broadleaf_forest), κ, ψh, iters)
    @test dp(cell) == aerodynamic_parameters(dp, cell)

    # A non-default vegetation class selects different parameters.
    grass = DragPartitionRoughness(Float64; vegetation_type = :grassland)
    @test aerodynamic_parameters(grass, (; lai = 1.5, canopy_height = 1.39)) ==
          canopy_roughness(1.5, 1.39, params(:grassland), κ, ψh, iters)
end

@testset "LAI dependence: monotone d0, skimming z0" begin
    p = params(:cropland)  # Λmax = 1.5
    Λs = 0.2:0.2:1.4
    d0(Λ) = zero_plane_displacement(Λ, canopy_wind_ratio(Λ, p, iters), 1.32, p)
    @test issorted(d0.(Λs))                                # d0 monotone increasing
    # Λmax caps only the wind ratio; displacement keeps rising past it toward the canopy top.
    @test d0(5.0) > d0(1.5)

    # z0 is non-monotonic and keeps skimming: it falls as density grows past Λmax.
    z0(Λ) = canopy_roughness(Λ, 1.32, p, κ, ψh, iters)[1]
    @test z0(3.0) < z0(1.0)                                # dense canopy skims → lower z0
    @test z0(5.0) < z0(3.0)                                # skimming continues past Λmax
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
        for (Λ, h) in ((FT(0), FT(1.32)), (FT(1e-6), FT(1.32)), (FT(3), FT(0)), (FT(1e3), FT(1.32)))
            z0, d0 = canopy_roughness(Λ, h, p, FT(0.4), FT(0.193), iters)
            @test isfinite(z0) && isfinite(d0)
            @test z0 isa FT && d0 isa FT
            @test z0 ≥ 0 && d0 ≥ 0
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
    Λ, hc = scalarfield(grid), scalarfield(grid)
    z0m, d0 = scalarfield(grid), scalarfield(grid)

    # Uniform evergreen-broadleaf canopy (Λ = 5, 24.72 m) → the builder matches the scalar closure.
    set!(Λ, 5); set!(hc, 24.72)
    compute_aerodynamic_roughness!(z0m, d0, dp, (; lai = Λ, canopy_height = hc), grid)
    z0ref, d0ref = canopy_roughness(5.0, 24.72, params(:evergreen_broadleaf_forest), κ, ψh, iters)
    @test all(≈(z0ref), interior(z0m))
    @test all(≈(d0ref), interior(d0))

    # A scalar height broadcasts identically to a uniform field.
    z0s, d0s = scalarfield(grid), scalarfield(grid)
    compute_aerodynamic_roughness!(z0s, d0s, dp, (; lai = Λ, canopy_height = 24.72), grid)
    @test interior(z0s) == interior(z0m)
    @test interior(d0s) == interior(d0)

    # Omitting canopy_height → the class representative height (EBF 24.72) fills every cell;
    # since hc was also 24.72, the result matches the explicit-field build.
    z0f, d0f = scalarfield(grid), scalarfield(grid)
    compute_aerodynamic_roughness!(z0f, d0f, dp, (; lai = Λ), grid)
    @test interior(z0f) == interior(z0m)
    @test interior(d0f) == interior(d0)

    # Invalid LAI → honest NaN gap.
    set!(Λ, 1e30)
    compute_aerodynamic_roughness!(z0m, d0, dp, (; lai = Λ, canopy_height = hc), grid)
    @test all(isnan, interior(z0m))
    @test all(isnan, interior(d0))
end

@testset "Per-cell canopy height and gaps" begin
    grid = grid2()
    Λ, hc = scalarfield(grid), scalarfield(grid)
    z0m, d0 = scalarfield(grid), scalarfield(grid)
    set!(Λ, 5)

    H = interior(hc)
    H[1, 1, 1] = 30.0     # tall measured canopy
    H[2, 1, 1] = 0.0      # non-forest reading → z0 = 0
    H[1, 2, 1] = NaN      # no observation → NaN gap
    H[2, 2, 1] = 10.0     # short measured canopy
    compute_aerodynamic_roughness!(z0m, d0, dp, (; lai = Λ, canopy_height = hc), grid)
    Z = interior(z0m)
    @test Z[1, 1, 1] ≈ canopy_roughness(5.0, 30.0, params(:evergreen_broadleaf_forest), κ, ψh, iters)[1]
    @test Z[2, 1, 1] == 0                                  # h = 0 → z0 = 0
    @test isnan(Z[1, 2, 1])                                # NaN height → gap
    @test Z[2, 2, 1] ≈ canopy_roughness(5.0, 10.0, params(:evergreen_broadleaf_forest), κ, ψh, iters)[1]
    @test Z[1, 1, 1] > Z[2, 2, 1]                          # taller canopy → rougher
end

@testset "Climatology driver" begin
    grid = grid2()
    hc = scalarfield(grid)
    set!(hc, 25.0)

    lai = FieldTimeSeries{Center, Center, Nothing}(grid, [0.0, 1.0])
    set!(lai[1], 0.5); set!(lai[2], 5.0)

    z0ts, d0ts = canopy_roughness_climatology(lai, hc)
    @test size(z0ts) == size(lai)
    # Displacement rises with the summer LAI increase; roughness skims down as the dense
    # summer canopy closes (past the z0 peak).
    @test mean(interior(d0ts[2])) > mean(interior(d0ts[1]))
    @test mean(interior(z0ts[2])) < mean(interior(z0ts[1]))

    # The static height feeds through: the driver matches the per-slice builder.
    z0check, d0check = scalarfield(grid), scalarfield(grid)
    compute_aerodynamic_roughness!(z0check, d0check, dp, (; lai = lai[2], canopy_height = hc), grid)
    @test interior(z0ts[2]) == interior(z0check)

    # Without a canopy height, the climatology uses the class representative height.
    z0rep, d0rep = canopy_roughness_climatology(lai)
    z0repcheck = scalarfield(grid); d0repcheck = scalarfield(grid)
    compute_aerodynamic_roughness!(z0repcheck, d0repcheck, dp, (; lai = lai[2]), grid)
    @test interior(z0rep[2]) == interior(z0repcheck)
end

@testset "Tunable LAI ceiling, Field and FieldTimeSeries canopy height" begin
    grid = grid2()

    # `maximum_valid_area_index` is a closure field the user can raise: a finite LAI above the
    # default ceiling (10) gaps, and lifting the ceiling admits it.
    hi = DragPartitionRoughness(Float64; maximum_valid_area_index = 15)
    @test all(isnan,    canopy_roughness(dp, 12.0, 24.72))    # 12 > default ceiling → NaN gap
    @test all(isfinite, canopy_roughness(hi, 12.0, 24.72))    # raised ceiling admits it

    # A spatially-varying canopy-height Field flows through the unified `canopy_roughness`.
    lai = scalarfield(grid); set!(lai, 5)
    hc  = scalarfield(grid)
    H = interior(hc); H[1, 1, 1] = 30.0; H[2, 2, 1] = 10.0
    z0, d0 = canopy_roughness(dp, lai, hc)
    @test interior(z0)[1, 1, 1] ≈ canopy_roughness(5.0, 30.0, params(:evergreen_broadleaf_forest), κ, ψh, iters)[1]
    @test interior(z0)[2, 2, 1] ≈ canopy_roughness(5.0, 10.0, params(:evergreen_broadleaf_forest), κ, ψh, iters)[1]

    # A FieldTimeSeries canopy height is indexed per period alongside `lai`.
    laits = FieldTimeSeries{Center, Center, Nothing}(grid, [0.0, 1.0])
    set!(laits[1], 5.0); set!(laits[2], 5.0)
    hcts = FieldTimeSeries{Center, Center, Nothing}(grid, [0.0, 1.0])
    set!(hcts[1], 20.0); set!(hcts[2], 30.0)
    z0ts, d0ts = canopy_roughness(dp, laits, hcts)
    @test all(interior(z0ts[1]) .≈ canopy_roughness(5.0, 20.0, params(:evergreen_broadleaf_forest), κ, ψh, iters)[1])
    @test all(interior(z0ts[2]) .≈ canopy_roughness(5.0, 30.0, params(:evergreen_broadleaf_forest), κ, ψh, iters)[1])
    @test interior(z0ts[2]) != interior(z0ts[1])             # height varies in time → z0 varies
end
