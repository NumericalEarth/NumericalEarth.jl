include("runtests_setup.jl")

using NumericalEarth.Lands
using NumericalEarth.Lands:
    DragPartitionParameters, DragPartitionRoughness, aerodynamic_parameters,
    canopy_roughness, canopy_wind_ratio,
    zero_plane_displacement, canopy_roughness_length,
    semiempirical_roughness, semiempirical_displacement,
    canopy_drag_parameters, drag_partition_group, is_vegetated,
    class_canopy_height, nonvegetated_roughness,
    compute_aerodynamic_roughness!, canopy_roughness_climatology

using Oceananigans.Fields: interior, set!
using Oceananigans.OutputReaders: FieldTimeSeries
using Statistics: mean

const κ  = 0.4
const ψh = 0.193
const iters = 20

params(g) = canopy_drag_parameters(Float64, g)
dp = DragPartitionRoughness(Float64)   # default config (κ, ψₕ, iters) = (0.4, 0.193, 20)

#####
##### Raupach (1994) / Jasinski (2005) drag-partition closure: per-class oracle + ordering.
#####

@testset "Drag-partition closure" begin
    # Per-class oracle: the full closure at representative growing-season Λ and per-class
    # height must land between the class-averaged (z0i) and semi-empirical (z0e) values of
    # Borak et al. (2025) Table 5, and reproduce the forest ≫ non-forest ordering.
    # (group, height, Λ, z0i, z0e, d0i, d0e)
    oracle = Dict(
        "EBF" => (2, 24.72, 6.0, 1.16, 3.30, 21.18, 16.48),
        "DBF" => (2, 17.43, 5.0, 0.98, 2.32, 14.42, 11.62),
        "ENF" => (1, 16.62, 4.0, 1.36, 2.22, 11.53, 11.08),
        "GRS" => (3,  1.39, 1.5, 0.14, 0.19,  0.97,  0.93),
        "CRP" => (4,  1.32, 3.0, 0.13, 0.18,  0.72,  0.88))

    for (_, (g, h, Λ, z0i, z0e, d0i, d0e)) in oracle
        z0, d0 = canopy_roughness(Λ, h, params(g), κ, ψh, iters)
        @test min(z0i, z0e) - 0.1 ≤ z0 ≤ max(z0i, z0e) + 0.1
        @test min(d0i, d0e) - 1.0 ≤ d0 ≤ max(d0i, d0e) + 1.0
    end

    # Forest roughness/displacement dwarf non-forest.
    z0_forest, d0_forest = canopy_roughness(6.0, 24.72, params(2), κ, ψh, iters)
    z0_crop,   d0_crop   = canopy_roughness(3.0,  1.32, params(4), κ, ψh, iters)
    @test z0_forest > 10z0_crop
    @test d0_forest > 10d0_crop

    # Semi-empirical relations are exact closed forms (d0 = 2h/3, z0 = d0/5).
    @test semiempirical_displacement(24.72) ≈ 2 * 24.72 / 3
    @test semiempirical_roughness(24.72)    ≈ 2 * 24.72 / 3 / 5

    # DragPartitionRoughness exposes the shared `aerodynamic_parameters(closure, cell)`
    # contract: over an EBF (IGBP 2) cell it reproduces the raw scalar closure for group 2.
    cell = (; land_cover = 2, lai = 6.0, canopy_height = 24.72, latitude = 0.0)
    @test aerodynamic_parameters(dp, cell) == canopy_roughness(6.0, 24.72, params(2), κ, ψh, iters)
    @test dp(cell) == aerodynamic_parameters(dp, cell)
end

@testset "LAI dependence: monotone d0, skimming z0" begin
    p = params(4)  # cropland, Λmax = 1.5
    Λs = 0.2:0.2:1.4
    d0s = [zero_plane_displacement(Λ, canopy_wind_ratio(Λ, p, iters), 1.32, p) for Λ in Λs]
    @test issorted(d0s)                                    # d0 monotone increasing below Λmax

    # z0 is non-monotonic: it must fall once Λ exceeds Λmax (the skimming effect).
    z0(Λ) = canopy_roughness(Λ, 1.32, p, κ, ψh, iters)[1]
    @test z0(3.0) < z0(1.0)                                # dense canopy skims → lower z0
    @test z0(3.0) ≈ z0(5.0)                                # capped at Λmax beyond the critical value
end

@testset "Class mapping and non-vegetated constants" begin
    @test drag_partition_group(1, 45)  == 1   # ENF
    @test drag_partition_group(2, 45)  == 2   # EBF
    @test drag_partition_group(10, 45) == 3   # grassland
    @test drag_partition_group(12, 45) == 4   # cropland
    @test drag_partition_group(6, 45)  == 5   # closed shrubland
    @test drag_partition_group(8, 30)  == 2   # woody savanna, non-boreal
    @test drag_partition_group(8, 60)  == 1   # woody savanna, boreal
    for nonveg in (13, 15, 16, 17)
        @test drag_partition_group(nonveg, 0) == 0
        @test !is_vegetated(nonveg)
    end
    @test is_vegetated(4)

    @test nonvegetated_roughness(Float64, 13) == (0.8000, 4.83)   # urban
    @test nonvegetated_roughness(Float64, 15) == (0.0024, 0.012)  # snow/ice
    @test nonvegetated_roughness(Float64, 17) == (0.0010, 0.005)  # water

    @test class_canopy_height(Float64, 12) == 1.32
    @test class_canopy_height(Float64, 2)  == 24.72
    @test class_canopy_height(Float64, 17) == 0        # non-veg: no canopy
end

@testset "Kernel safety: finite everywhere, correct eltype" begin
    for FT in (Float32, Float64)
        p = canopy_drag_parameters(FT, 4)
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

@testset "On-grid builder: class-height path" begin
    grid = grid2()
    Λ, lc = scalarfield(grid), scalarfield(grid)
    z0m, d0 = scalarfield(grid), scalarfield(grid)

    # Uniform cropland (IGBP 12) at Λ = 3, no height property → the builder must match the
    # scalar closure evaluated at the class-average crop height.
    set!(Λ, 3); set!(lc, 12)
    compute_aerodynamic_roughness!(z0m, d0, dp, (; land_cover = lc, lai = Λ), grid)
    z0ref, d0ref = canopy_roughness(3.0, class_canopy_height(Float64, 12), params(4), κ, ψh, iters)
    @test all(≈(z0ref), interior(z0m))
    @test all(≈(d0ref), interior(d0))

    # Omitting the canopy_height property is exactly `canopy_height = nothing`.
    z0n, d0n = scalarfield(grid), scalarfield(grid)
    compute_aerodynamic_roughness!(z0n, d0n, dp, (; land_cover = lc, lai = Λ, canopy_height = nothing), grid)
    @test interior(z0n) == interior(z0m)
    @test interior(d0n) == interior(d0)

    # Water (IGBP 17): prescribed constants regardless of LAI/height.
    set!(lc, 17)
    compute_aerodynamic_roughness!(z0m, d0, dp, (; land_cover = lc, lai = Λ), grid)
    @test all(≈(0.0010), interior(z0m))
    @test all(≈(0.005),  interior(d0))

    # Invalid LAI over a vegetated cell → honest NaN gap.
    set!(lc, 12); set!(Λ, 1e30)
    compute_aerodynamic_roughness!(z0m, d0, dp, (; land_cover = lc, lai = Λ), grid)
    @test all(isnan, interior(z0m))
    @test all(isnan, interior(d0))
end

@testset "Canopy height as a native field input" begin
    grid = grid2()
    Λ, lc = scalarfield(grid), scalarfield(grid)
    hc = scalarfield(grid)
    z0m, d0 = scalarfield(grid), scalarfield(grid)

    # Deciduous broadleaf forest (IGBP 4, class height 17.43), Λ = 5 everywhere.
    set!(Λ, 5); set!(lc, 4)
    hclass = class_canopy_height(Float64, 4)

    # A measured 30 m canopy taller than the class average must raise z0m and d0.
    set!(hc, 30.0)
    compute_aerodynamic_roughness!(z0m, d0, dp, (; land_cover = lc, lai = Λ, canopy_height = hc), grid)
    z0_meas, d0_meas = canopy_roughness(5.0, 30.0, params(2), κ, ψh, iters)
    z0_class, d0_class = canopy_roughness(5.0, hclass, params(2), κ, ψh, iters)
    @test all(≈(z0_meas), interior(z0m))
    @test all(≈(d0_meas), interior(d0))
    @test z0_meas > z0_class && d0_meas > d0_class

    # A scalar height broadcasts identically to a uniform field.
    z0s, d0s = scalarfield(grid), scalarfield(grid)
    compute_aerodynamic_roughness!(z0s, d0s, dp, (; land_cover = lc, lai = Λ, canopy_height = 30.0), grid)
    @test interior(z0s) == interior(z0m)
    @test interior(d0s) == interior(d0)

    # Per-cell fallback: measured height is used where finite & positive; the class-average
    # height fills cells where h_c is NaN (no observation) or 0 (non-forest reading).
    H = interior(hc)
    H[1, 1, 1] = 30.0     # measured → 30 m
    H[2, 1, 1] = 0.0      # zero reading → class height
    H[1, 2, 1] = NaN      # no observation → class height
    H[2, 2, 1] = 10.0     # measured → 10 m
    compute_aerodynamic_roughness!(z0m, d0, dp, (; land_cover = lc, lai = Λ, canopy_height = hc), grid)
    Z = interior(z0m)
    @test Z[1, 1, 1] ≈ canopy_roughness(5.0, 30.0,   params(2), κ, ψh, iters)[1]
    @test Z[2, 1, 1] ≈ canopy_roughness(5.0, hclass, params(2), κ, ψh, iters)[1]
    @test Z[1, 2, 1] ≈ canopy_roughness(5.0, hclass, params(2), κ, ψh, iters)[1]
    @test Z[2, 2, 1] ≈ canopy_roughness(5.0, 10.0,   params(2), κ, ψh, iters)[1]
end

@testset "Climatology driver with static canopy height" begin
    grid = grid2()
    lc = scalarfield(grid)
    hc = scalarfield(grid)
    set!(lc, 4)           # deciduous broadleaf
    set!(hc, 25.0)

    lai = FieldTimeSeries{Center, Center, Nothing}(grid, [0.0, 1.0])
    set!(lai[1], 0.5); set!(lai[2], 5.0)

    z0ts, d0ts = canopy_roughness_climatology(lai, lc, hc)
    @test size(z0ts) == size(lai)
    # Roughness/displacement rise with the summer LAI increase.
    @test mean(interior(z0ts[2])) > mean(interior(z0ts[1]))
    @test mean(interior(d0ts[2])) > mean(interior(d0ts[1]))

    # The static height feeds through: the driver matches the per-slice builder.
    z0check, d0check = scalarfield(grid), scalarfield(grid)
    compute_aerodynamic_roughness!(z0check, d0check, dp, (; land_cover = lc, lai = lai[2], canopy_height = hc), grid)
    @test interior(z0ts[2]) == interior(z0check)
end
