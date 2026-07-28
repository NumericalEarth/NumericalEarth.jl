include("runtests_setup.jl")

using NumericalEarth.DataWrangling: fill_gaps!, fill_seasonal_gaps!, gap_fill_provenance,
    gap_fill_denial, time_average
using Dates: DateTime, Day, Month, Year
using Statistics: mean

# A seasonal shape both the donor and the target share, so a fill that preserves shape is
# exactly checkable.
seasonal_shape(Nt; amplitude = 0.5) =
    Float32[1 + amplitude * sin(2π * (t - 1) / Nt) for t in 1:Nt]

# A uniform field of one class carrying `shape`, ready to have gaps punched in it.
function uniform_series(shape, Nx, Ny)
    Nt = length(shape)
    Λ = zeros(Float32, Nx, Ny, Nt)
    for j in 1:Ny, i in 1:Nx
        Λ[i, j, :] .= shape
    end
    return Λ
end

@testset "Per-column max_gap" begin
    # The scalar path is untouched: same results as before, byte for byte.
    scalar = Float32[1, NaN, NaN, 4, 5]
    array = copy(scalar)
    fill_gaps!(scalar; max_gap = 2)
    fill_gaps!(array; max_gap = 2)
    @test scalar == array

    stacked = fill(NaN32, 2, 1, 5)
    stacked[1, 1, :] .= Float32[1, NaN, NaN, 4, 5]
    stacked[2, 1, :] .= Float32[1, NaN, NaN, 4, 5]
    uniform = copy(stacked)

    fill_gaps!(uniform; max_gap = 2)
    @test_logs (:warn,) fill_gaps!(stacked; max_gap = reshape([2, 1], 2, 1))

    @test stacked[1, 1, :] == uniform[1, 1, :]         # tolerant column: filled
    @test all(isnan, stacked[2, 1, 2:3])               # strict column: left alone

    # A per-column tolerance has to be on the data's own lattice.
    @test_throws ArgumentError fill_gaps!(stacked; max_gap = fill(2, 3, 1))
end

@testset "Seasonal fill: the scaled donor preserves each cell's level" begin
    Nt = 12
    shape = seasonal_shape(Nt)
    Λ = uniform_series(shape, 8, 8)

    # One cell sits at twice its neighbours' level and is missing at one period.
    Λ[3, 3, :] .= 2 .* shape
    Λ[3, 3, 5] = NaN32
    classes = fill(Float32(2), 8, 8)

    filled = fill_seasonal_gaps!(Λ, classes; cyclic = true, maximum_gap = 0,
                                 block_size = 4, minimum_donors = 4)

    # The whole claim of the method: the donor supplies the shape, the cell keeps its level.
    @test Λ[3, 3, 5] ≈ 2 * shape[5] rtol = 1e-6
    @test filled.provenance[3, 3, 5] == gap_fill_provenance.scaled
    @test filled.reach[3, 3] == 1

    # Nothing that was observed moved.
    @test Λ[1, 1, :] == shape
    @test all(t -> t == 5 || Λ[3, 3, t] == 2 * shape[t], 1:Nt)
    @test all(==(gap_fill_provenance.observed), filled.provenance[1, 1, :])
end

@testset "Seasonal fill: the estimator's guards" begin
    Nt = 12
    shape = seasonal_shape(Nt)

    # The ratio of sums, not a mean of ratios: a donor curve that passes near zero at one
    # period would otherwise dominate every other one.
    near_zero = copy(shape)
    near_zero[4] = 1f-6
    Λ = uniform_series(near_zero, 8, 8)
    Λ[3, 3, :] .= 3 .* near_zero
    Λ[3, 3, 7] = NaN32
    classes = fill(Float32(2), 8, 8)

    fill_seasonal_gaps!(Λ, classes; cyclic = true, maximum_gap = 0, block_size = 4,
                        minimum_donors = 4)
    @test Λ[3, 3, 7] ≈ 3 * near_zero[7] rtol = 1e-5

    # A cell is excluded from its own donor pool: with the target left in, a cell at twice
    # its neighbours' level would drag the donor towards itself and the scaling towards one.
    Λ = uniform_series(shape, 4, 4)
    Λ[2, 2, :] .= 5 .* shape
    Λ[2, 2, 6] = NaN32
    fill_seasonal_gaps!(Λ, fill(Float32(2), 4, 4); cyclic = true, maximum_gap = 0,
                        block_size = 2, minimum_donors = 1)
    @test Λ[2, 2, 6] ≈ 5 * shape[6] rtol = 1e-5

    # Too few anchor periods to trust a scaling: fall through to the donor curve itself.
    Λ = uniform_series(shape, 4, 4)
    Λ[2, 2, :] .= 4 .* shape
    Λ[2, 2, 3:Nt] .= NaN32
    filled = fill_seasonal_gaps!(Λ, fill(Float32(2), 4, 4); cyclic = true, maximum_gap = 0,
                                 block_size = 2, minimum_donors = 1,
                                 minimum_anchor_periods = 6)
    @test Λ[2, 2, 7] ≈ shape[7] rtol = 1e-5
    @test filled.provenance[2, 2, 7] == gap_fill_provenance.class_mean

    # A cell with no valid period at all has no level to scale by, and reads the class mean.
    Λ = uniform_series(shape, 4, 4)
    Λ[2, 2, :] .= NaN32
    filled = fill_seasonal_gaps!(Λ, fill(Float32(2), 4, 4); cyclic = true, maximum_gap = 0,
                                 block_size = 2, minimum_donors = 1)
    @test Λ[2, 2, :] ≈ shape rtol = 1e-5
    @test all(==(gap_fill_provenance.class_mean), filled.provenance[2, 2, :])

    # The clamp bounds the estimate at the product's own range.
    Λ = uniform_series(shape, 4, 4)
    Λ[2, 2, :] .= 100 .* shape
    Λ[2, 2, 6] = NaN32
    fill_seasonal_gaps!(Λ, fill(Float32(2), 4, 4); cyclic = true, maximum_gap = 0,
                        block_size = 2, minimum_donors = 1, valid_range = (0, 10))
    @test Λ[2, 2, 6] == 10

    @test_throws ArgumentError fill_seasonal_gaps!(uniform_series(shape, 2, 2),
                                                   fill(Float32(2), 2, 2); scaling = :geometric)
end

@testset "Seasonal fill: donors are pooled by class" begin
    Nt = 12
    evergreen = seasonal_shape(Nt; amplitude = 0.1)
    deciduous = seasonal_shape(Nt; amplitude = 0.9)

    Λ = zeros(Float32, 8, 8, Nt)
    classes = zeros(Float32, 8, 8)
    for j in 1:8, i in 1:8
        woody = i ≤ 4
        Λ[i, j, :] .= woody ? evergreen : deciduous
        classes[i, j] = woody ? igbp_class_names.evergreen_broadleaf_forest :
                                igbp_class_names.deciduous_broadleaf_forest
    end

    # A deciduous cell at the class boundary must not be filled from its evergreen
    # neighbours: they are one cell away and carry the wrong seasonal shape entirely.
    Λ[5, 4, 3] = NaN32
    fill_seasonal_gaps!(Λ, classes; cyclic = true, maximum_gap = 0, block_size = 4,
                        minimum_donors = 4)

    @test Λ[5, 4, 3] ≈ deciduous[3] rtol = 1e-5
    @test !isapprox(Λ[5, 4, 3], evergreen[3]; rtol = 1e-3)
end

@testset "Seasonal fill: the stencil grows until it finds donors" begin
    Nt = 8
    shape = seasonal_shape(Nt; amplitude = 0.4)
    block = 4
    Λ = uniform_series(shape, 32, 32)

    # Period 5 is blank across three rings of blocks around block (2, 2) — synoptic overcast,
    # which is exactly the case a fixed neighbourhood is empty for.
    for j in 1:32, i in 1:32
        bi, bj = (i - 1) ÷ block + 1, (j - 1) ÷ block + 1
        (abs(bi - 2) ≤ 2 && abs(bj - 2) ≤ 2) && (Λ[i, j, 5] = NaN32)
    end

    filled = fill_seasonal_gaps!(Λ, fill(Float32(2), 32, 32); cyclic = true, maximum_gap = 0,
                                 block_size = block, initial_radius = 1, minimum_donors = 4,
                                 maximum_radius = 8)

    @test Λ[5, 5, 5] ≈ shape[5] rtol = 1e-5
    @test filled.reach[5, 5] == 3
    @test count(isnan, Λ) == 0

    # A donor that cannot be reached inside `maximum_radius` leaves the cell alone rather
    # than borrowing from the far side of the region.
    Λ = uniform_series(shape, 32, 32)
    Λ[:, :, 5] .= NaN32
    filled = fill_seasonal_gaps!(Λ, fill(Float32(2), 32, 32); cyclic = true, maximum_gap = 0,
                                 block_size = block, minimum_donors = 4, maximum_radius = 2)
    @test all(isnan, Λ[:, :, 5])
    @test all(==(gap_fill_provenance.unfilled), filled.provenance[:, :, 5])
end

@testset "Seasonal fill: stages, masks and provenance" begin
    Nt = 12
    shape = seasonal_shape(Nt)
    classes = fill(Float32(igbp_class_names.evergreen_broadleaf_forest), 4, 4)
    classes[1, 1] = igbp_class_names.water

    template = uniform_series(shape, 4, 4)
    template[1, 1, :] .= NaN32           # water carries no retrieval
    template[2, 2, 5] = NaN32            # a one-period gap, reachable by interpolation
    template[3, 3, 4:7] .= NaN32         # too long to bridge, reachable only by a donor

    Λ = copy(template)
    filled = fill_seasonal_gaps!(Λ, classes; cyclic = true, maximum_gap = 2, block_size = 2,
                                 minimum_donors = 1,
                                 unfilled_classes = (igbp_class_names.water,))

    @test filled.provenance[2, 2, 5] == gap_fill_provenance.temporal
    @test filled.provenance[3, 3, 5] == gap_fill_provenance.scaled
    @test filled.provenance[2, 2, 1] == gap_fill_provenance.observed

    # A class that is never filled stays missing at every period, and says so.
    @test all(isnan, Λ[1, 1, :])
    @test all(==(gap_fill_provenance.unfilled), filled.provenance[1, 1, :])

    # The chain only ever adds: every observed value is bit-identical afterwards.
    observed = .!isnan.(template)
    @test Λ[observed] == template[observed]

    # Every stage can be run alone, which is what makes the denial experiment per stage.
    Λ = copy(template)
    fill_seasonal_gaps!(Λ, classes; cyclic = true, maximum_gap = 2, block_size = 2,
                        minimum_donors = 1, stages = (:temporal,))
    @test !isnan(Λ[2, 2, 5])
    @test all(isnan, Λ[3, 3, 4:7])

    Λ = copy(template)
    fill_seasonal_gaps!(Λ, classes; cyclic = true, maximum_gap = 2, block_size = 2,
                        minimum_donors = 1, stages = (:scaled,))
    @test !isnan(Λ[2, 2, 5])              # the donor reaches it too, without the bridge
    @test !isnan(Λ[3, 3, 5])

    # A class array on the wrong lattice is an error, not a silent one-cell offset.
    @test_throws ArgumentError fill_seasonal_gaps!(copy(template), fill(Float32(2), 5, 4))
end

@testset "Seasonal fill: the anchored donor" begin
    Nt = 12
    Λ̄ = uniform_series(seasonal_shape(Nt), 4, 4)
    Λ = 1.5f0 .* Λ̄
    Λ[2, 2, 7] = NaN32

    # With a climatology in hand the donor is the cell's own curve, so no class is needed at
    # all: the fill keeps both the cell's level and its own seasonal shape.
    filled = fill_seasonal_gaps!(Λ, fill(NaN32, 4, 4); anchor = Λ̄, maximum_gap = 0)

    @test Λ[2, 2, 7] ≈ 1.5f0 * Λ̄[2, 2, 7] rtol = 1e-6
    @test filled.provenance[2, 2, 7] == gap_fill_provenance.scaled
    @test filled.reach[2, 2] == 0

    # The anchor is scaled, never copied: a target year that departs from the climatology
    # must keep its departure.
    @test !(Λ[2, 2, 7] ≈ Λ̄[2, 2, 7])

    # A shorter anchor is reused cyclically, and an explicit mapping is honoured.
    Λ̄season = uniform_series(seasonal_shape(4), 4, 4)
    Λ = zeros(Float32, 4, 4, 8)
    for t in 1:8
        Λ[:, :, t] .= 2 .* Λ̄season[:, :, mod1(t, 4)]
    end
    Λ[1, 1, 3] = NaN32
    fill_seasonal_gaps!(Λ, fill(NaN32, 4, 4); anchor = Λ̄season, maximum_gap = 0)
    @test Λ[1, 1, 3] ≈ 2 * Λ̄season[1, 1, 3] rtol = 1e-6

    @test_throws ArgumentError fill_seasonal_gaps!(zeros(Float32, 4, 4, 8), fill(NaN32, 4, 4);
                                                   anchor = Λ̄season,
                                                   anchor_periods = [1, 2, 3])
    @test_throws ArgumentError fill_seasonal_gaps!(zeros(Float32, 4, 4, 8), fill(NaN32, 4, 4);
                                                   anchor = uniform_series(seasonal_shape(4), 5, 4))
end

@testset "Data denial" begin
    Nt = 12
    shape = seasonal_shape(Nt)
    classes = fill(Float32(igbp_class_names.evergreen_broadleaf_forest), 16, 16)
    Λ = uniform_series(shape, 16, 16)

    # A field the chain reproduces exactly must score a perfect one, or a harness bug could
    # pass for a good result.
    rows = gap_fill_denial(Λ, classes; samples_per_class = 50, cyclic = true, maximum_gap = 0,
                           block_size = 4, minimum_donors = 4)
    row = only(rows)
    @test row.class == igbp_class_names.evergreen_broadleaf_forest
    @test row.withheld == 50
    @test row.estimated == 50
    @test row.cv_rmse < 1e-4
    @test row.R² ≈ 1 atol = 1e-4

    # It also has to be non-destructive: the series it scores is the caller's.
    @test count(isnan, Λ) == 0

    # A per-class breakdown, and a class that cannot be reconstructed scores worse.
    classes[9:16, :] .= igbp_class_names.deciduous_broadleaf_forest
    for j in 1:16, i in 9:16
        Λ[i, j, :] .= shape .* (1 + 0.1f0 * i)
    end
    rows = gap_fill_denial(Λ, classes; samples_per_class = 20, cyclic = true, maximum_gap = 0,
                           block_size = 4, minimum_donors = 4)
    @test length(rows) == 2
    @test [row.class for row in rows] == sort([igbp_class_names.evergreen_broadleaf_forest,
                                               igbp_class_names.deciduous_broadleaf_forest])
end

@testset "Time averaging" begin
    # Eight 8-day composites of a ramp, averaged onto calendar months. The windows do not
    # nest, so the samples that straddle an edge have to be split by days of overlap.
    dates = [DateTime(2019, 1, 1) + Day(8 * (n - 1)) for n in 1:8]
    bounds = [dates; dates[end] + Day(8)]
    ramp = reshape(Float32.(1:8), 1, 1, 8)

    averaged, edges = time_average(ramp, bounds, Month(1))

    @test edges == [DateTime(2019, 1, 1), DateTime(2019, 2, 1), DateTime(2019, 3, 1),
                    DateTime(2019, 3, 6)]
    @test size(averaged) == (1, 1, 3)

    # January holds composites 1-3 whole and seven of composite 4's eight days.
    @test averaged[1, 1, 1] ≈ (8 * 1 + 8 * 2 + 8 * 3 + 7 * 4) / 31
    # February takes composite 4's last day, 5-7 whole, and three days of 8.
    @test averaged[1, 1, 2] ≈ (1 * 4 + 8 * 5 + 8 * 6 + 8 * 7 + 3 * 8) / 28
    @test averaged[1, 1, 3] ≈ 8

    # An unweighted mean of the same samples is a different number, which is the reason the
    # weighting is not optional.
    @test !isapprox(averaged[1, 1, 1], mean(1:4))

    # NaN samples drop out and the rest are renormalized, rather than poisoning the window.
    gappy = copy(ramp)
    gappy[1, 1, 2] = NaN32
    averaged, _ = time_average(gappy, bounds, Month(1))
    @test averaged[1, 1, 1] ≈ (8 * 1 + 8 * 3 + 7 * 4) / 23

    # A window with nothing valid in it stays missing.
    empty_series = fill(NaN32, 1, 1, 8)
    averaged, _ = time_average(empty_series, bounds, Month(1))
    @test all(isnan, averaged)

    # A window as long as the record returns the record's own weighted mean.
    averaged, edges = time_average(ramp, bounds, Year(1))
    @test length(edges) == 2
    @test averaged[1, 1, 1] ≈ mean(1:8)

    @test_throws ArgumentError time_average(ramp, dates, Month(1))
end
