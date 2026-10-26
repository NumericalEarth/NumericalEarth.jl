include("runtests_setup.jl")

using NumericalEarth.DataWrangling: fill_gaps!, fill_seasonal_gaps!, gap_fill_provenance,
    gap_fill_denial
using NumericalEarth.DataWrangling.MODISLand: igbp_class_names, zero_non_vegetated!
using Oceananigans.Grids: halo_size
using Oceananigans.OutputReaders: Cyclical, InMemory
using Dates: DateTime, Day, Month
using JLD2: jldopen

# A seasonal shape both the donor and the target share, so a fill that preserves shape is
# exactly checkable.
seasonal_shape(Nt; amplitude = 0.5) =
    Float32[1 + amplitude * sin(2π * (t - 1) / Nt) for t in 1:Nt]

# A uniform field of one class carrying `shape`, ready to have gaps punched in it.
function uniform_series(shape, Nx, Ny)
    Nt = length(shape)
    LAI = zeros(Float32, Nx, Ny, Nt)
    for j in 1:Ny, i in 1:Nx
        LAI[i, j, :] .= shape
    end
    return LAI
end

@testset "Seasonal fill accepts horizontal Fields and arrays [$arch]" for arch in test_architectures
    grid = LatitudeLongitudeGrid(arch; size = (2, 2), longitude = (0, 1), latitude = (0, 1),
                                 topology = (Bounded, Bounded, Flat))
    classes = Field{Center, Center, Nothing}(grid)
    fill!(classes, igbp_class_names.evergreen_broadleaf_forest)
    codes = Array(interior(classes))

    for land_cover in (classes, codes, dropdims(codes; dims=3))
        series = repeat(reshape(Float32[1, NaN, 3], 1, 1, :), 2, 2, 1)
        filled = fill_seasonal_gaps!(series, land_cover; stages = (:temporal,))
        @test series == repeat(reshape(Float32[1, 2, 3], 1, 1, :), 2, 2, 1)
        @test all(==(gap_fill_provenance.temporal), filled.provenance[:, :, 2])
    end
end

@testset "Per-column max_gap" begin
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
    @test_throws ArgumentError fill_gaps!(stacked; max_gap = fill(2, 2, 1, 1))
end

@testset "Seasonal fill: the scaled donor preserves each cell's level" begin
    Nt = 12
    shape = seasonal_shape(Nt)
    LAI = uniform_series(shape, 8, 8)

    # One cell sits at twice its neighbors' level and is missing at one period.
    LAI[3, 3, :] .= 2 .* shape
    LAI[3, 3, 5] = NaN32
    classes = fill(Float32(2), 8, 8)

    filled = fill_seasonal_gaps!(LAI, classes; cyclic = true, max_gap = 0,
                                 block_size = 4, minimum_donors = 4)

    # The whole claim of the method: the donor supplies the shape, the cell keeps its level.
    @test LAI[3, 3, 5] ≈ 2 * shape[5] rtol = 1e-6
    @test filled.provenance[3, 3, 5] == gap_fill_provenance.scaled
    @test filled.reach[3, 3] == 1

    # Nothing that was observed moved.
    @test LAI[1, 1, :] == shape
    @test all(t -> t == 5 || LAI[3, 3, t] == 2 * shape[t], 1:Nt)
    @test all(==(gap_fill_provenance.observed), filled.provenance[1, 1, :])
end

# Overwrite the cached values, keeping the key that names the inputs they came from.
function overwrite_cached_series!(cache, series)
    key, provenance, reach = jldopen(cache, "r") do file
        file["key"], file["provenance"], file["reach"]
    end

    jldopen(cache, "w") do file
        file["key"] = key
        file["series"] = series
        file["provenance"] = provenance
        file["reach"] = reach
    end

    return nothing
end

@testset "Seasonal fill: a cached fill is read back" begin
    Nt = 12
    shape = seasonal_shape(Nt)
    LAI = uniform_series(shape, 8, 8)
    LAI[3, 3, :] .= 2 .* shape
    LAI[3, 3, 5] = NaN32
    classes = fill(Float32(2), 8, 8)

    cache = joinpath(mktempdir(), "fills", "seasonal_fill.jld2")   # a directory not made yet
    parameters = (; cyclic = true, max_gap = 0, block_size = 4, minimum_donors = 4)

    expected = fill_seasonal_gaps!(copy(LAI), classes; parameters...)
    filled = fill_seasonal_gaps!(copy(LAI), classes; cache, parameters...)

    @test isfile(cache)
    @test filled.series == expected.series
    @test filled.provenance == expected.provenance
    @test filled.reach == expected.reach

    # A value no fill would produce: a call that returns it read the file rather than
    # filling again.
    marker = fill(7f0, size(LAI))
    overwrite_cached_series!(cache, marker)
    @test fill_seasonal_gaps!(copy(LAI), classes; cache, parameters...).series == marker

    # Scoring the fill leaves the real series' cache alone.
    gap_fill_denial(LAI, classes; samples_per_class = 4, cache, parameters...)
    @test fill_seasonal_gaps!(copy(LAI), classes; cache, parameters...).series == marker

    # A parameter the fill depends on, and the series itself, are both part of what the
    # cache is keyed on, so neither reads the marker back.
    strict = merge(parameters, (; max_gap = 3))
    @test fill_seasonal_gaps!(copy(LAI), classes; cache, strict...).series != marker

    overwrite_cached_series!(cache, marker)
    perturbed = copy(LAI)
    perturbed[4, 4, 2] = NaN32
    @test fill_seasonal_gaps!(perturbed, classes; cache, parameters...).series != marker

    # A land-cover map holding the same codes in other places is different input.
    moved = copy(classes)
    moved[1, 1], moved[8, 8] = 3, 1
    swapped = copy(classes)
    swapped[1, 1], swapped[8, 8] = 1, 3
    fill_seasonal_gaps!(copy(LAI), moved; cache, parameters...)
    overwrite_cached_series!(cache, marker)
    @test fill_seasonal_gaps!(copy(LAI), swapped; cache, parameters...).series != marker
end

@testset "Seasonal fill: the estimator's guards" begin
    Nt = 12
    shape = seasonal_shape(Nt)

    # The ratio of sums, not a mean of ratios: a donor curve that passes near zero at one
    # period would otherwise dominate every other one.
    near_zero = copy(shape)
    near_zero[4] = 1f-6
    LAI = uniform_series(near_zero, 8, 8)
    LAI[3, 3, :] .= 3 .* near_zero
    LAI[3, 3, 7] = NaN32
    classes = fill(Float32(2), 8, 8)

    fill_seasonal_gaps!(LAI, classes; cyclic = true, max_gap = 0, block_size = 4,
                        minimum_donors = 4)
    @test LAI[3, 3, 7] ≈ 3 * near_zero[7] rtol = 1e-5

    # A cell is excluded from its own donor pool: with the target left in, a cell at twice
    # its neighbors' level would drag the donor toward itself and the scaling toward one.
    LAI = uniform_series(shape, 4, 4)
    LAI[2, 2, :] .= 5 .* shape
    LAI[2, 2, 6] = NaN32
    fill_seasonal_gaps!(LAI, fill(Float32(2), 4, 4); cyclic = true, max_gap = 0,
                        block_size = 2, minimum_donors = 1)
    @test LAI[2, 2, 6] ≈ 5 * shape[6] rtol = 1e-5

    # The scale comes from observed periods only: a bridged period is a linear estimate of a
    # curved cycle, and counting it would pull the level away from the observed one.
    LAI = uniform_series(shape, 8, 8)
    LAI[3, 3, :] .= 2 .* shape
    LAI[3, 3, 2] = NaN32
    LAI[3, 3, 6:9] .= NaN32
    fill_seasonal_gaps!(LAI, fill(Float32(2), 8, 8); cyclic = true, max_gap = 1,
                        block_size = 4, minimum_donors = 4)
    @test LAI[3, 3, 7] ≈ 2 * shape[7] rtol = 1e-6

    # Too few anchor periods to trust a scaling: fall through to the donor curve itself.
    LAI = uniform_series(shape, 4, 4)
    LAI[2, 2, :] .= 4 .* shape
    LAI[2, 2, 3:Nt] .= NaN32
    filled = fill_seasonal_gaps!(LAI, fill(Float32(2), 4, 4); cyclic = true, max_gap = 0,
                                 block_size = 2, minimum_donors = 1,
                                 minimum_anchor_periods = 6)
    @test LAI[2, 2, 7] ≈ shape[7] rtol = 1e-5
    @test filled.provenance[2, 2, 7] == gap_fill_provenance.class_mean

    # A cell with no valid period at all has no level to scale by, and reads the class mean.
    LAI = uniform_series(shape, 4, 4)
    LAI[2, 2, :] .= NaN32
    filled = fill_seasonal_gaps!(LAI, fill(Float32(2), 4, 4); cyclic = true, max_gap = 0,
                                 block_size = 2, minimum_donors = 1)
    @test LAI[2, 2, :] ≈ shape rtol = 1e-5
    @test all(==(gap_fill_provenance.class_mean), filled.provenance[2, 2, :])

    # The clamp bounds the estimate at the product's own range.
    LAI = uniform_series(shape, 4, 4)
    LAI[2, 2, :] .= 100 .* shape
    LAI[2, 2, 6] = NaN32
    fill_seasonal_gaps!(LAI, fill(Float32(2), 4, 4); cyclic = true, max_gap = 0,
                        block_size = 2, minimum_donors = 1, valid_range = (0, 10))
    @test LAI[2, 2, 6] == 10

    @test_throws ArgumentError fill_seasonal_gaps!(uniform_series(shape, 2, 2),
                                                   fill(Float32(2), 2, 2); scaling = :geometric)
end

@testset "Seasonal fill: donors are pooled by class" begin
    Nt = 12
    evergreen = seasonal_shape(Nt; amplitude = 0.1)
    deciduous = seasonal_shape(Nt; amplitude = 0.9)

    LAI = zeros(Float32, 8, 8, Nt)
    classes = zeros(Float32, 8, 8)
    for j in 1:8, i in 1:8
        woody = i ≤ 4
        LAI[i, j, :] .= woody ? evergreen : deciduous
        classes[i, j] = woody ? igbp_class_names.evergreen_broadleaf_forest :
                                igbp_class_names.deciduous_broadleaf_forest
    end

    # A deciduous cell at the class boundary must not be filled from its evergreen
    # neighbors: they are one cell away and carry the wrong seasonal shape entirely.
    LAI[5, 4, 3] = NaN32
    fill_seasonal_gaps!(LAI, classes; cyclic = true, max_gap = 0, block_size = 4,
                        minimum_donors = 4)

    @test LAI[5, 4, 3] ≈ deciduous[3] rtol = 1e-5
    @test !isapprox(LAI[5, 4, 3], evergreen[3]; rtol = 1e-3)
end

@testset "Seasonal fill: the stencil grows until it finds donors" begin
    Nt = 8
    shape = seasonal_shape(Nt; amplitude = 0.4)
    block = 4
    LAI = uniform_series(shape, 32, 32)

    # Period 5 is blank across three rings of blocks around block (2, 2) — synoptic overcast,
    # which is exactly the case a fixed neighborhood is empty for.
    for j in 1:32, i in 1:32
        bi, bj = (i - 1) ÷ block + 1, (j - 1) ÷ block + 1
        (abs(bi - 2) ≤ 2 && abs(bj - 2) ≤ 2) && (LAI[i, j, 5] = NaN32)
    end

    filled = fill_seasonal_gaps!(LAI, fill(Float32(2), 32, 32); cyclic = true, max_gap = 0,
                                 block_size = block, initial_radius = 1, minimum_donors = 4,
                                 maximum_radius = 8)

    @test LAI[5, 5, 5] ≈ shape[5] rtol = 1e-5
    @test filled.reach[5, 5] == 3
    @test count(isnan, LAI) == 0

    # A donor that cannot be reached inside `maximum_radius` leaves the cell alone rather
    # than borrowing from the far side of the region.
    LAI = uniform_series(shape, 32, 32)
    LAI[:, :, 5] .= NaN32
    filled = fill_seasonal_gaps!(LAI, fill(Float32(2), 32, 32); cyclic = true, max_gap = 0,
                                 block_size = block, minimum_donors = 4, maximum_radius = 2)
    @test all(isnan, LAI[:, :, 5])
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

    LAI = copy(template)
    filled = fill_seasonal_gaps!(LAI, classes; cyclic = true, max_gap = 2, block_size = 2,
                                 minimum_donors = 1,
                                 unfilled_classes = (igbp_class_names.water,))

    @test filled.provenance[2, 2, 5] == gap_fill_provenance.temporal
    @test filled.provenance[3, 3, 5] == gap_fill_provenance.scaled
    @test filled.provenance[2, 2, 1] == gap_fill_provenance.observed

    # A class that is never filled stays missing at every period, and says so.
    @test all(isnan, LAI[1, 1, :])
    @test all(==(gap_fill_provenance.unfilled), filled.provenance[1, 1, :])

    # Nor are its gaps reported as left unfilled.
    water = uniform_series(shape, 4, 4)
    water[1, 1, 3:6] .= NaN32
    @test_logs fill_seasonal_gaps!(water, classes; max_gap = 2,
                                   unfilled_classes = (igbp_class_names.water,))
    @test all(isnan, water[1, 1, 3:6])

    # The chain only ever adds: every observed value is bit-identical afterwards.
    observed = .!isnan.(template)
    @test LAI[observed] == template[observed]

    # Every stage can be run alone, which is what makes the denial experiment per stage.
    LAI = copy(template)
    fill_seasonal_gaps!(LAI, classes; cyclic = true, max_gap = 2, block_size = 2,
                        minimum_donors = 1, stages = (:temporal,))
    @test !isnan(LAI[2, 2, 5])
    @test all(isnan, LAI[3, 3, 4:7])

    LAI = copy(template)
    fill_seasonal_gaps!(LAI, classes; cyclic = true, max_gap = 2, block_size = 2,
                        minimum_donors = 1, stages = (:scaled,))
    @test !isnan(LAI[2, 2, 5])              # the donor reaches it too, without the bridge
    @test !isnan(LAI[3, 3, 5])

    # A class array on the wrong lattice is an error, not a silent one-cell offset.
    @test_throws ArgumentError fill_seasonal_gaps!(copy(template), fill(Float32(2), 5, 4))
end

@testset "Seasonal fill: the anchored donor" begin
    Nt = 12
    climatology = uniform_series(seasonal_shape(Nt), 4, 4)
    LAI = 1.5f0 .* climatology
    LAI[2, 2, 7] = NaN32

    # With a climatology in hand the donor is the cell's own curve, so no class is needed at
    # all: the fill keeps both the cell's level and its own seasonal shape.
    filled = fill_seasonal_gaps!(LAI, fill(NaN32, 4, 4); anchor = climatology, max_gap = 0)

    @test LAI[2, 2, 7] ≈ 1.5f0 * climatology[2, 2, 7] rtol = 1e-6
    @test filled.provenance[2, 2, 7] == gap_fill_provenance.scaled
    @test filled.reach[2, 2] == 0

    # The anchor is scaled, never copied: a target year that departs from the climatology
    # must keep its departure.
    @test !(LAI[2, 2, 7] ≈ climatology[2, 2, 7])

    # A gap at the start of the year too long to bridge comes from the anchor's shape, not
    # from the first observed value held back to the first period.
    LAI = 1.5f0 .* climatology
    LAI[3, 3, 1:3] .= NaN32
    filled = fill_seasonal_gaps!(LAI, fill(NaN32, 4, 4); anchor = climatology, max_gap = 1)
    @test LAI[3, 3, 1:3] ≈ 1.5f0 .* climatology[3, 3, 1:3] rtol = 1e-6
    @test all(==(gap_fill_provenance.scaled), filled.provenance[3, 3, 1:3])

    # A shorter anchor is reused cyclically, and an explicit mapping is honored.
    short_climatology = uniform_series(seasonal_shape(4), 4, 4)
    LAI = zeros(Float32, 4, 4, 8)
    for t in 1:8
        LAI[:, :, t] .= 2 .* short_climatology[:, :, mod1(t, 4)]
    end
    LAI[1, 1, 3] = NaN32
    fill_seasonal_gaps!(LAI, fill(NaN32, 4, 4); anchor = short_climatology, max_gap = 0)
    @test LAI[1, 1, 3] ≈ 2 * short_climatology[1, 1, 3] rtol = 1e-6

    @test_throws ArgumentError fill_seasonal_gaps!(zeros(Float32, 4, 4, 8), fill(NaN32, 4, 4);
                                                   anchor = short_climatology,
                                                   anchor_periods = [1, 2, 3])
    @test_throws ArgumentError fill_seasonal_gaps!(zeros(Float32, 4, 4, 8), fill(NaN32, 4, 4);
                                                   anchor = uniform_series(seasonal_shape(4), 5, 4))

    # A series that is not a whole number of cycles cannot rely on the cyclic default: it
    # would silently map its first time onto the anchor's first period.
    @test_throws ArgumentError fill_seasonal_gaps!(zeros(Float32, 4, 4, 6), fill(NaN32, 4, 4);
                                                   anchor = short_climatology)
end

@testset "Seasonal fill: scaling from leaf-off observations" begin
    anchor = reshape(Float32[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 2, 4, 6, 8, 6, 2], 1, 1, :)
    series = 2 .* anchor
    series[:, :, 7:12] .= NaN32

    filled = fill_seasonal_gaps!(series, fill(1, 1, 1); anchor, stages = (:scaled,))
    @test series ≈ 2 .* anchor
    @test all(==(gap_fill_provenance.scaled), filled.provenance[:, :, 7:12])

    anchor[:, :, 1:6] .= 0
    series = 2 .* anchor
    series[:, :, 7:12] .= NaN32
    filled = fill_seasonal_gaps!(series, fill(1, 1, 1); anchor, stages = (:scaled,))
    @test all(isnan, series[:, :, 7:12])
    @test all(==(gap_fill_provenance.unfilled), filled.provenance[:, :, 7:12])
end

@testset "Filled series boundary interpolation [$arch]" for arch in test_architectures
    grid = LatitudeLongitudeGrid(arch; size = (2, 2), longitude = (0, 1), latitude = (0, 1),
                                 topology = (Bounded, Bounded, Flat))
    series = FieldTimeSeries{Center, Center, Nothing}(grid, [0, 1, 2])
    fill!(series[1], 1)
    fill!(series[2], NaN)
    fill!(series[3], 3)

    boundary_grid = LatitudeLongitudeGrid(arch; size = (1, 1),
                                          longitude = (0, 0.2), latitude = (0.4, 0.6),
                                          topology = (Bounded, Bounded, Flat))
    boundary_value = Field{Center, Center, Nothing}(boundary_grid)

    fill_seasonal_gaps!(series, ones(2, 2); stages = (:temporal,))
    interpolate!(boundary_value, series[2])
    @test all(==(2), Array(interior(boundary_value)))

    zero_non_vegetated!(series, fill(igbp_class_names.urban, 2, 2))
    interpolate!(boundary_value, series[2])
    @test all(iszero, Array(interior(boundary_value)))
end

# The host-side chain reduces the whole record at once, so a sliding-window series is
# rejected rather than silently processed as if its resident slice were the record.
@testset "Whole-series guards" begin
    grid = LatitudeLongitudeGrid(CPU(), size = (2, 1, 1), longitude = (0, 1),
                                 latitude = (0, 1), z = (0, 1))
    windowed = FieldTimeSeries{Center, Center, Center}(grid, Float64.(0:7); backend = InMemory(2))

    @test_throws ArgumentError gap_fill_denial(windowed, fill(1f0, 2, 1))
    @test_throws ArgumentError zero_non_vegetated!(windowed, fill(1f0, 2, 1))
end

@testset "Data denial" begin
    Nt = 12
    shape = seasonal_shape(Nt)
    classes = fill(Float32(igbp_class_names.evergreen_broadleaf_forest), 16, 16)
    LAI = uniform_series(shape, 16, 16)

    # A field the chain reproduces exactly must score a perfect one, or a harness bug could
    # pass for a good result.
    rows = gap_fill_denial(LAI, classes; samples_per_class = 50, cyclic = true, max_gap = 0,
                           block_size = 4, minimum_donors = 4)
    row = only(rows)
    @test row.class == igbp_class_names.evergreen_broadleaf_forest
    @test row.withheld == 50
    @test row.estimated == 50
    @test row.relative_root_mean_square_error < 1e-4
    @test row.R² ≈ 1 atol = 1e-4

    # It also has to be non-destructive: the series it scores is the caller's.
    @test count(isnan, LAI) == 0

    # A per-class breakdown, and a class that cannot be reconstructed scores worse.
    classes[9:16, :] .= igbp_class_names.deciduous_broadleaf_forest
    for j in 1:16, i in 9:16
        LAI[i, j, :] .= shape .* (1 + 0.1f0 * i)
    end
    rows = gap_fill_denial(LAI, classes; samples_per_class = 20, cyclic = true, max_gap = 0,
                           block_size = 4, minimum_donors = 4)
    @test length(rows) == 2
    @test [row.class for row in rows] == sort([igbp_class_names.evergreen_broadleaf_forest,
                                               igbp_class_names.deciduous_broadleaf_forest])
end

# Eight 8-day composites of a ramp, averaged onto calendar months. The windows do not nest, so
# the samples that straddle an edge have to be split by days of overlap.
@testset "Time averaging onto calendar months [$arch]" for arch in test_architectures
    dates = [DateTime(2019, 1, 1) + Day(8 * (n - 1)) for n in 1:8]
    bounds = [dates; dates[end] + Day(8)]

    grid = LatitudeLongitudeGrid(arch, size = (2, 1, 4), longitude = (0, 1),
                                 latitude = (0, 1), z = (0, 1))
    ramp = FieldTimeSeries{Center, Center, Center}(grid, Float64.(0:7), indices = (:, :, 4),
                                                   time_indexing = Cyclical())
    for n in 1:8
        interior(ramp[n]) .= n
    end

    truncation = (:warn, "The last window spans 5 days rather than 1 month")
    averaged = @test_logs truncation time_average(ramp, bounds, Month(1))
    values = Array(interior(averaged))

    # January holds composites 1-3 whole and seven of composite 4's eight days.
    @test values[1, 1, 1, 1] ≈ (8 * 1 + 8 * 2 + 8 * 3 + 7 * 4) / 31
    # February takes composite 4's last day, 5-7 whole, and three days of 8.
    @test values[1, 1, 1, 2] ≈ (1 * 4 + 8 * 5 + 8 * 6 + 8 * 7 + 3 * 8) / 28
    # The record ends on 6 March, five days into the last window.
    @test values[1, 1, 1, 3] ≈ 8
    @test Array(averaged.times) ≈ [15.5, 45, 61.5] .* 86400

    @test averaged.indices == ramp.indices
    @test averaged.time_indexing isa Cyclical
    Hx, Hy, _ = halo_size(grid)
    @test Array(parent(averaged[3]))[1 + Hx, Hy, 1] ≈ 8
end
