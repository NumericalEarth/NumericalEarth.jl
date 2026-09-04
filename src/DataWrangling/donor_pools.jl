#####
##### Donor pools for the seasonal gap fill: `donor_series!` fills a reference year-curve for a cell
#####

"""
    BlockDonorTable

The neighbor-sourced donor pool: a cell's reference curve is the mean, period by period, of
the same-class cells around it. Valid values bin into per-(block, class, period) sums and
counts over blocks of `block_size`² cells, the block totals accumulate into 2-D prefix sums
so the total over any rectangle of blocks is a four-corner lookup (`window_total`), and the
stencil widens ring by ring until every period holds `minimum_donors` same-class cells:

    ┌───┬───┬───┬───┬───┐
    │ ░ │ ░ │ ░ │ ░ │ ░ │     The stencil opens `initial_radius` blocks (▒) around the
    ├───┼───┼───┼───┼───┤     block holding the target cell ●, and adds a ring (░)
    │ ░ │ ▒ │ ▒ │ ▒ │ ░ │     whenever some period still holds fewer than
    ├───┼───┼───┼───┼───┤     `minimum_donors` same-class cells, up to `maximum_radius`.
    │ ░ │ ▒ │ ● │ ▒ │ ░ │
    ├───┼───┼───┼───┼───┤
    │ ░ │ ▒ │ ▒ │ ▒ │ ░ │
    ├───┼───┼───┼───┼───┤
    │ ░ │ ░ │ ░ │ ░ │ ░ │
    └───┴───┴───┴───┴───┘

Each (block, class) curve is computed once and cached; the target cell's own contribution
is removed when it is read.

`BlockDonorTable(𝒜, class_index, Nclasses; ...)` takes the series and a matrix giving each
cell's class index, `0` marking cells in no class. Both tables carry a leading row and
column of zeros, so block `(bi, bj)` sits at array index `(bi + 1, bj + 1)`.
"""
struct BlockDonorTable{S, C}
    sums :: S
    counts :: C
    class_index :: Matrix{Int}
    block_size :: Int
    initial_radius :: Int
    maximum_radius :: Int
    minimum_donors :: Int
    curves :: Dict{Tuple{Int, Int, Int}, Tuple{Vector{Float64}, Vector{Int}, Int}}
end

function BlockDonorTable(𝒜, class_index, Nclasses;
                         block_size, initial_radius, maximum_radius, minimum_donors)

    Nx, Ny, Nt = size(𝒜)
    Nbx, Nby = cld(Nx, block_size), cld(Ny, block_size)
    Nc = max(Nclasses, 1)

    sums = zeros(Float64, Nbx + 1, Nby + 1, Nc, Nt)
    counts = zeros(Int, Nbx + 1, Nby + 1, Nc, Nt)

    for t in 1:Nt, j in 1:Ny, i in 1:Nx
        c = class_index[i, j]
        c == 0 && continue
        value = 𝒜[i, j, t]
        isnan(value) && continue
        # +1 for the block index, +1 again past the guard row of zeros at index 1.
        bi = (i - 1) ÷ block_size + 2
        bj = (j - 1) ÷ block_size + 2
        sums[bi, bj, c, t] += value
        counts[bi, bj, c, t] += 1
    end

    # In-place 2-D prefix sums over the block indices, per class and period.
    for t in 1:Nt, c in 1:Nc, bj in 2:(Nby + 1), bi in 2:(Nbx + 1)
        sums[bi, bj, c, t] += sums[bi - 1, bj, c, t] + sums[bi, bj - 1, c, t] -
                              sums[bi - 1, bj - 1, c, t]
        counts[bi, bj, c, t] += counts[bi - 1, bj, c, t] + counts[bi, bj - 1, c, t] -
                                counts[bi - 1, bj - 1, c, t]
    end

    curves = Dict{Tuple{Int, Int, Int}, Tuple{Vector{Float64}, Vector{Int}, Int}}()

    return BlockDonorTable(sums, counts, class_index, block_size, initial_radius,
                           maximum_radius, minimum_donors, curves)
end

# Summed-area lookup: `prefix` accumulates over both block indices, so the total over the
# window of blocks i₁:i₂ × j₁:j₂ is a four-corner difference, whatever the window's size —
#
#            j₁-1        j₂
#     i₁-1 ───A───────────B
#             │ ░░░░░░░░░ │        Σ(window) = D − B − C + A
#             │ ░░░░░░░░░ │
#     i₂   ───C───────────D
#
# with each corner at array index (block index + 1), past the guard row of zeros.
@inline window_total(prefix, i₁, i₂, j₁, j₂, c, t) =
    @inbounds prefix[i₂ + 1, j₂ + 1, c, t] - prefix[i₁, j₂ + 1, c, t] -
              prefix[i₂ + 1, j₁, c, t] + prefix[i₁, j₁, c, t]

function window_totals!(sums, counts, table, bi, bj, radius, c)
    Nbx, Nby = size(table.sums, 1) - 1, size(table.sums, 2) - 1
    i₁, i₂ = max(1, bi - radius), min(Nbx, bi + radius)
    j₁, j₂ = max(1, bj - radius), min(Nby, bj + radius)
    for t in eachindex(sums)
        sums[t]   = window_total(table.sums, i₁, i₂, j₁, j₂, c, t)
        counts[t] = window_total(table.counts, i₁, i₂, j₁, j₂, c, t)
    end
    return nothing
end

# The radius grows until every period holds `minimum_donors`.
function block_donor_curve(table, bi, bj, c)
    key = (bi, bj, c)
    haskey(table.curves, key) && return table.curves[key]

    Nt = size(table.sums, 4)
    sums = zeros(Float64, Nt)
    counts = zeros(Int, Nt)
    radius = table.initial_radius

    while true
        window_totals!(sums, counts, table, bi, bj, radius, c)
        (minimum(counts) ≥ table.minimum_donors || radius ≥ table.maximum_radius) && break
        radius += 1
    end

    table.curves[key] = (sums, counts, radius)
    return sums, counts, radius
end

function donor_series!(curve, table::BlockDonorTable, 𝒜, i, j)
    c = table.class_index[i, j]
    if c == 0
        fill!(curve, NaN)
        return 0
    end

    bi = (i - 1) ÷ table.block_size + 1
    bj = (j - 1) ÷ table.block_size + 1
    sums, counts, radius = block_donor_curve(table, bi, bj, c)

    for t in eachindex(curve)
        total, n = sums[t], counts[t]
        # The cell's own value leaves its reference curve.
        value = 𝒜[i, j, t]
        if !isnan(value)
            total -= value
            n -= 1
        end
        curve[t] = n > 0 ? total / n : NaN
    end

    return radius
end

"""
    AnchorDonorPool

The donor pool of a date-dependent fill: each cell's own climatological curve, read at the
period each target time falls in, so nothing is borrowed spatially.

`AnchorDonorPool(anchor, anchor_periods, Nt, spatial)` takes the climatology (a
`FieldTimeSeries` or an array whose last dimension is its periods) on the series' `spatial`
lattice, and the anchor period each of the series' `Nt` times falls in, or `nothing` for
cyclic reuse:

    series time    1   2   3   ⋯   46  47  48   ⋯   92     (two years of 8-day composites)
                   ↓   ↓   ↓        ↓   ↓   ↓         ↓
    anchor period  1   2   3   ⋯   46   1   2   ⋯   46     (the climatology's cycle)

Cyclic reuse pins series time 1 to anchor period 1, so the series must cover whole cycles.
"""
struct AnchorDonorPool{A}
    anchor :: A
    periods :: Vector{Int}
end

function AnchorDonorPool(anchor, anchor_periods, Nt, spatial)
    𝒜̄ = seasonal_array(anchor)
    size(𝒜̄)[1:2] == spatial ||
        throw(ArgumentError("The anchor climatology is $(size(𝒜̄)[1:2]) but the series is $spatial in space."))

    Na = size(𝒜̄, 3)

    if isnothing(anchor_periods)
        mod(Nt, Na) == 0 ||
            throw(ArgumentError("The series has $Nt times and the anchor $Na periods; pass `anchor_periods`."))
    end

    periods = isnothing(anchor_periods) ? [mod1(t, Na) for t in 1:Nt] : collect(anchor_periods)

    length(periods) == Nt ||
        throw(ArgumentError("anchor_periods must give one anchor period per time of the series ($Nt); got $(length(periods))."))
    all(p -> 1 ≤ p ≤ Na, periods) ||
        throw(ArgumentError("anchor_periods must index the anchor's $Na periods."))

    return AnchorDonorPool(𝒜̄, periods)
end

function donor_series!(curve, pool::AnchorDonorPool, 𝒜, i, j)
    for t in eachindex(curve)
        curve[t] = pool.anchor[i, j, pool.periods[t]]
    end
    return 0
end

# The ratio of sums, not a mean of ratios.
function donor_scaling(𝒜, curve, i, j, scaling, minimum_anchor_periods)
    observed_total = 0.0
    donor_total = 0.0
    anchors = 0

    for t in eachindex(curve)
        value, donor = 𝒜[i, j, t], curve[t]
        (isnan(value) || isnan(donor)) && continue
        observed_total += value
        donor_total += donor
        anchors += 1
    end

    anchors ≥ minimum_anchor_periods || return NaN
    scaling === :additive && return (observed_total - donor_total) / anchors
    donor_total > 0 || return NaN
    return observed_total / donor_total
end
