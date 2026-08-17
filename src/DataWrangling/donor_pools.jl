#####
##### Donor pools for the seasonal gap fill: where a gappy cell borrows its curve from,
##### and the scaling that adapts the borrowed shape to the cell's own level
#####

"""
    BlockDonorTable

Per-period sums and counts of the valid values of each class, aggregated over square blocks
of cells and accumulated into 2-D prefix sums, so the total over any block window costs four
lookups however wide the window is. That is what makes an *expanding* stencil affordable:
the search grows its radius until the donor count reaches `minimum_donors`, and growing by
one ring costs a handful of table lookups rather than thousands of cell visits.

    ┌───┬───┬───┬───┬───┐
    │ ░ │ ░ │ ░ │ ░ │ ░ │     Blocks of `block_size`² cells. The stencil opens
    ├───┼───┼───┼───┼───┤     `initial_radius` blocks (▒) around the block holding the
    │ ░ │ ▒ │ ▒ │ ▒ │ ░ │     target cell ●, and adds a ring (░) whenever some period
    ├───┼───┼───┼───┼───┤     still holds fewer than `minimum_donors` same-class cells,
    │ ░ │ ▒ │ ● │ ▒ │ ░ │     up to `maximum_radius`.
    ├───┼───┼───┼───┼───┤
    │ ░ │ ▒ │ ▒ │ ▒ │ ░ │
    ├───┼───┼───┼───┼───┤
    │ ░ │ ░ │ ░ │ ░ │ ░ │
    └───┴───┴───┴───┴───┘

The radius that satisfies `minimum_donors` is a property of the block and the class, so
each donor curve is computed once and cached — corrected per cell only by removing that
cell's own contribution, which would otherwise bias its scaling toward one.

Built with `BlockDonorTable(𝒜, class_index, Nclasses; ...)` from the series and a matrix
giving each cell's class index (`0` marks cells in no class, which join no pool): one pass
bins every valid value into its (block, class, period) slot, and a second turns the block
totals into prefix sums over the block indices. Both arrays carry a leading row and column
of zeros so `window_total`'s four-corner difference needs no boundary branch, which puts
block `(bi, bj)` at array index `(bi + 1, bj + 1)`.
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

# The radius grows until every period is supported, not only the ones a particular cell is
# missing — otherwise the cached curve would depend on whichever cell asked first.
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
period each target time falls in — preserving the cell's magnitude and seasonal shape, and
borrowing nothing spatially.

Built with `AnchorDonorPool(anchor, anchor_periods, Nt, spatial)`: `anchor` is the
climatology (a `FieldTimeSeries` or an array whose last dimension is its periods), checked
against the series' `spatial` size, and `anchor_periods` gives the anchor period each of
the series' `Nt` times falls in — or `nothing` for cyclic reuse:

    series time    1   2   3   ⋯   46  47  48   ⋯   92     (two years of 8-day composites)
                   ↓   ↓   ↓        ↓   ↓   ↓         ↓
    anchor period  1   2   3   ⋯   46   1   2   ⋯   46     (the climatology's cycle)

Cyclic reuse pins series time 1 to anchor period 1, so it is refused unless the series
covers whole cycles — a series opening mid-year would sit under the wrong periods with no
error to show for it.
"""
struct AnchorDonorPool{A}
    anchor :: A
    periods :: Vector{Int}
end

function AnchorDonorPool(anchor, anchor_periods, Nt, spatial)
    𝒜̄ = seasonal_array(anchor)
    size(𝒜̄)[1:2] == spatial ||
        throw(ArgumentError("The anchor climatology is $(size(𝒜̄)[1:2]) but the series is " *
                            "$spatial in space; both must be on the same lattice."))

    Na = size(𝒜̄, 3)

    if isnothing(anchor_periods)
        mod(Nt, Na) == 0 ||
            throw(ArgumentError("The series has $Nt times and the anchor $Na periods, so their " *
                                "alignment cannot be inferred. Pass `anchor_periods`, the anchor " *
                                "period each time of the series falls in."))
    end

    periods = isnothing(anchor_periods) ? [mod1(t, Na) for t in 1:Nt] : collect(anchor_periods)

    length(periods) == Nt ||
        throw(ArgumentError("anchor_periods must give one anchor index per time of the " *
                            "series ($Nt); got $(length(periods))."))
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

# The ratio of sums, never a mean of ratios: a donor curve that passes near zero at one
# period would otherwise dominate the estimate at every other one.
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
