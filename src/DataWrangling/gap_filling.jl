using Statistics: mean, cor

#####
##### Interpolation along time
#####

"""
    fill_gaps!(fts::FieldTimeSeries; max_gap=6, cyclic=false)
    fill_gaps!(data::AbstractArray; max_gap=6, cyclic=false)
    fill_gaps!(data::AbstractVector; max_gap=6, cyclic=false)

Fill NaN gaps along the time dimension using linear interpolation. For an
`AbstractArray`, the last dimension is assumed to be time, and each spatial
column is filled independently. For a `FieldTimeSeries`, `interior(fts)` is
copied to the CPU, filled in place, and copied back.

Gaps longer than `max_gap` points are left as NaN, and one warning summarizes what was
left behind across the whole call. A gap running to either end of an open series is the
exception: it is extended with its nearest value whatever `max_gap` says, because there
is no second value to interpolate towards. `max_gap` may instead be an array matching the
spatial dimensions of `data`, giving each column its own tolerance.

With `cyclic = true` the series is treated as one period of a periodic signal, so a
gap at either end interpolates across the wrap rather than being extended with its
nearest value — what a seasonal climatology needs, where December's neighbor is
January. An all-NaN column is left untouched either way.
"""
function fill_gaps!(fts::FieldTimeSeries; max_gap=6, cyclic=false)
    validate_whole_series(fts)
    data_cpu = Array(interior(fts))
    fill_gaps!(data_cpu; max_gap, cyclic)
    copyto!(interior(fts), data_cpu)
    return fts
end

# A series read through a sliding window holds only its in-memory slice, so filling it would
# quietly treat that window as the whole record — and, with `cyclic`, as the whole cycle.
function validate_whole_series(fts)
    size(interior(fts))[end] == length(fts.times) ||
        throw(ArgumentError("Filling gaps along time needs the whole series in memory, but " *
                            "this one holds $(size(interior(fts))[end]) of its " *
                            "$(length(fts.times)) times. Rebuild it with " *
                            "`time_indices_in_memory = $(length(fts.times))`."))
    return nothing
end

function fill_gaps!(data::AbstractArray; max_gap=6, cyclic=false)
    validate_maximum_gap(max_gap, size(data)[1:end-1])
    unfilled = Tuple{Int, Int}[]
    for I in CartesianIndices(size(data)[1:end-1])
        fill_column_gaps!(view(data, I, :), unfilled; max_gap=column_maximum_gap(max_gap, I), cyclic)
    end
    warn_unfilled_gaps(unfilled, max_gap)
    return data
end

function fill_gaps!(data::AbstractVector; max_gap=6, cyclic=false)
    unfilled = Tuple{Int, Int}[]
    fill_column_gaps!(data, unfilled; max_gap, cyclic)
    warn_unfilled_gaps(unfilled, max_gap)
    return data
end

@inline column_maximum_gap(max_gap, I) = max_gap
@inline column_maximum_gap(max_gap::AbstractArray, I) =
    @inbounds max_gap[ntuple(d -> I[d], ndims(max_gap))...]

validate_maximum_gap(max_gap, spatial_size) = nothing

function validate_maximum_gap(max_gap::AbstractArray, spatial_size)
    size(max_gap) == spatial_size[1:ndims(max_gap)] ||
        throw(ArgumentError("A per-column max_gap must match the spatial dimensions of the " *
                            "data, $(spatial_size); got $(size(max_gap))."))
    return nothing
end

maximum_gap_summary(max_gap) = string(max_gap)
maximum_gap_summary(max_gap::AbstractArray) = string(minimum(max_gap), "–", maximum(max_gap))

function warn_unfilled_gaps(unfilled, max_gap)
    isempty(unfilled) && return nothing
    longest = maximum(last(gap) - first(gap) + 1 for gap in unfilled)
    @warn "Left $(length(unfilled)) gap(s) of up to $longest points unfilled " *
          "(longer than max_gap = $(maximum_gap_summary(max_gap)))"
    return nothing
end

# Records the gaps it refused to bridge in `unfilled`, so the caller can warn once.
function fill_column_gaps!(data::AbstractVector, unfilled; max_gap, cyclic)
    cyclic && return fill_cyclic_column_gaps!(data, unfilled; max_gap)

    N = length(data)
    i = 1
    while i ≤ N
        if isnan(data[i])
            gap_start = i
            while i ≤ N && isnan(data[i])
                i += 1
            end
            gap_end = i - 1
            gap_length = gap_end - gap_start + 1

            if gap_start == 1 || gap_end == N
                if gap_start == 1 && gap_end < N
                    data[gap_start:gap_end] .= data[gap_end + 1]
                elseif gap_end == N && gap_start > 1
                    data[gap_start:gap_end] .= data[gap_start - 1]
                end
            elseif gap_length > max_gap
                push!(unfilled, (gap_start, gap_end))
            else
                v0 = data[gap_start - 1]
                v1 = data[gap_end + 1]
                for j in gap_start:gap_end
                    α = (j - gap_start + 1) / (gap_length + 1)
                    data[j] = v0 + α * (v1 - v0)
                end
            end
        else
            i += 1
        end
    end
    return data
end

# Rotating the series to start on a valid point turns every wrapped gap into an interior
# one, so a single sweep in the rotated order fills them all with the same interpolation.
function fill_cyclic_column_gaps!(data::AbstractVector, unfilled; max_gap)
    N = length(data)
    origin = findfirst(!isnan, data)
    isnothing(origin) && return data

    rotated(k) = mod1(origin + k - 1, N)

    k = 2
    while k ≤ N
        if isnan(data[rotated(k)])
            gap_start = k
            while k ≤ N && isnan(data[rotated(k)])
                k += 1
            end
            gap_end = k - 1
            gap_length = gap_end - gap_start + 1

            if gap_length > max_gap
                push!(unfilled, (gap_start, gap_end))
            else
                v0 = data[rotated(gap_start - 1)]
                v1 = data[rotated(mod1(gap_end + 1, N))]
                for j in gap_start:gap_end
                    α = (j - gap_start + 1) / (gap_length + 1)
                    data[rotated(j)] = v0 + α * (v1 - v0)
                end
            end
        else
            k += 1
        end
    end
    return data
end

#####
##### Class-aware seasonal gap filling
#####
##### Compositing a period across years and interpolating along the seasonal axis both
##### assume cloud is quasi-random across years at a given period. Where the gap is
##### phase-locked — the ITCZ, a monsoon, a windward slope — neither can see it: what is
##### left needs a donor from elsewhere, and a land-cover class is what makes borrowing
##### safe. Everything here is host-side: a one-time ingestion step, free to allocate and
##### free to branch.
#####

"""
    gap_fill_provenance

How a cell of a series returned by [`fill_seasonal_gaps!`](@ref) came by its value:

| code | name | meaning |
|---|---|---|
| `0x00` | `observed` | the input carried a value here; the chain never rewrites one |
| `0x01` | `temporal` | interpolated along time between this cell's own values |
| `0x02` | `scaled` | a donor's seasonal shape, scaled to this cell's own level |
| `0x03` | `class_mean` | the donor curve itself, for a cell with no level to scale by |
| `0xff` | `unfilled` | still missing |
"""
const gap_fill_provenance = (observed   = 0x00,
                             temporal   = 0x01,
                             scaled     = 0x02,
                             class_mean = 0x03,
                             unfilled   = 0xff)

# The chain is host-side, so a `Field` or `FieldTimeSeries` argument is materialized once.
horizontal_array(codes::AbstractArray) = codes

function horizontal_array(field::Field)
    codes = Array(interior(field))
    size(codes, 3) == 1 ||
        throw(ArgumentError("A land-cover field must be horizontal; this one has " *
                            "$(size(codes, 3)) levels."))
    return codes[:, :, 1]
end

# A `(Nx, Ny, Nt)` view of a series whose last dimension is time, sharing its memory so the
# fill is in place. `interior` of a horizontal `FieldTimeSeries` carries a singleton level.
function seasonal_array(data::AbstractArray)
    spatial = size(data)[1:end-1]
    Nt = size(data)[end]
    (length(spatial) == 2 || (length(spatial) == 3 && spatial[3] == 1)) ||
        throw(ArgumentError("A seasonal series must be horizontal in space; got spatial " *
                            "dimensions $spatial."))
    return reshape(data, spatial[1], spatial[2], Nt)
end

seasonal_array(fts::FieldTimeSeries) = seasonal_array(Array(interior(fts)))

# Compact 1-based indices for the classes actually present, so the donor table's class axis
# is as short as the region's legend rather than as long as the product's.
function class_indices(codes)
    present = sort!(unique(round.(Int, filter(isfinite, vec(codes)))))
    index = zeros(Int, size(codes))
    for I in eachindex(codes)
        isfinite(codes[I]) || continue
        index[I] = searchsortedfirst(present, round(Int, codes[I]))
    end
    return index, present
end

"""
    BlockDonorTable

Per-period sums and counts of the valid values of each class, aggregated over square blocks
of cells and accumulated into 2-D prefix sums, so the total over any block window costs four
lookups however wide the window is. That is what makes an *expanding* stencil affordable:
the search grows its radius until the donor count reaches `minimum_donors`, and growing by
one ring costs a handful of table lookups rather than thousands of cell visits.

The radius that satisfies `minimum_donors` is a property of the block and the class, so
each donor curve is computed once and cached — corrected per cell only by removing that
cell's own contribution, which would otherwise bias its scaling toward one.
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

function block_donor_table(𝒜, class_index, Nclasses;
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
        bi = (i - 1) ÷ block_size + 2
        bj = (j - 1) ÷ block_size + 2
        sums[bi, bj, c, t] += value
        counts[bi, bj, c, t] += 1
    end

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
"""
struct AnchorDonorPool{A}
    anchor :: A
    periods :: Vector{Int}
end

function anchor_donor_pool(anchor, anchor_periods, Nt, spatial)
    𝒜̄ = seasonal_array(anchor)
    size(𝒜̄)[1:2] == spatial ||
        throw(ArgumentError("The anchor climatology is $(size(𝒜̄)[1:2]) but the series is " *
                            "$spatial in space; both must be on the same lattice."))

    Na = size(𝒜̄, 3)

    if isnothing(anchor_periods)
        # Cyclic reuse maps time 1 onto anchor period 1, which is only an alignment when the
        # series opens the cycle and covers whole cycles: a June-to-December window would
        # otherwise borrow January's leaf-off curve for July with no error to show for it.
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

@inline apply_scaling(donor, scale, scaling) =
    scaling === :multiplicative ? scale * donor : donor + scale

@inline clamp_to_range(value, ::Nothing) = value
@inline clamp_to_range(value, range) = clamp(value, first(range), last(range))

"""
    fill_seasonal_gaps!(series, land_cover; kw...)

Fill what compositing and a temporal interpolation could not, using the land-cover class to
decide how far each cell may borrow and from whom. `series` is a `FieldTimeSeries` or an
array whose last dimension is time, `land_cover` a `Field` or array of class codes on the
same lattice.

Returns `(; series, provenance, reach)`: the filled series, a `UInt8` code per cell and
period naming the stage that produced it ([`gap_fill_provenance`](@ref)), and the donor
radius each cell reached, in blocks.

Three stages run in order, and each only ever writes where the previous left `NaN` —
observed values are never rewritten:

1. `:temporal` — [`fill_gaps!`](@ref) along the time axis, with `max_gap` free to be a
   per-cell array.
2. `:scaled` — a donor's seasonal *shape* scaled to the cell's own level,
   `𝒜(i,j,t) = s(i,j) · 𝒜̄(i,j,t)`, with `s` the ratio of the cell's sum to the donor's over
   the periods where both are valid.
3. `:class_mean` — the donor curve itself, for cells with no valid period to scale by.

The donor is same-class neighbors over an expanding block stencil, or — with `anchor` set
to a climatology on the same lattice — the cell's own climatological curve, which preserves
its seasonal shape too. Both go through the same estimator.

Keyword arguments
=================

- `anchor`: a climatology series on the same lattice, making stage 2's donor the cell's own
            curve instead of its neighbors'. Default: `nothing`.
- `anchor_periods`: the anchor index each time of `series` maps to. Defaults to cyclic reuse,
                    which assumes the series opens the anchor's cycle and covers whole cycles,
                    and is an error otherwise.
- `cyclic`: treat the time axis as one period of a periodic signal. Default: no `anchor`.
- `max_gap`: scalar or per-cell bridge length for stage 1. Default: `6`.
- `block_size`: cells per side of the donor table's blocks. Default: `32`.
- `initial_radius`, `maximum_radius`: the donor stencil's half-width in blocks, before and
                                      after growing. Defaults: `1` and `16`.
- `minimum_donors`: valid same-class cells a period needs before the stencil stops growing.
                    Default: `20`.
- `minimum_anchor_periods`: periods a cell needs before its scaling is trusted; below this
                            it falls through to `:class_mean`. Default: `6`.
- `scaling`: `:multiplicative` for a non-negative ratio-scale field, `:additive` for a
             signed one. Default: `:multiplicative`.
- `valid_range`: clamp applied to every estimate, e.g. `(0, 10)` for leaf area index.
                 Default: `nothing`.
- `unfilled_classes`: class codes that are never filled — water, urban, snow, barren.
                      Default: `()`.
- `stages`: which of `(:temporal, :scaled, :class_mean)` to run, so each can be scored
            alone. Default: all three.
"""
function fill_seasonal_gaps!(data::AbstractArray, land_cover;
                             anchor = nothing,
                             anchor_periods = nothing,
                             cyclic = isnothing(anchor),
                             max_gap = 6,
                             block_size = 32,
                             initial_radius = 1,
                             minimum_donors = 20,
                             maximum_radius = 16,
                             minimum_anchor_periods = 6,
                             scaling = :multiplicative,
                             valid_range = nothing,
                             unfilled_classes = (),
                             stages = (:temporal, :scaled, :class_mean))

    scaling in (:multiplicative, :additive) ||
        throw(ArgumentError("scaling must be :multiplicative or :additive; got $scaling."))

    𝒜 = seasonal_array(data)
    Nx, Ny, Nt = size(𝒜)

    codes = horizontal_array(land_cover)
    size(codes) == (Nx, Ny) ||
        throw(ArgumentError("The land-cover array is $(size(codes)) but the series is " *
                            "$((Nx, Ny)) in space. Both must be on the same lattice: a " *
                            "one-cell offset pairs every cell with its neighbor's class, " *
                            "which is worse than an error because the result still looks " *
                            "like a map."))

    class_index, classes_present = class_indices(codes)
    fillable = [!isfinite(code) || !(round(Int, code) in unfilled_classes) for code in codes]

    observed = .!isnan.(𝒜)
    provenance = fill(gap_fill_provenance.unfilled, Nx, Ny, Nt)
    provenance[observed] .= gap_fill_provenance.observed
    reach = zeros(Int, Nx, Ny)

    if :temporal in stages
        fill_gaps!(𝒜; max_gap, cyclic)

        for t in 1:Nt, j in 1:Ny, i in 1:Nx
            observed[i, j, t] && continue
            if !fillable[i, j]
                𝒜[i, j, t] = NaN
            elseif !isnan(𝒜[i, j, t])
                provenance[i, j, t] = gap_fill_provenance.temporal
            end
        end
    end

    scale_donors = :scaled in stages
    mean_donors  = :class_mean in stages
    (scale_donors || mean_donors) || return (; series = data, provenance, reach)

    # Built after the temporal stage, so a neighbor's short bridged gap is available as a donor.
    pool = isnothing(anchor) ?
        block_donor_table(𝒜, class_index, length(classes_present);
                          block_size, initial_radius, maximum_radius, minimum_donors) :
        anchor_donor_pool(anchor, anchor_periods, Nt, (Nx, Ny))

    curve = zeros(Float64, Nt)

    for j in 1:Ny, i in 1:Nx
        fillable[i, j] || continue
        any(t -> isnan(𝒜[i, j, t]), 1:Nt) || continue

        radius = donor_series!(curve, pool, 𝒜, i, j)
        scale = donor_scaling(𝒜, curve, i, j, scaling, minimum_anchor_periods)
        filled = false

        for t in 1:Nt
            isnan(𝒜[i, j, t]) || continue
            donor = curve[t]
            isnan(donor) && continue

            if scale_donors && !isnan(scale)
                𝒜[i, j, t] = clamp_to_range(apply_scaling(donor, scale, scaling), valid_range)
                provenance[i, j, t] = gap_fill_provenance.scaled
            elseif mean_donors
                𝒜[i, j, t] = clamp_to_range(donor, valid_range)
                provenance[i, j, t] = gap_fill_provenance.class_mean
            else
                continue
            end

            filled = true
        end

        filled && (reach[i, j] = radius)
    end

    return (; series = data, provenance, reach)
end

function fill_seasonal_gaps!(fts::FieldTimeSeries, land_cover; kw...)
    validate_whole_series(fts)
    data = Array(interior(fts))
    filled = fill_seasonal_gaps!(data, land_cover; kw...)
    copyto!(interior(fts), data)
    return (; series = fts, filled.provenance, filled.reach)
end

#####
##### Scoring the fill by data denial
#####

# Every `stride`-th eligible entry: spreads the sample over the region and repeats exactly,
# with no seed to report.
function strided_sample(candidates, count)
    length(candidates) ≤ count && return candidates
    stride = length(candidates) ÷ count
    return candidates[1:stride:(1 + stride * (count - 1))]
end

"""
    gap_fill_denial(series, land_cover; samples_per_class = 100, kw...)

Score [`fill_seasonal_gaps!`](@ref) by withholding values it would otherwise have kept:
`samples_per_class` observed cell-periods of each class are set to `NaN`, the chain is run
on the damaged copy, and the estimates are compared with the values that were removed. The
remaining keyword arguments are passed to the fill, so `stages` scores each stage alone.

Returns one row per class: `(; class, withheld, estimated, R², cv_rmse)`, where `R²` is the
squared correlation of estimate against truth, `cv_rmse` the root-mean-square error divided
by the mean of the withheld truth, and `estimated` the number of withheld entries the chain
reached at all.
"""
function gap_fill_denial(series, land_cover; samples_per_class = 100, kw...)
    series isa FieldTimeSeries && validate_whole_series(series)
    𝒜 = seasonal_array(series isa FieldTimeSeries ? series : copy(series))
    codes = horizontal_array(land_cover)
    Nx, Ny, Nt = size(𝒜)

    candidates = Dict{Int, Vector{CartesianIndex{3}}}()
    for t in 1:Nt, j in 1:Ny, i in 1:Nx
        isfinite(codes[i, j]) || continue
        isnan(𝒜[i, j, t]) && continue
        class = round(Int, codes[i, j])
        push!(get!(candidates, class, CartesianIndex{3}[]), CartesianIndex(i, j, t))
    end

    withheld = Dict(class => strided_sample(indices, samples_per_class)
                    for (class, indices) in candidates)
    truth = Dict(class => [𝒜[I] for I in indices] for (class, indices) in withheld)
    for indices in values(withheld), I in indices
        𝒜[I] = NaN
    end

    fill_seasonal_gaps!(𝒜, codes; kw...)

    rows = NamedTuple[]
    for class in sort!(collect(keys(withheld)))
        indices = withheld[class]
        estimate = [𝒜[I] for I in indices]
        reached = isfinite.(estimate)
        observed, modeled = truth[class][reached], estimate[reached]

        if length(observed) < 2
            push!(rows, (; class, withheld = length(indices), estimated = count(reached),
                         R² = NaN, cv_rmse = NaN))
            continue
        end

        rmse = sqrt(mean(abs2, modeled .- observed))

        push!(rows, (; class, withheld = length(indices), estimated = count(reached),
                     R² = cor(modeled, observed)^2, cv_rmse = rmse / mean(observed)))
    end

    return rows
end
