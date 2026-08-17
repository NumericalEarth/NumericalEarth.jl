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
##### assume cloud is quasi-random across years at a given period.
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

Three stages run in order, and each only ever writes where the previous left `NaN`s.
Observed values are never rewritten:

1. `:temporal` — [`fill_gaps!`](@ref) along the time axis, with `max_gap` free to be a
   per-cell array.
2. `:scaled` — a donor's seasonal *shape* scaled to the cell's own magnitude. Over the
   periods where the cell was observed and the donor covers, the cell's sum divided by
   the donor's sum gives the scale; each missing period then fills with the donor's value
   at that period multiplied by the scale.
3. `:class_mean` — the donor curve itself, for cells with no valid period to scale by.

The donor curve has two sources, and `anchor` chooses between them. By default it is
the mean cycle of same-class neighbors, pooled over an expanding block stencil. With
`anchor` set to a climatology on the same lattice it is the cell's *own* climatological
curve. This is the better donor when one is available, since it preserves the cell's
individual timing as well as its level.

Keyword arguments
=================

- `anchor`: `nothing`, or a climatology on the same lattice — a `FieldTimeSeries` or an
            array whose last dimension is its periods — making stage 2's donor the cell's
            own curve instead of its neighbors'. Default: `nothing`.
- `anchor_periods`: one integer per time of `series`, the anchor period that time falls in
                    ([`period_index`](@ref) computes it from a date). Default: cyclic reuse,
                    which assumes the series opens the anchor's cycle and covers whole
                    cycles, and is an error otherwise.
- `cyclic`: `true` to wrap stage 1's interpolation across the ends of the time axis, as one
            period of a periodic signal. Default: `true` without an `anchor` (a seasonal
            climatology wraps), `false` with one (a date window does not).
- `max_gap`: how many consecutive missing periods stage 1 may bridge — one `Int` for every
             cell, or an array over the spatial dimensions giving each cell its own
             ([`class_maximum_gap`](@ref) builds one from a class map). Default: `6`.
- `block_size`: cells per side of the donor table's blocks, an `Int`. Default: `32`.
- `initial_radius`, `maximum_radius`: the donor stencil's half-width in blocks, before and
                                      after growing, `Int`s. Defaults: `1` and `16`.
- `minimum_donors`: valid same-class cells every period needs before the stencil stops
                    growing, an `Int`. Default: `20`.
- `minimum_anchor_periods`: observed periods a cell must share with its donor before the
                            scale is trusted, an `Int`; a cell below it falls through to
                            `:class_mean`. Default: `6`.
- `scaling`: `:multiplicative` (the ratio of sums, for a non-negative field) or `:additive`
             (the mean offset, for a field that may be negative). Default: `:multiplicative`.
- `valid_range`: a `(minimum, maximum)` pair every estimate is clamped to, e.g. `(0, 10)`
                 for leaf area index, or `nothing` for no clamp. Default: `nothing`.
- `unfilled_classes`: a collection of class codes never written into;
                      [`igbp_non_vegetated_classes`](@ref) is the IGBP set (urban, snow,
                      barren, water). Default: `()`, so every class fills.
- `stages`: which stages run, any subset of `(:temporal, :scaled, :class_mean)`, so each
            can be scored alone. Default: all three.
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
