using KernelAbstractions: @kernel, @index

"""
    NearestNeighborInpainting{M}

A structure representing the nearest neighbor inpainting algorithm, where a missing value is
substituted with the average of the surrounding valid values. This process is repeated a maximum
of `maxiter` times or until the field is completely inpainted.
"""
struct NearestNeighborInpainting{M}
    maxiter :: M
end

propagate_horizontally!(field, ::Nothing, substituting_field=deepcopy(field); kw...) = field

"""
    propagate_horizontally!(inpainting, field, mask [, regions=nothing, substituting_field=deepcopy(field)])

Horizontally propagate the values of `field` into the `mask`.
In other words, cells where `mask[i, j, k] == false` are preserved,
and cells where `mask[i, j, k] == true` are painted over.

The first argument `inpainting` is the inpainting algorithm to use in the `_propagate_field!` step.

`regions` is an optional per-cell region-label `Field` (same grid and location as
`field`): a `NaN` gap is then filled only from neighbors sharing its label
(surface-aware inpainting). `regions === nothing` (default) leaves the fill
surface-agnostic — behavior identical to a single-class field.
"""
function propagate_horizontally!(inpainting::NearestNeighborInpainting, field, mask,
                                 regions = nothing,
                                 substituting_field = deepcopy(field))
    grid = field.grid
    arch = architecture(grid)

    launch!(arch, grid, size(field), _nan_mask!, field, mask)
    fill_halo_regions!(field)
    # The donor stencil reads regions[i±1, j±1], which land in halo cells.
    isnothing(regions) || fill_halo_regions!(regions)

    propagate_step!(inpainting, field, mask, regions, substituting_field)

    # Ungated cleanup: a gap with no reachable same-region donor (an isolated island
    # of its class within the window) is filled from the nearest valid cell of any
    # class so the field stays finite. Runs on the residual NaNs only, so the
    # same-region fills already made are preserved.
    if !isnothing(regions) && sum(isnan, field; condition = interior(mask)) > 0
        fill_halo_regions!(field)
        propagate_step!(inpainting, field, mask, nothing, substituting_field)
    end

    launch!(arch, grid, size(field), _fill_nans!, field)
    fill_halo_regions!(field)

    return field
end

# One propagate/substitute sweep repeated until the field is inpainted, `maxiter` is
# reached, or a pass fills nothing (stalled — no donor reachable for the remaining
# gaps, e.g. an all-NaN connected component or a class with no valid donor).
function propagate_step!(inpainting, field, mask, regions, substituting_field)
    grid = field.grid
    arch = architecture(grid)
    iter = 0
    previous_nans = -1

    # Temporary field to avoid a race condition between propagate and substitute.
    parent(substituting_field) .= parent(field)

    while true
        nans = sum(isnan, field; condition = interior(mask))
        (nans == 0 || nans == previous_nans || iter ≥ inpainting.maxiter) && break
        previous_nans = nans

        launch!(arch, grid, size(field), _propagate_field!, substituting_field, inpainting, field, regions)
        launch!(arch, grid, size(field), _substitute_values!, field, substituting_field)
        iter += 1
    end

    return field
end

# Region gating for the nearest-neighbor donor stencil. `regions === nothing` makes
# every gate `true` (surface-agnostic); otherwise a neighbor donates only if it shares
# the query cell's label. Mirrors Oceananigans' `active_weighted_ℑ*` gating
# (Operators/interpolation_operators.jl), with an equality-of-label gate replacing the
# peripheral-node gate.
@inline region_label(::Nothing, i, j, k) = false
@inline region_label(regions, i, j, k) = @inbounds regions[i, j, k]

@inline same_region(::Nothing, i, j, k, rc) = true
@inline same_region(regions, i, j, k, rc) = @inbounds regions[i, j, k] == rc

# Maybe we can remove this propagate field in lieu of a diffusion,
# Still we'll need to do this a couple of steps on the original grid
@kernel function _propagate_field!(substituting_field, ::NearestNeighborInpainting, field, regions)
    i, j, k = @index(Global, NTuple)

    rc = region_label(regions, i, j, k)
    @inbounds begin
        nw = field[i - 1, j, k]
        ns = field[i, j - 1, k]
        ne = field[i + 1, j, k]
        nn = field[i, j + 1, k]
    end

    # A neighbor donates only if it is valid and shares the query cell's region label.
    # (`false * NaN` is Julia's strong zero `0.0`, so masked/gated neighbors drop out.)
    cw = !isnan(nw) & same_region(regions, i - 1, j, k, rc)
    ce = !isnan(ne) & same_region(regions, i + 1, j, k, rc)
    cn = !isnan(nn) & same_region(regions, i, j + 1, k, rc)
    cs = !isnan(ns) & same_region(regions, i, j - 1, k, rc)

    FT = eltype(field)
    donors = cw + ce + cn + cs
    value  = cw * nw + ce * ne + cn * nn + cs * ns

    FT_NaN = convert(FT, NaN)
    @inbounds substituting_field[i, j, k] = ifelse(value == 0, FT_NaN, value / donors)
end

@kernel function _substitute_values!(field, substituting_field)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        needs_inpainting = isnan(field[i, j, k])
        field[i, j, k] = ifelse(needs_inpainting, substituting_field[i, j, k], field[i, j, k])
    end
end

@kernel function _nan_mask!(field, mask)
    i, j, k = @index(Global, NTuple)
    FT_NaN = convert(eltype(field), NaN)
    @inbounds field[i, j, k] = ifelse(mask[i, j, k], FT_NaN, field[i, j, k])
end

@kernel function _fill_nans!(field)
    i, j, k = @index(Global, NTuple)
    @inbounds field[i, j, k] *= !isnan(field[i, j, k])
end

"""
    inpaint_mask!(field, mask; inpainting=NearestNeighborInpainting(Inf), regions=nothing)

Inpaint `field` within `mask`, using values outside `mask`.
In other words, regions where `mask[i, j, k] == 1` is inpainted
and regions where `mask[i, j, k] == 0` are preserved.

Arguments
=========

- `field`: `Field` to be inpainted.

- `mask`: Boolean-valued `Field`, values where
          `mask[i, j, k] == true` are inpainted.

- `inpainting`: The inpainting algorithm to use. The only option is
                `NearestNeighborInpainting(maxiter)`, where an average
                of the valid surrounding values is used `maxiter` times.
                Default: `NearestNeighborInpainting(Inf)`.

- `regions`: Optional per-cell region-label `Field` (same grid and location as
             `field`) making the horizontal fill surface-aware — a gap is filled
             only from neighbors sharing its label. `nothing` (default) keeps the
             fill surface-agnostic. The vertical `continue_downwards!` step is not
             region-gated.
"""
function inpaint_mask!(field, mask; inpainting=NearestNeighborInpainting(Inf), regions=nothing)

    if inpainting isa Int
        inpainting = NearestNeighborInpainting(inpainting)
    end

    if size(field, 3) > 1
        continue_downwards!(field, mask)
    end

    propagate_horizontally!(inpainting, field, mask, regions)

    return field
end

#####
##### Vertical continuation of fields
#####

continue_downwards!(field, ::Nothing) = field

"""
    continue_downwards!(field, mask)

Continue downwards a field with missing values within `mask`.
Cells where `mask[i, k, k] == false` will be preserved.
"""
function continue_downwards!(field, mask)
    arch = architecture(field)
    grid = field.grid
    launch!(arch, grid, :xy, _continue_downwards!, field, grid, mask)
    return field
end

@kernel function _continue_downwards!(field, grid, mask)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)

    for k = Nz-1 : -1 : 1
        @inbounds field[i, j, k] = ifelse(mask[i, j, k], field[i, j, k+1], field[i, j, k])
    end
end

"""
    fill_gaps!(fts::FieldTimeSeries; max_gap=6)
    fill_gaps!(data::AbstractArray; max_gap=6)
    fill_gaps!(data::AbstractVector; max_gap=6)

Fill NaN gaps along the time dimension using linear interpolation. For an
`AbstractArray`, the last dimension is assumed to be time, and each spatial
column is filled independently. For a `FieldTimeSeries`, `interior(fts)` is
copied to the CPU, filled in place, and copied back.

Gaps longer than `max_gap` points are left as NaN with a warning.
"""
function fill_gaps!(fts::FieldTimeSeries; max_gap=6)
    data_cpu = Array(interior(fts))
    fill_gaps!(data_cpu; max_gap)
    copyto!(interior(fts), data_cpu)
    return fts
end

function fill_gaps!(data::AbstractArray; max_gap=6)
    spatial_inds = CartesianIndices(size(data)[1:end-1])
    for I in spatial_inds
        fill_gaps!(view(data, I, :); max_gap)
    end
    return data
end

function fill_gaps!(data::AbstractVector; max_gap=6)
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
                # Edge gap: fill with nearest valid value
                if gap_start == 1 && gap_end < N
                    data[gap_start:gap_end] .= data[gap_end + 1]
                elseif gap_end == N && gap_start > 1
                    data[gap_start:gap_end] .= data[gap_start - 1]
                end
            elseif gap_length > max_gap
                @warn "Large gap of $gap_length hours at indices $gap_start:$gap_end left unfilled"
            else
                # Linear interpolation
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
