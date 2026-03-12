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

"""
    DiffusiveInpainting{M, T}

Diffusive inpainting algorithm. Vertical levels are processed top-to-bottom sequentially:
at each level, missing values are first filled by horizontal neighbor propagation,
then smoothed by iteratively solving the discrete Laplace equation ``∇²ϕ = 0``.
Points that cannot be reached by horizontal propagation fall back to the value
from the level above (the "previous guess").

This produces smooth extrapolated values that preserve horizontal structure at each
depth, minimizing artificial gradients near coastlines.
"""
struct DiffusiveInpainting{M, T, R}
    maxiter                :: M
    tolerance              :: T
    relaxation_coefficient :: R
end

"""
    DiffusiveInpainting(; maxiter=10000, tolerance=1e-6, relaxation_coefficient=0.25)

Construct a `DiffusiveInpainting` algorithm.

Keyword Arguments
=================

- `maxiter`: Maximum number of Laplacian relaxation sweeps per level. Default: `10000`.
- `tolerance`: Convergence criterion — relaxation stops when the maximum pointwise hange falls below this value. Default: `1e-6`.
- `relaxation_coefficient`: Damping factor for the Laplacian smoothing step. Default: `0.25`.
"""
DiffusiveInpainting(; maxiter=10000, tolerance=1e-6, relaxation_coefficient=0.25) =
    DiffusiveInpainting(maxiter, tolerance, relaxation_coefficient)

propagate_horizontally!(field, ::Nothing, substituting_field=deepcopy(field); kw...) = field

function propagating(field, mask, iter, inpainting::NearestNeighborInpainting)
    nans = sum(isnan, field; condition=interior(mask))
    return nans > 0 && iter < inpainting.maxiter
end

"""
    propagate_horizontally!(inpainting, field, mask [, substituting_field=deepcopy(field)])

Horizontally propagate the values of `field` into the `mask`.
In other words, cells where `mask[i, j, k] == false` are preserved,
and cells where `mask[i, j, k] == true` are painted over.

The first argument `inpainting` is the inpainting algorithm to use in the `_propagate_field!` step.
"""
function propagate_horizontally!(inpainting::NearestNeighborInpainting, field, mask,
                                 substituting_field=deepcopy(field))
    iter  = 0
    grid  = field.grid
    arch  = architecture(grid)

    launch!(arch, grid, size(field), _nan_mask!, field, mask)
    fill_halo_regions!(field)

    # Need temporary field to avoid a race condition
    parent(substituting_field) .= parent(field)

    while propagating(field, mask, iter, inpainting)
        launch!(arch, grid, size(field), _propagate_field!,  substituting_field, field)
        launch!(arch, grid, size(field), _substitute_field!, field, substituting_field)

        @debug begin
            nans = sum(isnan, field; condition=interior(mask))
            "Propagate pass: $iter, remaining NaNs: $nans"
        end

        iter += 1
    end

    launch!(arch, grid, size(field), _fill_nans!, field)
    fill_halo_regions!(field)

    return field
end

@inline function neighbor_average(field, i, j, k)
    @inbounds begin
        nw = field[i - 1, j, k]
        ns = field[i, j - 1, k]
        ne = field[i + 1, j, k]
        nn = field[i, j + 1, k]
    end

    FT = eltype(field)
    donors = 0
    value = zero(FT)

    for n in (nw, ne, nn, ns)
        donors += !isnan(n)
        value += !isnan(n) * n
    end

    FT_NaN = convert(FT, NaN)
    return ifelse(value == 0, FT_NaN, value / donors)
end

@kernel function _propagate_field!(substituting_field, field)
    i, j, k = @index(Global, NTuple)
    @inbounds substituting_field[i, j, k] = neighbor_average(field, i, j, k)
end

@kernel function _substitute_field!(field, substituting_field)
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

# At each level, after horizontal filling, any remaining NaN is
# replaced by the previous level's value.
@kernel function _vertical_fallback!(field, mask, k)
    i, j = @index(Global, NTuple)
    @inbounds begin
        is_masked  = mask[i, j, k]
        is_nan     = isnan(field[i, j, k])
        needs_fill = is_masked & is_nan
        field[i, j, k] = ifelse(needs_fill, field[i, j, k + 1], field[i, j, k])
    end
end

# Zero out remaining NaN fill points and mark all fill points as good
@kernel function _finalize_stalled_fill!(field, good, mask)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        is_fill   = mask[i, j, k]
        still_nan = is_fill & isnan(field[i, j, k])
        field[i, j, k] = ifelse(still_nan, zero(field[i, j, k]), field[i, j, k])
        good[i, j, k]  = ifelse(is_fill, one(good[i, j, k]), good[i, j, k])
    end
end

# Laplacian relaxation: computes change = relc * (∑neighbors - n_neighbors * center) on fill points.
# Then applies change in-place: field += change.
# `mask` is the original fill mask (true = fill point).
# `good` is the good mask (true = valid data point).
@kernel function _laplacian_relaxation!(field, change, mask, good, relc)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        is_fill = mask[i, j, k]

        # Neighbor validity: max(good, fill) — a neighbor counts if it's
        # either original data or a fill point
        ge = max(good[i + 1, j, k], mask[i + 1, j, k])
        gw = max(good[i - 1, j, k], mask[i - 1, j, k])
        gn = max(good[i, j + 1, k], mask[i, j + 1, k])
        gs = max(good[i, j - 1, k], mask[i, j - 1, k])

        ϕc = field[i, j, k]
        Δϕ = relc * ((gs * field[i, j - 1, k] + gn * field[i, j + 1, k]) +
                     (gw * field[i - 1, j, k] + ge * field[i + 1, j, k]) -
                     ((gs + gn) + (gw + ge)) * ϕc)

        change[i, j, k] = ifelse(is_fill, Δϕ, zero(Δϕ))
        field[i, j, k]  = field[i, j, k] + ifelse(is_fill, Δϕ, zero(Δϕ))
    end
end

"""
    inpaint_mask!(field, mask; inpainting=DiffusiveInpainting())

Inpaint `field` within `mask`, using values outside `mask`.
In other words, regions where `mask[i, j, k] == 1` is inpainted
and regions where `mask[i, j, k] == 0` are preserved.

Arguments
=========

- `field`: `Field` to be inpainted.

- `mask`: Boolean-valued `Field`, values where
          `mask[i, j, k] == true` are inpainted.

- `inpainting`: The inpainting algorithm to use. Options:
    * `NearestNeighborInpainting(maxiter)`: fills missing values with the
       average of surrounding valid neighbors, repeated `maxiter` times.
       For 3D fields, values are first continued downward before horizontal propagation.
    * `DiffusiveInpainting(; maxiter, tolerance)`: processes levels top-to-bottom. 
        At each level: 
        (1) horizontal nearest-neighbor fill, 
        (2) fallback to the level above for unreachable points, 
        (3) Laplacian smoothing.
    Default: `NearestNeighborInpainting(Inf)`.
"""
function inpaint_mask!(field, mask; inpainting=DiffusiveInpainting())

    if inpainting isa Int
        inpainting = NearestNeighborInpainting(inpainting)
    end

    if inpainting isa DiffusiveInpainting
        diffusive_inpaint_mask!(field, mask, inpainting)
    else
        if size(field, 3) > 1
            continue_downwards!(field, mask)
        end
        propagate_horizontally!(inpainting, field, mask)
    end

    return field
end

# Diffusive inpainting in three phases, processing all levels simultaneously:
#   Phase 1: Horizontal nearest-neighbor fill (all levels at once)
#   Phase 2: Vertical fallback top-to-bottom for stalled points
#   Phase 3: Laplacian relaxation (all levels at once)
function diffusive_inpaint_mask!(field, mask, inpainting)
    grid = field.grid
    arch = architecture(grid)
    Nz = size(grid, 3)

    # NaN-mask all fill points
    launch!(arch, grid, size(field), _nan_mask!, field, mask)
    fill_halo_regions!(field)

    substituting_field = deepcopy(field)

    # `good` tracks which points have valid data (1 = valid, 0 = missing).
    good = deepcopy(mask)
    parent(good) .= 1 .- parent(mask)

    # Phase 1: Horizontal nearest-neighbor fill (all levels simultaneously)
    fill_horizontally!(field, mask, good, substituting_field, arch, grid)

    # Phase 2: Vertical fallback top-to-bottom for points still NaN after horizontal fill
    for k in Nz-1:-1:1
        launch!(arch, grid, :xy, _vertical_fallback!, field, mask, k)
    end

    # Finalize: zero any remaining NaN, mark all fill points as good
    launch!(arch, grid, size(field), _finalize_stalled_fill!, field, good, mask)
    fill_halo_regions!(field)
    fill_halo_regions!(good)

    # Phase 3: Laplacian relaxation (all levels simultaneously)
    smooth_field!(field, mask, good, substituting_field, inpainting, arch, grid)

    fill_halo_regions!(field)

    return field
end

# Horizontal nearest-neighbor fill across all levels simultaneously.
# One fill_halo_regions! per iteration instead of per level.
function fill_horizontally!(field, mask, good, substituting_field, arch, grid)
    nfill_prev = sum(isnan.(interior(field)) .* interior(mask))

    for iter in 1:10_000
        nfill_prev == 0 && break

        fill_halo_regions!(field)

        launch!(arch, grid, size(field), _propagate_field!, substituting_field, field)
        launch!(arch, grid, size(field), _substitute_field!, field, substituting_field)

        nfill = sum(isnan.(interior(field)) .* interior(mask))

        nfill == nfill_prev && break  # stalled globally
        nfill == 0 && break

        nfill_prev = nfill
    end

    fill_halo_regions!(field)
end

# Laplacian relaxation across all levels simultaneously.
function smooth_field!(field, mask, good, change_field, inpainting, arch, grid)
    relc = convert(eltype(field), inpainting.relaxation_coefficient)
    iter = 0
    δmax = Inf

    while iter < inpainting.maxiter && δmax > inpainting.tolerance
        fill_halo_regions!(field)

        # Compute and apply change in one kernel
        launch!(arch, grid, size(field), _laplacian_relaxation!, field, change_field, mask, good, relc)

        # Convergence: max |Δϕ| over fill points
        δmax  = maximum(abs.(interior(change_field)) .* interior(mask))
        iter += 1
    end
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
