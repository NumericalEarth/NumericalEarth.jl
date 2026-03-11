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

Diffusive inpainting algorithm. Missing values are first filled by horizontal
nearest-neighbor propagation, then unreachable points fall back to the level above.
Finally, all fill points are smoothed to satisfy ∇²ϕ = 0 using red-black
Gauss-Seidel iteration. Iteration stops when `max|ϕ_new - ϕ_old|` drops
below `tolerance`, or after `maxiter` iterations.
"""
struct DiffusiveInpainting{M, T}
    maxiter   :: M
    tolerance :: T
end

"""
    DiffusiveInpainting(; maxiter=10000, tolerance=1e-6)

Construct a `DiffusiveInpainting` algorithm.

Keyword Arguments
=================

- `maxiter`: Maximum number of Laplacian smoothing iterations. Default: `1000`.
- `tolerance`: Convergence threshold on `max|ϕ_new - ϕ_old| / max|ϕ_old|`. Default: `1e-4`.
"""
DiffusiveInpainting(; maxiter=1000, tolerance=1e-4) =
    DiffusiveInpainting(maxiter, tolerance)

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

# Red-black Gauss-Seidel kernel for ∇²ϕ = 0 on fill points.
# `color` is 0 (red) or 1 (black): a point (i,j) is active when (i+j) % 2 == color.
# Each color sweep reads only from the opposite color, so the update is parallel-safe
# without double-buffering.
@kernel function _gauss_seidel_sweep!(field, mask, color)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        is_fill   = mask[i, j, k]
        is_active = is_fill & (((i + j) & 1) == color)

        ϕ_avg = (field[i - 1, j, k] + field[i + 1, j, k] +
                 field[i, j - 1, k] + field[i, j + 1, k]) / 4

        field[i, j, k] = ifelse(is_active, ϕ_avg, field[i, j, k])
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
    * `DiffusiveInpainting(; maxiter, tolerance)`: (1) horizontal nearest-neighbor fill,
        (2) fallback to the level above for unreachable points,
        (3) red-black Gauss-Seidel smoothing (∇²ϕ = 0, until convergence or `maxiter`).
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
#   Phase 3: Red-black SOR for ∇²ϕ = 0 (all levels at once)
function diffusive_inpaint_mask!(field, mask, inpainting)
    grid = field.grid
    arch = architecture(grid)
    Nz = size(grid, 3)

    # NaN-mask all fill points
    launch!(arch, grid, size(field), _nan_mask!, field, mask)
    fill_halo_regions!(field)

    tmp_field = deepcopy(field)

    # Phase 1: Horizontal nearest-neighbor fill (all levels simultaneously)
    fill_horizontally!(field, mask, tmp_field, arch, grid)

    # Phase 2: Vertical fallback top-to-bottom for points still NaN after horizontal fill
    for k in Nz-1:-1:1
        launch!(arch, grid, :xy, _vertical_fallback!, field, mask, k)
    end

    # Zero any remaining NaN at fill points
    launch!(arch, grid, size(field), _fill_nans!, field)
    fill_halo_regions!(field)

    # Phase 3: Red-black SOR for ∇²ϕ = 0 (all levels simultaneously)
    smooth_field!(field, mask, tmp_field, inpainting, arch, grid)

    fill_halo_regions!(field)

    return field
end

# Horizontal nearest-neighbor fill across all levels simultaneously.
function fill_horizontally!(field, mask, tmp_field, arch, grid)
    nfill_prev = sum(isnan.(interior(field)) .* interior(mask))

    for iter in 1:10_000
        nfill_prev == 0 && break

        fill_halo_regions!(field)

        launch!(arch, grid, size(field), _propagate_field!, tmp_field, field)
        launch!(arch, grid, size(field), _substitute_field!, field, tmp_field)

        nfill = sum(isnan.(interior(field)) .* interior(mask))

        nfill == nfill_prev && break  # stalled globally
        nfill == 0 && break

        nfill_prev = nfill
    end

    fill_halo_regions!(field)
end

# Red-black Gauss-Seidel smoothing for ∇²ϕ = 0 on fill points.
# Convergence checked every `check_every` iterations via max|ϕ_new - ϕ_old| at fill points.
function smooth_field!(field, mask, tmp_field, inpainting, arch, grid)
    check_every = 10

    for iter in 1:inpainting.maxiter
        # Snapshot for convergence check
        if iter % check_every == 0
            parent(tmp_field) .= parent(field)
        end

        fill_halo_regions!(field)
        launch!(arch, grid, size(field), _gauss_seidel_sweep!, field, mask, 0) # red
        fill_halo_regions!(field)
        launch!(arch, grid, size(field), _gauss_seidel_sweep!, field, mask, 1) # black

        if iter % check_every == 0
            diff = interior(field) .- interior(tmp_field)
            M = maximum(abs, interior(field))
            δ = maximum(abs, diff .* interior(mask)) / M
            @debug "Gauss-Seidel smoothing" iter δ
            δ < inpainting.tolerance && break
        end
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
