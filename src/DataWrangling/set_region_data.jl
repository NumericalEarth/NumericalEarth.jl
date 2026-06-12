using Oceananigans.Utils: launch!
using Oceananigans.Architectures: AbstractArchitecture, architecture
using Oceananigans.Grids: AbstractGrid, Periodic, Bounded, λnodes, φnodes
using Oceananigans.Fields: Field, interior, interpolate!
using Oceananigans.Fields: convert_to_λ₀_λ₀_plus360
using GPUArraysCore: @allowscalar

#####
##### Region helpers shared across dataset backends
#####

function compute_bounding_nodes(grid, LH, hnodes)
    hg = hnodes(grid, LH())
    h₁ = @allowscalar minimum(hg)
    h₂ = @allowscalar maximum(hg)
    return h₁, h₂
end

# `ε` forgives Float32 to Float64 promotion noise so the slice doesn't lose a
# cell at each end when grid centers are compared against file centers.
function compute_bounding_indices(bounds::Tuple, hc)
    h₁, h₂ = bounds
    Nh = length(hc)
    ε  = eps(Float32) * max(one(eltype(hc)), abs(h₁), abs(h₂))
    i₁ = max(searchsortedfirst(hc, h₁ - ε), 1)
    i₂ = min( searchsortedlast(hc, h₂ + ε), Nh)
    return i₁, i₂
end

# Periodic only when the restricted span equals the full native span.
function infer_longitudinal_topology(full_longitude, restricted_longitude)
    full_span = full_longitude[end] - full_longitude[1]
    restricted_span = restricted_longitude[end] - restricted_longitude[1]
    return restricted_span ≈ full_span ? Periodic : Bounded
end

#####
##### Mangling utilities
#####

struct ShiftSouth end
struct AverageNorthSouth end

# `mangle(i, j, k, data, mangling)` reads file `data` at metadata-grid index `(i, j, k)`, accounting
# for staggered lat-axis offsets. Used inside the region-aware kernel.
@inline mangle(i, j, k, data, ::Nothing) = @inbounds data[i, j, k]
@inline mangle(i, j, k, data, ::ShiftSouth) = @inbounds data[i, max(j - 1, 1), k]
@inline mangle(i, j, k, data, ::AverageNorthSouth) = @inbounds (data[i, j, k] + data[i, j + 1, k]) / 2

#####
##### Region-aware filling for Fields and FieldTimeSeries via a single kernel.
#####
##### `read_data(data, i, j, k, region_info, mangling)` is the only access point: it composes the
##### file-axis offset (region) with the lat-axis remap (mangling). All region/mangling combinations
##### go through one kernel that handles NaN + unit conversion in the same pass.
#####

struct BoundingBoxOffset
    di :: Int
    dj :: Int
end

"""
    ColumnInfo{F, I}

Resolved location of a `Column` extraction inside the file grid. Built once per `set_region_data!` call by
`region_info(::Column, …)` and captured into `_set_region_kernel!` as a stack-friendly struct.

- `i⁻`, `i⁺`: bracketing longitude indices (`i⁺` wraps to `1` across the periodic seam).
- `j⁻`, `j⁺`: bracketing latitude indices.
- `wx`, `wy`: bilinear blend weights in `[0, 1]` (`0` → at `i⁻`/`j⁻`, `1` → at `i⁺`/`j⁺`).
- `ℑ`: interpolation kind, `Linear()` or `Nearest()`.
"""
struct ColumnInfo{F, I}
    i⁻ :: Int
    i⁺ :: Int
    j⁻ :: Int
    j⁺ :: Int
    wx :: F
    wy :: F
    ℑ  :: I
end

# `region_info` resolves the target's region to a kernel-friendly struct.
region_info(::Nothing, target, λc, φc) = nothing

function region_info(::BoundingBox, target, λc, φc)
    LX, LY, _  = Oceananigans.Fields.location(target)
    λmin, λmax = compute_bounding_nodes(target.grid, LX, λnodes)
    φmin, φmax = compute_bounding_nodes(target.grid, LY, φnodes)

    # Shift the target's longitude into the file's `[λc[1], λc[1]+360)`
    if !isempty(λc)
        λmin = convert_to_λ₀_λ₀_plus360(λmin, λc[1])
        λmax = convert_to_λ₀_λ₀_plus360(λmax, λc[1])
    end

    i₁, _ = compute_bounding_indices((λmin, λmax), λc)
    j₁, _ = compute_bounding_indices((φmin, φmax), φc)

    Nx, Ny, _ = size(target)
    di = clamp(i₁ - 1, 0, max(length(λc) - Nx, 0))
    dj = clamp(j₁ - 1, 0, max(length(φc) - Ny, 0))
    return BoundingBoxOffset(di, dj)
end

function region_info(col::Column, target, λc, φc)
    i⁻, i⁺, wx = bracket_with_weight(λc, col.longitude; period = infer_longitudinal_period(λc))
    j⁻, j⁺, wy = bracket_with_weight(φc, col.latitude)  # latitude is never cyclic
    FT = eltype(target)
    return ColumnInfo(i⁻, i⁺, j⁻, j⁺, FT(wx), FT(wy), col.interpolation)
end

# 360 if `λc` spans the full globe (cyclic), else `nothing`.
function infer_longitudinal_period(λc)
    length(λc) < 2 && return nothing
    Δ = λc[2] - λc[1]
    span = λc[end] - λc[1] + Δ
    return span ≈ 360 ? 360 : nothing
end

# Cyclic-aware bracketing. With `period`, the cell between `coords[end]` and `coords[1] + period` is the wrap cell:
# returns `(n, 1, w)` so the blend reads `data[n, …]` and `data[1, …]`.
function bracket_with_weight(coords, x; period = nothing)
    n = length(coords)

    # Single-cell axis: nothing to bracket — point both corners at the only cell.
    n ≤ 1 && return 1, 1, zero(x)

    if !isnothing(period)
        x = coords[1] + mod(x - coords[1], period)
        if x > coords[end]
            Δ = (coords[1] + period) - coords[end]
            w = (x - coords[end]) / Δ
            return n, 1, clamp(w, 0, 1)
        end
    end

    i⁺ = searchsortedfirst(coords, x)
    i⁺ = clamp(i⁺, 2, n)
    i⁻ = i⁺ - 1
    Δ = coords[i⁺] - coords[i⁻]
    w = Δ == 0 ? zero(x) : (x - coords[i⁻]) / Δ
    return i⁻, i⁺, clamp(w, 0, 1)
end

# `mangling_for` detects a file/grid lat-axis offset from the data shape.
function mangling_for(metadata, data_lat_count)
    Ny = size(metadata)[2]
    return data_lat_count == Ny - 1 ? ShiftSouth() :
           data_lat_count == Ny + 1 ? AverageNorthSouth() :
                                      nothing
end

# `read_data(data, i, j, k, region, mangling, FT)` returns the file value at
# the grid's (i, j, k) as `FT`, with `Missing` converted to NaN.
@inline read_data(data, i, j, k, ::Nothing,     mangling, missing_val, FT) = nan_convert_missing(FT, mangle(i, j, k, data, mangling), missing_val)
@inline read_data(data, i, j, k, b::BoundingBoxOffset, mangling, missing_val, FT) = nan_convert_missing(FT, mangle(i + b.di, j + b.dj, k, data, mangling), missing_val)
@inline read_data(data, _, _, k, c::ColumnInfo, mangling, missing_val, FT) = blend(c.ℑ, data, c, k, mangling, missing_val, FT)

# Land cells arrive as NaN through `nan_convert_missing`.
# A naive bilinear average of four corners would propagate that NaN into the
# interior, biasing every column whose stencil touches a coast. Instead we drop
# any NaN corner and renormalise the weights over the surviving wet corners,
# returning NaN only when all four are land.
@inline function blend(::Linear, data, c, k, mangling, missing_val, FT)
    d00 = nan_convert_missing(FT, mangle(c.i⁻, c.j⁻, k, data, mangling), missing_val)
    d10 = nan_convert_missing(FT, mangle(c.i⁺, c.j⁻, k, data, mangling), missing_val)
    d01 = nan_convert_missing(FT, mangle(c.i⁻, c.j⁺, k, data, mangling), missing_val)
    d11 = nan_convert_missing(FT, mangle(c.i⁺, c.j⁺, k, data, mangling), missing_val)
    w00 = (1 - c.wx) * (1 - c.wy) * !isnan(d00)
    w10 =      c.wx  * (1 - c.wy) * !isnan(d10)
    w01 = (1 - c.wx) *      c.wy  * !isnan(d01)
    w11 =      c.wx  *      c.wy  * !isnan(d11)
    Σw  = w00 + w10 + w01 + w11
    numerator = (w00 * ifelse(isnan(d00), zero(FT), d00) +
                 w10 * ifelse(isnan(d10), zero(FT), d10) +
                 w01 * ifelse(isnan(d01), zero(FT), d01) +
                 w11 * ifelse(isnan(d11), zero(FT), d11))
    denominator = ifelse(Σw == 0, one(FT), Σw)
    return ifelse(Σw == 0, convert(FT, NaN), numerator / denominator)
end

@inline function blend(::Nearest, data, c, k, mangling, missing_val, FT)
    i = ifelse(c.wx ≥ 0.5, c.i⁺, c.i⁻)
    j = ifelse(c.wy ≥ 0.5, c.j⁺, c.j⁻)
    near = nan_convert_missing(FT, mangle(i, j, k, data, mangling), missing_val)
    # If the closest corner is land, fall back to the NaN-aware Linear blend.
    return ifelse(isnan(near), blend(Linear(), data, c, k, mangling, missing_val, FT), near)
end

# Fallback dispatch that assumes missing_val = missing
blend(scheme, data, c, k, mangling, FT) = blend(scheme, data, c, k, mangling, missing, FT)

@kernel function _set_region_kernel!(dst, data, region, mangling, conversion, missing_val, FT)
    i, j, k = @index(Global, NTuple)
    d = read_data(data, i, j, k, region, mangling, missing_val, FT)
    d = convert_units(d, conversion)
    @inbounds dst[i, j, k] = d
end

"""
    set_region_data!(target, data, λc, φc, metadata)

Fill the region of `target` (Field or FieldTimeSeries) implied by `metadata.region` from `data`,
applying mangling, NaN conversion, and unit conversion in a single GPU-friendly kernel pass.
"""
function set_region_data!(target::Field, data, λc, φc, metadata;
                          mangling = mangling_for(metadata, size(data, 2)),
                          conversion = conversion_units(metadata))

    region      = region_info(metadata.region, target, λc, φc)
    FT          = eltype(target)
    grid        = target.grid
    arch        = architecture(grid)
    data        = on_architecture(arch, data)
    missing_val = missing_value(metadata)
    launch!(arch, grid, :xyz, _set_region_kernel!, interior(target), data, region, mangling, conversion, missing_val, FT)
    return nothing
end

function set_region_data!(target::FieldTimeSeries, data, λc, φc, metadata;
                          mangling = mangling_for(metadata, size(data, 2)),
                          conversion = conversion_units(metadata),
                          slot_indices = 1:size(target, 4))

    region      = region_info(metadata.region, target, λc, φc)
    grid        = target.grid
    arch        = architecture(grid)
    FT          = eltype(target)
    data        = on_architecture(arch, data)
    missing_val = missing_value(metadata)
    for (data_time, slot_time) in zip(axes(data, 4), slot_indices)
        dest = view(interior(target), :, :, :, slot_time)
        slice = view(data, :, :, :, data_time)
        launch!(arch, grid, :xyz, _set_region_kernel!, dest, slice, region, mangling, conversion, missing_val, FT)
    end
    return nothing
end
