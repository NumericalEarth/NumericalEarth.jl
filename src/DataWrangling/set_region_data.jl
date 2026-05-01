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

struct BBoxOffset
    di :: Int
    dj :: Int
end

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
    return BBoxOffset(di, dj)
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
@inline read_data(data, i, j, k, ::Nothing,     mangling, FT) = nan_convert_missing(FT, mangle(i, j, k, data, mangling))
@inline read_data(data, i, j, k, b::BBoxOffset, mangling, FT) = nan_convert_missing(FT, mangle(i + b.di, j + b.dj, k, data, mangling))
@inline read_data(data, _, _, k, c::ColumnInfo, mangling, FT) = blend(data, c, k, mangling, c.ℑ, FT)

# NaN-aware bilinear blend: drop NaN corners and renormalise weights. 
# Returns NaN only when all four corners are land.
@inline function blend(data, c, k, mangling, ::Linear, FT)
    d00 = nan_convert_missing(FT, mangle(c.i⁻, c.j⁻, k, data, mangling))
    d10 = nan_convert_missing(FT, mangle(c.i⁺, c.j⁻, k, data, mangling))
    d01 = nan_convert_missing(FT, mangle(c.i⁻, c.j⁺, k, data, mangling))
    d11 = nan_convert_missing(FT, mangle(c.i⁺, c.j⁺, k, data, mangling))
    w00 = (1 - c.wx) * (1 - c.wy) * !isnan(d00)
    w10 =      c.wx  * (1 - c.wy) * !isnan(d10)
    w01 = (1 - c.wx) *      c.wy  * !isnan(d01)
    w11 =      c.wx  *      c.wy  * !isnan(d11)
    Σw  = w00 + w10 + w01 + w11
    Σw == 0 && return convert(FT, NaN)
    return (w00 * ifelse(isnan(d00), zero(FT), d00) +
            w10 * ifelse(isnan(d10), zero(FT), d10) +
            w01 * ifelse(isnan(d01), zero(FT), d01) +
            w11 * ifelse(isnan(d11), zero(FT), d11)) / Σw
end

@inline function blend(data, c, k, mangling, ::Nearest, FT)
    i = c.wx ≥ 0.5 ? c.i⁺ : c.i⁻
    j = c.wy ≥ 0.5 ? c.j⁺ : c.j⁻
    near = nan_convert_missing(FT, mangle(i, j, k, data, mangling))
    # If the closest corner is land, fall back to the NaN-aware Linear blend.
    return isnan(near) ? blend(data, c, k, mangling, Linear(), FT) : near
end

@kernel function _set_region_kernel!(dst, data, region, mangling, conversion, FT)
    i, j, k = @index(Global, NTuple)
    d = read_data(data, i, j, k, region, mangling, FT)
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
                          
    region = region_info(metadata.region, target, λc, φc)
    FT     = eltype(target)
    grid   = target.grid
    arch   = architecture(grid)
    data   = on_architecture(arch, data)
    launch!(arch, grid, :xyz, _set_region_kernel!, interior(target), data, region, mangling, conversion, FT)
    return nothing
end

function set_region_data!(target::FieldTimeSeries, data, λc, φc, metadata;
                          mangling = mangling_for(metadata, size(data, 2)),
                          conversion = conversion_units(metadata),
                          slot_indices = 1:size(target, 4))

    region = region_info(metadata.region, target, λc, φc)
    grid   = target.grid
    arch   = architecture(grid)
    FT     = eltype(target)
    data   = on_architecture(arch, data)
    for (data_time, slot_time) in zip(axes(data, 4), slot_indices)
        dest = view(interior(target), :, :, :, slot_time)
        slice = view(data, :, :, :, data_time)
        launch!(arch, grid, :xyz, _set_region_kernel!, dest, slice, region, mangling, conversion, FT)
    end
    return nothing
end
