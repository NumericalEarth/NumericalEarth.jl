using Oceananigans.Utils: launch!
using Oceananigans.Architectures: AbstractArchitecture, architecture
using Oceananigans.Grids: AbstractGrid, Periodic, Bounded, λnodes, φnodes
using Oceananigans.Fields: Field, interior, interpolate!
using GPUArraysCore: @allowscalar

#####
##### Region helpers shared across dataset backends
#####

function node_extrema(grid, LH, hnodes)
    hg = hnodes(grid, LH())
    return @allowscalar extrema(hg)
end

# File index of the first ascending coordinate at or after `h`. `h` is a Float32 grid node
# promoted to Float64; `tolerance` absorbs the promotion drift, capped at a quarter cell because
# `eps(Float32) * |h|` alone exceeds half a cell on arc-second axes near 180°. On an axis whose
# labels sit half a cell from the grid's nodes (regional ERA5 latitude), the tie resolves to the
# label north/east of the node. A node more than half a cell before the first label extrapolates to
# an index below 1, marking a file that begins east/north of where the read starts.
function file_cell_index(h, hc)
    Nh = length(hc)
    Nh > 1 || return 1
    j = clamp(searchsortedlast(hc, h), 1, Nh - 1)
    Δ = hc[j+1] - hc[j]
    tolerance = min(8 * eps(Float32) * max(1, abs(h)), abs(Δ) / 4)
    h ≥ hc[1] && return clamp(searchsortedfirst(hc, h - tolerance), 1, Nh)
    return 1 + ceil(Int, (h - tolerance - hc[1]) / Δ)
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
#
# Clamp indices to avoid out-of-bounds access
@inline clamp_i(i, data) = clamp(i, 1, size(data, 1))
@inline clamp_j(j, data) = clamp(j, 1, size(data, 2))

# Longitude index of a window continuing across the seam of a file that spans the globe
# (`Nλ` columns); `Nλ = 0` marks a regional file, whose indices are clamped instead.
@inline wrapped_i(i, Nλ) = ifelse(Nλ > 0, mod1(i, max(Nλ, 1)), i)
@inline mangle(i, j, k, data, ::Nothing) = @inbounds data[clamp_i(i, data), clamp_j(j, data), k]
@inline mangle(i, j, k, data, ::ShiftSouth) = @inbounds data[clamp_i(i, data), clamp_j(j - 1, data), k]
@inline mangle(i, j, k, data, ::AverageNorthSouth) =
    @inbounds (data[clamp_i(i, data), clamp_j(j, data), k] + data[clamp_i(i, data), clamp_j(j + 1, data), k]) / 2

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
    Nλ :: Int # file longitude count when the file spans the globe, else 0 (no wrapping)
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
    LX, LY, _ = Oceananigans.Fields.location(target)
    λmin, λmax = node_extrema(target.grid, LX, λnodes)
    φmin, φmax = node_extrema(target.grid, LY, φnodes)

    # Shift the target's longitude into the file's `[λc[1], λc[1]+360)`, allowing half a cell of
    # slack at the west edge: a Float32 grid node can land a few ulps below `λc[1]`, and wrapping
    # such a node by +360° would place the window at the far end of the file.
    λfile = length(λc) > 1 ? convert_to_λ₀_λ₀_plus360(λmin, λc[1] - abs(λc[2] - λc[1]) / 2) : λmin

    di = file_cell_index(λfile, λc) - 1
    dj = file_cell_index(φmin, φc) - 1

    Nx, Ny, _ = size(target)

    # A window overhanging the file's east edge continues across the seam when the file spans the
    # globe; a regional file has no data beyond its edge, so its window has to stay inside.
    Nλ = isnothing(infer_longitudinal_period(λc)) ? 0 : length(λc)
    Nλ > 0 || validate_file_covers_grid(di, Nx, λmin, λmax, λc, "longitude")
    validate_file_covers_grid(dj, Ny, φmin, φmax, φc, "latitude")

    return BoundingBoxOffset(di, dj, Nλ)
end

function region_info(col::Column, target, λc, φc)
    i⁻, i⁺, wx = bracket_with_weight(λc, col.longitude; period = infer_longitudinal_period(λc))
    j⁻, j⁺, wy = bracket_with_weight(φc, col.latitude)  # latitude is never cyclic
    FT = eltype(target)
    return ColumnInfo(i⁻, i⁺, j⁻, j⁺, FT(wx), FT(wy), col.interpolation)
end

# Data lands at one fixed offset into the file, so the file has to label every cell it fills.
# `restrict` builds the native grid by bracketing the region with native cell centers, which
# reaches one cell past an edge that falls on a cell face — a file materialized on the requested
# region alone then comes up short, and its values would land on the wrong cells.
function validate_file_covers_grid(offset, N, hmin, hmax, hc, axis)
    0 ≤ offset && offset + N ≤ length(hc) && return nothing

    throw(ArgumentError("the file does not cover the native grid in $axis: the read needs $N cells " *
                        "from $hmin to $hmax, but the file labels $(length(hc)) cells " *
                        "from $(first(hc)) to $(last(hc)). Materialize the file on " *
                        "`native_grid(metadatum)`, or fetch a margin of native cells around the " *
                        "region the way `era5_request_area` does."))
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
@inline read_data(data, i, j, k, b::BoundingBoxOffset, mangling, missing_val, FT) = nan_convert_missing(FT, mangle(wrapped_i(i + b.di, b.Nλ), j + b.dj, k, data, mangling), missing_val)
@inline read_data(data, _, _, k, c::ColumnInfo, mangling, missing_val, FT) = blend(c.ℑ, data, c, k, mangling, missing_val, FT)

# Land cells arrive as NaN through `nan_convert_missing`.
# A naive bilinear average of four corners would propagate that NaN into the
# interior, biasing every column whose stencil touches a coast. Instead we drop
# any NaN corner and renormalize the weights over the surviving wet corners,
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

# TODO: upstream to Oceananigans.Architectures alongside its SubArray/OffsetArray methods.
# `on_architecture` has no `Base.ReshapedArray` method, so host data arriving reshaped — e.g. a
# 2-D NetCDF variable reshaped to (Nx, Ny, 1) — falls through the generic identity fallback and
# reaches GPU kernels as CPU memory (kernel compilation failure).
architecture_ready(arch, data) = on_architecture(arch, data)
architecture_ready(arch, data::Base.ReshapedArray) =
    reshape(on_architecture(arch, parent(data)), size(data))

"""
    set_region_data!(target, data, λc, φc, metadata)

Fill the region of `target` (Field or FieldTimeSeries) implied by `metadata.region` from `data`,
applying mangling, NaN conversion, and unit conversion in a single GPU-friendly kernel pass.
"""
function set_region_data!(target::Field, data, λc, φc, metadata;
                          mangling = mangling_for(metadata, size(data, 2)),
                          conversion = conversion_units(metadata),
                          region = region_info(metadata.region, target, λc, φc),
                          parameters = :xyz)

    FT          = eltype(target)
    grid        = target.grid
    arch        = architecture(grid)
    data        = architecture_ready(arch, data)
    missing_val = missing_value(metadata)
    # `target` rather than `interior(target)`: the two index identically over the interior, and a
    # windowed field is then filled over its own indices.
    launch!(arch, grid, parameters, _set_region_kernel!, target, data, region, mangling, conversion, missing_val, FT)
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
    data        = architecture_ready(arch, data)
    missing_val = missing_value(metadata)
    for (data_time, slot_time) in zip(axes(data, 4), slot_indices)
        dest = view(interior(target), :, :, :, slot_time)
        slice = view(data, :, :, :, data_time)
        launch!(arch, grid, :xyz, _set_region_kernel!, dest, slice, region, mangling, conversion, missing_val, FT)
    end
    return nothing
end
