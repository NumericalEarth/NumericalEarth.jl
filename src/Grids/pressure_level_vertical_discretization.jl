using Adapt
using KernelAbstractions: @kernel, @index
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: Field, interior
using Oceananigans.Grids: AbstractVerticalCoordinate, AbstractUnderlyingGrid, Center, Face
using Oceananigans.OutputReaders: TimeSeriesInterpolation
using Oceananigans.Utils: launch!

import Oceananigans.Architectures: on_architecture
import Oceananigans.Grids: rnode, rnodes, generate_coordinate, validate_dimension_specification
import Oceananigans.Fields: _fractional_indices, fractional_x_index,
                            fractional_y_index, FractionalIndices, index_binary_search

"""
    PressureLevelVerticalDiscretization{G, Geo}

A vertical discretization for pressure-level reanalysis data on a
`LatitudeLongitudeGrid`. There is no 1-D z-axis: every cell `(i, j, k)` has its
own height `geopotential[i, j, k] / gravitational_acceleration`. `znode`/`rnode`
read that directly; `znodes`/`rnodes` (the plural, 1-D forms) error explicitly
because they have no meaningful answer on this grid.

`geopotential` (units m²/s²) is a 3-D `Field` or a `TimeSeriesInterpolation`
over a `FieldTimeSeries`. The former gives a static z-coordinate; the latter
gives a time-evolving one driven by an attached `Clock`.

The `LatitudeLongitudeGrid` constructor still needs a value for `Lz`; we
compute it as `extrema(geopotential) / g` inside `generate_coordinate`.
"""
struct PressureLevelVerticalDiscretization{G, Geo} <: AbstractVerticalCoordinate
    gravitational_acceleration :: G
    geopotential               :: Geo
end

"""
    PressureLevelVerticalDiscretization(geopotential;
                                        gravitational_acceleration,
                                        surface_geopotential = nothing)

Build a discretization backed by per-column `geopotential` (m²/s²). `znode`
divides by `gravitational_acceleration` at read time.

If `surface_geopotential` is provided (a 2-D `Field`, m²/s²), columns are
clipped so that `geopotential[i,j,k] ≥ surface_geopotential[i,j]`. Required
when the source is ERA5 pressure-level data, because sub-surface levels are
filled with non-physical extrapolations that would break the column-monotonicity
assumed by `_fractional_indices`.
"""
function PressureLevelVerticalDiscretization(geopotential;
                                              gravitational_acceleration,
                                              surface_geopotential = nothing)
    isnothing(surface_geopotential) || clip_subsurface!(geopotential, surface_geopotential)
    return PressureLevelVerticalDiscretization(gravitational_acceleration, geopotential)
end

# Skip the generic validator (which would `length`-check the missing 1-D fields).
validate_dimension_specification(T, ξ::PressureLevelVerticalDiscretization, dir, N, FT) = ξ

# Compute `Lz` from the actual geopotential data instead of a placeholder vector.
# The returned discretization carries the geopotential (moved to `arch`) but
# nothing else — `znodes` will error and `rnode` reads `geopotential` directly.
function generate_coordinate(FT, topo, sz, halo,
                             coord::PressureLevelVerticalDiscretization,
                             coordinate_name, dim::Int, arch)
    dim == 3              || throw(ArgumentError("PressureLevelVerticalDiscretization requires dim=3"))
    coordinate_name == :z || throw(ArgumentError("PressureLevelVerticalDiscretization requires coordinate_name=:z"))

    g = coord.gravitational_acceleration
    Φi = _geopotential_data_for_extrema(coord.geopotential)
    z_lo, z_hi = FT.(extrema(Φi) ./ g)
    Lz = z_hi - z_lo

    moved = PressureLevelVerticalDiscretization(g, on_architecture(arch, coord.geopotential))
    return Lz, moved
end

_geopotential_data_for_extrema(Φ::Field) = interior(Φ)
# NOTE: For a TSI this reads every time slice. Fine while the FTS path isn't
# exercised; switch to a per-time-slice extent if we ever advance the clock here.
_geopotential_data_for_extrema(Φ::TimeSeriesInterpolation) = parent(Φ.time_series)

Adapt.adapt_structure(to, c::PressureLevelVerticalDiscretization) =
    PressureLevelVerticalDiscretization(c.gravitational_acceleration,
                                        Adapt.adapt(to, c.geopotential))

on_architecture(arch, c::PressureLevelVerticalDiscretization) =
    PressureLevelVerticalDiscretization(c.gravitational_acceleration,
                                        on_architecture(arch, c.geopotential))

function Base.show(io::IO, z::PressureLevelVerticalDiscretization)
    print(io, "PressureLevelVerticalDiscretization with $(size(z.geopotential, 3)) levels, ",
              "g = ", z.gravitational_acceleration, " m/s²")
end

"""
    PressureLevelGrid

Type alias for any underlying grid whose vertical coordinate is a
[`PressureLevelVerticalDiscretization`](@ref).
"""
const PressureLevelGrid =
    AbstractUnderlyingGrid{<:Any, <:Any, <:Any, <:Any, <:PressureLevelVerticalDiscretization}

#####
##### Node accessors: per-cell znode works; 1-D znodes / rnodes error.
#####

# znode falls through to rnode in Oceananigans; only rnode needs overriding.
@inline rnode(i, j, k, grid::PressureLevelGrid, ℓx, ℓy, ℓz) =
    @inbounds grid.z.geopotential[i, j, k] / grid.z.gravitational_acceleration

# For the 3-location form `rnodes(grid, ℓx, ℓy, ℓz)` we have a genuine 3-D
# answer: wrap `rnode(i, j, k, grid, locs...)` in a `KernelFunctionOperation`
# so the caller gets a lazy field of per-cell heights. The 1-location
# `rnodes(grid, ℓz)` (and the equivalent `znodes` call) has no sensible
# answer on a grid whose z varies per (i, j); throw explicitly.

const _PL_NO_1D_Z_MSG = "PressureLevelGrid has a 3-D z-coordinate; " *
                        "use `rnodes(grid, ℓx, ℓy, ℓz)` for the 3-D form, or " *
                        "`znode(i, j, k, grid, locs...)` per cell."

@inline _znode_op(i, j, k, grid, ℓx, ℓy, ℓz) = rnode(i, j, k, grid, ℓx, ℓy, ℓz)

@inline rnodes(::PressureLevelGrid, ::Face;   kwargs...) = throw(ArgumentError(_PL_NO_1D_Z_MSG))
@inline rnodes(::PressureLevelGrid, ::Center; kwargs...) = throw(ArgumentError(_PL_NO_1D_Z_MSG))

@inline rnodes(grid::PressureLevelGrid, ℓx, ℓy, ℓz; kwargs...) =
    KernelFunctionOperation{typeof(ℓx), typeof(ℓy), typeof(ℓz)}(_znode_op, grid, ℓx, ℓy, ℓz)

#####
##### interpolate! hook: column-aware fractional z index
#####

# A 1-D `getindex`-only view of column `(i, j)` of a `PressureLevelGrid`'s z,
# used to feed Oceananigans' `index_binary_search` without materializing the
# column. Stack-allocated; bitstype-clean for GPU kernels when `grid` is.
struct _ColumnView{G}
    grid :: G
    i    :: Int
    j    :: Int
end

@inline Base.getindex(c::_ColumnView, k::Int) =
    rnode(c.i, c.j, k, c.grid, Center(), Center(), Center())

@inline function _fractional_indices((x, y, z)::NTuple{3, Any},
                                      grid::PressureLevelGrid, ℓx, ℓy, ℓz)
    ii = fractional_x_index(x, (ℓx, ℓy, ℓz), grid)
    jj = fractional_y_index(y, (ℓx, ℓy, ℓz), grid)
    kk = column_fractional_z_index(z, ii, jj, (ℓx, ℓy, ℓz), grid)
    return FractionalIndices(ii, jj, kk)
end

# Column-region source (Flat-Flat-Bounded): there's only one (i,j)=(1,1), so
# bisect that single column directly. Mirrors the 3-D form's column logic.
@inline function _fractional_indices((z,)::NTuple{1, Any},
                                      grid::PressureLevelGrid, ::Nothing, ::Nothing, ℓz)
    kk = column_fractional_z_index(z, 1, 1, (nothing, nothing, ℓz), grid)
    return FractionalIndices(nothing, nothing, kk)
end

@inline function column_fractional_z_index(z, ii, jj, locs, grid)
    i = clamp(Base.unsafe_trunc(Int, ii), 1, grid.Nx)
    j = clamp(Base.unsafe_trunc(Int, jj), 1, grid.Ny)
    column = _ColumnView(grid, i, j)
    low, high = index_binary_search(column, z, grid.Nz)
    z_lo = @inbounds column[low]
    z_hi = @inbounds column[high]
    # Degenerate column-shelf case (clipped sub-surface levels with z_lo == z_hi):
    # snap to the integer index rather than divide by zero.
    kk = ifelse(z_hi == z_lo, oftype(z, low),
                (high - low) / (z_hi - z_lo) * (z - z_lo) + low)
    FT = eltype(grid)
    return convert(FT, kk)
end

#####
##### Sub-surface clip helper (operates on raw geopotential, units m²/s²).
#####

@kernel function _clip_subsurface_kernel!(Φ, Φ_sfc)
    i, j, k = @index(Global, NTuple)
    @inbounds Φ[i, j, k] = max(Φ[i, j, k], Φ_sfc[i, j, 1])
end

"""
    clip_subsurface!(geopotential, surface_geopotential)

Clip each column of `geopotential` so that values below the local surface
geopotential are replaced by the surface value. Required to keep columns
monotonically increasing in z for the column bisection in
[`_fractional_indices`](@ref).

Works for either a `Field` (3-D geopotential at a single time) or a
`TimeSeriesInterpolation` wrapping a `FieldTimeSeries` of geopotential.
"""
function clip_subsurface!(Φ::Field, Φ_sfc)
    grid = Φ.grid
    arch = architecture(Φ)
    launch!(arch, grid, :xyz, _clip_subsurface_kernel!, Φ, Φ_sfc)
    # `Field(metadatum)` ran `fill_halo_regions!` before clipping, so halo cells
    # still hold the pre-clip (possibly sub-surface) values. Refill so any halo
    # read sees the clipped data.
    fill_halo_regions!(Φ)
    return Φ
end

# TSI path is CPU-only for now (no FTS-backed discretization is exercised in
# the current pipeline). When that lands, swap the loop for a 4-D launch.
function clip_subsurface!(geopotential::TimeSeriesInterpolation, surface_geopotential)
    fts_parent = parent(geopotential.time_series)
    Φs = interior(surface_geopotential)
    Nx, Ny, Nz, Nt = size(fts_parent)
    @inbounds for t in 1:Nt, k in 1:Nz, j in 1:Ny, i in 1:Nx
        fts_parent[i, j, k, t] = max(fts_parent[i, j, k, t], Φs[i, j, 1])
    end
    return geopotential
end
