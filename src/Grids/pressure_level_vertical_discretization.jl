using Adapt
using Oceananigans.Grids: AbstractVerticalCoordinate, AbstractUnderlyingGrid,
                          Center, Face, topology
using Oceananigans.Fields: Field, interior
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.OutputReaders: TimeSeriesInterpolation

import Oceananigans.Architectures: on_architecture
import Oceananigans.Grids: rnode, rnodes, generate_coordinate, validate_dimension_specification
import Oceananigans.Fields: _fractional_indices, fractional_x_index,
                            fractional_y_index, FractionalIndices

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

    Φi   = _geopotential_data_for_extrema(coord.geopotential)
    g    = coord.gravitational_acceleration
    z_lo, z_hi = FT.(extrema(Φi) ./ g)
    Lz   = z_hi - z_lo

    moved = PressureLevelVerticalDiscretization(g, on_architecture(arch, coord.geopotential))
    return Lz, moved
end

_geopotential_data_for_extrema(Φ::Field)                    = interior(Φ)
_geopotential_data_for_extrema(Φ::TimeSeriesInterpolation)  = parent(Φ.time_series)

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

# The 1-D `rnodes(grid, ℓz)` (and `znodes` via `rnodes`) has no sensible answer
# on a grid whose z varies per (i, j); throw an explicit error if anything asks
# for it. `rnode(i, j, k, grid, ...)` is the per-cell entry point.
function rnodes(grid::PressureLevelGrid, args...; kwargs...)
    throw(ArgumentError("PressureLevelGrid has a 3-D z-coordinate; " *
                        "use znode(i, j, k, grid, locs...) per cell instead of znodes/rnodes."))
end

#####
##### interpolate! hook: column-aware fractional z index
#####

@inline function _fractional_indices((x, y, z), grid::PressureLevelGrid, ℓx, ℓy, ℓz)
    ii = fractional_x_index(x, (ℓx, ℓy, ℓz), grid)
    jj = fractional_y_index(y, (ℓx, ℓy, ℓz), grid)
    kk = column_fractional_z_index(z, ii, jj, (ℓx, ℓy, ℓz), grid)
    return FractionalIndices(ii, jj, kk)
end

@inline function column_fractional_z_index(z, ii, jj, locs, grid)
    i = clamp(Base.unsafe_trunc(Int, ii), 1, grid.Nx)
    j = clamp(Base.unsafe_trunc(Int, jj), 1, grid.Ny)
    low, high = column_index_binary_search(z, i, j, grid, grid.Nz)
    z_lo = rnode(i, j, low,  grid, locs...)
    z_hi = rnode(i, j, high, grid, locs...)
    # Degenerate column-shelf case (clipped sub-surface levels with z_lo == z_hi):
    # snap to the integer index rather than divide by zero.
    kk = ifelse(z_hi == z_lo, oftype(z, low),
                (high - low) / (z_hi - z_lo) * (z - z_lo) + low)
    FT = eltype(grid)
    return convert(FT, kk)
end

@inline function column_index_binary_search(z, i, j, grid, Nz)
    low, high = 0, Nz - 1
    while low + 1 < high
        mid = (low + high) >> 1
        zm = @inbounds rnode(i, j, mid + 1, grid, Center(), Center(), Center())
        if zm == z
            return (mid + 1, mid + 1)
        elseif zm < z
            low = mid
        else
            high = mid
        end
    end
    return (low + 1, high + 1)
end

#####
##### Sub-surface clip helper (operates on raw geopotential, units m²/s²).
#####

"""
    clip_subsurface!(geopotential, surface_geopotential)

Clip each column of `geopotential` so that values below the local surface
geopotential are replaced by the surface value. Required to keep columns
monotonically increasing in z for `column_index_binary_search`.

Works for either a `Field` (3-D geopotential at a single time) or a
`TimeSeriesInterpolation` wrapping a `FieldTimeSeries` of geopotential.
"""
function clip_subsurface!(geopotential::Field, surface_geopotential)
    Nx, Ny, Nz = size(geopotential)
    Φi = interior(geopotential)
    Φs = interior(surface_geopotential)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        Φi[i, j, k] = max(Φi[i, j, k], Φs[i, j, 1])
    end
    # `Field(metadatum)` ran `fill_halo_regions!` before clipping, so halo cells
    # still hold the pre-clip (possibly sub-surface) values. Refill so any halo
    # read sees the clipped data.
    fill_halo_regions!(geopotential)
    return geopotential
end

function clip_subsurface!(geopotential::TimeSeriesInterpolation, surface_geopotential)
    fts_parent = parent(geopotential.time_series)
    Φs = interior(surface_geopotential)
    Nx, Ny, Nz, Nt = size(fts_parent)
    @inbounds for t in 1:Nt, k in 1:Nz, j in 1:Ny, i in 1:Nx
        fts_parent[i, j, k, t] = max(fts_parent[i, j, k, t], Φs[i, j, 1])
    end
    return geopotential
end
