using Adapt
using KernelAbstractions: @kernel, @index
using Statistics: mean
using Oceananigans.AbstractOperations: KernelFunctionOperation
using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans: instantiated_location
using Oceananigans.Fields: Field, compute!, interior
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: AbstractVerticalCoordinate, AbstractUnderlyingGrid, Center, Face, Flat, LatitudeLongitudeGrid, topology
using Oceananigans.OutputReaders: TimeSeriesInterpolation
using Oceananigans.Utils: launch!

import Oceananigans.Architectures: on_architecture
import Oceananigans.Grids: rnode, rnodes, znodes, generate_coordinate,
                           validate_dimension_specification
import Oceananigans.Fields: _fractional_indices, fractional_x_index,
                            fractional_y_index, FractionalIndices, index_binary_search

"""
    PressureLevelVerticalDiscretization{G, Geo}

A vertical discretization for pressure-level reanalysis data on a
`LatitudeLongitudeGrid`. Per-cell heights are `geopotential[i, j, k] /
gravitational_acceleration`; `znode(i, j, k, grid)` returns that directly.

Two `znodes` paths:

- `znodes(grid, ℓ...)` returns the **column-mean** z profile as a 1-D
  `Vector{Float64}` of length `Nz` — the representative axis that plot
  recipes, `Lz`, and length consumers expect when only the grid is in hand.
- `znodes(field)` returns the column-mean `Vector{Float64}` when the
  field has no horizontal extent (`Flat` topology or `Nothing` horizontal
  locations, e.g. after `mean(...; dims=(1, 2))`); otherwise it returns
  a 3-D `Field` of per-cell heights at the field's own location.

`geopotential` (units m²/s²) is a 3-D `Field` or a `TimeSeriesInterpolation`
over a `FieldTimeSeries`. The former gives a static z-coordinate; the latter
gives a time-evolving one driven by an attached `Clock`.

The `LatitudeLongitudeGrid` constructor needs a value for `Lz`; we compute it
as `extrema(geopotential) / g` inside `generate_coordinate`.
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
# The returned discretization carries the geopotential (transferred to `arch`)
# but nothing else — `znodes` will error and `rnode` reads `geopotential` directly.
function generate_coordinate(FT, topo, sz, halo,
                             coord::PressureLevelVerticalDiscretization,
                             coordinate_name, dim::Int, arch)
    dim == 3              || throw(ArgumentError("PressureLevelVerticalDiscretization requires dim=3"))
    coordinate_name == :z || throw(ArgumentError("PressureLevelVerticalDiscretization requires coordinate_name=:z"))

    g = coord.gravitational_acceleration
    Φi = geopotential_data_for_extrema(coord.geopotential)
    z_lo, z_hi = FT.(extrema(Φi) ./ g)
    Lz = z_hi - z_lo

    arch_discretization = PressureLevelVerticalDiscretization(g, on_architecture(arch, coord.geopotential))
    return Lz, arch_discretization
end

geopotential_data_for_extrema(Φ::Field) = interior(Φ)
# Use `interior(fts)` — not `parent(fts)` — so halo zeros don't dominate the
# extrema / column mean.
# NOTE: For a TSI this reads every time slice. Fine while the FTS path isn't
# exercised; switch to a per-time-slice extent if we ever advance the clock here.
geopotential_data_for_extrema(Φ::TimeSeriesInterpolation) = interior(Φ.time_series)

Adapt.adapt_structure(to, z::PressureLevelVerticalDiscretization) =
    PressureLevelVerticalDiscretization(z.gravitational_acceleration,
                                        Adapt.adapt(to, z.geopotential))

on_architecture(arch, z::PressureLevelVerticalDiscretization) =
    PressureLevelVerticalDiscretization(z.gravitational_acceleration,
                                        on_architecture(arch, z.geopotential))

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

# Override the LLG show, which reads `grid.z.cᵃᵃᶠ` and crashes on PLVD.
# We print the horizontal axes and report the z coordinate via the
# `PressureLevelVerticalDiscretization` `show` defined above. Targeting the
# LLG{...PLVD} concrete combination is strictly more specific than either
# `show(::IO, ::LatitudeLongitudeGrid, withsummary=true)` or
# `show(::IO, ::PressureLevelGrid)`, so dispatch is unambiguous.
const _LLG_PLG = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any,
                                       <:PressureLevelVerticalDiscretization}

function Base.show(io::IO, grid::_LLG_PLG, withsummary::Bool=true)
    TX, TY, _ = topology(grid)
    if withsummary
        print(io, summary(grid), "\n")
    end
    print(io, "├── topology: (", TX, ", ", TY, ", Bounded)\n",
              "├── size: (Nx=", grid.Nx, ", Ny=", grid.Ny, ", Nz=", grid.Nz, ")\n",
              "├── halo: (Hx=", grid.Hx, ", Hy=", grid.Hy, ", Hz=", grid.Hz, ")\n",
              "├── Lz (column-mean span): ", grid.Lz, " m\n",
              "└── ", grid.z)
end

#####
##### Node accessors: per-cell znode works; 1-D znodes / rnodes error.
#####

# znode falls through to rnode in Oceananigans; only rnode needs overriding.
@inline rnode(i, j, k, grid::PressureLevelGrid, ℓx, ℓy, ℓz) =
    @inbounds grid.z.geopotential[i, j, k] / grid.z.gravitational_acceleration

# `rnodes(grid, ...)` returns the column-mean z profile as a 1-D
# `Vector{Float64}` of length `Nz`. This matches what plot recipes, `Lz`,
# and length consumers expect when only the grid is in hand. Per-cell access
# via a `Field` is exposed through `znodes(::Field)` below.
@inline rnodes(grid::PressureLevelGrid, ℓz::Center;          kwargs...) = mean_height_profile(grid)
@inline rnodes(grid::PressureLevelGrid, ℓz::Face;            kwargs...) = mean_height_profile(grid)
@inline rnodes(grid::PressureLevelGrid, ℓx, ℓy, ℓz;          kwargs...) = mean_height_profile(grid)
@inline rnodes(grid::PressureLevelGrid, ::Nothing, ::Nothing, ℓz; kwargs...) = mean_height_profile(grid)

function mean_height_profile(grid::PressureLevelGrid)
    g = grid.z.gravitational_acceleration
    Φi = geopotential_data_for_extrema(grid.z.geopotential)
    Nz = grid.Nz
    # `selectdim(Φi, 3, k)` works for both 3-D (Field) and 4-D (TSI parent)
    # geopotential storage; the trailing dims are reduced away by `mean`.
    return [mean(selectdim(Φi, 3, k)) / g for k in 1:Nz]
end

# `znodes(::Field)` on a `PressureLevelGrid`:
# - When the field has no horizontal extent (`Flat` topology or `Nothing`
#   horizontal locations from a `mean(...; dims=(1, 2))`-style reduction),
#   return the column-mean 1-D `Vector{Float64}` — same as the grid-level
#   `rnodes(grid, ℓz)` 1-arg form.
# - Otherwise, build a 3-D per-cell `Field` via a `KernelFunctionOperation`
#   over `rnode`.
@inline znode_op(i, j, k, grid) =
    @inbounds grid.z.geopotential[i, j, k] / grid.z.gravitational_acceleration

# Grid sits at different type-parameter positions in `Field` and
# `FieldTimeSeries`, so we list both concrete types and dispatch each through
# the same `znodes_for_plg` helper. `Field` is more specific than the
# Oceananigans `znodes(::Field)` default; `FieldTimeSeries` is more specific
# than `znodes(::AbstractField)`. Both wins on dispatch.
const PressureLevelField =
    Field{<:Any, <:Any, <:Any, <:Any, <:PressureLevelGrid}

const PressureLevelFieldTimeSeries =
    FieldTimeSeries{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:PressureLevelGrid}

znodes(f::PressureLevelField; kwargs...)            = znodes_for_plg(f; kwargs...)
znodes(fts::PressureLevelFieldTimeSeries; kwargs...) = znodes_for_plg(fts; kwargs...)

function znodes_for_plg(f; kwargs...)
    grid = f.grid
    TX, TY, _ = topology(grid)
    locs = instantiated_location(f)

    if (TX === Flat || locs[1] isa Nothing) &&
       (TY === Flat || locs[2] isa Nothing)
        return rnodes(grid, locs[3]; kwargs...)
    end

    LX, LY, LZ = map(typeof, locs)
    op = KernelFunctionOperation{LX, LY, LZ}(znode_op, grid)
    return compute!(Field(op))
end

#####
##### interpolate! hook: column-aware fractional z index
#####

# A 1-D `getindex`-only view of column `(i, j)` of a `PressureLevelGrid`'s z,
# used to feed Oceananigans' `index_binary_search` without materializing the
# column. Stack-allocated; bitstype-clean for GPU kernels when `grid` is.
struct ColumnView{G}
    grid :: G
    i    :: Int
    j    :: Int
end

@inline Base.getindex(c::ColumnView, k::Int) =
    rnode(c.i, c.j, k, c.grid, Center(), Center(), Center())

@inline function _fractional_indices((x, y, z)::NTuple{3, Any},
                                      grid::PressureLevelGrid, ℓx, ℓy, ℓz)
    ii = fractional_x_index(x, (ℓx, ℓy, ℓz), grid)
    jj = fractional_y_index(y, (ℓx, ℓy, ℓz), grid)
    kk = column_fractional_z_index(z, ii, jj, grid)
    return FractionalIndices(ii, jj, kk)
end

# Column-region source (Flat-Flat-Bounded): there's only one (i,j)=(1,1), so
# bisect that single column directly. Mirrors the 3-D form's column logic.
@inline function _fractional_indices((z,)::NTuple{1, Any},
                                      grid::PressureLevelGrid, ::Nothing, ::Nothing, ℓz)
    kk = column_fractional_z_index(z, 1, 1, grid)
    return FractionalIndices(nothing, nothing, kk)
end

@inline function column_fractional_z_index(z, ii, jj, grid)
    i = clamp(Base.unsafe_trunc(Int, ii), 1, grid.Nx)
    j = clamp(Base.unsafe_trunc(Int, jj), 1, grid.Ny)
    column = ColumnView(grid, i, j)
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
    fts = geopotential.time_series
    for t in 1:length(fts.times)
        clip_subsurface!(fts[t], surface_geopotential)
    end
    return geopotential
end
