#####
##### Apply a roughness closure over a grid: sample the per-cell surface `properties`, hand
##### each cell to `aerodynamic_parameters(closure, cell)`, and write the momentum roughness
##### length z0 and zero-plane displacement d0. The `DragPartitionRoughness` method below runs
##### the drag partition on the cell's canopy height and area index using the closure's
##### vegetation-type parameters.
#####

const MAXIMUM_VALID_LAI = 10       # physical LAI ceiling; larger values are fill/artifacts

"""
$(TYPEDSIGNATURES)

Momentum roughness length `z0` and zero-plane displacement `d0` (meters) for one cell under
the drag-partition canopy closure. `cell` carries `lai` and `canopy_height`; the closure's
vegetation-type drag parameters set `(z0, d0)` from them. A non-finite or out-of-range `lai`,
or a non-finite/negative `canopy_height`, returns `NaN` gaps.
"""
@inline function aerodynamic_parameters(closure::DragPartitionRoughness{FT}, cell) where FT
    Λ₀ = convert(FT, cell.lai)
    h₀ = convert(FT, cell.canopy_height)

    valid = isfinite(Λ₀) & (Λ₀ ≥ 0) & (Λ₀ ≤ MAXIMUM_VALID_LAI) & isfinite(h₀) & (h₀ ≥ 0)
    Λ = ifelse(valid, Λ₀, zero(FT))
    h = ifelse(valid, h₀, zero(FT))

    z0, d0 = canopy_roughness(Λ, h, closure.parameters, closure.von_karman_constant,
                              closure.sublayer_influence, closure.iterations)
    gap = convert(FT, NaN)
    return ifelse(valid, z0, gap), ifelse(valid, d0, gap)
end

@inline (closure::DragPartitionRoughness)(cell) = aerodynamic_parameters(closure, cell)

@kernel function _compute_aerodynamic_roughness!(z0m, d0, closure, lai, canopy_height)
    i, j = @index(Global, NTuple)
    cell = (lai           = property_value(lai, i, j),
            canopy_height = property_value(canopy_height, i, j))
    z0ij, d0ij = aerodynamic_parameters(closure, cell)
    @inbounds z0m[i, j, 1] = z0ij
    @inbounds d0[i, j, 1]  = d0ij
end

"""
$(TYPEDSIGNATURES)

Fill `z0m` and `d0` (meters) in place by applying `closure` over `grid`, reading the per-cell
surface fields from the `properties` NamedTuple (each a scalar, array or `Field`). For
[`DragPartitionRoughness`](@ref) the properties are `(; lai, canopy_height)`; `canopy_height`
may be omitted, in which case the closure's representative canopy height fills every cell.
"""
function compute_aerodynamic_roughness!(z0m, d0, closure::DragPartitionRoughness, properties, grid)
    arch = architecture(grid)
    canopy_height = get(properties, :canopy_height, closure.representative_height)
    launch!(arch, grid, :xy, _compute_aerodynamic_roughness!,
            z0m, d0, closure, properties.lai, canopy_height)
    return z0m, d0
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length `z0` and zero-plane displacement `d0` (meters) from a
`DragPartitionRoughness` `closure` applied to a leaf-area index `lai`, returning output that
matches the shape of `lai`:

  - `lai::Number` → a scalar `(z0, d0)` pair;
  - `lai::AbstractField` → a `(z0, d0)` pair of `Field`s on `lai`'s grid;
  - `lai::FieldTimeSeries` → a `(z0, d0)` pair of `FieldTimeSeries` (a cyclic climatology,
    sharing `lai`'s grid and times).

`canopy_height` drives the height and defaults to the closure's representative canopy height;
pass a `Number` or a `Field` to override it. A non-finite or out-of-range `lai`, or a
non-finite/negative `canopy_height`, yields `NaN` gaps.

```jldoctest
julia> using NumericalEarth.Lands

julia> closure = DragPartitionRoughness(Float64; vegetation_type = :evergreen_broadleaf_forest);

julia> z0, d0 = canopy_roughness(closure, 6.0, 24.72);

julia> round.((z0, d0), digits = 2)
(1.22, 21.05)
```
"""
@inline canopy_roughness(closure::DragPartitionRoughness, lai::Number,
                         canopy_height::Number = closure.representative_height) =
    aerodynamic_parameters(closure, (; lai, canopy_height))

function canopy_roughness(closure::DragPartitionRoughness, lai::AbstractField,
                          canopy_height = closure.representative_height)
    grid = lai.grid
    z0 = Field{Center, Center, Nothing}(grid)
    d0 = Field{Center, Center, Nothing}(grid)
    compute_aerodynamic_roughness!(z0, d0, closure, (; lai, canopy_height), grid)
    return z0, d0
end

function canopy_roughness(closure::DragPartitionRoughness, lai::FieldTimeSeries,
                          canopy_height = closure.representative_height)
    grid  = lai.grid
    times = lai.times
    z0 = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    d0 = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    for n in eachindex(times)
        compute_aerodynamic_roughness!(z0[n], d0[n], closure, (; lai = lai[n], canopy_height), grid)
    end
    return z0, d0
end

"""
$(TYPEDSIGNATURES)

Cyclic climatology of momentum roughness length `z0m` and zero-plane displacement `d0` from a
leaf-area-index `FieldTimeSeries` `lai` (one seasonal cycle of periods). A convenience wrapper
that builds a [`DragPartitionRoughness`](@ref) for `vegetation_type` (default
`:evergreen_broadleaf_forest`) — forwarding `von_karman_constant`, `sublayer_influence`, and
`iterations` — and applies it to `lai` through [`canopy_roughness`](@ref). Pass a static
`canopy_height` to drive the height, or leave it `nothing` for the class's representative
height. Returns `(z0m, d0)` as `FieldTimeSeries`.
"""
function canopy_roughness_climatology(lai::FieldTimeSeries, canopy_height = nothing;
                                      vegetation_type = :evergreen_broadleaf_forest, kw...)
    closure = DragPartitionRoughness(eltype(lai.grid); vegetation_type, kw...)
    return isnothing(canopy_height) ? canopy_roughness(closure, lai) :
                                      canopy_roughness(closure, lai, canopy_height)
end
