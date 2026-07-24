#####
##### Apply a roughness closure over a grid: sample the per-cell surface `properties`, hand
##### each cell to `aerodynamic_parameters(closure, cell)`, and write the momentum roughness
##### length ℓᵐ and zero-plane displacement d. The `DragPartitionRoughness` method below runs
##### the drag partition on the cell's canopy height and leaf area index using the closure's
##### vegetation-type parameters.
#####

"""
$(TYPEDSIGNATURES)

Momentum roughness length `ℓᵐ` and zero-plane displacement `d` (meters) for one cell under
the drag-partition canopy closure. `cell` carries `leaf_area_index` and `canopy_height`; the
closure's vegetation-type drag parameters set `(ℓᵐ, d)` from them. A non-finite or out-of-range
`leaf_area_index`, or a non-finite/negative `canopy_height`, returns `NaN` gaps.
"""
@inline function aerodynamic_parameters(closure::DragPartitionRoughness{FT}, cell) where FT
    𝒜 = convert(FT, cell.leaf_area_index)
    h = convert(FT, cell.canopy_height)

    valid = isfinite(𝒜) & (𝒜 ≥ 0) & (𝒜 ≤ closure.maximum_valid_leaf_area_index) &
            isfinite(h) & (h ≥ 0)

    ℓᵐ, d = canopy_roughness(ifelse(valid, 𝒜, zero(FT)), ifelse(valid, h, zero(FT)),
                             closure.parameters, closure.von_karman_constant,
                             closure.sublayer_influence, closure.iterations)
    gap = convert(FT, NaN)
    return ifelse(valid, ℓᵐ, gap), ifelse(valid, d, gap)
end

@inline (closure::DragPartitionRoughness)(cell) = aerodynamic_parameters(closure, cell)

@kernel function _compute_aerodynamic_roughness!(ℓᵐ, d, closure, leaf_area_index, canopy_height)
    i, j = @index(Global, NTuple)
    cell = (leaf_area_index = property_value(leaf_area_index, i, j),
            canopy_height   = property_value(canopy_height, i, j))
    ℓᵐij, dij = aerodynamic_parameters(closure, cell)
    @inbounds ℓᵐ[i, j, 1] = ℓᵐij
    @inbounds d[i, j, 1]  = dij
end

"""
$(TYPEDSIGNATURES)

Fill `ℓᵐ` and `d` (meters) in place by applying `closure` over `grid`, reading the per-cell
surface fields from the `properties` NamedTuple (each a scalar, array or `Field`). For
[`DragPartitionRoughness`](@ref) the properties are `(; leaf_area_index, canopy_height)`;
`canopy_height` may be omitted, in which case the closure's representative canopy height fills
every cell.
"""
function compute_aerodynamic_roughness!(ℓᵐ, d, closure::DragPartitionRoughness, properties, grid)
    arch = architecture(grid)
    canopy_height = get(properties, :canopy_height, closure.representative_height)
    launch!(arch, grid, :xy, _compute_aerodynamic_roughness!,
            ℓᵐ, d, closure, properties.leaf_area_index, canopy_height)
    return ℓᵐ, d
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length `ℓᵐ` and zero-plane displacement `d` (meters) from a
`DragPartitionRoughness` `closure` applied to a `leaf_area_index`, returning output that
matches the shape of `leaf_area_index`:

  - `leaf_area_index::Number` → a scalar `(ℓᵐ, d)` pair;
  - `leaf_area_index::AbstractField` → a `(ℓᵐ, d)` pair of `Field`s on its grid;
  - `leaf_area_index::FieldTimeSeries` → a `(ℓᵐ, d)` pair of `FieldTimeSeries` (a cyclic
    climatology, sharing its grid and times).

`canopy_height` drives the height and defaults to the closure's representative canopy height;
override it with a `Number`, a spatially-varying `Field`, or — with a
`leaf_area_index::FieldTimeSeries` — a `FieldTimeSeries` sharing its periods. A non-finite or
out-of-range `leaf_area_index`, or a non-finite/negative `canopy_height`, yields `NaN` gaps.

```jldoctest
julia> using NumericalEarth.Lands

julia> closure = DragPartitionRoughness(Float64; vegetation_type = :evergreen_broadleaf_forest);

julia> ℓᵐ, d = canopy_roughness(closure, 6.0, 24.72);

julia> round.((ℓᵐ, d), digits = 2)
(1.22, 21.05)
```
"""
@inline canopy_roughness(closure::DragPartitionRoughness, leaf_area_index::Number,
                         canopy_height::Number = closure.representative_height) =
    aerodynamic_parameters(closure, (; leaf_area_index, canopy_height))

function canopy_roughness(closure::DragPartitionRoughness, leaf_area_index::AbstractField,
                          canopy_height = closure.representative_height)
    grid = leaf_area_index.grid
    ℓᵐ = Field{Center, Center, Nothing}(grid)
    d  = Field{Center, Center, Nothing}(grid)
    compute_aerodynamic_roughness!(ℓᵐ, d, closure, (; leaf_area_index, canopy_height), grid)
    return ℓᵐ, d
end

# Canopy height for period `n`: index a FieldTimeSeries per period; share a scalar or static Field.
@inline canopy_height_at_period(canopy_height, n) = canopy_height
@inline canopy_height_at_period(canopy_height::FieldTimeSeries, n) = canopy_height[n]

function canopy_roughness(closure::DragPartitionRoughness, leaf_area_index::FieldTimeSeries,
                          canopy_height = closure.representative_height)
    grid  = leaf_area_index.grid
    times = leaf_area_index.times
    ℓᵐ = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    d  = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    for n in eachindex(times)
        hₙ = canopy_height_at_period(canopy_height, n)
        compute_aerodynamic_roughness!(ℓᵐ[n], d[n], closure,
                                       (; leaf_area_index = leaf_area_index[n], canopy_height = hₙ), grid)
    end
    return ℓᵐ, d
end

"""
$(TYPEDSIGNATURES)

Cyclic climatology of momentum roughness length `ℓᵐ` and zero-plane displacement `d` from a
`leaf_area_index` `FieldTimeSeries` (one seasonal cycle of periods). A convenience wrapper
that builds a [`DragPartitionRoughness`](@ref) for `vegetation_type` (default
`:evergreen_broadleaf_forest`) — forwarding `von_karman_constant`, `sublayer_influence`, and
`iterations` — and applies it through [`canopy_roughness`](@ref). Pass a static
`canopy_height` to drive the height, or leave it `nothing` for the class's representative
height. Returns `(ℓᵐ, d)` as `FieldTimeSeries`.
"""
function canopy_roughness_climatology(leaf_area_index::FieldTimeSeries, canopy_height = nothing;
                                     vegetation_type = :evergreen_broadleaf_forest, kw...)
    closure = DragPartitionRoughness(eltype(leaf_area_index.grid); vegetation_type, kw...)
    return isnothing(canopy_height) ? canopy_roughness(closure, leaf_area_index) :
                                      canopy_roughness(closure, leaf_area_index, canopy_height)
end
