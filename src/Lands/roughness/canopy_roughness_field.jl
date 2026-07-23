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

Momentum roughness length `z0` and zero-plane displacement `d0` (metres) for one cell under
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

Fill `z0m` and `d0` (metres) in place by applying `closure` over `grid`, reading the per-cell
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

Build cyclic climatologies of momentum roughness length `z0m` and zero-plane displacement
`d0` from a leaf-area-index `FieldTimeSeries` `lai` (one seasonal cycle of periods), using the
drag parameters of `vegetation_type` (default `:evergreen_broadleaf_forest`). Pass a static
`canopy_height` to drive the height, or leave it `nothing` to use the class's representative
height. Returns `(z0m, d0)` as `FieldTimeSeries` sharing `lai`'s grid and times.
"""
function canopy_roughness_climatology(lai::FieldTimeSeries, canopy_height = nothing;
                                      vegetation_type = :evergreen_broadleaf_forest,
                                      von_karman_constant = VON_KARMAN_CONSTANT,
                                      sublayer_influence = SUBLAYER_INFLUENCE,
                                      iterations = CLOSURE_ITERATIONS)
    grid    = lai.grid
    times   = lai.times
    closure = DragPartitionRoughness(eltype(grid); vegetation_type, von_karman_constant,
                                     sublayer_influence, iterations)

    z0m = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    d0  = FieldTimeSeries{Center, Center, Nothing}(grid, times)

    for n in eachindex(times)
        properties = isnothing(canopy_height) ? (; lai = lai[n]) : (; lai = lai[n], canopy_height)
        compute_aerodynamic_roughness!(z0m[n], d0[n], closure, properties, grid)
    end

    return z0m, d0
end
