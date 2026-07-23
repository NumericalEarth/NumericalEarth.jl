#####
##### Apply a roughness closure over a grid: sample the per-cell surface `properties`, hand
##### each cell to `aerodynamic_parameters(closure, cell)`, and write the momentum roughness
##### length z0 and zero-plane displacement d0. The `DragPartitionRoughness` method below is
##### the canopy closure: it selects the drag group from the IGBP land cover, uses the
##### measured canopy height where valid and the class-average height otherwise, and returns
##### the prescribed constants over non-vegetated classes.
#####

const MAXIMUM_VALID_LAI = 10       # physical LAI ceiling; larger values are fill/artifacts

"""
$(TYPEDSIGNATURES)

Momentum roughness length `z0` and zero-plane displacement `d0` (metres) for one cell under
the drag-partition canopy closure. `cell` carries `land_cover` (IGBP class), `lai`,
`canopy_height` and `latitude`. The IGBP class selects the vegetation drag group (woody
savanna splits on latitude); the measured `canopy_height` is used where finite and positive,
the class-average height otherwise. Non-vegetated classes take the prescribed constants, and
vegetated cells with no valid LAI (or cells with no valid land cover) return `NaN` gaps.
"""
@inline function aerodynamic_parameters(closure::DragPartitionRoughness{FT}, cell) where FT
    cover = convert(FT, cell.land_cover)
    Λ₀    = convert(FT, cell.lai)
    h₀    = convert(FT, cell.canopy_height)
    φ     = cell.latitude

    valid_cover = isfinite(cover) & (cover ≥ 1) & (cover ≤ 17)
    safe_cover  = ifelse(valid_cover, cover, one(cover))  # finite valid class before round (ifelse eager)
    igbp = round(Int, safe_cover)

    valid_lai = isfinite(Λ₀) & (Λ₀ ≥ 0) & (Λ₀ ≤ MAXIMUM_VALID_LAI)
    Λ = ifelse(valid_lai, Λ₀, zero(FT))

    vegetated = is_vegetated(igbp)
    group = drag_partition_group(igbp, φ)
    p = canopy_drag_parameters(FT, max(group, 1))       # group=0 (non-veg) discarded below

    valid_h = isfinite(h₀) & (h₀ > 0)
    h = ifelse(valid_h, h₀, class_canopy_height(FT, igbp))

    z0ᵛ, d0ᵛ = canopy_roughness(Λ, h, p, closure.von_karman_constant,
                                closure.sublayer_influence, closure.iterations)
    z0ⁿ, d0ⁿ = nonvegetated_roughness(FT, igbp)

    # Vegetated cells with no valid LAI retrieval — and cells with no valid land cover —
    # become honest gaps (NaN); non-vegetated cells take their prescribed constants.
    gap = convert(FT, NaN)
    z0 = ifelse(valid_cover, ifelse(vegetated, ifelse(valid_lai, z0ᵛ, gap), z0ⁿ), gap)
    d0 = ifelse(valid_cover, ifelse(vegetated, ifelse(valid_lai, d0ᵛ, gap), d0ⁿ), gap)
    return z0, d0
end

@inline (closure::DragPartitionRoughness)(cell) = aerodynamic_parameters(closure, cell)

@kernel function _compute_aerodynamic_roughness!(z0m, d0, closure, land_cover, lai, canopy_height, grid)
    i, j = @index(Global, NTuple)
    φ = φnode(i, j, 1, grid, Center(), Center(), Center())
    cell = (land_cover    = property_value(land_cover, i, j),
            lai           = property_value(lai, i, j),
            canopy_height = property_value(canopy_height, i, j),
            latitude      = φ)
    z0ij, d0ij = aerodynamic_parameters(closure, cell)
    @inbounds z0m[i, j, 1] = z0ij
    @inbounds d0[i, j, 1]  = d0ij
end

"""
$(TYPEDSIGNATURES)

Fill `z0m` and `d0` (metres) in place by applying `closure` over `grid`, reading the per-cell
surface fields from the `properties` NamedTuple (each a scalar, array or `Field`). For
[`DragPartitionRoughness`](@ref) the properties are `(; land_cover, lai, canopy_height)`;
`canopy_height` may be omitted (or `nothing`), in which case the class-average height fills
wherever the measured height is missing or non-positive. The builder injects the grid latitude
into each cell.
"""
function compute_aerodynamic_roughness!(z0m, d0, closure::DragPartitionRoughness, properties, grid)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _compute_aerodynamic_roughness!,
            z0m, d0, closure, properties.land_cover, properties.lai,
            get(properties, :canopy_height, nothing), grid)
    return z0m, d0
end

"""
$(TYPEDSIGNATURES)

Build cyclic climatologies of momentum roughness length `z0m` and zero-plane displacement
`d0` from a leaf-area-index `FieldTimeSeries` `lai` (one seasonal cycle of periods), a static
IGBP `land_cover` field, and a static `canopy_height` (defaults to the class heights).
Returns `(z0m, d0)` as `FieldTimeSeries` sharing `lai`'s grid and times.
"""
function canopy_roughness_climatology(lai::FieldTimeSeries, land_cover, canopy_height = nothing;
                                      von_karman_constant = VON_KARMAN_CONSTANT,
                                      sublayer_influence = SUBLAYER_INFLUENCE,
                                      iterations = CLOSURE_ITERATIONS)
    grid    = lai.grid
    times   = lai.times
    closure = DragPartitionRoughness(eltype(grid); von_karman_constant, sublayer_influence, iterations)

    z0m = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    d0  = FieldTimeSeries{Center, Center, Nothing}(grid, times)

    for n in eachindex(times)
        compute_aerodynamic_roughness!(z0m[n], d0[n], closure,
                                       (; land_cover, lai = lai[n], canopy_height), grid)
    end

    return z0m, d0
end
