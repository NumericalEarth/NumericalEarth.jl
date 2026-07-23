#####
##### Grid builder for the urban closures: sample each cell's inputs from the `properties`
##### NamedTuple (scalar / array / Field), assemble the `cell`, and evaluate the closure to
##### the aerodynamic parameters (z0, d0). Shares the `compute_aerodynamic_roughness!` /
##### `aerodynamic_parameters(closure, cell)` contract with the canopy roughness closures.
#####

@kernel function _compute_urban_aerodynamic_roughness!(z0m, d0, closure, plan_area_fraction, building_height, grid)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)
    λp = convert(FT, property_value(plan_area_fraction, i, j))
    h  = convert(FT, property_value(building_height, i, j))
    φ  = φnode(i, j, 1, grid, Center(), Center(), Center())
    cell = (plan_area_fraction = λp, building_height = h, latitude = φ)
    z0ij, d0ij = aerodynamic_parameters(closure, cell)
    @inbounds z0m[i, j, 1] = z0ij
    @inbounds d0[i, j, 1] = d0ij
end

"""
$(TYPEDSIGNATURES)

Fill `z0m` and `d0` (meters) in place by applying an urban `closure`
([`AbstractUrbanRoughness`](@ref)) over every cell of `grid`. `properties` is a NamedTuple
of the closure's per-cell inputs — scalars, arrays or `Field`s — read with `property_value`;
the urban closures expect `plan_area_fraction` and `building_height`. Shared entry point with
the canopy roughness closures.
"""
function compute_aerodynamic_roughness!(z0m, d0, closure::AbstractUrbanRoughness, properties, grid)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _compute_urban_aerodynamic_roughness!,
            z0m, d0, closure, properties.plan_area_fraction, properties.building_height, grid)
    return z0m, d0
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length `z0m` and zero-plane displacement `d0` (as `Field`s on the grid
of `H`) for the urban tile, from a mean building-height field `H` and a built-up plan-area
fraction field `λp`. Convenience wrapper around [`compute_aerodynamic_roughness!`](@ref);
pass a `closure` (default [`KandaRoughness`](@ref)) to select the morphometry. Where
`λp → 0` the result reduces to a bare-soil roughness.
"""
function urban_roughness(H, λp; closure = KandaRoughness(eltype(H.grid)))
    grid = H.grid
    z0m = Field{Center, Center, Nothing}(grid)
    d0  = Field{Center, Center, Nothing}(grid)
    compute_aerodynamic_roughness!(z0m, d0, closure, (; plan_area_fraction = λp, building_height = H), grid)
    return z0m, d0
end
