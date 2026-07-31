#####
##### Grid builder for the urban closures: sample each cell's inputs from the `properties`
##### NamedTuple (scalar / array / Field), assemble the `cell`, and evaluate the closure to
##### the aerodynamic parameters (ℓᵐ, d). Shares the `compute_aerodynamic_roughness!` /
##### `aerodynamic_parameters(closure, cell)` contract with the canopy roughness closures.
#####

@kernel function _compute_urban_aerodynamic_roughness!(ℓᵐ, d, closure, plan_area_fraction, building_height, grid)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)
    λᵖ = convert(FT, property_value(plan_area_fraction, i, j))
    h  = convert(FT, property_value(building_height, i, j))
    φ  = φnode(i, j, 1, grid, Center(), Center(), Center())
    cell = (plan_area_fraction = λᵖ, building_height = h, latitude = φ)
    ℓᵐᵢⱼ, dᵢⱼ = aerodynamic_parameters(closure, cell)
    @inbounds ℓᵐ[i, j, 1] = ℓᵐᵢⱼ
    @inbounds d[i, j, 1] = dᵢⱼ
end

"""
$(TYPEDSIGNATURES)

Fill the momentum roughness length `ℓᵐ` and zero-plane displacement `d` (meters) in place
by applying an urban `closure` ([`AbstractUrbanRoughness`](@ref)) over every cell of `grid`.
`properties` is a NamedTuple of the closure's per-cell inputs — scalars, arrays or `Field`s —
read with `property_value`; the urban closures expect `plan_area_fraction` and
`building_height`. Shared entry point with the canopy roughness closures.
"""
function compute_aerodynamic_roughness!(ℓᵐ, d, closure::AbstractUrbanRoughness, properties, grid)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _compute_urban_aerodynamic_roughness!,
            ℓᵐ, d, closure, properties.plan_area_fraction, properties.building_height, grid)
    return ℓᵐ, d
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length `ℓᵐ` and zero-plane displacement `d` (as `Field`s on the grid
of `h`) for the urban tile, from a mean building-height field `h` and a built-up plan-area
index field `λᵖ`. Convenience wrapper around [`compute_aerodynamic_roughness!`](@ref);
pass a `closure` (default [`MorphometricRoughness`](@ref)) to select the morphometry. Where
`λᵖ → 0` the result reduces to a bare-soil roughness.
"""
function urban_roughness(h, λᵖ; closure = MorphometricRoughness(eltype(h.grid)))
    grid = h.grid
    ℓᵐ = Field{Center, Center, Nothing}(grid)
    d  = Field{Center, Center, Nothing}(grid)
    compute_aerodynamic_roughness!(ℓᵐ, d, closure, (; plan_area_fraction = λᵖ, building_height = h), grid)
    return ℓᵐ, d
end
