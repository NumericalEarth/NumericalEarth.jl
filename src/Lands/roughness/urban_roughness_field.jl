#####
##### On-grid evaluation of the urban roughness closures: (λᵖ, h) → (ℓᵐ, d) fields.
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
`properties` is a NamedTuple of the closure's per-cell inputs — scalars, arrays or `Field`s;
the urban closures expect `plan_area_fraction` and `building_height`.
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
of `h`) from a mean building-height field `h` and a built-up plan-area index field `λᵖ`,
under `closure` (default [`MorphometricRoughness`](@ref)). Where `λᵖ → 0` the result
reduces to a bare-soil roughness.
"""
function urban_roughness(h, λᵖ; closure = MorphometricRoughness(eltype(h.grid)))
    grid = h.grid
    ℓᵐ = Field{Center, Center, Nothing}(grid)
    d  = Field{Center, Center, Nothing}(grid)
    compute_aerodynamic_roughness!(ℓᵐ, d, closure, (; plan_area_fraction = λᵖ, building_height = h), grid)
    return ℓᵐ, d
end

#####
##### Measured-morphometry builder: per-cell σʰ, hᵐᵃˣ and λᶠ from a footprint-level dataset
##### instead of the closure's regressions and frontal-area estimator.
#####

@kernel function _compute_measured_urban_aerodynamic_roughness!(ℓᵐ, d, closure,
                                                                plan_area_fraction, building_height,
                                                                height_deviation, maximum_height,
                                                                frontal_area, grid)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)
    λᵖ   = convert(FT, property_value(plan_area_fraction, i, j))
    h    = convert(FT, property_value(building_height, i, j))
    σʰ   = convert(FT, property_value(height_deviation, i, j))
    hᵐᵃˣ = convert(FT, property_value(maximum_height, i, j))
    λᶠ   = convert(FT, property_value(frontal_area, i, j))
    ℓᵐᵢⱼ, dᵢⱼ = aerodynamic_parameters(closure, λᵖ, h, σʰ, hᵐᵃˣ, λᶠ)
    @inbounds ℓᵐ[i, j, 1] = ℓᵐᵢⱼ
    @inbounds d[i, j, 1] = dᵢⱼ
end

const MeasuredMorphometryProperties = NamedTuple{(:plan_area_fraction, :building_height,
                                                  :height_deviation, :maximum_height, :frontal_area_index)}

# Selected over the estimator-based method by the richer `properties` NamedTuple.
function compute_aerodynamic_roughness!(ℓᵐ, d, closure::AbstractUrbanRoughness,
                                        properties::MeasuredMorphometryProperties, grid)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _compute_measured_urban_aerodynamic_roughness!,
            ℓᵐ, d, closure, properties.plan_area_fraction, properties.building_height,
            properties.height_deviation, properties.maximum_height, properties.frontal_area_index, grid)
    return ℓᵐ, d
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length `ℓᵐ` and zero-plane displacement `d` (as `Field`s on the grid of
`h`) from **measured** per-cell morphometry: the mean building height `h`, built-up
plan-area index `λᵖ`, height standard deviation `σʰ`, maximum height `hᵐᵃˣ` and
frontal-area index `λᶠ` — the fields a footprint-level dataset such as
`GlobalBuildingFootprints3D` aggregates. The closure (default
[`MorphometricRoughness`](@ref)) is fed the measured height heterogeneity in place of its
frontal-area estimator and `σʰ`/`hᵐᵃˣ` regressions. Where `λᵖ → 0` the result reduces to
a bare-soil roughness.
"""
function urban_roughness(h, λᵖ, σʰ, hᵐᵃˣ, λᶠ; closure = MorphometricRoughness(eltype(h.grid)))
    grid = h.grid
    ℓᵐ = Field{Center, Center, Nothing}(grid)
    d  = Field{Center, Center, Nothing}(grid)
    properties = (; plan_area_fraction = λᵖ, building_height = h,
                    height_deviation = σʰ, maximum_height = hᵐᵃˣ, frontal_area_index = λᶠ)
    compute_aerodynamic_roughness!(ℓᵐ, d, closure, properties, grid)
    return ℓᵐ, d
end
