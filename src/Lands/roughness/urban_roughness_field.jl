#####
##### On-grid evaluation of the urban roughness closures: (λᵖ, h) → (ℓᵐ, d) fields.
#####

@kernel function _compute_urban_aerodynamic_roughness!(ℓᵐ, d, closure, plan_area_index, mean_building_height, grid)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)
    λᵖ = convert(FT, property_value(plan_area_index, i, j))
    h  = convert(FT, property_value(mean_building_height, i, j))
    φ  = φnode(i, j, 1, grid, Center(), Center(), Center())
    cell = (plan_area_index = λᵖ, mean_building_height = h, latitude = φ)
    ℓᵐᵢⱼ, dᵢⱼ = aerodynamic_parameters(closure, cell)
    @inbounds ℓᵐ[i, j, 1] = ℓᵐᵢⱼ
    @inbounds d[i, j, 1] = dᵢⱼ
end

"""
$(TYPEDSIGNATURES)

Fill the momentum roughness length `ℓᵐ` and zero-plane displacement `d` (meters) in place
by applying an urban `closure` ([`AbstractUrbanRoughness`](@ref)) over every cell of `grid`.
`properties` is a NamedTuple of the closure's per-cell inputs — scalars, arrays or `Field`s;
the urban closures expect `plan_area_index` and `mean_building_height`.
"""
function compute_aerodynamic_roughness!(ℓᵐ, d, closure::AbstractUrbanRoughness, properties, grid)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _compute_urban_aerodynamic_roughness!,
            ℓᵐ, d, closure, properties.plan_area_index, properties.mean_building_height, grid)
    return ℓᵐ, d
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length `ℓᵐ` and zero-plane displacement `d` (as `Field`s on the grid
of `h`) from a mean building-height field `h` and a plan-area index field `λᵖ`,
under `closure` (default [`MorphometricRoughness`](@ref)). Where `λᵖ → 0` the result
reduces to a bare-soil roughness, and cells of invalid morphometry are `NaN`; fill them
with [`fill_aerodynamic_roughness_gaps!`](@ref) before passing the pair to a flux closure.
"""
function urban_roughness(h, λᵖ; closure = MorphometricRoughness(eltype(h.grid)))
    grid = h.grid
    ℓᵐ = Field{Center, Center, Nothing}(grid)
    d  = Field{Center, Center, Nothing}(grid)
    compute_aerodynamic_roughness!(ℓᵐ, d, closure, (; plan_area_index = λᵖ, mean_building_height = h), grid)
    return ℓᵐ, d
end

#####
##### Gap filling
#####

# A similarity profile needs a finite, positive roughness length.
@inline roughness_gap(ℓᵐ, d) = !(isfinite(ℓᵐ) & (ℓᵐ > 0)) | !isfinite(d)

@kernel function _fill_aerodynamic_roughness_gaps!(ℓᵐ, d, ℓᵘ, dᵘ)
    i, j = @index(Global, NTuple)
    @inbounds begin
        ℓᵐᵢⱼ = ℓᵐ[i, j, 1]
        dᵢⱼ  = d[i, j, 1]
        gap  = roughness_gap(ℓᵐᵢⱼ, dᵢⱼ)
        ℓᵐ[i, j, 1] = ifelse(gap, ℓᵘ, ℓᵐᵢⱼ)
        d[i, j, 1]  = ifelse(gap, dᵘ, dᵢⱼ)
    end
end

"""
$(TYPEDSIGNATURES)

Replace, in place, the gaps left where `closure` could not evaluate a cell with the
`unbuilt` pair `(ℓ, d)`, by default the closure's own bare-soil endpoint
`aerodynamic_parameters(closure, 0, 0)`. A gap is a non-finite `ℓᵐ` or `d`, or a
non-positive `ℓᵐ`, which a similarity profile cannot take either. The roughness length `ℓᵐ`
and zero-plane displacement `d` are filled together, so a gap in either becomes unbuilt
surface in both.

GHSL publishes no tile for the all-ocean cells of its Mollweide grid, so the offshore part
of a coastal window comes back as gaps, as does any no-data pixel inside a published tile.
The fill makes those cells evaluable and no more: bare soil (`ℓᵐ = 0.03` m by default) is
two orders of magnitude rougher than a water surface, and the atmosphere-land flux closure
applies land similarity theory wherever the land component runs. Where the gaps are known
to be open water, pass a water roughness such as `unbuilt = (1e-4, 0)`, or keep the land
component off those cells. For gaps that fall inside built-up land, inpaint `λᵖ` and `h`
with `DataWrangling.inpaint_mask!` before evaluating the closure instead, so the fill comes
from horizontal neighbors rather than from a constant.
"""
function fill_aerodynamic_roughness_gaps!(ℓᵐ, d, closure::AbstractUrbanRoughness;
                                          unbuilt = aerodynamic_parameters(closure, 0, 0))
    ℓᵘ, dᵘ = unbuilt
    ℓᵘ > 0 || throw(ArgumentError("the unbuilt roughness length must be positive, got $ℓᵘ"))
    dᵘ >= 0 || throw(ArgumentError("the unbuilt zero-plane displacement must be non-negative, got $dᵘ"))

    grid = ℓᵐ.grid
    FT = eltype(ℓᵐ)
    launch!(architecture(grid), grid, :xy, _fill_aerodynamic_roughness_gaps!,
            ℓᵐ, d, convert(FT, ℓᵘ), convert(FT, dᵘ))
    return ℓᵐ, d
end

#####
##### Measured-morphometry builder: per-cell σʰ, hᵐᵃˣ and λᶠ from a footprint-level dataset
##### instead of the closure's regressions and frontal-area estimator.
#####

@kernel function _compute_measured_urban_aerodynamic_roughness!(ℓᵐ, d, closure,
                                                                plan_area_index, mean_building_height,
                                                                building_height_deviation, maximum_building_height,
                                                                frontal_area, grid)
    i, j = @index(Global, NTuple)
    FT = eltype(grid)
    λᵖ   = convert(FT, property_value(plan_area_index, i, j))
    h    = convert(FT, property_value(mean_building_height, i, j))
    σʰ   = convert(FT, property_value(building_height_deviation, i, j))
    hᵐᵃˣ = convert(FT, property_value(maximum_building_height, i, j))
    λᶠ   = convert(FT, property_value(frontal_area, i, j))
    ℓᵐᵢⱼ, dᵢⱼ = aerodynamic_parameters(closure, λᵖ, h, σʰ, hᵐᵃˣ, λᶠ)
    @inbounds ℓᵐ[i, j, 1] = ℓᵐᵢⱼ
    @inbounds d[i, j, 1] = dᵢⱼ
end

const MeasuredMorphometryProperties = NamedTuple{(:plan_area_index, :mean_building_height,
                                                  :building_height_deviation, :maximum_building_height,
                                                  :frontal_area_index)}

# Selected over the estimator-based method by the richer `properties` NamedTuple.
function compute_aerodynamic_roughness!(ℓᵐ, d, closure::AbstractUrbanRoughness,
                                        properties::MeasuredMorphometryProperties, grid)
    arch = architecture(grid)
    launch!(arch, grid, :xy, _compute_measured_urban_aerodynamic_roughness!,
            ℓᵐ, d, closure, properties.plan_area_index, properties.mean_building_height,
            properties.building_height_deviation, properties.maximum_building_height,
            properties.frontal_area_index, grid)
    return ℓᵐ, d
end

"""
$(TYPEDSIGNATURES)

Momentum roughness length `ℓᵐ` and zero-plane displacement `d` (as `Field`s on the grid of
`h`) from **measured** per-cell morphometry: the mean building height `h`, plan-area index
`λᵖ`, height standard deviation `σʰ`, maximum height `hᵐᵃˣ` and frontal-area index `λᶠ` —
the fields a footprint-level dataset such as `GlobalBuildingFootprints3D` aggregates. The
closure (default [`MorphometricRoughness`](@ref)) is fed the measured height heterogeneity
in place of its frontal-area estimator and `σʰ`/`hᵐᵃˣ` regressions. Where `λᵖ → 0` the
result reduces to a bare-soil roughness. Cells the closure cannot evaluate are `NaN`; fill
them with [`fill_aerodynamic_roughness_gaps!`](@ref) before passing the pair to a flux closure.
"""
function urban_roughness(h, λᵖ, σʰ, hᵐᵃˣ, λᶠ; closure = MorphometricRoughness(eltype(h.grid)))
    grid = h.grid
    ℓᵐ = Field{Center, Center, Nothing}(grid)
    d  = Field{Center, Center, Nothing}(grid)
    properties = (; plan_area_index = λᵖ, mean_building_height = h,
                    building_height_deviation = σʰ, maximum_building_height = hᵐᵃˣ,
                    frontal_area_index = λᶠ)
    compute_aerodynamic_roughness!(ℓᵐ, d, closure, properties, grid)
    return ℓᵐ, d
end
