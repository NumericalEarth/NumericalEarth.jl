using ..EarthSystemModels: EarthSystemModels, BoundaryPrescribedComponent, relaxation_zone_width

"""
$(TYPEDSIGNATURES)

Return a [`BoundaryPrescribedComponent`](@ref NumericalEarth.EarthSystemModels.BoundaryPrescribedComponent)
whose strips are `Component`s holding `dataset` at `dates` along the lateral boundaries of `grid`.
Each strip spans its whole side and reaches `width` cells of `grid` inward, matching a Davies
relaxation zone of `width` cells; every strip is padded by `padding` degrees. Keyword arguments
`kw` go to each strip's `Component(region, dates, dataset; kw...)`, for example
`Component = PrescribedAtmosphere`.
"""
function EarthSystemModels.BoundaryPrescribedComponent(Component, grid::AbstractGrid, dates, dataset;
                                                       width,
                                                       padding = default_horizontal_padding(dataset),
                                                       sides = (:west, :east, :south, :north),
                                                       kw...)
    box = BoundingBox(grid)
    λ₁, λ₂ = box.longitude
    φ₁, φ₂ = box.latitude
    w = relaxation_zone_width(grid, width)
    λ = (λ₁ - padding, λ₂ + padding)
    φ = (φ₁ - padding, φ₂ + padding)

    regions = (west  = BoundingBox(longitude = (λ₁ - padding, λ₁ + w + padding), latitude = φ),
               east  = BoundingBox(longitude = (λ₂ - w - padding, λ₂ + padding), latitude = φ),
               south = BoundingBox(longitude = λ, latitude = (φ₁ - padding, φ₁ + w + padding)),
               north = BoundingBox(longitude = λ, latitude = (φ₂ - w - padding, φ₂ + padding)))

    strips = NamedTuple{sides}(Component(regions[side], dates, dataset; architecture = architecture(grid), kw...)
                               for side in sides)

    return BoundaryPrescribedComponent(; strips...)
end
