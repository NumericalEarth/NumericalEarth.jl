using ..EarthSystemModels: EarthSystemModels, BoundaryPrescribedComponent, relaxation_zone_width

"""
$(TYPEDSIGNATURES)

Return a [`BoundaryPrescribedComponent`](@ref NumericalEarth.EarthSystemModels.BoundaryPrescribedComponent)
whose strips are `Component`s holding `dataset` at `dates` along the lateral boundaries of `grid`.
Each strip spans its whole side and reaches `width` cells of `grid` inward, matching a lateral boundary
relaxation zone of `width` cells; every strip is padded by `padding` degrees, a number or a
`(longitude, latitude)` pair. Keyword arguments `kw` go to each strip's
`Component(region, dates, dataset; kw...)`, for example `Component = PrescribedAtmosphere`.
"""
function EarthSystemModels.BoundaryPrescribedComponent(Component, grid::AbstractGrid, dates, dataset;
                                                       width,
                                                       padding = default_horizontal_padding(dataset),
                                                       sides = (:west, :east, :south, :north),
                                                       kw...)
    regions = boundary_strip_regions(grid, width, padding)
    strips = map(region -> Component(region, dates, dataset; architecture = architecture(grid), kw...), regions[sides])

    return BoundaryPrescribedComponent(; strips...)
end

function boundary_strip_regions(grid, width, padding)
    box = BoundingBox(grid)
    λ₁, λ₂ = box.longitude
    φ₁, φ₂ = box.latitude
    δλ, δφ = axis_padding(padding)
    w = relaxation_zone_width(grid, width)
    λ = (λ₁ - δλ, λ₂ + δλ)
    φ = (φ₁ - δφ, φ₂ + δφ)

    return (west  = BoundingBox(longitude = (λ₁ - δλ, λ₁ + w + δλ), latitude = φ),
            east  = BoundingBox(longitude = (λ₂ - w - δλ, λ₂ + δλ), latitude = φ),
            south = BoundingBox(longitude = λ, latitude = (φ₁ - δφ, φ₁ + w + δφ)),
            north = BoundingBox(longitude = λ, latitude = (φ₂ - w - δφ, φ₂ + δφ)))
end

# The sides of a curvilinear grid are not lines of longitude or latitude: each strip bounds the corners of its band.
function boundary_strip_regions(grid::OrthogonalSphericalShellGrid, width, padding)
    Nx, Ny, _ = size(grid)
    return (west  = BoundingBox(grid; padding, i = 1:width+1),
            east  = BoundingBox(grid; padding, i = Nx-width+1:Nx+1),
            south = BoundingBox(grid; padding, j = 1:width+1),
            north = BoundingBox(grid; padding, j = Ny-width+1:Ny+1))
end
