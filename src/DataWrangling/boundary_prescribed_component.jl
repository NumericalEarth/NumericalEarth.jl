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
    regions = boundary_strip_boxes(grid, width, padding)
    strips = map(region -> Component(region, dates, dataset; architecture = architecture(grid), kw...), regions[sides])

    return BoundaryPrescribedComponent(; strips...)
end

function boundary_strip_boxes(grid, width, padding)
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

# The sides of a curvilinear grid need not follow meridians and parallels, so each strip spans the
# nodes of the columns or rows it covers.
function boundary_strip_boxes(grid::OrthogonalSphericalShellGrid, width, padding)
    box = BoundingBox(grid)
    Nx, Ny, _ = size(grid)
    w = relaxation_zone_width(grid, width)
    Wx = ceil(Int, w * Nx / (box.longitude[2] - box.longitude[1]))
    Wy = ceil(Int, w * Ny / (box.latitude[2] - box.latitude[1]))

    return (west  = BoundingBox(grid, 1:Wx+1, 1:Ny+1; padding),
            east  = BoundingBox(grid, Nx-Wx+1:Nx+1, 1:Ny+1; padding),
            south = BoundingBox(grid, 1:Nx+1, 1:Wy+1; padding),
            north = BoundingBox(grid, 1:Nx+1, Ny-Wy+1:Ny+1; padding))
end
