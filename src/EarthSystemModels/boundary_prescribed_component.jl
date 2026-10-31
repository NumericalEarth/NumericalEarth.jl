using Oceananigans.Grids: AbstractGrid
using ..DataWrangling: BoundingBox, default_horizontal_padding, axis_padding

#####
##### A prescribed component known only on strips along the lateral boundaries of a limited-area domain
#####

struct BoundaryPrescribedComponent{W, E, S, N, C}
    west  :: W
    east  :: E
    south :: S
    north :: N
    clock :: C
end

"""
$(TYPEDSIGNATURES)

A prescribed component, such as a `PrescribedAtmosphere`, known only on strips along the lateral
boundaries of a limited-area domain. Each of `west`, `east`, `south` and `north` is a prescribed
component whose region covers that side's boundary and relaxation zone, or `nothing` for a side
without one.
"""
function BoundaryPrescribedComponent(; west=nothing, east=nothing, south=nothing, north=nothing)
    first_strip = something(west, east, south, north)
    clock = Clock(time = first_strip.clock.time)
    return BoundaryPrescribedComponent(west, east, south, north, clock)
end

"""
$(TYPEDSIGNATURES)

The strips of `component`, as a `NamedTuple` keyed by side.
"""
function boundary_strips(component::BoundaryPrescribedComponent)
    sides = filter(side -> !isnothing(getproperty(component, side)), (:west, :east, :south, :north))
    return NamedTuple{sides}(map(side -> getproperty(component, side), sides))
end

# Width of a boundary relaxation zone spanning `width` cells along the coarser horizontal direction
# of `grid`, in the `grid`’s native coordinates.
function relaxation_zone_width(grid, width)
    λ₁, λ₂ = x_domain(grid)
    φ₁, φ₂ = y_domain(grid)
    Nx, Ny, _ = size(grid)
    return width * max((λ₂ - λ₁) / Nx, (φ₂ - φ₁) / Ny)
end

function Oceananigans.TimeSteppers.time_step!(component::BoundaryPrescribedComponent, Δt)
    tick!(component.clock, Δt)
    foreach(strip -> time_step!(strip, Δt), boundary_strips(component))
    return nothing
end

Base.summary(component::BoundaryPrescribedComponent) =
    string("BoundaryPrescribedComponent with sides ", join(keys(boundary_strips(component)), ", "))

function Base.show(io::IO, component::BoundaryPrescribedComponent)
    strips = boundary_strips(component)
    print(io, summary(component), ":")
    for (n, side) in enumerate(keys(strips))
        prefix = n == length(strips) ? "└── " : "├── "
        print(io, '\n', prefix, side, ": ", summary(strips[side]))
    end
end

#####
##### Boundary strips built from a dataset
#####

"""
$(TYPEDSIGNATURES)

Return a [`BoundaryPrescribedComponent`](@ref)
whose strips are `Component`s holding `dataset` at `dates` along the lateral boundaries of `grid`.
Each strip spans its whole side and reaches `width` cells of `grid` inward, matching a lateral boundary
relaxation zone of `width` cells; every strip is padded by `padding` degrees, a number or a
`(longitude, latitude)` pair. Keyword arguments `kw` go to each strip's
`Component(region, dates, dataset; kw...)`, for example `Component = PrescribedAtmosphere`.
"""
function BoundaryPrescribedComponent(Component, grid::AbstractGrid, dates, dataset;
                                                       width,
                                                       padding = default_horizontal_padding(dataset),
                                                       sides = (:west, :east, :south, :north),
                                                       kw...)
    box = BoundingBox(grid)
    λ₁, λ₂ = box.longitude
    φ₁, φ₂ = box.latitude
    δλ, δφ = axis_padding(padding)
    w = relaxation_zone_width(grid, width)
    λ = (λ₁ - δλ, λ₂ + δλ)
    φ = (φ₁ - δφ, φ₂ + δφ)

    regions = (west  = BoundingBox(longitude = (λ₁ - δλ, λ₁ + w + δλ), latitude = φ),
               east  = BoundingBox(longitude = (λ₂ - w - δλ, λ₂ + δλ), latitude = φ),
               south = BoundingBox(longitude = λ, latitude = (φ₁ - δφ, φ₁ + w + δφ)),
               north = BoundingBox(longitude = λ, latitude = (φ₂ - w - δφ, φ₂ + δφ)))

    strips = map(region -> Component(region, dates, dataset; architecture = architecture(grid), kw...), regions[sides])

    return BoundaryPrescribedComponent(; strips...)
end
