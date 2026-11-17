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
    (λ₁, λ₂), (φ₁, φ₂) = horizontal_extents(grid)
    Nx, Ny, _ = size(grid)
    return width * max((λ₂ - λ₁) / Nx, (φ₂ - φ₁) / Ny)
end

horizontal_extents(grid) = (x_domain(grid), y_domain(grid))

# `x_domain` and `y_domain` reduce over the whole coordinate array, and a `ConformalConicGrid` leaves
# its coordinate halo unfilled. The interior slice is copied before reducing, as `BoundingBox` does,
# so the reduction stays on the device.
function horizontal_extents(grid::OrthogonalSphericalShellGrid)
    Nx, Ny, _ = size(grid)
    λ = grid.λᶠᶠᵃ[1:(Nx + 1), 1:(Ny + 1)]
    φ = grid.φᶠᶠᵃ[1:(Nx + 1), 1:(Ny + 1)]
    return ((minimum(λ), maximum(λ)), (minimum(φ), maximum(φ)))
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
