#####
##### A prescribed atmosphere known only on strips along the lateral boundaries of a limited-area domain
#####

struct BoundaryPrescribedAtmosphere{W, E, S, N, C}
    west  :: W
    east  :: E
    south :: S
    north :: N
    clock :: C
end

"""
$(TYPEDSIGNATURES)

A prescribed atmosphere known only on strips along the lateral boundaries of a limited-area
domain. Each of `west`, `east`, `south` and `north` is a [`PrescribedAtmosphere`](@ref) whose
region covers that side's boundary and relaxation zone, or `nothing` for a side without one.
"""
function BoundaryPrescribedAtmosphere(; west=nothing, east=nothing, south=nothing, north=nothing)
    first_strip = something(west, east, south, north)
    clock = Clock(time = first_strip.clock.time)
    return BoundaryPrescribedAtmosphere(west, east, south, north, clock)
end

"""
$(TYPEDSIGNATURES)

The `PrescribedAtmosphere` strips of `atmosphere`, as a `NamedTuple` keyed by side.
"""
function boundary_strips(atmosphere::BoundaryPrescribedAtmosphere)
    sides = filter(side -> !isnothing(getproperty(atmosphere, side)), (:west, :east, :south, :north))
    return NamedTuple{sides}(map(side -> getproperty(atmosphere, side), sides))
end

# Width of a Davies relaxation zone spanning `width` cells along the coarser horizontal direction of `grid`.
function relaxation_zone_width(grid, width)
    λ₁, λ₂ = x_domain(grid)
    φ₁, φ₂ = y_domain(grid)
    Nx, Ny, _ = size(grid)
    return width * max((λ₂ - λ₁) / Nx, (φ₂ - φ₁) / Ny)
end

function Oceananigans.TimeSteppers.time_step!(atmosphere::BoundaryPrescribedAtmosphere, Δt)
    tick!(atmosphere.clock, Δt)
    foreach(strip -> Oceananigans.TimeSteppers.time_step!(strip, Δt), boundary_strips(atmosphere))
    return nothing
end

Base.summary(atmosphere::BoundaryPrescribedAtmosphere) =
    string("BoundaryPrescribedAtmosphere with sides ", join(keys(boundary_strips(atmosphere)), ", "))

function Base.show(io::IO, atmosphere::BoundaryPrescribedAtmosphere)
    strips = boundary_strips(atmosphere)
    print(io, summary(atmosphere), ":")
    for (n, side) in enumerate(keys(strips))
        prefix = n == length(strips) ? "└── " : "├── "
        print(io, '\n', prefix, side, ": ", summary(strips[side]))
    end
end
