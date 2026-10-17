using Oceananigans.OrthogonalSphericalShellGrids: LambertConformalConic, LambertConformalConicGrid,
                                                 lcc_forward, lcc_inverse, lcc_cone_constant

struct ConformalConicBox{FT}
    origin :: NTuple{2, FT}
    orientation :: FT
    extent :: NTuple{2, FT}
    standard_parallels :: NTuple{2, FT}
    radius :: FT
end

"""
$(TYPEDSIGNATURES)

Construct a rectangle in spherical conformal conic coordinates. `origin` is the
center's `(longitude, latitude)` in degrees, `orientation` is the counterclockwise
angle of the x-axis from local east in degrees, and `extent` gives the full projected
widths in meters. `radius` is the sphere radius in meters.

`standard_parallels` gives the two latitudes of unit projection scale in degrees.
With `nothing`, parallels are placed one-sixth and five-sixths through the latitude
span of a provisional box tangent at the origin. This requires a non-equatorial origin.
The box must exclude the cone apex and remain within the projection's longitude branch.

```jldoctest
using NumericalEarth

box = ConformalConicBox(origin=(-105, 40), extent=(4e6, 3e6), standard_parallels=(30, 60));
show(box)

# output
ConformalConicBox(origin=(-105.0, 40.0), orientation=0.0°, extent=(4.0e6, 3.0e6) m, standard_parallels=(30.0, 60.0))
```
"""
function ConformalConicBox(FT::DataType = Oceananigans.defaults.FloatType;
                          origin,
                          extent,
                          orientation = 0,
                          standard_parallels = nothing,
                          radius = Oceananigans.defaults.planet_radius)
    origin = NTuple{2, FT}(origin)
    extent = NTuple{2, FT}(extent)
    orientation = FT(orientation)
    radius = FT(radius)
    all(isfinite, origin) && abs(origin[2]) < 90 ||
        throw(ArgumentError("origin must have finite coordinates and latitude strictly between -90° and 90°."))
    all(width -> isfinite(width) && width > 0, extent) ||
        throw(ArgumentError("extent must contain two finite positive lengths."))
    isfinite(orientation) || throw(ArgumentError("orientation must be finite."))

    if isnothing(standard_parallels)
        origin[2] == 0 && throw(ArgumentError("An equatorial origin requires explicit standard_parallels."))
        provisional = ConformalConicBox(origin, orientation, extent, (origin[2], origin[2]), radius)
        south, north = BoundingBox(provisional).latitude
        span = north - south
        standard_parallels = (south + span / 6, north - span / 6)
    end

    box = ConformalConicBox(origin, orientation, extent, NTuple{2, FT}(standard_parallels), radius)
    BoundingBox(box)
    return box
end

Base.summary(box::ConformalConicBox) = string("ConformalConicBox(origin=", box.origin,
                                             ", orientation=", box.orientation, "°",
                                             ", extent=", box.extent, " m",
                                             ", standard_parallels=", box.standard_parallels, ")")
Base.show(io::IO, box::ConformalConicBox) = print(io, summary(box))

function conic_mapping(box::ConformalConicBox{FT}) where FT
    cone_constant = lcc_cone_constant(FT, deg2rad(box.standard_parallels[1]), deg2rad(box.standard_parallels[2]))
    # Meridian convergence at the origin is minus the requested grid orientation.
    central_longitude = box.origin[1] + box.orientation / cone_constant
    return LambertConformalConic(FT;
                                 central_longitude,
                                 latitude_of_origin = box.origin[2],
                                 standard_parallels = box.standard_parallels,
                                 radius = box.radius,
                                 x₁ = 0, y₁ = 0, Δx = 1, Δy = 1)
end

"""
$(TYPEDSIGNATURES)

Return the geographic envelope of `box`, expanded by `padding` degrees on each side.
Longitude bounds are continuous about the box's origin and may extend beyond ±180°.
Latitude extrema include points along edges, not just the four corners.
"""
function BoundingBox(box::ConformalConicBox; padding = 0)
    projection = conic_mapping(box)
    center_x, center_y = lcc_forward(projection, box.origin...)
    lower_x, upper_x = center_x .+ (-box.extent[1] / 2, box.extent[1] / 2)
    lower_y, upper_y = center_y .+ (-box.extent[2] / 2, box.extent[2] / 2)
    corners = ((lower_x, lower_y), (upper_x, lower_y), (upper_x, upper_y), (lower_x, upper_y))
    cone_constant = projection.cone_constant
    apex_y = projection.origin_radius
    angles = map(corners) do (x, y)
        atan(sign(cone_constant) * x, sign(cone_constant) * (apex_y - y))
    end

    crosses_cut = lower_x <= 0 <= upper_x &&
                  min(sign(cone_constant) * (apex_y - lower_y), sign(cone_constant) * (apex_y - upper_y)) <= 0
    !crosses_cut && abs(box.orientation) < 180abs(cone_constant) &&
        maximum(abs, angles) < π * abs(cone_constant) ||
        throw(ArgumentError("The box must exclude the cone apex and fit within the projection's longitude branch."))

    longitudes = map(angle -> rad2deg(projection.central_longitude + angle / cone_constant), angles)
    latitudes = map(corners) do corner
        _, latitude = lcc_inverse(projection, corner...)
        latitude
    end
    # Latitude is monotone in distance from the cone apex.
    _, closest_latitude = lcc_inverse(projection, clamp(0, lower_x, upper_x), clamp(apex_y, lower_y, upper_y))
    return BoundingBox(longitude = (minimum(longitudes) - padding, maximum(longitudes) + padding),
                       latitude = (max(-90, min(minimum(latitudes), closest_latitude) - padding),
                                   min(90, max(maximum(latitudes), closest_latitude) + padding)))
end

"""
$(TYPEDSIGNATURES)

Discretize `box` with horizontal dimensions `size[1:2]` and vertical coordinate `z`.
The returned grid is an Oceananigans `LambertConformalConicGrid`; additional keyword
arguments such as `halo` are passed to its constructor.
"""
function ConformalConicGrid(box::ConformalConicBox{FT}, arch::AbstractArchitecture = CPU(); kwargs...) where FT
    projection = conic_mapping(box)
    return LambertConformalConicGrid(arch, FT;
                                     center = box.origin,
                                     extent = box.extent,
                                     standard_parallels = box.standard_parallels,
                                     central_longitude = rad2deg(projection.central_longitude),
                                     latitude_of_origin = box.origin[2],
                                     radius = box.radius,
                                     kwargs...)
end

dataset_region(dataset, box::ConformalConicBox) = BoundingBox(box)
