module NumericalEarthNaturalEarthExt

using NumericalEarth: NumericalEarth
using NaturalEarth: naturalearth
import GeoInterface as GI

# Flatten a (Multi)LineString geometry into the running `lons`/`lats`, appending a
# `NaN` after each line so a plotting backend renders the parts as disjoint segments.
append_border!(lons, lats, geom) = append_border!(lons, lats, GI.geomtrait(geom), geom)

function append_border!(lons, lats, ::GI.LineStringTrait, line)
    for p in GI.getpoint(line)
        push!(lons, GI.x(p))
        push!(lats, GI.y(p))
    end
    push!(lons, NaN)
    push!(lats, NaN)
    return nothing
end

append_border!(lons, lats, ::GI.MultiLineStringTrait, multiline) =
    foreach(line -> append_border!(lons, lats, line), GI.getgeom(multiline))

# Non-line geometries (points, polygons) are not borders — skip them.
append_border!(lons, lats, ::Any, geom) = nothing

function NumericalEarth.natural_earth_lines(name; scale = 50)
    lons, lats = Float64[], Float64[]
    for feature in naturalearth(name, scale)
        geom = GI.geometry(feature)
        isnothing(geom) || append_border!(lons, lats, geom)
    end
    return lons, lats
end

end # module NumericalEarthNaturalEarthExt
