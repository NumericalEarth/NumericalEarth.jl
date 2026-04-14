using ConservativeRegridding: Trees
import ConservativeRegridding.Trees: treeify
import GeometryOpsCore as GOCore
import GeometryOps as GO
import GeoInterface as GI

using StaticArrays: SA

GOCore.best_manifold(grid::RingGrids.AbstractGrid) = GO.Spherical()
GOCore.best_manifold(sg::SpeedyWeather.SpectralGrid) = GOCore.best_manifold(sg.grid)

treeify(manifold::GOCore.Spherical, sg::SpeedyWeather.SpectralGrid) = treeify(manifold, sg.grid)

function treeify(manifold::GOCore.Spherical, grid::RingGrids.AbstractGrid)
    # get_gridcell_polygons returns a 5×npoints matrix of (lon, lat) tuples
    # with vertices in clockwise order: E, S, W, N, E (closed).
    # We reverse to counter-clockwise (E, N, W, S, E) for correct signed area.
    polygons_matrix = RingGrids.get_gridcell_polygons(grid)
    npoints = size(polygons_matrix, 2)
    lonlat_to_usp = GO.UnitSphereFromGeographic()

    polys = Vector{GI.Polygon}(undef, npoints)
    for ij in 1:npoints
        vE = lonlat_to_usp(polygons_matrix[1, ij])
        vS = lonlat_to_usp(polygons_matrix[2, ij])
        vW = lonlat_to_usp(polygons_matrix[3, ij])
        vN = lonlat_to_usp(polygons_matrix[4, ij])
        # CCW: E → N → W → S → E
        polys[ij] = GI.Polygon(SA[GI.LinearRing(SA[vE, vN, vW, vS, vE])])
    end

    return Trees.treeify(manifold, polys)
end
