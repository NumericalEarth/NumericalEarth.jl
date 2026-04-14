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
    polygons_matrix = RingGrids.get_gridcell_polygons(grid)
    npoints = size(polygons_matrix, 2)
    lonlat_to_usp = GO.UnitSphereFromGeographic()

    polys = Vector{GI.Polygon}(undef, npoints)
    for ij in 1:npoints
        v1 = lonlat_to_usp(polygons_matrix[1, ij])
        v2 = lonlat_to_usp(polygons_matrix[2, ij])
        v3 = lonlat_to_usp(polygons_matrix[3, ij])
        v4 = lonlat_to_usp(polygons_matrix[4, ij])
        polys[ij] = GI.Polygon(SA[GI.LinearRing(SA[v1, v2, v3, v4, v1])])
    end

    return Trees.treeify(manifold, polys)
end
