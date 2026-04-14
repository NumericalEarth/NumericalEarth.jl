using ConservativeRegridding: Trees, Regridder
import ConservativeRegridding.Trees: treeify
import GeometryOpsCore as GOCore
import GeometryOps as GO
import GeoInterface as GI

using StaticArrays: SA

GOCore.best_manifold(grid::RingGrids.AbstractGrid) = GO.Spherical()
GOCore.best_manifold(sg::SpeedyWeather.SpectralGrid) = GOCore.best_manifold(sg.grid)

treeify(manifold::GOCore.Spherical, sg::SpeedyWeather.SpectralGrid) = treeify(manifold, sg.grid)

function _to_ccw_unit_sphere_polygon(lonlat_to_usp, polygons_matrix, ij)
    vE = lonlat_to_usp(polygons_matrix[1, ij])
    vS = lonlat_to_usp(polygons_matrix[2, ij])
    vW = lonlat_to_usp(polygons_matrix[3, ij])
    vN = lonlat_to_usp(polygons_matrix[4, ij])
    # get_gridcell_polygons returns CW (E, S, W, N); reverse to CCW (E, N, W, S)
    return GI.Polygon(SA[GI.LinearRing(SA[vE, vN, vW, vS, vE])])
end

# Full grids: all rings have the same nlon, so no padding needed.
# Linear index in ExplicitPolygonGrid matches the RingGrid flat index.
function treeify(manifold::GOCore.Spherical, grid::RingGrids.AbstractFullGrid)
    polygons_matrix = RingGrids.get_gridcell_polygons(grid)
    lonlat_to_usp = GO.UnitSphereFromGeographic()

    nlat = RingGrids.get_nlat(grid)
    nlon = RingGrids.get_nlon(grid)

    poly_matrix = [_to_ccw_unit_sphere_polygon(lonlat_to_usp, polygons_matrix, (j - 1) * nlon + i)
                   for i in 1:nlon, j in 1:nlat]

    epg = Trees.ExplicitPolygonGrid(manifold, poly_matrix)
    tree = Trees.TopDownQuadtreeCursor(epg)
    return Trees.KnownFullSphereExtentWrapper(tree)
end

# Reduced grids: variable nlon per ring. Build one sub-tree per latitude ring
# and combine with MultiTreeWrapper. Each ring is an ExplicitPolygonGrid(nlon_ring, 1)
# wrapped in IndexOffsetQuadtreeCursor so that emitted indices match RingGrid flat indices.
# ncells = npoints exactly — no ghost cells, no size mismatch in regrid!.
function treeify(manifold::GOCore.Spherical, grid::RingGrids.AbstractGrid)
    polygons_matrix = RingGrids.get_gridcell_polygons(grid)
    lonlat_to_usp = GO.UnitSphereFromGeographic()

    rings = RingGrids.eachring(grid)
    nlat  = length(rings)

    # Build one sub-tree per latitude ring
    poly_type = typeof(_to_ccw_unit_sphere_polygon(lonlat_to_usp, polygons_matrix, 1))

    subtrees = Trees.IndexOffsetQuadtreeCursor[]
    offsets  = Int[]
    cumulative = 0

    for ring in rings
        nlon_ring = length(ring)
        poly_ring = Matrix{poly_type}(undef, nlon_ring, 1)
        for (i, ij) in enumerate(ring)
            poly_ring[i, 1] = _to_ccw_unit_sphere_polygon(lonlat_to_usp, polygons_matrix, ij)
        end
        epg = Trees.ExplicitPolygonGrid(manifold, poly_ring)
        cursor = Trees.IndexOffsetQuadtreeCursor(epg, first(ring) - 1)
        push!(subtrees, cursor)
        cumulative += nlon_ring
        push!(offsets, cumulative)
    end

    return Trees.MultiTreeWrapper(subtrees, offsets)
end
