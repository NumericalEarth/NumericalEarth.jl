using ConservativeRegridding: Trees, Regridder
import ConservativeRegridding.Trees: treeify
import GeometryOpsCore as GOCore
import GeometryOps as GO
import GeoInterface as GI

using StaticArrays: SA

GOCore.best_manifold(grid::RingGrids.AbstractGrid) = GO.Spherical()
GOCore.best_manifold(sg::SpeedyWeather.SpectralGrid) = GOCore.best_manifold(sg.grid)

treeify(manifold::GOCore.Spherical, sg::SpeedyWeather.SpectralGrid) = treeify(manifold, sg.grid)

# get_gridcell_polygons returns CW (E, S, W, N); reverse to CCW (E, N, W, S)
function ccw_unit_sphere_polygon(polygons_matrix, ij)

    USFG = GO.UnitSphereFromGeographic()

    vE = USFG(polygons_matrix[1, ij])
    vN = USFG(polygons_matrix[4, ij])
    vW = USFG(polygons_matrix[3, ij])
    vS = USFG(polygons_matrix[2, ij])
    return GI.Polygon(SA[GI.LinearRing(SA[vE, vN, vW, vS, vE])])
end

# Full grids: all rings have the same nlon, so no padding needed.
# Linear index in ExplicitPolygonGrid matches the RingGrid flat index.
function treeify(manifold::GOCore.Spherical, grid::RingGrids.AbstractFullGrid)
    polygons = RingGrids.get_gridcell_polygons(grid)
    nlat = RingGrids.get_nlat(grid)
    nlon = RingGrids.get_nlon(grid)

    poly_matrix = [ccw_unit_sphere_polygon(polygons, (j - 1) * nlon + i)
                   for i in 1:nlon, j in 1:nlat]

    epg  = Trees.ExplicitPolygonGrid(manifold, poly_matrix)
    tree = Trees.TopDownQuadtreeCursor(epg)
    return Trees.KnownFullSphereExtentWrapper(tree)
end

# Reduced grids: variable nlon per ring. Build one sub-tree per latitude ring
# and combine with MultiTreeWrapper. Each ring is an ExplicitPolygonGrid(nlon_ring, 1)
# wrapped in IndexOffsetQuadtreeCursor so that emitted indices match RingGrid flat indices.
# ncells = npoints exactly — no ghost cells, no size mismatch in regrid!.
function treeify(manifold::GOCore.Spherical, grid::RingGrids.AbstractGrid)
    polygons = RingGrids.get_gridcell_polygons(grid)
    rings    = RingGrids.eachring(grid)
    subtrees = Trees.IndexOffsetQuadtreeCursor[]
    offsets  = Int[]
    cumulative = 0

    for ring in rings
        nlon_ring = length(ring)
        poly_ring = [ccw_unit_sphere_polygon(polygons, ij) for ij in ring]
        poly_ring = reshape(poly_ring, nlon_ring, 1)
        epg    = Trees.ExplicitPolygonGrid(manifold, poly_ring)
        cursor = Trees.IndexOffsetQuadtreeCursor(epg, first(ring) - 1)
        push!(subtrees, cursor)
        cumulative += nlon_ring
        push!(offsets, cumulative)
    end

    return Trees.MultiTreeWrapper(subtrees, offsets)
end
