using ConservativeRegridding: Trees, Regridder
import ConservativeRegridding.Trees: treeify
import GeometryOpsCore as GOCore
import GeometryOps as GO
import GeoInterface as GI

using StaticArrays: SA

GOCore.best_manifold(grid::RingGrids.AbstractGrid) = GO.Spherical()
GOCore.best_manifold(sg::SpeedyWeather.SpectralGrid) = GOCore.best_manifold(sg.grid)

treeify(manifold::GOCore.Spherical, sg::SpeedyWeather.SpectralGrid) = treeify(manifold, sg.grid)

function _make_degenerate_polygon(lonlat_to_usp)
    p = lonlat_to_usp((0.0, 0.0))
    return GI.Polygon(SA[GI.LinearRing(SA[p, p, p, p, p])])
end

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

# Reduced grids: variable nlon per ring, pad shorter rings with degenerate polygons.
# The ReorderedTopDownQuadtreeCursor remaps 2D (i,j) → flat RingGrid indices 1:npoints.
# Ghost cells (padding) get indices npoints+1:ntotal and have zero intersection area.
function treeify(manifold::GOCore.Spherical, grid::RingGrids.AbstractGrid)
    polygons_matrix = RingGrids.get_gridcell_polygons(grid)
    npoints = size(polygons_matrix, 2)
    lonlat_to_usp = GO.UnitSphereFromGeographic()

    rings    = RingGrids.eachring(grid)
    nlat     = length(rings)
    nlon_max = RingGrids.get_nlon_max(grid)
    ntotal   = nlon_max * nlat

    degenerate = _make_degenerate_polygon(lonlat_to_usp)
    poly_type  = typeof(degenerate)
    poly_matrix = Matrix{poly_type}(undef, nlon_max, nlat)

    # Reordering: real cells → RingGrid flat index (1:npoints)
    # Ghost cells → npoints+1:ntotal
    cart2lin = zeros(Int, nlon_max, nlat)
    lin2cart = Vector{CartesianIndex{2}}(undef, ntotal)

    n_padding = 0
    for (j, ring) in enumerate(rings)
        nlon_ring = length(ring)
        for (i_local, ij) in enumerate(ring)
            poly_matrix[i_local, j] = _to_ccw_unit_sphere_polygon(lonlat_to_usp, polygons_matrix, ij)
            cart2lin[i_local, j] = ij
            lin2cart[ij] = CartesianIndex(i_local, j)
        end
        for i_pad in (nlon_ring + 1):nlon_max
            poly_matrix[i_pad, j] = degenerate
            n_padding += 1
            pad_idx = npoints + n_padding
            cart2lin[i_pad, j] = pad_idx
            lin2cart[pad_idx] = CartesianIndex(i_pad, j)
        end
    end

    epg = Trees.ExplicitPolygonGrid(manifold, poly_matrix)
    ordering = Trees.Reorderer2D(cart2lin, lin2cart)
    tree = Trees.ReorderedTopDownQuadtreeCursor(epg, ordering)
    return Trees.KnownFullSphereExtentWrapper(tree)
end
