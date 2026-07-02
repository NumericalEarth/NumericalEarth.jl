module NumericalEarthMakieExt

using NumericalEarth: NumericalEarth, BoundingBox, ETOPO2022, regrid_bathymetry, natural_earth_lines
using Oceananigans: CPU, LatitudeLongitudeGrid, Bounded, Center, Face, λnodes, φnodes, interior
using Makie: Figure, Axis, Colorbar, DataAspect, axislegend, cgrad,
             heatmap!, lines!, scatter!, xlims!, ylims!

bounding_box(region::BoundingBox) = region
bounding_box(grid) = BoundingBox(longitude = extrema(λnodes(grid, Face(), Center(), Center())),
                                 latitude  = extrema(φnodes(grid, Center(), Face(), Center())))

# Closed rectangle path tracing the perimeter of `region`.
function region_outline(region)
    λ₁, λ₂ = region.longitude
    φ₁, φ₂ = region.latitude
    return [λ₁, λ₂, λ₂, λ₁, λ₁], [φ₁, φ₁, φ₂, φ₂, φ₁]
end

# Recenter `:topo` so its sea-level break sits at relief = 0 while the colorrange stays linear
# in metres; regions entirely above (below) sea level use the land (ocean) half of the gradient.
function relief_colormap(minimum_relief, maximum_relief)
    topography_gradient = cgrad(:topo)
    remap = if minimum_relief >= 0
        g -> 1/2 + g/2
    elseif maximum_relief <= 0
        g -> g/2
    else
        sea_level = -minimum_relief / (maximum_relief - minimum_relief)
        g -> g <= sea_level ? g / (2sea_level) : 1/2 + (g - sea_level) / (2 * (1 - sea_level))
    end
    return [topography_gradient[remap(g)] for g in range(0, 1, length = 512)]
end

function NumericalEarth.visualize_nested_domain(grid;
                                         parent = nothing,
                                         padding = 2.5,
                                         resolution = 1/30,
                                         dataset = ETOPO2022(),
                                         boundaries = true,
                                         landmarks = tuple(),
                                         label = "grid",
                                         parent_label = "parent",
                                         title = "")

    grid_region  = bounding_box(grid)
    outer_region = isnothing(parent) ? grid_region : bounding_box(parent)

    longitude = (outer_region.longitude[1] - padding, outer_region.longitude[2] + padding)
    latitude  = (outer_region.latitude[1]  - padding, outer_region.latitude[2]  + padding)

    basemap_grid = LatitudeLongitudeGrid(CPU();
                                         longitude, latitude,
                                         z        = (0, 1),
                                         size     = (round(Int, (longitude[2] - longitude[1]) / resolution),
                                                     round(Int, (latitude[2]  - latitude[1])  / resolution), 1),
                                         topology = (Bounded, Bounded, Bounded))

    # `regrid_bathymetry` retains the full relief (negative over ocean), so the basemap
    # shows true bathymetry as well as topography — the coastline is rendered by the coloring.
    relief = Array(interior(regrid_bathymetry(basemap_grid; dataset)))[:, :, 1]
    minimum_relief, maximum_relief = extrema(relief)

    figure = Figure(size = (840, 760), fontsize = 13)
    axis   = Axis(figure[1, 1]; xlabel = "longitude (°)", ylabel = "latitude (°)",
                  title, aspect = DataAspect())

    relief_heatmap = heatmap!(axis,
                              collect(λnodes(basemap_grid, Center(), Center(), Center())),
                              collect(φnodes(basemap_grid, Center(), Center(), Center())),
                              relief;
                              colormap   = relief_colormap(minimum_relief, maximum_relief),
                              colorrange = (minimum_relief, maximum_relief))
    Colorbar(figure[1, 2], relief_heatmap; label = "elevation / depth (m)")

    if boundaries
        isempty(methods(natural_earth_lines)) &&
            throw(ArgumentError("`boundaries = true` requires NaturalEarth and GeoInterface to be loaded " *
                                "(`using NaturalEarth`); or pass `boundaries = false`."))
        for (name, color, linewidth) in (("admin_1_states_provinces_lines", (:gray20, 0.55), 0.7),
                                         ("admin_0_boundary_lines_land",    (:black,  0.75), 1.4))
            boundary_longitudes, boundary_latitudes = natural_earth_lines(name)
            lines!(axis, boundary_longitudes, boundary_latitudes; color, linewidth)
        end
    end

    isnothing(parent) ||
        lines!(axis, region_outline(outer_region)...; color = :dodgerblue, linewidth = 3, label = parent_label)
    lines!(axis, region_outline(grid_region)...; color = :crimson, linewidth = 3, label)

    for (landmark_label, (λ, φ)) in landmarks
        scatter!(axis, [λ], [φ]; color = :black, marker = :star5, markersize = 18, label = landmark_label)
    end

    axislegend(axis; position = :rt, framevisible = true, backgroundcolor = (:white, 0.85))

    # Clip to the padded map region — the Natural Earth lines span the globe.
    xlims!(axis, longitude...)
    ylims!(axis, latitude...)

    return figure
end

end # module NumericalEarthMakieExt
