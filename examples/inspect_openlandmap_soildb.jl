# # OpenLandMap soil texture over the Grand Canyon
#
# We read the OpenLandMap-soilDB sand, silt, and clay fractions and the bulk density at
# their native 30 m resolution, straight from the cloud-optimized GeoTIFFs (no credentials
# needed), and map the surface layer.

using NumericalEarth
using Oceananigans
using ArchGDAL # activates the windowed cloud-optimized GeoTIFF reader
using CairoMakie
using Statistics

region = BoundingBox(longitude = (-112.3, -111.9), latitude = (36.0, 36.4))

panels = [(:sand_fraction, "Sand fraction", "kg/kg", :YlOrBr),
          (:silt_fraction, "Silt fraction", "kg/kg", :YlGnBu),
          (:clay_fraction, "Clay fraction", "kg/kg", :OrRd),
          (:bulk_density,  "Bulk density",  "kg/m³", :dense)]

fields = map(p -> Field(Metadatum(p[1]; dataset = OpenLandMapSoilDB(), region), CPU()), panels)

# Each field has three depth intervals (0–30, 30–60, and 60–100 cm), stored deepest first,
# so the 0–30 cm surface layer is the third.

surface_level = 3

fig = Figure(size = (1100, 980), fontsize = 15)
Label(fig[0, 1:2], "OpenLandMap-soilDB at 30 m, 0–30 cm, Grand Canyon"; fontsize = 18, font = :bold)

for (n, ((_, title, units, colormap), field)) in enumerate(zip(panels, fields))
    surface_layer = view(field, :, :, surface_level)
    finite_percentage = round(100 * mean(isfinite, surface_layer); digits = 1)
    row, column = fldmod1(n, 2)
    ax = Axis(fig[row, column]; title = "$title ($finite_percentage% finite)",
              xlabel = "longitude (°)", ylabel = "latitude (°)", aspect = DataAspect())
    hm = heatmap!(ax, surface_layer; colormap, nan_color = :lightgray)
    Colorbar(fig[row, column][1, 2], hm; label = units)
end

save("openlandmap_soildb_texture_map.png", fig)
