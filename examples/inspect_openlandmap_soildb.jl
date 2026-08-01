using NumericalEarth   # OpenLandMapSoilDB, BoundingBox, Metadatum
using Oceananigans     # Field, CPU, interior
using ArchGDAL         # activates the windowed cloud-optimized-GeoTIFF reader
using CairoMakie

region = BoundingBox(longitude = (-112.3, -111.9), latitude = (36.0, 36.4))

# Native 30 m horizontal window × three depth intervals (0–30, 30–60, 60–100 cm),
# read straight from the cloud-optimized GeoTIFFs. No credentials needed.
panels = [(:sand_fraction, "sand fraction", "kg/kg", :YlOrBr),
          (:silt_fraction, "silt fraction", "kg/kg", :YlGnBu),
          (:clay_fraction, "clay fraction", "kg/kg", :OrRd),
          (:bulk_density,  "bulk density",  "kg/m³", :dense)]

fields = map(p -> Field(Metadatum(p[1]; dataset = OpenLandMapSoilDB(), region), CPU()), panels)

ksurf = 3   # surface 0–30 cm layer (depths stored deepest-first)

fig = Figure(size = (1100, 980), fontsize = 15)
Label(fig[0, 1:2], "OpenLandMap-soilDB 30 m — 0–30 cm — Grand Canyon window"; fontsize = 18, font = :bold)

for (n, ((_, title, unit, cmap), field)) in enumerate(zip(panels, fields))
    values = interior(field, :, :, ksurf)
    finite_pct = round(100 * count(isfinite, values) / length(values); digits = 1)
    row, col = fldmod1(n, 2)
    ax = Axis(fig[row, col]; title = "$title  ($finite_pct% finite)",
              xlabel = "longitude (°)", ylabel = "latitude (°)", aspect = DataAspect())
    hm = heatmap!(ax, view(field, :, :, ksurf); colormap = cmap,
                  colorrange = extrema(filter(isfinite, values)),
                  nan_color = RGBAf(0.85, 0.85, 0.85, 1))
    Colorbar(fig[row, col][1, 2], hm; label = unit)
end

save("openlandmap_soildb_texture_map.png", fig)
