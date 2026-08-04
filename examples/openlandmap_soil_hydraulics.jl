using NumericalEarth   # OpenLandMapSoilDB, BoundingBox, MetadataSet, soil_hydraulic_properties
using Oceananigans     # Field, CPU, interior
using ArchGDAL         # activates the windowed cloud-optimized-GeoTIFF reader
using CairoMakie

# Derive van Genuchten hydraulic parameters for a `VariablySaturatedHydrology` slab
# straight from 30 m soil texture. OpenLandMap-soilDB supplies sand/silt/clay and
# bulk density over three depth intervals; the pedotransfer function converts each
# interval to (ν, θʳ, α, n, Kₛ), and the depth-layer combination collapses them to
# one effective column per grid point.

z_interfaces = [-1.0, -0.6, -0.3, 0.0]   # depth faces, deepest-first: 60–100, 30–60, 0–30 cm
region = BoundingBox(longitude = (-112.2, -112.0), latitude = (36.0, 36.2))

# Native 30 m horizontal window × three depth intervals, read straight from the
# cloud-optimized GeoTIFFs. No credentials needed.
metadata = MetadataSet(:sand_fraction, :silt_fraction, :clay_fraction, :bulk_density;
                       dataset = OpenLandMapSoilDB(), region)

soil = map(m -> Field(m, CPU()), NamedTuple(metadata))

# Wösten HYPRES per depth layer — each layer read as topsoil or subsoil by its depth —
# then upscale each parameter over `slab_depth` with its own law: arithmetic for ν/θʳ,
# harmonic for Kₛ, resistance-weighted for α, and geometric in n − 1 for n.
properties = soil_hydraulic_properties(soil.sand_fraction, soil.silt_fraction,
                                       soil.clay_fraction, soil.bulk_density;
                                       slab_depth = 1.0, z_interfaces)

# θʳ is a fixed constant of the pedotransfer function, so only four parameters vary
# in space. Kₛ spans orders of magnitude, so it is mapped in log₁₀.
panels = [("porosity ν",                 "–",            properties.porosity,                 :viridis),
          ("inverse air-entry head α",   "m⁻¹",          properties.inverse_air_entry_head,    :plasma),
          ("pore-size uniformity n",     "–",            properties.pore_size_uniformity,     :plasma),
          ("saturated conductivity Kₛ",  "log₁₀(m s⁻¹)", Field(log10(properties.K_saturated)), :turbo)]

fig = Figure(size = (1150, 980), fontsize = 15)
Label(fig[0, 1:2], "Soil hydraulic parameters from OpenLandMap-soilDB 30 m — 0–100 cm — Grand Canyon window";
      fontsize = 18, font = :bold)

for (panel_number, (title, unit, field, cmap)) in enumerate(panels)
    ## Rock, water, and out-of-coverage cells arrive as NaN and propagate through the PTF.
    finite_values = filter(isfinite, interior(field))
    row, col = fldmod1(panel_number, 2)
    ax = Axis(fig[row, col]; title, xlabel = "longitude (°)", ylabel = "latitude (°)",
              aspect = DataAspect())
    hm = heatmap!(ax, field; colormap = cmap, colorrange = extrema(finite_values),
                  nan_color = RGBAf(0.85, 0.85, 0.85, 1))
    Colorbar(fig[row, col][1, 2], hm; label = unit)
end

save("soil_hydraulic_parameters_map.png", fig)
