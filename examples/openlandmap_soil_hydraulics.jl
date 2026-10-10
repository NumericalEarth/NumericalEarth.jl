# # Soil hydraulic parameters from OpenLandMap-soilDB
#
# We derive the van Genuchten hydraulic parameters of a `VariablySaturatedHydrology` slab
# directly from 30 m soil texture. OpenLandMap-soilDB supplies the sand, silt, and clay
# fractions and the bulk density over three depth intervals; a pedotransfer function converts
# each interval to (ν, θʳ, αᵃᵉ, 𝓃, K₀, ηᴷ), and combining the depth layers collapses them into
# one effective column per grid point.

using NumericalEarth   # OpenLandMapSoilDB, BoundingBox, MetadataSet, soil_hydraulic_properties
using Oceananigans     # Field, CPU, interior
using ArchGDAL         # activates the windowed cloud-optimized-GeoTIFF reader
using CairoMakie
using Statistics       # quantile, for robust color limits
using NumericalEarth.DataWrangling: NearestNeighborInpainting

region = BoundingBox(longitude = (-112.2, -112.0), latitude = (36.0, 36.2))

# The data come at the native 30 m horizontal resolution on three depth intervals (60–100, 30–60,
# and 0–30 cm), read directly from the cloud-optimized GeoTIFFs. No credentials are needed.
metadata = MetadataSet(:sand_fraction, :silt_fraction, :clay_fraction, :bulk_density;
                       dataset = OpenLandMapSoilDB(), region)

# The canyon walls and the river, 16 % of this window, carry no texture. Inpainting fills them
# from the neighboring soil before the pedotransfer function runs, which keeps all six parameters
# of a filled cell mutually consistent.
soil = map(m -> Field(m, CPU(); inpainting = NearestNeighborInpainting(20)), NamedTuple(metadata))

# The Weynants pedotransfer function is applied to each depth layer, and the layers are then
# combined over `slab_depth`: αᵃᵉ and 𝓃 are matched to the thickness-weighted mean retention
# curve, while K₀ is upscaled harmonically.
properties = soil_hydraulic_properties(soil.sand_fraction, soil.silt_fraction,
                                       soil.clay_fraction, soil.bulk_density;
                                       slab_depth = 1.0)

# The property names match the keyword arguments of the closures they belong to, so the
# parameters go straight into a hydrology model.
retention_curve = VanGenuchtenRetention(inverse_air_entry_head = properties.inverse_air_entry_head,
                                        pore_size_uniformity = properties.pore_size_uniformity)

hydraulic_conductivity = VanGenuchtenConductivity(matching_point_conductivity = properties.matching_point_conductivity,
                                                  pore_size_uniformity = properties.pore_size_uniformity,
                                                  pore_connectivity_exponent = properties.pore_connectivity_exponent)

hydrology = VariablySaturatedHydrology(slab_depth = 1.0,
                                       storage_height = 1000,
                                       porosity = properties.porosity,
                                       residual_liquid_fraction = properties.residual_liquid_fraction,
                                       retention_curve,
                                       hydraulic_conductivity,
                                       deep_liquid_flux = FreeDrainageFlux())

# This pedotransfer function gives θʳ = 0 everywhere, so only five parameters vary in space.
# Its K₀ is the *matrix* matching-point conductivity that the conductivity closure expects; an
# infiltration cap instead needs the macropore-inclusive Cosby K⁺, which we map alongside for contrast.
K₀ = properties.matching_point_conductivity
infiltration_capacity = Field(3_600_000 * saturated_conductivity(CosbyConductivity(),
                                                                 soil.sand_fraction))

panels = [("porosity ν",                    "–",            properties.porosity,                   :viridis),
          ("inverse air-entry head αᵃᵉ",      "m⁻¹",          properties.inverse_air_entry_head,     :plasma),
          ("pore-size uniformity 𝓃",        "–",            properties.pore_size_uniformity,       :plasma),
          ("pore-connectivity exponent ηᴷ", "–",            properties.pore_connectivity_exponent, :batlow),
          ("matching-point K₀",             "log₁₀(m s⁻¹)", Field(log10(K₀)),                      :turbo),
          ("Cosby saturated K⁺ (0–30 cm)",  "mm hour⁻¹",    view(infiltration_capacity, :, :, 3),  :turbo)]

# Every parameter here has a thin tail: for 𝓃, the full minimum-to-maximum range spends 77 % of
# the colormap on fewer than 2 % of the cells, which flattens everything else. We span the 1st to
# 99th percentiles and let the tails saturate; the colorbar marks them with pointed ends.
function percentile_range(field, low = 0.01, high = 0.99)
    values = sort!(filter(isfinite, interior(field)))
    return quantile(values, low, sorted=true), quantile(values, high, sorted=true)
end

fig = Figure(size = (1150, 1450), fontsize = 15)
Label(fig[0, 1:2], "Soil hydraulic parameters from OpenLandMap-soilDB 30 m — 0–100 cm — Grand Canyon window";
      fontsize = 18, font = :bold)

for (panel_number, (title, unit, field, colormap)) in enumerate(panels)
    row, col = fldmod1(panel_number, 2)
    ax = Axis(fig[row, col]; title, xlabel = "longitude (°)", ylabel = "latitude (°)",
              aspect = DataAspect())
    hm = heatmap!(ax, field; colormap, colorrange = percentile_range(field),
                  lowclip = :grey25, highclip = :grey85,
                  nan_color = RGBAf(0.85, 0.85, 0.85, 1))
    Colorbar(fig[row, col][1, 2], hm; label = unit)
end

save("soil_hydraulic_parameters_map.png", fig)

hydrology
