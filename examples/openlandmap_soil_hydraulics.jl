using NumericalEarth   # OpenLandMapSoilDB, BoundingBox, MetadataSet, soil_hydraulic_properties
using Oceananigans     # Field, CPU, interior
using ArchGDAL         # activates the windowed cloud-optimized-GeoTIFF reader
using CairoMakie
using Statistics       # quantile, for robust color limits
using NumericalEarth.DataWrangling: NearestNeighborInpainting

# Derive van Genuchten hydraulic parameters for a `VariablySaturatedHydrology` slab
# straight from 30 m soil texture. OpenLandMap-soilDB supplies sand/silt/clay and
# bulk density over three depth intervals; the pedotransfer function converts each
# interval to (ν, θʳ, α, n, K₀, ηᴷ), and the depth-layer combination collapses them to
# one effective column per grid point.

region = BoundingBox(longitude = (-112.2, -112.0), latitude = (36.0, 36.2))

# Native 30 m horizontal window × three depth intervals (60–100, 30–60 and 0–30 cm),
# read straight from the cloud-optimized GeoTIFFs. No credentials needed.
metadata = MetadataSet(:sand_fraction, :silt_fraction, :clay_fraction, :bulk_density;
                       dataset = OpenLandMapSoilDB(), region)

# The canyon walls and the river carry no texture, 16 % of this window. Inpainting fills them
# from neighboring soil before the pedotransfer function runs, which keeps all six parameters
# of a filled cell mutually consistent.
soil = map(m -> Field(m, CPU(); inpainting = NearestNeighborInpainting(20)), NamedTuple(metadata))

# Weynants per depth layer, then combined over `slab_depth`: α and n are matched to the
# thickness-weighted mean retention curve, K₀ upscales harmonically.
properties = soil_hydraulic_properties(soil.sand_fraction, soil.silt_fraction,
                                       soil.clay_fraction, soil.bulk_density;
                                       slab_depth = 1.0)

# The keys are the keyword arguments of the closures they belong to, so the parameter set
# goes straight into a hydrology.
hydrology = VariablySaturatedHydrology(
    slab_depth = 1.0,
    storage_height = 1000,
    porosity = properties.porosity,
    residual_liquid_fraction = properties.residual_liquid_fraction,
    retention_curve = VanGenuchtenRetention(
        inverse_air_entry_head = properties.inverse_air_entry_head,
        pore_size_uniformity = properties.pore_size_uniformity),
    hydraulic_conductivity = VanGenuchtenConductivity(
        matching_point_conductivity = properties.matching_point_conductivity,
        pore_size_uniformity = properties.pore_size_uniformity,
        pore_connectivity_exponent = properties.pore_connectivity_exponent),
    deep_liquid_flux = FreeDrainageFlux())

# θʳ is zero throughout for this pedotransfer function, so five parameters vary in space.
# Its K₀ is the *matrix* matching point the conductivity closure wants; an infiltration cap
# wants the macropore-inclusive Cosby Kˢᵃᵗ, mapped alongside it for contrast.
K₀ = properties.matching_point_conductivity
infiltration_capacity = Field(3_600_000 * saturated_conductivity(CosbyConductivity(),
                                                                 soil.sand_fraction))

panels = [("porosity ν",                     "–",            properties.porosity,                   :viridis),
          ("inverse air-entry head α",       "m⁻¹",          properties.inverse_air_entry_head,      :plasma),
          ("pore-size uniformity n",         "–",            properties.pore_size_uniformity,       :plasma),
          ("pore-connectivity exponent ηᴷ",  "–",            properties.pore_connectivity_exponent, :batlow),
          ("matching-point K₀",              "log₁₀(m s⁻¹)", Field(log10(K₀)),                       :turbo),
          ("Cosby saturated Kˢᵃᵗ (0–30 cm)", "mm hour⁻¹",    view(infiltration_capacity, :, :, 3),   :turbo)]

# Every parameter here has a thin tail: for n, the full min-to-max range spends 77 % of the
# colormap on under 2 % of the cells, which flattens everything else. Span the 1st to 99th
# percentile instead and let the tails saturate, which the colorbar marks with pointed ends.
function percentile_range(field, low = 0.01, high = 0.99)
    values = sort!(filter(isfinite, vec(Array(interior(field)))))
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
