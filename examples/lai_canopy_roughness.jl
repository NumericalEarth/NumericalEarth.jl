# # Vegetation roughness climatology from MODIS LAI
#
# Derive momentum roughness length `z₀` and zero-plane displacement `d₀` from MODIS
# leaf-area index and IGBP land cover using the Raupach (1994) drag-partition closure
# (as parameterized by Jasinski et al. 2005; cf. Borak et al. 2025), and assemble a
# monthly climatology consumable by the land–atmosphere surface-flux solver.
#
# Requirements: `using ArchGDAL` (with a GDAL_jll HDF4 driver) and NASA Earthdata
# credentials in `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` for the MODIS download;
# `using CairoMakie` for the figures. See `src/DataWrangling/MODISLand/README.md`.

using NumericalEarth
using NumericalEarth.DataWrangling: BoundingBox, Metadatum
using NumericalEarth.DataWrangling.MODISLand: MCD15A2H, MCD12Q1
using NumericalEarth.DataWrangling.CanopyRoughness: canopy_roughness_climatology, fill_temporal_gaps!
using Oceananigans
using Oceananigans.Fields: set!, interior
using Oceananigans.OutputReaders: FieldTimeSeries
using ArchGDAL
using Dates

# A small demonstration box over the Missouri Ozarks (deciduous forest + cropland),
# on a target grid near the 500 m native resolution.
region = BoundingBox(longitude = (-92.0, -91.0), latitude = (37.0, 38.0))
grid = LatitudeLongitudeGrid(CPU(), Float32; size = (96, 96),
                             longitude = (-92.0, -91.0), latitude = (37.0, 38.0),
                             topology = (Bounded, Bounded, Flat))

# Static IGBP land cover (choose the `:IGBP` legend so classes match the closure's tables).
land_cover = Field(Metadatum(:landcover_igbp; dataset = MCD12Q1(legend = :IGBP),
                             region, date = DateTime(2020, 1, 1)), grid)

# One 8-day LAI composite per ~month across 2020 (composite start DOYs, spaced 32 days).
doys  = [9, 41, 73, 105, 137, 169, 201, 233, 265, 297, 329, 361]
dates = [DateTime(2020) + Day(d - 1) for d in doys]

lai = FieldTimeSeries{Center, Center, Nothing}(grid, Float64.(doys .* 86400))
for (n, date) in enumerate(dates)
    set!(lai[n], Field(Metadatum(:leaf_area_index; dataset = MCD15A2H(), region, date), grid))
end

# Repair cloud/snow-screened gaps (heaviest in winter) by cyclic temporal interpolation per
# pixel — so every land cell carries a value the flux solver can use. Returns the pre-fill
# missing fraction (reported, not silently dropped); multi-year compositing is the production
# complement (loop years, average per period).
missing_fraction = fill_temporal_gaps!(lai)
@info "LAI missing fraction before gap-fill" missing_fraction

# Apply the drag-partition closure per period → z₀ and d₀ climatologies (metres).
z0m, d0 = canopy_roughness_climatology(lai, land_cover)

# ## Plot the monthly climatology
using CairoMakie
using Statistics: mean

month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
λ = Array(Oceananigans.Grids.λnodes(grid, Center()))
φ = Array(Oceananigans.Grids.φnodes(grid, Center()))
slice(fts, m) = Array(interior(fts[m], :, :, 1))

fig = Figure(size = (1400, 520))
for m in 1:12  # animate through the year
    for (col, (fts, crange, cmap, name)) in enumerate(((lai, (0, 6), :YlGn, "LAI"),
                                                       (z0m, (0, 2), :viridis, "z₀ (m)"),
                                                       (d0,  (0, 15), :magma, "d₀ (m)")))
        ax = Axis(fig[1, col]; title = "$name — $(month_names[m])", aspect = DataAspect())
        heatmap!(ax, λ, φ, slice(fts, m); colorrange = crange, colormap = cmap, nan_color = :gray85)
    end
end
save("canopy_roughness_climatology.png", fig)
