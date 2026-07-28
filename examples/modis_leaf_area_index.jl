# # Leaf area index from MODIS, as a seasonal climatology
#
# This example builds a leaf-area-index (``Λ``) boundary condition from MODIS MCD15A2H —
# 500 m combined Terra + Aqua composites, one every eight days since 2002.
#
# The region is the central Amazon around the Rio Negro–Solimões confluence, on the same grid
# the bare-earth terrain and canopy-height examples use, so leaf area index and canopy height
# land on one grid — the pair an aerodynamic-roughness closure consumes.
#
# Two things about this record shape the example. A single 8-day composite is not usable as a
# boundary condition, because cloud takes a fifth of a good scene and most of a bad one; the fix
# is to composite the same period across years, which [`MODISLAIClimatology`](@ref) does, giving
# 46 periods per year as a cyclic `FieldTimeSeries`. And unlike the two-stream inversion records,
# MCD15A2H targets *true* leaf area index, applying per-biome foliage clumping inside its own
# retrieval — the quantity a canopy closure calibrated on MODIS expects.
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add NumericalEarth ArchGDAL Oceananigans CairoMakie"
# ```
#
# `ArchGDAL` provides GDAL's HDF4 driver, which reads the sinusoidal HDF-EOS granules, and a
# free [NASA Earthdata](https://urs.earthdata.nasa.gov) login supplies the download credentials
# through `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`.

using NumericalEarth
using ArchGDAL
using Oceananigans
using Oceananigans.Fields: interpolate!
using Dates
using Printf
using Statistics
using CairoMakie

# ## The region and the model grid
#
# The dataset object carries product identity only; where and when to read comes from the
# `Metadata`. Granules are sinusoidal tiles, so a region is discovered from NASA's metadata
# repository, mosaicked, and reprojected onto a regional latitude-longitude window at download
# time — a global read is never attempted. This box straddles a tile boundary, so each date is
# assembled from two granules.

latitude = (-3.5, -2.4)
longitude = (-60.5, -59.0)

region = BoundingBox(; longitude, latitude)

grid = LatitudeLongitudeGrid(CPU(), Float32;
                             size = (1500, 1100),
                             longitude, latitude,
                             topology = (Bounded, Bounded, Flat))

nanmean(field) = mean(filter(!isnan, vec(Array(interior(field)))))
gap_fraction(field) = mean(isnan.(Array(interior(field))))

# ## Why a single composite looks so sparse
#
# A single 8-day map of this box is missing a quarter of its cells, which looks alarming until
# the three contributions are separated. They are different in kind, and only one of them is a
# limitation of the retrieval.

date = DateTime(2021, 7, 20)

single = Metadatum(:leaf_area_index; dataset = MCD15A2H(), region, date)

Λsingle = Field(single)

# **The first is not missing data at all.** Where the product does not attempt a retrieval it
# writes a land-cover class instead — water, urban, snow, barren — and the adapter reads those
# codes back as `:landcover_code`. Over this box they are the river network and the city of
# Manaus, so they are permanently unretrievable by design, and reporting them as leaf area of
# zero would be worse than reporting them as missing.

classes = Metadatum(:landcover_code; dataset = MCD15A2H(), region, date)

class_fraction = mean(.!isnan.(Array(interior(Field(classes)))))

@printf("Single 8-day composite, %s, on the product's own 500 m grid\n",
        Dates.format(date, "yyyy-mm-dd"))
@printf("  a land-cover class, not a failed retrieval : %.2f%%\n", 100 * class_fraction)
@printf("  total missing                              : %.2f%%\n", 100 * gap_fraction(Λsingle))
@printf("  → attributable to cloud and quality        : %.2f%%\n",
        100 * (gap_fraction(Λsingle) - class_fraction))

# **The second is cloud, and compositing across years removes it — given enough years.** The
# same period in another year is usually clear, so the composite's gap collapses onto the
# permanent land-cover classes and stops there, which is the signature of the compositing working
# rather than of it papering over anything. How many years "enough" is turns out to depend
# sharply on the season, and the seasonal section below measures it. `years` defaults to
# 2003–2019, the span over which both Terra and Aqua held their equatorial crossing times.

period = 26   # the 8-day period containing 20 July

deep = MODISLAIClimatology(years = 2003:2019)

build_lai_climatology!(deep; region, periods = period:period)

stamp = DateTime(2018, 1, 1) + Day(8 * (period - 1))

deep_composite = Metadatum(:leaf_area_index; dataset = deep, region, date = stamp)

Λdeep = Field(deep_composite)

@printf("\n%s composite of the same period\n", deep.years)
@printf("  total missing : %.2f%%  (the land-cover classes are %.2f%%)\n",
        100 * gap_fraction(Λdeep), 100 * class_fraction)
@printf("  mean Λ        : %.2f m² m⁻²\n", nanmean(Λdeep))

# **The third is the regrid, and it is the one worth knowing about.** Landing a field with
# scattered gaps on a model grid goes through the shared bilinear interpolation, whose 2×2
# stencil is not `NaN`-aware: one missing native cell makes up to four missing model cells.
# Dilating the native mask by that stencil predicts the model-grid gap exactly, and the inflation
# is the same at every target resolution — the signature of a stencil rather than of resolution.

function stencil_dilated(field)
    m = isnan.(Array(interior(field, :, :, 1)))
    d = copy(m)
    for i in 2:size(m, 1), j in 2:size(m, 2)
        d[i, j] = m[i, j] | m[i-1, j] | m[i, j-1] | m[i-1, j-1]
    end
    return mean(d)
end

@printf("\nWhat the regrid onto the model grid costs\n")
@printf("  native 500 m grid            : %.2f%% missing\n", 100 * gap_fraction(Λdeep))
@printf("  the 2x2 stencil would dilate : %.2f%%\n", 100 * stencil_dilated(Λdeep))
@printf("  measured on the model grid   : %.2f%%\n",
        100 * gap_fraction(Field(deep_composite, grid)))

# ## Is the magnitude right?
#
# Worth checking before the field is used for anything, because a leaf-area record can be
# self-consistent and still sit in the wrong place. Closed rainforest carries a *true* leaf area
# index of roughly 5–6, and this box is closed rainforest, so the composite has to land there —
# a two-stream inversion record reporting the *effective* quantity over the same box would read
# considerably lower, and the difference is one of definition rather than of quality.

@printf("\nCentral Amazon, %s composite\n", deep.years)
@printf("  MCD15A2H mean Λ : %.2f m² m⁻²   (closed rainforest is 5-6)\n", nanmean(Λdeep))

axis_kw = (xlabel = "Longitude (ᵒ)", ylabel = "Latitude (ᵒ)")
Λrange = (0, 7)

fig = Figure(size = (1500, 900))

ax = Axis(fig[1, 1]; title = "One 8-day composite, $(Dates.format(date, "u dd yyyy"))", axis_kw...)
heatmap!(ax, Field(single, grid), colormap = :viridis, colorrange = Λrange,
         nan_color = :midnightblue)

ax = Axis(fig[1, 2]; title = "$(deep.years) composite, same period", axis_kw...)
plot = heatmap!(ax, Field(deep_composite, grid), colormap = :viridis, colorrange = Λrange,
                nan_color = :midnightblue)
Colorbar(fig[1, 3], plot, label = "Λ (m² m⁻²)")

# The classes the product names instead of retrieving. They are the reason the composite's gap
# does not close further, and their shape — a connected river network, not scattered speckle —
# is what a correct geolocation looks like.
#
# Plotted on the product's own 500 m grid, deliberately: these are class *codes*, so
# interpolating them onto the model grid would average urban (250) against water (254) into
# snow/ice (252) along every riverbank. Continuous fields regrid; categorical ones do not.

ax = Axis(fig[2, 1]; title = "Land-cover code (native 500 m grid)", axis_kw...)
heatmap!(ax, Field(classes), colormap = cgrad(:tab10, 6, categorical = true),
         colorrange = (249.5, 255.5), nan_color = (:seagreen, 0.25))

ax = Axis(fig[2, 2]; xlabel = "Λ (m² m⁻²)", ylabel = "Cells",
          title = "Distribution over the retained canopy")
hist!(ax, filter(!isnan, vec(Array(interior(Λdeep)))), bins = 50, color = (:seagreen, 0.8))

save("modis_leaf_area_index.png", fig)
nothing #hide

# ![](modis_leaf_area_index.png)
#
# The rivers read as missing rather than as zero leaf area, because water is a surface this
# retrieval does not apply to, and the canopy between the channels sits where closed rainforest
# should. That is the practical reason to prefer this record where a canopy closure has been
# calibrated against it, and the reason the two records are worth keeping side by side: they
# report different quantities from different retrieval lineages, and each one's magnitude is only
# meaningful against the closure it was fitted to.

# ## The whole seasonal cycle
#
# All 46 periods, over the same box, at three years — which keeps the download tractable (46
# periods × 3 years × 2 tiles is ~280 granules and tens of minutes, where the default span would
# be six times that) and, as it turns out, is *not* enough here. That failure is worth showing
# rather than hiding: it is what sets the `years` default.

climatology = MODISLAIClimatology(years = 2017:2019)

build_lai_climatology!(climatology; region)

metadata = Metadata(:leaf_area_index; dataset = climatology, region)

# The series is built on the product's own grid, not the model grid. Regional means do not care,
# the gap fractions then stay comparable with the numbers above, and the temporal fill below wants
# it this way — filling before the regrid leaves the 2×2 stencil almost nothing to dilate.

Λclimatology = FieldTimeSeries(metadata; time_indices_in_memory = 4)

periods = eachindex(metadata.dates)

seasonal_mean = [nanmean(Λclimatology[n]) for n in periods]
seasonal_gaps = [gap_fraction(Λclimatology[n]) for n in periods]

# A single year of the same 46 periods is the honest comparison: the same cadence, the same
# screen, one year of observations instead of three.

single_year = Metadata(:leaf_area_index; dataset = MCD15A2H(), region,
                       dates = [DateTime(2019, 1, 1) + Day(8 * (n - 1)) for n in periods])

Λyear = FieldTimeSeries(single_year; time_indices_in_memory = 4)

year_mean = [nanmean(Λyear[n]) for n in periods]
year_gaps = [gap_fraction(Λyear[n]) for n in periods]

low_period, peak_period = argmin(seasonal_mean), argmax(seasonal_mean)

period_date(n) = DateTime(2018, 1, 1) + Day(8 * (n - 1))

@printf("\nSeasonal cycle over the 46 periods\n")
@printf("  annual low : Λ = %.2f  (%s)\n", seasonal_mean[low_period],
        Dates.format(period_date(low_period), "u dd"))
@printf("  peak       : Λ = %.2f  (%s)\n", seasonal_mean[peak_period],
        Dates.format(period_date(peak_period), "u dd"))
@printf("  peak / low : %.2f×  — evergreen canopy holds its leaf area year-round\n",
        seasonal_mean[peak_period] / seasonal_mean[low_period])

# That ratio needs one check before it is believed. A regional mean is taken over the cells that
# have a retrieval, and at three years the cloudiest periods have very few — so an apparent low
# may be a biased sample rather than a real minimum. Rebuilding the low period at the default span
# tests it directly.

build_lai_climatology!(deep; region, periods = low_period:low_period)

Λlow_deep = Field(Metadatum(:leaf_area_index; dataset = deep, region,
                            date = period_date(low_period)))

@printf("  the low period at %s : Λ = %.2f over %.0f%% of cells, against %.2f over %.0f%%\n",
        deep.years, nanmean(Λlow_deep), 100 * (1 - gap_fraction(Λlow_deep)),
        seasonal_mean[low_period], 100 * (1 - seasonal_gaps[low_period]))
@printf("  → peak / low on the deeper sample : %.2f×\n",
        seasonal_mean[peak_period] / nanmean(Λlow_deep))
@printf("  gap fraction : one year mean %.1f%% (worst %.1f%%) → composite mean %.1f%%\n",
        100 * mean(year_gaps), 100 * maximum(year_gaps), 100 * mean(seasonal_gaps))

# Whatever compositing leaves behind is filled along the time axis. A seasonal series is
# periodic, so December's neighbor is January: `fill_gaps!(cyclic = true)` interpolates across
# the wrap instead of extending the last valid value, which is what an open-series fill would do
# at both ends. The rivers stay missing at every period, so the fill cannot close them — and
# should not.

Λfilled = FieldTimeSeries(metadata; time_indices_in_memory = length(periods))

fill_gaps!(Λfilled; max_gap = 4, cyclic = true)

filled_gaps = [gap_fraction(Λfilled[n]) for n in periods]

@printf("  still missing after the cyclic fill : mean %.2f%%  (land-cover classes are %.2f%%)\n",
        100 * mean(filled_gaps), 100 * class_fraction)

day_of_year = 8 .* (periods .- 1) .+ 1

fig = Figure(size = (1400, 1050))

ax = Axis(fig[1, 1:3];
          xlabel = "Day of year", ylabel = "Λ (m² m⁻²)",
          title = "Regional-mean leaf area index")
scatterlines!(ax, day_of_year, year_mean, color = (:goldenrod, 0.8), markersize = 7,
              label = "2019 alone")
scatterlines!(ax, day_of_year, seasonal_mean, color = :seagreen, markersize = 9,
              label = "$(climatology.years) composite")
axislegend(ax, position = :lb)
ylims!(ax, 0, 1.15 * maximum(seasonal_mean))

# Cloud over the Amazon is strongly seasonal, so the gap fraction is a wet-season signal rather
# than noise — and three years is nowhere near enough to composite through it. Rebuilding just the
# cloudiest period at the default 2003–2019 span shows where the floor really is, for the price of
# one period's worth of extra granules instead of the whole series'.

wettest = argmax(seasonal_gaps)

build_lai_climatology!(deep; region, periods = wettest:wettest)

Λwettest = Field(Metadatum(:leaf_area_index; dataset = deep, region,
                           date = period_date(wettest)))

@printf("  cloudiest period (%s) : %.1f%% at %s → %.1f%% at %s\n",
        Dates.format(period_date(wettest), "u dd"),
        100 * seasonal_gaps[wettest], climatology.years,
        100 * gap_fraction(Λwettest), deep.years)

ax = Axis(fig[2, 1:3];
          xlabel = "Day of year", ylabel = "Cells with no retrieval (%)",
          title = "Gap fraction")
scatterlines!(ax, day_of_year, 100 .* year_gaps, color = (:goldenrod, 0.8), markersize = 7,
              label = "2019 alone")
scatterlines!(ax, day_of_year, 100 .* seasonal_gaps, color = :seagreen, markersize = 9,
              label = "$(climatology.years) composite")
scatterlines!(ax, day_of_year, 100 .* filled_gaps, color = :steelblue, markersize = 9,
              label = "after the cyclic fill")
scatter!(ax, [day_of_year[wettest]], [100 * gap_fraction(Λwettest)],
         color = :firebrick, marker = :star5, markersize = 20,
         label = "cloudiest period, $(deep.years)")
hlines!(ax, [100 * class_fraction], color = :gray40, linestyle = :dash,
        label = "land-cover classes (the floor)")
axislegend(ax, position = :rt)

# The two extremes on the model grid, interpolated from the *filled* series rather than re-read
# from the cache — filling first and regridding second is the order that keeps the stencil from
# dilating gaps, and this is what a simulation would actually be handed.
#
# The two periods inflate by different factors even though the stencil is the same, because
# dilation is a perimeter effect rather than an area one. The peak period's gaps are almost
# entirely the connected river network, which has little perimeter for its area and grows by
# about a third; the low period's are the scattered cells the temporal fill could not reach, and
# each isolated cell recruits its three neighbors. Gap *geometry*, not gap fraction, sets what
# the regrid costs.

function on_model_grid(native)
    field = Field{Center, Center, Nothing}(grid)
    interpolate!(field, native)
    return field
end

@printf("  worst period after the cyclic fill  : %.2f%%  (%d of %d still above 15%%)\n",
        100 * maximum(filled_gaps), count(>(0.15), filled_gaps), length(periods))
@printf("  on the model grid, after filling then regridding : %.2f%% at the low, %.2f%% at the peak\n",
        100 * gap_fraction(on_model_grid(Λfilled[low_period])),
        100 * gap_fraction(on_model_grid(Λfilled[peak_period])))

ax = Axis(fig[3, 1]; title = "Annual low, $(Dates.format(period_date(low_period), "u dd"))", axis_kw...)
heatmap!(ax, on_model_grid(Λfilled[low_period]), colormap = :viridis, colorrange = Λrange,
         nan_color = :midnightblue)

ax = Axis(fig[3, 2]; title = "Seasonal peak, $(Dates.format(period_date(peak_period), "u dd"))", axis_kw...)
plot = heatmap!(ax, on_model_grid(Λfilled[peak_period]), colormap = :viridis,
                colorrange = Λrange, nan_color = :midnightblue)
Colorbar(fig[3, 3], plot, label = "Λ (m² m⁻²)")

save("modis_leaf_area_index_seasonal.png", fig)
nothing #hide

# ![](modis_leaf_area_index_seasonal.png)
#
# A regional mean over a heavily gapped period is a mean over whichever cells were clear, not over
# the region — and the cloudiest period here turns out to be the apparent annual low, which is a
# warning rather than a coincidence. On the deeper composite the same period covers twice as many
# cells and reads higher, because the cells cloud hides are the denser ones. The seasonal *shape*
# survives; a peak-to-low ratio read off the three-year curve alone does not.
#
# The composite is the smoother of the two curves because each of its points draws on more
# observations, not because it has been smoothed. The annual range is small against the mean:
# closed evergreen canopy holds its leaf area through the year, where a temperate deciduous stand
# swings by close to an order of magnitude between leaf-off and full canopy. For a domain like
# this one a single peak-season field is therefore a defensible boundary condition, and
# `build_lai_climatology!(…; reducer = maximum)` produces exactly that; for a mid-latitude domain
# the seasonal series is not optional.
#
# The gap panel is the one that sets the `years` default. Three years reaches the land-cover floor
# through the dry season and fails badly through the wet one — the Amazon's wet-season cloud is
# persistent enough that three samples per period frequently contain no clear view at all. The
# full 2003–2019 span brings even the cloudiest period down to the floor, so over this region the
# default is not a nicety; it is what makes half the year usable. A temperate box needs far fewer
# years for the same result, which is worth knowing before budgeting a download.
#
# The residual speckle at the pixel scale is the retrieval's own uncertainty rather than anything
# the ingestion adds: the product reports it as `:leaf_area_index_uncertainty`, and it is close to
# independent between years, so it averages down as `√n`. Three years buys a factor 1.7 and the
# 2003–2019 default buys 3.6, which is why the deep composite above is visibly smoother than these
# panels.
