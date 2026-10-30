# # Leaf area index and land cover from MODIS
#
# This example builds a leaf-area-index (LAI) boundary condition from MODIS MCD15A2H —
# 500 m combined Terra + Aqua composites, one every eight days since 2002 — together with the
# MCD12Q1 land-cover classes that make it safe to fill.
#
# Two things about this record shape the example. A single 8-day composite is not usable as a
# boundary condition, because cloud takes a fifth of a good scene and most of a bad one; the fix
# is to composite the same period across years, which [`MODISLAIClimatology`](@ref) does, giving
# 46 periods per year as a cyclic `FieldTimeSeries`. And MCD15A2H targets *true* leaf area index,
# applying per-biome foliage clumping inside its own retrieval — the quantity a canopy closure
# calibrated on MODIS expects.
#
# Compositing across years removes cloud that is random from one year to the next. It cannot
# remove cloud that recurs at the same calendar period every year — the ITCZ, a monsoon, a
# windward slope — and neither can interpolating along the seasonal axis, because the
# neighboring periods are cloudy for the same reason. What is left has to be borrowed from
# somewhere else, and that is only safe if the borrowing knows what it is borrowing from:
# averaging across a forest/cropland boundary injects a biased value, and over a dense canopy,
# where an aerodynamic-roughness closure gives more leaf area *less* roughness, a smeared value
# becomes a roughness error of the wrong sign. MODIS MCD12Q1 supplies the class field, at 500 m
# and on exactly the lattice the MCD15A2H leaf-area reads land on, so the two pair cell for cell
# with no aggregation in between.
#
# The region is Cyprus. Its classes have *different* seasonal behavior: the crops of the
# Mesaoria plain and the grassland around them green up with the winter rain and dry out by
# early summer, while the pine forest of the Troodos mountains holds its leaf area through the
# year. That contrast is the point — a class-keyed temporal tolerance and a shape-preserving
# fill can only be falsified where the classes behave differently, which a single-class box
# cannot show. The weather does the rest: winters are cloudy and summers almost cloudless, so
# the gaps arrive exactly when the crops are green.
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
using NumericalEarth.DataWrangling: fill_seasonal_gaps!, gap_fill_provenance, gap_fill_denial,
                                    window_bounds, class_fractions, valid_area_mean!
using NumericalEarth.DataWrangling.MODISLand: landcover_class_names, igbp_class_names,
                                              igbp_non_vegetated_classes, class_maximum_gap,
                                              zero_non_vegetated!, period_index,
                                              retained_retrieval_metadatum
using ArchGDAL
using Oceananigans
using Oceananigans.Units: Time   ## `Dates` exports a `Time` too, and it is not this one
using Dates
using Printf
using Statistics
using CairoMakie

# ## The region
#
# The dataset object carries product identity only; where and when to read comes from the
# `Metadata`. Granules are sinusoidal tiles, so a region is discovered from NASA's metadata
# repository, mosaicked, and reprojected onto a regional latitude-longitude window at download
# time — a global read is never attempted.

latitude = (34.5, 35.75)
longitude = (32.2, 34.65)

region = BoundingBox(; longitude, latitude)

# The MCD12Q1 land cover, on the same 500 m lattice as the leaf-area composites, also says which
# cells are land. Most of this box is sea, so every gap fraction below is a fraction of the island.

land_cover = Metadatum(:landcover_class; dataset = MCD12Q1(), region, date = DateTime(2019))

classes = Field(land_cover)
class_codes = Array(interior(classes, :, :, 1))
land = [!isnan(code) && code != igbp_class_names.water for code in class_codes]

nanmean(field) = mean(filter(!isnan, vec(Array(interior(field)))))
gap_fraction(field) = mean(isnan, Array(interior(field, :, :, 1))[land])
land_fraction(mask) = mean(mask[land])

# ## Why a single composite looks so sparse
#
# One 8-day composite first, because the problem the rest of the example solves is visible in
# it. A fifth of the island is missing in this February composite, and the contributions are
# different in kind. The product
# writes a land-cover code in place of LAI where the surface is outside the retrieval's scope,
# available as `:landcover_code`; urban and barren labels do not establish zero leaf area, and
# water is unretrievable by design. The rest is cloud and quality screening.

date = DateTime(2019, 2, 18)

single_composite = Metadatum(:leaf_area_index; dataset = MCD15A2H(), region, date)
in_band_codes = Metadatum(:landcover_code; dataset = MCD15A2H(), region, date)

single_composite_leaf_area_index = Field(single_composite)

in_band_fraction = land_fraction(isfinite.(Array(interior(Field(in_band_codes), :, :, 1))))

@printf("Single 8-day composite, %s\n", Dates.format(date, "yyyy-mm-dd"))
@printf("  land-cover codes, without a retrieval       : %.2f%%\n", 100 * in_band_fraction)
@printf("  total missing                              : %.2f%%\n",
        100 * gap_fraction(single_composite_leaf_area_index))
@printf("  → attributable to cloud and quality        : %.2f%%\n",
        100 * (gap_fraction(single_composite_leaf_area_index) - in_band_fraction))

# Cloud is what compositing across years removes. The same period in another year is usually
# clear, so the composite's gap collapses onto the permanent land-cover classes and stops there,
# which is the signature of the compositing working rather than of it papering over anything.
# `years` defaults to 2003–2019, the span over which both Terra and Aqua held their equatorial
# crossing times.

deep = MODISLAIClimatology(years = 2003:2019)

period = period_index(date, MCD15A2H())

build_lai_climatology!(deep; region, periods = period:period)

deep_composite = Metadatum(:leaf_area_index; dataset = deep, region,
                           date = Metadata(:leaf_area_index; dataset = deep, region).dates[period])

deep_composite_leaf_area_index = Field(deep_composite)

@printf("\n%s composite of the same period\n", deep.years)
@printf("  total missing : %.2f%%  (the land-cover codes are %.2f%%)\n",
        100 * gap_fraction(deep_composite_leaf_area_index), 100 * in_band_fraction)
@printf("  mean LAI      : %.2f m² m⁻²\n", nanmean(deep_composite_leaf_area_index))

# The residual speckle at the pixel scale is retrieval error rather than anything the ingestion
# adds. The part of it that varies from year to year averages down with the number of years
# retained; the systematic part, such as the retrieval saturating over closed canopy, survives
# any number of years. `:leaf_area_index_uncertainty` measures neither on its own: it is the
# spread of the retrieval's acceptable solutions for one observation.

# The class field comes from a separate annual product, read on the same lattice. Class
# *codes* are not interpolable — a bilinear regrid averages urban (13) against water (17)
# into permanent snow (15) — so `Field(metadatum)` lands them on the product's own grid and
# the check below is that nothing fractional came back.

present = sort(Int.(unique(filter(!isnan, vec(class_codes)))))
class_names = landcover_class_names(MCD12Q1())
class_name(code) = only(key for (key, value) in pairs(class_names) if value == code)

@printf("\nMCD12Q1 IGBP classes over the island, on the product's own 500 m grid\n")
for code in filter(!=(igbp_class_names.water), present)
    @printf("  %-28s (%2d) : %5.2f%%\n", class_name(code), code, 100 * land_fraction(class_codes .== code))
end
@printf("  no fractional codes : %s\n",
        all(code -> isnan(code) || code == round(code), class_codes))

# The safe way to carry the same information onto a model grid is per-class *fractions*.
# Aggregating whole 500 m cells into each model cell keeps them exact, they are continuous
# fields rather than codes, and they say more than a dominant class does: a cell that is 60%
# forest and 40% crop is not a forest cell.
#
# The model grid is therefore built from the product's own window rather than from the
# request — which carries a margin cell on each side, so it is not the nominal 240 across.
# Eleven 500 m cells a side puts it at about 5 km, a plausible mesoscale cell, and gives each
# fraction 121 samples rather than a handful of quantized steps.

native = native_grid(land_cover)

function native_field(data)
    field = Field{Center, Center, Nothing}(native)
    interior(field, :, :, 1) .= Float32.(data)
    return field
end

aggregation_factor = 11

grid = LatitudeLongitudeGrid(CPU(), Float32;
                             size = size(class_codes) .÷ aggregation_factor,
                             longitude = extrema(λnodes(native, Face(), Center(), Center())),
                             latitude = extrema(φnodes(native, Center(), Face(), Center())),
                             topology = (Bounded, Bounded, Flat))

fractions = class_fractions(grid, classes, MCD12Q1())

fraction_sum = sum(Array(interior(fraction)) for fraction in fractions)

@printf("  per-class fractions on the model grid sum to %.6f (largest departure %.2e)\n",
        mean(filter(!isnan, vec(fraction_sum))),
        maximum(abs, filter(!isnan, vec(fraction_sum)) .- 1))

# The two products name the non-vegetated surfaces independently: MCD12Q1 as classes 13, 15,
# 16 and 17, and MCD15A2H as the in-band codes it writes in place of a retrieval. They come
# from related land-cover inputs, so they should largely agree, and where they do not the
# disagreement localizes a geolocation or a vintage difference rather than a bug in either.

non_vegetated = [!isnan(code) && Int(code) in igbp_non_vegetated_classes for code in class_codes]
in_band = .!isnan.(Array(interior(Field(in_band_codes), :, :, 1)))

@printf("\nThe two non-vegetated masks\n")
@printf("  MCD12Q1 says non-vegetated : %d cells\n", count(non_vegetated))
@printf("  MCD15A2H wrote a class code: %d cells\n", count(in_band))
@printf("  they disagree on           : %d cells (%.2f%%)\n",
        count(non_vegetated .⊻ in_band), 100 * land_fraction(non_vegetated .⊻ in_band))

axis_kw = (xlabel = "Longitude (ᵒ)", ylabel = "Latitude (ᵒ)")
leaf_area_index_range = (0, 4)

## Only twelve of the seventeen IGBP classes occur in this box, so the map is drawn over a
## compact index of the ones present and the colorbar spells them out. A 1-17 numeric scale
## would be unreadable and would spend most of its range on classes that are not here.
class_position = Dict(code => n for (n, code) in enumerate(present))
class_index = [isnan(code) ? NaN32 : Float32(class_position[Int(code)]) for code in class_codes]
class_labels = [replace(String(class_name(code)), "_" => " ") for code in present]

fig = Figure(size = (1600, 1400))

ax = Axis(fig[1, 1]; title = "One 8-day composite, $(Dates.format(date, "u dd yyyy"))", axis_kw...)
heatmap!(ax, single_composite_leaf_area_index, colormap = :viridis,
         colorrange = leaf_area_index_range, nan_color = :gray80)

ax = Axis(fig[1, 3]; title = "$(deep.years) composite, same period", axis_kw...)
plot = heatmap!(ax, deep_composite_leaf_area_index, colormap = :viridis,
                colorrange = leaf_area_index_range, nan_color = :gray80)
Colorbar(fig[1, 4], plot, label = "LAI (m² m⁻²)")

ax = Axis(fig[2, 1]; title = "MCD12Q1 IGBP class (native 500 m grid)", axis_kw...)
plot = heatmap!(ax, native_field(class_index),
                colormap = cgrad(:tab10, length(present), categorical = true),
                colorrange = (0.5, length(present) + 0.5), nan_color = :gray80)
Colorbar(fig[2, 2], plot, ticks = (1:length(present), class_labels))

ax = Axis(fig[3, 1]; title = "Cropland fraction (model grid)", axis_kw...)
plot = heatmap!(ax, fractions.cropland,
                colormap = :magma, colorrange = (0, 1))
Colorbar(fig[3, 2], plot, label = "area fraction")

ax = Axis(fig[3, 3]; title = "Evergreen needleleaf forest fraction (model grid)", axis_kw...)
plot = heatmap!(ax, fractions.evergreen_needleleaf_forest,
                colormap = :magma, colorrange = (0, 1))
Colorbar(fig[3, 4], plot, label = "area fraction")

save("modis_leaf_area_index_fields.png", fig)
nothing #hide

# ![](modis_leaf_area_index_fields.png)

# ## The climatology, and the chain that fills it
#
# Five years of 46 periods over this box is 230 granules and tens of minutes. That is enough
# to anchor the single-year fill below and enough to make the residual gaps visible; the
# 2003–2019 default is what a published product would use.

climatology = MODISLAIClimatology(years = 2015:2019)

build_lai_climatology!(climatology; region)

metadata = Metadata(:leaf_area_index; dataset = climatology, region)

periods = eachindex(metadata.dates)

# The retained-retrieval count says how many of the five years survived screening in each
# cell. A period whose minimum is zero has cells no year observed, and a period whose mean is
# far below the number of years is a suspect minimum rather than a measured one.

raw_leaf_area_index = FieldTimeSeries(metadata; time_indices_in_memory = length(periods))

raw_gaps = [gap_fraction(raw_leaf_area_index[n]) for n in periods]
worst = argmax(raw_gaps)

retained = Field(retained_retrieval_metadatum(
    Metadatum(:leaf_area_index; dataset = climatology, region, date = metadata.dates[worst])))

@printf("\nThe %s climatology, %d periods\n", climatology.years, length(periods))
@printf("  gap fraction  : mean %.2f%%, worst %.2f%% (%s)\n",
        100 * mean(raw_gaps), 100 * raw_gaps[worst],
        Dates.format(metadata.dates[worst], "u dd"))
retained_on_land = Array(interior(retained, :, :, 1))[land]

@printf("  retained count at the worst period : mean %.2f of %d, minimum %d\n",
        mean(retained_on_land), length(climatology.years), Int(minimum(retained_on_land)))

# The chain runs on the assembled series. Each stage only writes where the previous one left
# a gap, and none of them rewrites an observation.
#
# The tolerance of the first stage is the class table. Over Cyprus that is one period for the
# grassland and cropland covering four fifths of the island — where more than one period across
# the winter green-up puts the ramp in the wrong place — three for the savanna, six for the
# Troodos pine, and zero for urban and water, which are never filled at all.

max_gap = class_maximum_gap(classes)

@printf("  class-keyed max_gap : %d–%d periods (a uniform table would use one number)\n",
        minimum(max_gap), maximum(max_gap))

filled_leaf_area_index = FieldTimeSeries(metadata; time_indices_in_memory = length(periods))
fill_record = fill_seasonal_gaps!(filled_leaf_area_index, classes; cyclic = true, max_gap,
                                  valid_range = (0, 10),
                                  unfilled_classes = igbp_non_vegetated_classes)

# Every stage writes only where the stages before it left a gap, so the provenance codes give
# the gap each stage leaves behind. That is what makes the gap-fraction panel below readable:
# each line is one more stage than the line above it, and each has to sit at or below it.

gaps_after(stage) = [land_fraction(fill_record.provenance[:, :, n] .> stage) for n in periods]

temporal_gaps = gaps_after(gap_fill_provenance.temporal)
scaled_gaps = gaps_after(gap_fill_provenance.scaled)
filled_gaps = gaps_after(gap_fill_provenance.class_mean)

non_vegetated_fraction = land_fraction(non_vegetated)

@printf("\nGap fraction, cumulative by stage (mean over the %d periods)\n", length(periods))
@printf("  composited only          : %.2f%%\n", 100 * mean(raw_gaps))
@printf("  + temporal interpolation : %.2f%%\n", 100 * mean(temporal_gaps))
@printf("  + scaled donor curve     : %.2f%%\n", 100 * mean(scaled_gaps))
@printf("  + localized class mean   : %.2f%%\n", 100 * mean(filled_gaps))
@printf("  the non-vegetated floor  : %.2f%%\n", 100 * non_vegetated_fraction)
@printf("  each stage is monotone   : %s\n",
        all(raw_gaps .≥ temporal_gaps .≥ scaled_gaps .≥ filled_gaps))

# Three checks with no figure, each of which catches a real bug and none of which a plausible
# map would reveal.

## A horizontal series carries a singleton level, which the plots and the checks below have
## no use for.
horizontal(series) = dropdims(Array(interior(series)); dims = 3)

before = horizontal(raw_leaf_area_index)
after = horizontal(filled_leaf_area_index)
observed = .!isnan.(before)

@printf("\nChecks on the filled series\n")
@printf("  observed values are bit-identical  : %s\n", after[observed] == before[observed])
@printf("  every filled value is within [0, 10]: %s\n",
        all(v -> isnan(v) || 0 ≤ v ≤ 10, after))
@printf("  provenance accounts for every cell : %s\n",
        sum(count(==(code), fill_record.provenance) for code in values(gap_fill_provenance)) ==
        length(fill_record.provenance))

seasonal_mean = [nanmean(filled_leaf_area_index[n]) for n in periods]
low, peak = argmin(seasonal_mean), argmax(seasonal_mean)

@printf("  annual low  : LAI = %.2f  (%s)\n", seasonal_mean[low],
        Dates.format(metadata.dates[low], "u dd"))
@printf("  seasonal peak: LAI = %.2f  (%s)\n", seasonal_mean[peak],
        Dates.format(metadata.dates[peak], "u dd"))
@printf("  peak / low  : %.2f×  — a winter-green island, not an evergreen one\n",
        seasonal_mean[peak] / seasonal_mean[low])

# The provenance and reach fields are what make the result defensible rather than opaque. A
# fill that reached four hundred kilometers is not the same datum as one that reached fifteen.
#
# Read the shares below before drawing conclusions about the donor stages from this
# climatology: five years of compositing bring the residual down to the non-vegetated floor
# from spring to autumn, when the sky is clear, and only the winter periods leave cloud to fill,
# so the stages supply about one cell-period in a hundred. That is the favorable case, not the
# case they exist for. The single year below is where they carry the result.

provenance_share = Dict(name => mean(land_fraction(fill_record.provenance[:, :, n] .== code) for n in periods)
                        for (name, code) in pairs(gap_fill_provenance))

@printf("\nHow every cell-period got its value\n")
for name in keys(gap_fill_provenance)
    @printf("  %-11s : %.2f%%\n", name, 100 * provenance_share[name])
end

reached = filter(>(0), vec(fill_record.reach))
if !isempty(reached)
    @printf("  donor reach: median %d blocks, maximum %d of 16 (%.0f km at 32 cells a block)\n",
            round(Int, median(reached)), maximum(reached), 15 * maximum(reached))
end

## The unfilled code is 255 so it cannot be mistaken for a stage; give it a plottable slot.
provenance_map(slice) =
    native_field(ifelse.(slice .== gap_fill_provenance.unfilled, UInt8(4), slice))

provenance_colors = cgrad([:gray70, :goldenrod, :seagreen, :steelblue, :firebrick], 5,
                           categorical = true)
provenance_ticks = (0:4, ["observed", "temporal", "scaled", "class mean", "unfilled"])

day_of_year = dayofyear.(metadata.dates)

# The product first: the two extremes of the filled climatology, which is the field a
# simulation is actually handed. Everything else on this page is a diagnostic of how it was
# made.

fig = Figure(size = (1500, 900))

ax = Axis(fig[1, 1]; axis_kw...,
          title = "Filled climatology, annual low ($(Dates.format(metadata.dates[low], "u dd")))")
heatmap!(ax, native_field(after[:, :, low]), colormap = :viridis,
         colorrange = leaf_area_index_range, nan_color = :gray80)

ax = Axis(fig[1, 2]; axis_kw...,
          title = "Filled climatology, seasonal peak ($(Dates.format(metadata.dates[peak], "u dd")))")
plot = heatmap!(ax, native_field(after[:, :, peak]), colormap = :viridis,
                colorrange = leaf_area_index_range, nan_color = :gray80)
Colorbar(fig[1, 3], plot, label = "LAI (m² m⁻²)")

ax = Axis(fig[2, 1]; xlabel = "Day of year", ylabel = "LAI (m² m⁻²)",
          title = "Regional-mean seasonal cycle")
scatterlines!(ax, day_of_year, seasonal_mean, color = :seagreen, markersize = 8)
scatter!(ax, day_of_year[[low, peak]], seasonal_mean[[low, peak]], color = :firebrick,
         markersize = 14)

ax = Axis(fig[2, 2]; xlabel = "Day of year", ylabel = "Cells with no value (%)",
          title = "Gap fraction, cumulative by stage")
scatterlines!(ax, day_of_year, 100 .* raw_gaps, color = :gray40, markersize = 6,
              label = "composited only")
scatterlines!(ax, day_of_year, 100 .* temporal_gaps, color = :goldenrod, markersize = 6,
              label = "+ temporal")
scatterlines!(ax, day_of_year, 100 .* scaled_gaps, color = :seagreen, markersize = 6,
              label = "+ scaled donor")
scatterlines!(ax, day_of_year, 100 .* filled_gaps, color = :steelblue, markersize = 6,
              label = "+ class mean")
hlines!(ax, [100 * non_vegetated_fraction], color = :firebrick, linestyle = :dash,
        label = "non-vegetated floor")
axislegend(ax, position = :rt, labelsize = 11)

save("modis_leaf_area_index_climatology.png", fig)
nothing #hide

# ![](modis_leaf_area_index_climatology.png)
#
# The peak and the low share one color scale, which is the point of putting them side by
# side: the island's leaf area more than triples between the November low and the
# late-February peak, almost all of it on the Mesaoria plain, while the Troodos barely changes.
# The gray cells are the sea and the built-up surfaces that are never filled; Nicosia is the
# hole in the middle.
#
# The gap panel follows the weather. From spring to autumn the composited line sits on the
# non-vegetated floor, because five clear summers leave nothing to fill. The winter periods are
# where the stages work, with up to 8.5% of the island missing in early December and the
# temporal stage taking most of the excess. The filled line ends a few tenths of a percent
# below the floor because the floor counts every urban and barren cell, and the product
# retrieves leaf area over some of them.

# ## The time axis
#
# A model asks a `FieldTimeSeries` for a value at its own clock time, not at a composite
# stamp, so the mapping from date to time has to be right. An 8-day composite is stamped at
# the *start* of its window, which means the value it holds represents four days later; the
# read path applies that offset, and a series that did not would hand every interpolation a
# four-day phase lead.

## A representative cell for every curve below: cropland, nearest the box center.
center = size(class_codes) .÷ 2
cropland = findall(==(Float32(igbp_class_names.cropland)), class_codes)
i, j = Tuple(argmin(cell -> (cell[1] - center[1])^2 + (cell[2] - center[2])^2, cropland))

times = filled_leaf_area_index.times

@printf("\nThe time axis of the 46-period cyclic series\n")
@printf("  first stamp sits at day %.1f, not day 0 — half a composite period\n",
        times[1] / 86400)
@printf("  the 46 periods span %.1f days, so the cycle closes on the year\n",
        (times[end] - times[1] + (times[end] - times[end-1])) / 86400)

# Reading at a node returns the node, reading between two returns their mean, and the series
# is continuous across the turn of the year.

sample(t) = filled_leaf_area_index[i, j, 1, Time(t)]

@printf("  at a composite's own time   : %.6f vs the composite's %.6f\n",
        sample(times[10]), after[i, j, 10])
@printf("  midway between two          : %.6f vs their mean %.6f\n",
        sample((times[10] + times[11]) / 2), (after[i, j, 10] + after[i, j, 11]) / 2)
@printf("  across the turn of the year : %.6f then %.6f\n",
        sample(365 * 86400 - 1), sample(1))

green_up = 290:0.5:365

fig = Figure(size = (1200, 450))
ax = Axis(fig[1, 1]; xlabel = "Day of year", ylabel = "LAI (m² m⁻²)",
          title = "Interpolating the series through the autumn green-up")
lines!(ax, collect(green_up), [sample((day - 1) * 86400) for day in green_up],
       color = :seagreen, label = "interpolated")
scatter!(ax, times ./ 86400 .+ 1, after[i, j, :], color = :firebrick, markersize = 9,
         label = "composites, at their window centers")
xlims!(ax, first(green_up), last(green_up))
axislegend(ax, position = :lt)

save("modis_leaf_area_index_time_axis.png", fig)
nothing #hide

# ![](modis_leaf_area_index_time_axis.png)

# ## One specific year, anchored on the climatology
#
# A climatology is not the only thing this chain produces. A single year's seasonal cycle is
# the same dataset with calendar dates instead of a nominal year, and the same estimator with
# a different donor: that cell's own climatological curve, scaled to the year's own level.
# There is no second dataset type — `MCD15A2H` plus a date window plus the `anchor` keyword.

target = Metadata(:leaf_area_index; dataset = MCD15A2H(), region,
                  dates = (DateTime(2019, 1, 1), DateTime(2019, 12, 31)))

single_year_leaf_area_index = FieldTimeSeries(target; time_indices_in_memory = length(target.dates))

raw_2019 = horizontal(single_year_leaf_area_index)

single_year_gaps = [gap_fraction(single_year_leaf_area_index[n]) for n in eachindex(target.dates)]
worst_2019 = argmax(single_year_gaps)

# The window expands to the product's own composites and is *bracketed*, so it runs one
# composite past 31 December: `period_index` is what maps each of them onto the climatology's
# periods, and it is the one thing the anchored fill needs to know about the calendar.

anchor_periods = [period_index(date, MCD15A2H()) for date in target.dates]

@printf("\n2019 alone against the %s composite\n", climatology.years)
@printf("  gap fraction before filling : %.2f%% for 2019, %.2f%% for the composite\n",
        100 * mean(single_year_gaps), 100 * mean(raw_gaps))

# One year of 8-day composites is temporally autocorrelated — an overcast spell covers the
# whole of several consecutive periods — so the residual entering the donor stages is much
# larger than in the climatology case, and their accuracy becomes the product's accuracy
# rather than a correction to it.

anchored = fill_seasonal_gaps!(single_year_leaf_area_index, classes;
                               anchor = filled_leaf_area_index, anchor_periods,
                               max_gap, cyclic = false, valid_range = (0, 10),
                               unfilled_classes = igbp_non_vegetated_classes)

filled_2019 = horizontal(single_year_leaf_area_index)
climatology_2019 = after[:, :, anchor_periods]

departure = filter(!isnan, vec(filled_2019 .- climatology_2019))

@printf("  gap fraction after filling  : %.2f%%\n",
        100 * mean(land_fraction(isnan.(filled_2019[:, :, n])) for n in eachindex(target.dates)))
@printf("  2019 minus the climatology  : mean %+.3f, RMS %.3f m² m⁻²\n",
        mean(departure), sqrt(mean(abs2, departure)))
@printf("  → the anchor is scaled, not copied : %s\n", sqrt(mean(abs2, departure)) > 0.05)

# The averaging cadence is a plain function on the assembled series, not another product.
# Composites are window averages already and 8-day periods do not nest inside months, so the
# samples that straddle each edge have to be split by their days of overlap; the unweighted
# alternative is worst exactly where the field moves fastest.

bounds = window_bounds(target)
bimonthly_leaf_area_index = time_average(single_year_leaf_area_index, bounds, Month(2))

edges = [first(bounds):Month(2):last(bounds); last(bounds)]
windows = eachindex(bimonthly_leaf_area_index.times)

unweighted = [mean(filter(!isnan, vec(filled_2019[:, :, n])))
              for n in eachindex(target.dates)]

bimonthly_mean = [nanmean(bimonthly_leaf_area_index[w]) for w in windows]

dry_down_window = findfirst(w -> Dates.month(edges[w]) == 3, windows)
dry_down_samples = [n for n in eachindex(target.dates)
                    if edges[dry_down_window] ≤ target.dates[n] < edges[dry_down_window + 1]]

# The bracketing composite runs a few days past 31 December, so the tiling ends in a short
# stub window rather than being truncated mid-composite. That is the honest thing to do with
# it: a two-month mean of eight days of data is not a two-month mean.

@printf("\nTwo-month means of the 2019 series\n")
for w in windows
    @printf("  %s – %s : LAI = %.3f\n", Dates.format(edges[w], "u dd"),
            Dates.format(edges[w + 1], "u dd"), bimonthly_mean[w])
end
@printf("  spring dry-down window, overlap-weighted %.4f vs unweighted %.4f (%.1f%% apart)\n",
        bimonthly_mean[dry_down_window], mean(unweighted[dry_down_samples]),
        100 * abs(bimonthly_mean[dry_down_window] - mean(unweighted[dry_down_samples])) /
        bimonthly_mean[dry_down_window])

## The bracketing composite lands in the next January, so measure days from the window's own
## start rather than wrapping its day-of-year back to 1.
elapsed(date) = Dates.value(Dates.Day(DateTime(date) - first(target.dates))) + 1

peak_2019 = findfirst(==(peak), anchor_periods)

worst_date = Dates.format(target.dates[worst_2019], "u dd")

fig = Figure(size = (1600, 1150))

# The top row is the fill doing its job: the same period before and after, side by side, and
# what produced each cell.

ax = Axis(fig[1, 1]; axis_kw...,
          title = "2019-$(worst_date) as read ($(round(100 * single_year_gaps[worst_2019], digits = 1))% missing)")
heatmap!(ax, native_field(raw_2019[:, :, worst_2019]), colormap = :viridis,
         colorrange = leaf_area_index_range, nan_color = :gray80)

ax = Axis(fig[1, 2]; axis_kw..., title = "2019-$worst_date after the chain")
plot = heatmap!(ax, native_field(filled_2019[:, :, worst_2019]), colormap = :viridis,
                colorrange = leaf_area_index_range, nan_color = :gray80)
Colorbar(fig[1, 3], plot, label = "LAI (m² m⁻²)")

ax = Axis(fig[1, 4]; axis_kw..., title = "What produced each cell")
plot = heatmap!(ax, provenance_map(anchored.provenance[:, :, worst_2019]),
                colormap = provenance_colors, colorrange = (-0.5, 4.5))
Colorbar(fig[1, 5], plot, ticks = provenance_ticks)

ax = Axis(fig[2, 1:3]; xlabel = "Days from 1 January 2019", ylabel = "LAI (m² m⁻²)",
          title = "One cropland cell: 2019 against the $(climatology.years) climatology")
lines!(ax, elapsed.(target.dates), climatology_2019[i, j, :], color = :gray40,
       linestyle = :dash, label = "climatology (the anchor)")
lines!(ax, elapsed.(target.dates), filled_2019[i, j, :], color = :seagreen, linewidth = 2,
       label = "2019, filled")
scatter!(ax, elapsed.(target.dates)[isnan.(raw_2019[i, j, :])],
         filled_2019[i, j, isnan.(raw_2019[i, j, :])], color = :goldenrod, markersize = 11,
         label = "periods the chain supplied")
scatterlines!(ax, [elapsed(edges[w]) for w in windows],
              [Array(interior(bimonthly_leaf_area_index[w], :, :, 1))[i, j] for w in windows],
              color = :firebrick, markersize = 11, label = "two-month means")
axislegend(ax, position = :lt)

ax = Axis(fig[2, 4:5]; axis_kw..., title = "2019 minus the climatology, seasonal peak")
plot = heatmap!(ax, native_field(filled_2019[:, :, peak_2019] .- climatology_2019[:, :, peak_2019]),
                colormap = :balance, colorrange = (-2, 2))
Colorbar(fig[2, 6], plot, label = "ΔLAI (m² m⁻²)")

save("modis_leaf_area_index_single_year.png", fig)
nothing #hide

# ![](modis_leaf_area_index_single_year.png)
#
# Nearly half of the island is missing in that January period before the fill, and none of its
# vegetated land after. The result is spatially coherent rather than patched — the provenance
# map shows the temporal stage carrying most of it, with the scaled donor taking coherent
# blocks where the overcast was too wide to bridge.
#
# The lower panels show why a single year is worth having. This cropland cell runs well above
# its climatology through the 2019 winter peak, and the difference map at the peak shows the
# same across the Mesaoria plain. The anchor is scaled to the year rather than copied, so the
# periods the chain supplied follow 2019's level, not the five-year mean.

# ## Scoring the fill
#
# The numbers above say how much was filled, not how well. Withholding observed values from
# the composited series, running the chain on the damaged copy, and comparing the estimates
# against what was removed turns "good quality" into a per-class score — and needs no new
# downloads, because the granules are already cached. The series scored is the one as read,
# before the chain: a filled series would hand the chain its own estimates back as truth.

scores = gap_fill_denial(raw_leaf_area_index, classes; samples_per_class = 100, cyclic = true,
                         max_gap, valid_range = (0, 10),
                         unfilled_classes = igbp_non_vegetated_classes)

@printf("\nData denial over the %s climatology\n", climatology.years)
@printf("  %-28s %8s %10s %8s %10s\n", "class", "withheld", "estimated", "R²", "CV(RMSE)")
for row in filter(row -> row.estimated > 0, scores)
    @printf("  %-28s %8d %10d %8.2f %10.2f\n", class_name(row.class), row.withheld,
            row.estimated, row.R², row.relative_root_mean_square_error)
end

# Compositing seventeen — or here five — years before the donors are drawn is what separates
# these scores from the published single-date experiments they are comparable with: both the
# donors and the truth are multi-year means, in which the part of the retrieval error that
# varies from year to year has averaged down, though the systematic part has not. Comparisons
# with single-date studies also depend on their validation data and gap patterns.
#
# Cropland scores lowest by CV(RMSE): its winter peak varies most from year to year, as the
# 2019 panel above shows.
#
# Withheld entries can adjoin existing gaps or other withheld entries. Short, isolated gaps
# may still dominate these scores, making them optimistic for long, clustered cloud gaps;
# they do not bound accuracy on those gaps. `stages` scores the donor stages on their own.
#
# The remaining caveat this pipeline's design makes necessary, and the one to carry forward to
# whatever consumes the field: compare the roughness distribution over **filled** cells against
# **observed** ones. If they differ systematically, the gap fill is biasing the field it exists
# to serve, and every check above can still pass.

# ## What the model is handed
#
# The remaining gaps are counted by class below. This example explicitly assigns zero leaf
# area to urban, permanent snow and barren classes, and leaves water missing. Zero in urban
# and barren cells is a modeling assumption; those classes can contain vegetation.

water = class_codes .== igbp_class_names.water

for (name, series) in ("climatology" => filled_leaf_area_index,
                       "2019" => single_year_leaf_area_index)
    left = isnan.(horizontal(series))
    vegetated_left = count(t -> left[t] && !non_vegetated[t[1], t[2]], CartesianIndices(left))
    @printf("\n%s: %.3f%% left unfilled, of which %d cell-periods are vegetated\n",
            name, 100 * mean(land_fraction(left[:, :, n]) for n in axes(left, 3)), vegetated_left)

    zero_non_vegetated!(series, classes)
    remaining = isnan.(horizontal(series))
    @printf("  after applying the class map : %d NaN remaining, %d of them off water\n",
            count(remaining), count(remaining .& .!water))
end

# The regrid is conservative and normalized by the land area that holds a value, so each model
# cell takes the area mean of the 500 m land cells under it: a coastal model cell half sea and
# half cropland carries the cropland's leaf area per unit of its land. Urban, snow and barren
# count toward that land at zero; the sea does not count at all, and a cell that is all water
# has no value.

model_grid_leaf_area_index = valid_area_mean!(Field{Center, Center, Nothing}(grid),
                                              filled_leaf_area_index[peak])

@printf("\nOn the %d x %d model grid at the seasonal peak\n", size(grid, 1), size(grid, 2))
@printf("  cells with no value : %d\n", count(isnan, Array(interior(model_grid_leaf_area_index))))
@printf("  regional mean LAI   : %.3f m² m⁻² of land\n", nanmean(model_grid_leaf_area_index))

# The class field has to travel with it. This field says how much leaf area the land carries; it
# does not say how much of a cell is land, nor whether its leafless land is a car park or bare
# rock, and those want roughness lengths orders of magnitude apart. A canopy closure handed this
# field alone would read a city as smoother than a wheat field, which is why `class_fractions`
# above is part of the deliverable and not a diagnostic.
