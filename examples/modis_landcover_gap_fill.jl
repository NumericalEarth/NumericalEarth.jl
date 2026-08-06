# # Land cover, and the gap fill it makes safe
#
# Compositing a leaf-area period across years removes cloud that is random from one year to
# the next. It cannot remove cloud that recurs at the same calendar period every year — the
# ITCZ, a monsoon, a windward slope — and neither can interpolating along the seasonal axis,
# because the neighboring periods are cloudy for the same reason. What is left has to be
# borrowed from somewhere else, and that is only safe if the borrowing knows what it is
# borrowing from: averaging across a forest/cropland boundary injects a biased value, and an
# aerodynamic-roughness closure downstream sits where more leaf area gives *less* roughness,
# so a smeared value there becomes a roughness error of the wrong sign.
#
# MODIS MCD12Q1 supplies the class field, at 500 m and on exactly the lattice the MCD15A2H
# leaf-area reads land on, so the two pair cell for cell with no aggregation in between.
#
# The region is the Missouri Ozarks: deciduous broadleaf forest, savanna, cropland and open
# water in one box, with a leaf-area cycle that swings by a factor of several between
# leaf-off and full canopy. That range is the point — a class-keyed temporal tolerance and a
# shape-preserving fill can only be falsified where the classes have *different* seasonal
# behavior, which a single-class evergreen box cannot show.
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add NumericalEarth ArchGDAL Oceananigans CairoMakie"
# ```
#
# `ArchGDAL` provides GDAL's HDF4 driver for the sinusoidal HDF-EOS granules, and a free
# [NASA Earthdata](https://urs.earthdata.nasa.gov) login supplies `EARTHDATA_USERNAME` and
# `EARTHDATA_PASSWORD`.

using NumericalEarth
using ArchGDAL
using Oceananigans
using Oceananigans.Units: Time   ## `Dates` exports a `Time` too, and it is not this one
using Dates
using Printf
using Statistics
using CairoMakie

latitude = (36.5, 37.5)
longitude = (-92.5, -91.5)

region = BoundingBox(; longitude, latitude)

nanmean(field) = mean(filter(!isnan, vec(Array(interior(field)))))
gap_fraction(field) = mean(isnan.(Array(interior(field))))

# ## What the two adapters deliver
#
# One 8-day composite first, because the problem the rest of the example solves is visible in
# it. A quarter of the box is missing, and the three contributions are different in kind.

date = DateTime(2019, 7, 4)

single = Metadatum(:leaf_area_index; dataset = MCD15A2H(), region, date)
codes = Metadatum(:landcover_code; dataset = MCD15A2H(), region, date)

Λsingle = Field(single)

in_band_fraction = mean(.!isnan.(Array(interior(Field(codes)))))

@printf("Single 8-day composite, %s\n", Dates.format(date, "yyyy-mm-dd"))
@printf("  a land-cover class, not a failed retrieval : %.2f%%\n", 100 * in_band_fraction)
@printf("  total missing                              : %.2f%%\n", 100 * gap_fraction(Λsingle))
@printf("  → attributable to cloud and quality        : %.2f%%\n",
        100 * (gap_fraction(Λsingle) - in_band_fraction))

# The class field comes from a separate annual product, read on the same lattice. Class
# *codes* are not interpolable — a bilinear regrid averages urban (13) against water (17)
# into permanent snow (15) — so `Field(metadatum)` lands them on the product's own grid and
# the check below is that nothing fractional came back.

land_cover = Metadatum(:landcover_class; dataset = MCD12Q1(), region, date = DateTime(2019))

classes = Field(land_cover)
class_codes = Array(interior(classes, :, :, 1))

present = sort(Int.(unique(filter(!isnan, vec(class_codes)))))
class_names = landcover_class_names(MCD12Q1())
class_name(code) = only(key for (key, value) in pairs(class_names) if value == code)

@printf("\nMCD12Q1 IGBP classes over the box, on the product's own 500 m grid\n")
for code in present
    @printf("  %-28s (%2d) : %5.2f%%\n", class_name(code), code,
            100 * class_fraction(class_codes, code))
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

fractions = class_fractions(class_codes, present, aggregation_factor)

fraction_sum = sum(values(fractions))

function model_field(data)
    field = Field{Center, Center, Nothing}(grid)
    interior(field, :, :, 1) .= Float32.(data)
    return field
end

@printf("  per-class fractions on the model grid sum to %.6f (largest departure %.2e)\n",
        mean(filter(!isnan, vec(fraction_sum))),
        maximum(abs, filter(!isnan, vec(fraction_sum)) .- 1))

# The two products name the non-vegetated surfaces independently: MCD12Q1 as classes 13, 15,
# 16 and 17, and MCD15A2H as the in-band codes it writes in place of a retrieval. They come
# from related land-cover inputs, so they should largely agree, and where they do not the
# disagreement localizes a geolocation or a vintage difference rather than a bug in either.

in_band_codes = Array(interior(Field(codes), :, :, 1))

non_vegetated = [!isnan(code) && Int(code) in igbp_non_vegetated_classes for code in class_codes]
in_band = .!isnan.(in_band_codes)

@printf("\nThe two non-vegetated masks\n")
@printf("  MCD12Q1 says non-vegetated : %d cells\n", count(non_vegetated))
@printf("  MCD15A2H wrote a class code: %d cells\n", count(in_band))
@printf("  they disagree on           : %d cells (%.2f%%)\n",
        count(non_vegetated .⊻ in_band), 100 * mean(non_vegetated .⊻ in_band))

axis_kw = (xlabel = "Longitude (ᵒ)", ylabel = "Latitude (ᵒ)")
Λrange = (0, 6)

## Only nine of the seventeen IGBP classes occur in this box, so the map is drawn over a
## compact index of the ones present and the colorbar spells them out. A 1-17 numeric scale
## would be unreadable and would spend most of its range on classes that are not here.
class_position = Dict(code => n for (n, code) in enumerate(present))
class_index = [isnan(code) ? NaN32 : Float32(class_position[Int(code)]) for code in class_codes]
class_labels = [replace(String(class_name(code)), "_" => " ") for code in present]

fig = Figure(size = (1600, 950))

ax = Axis(fig[1, 1]; title = "One 8-day composite, $(Dates.format(date, "u dd yyyy"))", axis_kw...)
plot = heatmap!(ax, Λsingle, colormap = :viridis, colorrange = Λrange,
                nan_color = :gray80)
Colorbar(fig[1, 2], plot, label = "Λ (m² m⁻²)")

ax = Axis(fig[1, 3]; title = "MCD12Q1 IGBP class (native 500 m grid)", axis_kw...)
plot = heatmap!(ax, native_field(class_index),
                colormap = cgrad(:tab10, length(present), categorical = true),
                colorrange = (0.5, length(present) + 0.5), nan_color = :gray80)
Colorbar(fig[1, 4], plot, ticks = (1:length(present), class_labels))

ax = Axis(fig[2, 1]; title = "Deciduous broadleaf fraction (model grid)", axis_kw...)
plot = heatmap!(ax, model_field(fractions[igbp_class_names.deciduous_broadleaf_forest]),
                colormap = :magma, colorrange = (0, 1))
Colorbar(fig[2, 2], plot, label = "area fraction")

ax = Axis(fig[2, 3]; title = "Woody savanna fraction (model grid)", axis_kw...)
plot = heatmap!(ax, model_field(fractions[igbp_class_names.woody_savanna]),
                colormap = :magma, colorrange = (0, 1))
Colorbar(fig[2, 4], plot, label = "area fraction")

save("modis_landcover_fields.png", fig)
nothing #hide

# ![](modis_landcover_fields.png)

# ## The climatology, and the chain that fills it
#
# Five years of 46 periods over this box is 230 granules and tens of minutes. That is enough
# to anchor the single-year fill below and enough to make the residual gaps visible; the
# 2003–2019 default is what a published product would use.

climatology = MODISLAIClimatology(years = 2015:2019)

build_lai_climatology!(climatology; region)

metadata = Metadata(:leaf_area_index; dataset = climatology, region)

periods = eachindex(metadata.dates)
period_date(n) = DateTime(2018, 1, 1) + Day(8 * (n - 1))

# The retained-retrieval count says how many of the five years survived screening in each
# cell. A period whose minimum is zero has cells no year observed, and a period whose mean is
# far below the number of years is a suspect minimum rather than a measured one.

Λraw = FieldTimeSeries(metadata; time_indices_in_memory = length(periods))

raw_gaps = [gap_fraction(Λraw[n]) for n in periods]
worst = argmax(raw_gaps)

retained = Field(retained_retrieval_metadatum(
    Metadatum(:leaf_area_index; dataset = climatology, region, date = period_date(worst))))

@printf("\nThe %s climatology, %d periods\n", climatology.years, length(periods))
@printf("  gap fraction  : mean %.2f%%, worst %.2f%% (%s)\n",
        100 * mean(raw_gaps), 100 * raw_gaps[worst],
        Dates.format(period_date(worst), "u dd"))
@printf("  retained count at the worst period : mean %.2f of %d, minimum %d\n",
        nanmean(retained), length(climatology.years),
        Int(minimum(Array(interior(retained)))))

# The chain runs on the assembled series. Each stage only writes where the previous one left
# a gap, and none of them rewrites an observation.
#
# The tolerance of the first stage is the class table. Over this box that is three periods
# for the savanna covering half of it, one for the deciduous forest covering a third — where
# more than one period across green-up puts the ramp in the wrong place — and zero for urban
# and water, which are never filled at all.

max_gap = class_maximum_gap(classes)

@printf("  class-keyed max_gap : %d–%d periods (a uniform table would use one number)\n",
        minimum(max_gap), maximum(max_gap))

# Running the stages cumulatively is what makes the gap-fraction panel below readable: each
# line is one more stage than the line above it, and each has to sit at or below it.

function run_chain(stages)
    Λ = FieldTimeSeries(metadata; time_indices_in_memory = length(periods))
    filled = fill_seasonal_gaps!(Λ, classes; cyclic = true, max_gap,
                                 valid_range = (0, 10),
                                 unfilled_classes = igbp_non_vegetated_classes,
                                 stages)
    return Λ, filled
end

Λtemporal, _ = run_chain((:temporal,))
Λscaled, _ = run_chain((:temporal, :scaled))
Λfilled, filled = run_chain((:temporal, :scaled, :class_mean))

temporal_gaps = [gap_fraction(Λtemporal[n]) for n in periods]
scaled_gaps = [gap_fraction(Λscaled[n]) for n in periods]
filled_gaps = [gap_fraction(Λfilled[n]) for n in periods]

non_vegetated_fraction = mean(non_vegetated)

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

before = horizontal(Λraw)
after = horizontal(Λfilled)
observed = .!isnan.(before)

@printf("\nChecks on the filled series\n")
@printf("  observed values are bit-identical  : %s\n", after[observed] == before[observed])
@printf("  every filled value is within [0, 10]: %s\n",
        all(v -> isnan(v) || 0 ≤ v ≤ 10, after))
@printf("  provenance accounts for every cell : %s\n",
        sum(count(==(code), filled.provenance) for code in values(gap_fill_provenance)) ==
        length(filled.provenance))

seasonal_mean = [nanmean(Λfilled[n]) for n in periods]
low, peak = argmin(seasonal_mean), argmax(seasonal_mean)

@printf("  annual low  : Λ = %.2f  (%s)\n", seasonal_mean[low],
        Dates.format(period_date(low), "u dd"))
@printf("  seasonal peak: Λ = %.2f  (%s)\n", seasonal_mean[peak],
        Dates.format(period_date(peak), "u dd"))
@printf("  peak / low  : %.2f×  — a deciduous box, not an evergreen one\n",
        seasonal_mean[peak] / seasonal_mean[low])

# The provenance and reach fields are what make the result defensible rather than opaque. A
# fill that reached four hundred kilometers is not the same datum as one that reached fifteen.
#
# Read the shares below before drawing conclusions about the donor stages from this box:
# five years of compositing over a mid-latitude continental interior already brings the
# residual to within a fraction of a percent of the non-vegetated floor, so almost everything
# is observed and the donor stages barely fire. That is the favorable case, not the case
# they exist for. The single year below is where they carry the result.

provenance_share = Dict(name => mean(filled.provenance .== code)
                        for (name, code) in pairs(gap_fill_provenance))

@printf("\nHow every cell-period got its value\n")
for name in keys(gap_fill_provenance)
    @printf("  %-11s : %.2f%%\n", name, 100 * provenance_share[name])
end

reached = filter(>(0), vec(filled.reach))
if !isempty(reached)
    @printf("  donor reach: median %d blocks, maximum %d of 16 (%.0f km at 32 cells a block)\n",
            round(Int, median(reached)), maximum(reached), 15 * maximum(reached))
end

# Land cover changed inside the span for some cells, and their composite is then a mean over
# two surfaces. A single year's label at 500 m is not reliable enough to call that on, so the
# test is persistence at each end of the span rather than a first-versus-last difference.
# This is a flag, not a correction: which surface a simulation wants is the caller's call.
#
# Five years only leaves room for a two-year window at each end, which is a weak persistence
# test — much of what it flags here will be label noise rather than change. Read the fraction
# as an upper bound, and widen the window on a longer span.

annual_classes = stack(Array(interior(Field(Metadatum(:landcover_class; dataset = MCD12Q1(),
                                                      region, date = DateTime(year))), :, :, 1))
                       for year in climatology.years)

changed = landcover_change_flag(annual_classes; window = 2)

@printf("\nLand cover over %s\n", climatology.years)
@printf("  cells whose class is stable at both ends and differs : %.2f%%\n",
        100 * mean(changed))

## The unfilled code is 255 so it cannot be mistaken for a stage; give it a plottable slot.
provenance_map(slice) =
    native_field(ifelse.(slice .== gap_fill_provenance.unfilled, UInt8(4), slice))

provenance_colors = cgrad([:gray70, :goldenrod, :seagreen, :steelblue, :firebrick], 5,
                           categorical = true)
provenance_ticks = (0:4, ["observed", "temporal", "scaled", "class mean", "unfilled"])

day_of_year = 8 .* (periods .- 1) .+ 1

# The product first: the two extremes of the filled climatology, which is the field a
# simulation is actually handed. Everything else on this page is a diagnostic of how it was
# made.

fig = Figure(size = (1500, 900))

ax = Axis(fig[1, 1]; axis_kw...,
          title = "Filled climatology, annual low ($(Dates.format(period_date(low), "u dd")))")
heatmap!(ax, native_field(after[:, :, low]), colormap = :viridis, colorrange = Λrange,
         nan_color = :gray80)

ax = Axis(fig[1, 2]; axis_kw...,
          title = "Filled climatology, seasonal peak ($(Dates.format(period_date(peak), "u dd")))")
plot = heatmap!(ax, native_field(after[:, :, peak]), colormap = :viridis,
                colorrange = Λrange, nan_color = :gray80)
Colorbar(fig[1, 3], plot, label = "Λ (m² m⁻²)")

ax = Axis(fig[2, 1]; xlabel = "Day of year", ylabel = "Λ (m² m⁻²)",
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

save("modis_landcover_gap_fill_climatology.png", fig)
nothing #hide

# ![](modis_landcover_gap_fill_climatology.png)
#
# The peak and the low share one color scale, which is the point of putting them side by
# side: this canopy swings by more than a factor of eight, and the gray cells are the water
# and built-up surfaces that are never filled.
#
# The gap panel is the honest result rather than a flattering one. Five years of compositing
# over a mid-latitude continental interior lands within a tenth of a percent of the
# non-vegetated floor before any filling, so the three filled lines sit on top of each other
# just above it and only the gray unfilled line has any structure. This box does not need the
# donor stages. The single year below does, and that is where they can be seen working.

# ## The time axis
#
# A model asks a `FieldTimeSeries` for a value at its own clock time, not at a composite
# stamp, so the mapping from date to time has to be right. An 8-day composite is stamped at
# the *start* of its window, which means the value it holds represents four days later; the
# read path applies that offset, and a series that did not would hand every interpolation a
# four-day phase lead.

## A representative cell for every curve below: deciduous broadleaf, nearest the box center.
center = size(class_codes) .÷ 2
deciduous = findall(==(Float32(igbp_class_names.deciduous_broadleaf_forest)), class_codes)
i, j = Tuple(argmin(cell -> (cell[1] - center[1])^2 + (cell[2] - center[2])^2, deciduous))

times = Λfilled.times

@printf("\nThe time axis of the 46-period cyclic series\n")
@printf("  first stamp sits at day %.1f, not day 0 — half a composite period\n",
        times[1] / 86400)
@printf("  the 46 periods span %.1f days, so the cycle closes on the year\n",
        (times[end] - times[1] + (times[end] - times[end-1])) / 86400)

# Reading at a node returns the node, reading between two returns their mean, and the series
# is continuous across the turn of the year.

sample(t) = Array(interior(Λfilled[Time(t)], :, :, 1))[i, j]

@printf("  at a composite's own time   : %.6f vs the composite's %.6f\n",
        sample(times[10]), after[i, j, 10])
@printf("  midway between two          : %.6f vs their mean %.6f\n",
        sample((times[10] + times[11]) / 2), (after[i, j, 10] + after[i, j, 11]) / 2)
@printf("  across the turn of the year : %.6f then %.6f\n",
        sample(365 * 86400 - 1), sample(1))

green_up = 60:0.5:160

fig = Figure(size = (1200, 450))
ax = Axis(fig[1, 1]; xlabel = "Day of year", ylabel = "Λ (m² m⁻²)",
          title = "Interpolating the series through green-up")
lines!(ax, collect(green_up), [sample((day - 1) * 86400) for day in green_up],
       color = :seagreen, label = "interpolated")
scatter!(ax, times ./ 86400 .+ 1, after[i, j, :], color = :firebrick, markersize = 9,
         label = "composites, at their window centers")
xlims!(ax, first(green_up), last(green_up))
axislegend(ax, position = :lt)

save("modis_landcover_gap_fill_time_axis.png", fig)
nothing #hide

# ![](modis_landcover_gap_fill_time_axis.png)

# ## One specific year, anchored on the climatology
#
# A climatology is not the only thing this chain produces. A single year's seasonal cycle is
# the same dataset with calendar dates instead of a nominal year, and the same estimator with
# a different donor: that cell's own climatological curve, scaled to the year's own level.
# There is no second dataset type — `MCD15A2H` plus a date window plus the `anchor` keyword.

target = Metadata(:leaf_area_index; dataset = MCD15A2H(), region,
                  dates = (DateTime(2019, 1, 1), DateTime(2019, 12, 31)))

Λ2019 = FieldTimeSeries(target; time_indices_in_memory = length(target.dates))

raw_2019 = horizontal(Λ2019)

single_year_gaps = [gap_fraction(Λ2019[n]) for n in eachindex(target.dates)]
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

anchored = fill_seasonal_gaps!(Λ2019, classes; anchor = Λfilled, anchor_periods,
                               max_gap, cyclic = false, valid_range = (0, 10),
                               unfilled_classes = igbp_non_vegetated_classes)

filled_2019 = horizontal(Λ2019)
climatology_2019 = after[:, :, anchor_periods]

departure = filter(!isnan, vec(filled_2019 .- climatology_2019))

@printf("  gap fraction after filling  : %.2f%%\n", 100 * mean(isnan.(filled_2019)))
@printf("  2019 minus the climatology  : mean %+.3f, RMS %.3f m² m⁻²\n",
        mean(departure), sqrt(mean(abs2, departure)))
@printf("  → the anchor is scaled, not copied : %s\n", sqrt(mean(abs2, departure)) > 0.05)

# The averaging cadence is a plain function on the assembled series, not another product.
# Composites are window averages already and 8-day periods do not nest inside months, so the
# samples that straddle each edge have to be split by their days of overlap; the unweighted
# alternative is worst exactly where the field moves fastest.

Λbimonthly, edges = time_average(Λ2019, target, Month(2))

windows = 1:length(edges)-1

unweighted = [mean(filter(!isnan, vec(filled_2019[:, :, n])))
              for n in eachindex(target.dates)]

bimonthly_mean = [nanmean(Λbimonthly[w]) for w in windows]

green_up_window = findfirst(w -> Dates.month(edges[w]) == 3, windows)
green_up_samples = [n for n in eachindex(target.dates)
                    if edges[green_up_window] ≤ target.dates[n] < edges[green_up_window + 1]]

# The bracketing composite runs a few days past 31 December, so the tiling ends in a short
# stub window rather than being truncated mid-composite. That is the honest thing to do with
# it: a two-month mean of eight days of data is not a two-month mean.

@printf("\nTwo-month means of the 2019 series\n")
for w in windows
    @printf("  %s – %s : Λ = %.3f\n", Dates.format(edges[w], "u dd"),
            Dates.format(edges[w + 1], "u dd"), bimonthly_mean[w])
end
@printf("  green-up window, overlap-weighted %.4f vs unweighted %.4f (%.1f%% apart)\n",
        bimonthly_mean[green_up_window], mean(unweighted[green_up_samples]),
        100 * abs(bimonthly_mean[green_up_window] - mean(unweighted[green_up_samples])) /
        bimonthly_mean[green_up_window])

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
         colorrange = Λrange, nan_color = :gray80)

ax = Axis(fig[1, 2]; axis_kw..., title = "2019-$worst_date after the chain")
plot = heatmap!(ax, native_field(filled_2019[:, :, worst_2019]), colormap = :viridis,
                colorrange = Λrange, nan_color = :gray80)
Colorbar(fig[1, 3], plot, label = "Λ (m² m⁻²)")

ax = Axis(fig[1, 4]; axis_kw..., title = "What produced each cell")
plot = heatmap!(ax, provenance_map(anchored.provenance[:, :, worst_2019]),
                colormap = provenance_colors, colorrange = (-0.5, 4.5))
Colorbar(fig[1, 5], plot, ticks = provenance_ticks)

ax = Axis(fig[2, 1:3]; xlabel = "Days from 1 January 2019", ylabel = "Λ (m² m⁻²)",
          title = "One deciduous cell: 2019 against the $(climatology.years) climatology")
lines!(ax, elapsed.(target.dates), climatology_2019[i, j, :], color = :gray40,
       linestyle = :dash, label = "climatology (the anchor)")
lines!(ax, elapsed.(target.dates), filled_2019[i, j, :], color = :seagreen, linewidth = 2,
       label = "2019, filled")
scatter!(ax, elapsed.(target.dates)[isnan.(raw_2019[i, j, :])],
         filled_2019[i, j, isnan.(raw_2019[i, j, :])], color = :goldenrod, markersize = 11,
         label = "periods the chain supplied")
scatterlines!(ax, [elapsed(edges[w]) for w in windows],
              [Array(interior(Λbimonthly[w], :, :, 1))[i, j] for w in windows],
              color = :firebrick, markersize = 11, label = "two-month means")
axislegend(ax, position = :lt)

ax = Axis(fig[2, 4:5]; axis_kw..., title = "2019 minus the climatology, seasonal peak")
plot = heatmap!(ax, native_field(filled_2019[:, :, peak_2019] .- climatology_2019[:, :, peak_2019]),
                colormap = :balance, colorrange = (-2, 2))
Colorbar(fig[2, 6], plot, label = "ΔΛ (m² m⁻²)")

save("modis_landcover_gap_fill_single_year.png", fig)
nothing #hide

# ![](modis_landcover_gap_fill_single_year.png)
#
# Three quarters of that period is missing before the fill and none of it after, and the
# result is spatially coherent rather than patched — the provenance map shows the temporal
# stage carrying most of it with the scaled donor taking coherent blocks where the overcast
# was too wide to bridge.
#
# The single-cell panel also shows what a single year costs. The sharp mid-August excursion is
# an *observed* composite, not a fill: the recommended screen keeps cloud shadow and aerosol,
# which a multi-year composite averages away and one year does not. Filling cannot repair
# that, and a simulation reading a single year should expect it.

# ## Scoring the fill
#
# The numbers above say how much was filled, not how well. Withholding values the chain would
# otherwise have kept, running it on the damaged copy, and comparing the estimates against
# what was removed turns "good quality" into a per-class score — and needs no new downloads,
# because the granules are already cached.

scores = gap_fill_denial(Λfilled, classes; samples_per_class = 100, cyclic = true,
                         max_gap, valid_range = (0, 10),
                         unfilled_classes = igbp_non_vegetated_classes)

@printf("\nData denial over the %s climatology\n", climatology.years)
@printf("  %-28s %8s %10s %8s %10s\n", "class", "withheld", "estimated", "R²", "CV(RMSE)")
for row in scores
    @printf("  %-28s %8d %10d %8.2f %10.2f\n", class_name(row.class), row.withheld,
            row.estimated, row.R², row.cv_rmse)
end

# Compositing seventeen — or here five — years before the donors are drawn is what separates
# these scores from the published single-date experiments they are comparable with: both the
# donors and the truth are multi-year means, carrying roughly `√n` less retrieval noise than a
# single retrieval does. A stage that scores *worse* than the single-date literature is a
# stage that is wrong, not a stage that is working hard.
#
# The remaining caveat this pipeline's design makes necessary, and the one to carry forward to
# whatever consumes the field: compare the roughness distribution over **filled** cells against
# **observed** ones. If they differ systematically, the gap fill is biasing the field it exists
# to serve, and every check above can still pass.

# ## What the model is handed
#
# Everything the chain leaves missing is water, urban, permanent snow or barren — measured
# below, it is *all* of it, with no vegetated cell left dry in either treatment. Those cells
# are not missing a value. Their leaf area per unit ground area is zero, and that is the value
# to write.
#
# Scoring came first on purpose: `gap_fill_denial` samples cells that carry a value, so zeroed
# water entering the sample as truth the chain reproduces exactly would inflate every score.

for (name, series) in ("climatology" => Λfilled, "2019" => Λ2019)
    left = isnan.(horizontal(series))
    vegetated_left = count(t -> left[t] && !non_vegetated[t[1], t[2]], CartesianIndices(left))
    @printf("\n%s: %.3f%% left unfilled, of which %d cell-periods are vegetated\n",
            name, 100 * mean(left), vegetated_left)

    zero_non_vegetated!(series, classes)
    @printf("  after zeroing the non-vegetated classes : %d NaN remaining\n",
            count(isnan, horizontal(series)))
end

# Zeroing before the regrid rather than after is what keeps the model grid clean: a `NaN` at a
# shoreline dilates into its neighbors through the interpolation stencil, while a zero blends
# into the cell mean correctly, because leaf area is already per unit *ground* area. A model
# cell half lake and half forest genuinely carries half the forest's leaf area.

Λmodel = Field{Center, Center, Nothing}(grid)
Oceananigans.Fields.interpolate!(Λmodel, Λfilled[peak])

@printf("\nOn the %d x %d model grid at the seasonal peak\n", size(grid, 1), size(grid, 2))
@printf("  cells with no value : %d\n", count(isnan, Array(interior(Λmodel))))
@printf("  regional mean Λ     : %.3f m² m⁻²\n", nanmean(Λmodel))

# The class field has to travel with it. Zero says there is no canopy; it does not say whether
# the surface is a lake or a car park, and those want roughness lengths four orders of
# magnitude apart. A canopy closure handed this field alone would read a city as smoother than
# a wheat field — which is why `class_fractions` above is part of the deliverable and not a
# diagnostic.
