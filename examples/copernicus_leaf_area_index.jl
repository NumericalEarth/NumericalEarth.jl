# # Leaf area index from the C3S 300 m record
#
# This example builds a leaf-area-index (``Λ``) series over a vegetated region from the
# Copernicus Climate Change Service 300 m climate data record, retrieved from Sentinel-3
# OLCI and SLSTR observations every ten days.
#
# The region is the central Amazon around the Rio Negro–Solimões confluence — the same box
# the bare-earth terrain and canopy-height examples use, so leaf area index and canopy
# height land on one grid, which is the pair an aerodynamic-roughness closure consumes.
#
# Two things distinguish this dataset from a coarse vegetation climatology. It resolves
# ``Λ`` at 300 m, fine enough to separate closed canopy from river margins and clearings,
# and it carries a per-pixel retrieval flag describing how much to trust each inversion.
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add NumericalEarth CDSAPI Oceananigans CairoMakie"
# ```
#
# You also need CDS API credentials in `~/.cdsapirc`, and to accept the *Copernicus
# Global Land product licence* on the
# [dataset page](https://cds.climate.copernicus.eu/datasets/satellite-lai-fapar).
# See <https://cds.climate.copernicus.eu/how-to-api> for setup instructions.

using NumericalEarth
using CDSAPI
using Oceananigans
using Dates
using Printf
using Statistics
using CairoMakie

# ## The region and the model grid
#
# The dataset object carries product identity only. Where and when to read comes from the
# `Metadata`: a `BoundingBox` for the region and the dates, which the Climate Data Store
# subsets server-side — the global 300 m grid holds 5.7 billion cells per dekad, so only
# the region is ever downloaded.

latitude = (-3.5, -2.4)
longitude = (-60.5, -59.0)

region = BoundingBox(; longitude, latitude)

grid = LatitudeLongitudeGrid(CPU(), Float32;
                             size = (150, 110),
                             longitude, latitude,
                             topology = (Bounded, Bounded, Flat))

dataset = CopernicusVegetation()

# ## A year of composites
#
# One composite every other month through 2021. `Metadata` takes the date list and hands
# back a `FieldTimeSeries` on the model grid, keeping only a couple of slices in memory at
# a time.

dates = [DateTime(2021, month, 20) for month in 1:2:11]

metadata = Metadata(:leaf_area_index; dataset, region, dates)

Λseries = FieldTimeSeries(metadata, grid; time_indices_in_memory = 2)

# The product reports fill values over pixels it could not retrieve, so every average has
# to skip `NaN`s.

slice(Λ, n) = Array(interior(Λ[n], :, :, 1))

nanmean(field) = mean(filter(!isnan, vec(field)))

seasonal_mean = [nanmean(slice(Λseries, n)) for n in 1:length(dates)]

for (date, Λ) in zip(dates, seasonal_mean)
    @printf("%s   Λ = %.2f\n", Dates.format(date, "yyyy-mm-dd"), Λ)
end

# ### Check 1 — seasonality
#
# Evergreen broadleaf forest holds its canopy year-round, so the peak should sit only
# modestly above the annual low. The same code over a temperate deciduous stand gives a
# ratio of several times, because its leaf-off state is near-bare.

low_index = argmin(seasonal_mean)
peak_index = argmax(seasonal_mean)

@printf("Lowest mean Λ  : %.2f  (%s)\n", seasonal_mean[low_index],
        Dates.format(dates[low_index], "yyyy-mm-dd"))
@printf("Peak mean Λ    : %.2f  (%s)\n", seasonal_mean[peak_index],
        Dates.format(dates[peak_index], "yyyy-mm-dd"))
@printf("Peak / low     : %.1f×\n", seasonal_mean[peak_index] / seasonal_mean[low_index])

# ### Check 2 — the rivers carry no retrieval
#
# Water is not a surface with zero leaf area; it is a surface where this retrieval does
# not apply. The unretrieved pixels should form a connected, season-independent shape —
# the river network — rather than moving around with the vegetation.

Λlow = slice(Λseries, low_index)
Λpeak = slice(Λseries, peak_index)

unretrieved_low = isnan.(Λlow)
unretrieved_peak = isnan.(Λpeak)

@printf("Unretrieved pixels: %.1f%% at the annual low, %.1f%% at peak\n",
        100 * count(unretrieved_low) / length(Λlow),
        100 * count(unretrieved_peak) / length(Λpeak))
@printf("Overlap between the two masks: %.1f%% of the low-month mask\n",
        100 * count(unretrieved_low .& unretrieved_peak) / max(count(unretrieved_low), 1))

# ### Check 3 — denser canopy carries more leaf area than open ground
#
# Splitting the peak-season field at its own median separates the closed-canopy pixels
# from river margins, clearings and disturbed ground. The two populations should stay
# ordered the same way at the annual low, and in a closed evergreen forest the gap should
# persist rather than collapse the way a deciduous canopy's would.

denser = .!unretrieved_peak .& (Λpeak .≥ median(filter(!isnan, vec(Λpeak))))
sparser = .!unretrieved_peak .& .!denser

peak_contrast = mean(Λpeak[denser]) - mean(Λpeak[sparser])
low_contrast = mean(filter(!isnan, Λlow[denser])) - mean(filter(!isnan, Λlow[sparser]))

@printf("Peak month:      denser %.2f vs sparser %.2f  (Δ = %.2f)\n",
        mean(Λpeak[denser]), mean(Λpeak[sparser]), peak_contrast)
@printf("Annual low:      denser %.2f vs sparser %.2f  (Δ = %.2f)\n",
        mean(filter(!isnan, Λlow[denser])), mean(filter(!isnan, Λlow[sparser])), low_contrast)

# ### Check 4 — what the retrieval flags remove
#
# Screening is off by default, so a read returns the product as distributed. Passing
# `screened_flags` drops the pixels whose retrieval the product marks as unusable —
# a fill value, an untrusted inversion, or unusable observations. It must only ever
# remove data, never change a value it keeps.

screened_dataset = CopernicusVegetation(screened_flags = unusable_retrieval_flags())

screened = Metadatum(:leaf_area_index; dataset = screened_dataset, region,
                     date = dates[peak_index])

Λscreened = Array(interior(Field(screened, grid), :, :, 1))

removed = isnan.(Λscreened) .& .!isnan.(Λpeak)
kept = .!isnan.(Λscreened)

@printf("Screen removes %d pixels (%.2f%%); values it keeps are unchanged: %s\n",
        count(removed), 100 * count(removed) / length(removed),
        all(Λscreened[kept] .≈ Λpeak[kept]))

# ## Figures

λ = λnodes(grid, Center(), Center(), Center())
φ = φnodes(grid, Center(), Center(), Center())

fig = Figure(size = (1250, 1180))

axis_kw = (xlabel = "Longitude (ᵒ)", ylabel = "Latitude (ᵒ)")
Λrange = (0, 5)

# The annual low and the peak side by side. The rivers read as missing in both.

ax = Axis(fig[1, 1]; title = "Λ, annual low ($(Dates.format(dates[low_index], "u yyyy")))",
          axis_kw...)
heatmap!(ax, λ, φ, Λlow, colormap = :viridis, colorrange = Λrange, nan_color = :midnightblue)

ax = Axis(fig[1, 2]; title = "Λ, peak ($(Dates.format(dates[peak_index], "u yyyy")))",
          axis_kw...)
plot = heatmap!(ax, λ, φ, Λpeak, colormap = :viridis, colorrange = Λrange,
                nan_color = :midnightblue)
Colorbar(fig[1, 3], plot, label = "Λ (m² m⁻²)")

# The annual trajectory of the regional mean, on an axis starting at zero so a weak cycle
# reads as weak rather than being visually amplified.

ax = Axis(fig[2, 1:2];
          xlabel = "Month of 2021",
          ylabel = "Λ (m² m⁻²)",
          title = "Regional-mean leaf area index",
          xticks = (Dates.month.(dates), Dates.format.(dates, "u")))
scatterlines!(ax, Dates.month.(dates), seasonal_mean, color = :seagreen, markersize = 12)
ylims!(ax, 0, 1.15 * maximum(seasonal_mean))

# Distributions at the annual low and the peak.

ax = Axis(fig[3, 1]; xlabel = "Λ (m² m⁻²)", ylabel = "Pixels",
          title = "Distribution, annual low vs peak")
hist!(ax, filter(!isnan, vec(Λlow)), bins = 40, color = (:goldenrod, 0.7), label = "annual low")
hist!(ax, filter(!isnan, vec(Λpeak)), bins = 40, color = (:seagreen, 0.7), label = "peak")
axislegend(ax, position = :lt)

# Where a retrieval is missing (the river network) and where the quality screen removes one.

mask = fill(0f0, size(Λpeak))
mask[removed] .= 1
mask[unretrieved_peak] .= 2

ax = Axis(fig[3, 2]; title = "Retrieved / screened / unretrieved", axis_kw...)
heatmap!(ax, λ, φ, mask, colormap = [:seagreen, :firebrick, :midnightblue],
         colorrange = (0, 2))

save("copernicus_leaf_area_index.png", fig)
nothing #hide

# ![](copernicus_leaf_area_index.png)
#
# The maps carry the river network as missing data at every date while the canopy between
# the channels stays high through the year, which is the behaviour to expect if the region,
# the half-cell coordinate convention, and the north→south latitude order are all being
# handled correctly: a coordinate error would smear the rivers or move them between dates.
