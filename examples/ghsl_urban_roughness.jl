# # Urban aerodynamic roughness from GHSL building morphometry
#
# Ingest the Global Human Settlement Layer (GHSL R2023A) mean building height and
# built-up fraction over a metropolitan area, derive the momentum roughness length
# `ℓᵐ` and zero-plane displacement `d` per cell with the urban morphometric closure
# (Macdonald 1998 / Kanda 2013), and render diagnostic maps + profiles.
#
# The model runs on the built fraction's native **10 m** grid (`GHSBuiltS(resolution = GHSBuiltS10m)`),
# so the maps resolve street-block structure. Building height is 100 m (the only resolution
# GHSL publishes); since `ℓᵐ` and `d` scale with height, the roughness magnitude carries
# 100 m structure textured by the 10 m coverage.
#
# Requirements:
#   * `using ArchGDAL` (for the World-Mollweide → EPSG:4326 reprojection)
#   * `using CairoMakie` for the figures
# GHSL is open access — no authentication. The first run downloads the intersecting
# Mollweide tiles (the 10 m built-surface tile is ~470 MB; cached afterwards).

using NumericalEarth
using Oceananigans
using ArchGDAL
using CairoMakie
using Statistics: mean

output_directory = joinpath(@__DIR__, "ghsl_urban_roughness_figures")
mkpath(output_directory)

# ## Region and target grid
# Inner London (City / Westminster / Docklands out to the inner suburbs): dense core →
# suburb gradient. We run on the built fraction's native 10 m grid (~3800 × 1560 cells) so
# the fine raster is used at full resolution rather than downsampled.
region = BoundingBox(longitude = (-0.28, 0.06), latitude = (51.42, 51.56))

# ## Ingest the building morphometry
# `λᵖ` — plan-area built fraction from the 10 m built-surface product (m²/cell → fraction),
# built directly on its native 10 m grid. `h` — mean net building height (ANBH, 100 m),
# interpolated up onto that grid. Both reprojected from Mollweide in the adapter.
λᵖ   = Field(Metadatum(:built_up_fraction; dataset = GHSBuiltS(resolution = GHSBuiltS10m), region), CPU())
grid = λᵖ.grid
h    = Field(Metadatum(:building_height; dataset = GHSBuiltH(), region), grid)

# ## Urban roughness closure
# One morphometric closure, two height distributions: `VariableHeight` (the default, after
# Kanda et al. 2013) parameterizes the building-height spread, while `UniformHeight` takes
# the idealized equal-height obstacle array of Macdonald et al. (1998). Both consume the
# same `(h, λᵖ)` fields.
FT = eltype(grid)
variable_height_roughness, variable_height_displacement =
    urban_roughness(h, λᵖ; closure = MorphometricRoughness(FT))
uniform_height_roughness, uniform_height_displacement =
    urban_roughness(h, λᵖ; closure = MorphometricRoughness(FT, height_distribution = UniformHeight()))

# ## Figures
function panel!(figure, position, title, field, colorrange, colormap, label)
    axis = Axis(figure[position...]; title, aspect = DataAspect())
    hidedecorations!(axis)
    heatmap_plot = heatmap!(axis, field; colorrange, colormap, nan_color = :gray82)
    Colorbar(figure[position[1], position[2] + 1], heatmap_plot; label, width = 11)
    return axis
end

# (1) The two height distributions side by side. Top row: the closure inputs (h, λᵖ).
# Middle rows: ℓᵐ and d from each (same columns → same closure). Bottom row: the anomaly,
# largest over the dense, height-heterogeneous core.
## Ranges are the 99th percentile of the variable-height fields, so the bulk of the
## domain resolves rather than saturating on the few supertall cells.
roughness_range = (0, 3.5)
displacement_range = (0, 40)
fig = Figure(size = (1150, 1500))
Label(fig[0, 1:4], "GHSL urban morphometry → roughness — Inner London (10 m built fraction)\nvariable-height vs uniform-height distribution"; fontsize = 20, font = :bold)
panel!(fig, (1, 1), "building height h (m)", h,  (0, 25), :inferno, "m")
panel!(fig, (1, 3), "built fraction λᵖ",     λᵖ, (0, 1),  :turbo,   "–")
panel!(fig, (2, 1), "ℓᵐ — variable height (m)", variable_height_roughness,    roughness_range,    :viridis, "m")
panel!(fig, (2, 3), "ℓᵐ — uniform height (m)",  uniform_height_roughness,     roughness_range,    :viridis, "m")
panel!(fig, (3, 1), "d — variable height (m)",  variable_height_displacement, displacement_range, :magma,   "m")
panel!(fig, (3, 3), "d — uniform height (m)",   uniform_height_displacement,  displacement_range, :magma,   "m")
panel!(fig, (4, 1), "ℓᵐ anomaly — variable − uniform (m)", variable_height_roughness - uniform_height_roughness, (-2.5, 2.5), :balance, "Δm")
panel!(fig, (4, 3), "d anomaly — variable − uniform (m)", variable_height_displacement - uniform_height_displacement, (-25, 25), :balance, "Δm")
save(joinpath(output_directory, "fig1_overview.png"), fig)

# (2) The diagnostic curve: ℓᵐ rises then falls with λᵖ (isolated → wake → skimming),
# peaking at intermediate coverage — binned mean over the domain.
built_fraction = interior(λᵖ, :, :, 1)
edges = range(0, 1; length = 21)
centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
function binned_mean(field)
    bins = [Float64[] for _ in centers]
    for (fraction, value) in zip(built_fraction, interior(field, :, :, 1))
        (isfinite(fraction) && isfinite(value)) || continue
        index = clamp(searchsortedlast(edges, fraction), 1, length(centers))
        push!(bins[index], value)
    end
    return [isempty(bin) ? NaN : mean(bin) for bin in bins]
end
fig = Figure(size = (760, 500))
axis = Axis(fig[1, 1]; xlabel = "built fraction λᵖ", ylabel = "domain-mean ℓᵐ (m)",
            title = "Roughness peaks at intermediate built fraction")
lines!(axis, centers, binned_mean(variable_height_roughness); linewidth = 3, label = "variable height")
lines!(axis, centers, binned_mean(uniform_height_roughness);  linewidth = 3, label = "uniform height")
axislegend(axis; position = :rt)
save(joinpath(output_directory, "fig2_roughness_vs_built_fraction.png"), fig)

# (3) West→east transect through the core. At 10 m each cell alternates building/street,
# so we average over a ~1 km latitudinal band (and lightly along-track) to read the
# core→suburb envelope rather than per-building spikes.
longitude = λnodes(grid, Center())
latitude = φnodes(grid, Center())
midpoint = size(grid, 2) ÷ 2
band = (midpoint - 50):(midpoint + 50)
moving_mean(series, width) =
    [mean(@view series[max(1, i - width):min(length(series), i + width)]) for i in eachindex(series)]
function band_profile(field)
    samples = interior(field, :, :, 1)
    return moving_mean([mean(filter(isfinite, @view samples[i, band])) for i in axes(samples, 1)], 25)
end
fig = Figure(size = (1100, 640))
Label(fig[0, 1:1], "Transect at $(round(latitude[midpoint], digits = 3))°N — core → inner suburb (1 km band mean)"; fontsize = 16, font = :bold)
height_axis = Axis(fig[1, 1]; ylabel = "h (m) / d (m)", xlabel = "longitude")
roughness_axis = Axis(fig[1, 1]; ylabel = "λᵖ / ℓᵐ (m)", yaxisposition = :right)
hidespines!(roughness_axis); hidexdecorations!(roughness_axis)
height_line = lines!(height_axis, longitude, band_profile(h); color = :black, linewidth = 2)
displacement_line = lines!(height_axis, longitude, band_profile(variable_height_displacement); color = :firebrick, linewidth = 2)
fraction_line = lines!(roughness_axis, longitude, band_profile(λᵖ); color = :seagreen, linewidth = 2)
roughness_line = lines!(roughness_axis, longitude, band_profile(variable_height_roughness); color = :navy, linewidth = 2)
axislegend(height_axis, [height_line, displacement_line, fraction_line, roughness_line],
           ["h", "d", "λᵖ", "ℓᵐ"]; position = :rt)
save(joinpath(output_directory, "fig3_transect.png"), fig)

# ## Sanity check against literature ranges
# Dense-core ℓᵐ is order 1 m and peaks at intermediate λᵖ rather than at maximum coverage;
# λᵖ → 0 reduces to a bare-soil roughness (~0.03 m). Under `VariableHeight` the
# displacement exceeds the mean building height, since it is referenced to the tallest
# element rather than to `h`.
#
# One caveat on reading these maps: the height-distribution and frontal-area regressions
# were fitted to 1 km urban districts, while this grid is 10 m. Per-cell `σʰ` and `hᵐᵃˣ`
# are therefore district statistics attached to street-width pixels, and the derived
# `hᵐᵃˣ` runs several times the cell's own building height. Aggregate onto a coarser
# target grid to keep the inputs and the closure at the same scale.
finite_mean(values) = mean(filter(isfinite, values))
roughness = interior(variable_height_roughness, :, :, 1)
displacement = interior(variable_height_displacement, :, :, 1)
dense = built_fraction .> 0.4
bare = built_fraction .< 0.02

@info "Dense-core (λᵖ > 0.4) morphometry" cells=count(dense) roughness=finite_mean(roughness[dense]) displacement=finite_mean(displacement[dense])
@info "Bare (λᵖ < 0.02) roughness" roughness=finite_mean(roughness[bare])
