# # Real ETH canopy in the DSM bare-earth workflow — central Amazon
#
# This is the `bare_earth_terrain.jl` scene (Rio Negro–Solimões confluence near Manaus),
# but with the **measured ETH canopy height** in place of the synthetic elevation-gated
# canopy that example stands in with. The same canopy field then feeds two consumers:
#
#   1. **Bare-earth terrain** — `bare_earth_elevation(z_DSM, h)` = `max(z_DSM − h, 0)`.
#   2. **Aerodynamic roughness** — `compute_aerodynamic_roughness!` (Raupach drag partition).
#
# The canopy the DSM overstates the ground by is exactly the canopy that sets the surface
# roughness — the terrain subtraction and the roughness closure share one measured field.
# A final section verifies the roughness closure itself against idealized cases.
#
# DSM stand-in: ETOPO 2022 (token-free; `GLO30()` is the commercial-use 30 m surface model).
# The ETH product is 10 m; `canopy_height_field` area-averages it onto the land grid straight
# from the windowed COGs (anonymous `/vsicurl/`). Needs `using ArchGDAL` and `using CairoMakie`.

using NumericalEarth
using NumericalEarth.DataWrangling.ETHSentinel2Canopy: ETHSentinel2CanopyHeight, canopy_height_field
using NumericalEarth.Lands: DragPartitionRoughness, compute_aerodynamic_roughness!,
                            canopy_roughness, canopy_wind_ratio,
                            canopy_drag_parameters, representative_canopy_height
using Oceananigans
using ArchGDAL   # activates the COG-read extension used by canopy_height_field
using CairoMakie

# ## Domain and DSM (identical to `bare_earth_terrain.jl`)
latitude  = -3.5, -2.4
longitude = -60.5, -59.0
grid = LatitudeLongitudeGrid(CPU(); latitude, longitude, size = (1500, 1100),
                             topology = (Bounded, Bounded, Flat))   # ~110 m across the basin

z_dsm = regrid_topography(grid; dataset = ETOPO2022())

# ## Real canopy height on the model grid
#
# `canopy_height_field` area-averages the ETH 10 m COG pixels within each model cell
# (coarse-graining, not point interpolation), reading only the windowed COG blocks via
# `/vsicurl/`. Canopy height over non-forest is a valid `0`; only the no-data byte is
# masked to `NaN`.
canopy_height = canopy_height_field(grid, ETHSentinel2CanopyHeight())

# ## (1) Bare-earth terrain — DSM minus the measured canopy
z_bare  = bare_earth_elevation(z_dsm, canopy_height)
removed = compute!(Field(z_dsm - z_bare))      # the canopy lift removed from the terrain

# ## (2) Roughness — the same canopy through the Raupach drag partition
# The basin is evergreen broadleaf forest, a uniform dense canopy (leaf area index 5) here. The closure
# returns the momentum roughness length `ℓᵐ` and the zero-plane displacement `d`.
# `DragPartitionRoughness()` defaults to the `:evergreen_broadleaf_forest` class.
leaf_area_index = Field{Center, Center, Nothing}(grid); set!(leaf_area_index, 5)
ℓᵐ, d = Field{Center, Center, Nothing}(grid), Field{Center, Center, Nothing}(grid)
compute_aerodynamic_roughness!(ℓᵐ, d, DragPartitionRoughness(),
                               (; leaf_area_index, canopy_height), grid)

# ## Figures
outdir = joinpath(@__DIR__, "eth_canopy_bare_earth_figures"); mkpath(outdir)
finite(f) = filter(isfinite, interior(f))
zmax = maximum(finite(z_dsm))

function heat!(fig, pos, title, field, crange, cmap)
    ax = Axis(fig[pos...]; title, aspect = DataAspect()); hidedecorations!(ax)
    hm = heatmap!(ax, field; colorrange = crange, colormap = cmap, nan_color = :gray82)
    Colorbar(fig[pos[1], pos[2] + 1], hm; width = 11)
end

fig = Figure(size = (1500, 950))
Label(fig[0, 1:4], "Real ETH canopy in the DSM bare-earth workflow — Rio Negro–Solimões, Amazon"; fontsize = 19, font = :bold)
heat!(fig, (1, 1), "DSM elevation (ETOPO stand-in)", z_dsm,         (0, zmax), :terrain)
heat!(fig, (1, 3), "canopy height h (ETH)",          canopy_height, (0, maximum(finite(canopy_height))), :YlGn)
heat!(fig, (2, 1), "bare-earth DTM = DSM − canopy",  z_bare,        (0, zmax), :terrain)
heat!(fig, (2, 3), "ℓᵐ roughness from the same h",   ℓᵐ,            (0, maximum(finite(ℓᵐ))), :viridis)
save(joinpath(outdir, "fig1_canopy_dsm_roughness.png"), fig)

# A west–east transect at 3.0°S: the DSM rides a canopy height above the bare-earth line.
# A `view` into each field plots straight through the Oceananigans Makie extension, which
# supplies the longitude coordinates.
jrow = size(grid, 2) ÷ 2
fig = Figure(size = (1200, 440))
ax = Axis(fig[1, 1]; xlabel = "longitude (°)", ylabel = "elevation (m)",
          title = "DSM vs bare-earth along 3°S: the gap is the ETH canopy")
lines!(ax, view(z_dsm,   :, jrow, 1); linewidth = 2, label = "DSM")
lines!(ax, view(z_bare,  :, jrow, 1); linewidth = 2, label = "bare-earth")
lines!(ax, view(removed, :, jrow, 1); linewidth = 2, color = :seagreen, label = "removed canopy")
axislegend(ax; position = :rt)
save(joinpath(outdir, "fig2_transect.png"), fig)

@info "canopy + DSM" canopy_height_max = round(maximum(finite(canopy_height)), digits = 1) dsm_range = round.((minimum(finite(z_dsm)), zmax), digits = 1) removed_canopy_max = round(maximum(finite(removed)), digits = 1) roughness_mean = round(sum(finite(ℓᵐ)) / length(finite(ℓᵐ)), digits = 2)

# ## (3) Is the roughness closure right? Idealized verification
#
# The `ℓᵐ` map above is only as trustworthy as the closure behind it. Driving that same
# closure with idealized inputs recovers its mathematical signatures and its physical
# magnitudes — the behavior expected from Raupach (1994) and the class values tabulated by
# Borak et al. (2025). `canopy_roughness(closure, leaf_area_index, h)` is the scalar form of
# the operator that filled the `ℓᵐ` field above; it also accepts a `Field` or a `FieldTimeSeries`.

closure = DragPartitionRoughness()                  # evergreen broadleaf forest defaults
skimming_ceiling = closure.parameters.critical_leaf_area_index
h₀ = representative_canopy_height(Float64, :evergreen_broadleaf_forest)

fig = Figure(size = (1500, 1080), fontsize = 15)
Label(fig[0, 1:2], "Drag-partition roughness closure — idealized verification"; fontsize = 21, font = :bold)

# **(1) Leaf-area-index response.** Sweeping the leaf area index at the reference height (lengths as a
# fraction of canopy height h): displacement `d/h` climbs monotonically toward the canopy top,
# while roughness `ℓᵐ/h` is non-monotonic — it peaks then *keeps falling* as the canopy
# densifies. That falling branch is the signature of skimming flow: a denser canopy is
# aerodynamically smoother. The skimming ceiling caps only the wind ratio `u★/Uh`; both lengths
# use the full index.
leaf_area_indices = range(0, 6; length = 241)
displacement_fraction = [canopy_roughness(closure, 𝒜, h₀)[2] / h₀ for 𝒜 in leaf_area_indices]
roughness_fraction    = [canopy_roughness(closure, 𝒜, h₀)[1] / h₀ for 𝒜 in leaf_area_indices]

ax1 = Axis(fig[1, 1]; xlabel = "leaf area index", ylabel = "fraction of canopy height h",
           title = "(1) leaf-area-index response: d/h monotone, ℓᵐ/h skims past the ceiling")
lines!(ax1, leaf_area_indices, displacement_fraction; linewidth = 3, color = :navy,    label = "d/h  (displacement)")
lines!(ax1, leaf_area_indices, roughness_fraction;    linewidth = 3, color = :crimson, label = "ℓᵐ/h  (roughness)")
vlines!(ax1, [skimming_ceiling]; color = :gray50, linestyle = :dash)
text!(ax1, skimming_ceiling, 0.02; text = " skimming ceiling (u★/Uh cap)", color = :gray40, align = (:left, :bottom))
axislegend(ax1; position = :rc)

# **(2) Wind ratio.** The internal ratio γ = Uh/u★ *falls* as the canopy densifies (form drag
# raises u★/Uh) and is floored at each class's cap 1/(u★/Uh)ₘₐₓ, so the friction ratio never
# runs away; every vegetation class relaxes onto its own floor. Log–log axes spread the steep
# small-index decay and separate the per-class floors.
log_indices = 10 .^ range(log10(0.05), log10(6); length = 200)   # log-spaced; index > 0 for the log axis
ax2 = Axis(fig[1, 2]; xlabel = "leaf area index", ylabel = "wind ratio  γ = Uh/u★",
           xscale = log10, yscale = log10,
           title = "(2) γ falls with leaf area index, floored at its cap 1/(u★/Uh)ₘₐₓ  (log–log)")
for (class, color) in ((:evergreen_broadleaf_forest, :seagreen),
                       (:cropland, :sienna), (:grassland, :goldenrod))
    parameters = canopy_drag_parameters(Float64, class)
    lines!(ax2, log_indices, [canopy_wind_ratio(𝒜, parameters, closure.iterations) for 𝒜 in log_indices];
           linewidth = 3, color, label = replace(String(class), "_" => " "))
    hlines!(ax2, [1 / parameters.maximum_friction_ratio]; color, linestyle = :dash)
end
axislegend(ax2; position = :rt)

# **(3) Height scaling.** At fixed leaf area index 𝒜 both lengths are *exactly* linear in canopy
# height h (`ℓᵐ = h·f(𝒜)`, `d = h·g(𝒜)`), passing through the origin, and sit close to the
# height-only rules of thumb `d ≈ 2h/3` and `ℓᵐ ≈ d/5` (Brutsaert 1982).
heights = range(0, 30; length = 121)
displacement_heights = [canopy_roughness(closure, 5.0, h)[2] for h in heights]
roughness_lengths    = [canopy_roughness(closure, 5.0, h)[1] for h in heights]

ax3 = Axis(fig[2, 1]; xlabel = "canopy height h (m)", ylabel = "length (m)",
           title = "(3) ℓᵐ, d scale linearly with h  (vs semi-empirical rules)")
lines!(ax3, heights, displacement_heights; linewidth = 3, color = :navy,    label = "d closure (index 5)")
lines!(ax3, heights, roughness_lengths;    linewidth = 3, color = :crimson, label = "ℓᵐ closure (index 5)")
lines!(ax3, heights, 2 .* heights ./ 3;  linewidth = 2, linestyle = :dash, color = :navy,    label = "d = 2h/3 (Brutsaert)")
lines!(ax3, heights, 2 .* heights ./ 15; linewidth = 2, linestyle = :dash, color = :crimson, label = "ℓᵐ = d/5")
axislegend(ax3; position = :lt)

# **(4) Magnitudes.** Evaluated at each IGBP class's representative leaf area index and height,
# the closure `ℓᵐ` reproduces `ℓᵐⁱ`, the class-integrated satellite estimate of Borak et al.
# (2025, Table 5) — which sits below the height-only semi-empirical `ℓᵐᵉ` because the drag
# partition skims roughness down relative to the rule of thumb — and reproduces the
# forest ≫ crop, grass order.
oracle = [("Evergreen\nbroadleaf",  :evergreen_broadleaf_forest,  24.72, 6.0, 1.16, 3.30),
          ("Deciduous\nbroadleaf",  :deciduous_broadleaf_forest,  17.43, 5.0, 0.98, 2.32),
          ("Evergreen\nneedleleaf", :evergreen_needleleaf_forest, 16.62, 4.0, 1.36, 2.22),
          ("Grassland",             :grassland,                    1.39, 1.5, 0.14, 0.19),
          ("Cropland",              :cropland,                     1.32, 3.0, 0.13, 0.18)]
x = 1:length(oracle)
class_roughness = [canopy_roughness(DragPartitionRoughness(Float64; vegetation_type = class), 𝒜, h)[1]
                   for (_, class, h, 𝒜, _, _) in oracle]

ax4 = Axis(fig[2, 2]; ylabel = "momentum roughness ℓᵐ (m)",
           xticks = (x, [row[1] for row in oracle]),
           title = "(4) magnitudes vs Borak et al. (2025) Table 5")
scatter!(ax4, x, [row[6] for row in oracle]; marker = :utriangle, markersize = 15, color = :gray70,
         label = "Borak ℓᵐᵉ  (semi-empirical, height-only)")
scatter!(ax4, x, [row[5] for row in oracle]; marker = :circle, markersize = 15, color = :black,
         label = "Borak ℓᵐⁱ  (class-integrated)")
scatter!(ax4, x, class_roughness; marker = :diamond, markersize = 15, color = :crimson, label = "closure")
axislegend(ax4; position = :rt)

save(joinpath(outdir, "fig3_closure_verification.png"), fig)

# ![](eth_canopy_bare_earth_figures/fig3_closure_verification.png)

fig
