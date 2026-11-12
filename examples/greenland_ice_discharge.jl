# # Greenland ice discharge from GEUS
#
# The Greenland ice sheet loses about 500 Gt of ice per year by discharge across the
# grounding lines of its marine-terminating glaciers. NumericalEarth can force an ocean
# with this solid ice discharge from two datasets:
#
# - JRA55-do, whose iceberg calving field `licalvf` [tsujino2018jra](@citep) puts the land-ice
#   discharge of [Bamber et al. (2018)](@cite Bamber2018) on coastal cells. `JRA55PrescribedLand`
#   already includes it, as the `icebergs` component.
# - The GEUS/PROMICE solid ice discharge [Mankoff2020solid](@citep), observed through flux gates
#   upstream of each glacier front from 1986 onward, with `GEUSPrescribedLand`.
#
# The freshwater sources around Greenland and the Arctic, and which dataset covers them:
#
# |                                           | Typical magnitude              | GloFAS v4                              | JRA55-do                                   | GEUS solid discharge                  |
# |:------------------------------------------|:-------------------------------|:---------------------------------------|:-------------------------------------------|:--------------------------------------|
# | **Method**                                |                                | LISFLOOD hydrology model, ERA5-forced  | CaMa-Flood river routing of JRA-55 land runoff, plus Bamber et al. (2018) for land ice | Ice velocity and thickness through flux gates |
# | **Major Arctic rivers**                   | ~4000 km³/yr                   | ✅ Calibrated against gauges            | ✅ Rescaled to observed basin totals        | ❌                                     |
# | **Other Arctic glaciers**                 | ~100–200 km³/yr                | ~ Treated as deep snow                 | ✅ Liquid runoff from Bamber et al. (2018)  | ❌                                     |
# | **Greenland liquid runoff**               | ~400–550 km³/yr                | ⚠️ ~250 km³/yr, too low                 | ✅ Bamber et al. (2018)                     | ❌                                     |
# | **Greenland ice discharge**               | ~500 km³/yr                    | ❌                                      | ✅ `licalvf`                                | ✅ Per gate                            |
# | **Basal melt**                            | ~20 km³/yr                     | ❌                                      | ❌                                          | ❌                                     |
# | **In NumericalEarth**                     |                                | `GloFASPrescribedLand`                 | `JRA55PrescribedLand`                      | `GEUSPrescribedLand`                  |
#
# 1 km³/yr of water is about 1 Gt/yr, or 32 m³/s. Typical magnitudes are rough literature values.
#
# Which land to use depends on the atmosphere:
#
# - **ERA5** has no river discharge and no ice discharge. With an ERA5 atmosphere, add both:
#   GloFAS rivers and GEUS solid ice discharge (see below).
# - **JRA55-do** comes with its own rivers and ice discharge. Use `JRA55PrescribedLand` alone:
#   GloFAS and GEUS describe the same freshwater, which would then be counted twice.
#
# Here we first add the GEUS solid ice discharge to a model of the ocean around Greenland,
# then look at where it enters the ocean and how it compares with JRA55-do.
#
# ## Install dependencies
#
# ```julia
# using Pkg
# pkg"add NumericalEarth Oceananigans CairoMakie NCDatasets"
# ```

using NumericalEarth
using Oceananigans
using CairoMakie
using Dates
using Downloads: download
using NCDatasets: NCDataset
using Statistics: mean

# ## Adding GEUS solid ice discharge to a model
#
# We use a 1/4ᵒ Greenland grid with one 100 m level, and ETOPO bathymetry. Cells whose seafloor
# is shallower than the 50 m cell center are land.

grid = LatitudeLongitudeGrid(size = (300, 120, 1), longitude = (-80, -5), latitude = (55, 85), z = (-100, 0))
bottom_height = regrid_bathymetry(grid; minimum_depth = 10)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom_height))

# `GEUSPrescribedLand` downloads a GEUS release (by default `GEUSSolidIceDischarge(version = 132)`),
# takes monthly means of the discharge across each gate, and spreads each gate over the
# nearest wet cells, with the same routing as GloFAS river mouths. The discharge enters the
# ocean as the `icebergs` component, so the ocean supplies the latent heat to melt it.

dates = (DateTime(1987, 1, 1), DateTime(2019, 12, 31))
land = GEUSPrescribedLand(grid; dates)

# With an ERA5 atmosphere, `merge` combines the GloFAS rivers and the GEUS icebergs into one
# `PrescribedLand`, which goes into the coupled model like any other land:
#
# ```julia
# rivers = GloFASPrescribedLand(grid; start_date = first(dates), end_date = last(dates))
# land = merge(rivers, GEUSPrescribedLand(grid; dates))
# coupled_model = OceanSeaIceModel(ocean, sea_ice; atmosphere, land, radiation)
# ```
#
# ## Where the discharge enters the ocean
#
# The discharge of each gate, in kg s⁻¹, is a `FieldTimeSeries` of monthly means. Its records
# are centered on each month and pad the dates by one month on each side.

icebergs = land.freshwater_flux.icebergs
months = [first(dates) + Millisecond(round(Int, 1000t)) for t in icebergs.times]
in_dates = first(dates) .≤ months .≤ last(dates)
discharge = interior(icebergs)[:, 1, 1, in_dates]
months = months[in_dates]

gigatonnes_per_year(Q) = Q * 365.25 * 86400 / 1e12

gate_λ, gate_φ = NCDataset(ds -> (ds["mean_lon"][:], ds["mean_lat"][:]), download(GEUSSolidIceDischarge()))
gate_discharge = vec(mean(discharge[:, 2000 .≤ year.(months)], dims = 2))
nothing #hide

# `land.river_routing.icebergs` records the ocean cells that receive each gate's discharge,
# and the weight per unit area each cell gets. We use it to map the 2000–2019 mean freshwater
# flux the ocean receives, and draw a line from each gate to its cells.

routing = land.river_routing.icebergs
λ = λnodes(grid, Center())
φ = φnodes(grid, Center())

ocean_flux = fill(NaN, size(grid, 1), size(grid, 2))   # m yr⁻¹ of water
connections = Point2f[]
for c in eachindex(routing.target_i)
    i, j = routing.target_i[c], routing.target_j[c]
    ocean_flux[i, j] = 0
    for k in routing.offsets[c]:routing.offsets[c+1]-1
        gate = routing.contribution_outlet_i[k]
        ocean_flux[i, j] += routing.contribution_weight[k] * gate_discharge[gate] * 365.25 * 86400 / 1000
        push!(connections, Point2f(gate_λ[gate], gate_φ[gate]), Point2f(λ[i], φ[j]))
    end
end

fig = Figure(size = (800, 850))
ax = Axis(fig[1, 1]; xlabel = "Longitude (ᵒ)", ylabel = "Latitude (ᵒ)", aspect = AxisAspect(75 * cosd(72) / 30),
          title = "GEUS gates routed onto a 1/4ᵒ grid, 2000–2019 mean")
heatmap!(ax, bottom_height; colormap = [:white, :white], colorrange = (-10000, -50), highclip = :gray80)
linesegments!(ax, connections; color = (:black, 0.3), linewidth = 0.5)
hm = heatmap!(ax, λ, φ, ocean_flux; colormap = :viridis, colorscale = log10, nan_color = :transparent)
scatter!(ax, gate_λ, gate_φ; markersize = 3 .+ 2 .* sqrt.(gigatonnes_per_year.(gate_discharge)),
         color = (:orange, 0.8), strokecolor = :black, strokewidth = 0.4)
limits!(ax, -76, -8, 58, 84)
Colorbar(fig[1, 2], hm; label = "ice discharge entering the ocean (m yr⁻¹ of water)")
fig

# The gates (orange, sized by discharge) sit up-fjord, behind fjords a 1/4ᵒ grid does not
# resolve, so their discharge enters the ocean tens of kilometers away, at the coast.
#
# ## Comparison with JRA55-do
#
# We sum the JRA55-do v1.5 calving over a box around Greenland, for each year, and compare
# with the GEUS annual means. The default of `JRA55PrescribedLand`, the repeat year
# May 1990 – April 1991, is shown too.

function greenland_calving(metadatum)
    return NCDataset(download(metadatum)) do ds
        i = findall(λ -> 270 ≤ λ ≤ 355, ds["lon"][:])
        j = findall(φ -> 55 ≤ φ ≤ 85, ds["lat"][:])
        area = [6371e3^2 * deg2rad(0.25) * (sind(φ + 0.125) - sind(φ - 0.125)) for λ in i, φ in ds["lat"][j]]
        calving = coalesce.(ds["licalvf"][i, j, :], 0)   # kg m⁻² s⁻¹
        mean(sum(calving[:, :, t] .* area) for t in axes(calving, 3))
    end
end

years = year(first(dates)):year(last(dates))
jra55 = [greenland_calving(Metadatum(:iceberg_freshwater_flux; dataset = MultiYearJRA55(), date = DateTime(y, 1, 1, 12)))
         for y in years]
repeat_year = greenland_calving(Metadatum(:iceberg_freshwater_flux; dataset = RepeatYearJRA55(), date = DateTime(1990, 1, 1, 12)))
geus = [mean(sum(discharge[:, year.(months) .== y], dims = 1)) for y in years]

fig = Figure(size = (800, 400))
ax = Axis(fig[1, 1]; xlabel = "Year", ylabel = "Greenland ice discharge (Gt yr⁻¹)")
scatterlines!(ax, years, gigatonnes_per_year.(geus); color = :black, label = "GEUS")
scatterlines!(ax, years, gigatonnes_per_year.(jra55); color = :firebrick, label = "JRA55-do v1.5")
hlines!(ax, gigatonnes_per_year(repeat_year); color = :dodgerblue, linestyle = :dash, label = "JRA55-do repeat year")
axislegend(ax; position = :lt)
fig

# Over 1987–2019 the two agree on average. JRA55-do is lower than GEUS in the 1990s and
# higher after the early 2000s, when its discharge rises faster.
