# Figures and animations for the CONUS slab-canopy run, from the saved JLD2 outputs.
# Run after the simulation, in the run directory:
#   julia --project=<docs> render_conus_figures.jl [TAG] [PREVIOUS_STATIC]
# TAG defaults to conus12km_v2; PREVIOUS_STATIC is an optional static file of an earlier run
# (SoilGrids-based soil) for the soil-parameter comparison panel.

using Oceananigans
using NumericalEarth
using CairoMakie
using JLD2
using Printf
using Statistics: mean, median, quantile
import Dates
import Dates: DateTime

tag = isempty(ARGS) ? get(ENV, "TAG", "conus12km_v2") : ARGS[1]
previous_static = length(ARGS) ≥ 2 ? ARGS[2] : get(ENV, "PREVIOUS_STATIC", "")

start_date = DateTime(2011, 5, 17, 0)
case_start = DateTime(2011, 5, 20, 0)
sgp = (-97.485, 36.605)

date_of(t) = start_date + Dates.Second(round(Int, t))
hours_of(t) = t / 3600

series(file, name) = FieldTimeSeries(file, name; backend = OnDisk())

land_file = "$(tag)_land.jld2"
canopy = jldopen(f -> haskey(f["timeseries"], "Tᵛ"), land_file)   # bucket runs write the common fields only
𝒮_ts   = series(land_file, "𝒮");   T_ts  = series(land_file, "Tˡᵃ")
LE_ts  = series(land_file, "LE");  H_ts  = series(land_file, "H")
W_ts   = series(land_file, "W");   Jʳⁿ_ts = series(land_file, "Jʳⁿ")
SW_ts  = series(land_file, "ℐꜜˢʷ"); LW_ts = series(land_file, "ℐꜜˡʷ")
u★_ts  = series(land_file, "u★")
if canopy
    LST_ts = series(land_file, "LST"); Tᵛ_ts = series(land_file, "Tᵛ")
    Tᵍ_ts  = series(land_file, "Tᵍ");  Tᵃᶜ_ts = series(land_file, "Tᵃᶜ")
    LEᶜ_ts = series(land_file, "LEᶜ"); LEᵍ_ts = series(land_file, "LEᵍ")
    Eʷ_ts  = series(land_file, "Eʷ");  Gᶜ_ts = series(land_file, "Gᶜ")
    Wᶜ_ts  = series(land_file, "Wᶜ");  Wᵖ_ts = series(land_file, "Wᵖ")
    P_ts   = series(land_file, "P");   E_ts  = series(land_file, "E")
    R_ts   = series(land_file, "R");   D_ts  = series(land_file, "D")
    α_ts   = series(land_file, "αᵉᶠᶠ")
else
    LST_ts = T_ts
end

times = LST_ts.times
Nt = length(times)
grid = LST_ts.grid
λ, φ, _ = nodes(grid, Center(), Center(), Center())

static = jldopen("$(tag)_static.jld2")
water = static["water"]
land_cells = .!water
mask_water(a) = ifelse.(water, NaN, a)

iˢᵍᵖ = argmin(abs.(λ .- sgp[1]))
jˢᵍᵖ = argmin(abs.(φ .- sgp[2]))

at_sgp(fts, n) = interior(fts[n], iˢᵍᵖ, jˢᵍᵖ, 1)[]
land_mean(fts, n) = mean(interior(fts[n], :, :, 1)[land_cells])
land_quantile(fts, n, q) = quantile(interior(fts[n], :, :, 1)[land_cells], q)

function map_panels!(fig, panels; ncolumns = 4)
    for (k, (title, data, colormap, colorrange)) in enumerate(panels)
        row, column = fldmod1(k, ncolumns)
        ax = Axis(fig[row, 2column - 1]; title, aspect = DataAspect())
        hidedecorations!(ax)
        colorscale = occursin("log scale", title) ? log10 : identity
        hm = heatmap!(ax, λ, φ, mask_water(data); colormap, colorrange, colorscale, nan_color = :lightsteelblue1)
        Colorbar(fig[row, 2column], hm)
    end
end

# ## The ingested land surface: vegetation, soil, urban, radiative properties

panels = (("leaf area index (MODIS)",        static["leaf_area_index"],                    :algae,   (0, 5)),
          ("vegetation-class fraction ($(static["landcover_source"]))", static["vegetation_fraction"], :speed, (0, 1)),
          ("canopy height (m, log scale)",   static["canopy_height"],                      :speed,   (1, 30)),
          ("canopy roughness ℓᵐ (m)",        static["momentum_roughness_vegetated"],       :turbid,  (0, 1.5)),
          ("built-up land fraction",         static["urban_cover"],                        :amp,     (0, 0.5)),
          ("GHSL building height (m)",       static["building_height"],                    :inferno, (0, 15)),
          ("urban ℓᵐ (GHSL morphometry, m)", static["urban_roughness"],                    :viridis, (0, 2)),
          ("bare-tile ℓᵐ (log₁₀ m)",         log10.(static["momentum_roughness_bare"]),    :thermal, (-3, 0)),
          ("sand fraction (OpenLandMap)",    static["sand"],                               :YlOrBr,  (0.1, 0.8)),
          ("porosity ν (Weynants PTF)",      static["porosity"],                           :viridis, (0.3, 0.55)),
          ("matching K₀ (log₁₀ m s⁻¹)",      log10.(static["matching_point_conductivity"]), :turbo,  (-8, -4)),
          ("pore-size uniformity n",         static["pore_size_uniformity"],               :plasma,  (1.05, 1.4)),
          ("blue-sky albedo (Copernicus)",   static["albedo"],                             :grays,   (0.05, 0.35)),
          ("broadband emissivity (ASTER GED)", static["emissivity"],                       :viridis, (0.92, 0.99)),
          ("initial soil water θ (ERA5-Land)", static["initial_soil_water"],               :dense,   (0.05, 0.45)),
          ("deep temperature (K, ERA5-Land)", static["deep_temperature"],                  :thermal, (275, 300)))

fig = Figure(size = (2000, 1500), fontsize = 15)
map_panels!(fig, panels)
Label(fig[0, 1:8], "CONUS $(tag) — the ingested land surface", fontsize = 20)
save("$(tag)_ingestion.png", fig)
@info "Saved $(tag)_ingestion.png"

# ## Measured vs class canopy height (ETH run only)

if haskey(static, "eth_canopy_height") && !isnothing(static["eth_canopy_height"])
    fig = Figure(size = (1800, 900), fontsize = 15)
    map_panels!(fig, (("ETH cell-mean tree height (m, log scale)", max.(static["eth_canopy_height"], 0.1), :speed, (0.1, 30)),
                      ("tall-canopy (≥ 2 m) area fraction",         static["tall_canopy_fraction"],           :algae, (0, 1)),
                      ("IGBP class height (m, log scale)",           static["class_canopy_height"],            :speed, (1, 30)),
                      ("canopy height used (m, log scale)",          static["canopy_height"],                  :speed, (1, 30)),
                      ("used − class (m)",                           static["canopy_height"] .- static["class_canopy_height"], :balance, (-10, 10)),
                      ("canopy roughness ℓᵐ (m, log scale)",         static["momentum_roughness_vegetated"],   :turbid, (0.01, 3))); ncolumns = 3)
    Label(fig[0, 1:6], "ETH Sentinel-2 canopy height against the IGBP class heights", fontsize = 18)
    save("$(tag)_canopy_height.png", fig)
    @info "Saved $(tag)_canopy_height.png"
end

# ## Urban detail: the GHSL roughness around the largest metropolitan areas

cities = (("New York", -74.0, 40.7), ("Chicago", -87.7, 41.85), ("Houston", -95.4, 29.8),
          ("Los Angeles", -118.2, 34.05), ("Dallas–Fort Worth", -97.0, 32.8), ("Atlanta", -84.4, 33.75))
fig = Figure(size = (1800, 1100), fontsize = 14)
for (k, (name, λc, φc)) in enumerate(cities)
    row, column = fldmod1(k, 3)
    window_i = findall(x -> abs(x - λc) < 1.5, λ)
    window_j = findall(y -> abs(y - φc) < 1.2, φ)
    ax = Axis(fig[row, 2column - 1]; title = "$name — urban ℓᵐ (m)", aspect = DataAspect())
    hidedecorations!(ax)
    hm = heatmap!(ax, λ[window_i], φ[window_j], static["urban_roughness"][window_i, window_j];
                  colormap = :viridis, colorrange = (0, 2), nan_color = :gray90)
    contour!(ax, λ[window_i], φ[window_j], static["urban_cover"][window_i, window_j];
             levels = [0.1, 0.3, 0.5], color = :white, linewidth = 1)
    Colorbar(fig[row, 2column], hm)
end
Label(fig[0, 1:6], "GHSL morphometric roughness of the built pixels (contours: built-up land fraction 0.1/0.3/0.5)", fontsize = 17)
save("$(tag)_urban.png", fig)
@info "Saved $(tag)_urban.png"

# ## OpenLandMap vs the previous run's SoilGrids pedotransfer parameters

if !isempty(previous_static) && isfile(previous_static)
    previous = jldopen(previous_static)
    old_porosity = interior(previous["porosity"], :, :, 1)
    old_conductivity = interior(previous["conductivity"], :, :, 1)
    close(previous)
    fig = Figure(size = (1800, 900), fontsize = 15)
    map_panels!(fig, (("porosity — OpenLandMap 30 m (overview read)", static["porosity"], :viridis, (0.3, 0.55)),
                      ("porosity — SoilGrids 2.0 (10 km)",            old_porosity,       :viridis, (0.3, 0.55)),
                      ("Δ porosity (OpenLandMap − SoilGrids)",         static["porosity"] .- old_porosity, :balance, (-0.08, 0.08)),
                      ("log₁₀ K₀ — OpenLandMap",                       log10.(static["matching_point_conductivity"]), :turbo, (-8, -4)),
                      ("log₁₀ K₀ — SoilGrids",                         log10.(old_conductivity), :turbo, (-8, -4)),
                      ("Δ log₁₀ K₀",                                   log10.(static["matching_point_conductivity"]) .- log10.(old_conductivity), :balance, (-1, 1)));
                ncolumns = 3)
    Label(fig[0, 1:6], "Weynants pedotransfer parameters from two texture products", fontsize = 18)
    save("$(tag)_soil_comparison.png", fig)
    @info "Saved $(tag)_soil_comparison.png"
end

# ## Spin-up: the land settles onto a repeating diurnal envelope

hours = hours_of.(times)
if canopy
fig = Figure(size = (1600, 1000), fontsize = 15)

ax1 = Axis(fig[1, 1]; title = "land-mean surface temperatures", xlabel = "hours since 17 May 00 UTC", ylabel = "T (K)")
lines!(ax1, hours, [land_mean(LST_ts, n) for n in 1:Nt]; color = :firebrick,  label = "radiative LST")
lines!(ax1, hours, [land_mean(Tᵛ_ts, n)  for n in 1:Nt]; color = :seagreen,  label = "canopy leaf")
lines!(ax1, hours, [land_mean(Tᵍ_ts, n)  for n in 1:Nt]; color = :chocolate, label = "soil skin")
lines!(ax1, hours, [land_mean(T_ts, n)   for n in 1:Nt]; color = :gray,      label = "bulk slab")
vspan!(ax1, 72, 96; color = (:gold, 0.15))
axislegend(ax1; position = :lt)

ax2 = Axis(fig[1, 2]; title = "land-mean turbulent fluxes", xlabel = "hours", ylabel = "flux (W m⁻²)")
band!(ax2, hours, [land_quantile(LE_ts, n, 0.1) for n in 1:Nt],
                  [land_quantile(LE_ts, n, 0.9) for n in 1:Nt]; color = (:navy, 0.2))
band!(ax2, hours, [land_quantile(H_ts, n, 0.1) for n in 1:Nt],
                  [land_quantile(H_ts, n, 0.9) for n in 1:Nt]; color = (:orangered, 0.2))
lines!(ax2, hours, [land_mean(LE_ts, n) for n in 1:Nt]; color = :navy,      label = "latent (mean, 10–90%)")
lines!(ax2, hours, [land_mean(H_ts, n)  for n in 1:Nt]; color = :orangered, label = "sensible")
vspan!(ax2, 72, 96; color = (:gold, 0.15))
axislegend(ax2; position = :lt)

ax3 = Axis(fig[2, 1]; title = "land-mean water reservoirs", xlabel = "hours", ylabel = "𝒮")
lines!(ax3, hours, [land_mean(𝒮_ts, n) for n in 1:Nt]; color = :navy, label = "saturation ⟨𝒮⟩")
ax3b = Axis(fig[2, 1]; ylabel = "canopy + pond (kg m⁻²)", yaxisposition = :right)
hidespines!(ax3b); hidexdecorations!(ax3b)
lines!(ax3b, hours, [land_mean(Wᶜ_ts, n) for n in 1:Nt]; color = :seagreen, label = "canopy store")
lines!(ax3b, hours, [land_mean(Wᵖ_ts, n) for n in 1:Nt]; color = :steelblue, label = "pond")
axislegend(ax3; position = :lt); axislegend(ax3b; position = :rt)

ax4 = Axis(fig[2, 2]; title = "land-mean rain and evaporation", xlabel = "hours", ylabel = "mm hr⁻¹")
lines!(ax4, hours, [3600 * land_mean(Jʳⁿ_ts, n) for n in 1:Nt]; color = :steelblue, label = "incident rain")
lines!(ax4, hours, [3600 * land_mean(P_ts, n) for n in 1:Nt];   color = :navy, linestyle = :dash, label = "throughfall")
lines!(ax4, hours, [3600 * land_mean(E_ts, n) for n in 1:Nt];   color = :darkorange, label = "soil evaporation")
vspan!(ax4, 72, 96; color = (:gold, 0.15))
axislegend(ax4; position = :lt)

Label(fig[0, 1:2], "CONUS $(tag) — three spin-up diurnal cycles, then the MC3E case day (shaded)", fontsize = 18)
save("$(tag)_spinup.png", fig)
@info "Saved $(tag)_spinup.png"

# ## The ARM SGP column: canopy-air-space physics through the case day

fig = Figure(size = (1700, 950), fontsize = 15)

ax1 = Axis(fig[1, 1]; title = "temperature hierarchy at ARM SGP", xlabel = "hours", ylabel = "T (K)")
for (fts, color, label) in ((Tᵛ_ts, :seagreen, "leaf Tᵛ"), (Tᵍ_ts, :chocolate, "soil skin Tᵍ"),
                            (Tᵃᶜ_ts, :purple, "canopy air Tᵃᶜ"), (LST_ts, :firebrick, "radiative LST"),
                            (T_ts, :gray, "bulk slab"))
    lines!(ax1, hours, [at_sgp(fts, n) for n in 1:Nt]; color, label)
end
vspan!(ax1, 72, 96; color = (:gold, 0.15))
axislegend(ax1; position = :lt)

ax2 = Axis(fig[1, 2]; title = "flux partition at ARM SGP", xlabel = "hours", ylabel = "W m⁻²")
lines!(ax2, hours, [at_sgp(LE_ts, n)  for n in 1:Nt]; color = :navy,      label = "blended LE")
lines!(ax2, hours, [at_sgp(H_ts, n)   for n in 1:Nt]; color = :orangered, label = "blended H")
lines!(ax2, hours, [at_sgp(LEᶜ_ts, n) for n in 1:Nt]; color = :seagreen,  linestyle = :dash, label = "canopy LEᶜ")
lines!(ax2, hours, [at_sgp(LEᵍ_ts, n) for n in 1:Nt]; color = :chocolate, linestyle = :dash, label = "soil LEᵍ")
lines!(ax2, hours, [at_sgp(Gᶜ_ts, n)  for n in 1:Nt]; color = :gray,      label = "ground heat Gᶜ")
vspan!(ax2, 72, 96; color = (:gold, 0.15))
axislegend(ax2; position = :lt)

ax3 = Axis(fig[2, 1]; title = "water at ARM SGP", xlabel = "hours", ylabel = "rain (mm hr⁻¹)")
lines!(ax3, hours, [3600 * at_sgp(Jʳⁿ_ts, n) for n in 1:Nt]; color = :steelblue, label = "incident rain")
lines!(ax3, hours, [3600 * at_sgp(P_ts, n) for n in 1:Nt]; color = :navy, linestyle = :dash, label = "throughfall")
ax3b = Axis(fig[2, 1]; ylabel = "Wᶜ (kg m⁻²)", yaxisposition = :right)
hidespines!(ax3b); hidexdecorations!(ax3b)
lines!(ax3b, hours, [at_sgp(Wᶜ_ts, n) for n in 1:Nt]; color = :seagreen)
axislegend(ax3; position = :lt)

# Surface energy balance at SGP from the run's own effective albedo and the ASTER emissivity.
σˢᵇ = 5.670374419e-8
εˢᵍᵖ = static["emissivity"][iˢᵍᵖ, jˢᵍᵖ]
Rn = [(1 - at_sgp(α_ts, n)) * at_sgp(SW_ts, n) + εˢᵍᵖ * (at_sgp(LW_ts, n) - σˢᵇ * at_sgp(LST_ts, n)^4) for n in 1:Nt]
ax4 = Axis(fig[2, 2]; title = "surface energy balance at ARM SGP", xlabel = "hours", ylabel = "W m⁻²")
lines!(ax4, hours, Rn; color = :black, label = "net radiation (from LST, αᵉᶠᶠ)")
lines!(ax4, hours, [at_sgp(H_ts, n) + at_sgp(LE_ts, n) + at_sgp(Gᶜ_ts, n) for n in 1:Nt];
       color = :crimson, linestyle = :dash, label = "H + LE + G")
axislegend(ax4; position = :lt)

Label(fig[0, 1:2], "CONUS $(tag) — the canopy air space at the ARM SGP pixel", fontsize = 18)
save("$(tag)_sgp_column.png", fig)
@info "Saved $(tag)_sgp_column.png"

# ## The mechanism map: soil moisture organizes the flux partition (case day, 1300 CST)

n1300 = argmin(abs.(times .- (times[1] + (3 * 24 + 19) * 3600)))   # 19 UTC on 20 May
𝒮map  = interior(𝒮_ts[n1300], :, :, 1)
Hmap  = interior(H_ts[n1300], :, :, 1)
LEmap = interior(LE_ts[n1300], :, :, 1)
LSTmap = interior(LST_ts[n1300], :, :, 1)
u★map = interior(u★_ts[n1300], :, :, 1)
bowen = Hmap ./ max.(LEmap, 10)

fig = Figure(size = (1800, 1300), fontsize = 15)
map_panels!(fig, (("saturation 𝒮", 𝒮map, :dense, (0, 1)),
                  ("Bowen ratio H/LE", bowen, :balance, (0, 4)),
                  ("latent heat (W m⁻²)", LEmap, :solar, (0, 500)),
                  ("sensible heat (W m⁻²)", Hmap, :lajolla, (0, 400)),
                  ("radiative LST (K)", LSTmap, :thermal, (285, 325)),
                  ("friction velocity u★ (m s⁻¹)", u★map, :speed, (0, 1))); ncolumns = 2)
Label(fig[0, 1:4], "Soil moisture and roughness organize the surface fluxes — 20 May 2011, 1300 CST", fontsize = 18)
save("$(tag)_mechanism.png", fig)
@info "Saved $(tag)_mechanism.png"

# Urban signature: LST and H of built-up cells against their non-urban surroundings, at midday.
urban_cells = land_cells .& (static["urban_cover"] .> 0.3)
rural_cells = land_cells .& (static["urban_cover"] .< 0.02) .& (static["vegetation_fraction"] .> 0.3)
@info @sprintf("1300 CST: %d urban cells (built-up > 0.3) LST %.1f K, H %.0f W m⁻², u★ %.2f; rural LST %.1f K, H %.0f, u★ %.2f",
               count(urban_cells), mean(LSTmap[urban_cells]), mean(Hmap[urban_cells]), mean(u★map[urban_cells]),
               mean(LSTmap[rural_cells]), mean(Hmap[rural_cells]), mean(u★map[rural_cells]))

# ## Land water budget closure (domain integral over land)

Δts = diff(times)
cumulative(fts; flip = false) = begin
    out = zeros(Nt)
    for n in 2:Nt
        rate = land_mean(fts, n)
        out[n] = out[n-1] + (flip ? -rate : rate) * Δts[n-1]
    end
    out
end

rain_in    = cumulative(Jʳⁿ_ts)
evap_out   = cumulative(E_ts)
canopy_out = cumulative(Eʷ_ts)
runoff_out = cumulative(R_ts)
drain_out  = cumulative(D_ts; flip = true)   # positive downward out of the slab
ΔW  = [land_mean(W_ts, n) - land_mean(W_ts, 1) for n in 1:Nt]
ΔWᶜ = [land_mean(Wᶜ_ts, n) - land_mean(Wᶜ_ts, 1) for n in 1:Nt]
ΔWᵖ = [land_mean(Wᵖ_ts, n) - land_mean(Wᵖ_ts, 1) for n in 1:Nt]
storage = ΔW .+ ΔWᶜ .+ ΔWᵖ
losses  = evap_out .+ canopy_out .+ runoff_out .+ drain_out

fig = Figure(size = (1200, 700), fontsize = 15)
ax = Axis(fig[1, 1]; title = "land-mean water budget (kg m⁻² accumulated)",
          xlabel = "hours since 17 May 00 UTC", ylabel = "kg m⁻²")
lines!(ax, hours, rain_in;  color = :steelblue, label = "rain in")
lines!(ax, hours, storage;  color = :navy,      label = "Δ storage (soil + canopy + pond)")
lines!(ax, hours, losses;   color = :darkorange, label = "evaporation + runoff + drainage")
lines!(ax, hours, rain_in .- losses .- storage; color = :crimson, linestyle = :dash, label = "residual")
axislegend(ax; position = :lt)
save("$(tag)_water_budget.png", fig)
@info @sprintf("water budget over %.0f h: rain %.2f = evap %.2f + wet canopy %.2f + runoff %.2f + drainage %.2f + Δstorage %.2f, residual %.2f kg m⁻²",
               hours[end], rain_in[end], evap_out[end], canopy_out[end], runoff_out[end], drain_out[end], storage[end],
               rain_in[end] - losses[end] - storage[end])
else
fig = Figure(size = (1600, 500), fontsize = 15)
ax1 = Axis(fig[1, 1]; title = "land-mean slab temperature", xlabel = "hours since 17 May 00 UTC", ylabel = "T (K)")
lines!(ax1, hours, [land_mean(T_ts, n) for n in 1:Nt]; color = :firebrick)
ax2 = Axis(fig[1, 2]; title = "land-mean turbulent fluxes and rain", xlabel = "hours", ylabel = "W m⁻²")
lines!(ax2, hours, [land_mean(LE_ts, n) for n in 1:Nt]; color = :navy, label = "LE")
lines!(ax2, hours, [land_mean(H_ts, n) for n in 1:Nt]; color = :orangered, label = "H")
lines!(ax2, hours, [3600 * 100 * land_mean(Jʳⁿ_ts, n) for n in 1:Nt]; color = :steelblue, label = "rain × 100 (mm hr⁻¹)")
axislegend(ax2; position = :lt)
save("$(tag)_spinup.png", fig)
end

# ## Case-day land animation

case = findall(t -> date_of(t) >= case_start, times)

fig = Figure(size = (1800, 950), fontsize = 14)
n = Observable(first(case))
LSTn = @lift mask_water(interior(LST_ts[$n], :, :, 1))
𝒮n   = @lift mask_water(interior(𝒮_ts[$n], :, :, 1))
LEn  = @lift mask_water(interior(LE_ts[$n], :, :, 1))
Hn   = @lift mask_water(interior(H_ts[$n], :, :, 1))
for (k, (title, obs, colormap, colorrange)) in enumerate(
        (("radiative LST (K)", LSTn, :thermal, (275, 320)),
         ("saturation 𝒮", 𝒮n, :dense, (0, 1)),
         ("latent heat (W m⁻²)", LEn, :solar, (0, 500)),
         ("sensible heat (W m⁻²)", Hn, :lajolla, (0, 400))))
    row, column = fldmod1(k, 2)
    ax = Axis(fig[row, 2column - 1]; title, aspect = DataAspect())
    hidedecorations!(ax)
    hm = heatmap!(ax, λ, φ, obs; colormap, colorrange, nan_color = :lightsteelblue1)
    Colorbar(fig[row, 2column], hm)
end
label = Label(fig[0, 1:4], ""; fontsize = 18)
CairoMakie.record(fig, "$(tag)_land.mp4", case; framerate = 6) do nn
    n[] = nn
    label.text = "CONUS land — " * Dates.format(date_of(times[nn]), "dd u yyyy HH:MM") * " UTC"
end
@info "Saved $(tag)_land.mp4"

# ## Case-day atmosphere animation (squall line)

surface_file = "$(tag)_surface.jld2"
aloft_file = "$(tag)_aloft.jld2"
θᵥ_ts = series(surface_file, "θᵥ"); U_ts = series(surface_file, "U")
qʳ_ts = series(aloft_file, "qʳ");  w_ts = series(aloft_file, "w")

fig = Figure(size = (1800, 950), fontsize = 14)
m = Observable(first(case))
θᵥn = @lift interior(θᵥ_ts[$m], :, :, 1)
Un  = @lift interior(U_ts[$m], :, :, 1)
qʳn = @lift interior(qʳ_ts[$m], :, :, 1)
wn  = @lift interior(w_ts[$m], :, :, 1)
for (k, (title, obs, colormap, colorrange)) in enumerate(
        (("surface θᵥ (K)", θᵥn, :thermal, (285, 320)),
         ("surface wind speed (m s⁻¹)", Un, :speed, (0, 25)),
         ("rain water qʳ at 2 km (kg kg⁻¹)", qʳn, :dense, (0, 2e-3)),
         ("w at 2 km (m s⁻¹)", wn, :balance, (-3, 3))))
    row, column = fldmod1(k, 2)
    ax = Axis(fig[row, 2column - 1]; title, aspect = DataAspect())
    hidedecorations!(ax)
    hm = heatmap!(ax, λ, φ, obs; colormap, colorrange)
    Colorbar(fig[row, 2column], hm)
end
label2 = Label(fig[0, 1:4], ""; fontsize = 18)
CairoMakie.record(fig, "$(tag)_atmosphere.mp4", case; framerate = 6) do nn
    m[] = nn
    label2.text = "CONUS atmosphere — " * Dates.format(date_of(times[nn]), "dd u yyyy HH:MM") * " UTC"
end
@info "Saved $(tag)_atmosphere.mp4"

# ## 500 hPa height + vertical velocity through the case day

iso_file = "$(tag)_isobaric.jld2"
if isfile(iso_file)
Z500_ts = series(iso_file, "Z_500"); w500_ts = series(iso_file, "w_500")

fig = Figure(size = (1500, 650), fontsize = 14)
p = Observable(first(case))
w500 = @lift interior(w500_ts[$p], :, :, 1)
Z500 = @lift interior(Z500_ts[$p], :, :, 1)
ax = Axis(fig[1, 1]; title = "w (fill) and Z (contours) at 500 hPa", aspect = DataAspect())
hidedecorations!(ax)
hm = heatmap!(ax, λ, φ, w500; colormap = :balance, colorrange = (-1.5, 1.5))
contour!(ax, λ, φ, Z500; color = :black, levels = 5400:60:5940)
Colorbar(fig[1, 2], hm)
label3 = Label(fig[0, 1:2], ""; fontsize = 18)
CairoMakie.record(fig, "$(tag)_500hPa.mp4", case; framerate = 6) do nn
    p[] = nn
    label3.text = "500 hPa — " * Dates.format(date_of(times[nn]), "dd u yyyy HH:MM") * " UTC"
end
@info "Saved $(tag)_500hPa.mp4"
end

close(static)
@info "All figures rendered for $(tag)."
