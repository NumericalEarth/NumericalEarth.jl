# Figures and animations for the CONUS slab-canopy run, from the saved JLD2 outputs.
# Run after the simulation, in the run directory:
#   julia --project=<docs> render_conus_figures.jl [TAG]
# TAG defaults to conus12km_veg.

using Oceananigans
using NumericalEarth
using CairoMakie
using JLD2
using Printf
using Statistics: mean, quantile
import Dates
import Dates: DateTime

tag = isempty(ARGS) ? get(ENV, "TAG", "conus12km_veg") : ARGS[1]

start_date = DateTime(2011, 5, 17, 0)
case_start = DateTime(2011, 5, 20, 0)
sgp = (-97.485, 36.605)

date_of(t) = start_date + Dates.Second(round(Int, t))
hours_of(t) = t / 3600

series(file, name) = FieldTimeSeries(file, name; backend = OnDisk())

land_file = "$(tag)_land.jld2"
LST_ts = series(land_file, "LST"); 𝒮_ts  = series(land_file, "𝒮")
T_ts   = series(land_file, "Tˡᵃ"); Tᵛ_ts = series(land_file, "Tᵛ")
Tᵍ_ts  = series(land_file, "Tᵍ");  Tᵃᶜ_ts = series(land_file, "Tᵃᶜ")
LE_ts  = series(land_file, "LE");  H_ts  = series(land_file, "H")
LEᶜ_ts = series(land_file, "LEᶜ"); LEᵍ_ts = series(land_file, "LEᵍ")
Eʷ_ts  = series(land_file, "Eʷ");  Gᶜ_ts = series(land_file, "Gᶜ")
W_ts   = series(land_file, "W");   Wᶜ_ts = series(land_file, "Wᶜ")
Wᵖ_ts  = series(land_file, "Wᵖ");  P_ts  = series(land_file, "P")
E_ts   = series(land_file, "E");   R_ts  = series(land_file, "R")
D_ts   = series(land_file, "D");   Jʳⁿ_ts = series(land_file, "Jʳⁿ")
SW_ts  = series(land_file, "ℐꜜˢʷ"); LW_ts = series(land_file, "ℐꜜˡʷ")

times = LST_ts.times
Nt = length(times)
grid = LST_ts.grid
λ, φ, _ = nodes(grid, Center(), Center(), Center())

static = jldopen("$(tag)_static.jld2")
water = static["water"]
albedo = interior(static["albedo"], :, :, 1)
mask_water(a) = ifelse.(water, NaN, a)
land_cells = .!water

iˢᵍᵖ = argmin(abs.(λ .- sgp[1]))
jˢᵍᵖ = argmin(abs.(φ .- sgp[2]))

at_sgp(fts, n) = interior(fts[n], iˢᵍᵖ, jˢᵍᵖ, 1)[]
land_mean(fts, n) = mean(interior(fts[n], :, :, 1)[land_cells])
land_quantile(fts, n, q) = quantile(interior(fts[n], :, :, 1)[land_cells], q)

# ## The ingested land surface

panels = (("leaf area index (MODIS)",      interior(static["leaf_area_index"], :, :, 1),           :algae,   (0, 5)),
          ("vegetated fraction",           interior(static["vegetation_fraction"], :, :, 1),       :speed,   (0, 1)),
          ("canopy height (m, IGBP class)", interior(static["canopy_height"], :, :, 1),            :speed,   (0, 20)),
          ("canopy roughness ℓᵐ (m)",      interior(static["momentum_roughness"], :, :, 1),        :turbid,  (0, 1.5)),
          ("displacement d (m)",           interior(static["displacement"], :, :, 1),              :turbid,  (0, 14)),
          ("bare ℓᵐ (log₁₀ m)",            log10.(interior(static["bare_roughness"], :, :, 1)),    :thermal, (-3.5, -1.5)),
          ("porosity ν (SoilGrids PTF)",   interior(static["porosity"], :, :, 1),                  :viridis, (0.3, 0.55)),
          ("matching K₀ (log₁₀ m s⁻¹)",    log10.(interior(static["conductivity"], :, :, 1)),      :turbo,   (-8, -4)),
          ("blue-sky albedo (Copernicus)", albedo,                                                 :grays,   (0.05, 0.35)),
          ("dry heat capacity (J m⁻² K⁻¹)", interior(static["dry_heat_capacity"], :, :, 1),        :amp,     (1.2e5, 2.2e5)),
          ("initial soil water θ (ERA5-Land)", interior(static["initial_soil_water"], :, :, 1),    :dense,   (0.05, 0.45)),
          ("deep temperature (K, ERA5-Land)", interior(static["deep_temperature"], :, :, 1),       :thermal, (275, 300)))

fig = Figure(size = (2000, 1150), fontsize = 15)
for (k, (title, data, colormap, colorrange)) in enumerate(panels)
    row, column = fldmod1(k, 4)
    ax = Axis(fig[row, 2column - 1]; title, aspect = DataAspect())
    hidedecorations!(ax)
    hm = heatmap!(ax, λ, φ, mask_water(data); colormap, colorrange, nan_color = :lightsteelblue1)
    Colorbar(fig[row, 2column], hm)
end
Label(fig[0, 1:8], "CONUS $(tag) — the ingested land surface", fontsize = 20)
save("$(tag)_ingestion.png", fig)
@info "Saved $(tag)_ingestion.png"

# ## Spin-up: the land settles onto a repeating diurnal envelope

hours = hours_of.(times)
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

ax3 = Axis(fig[2, 1]; title = "land-mean water reservoirs", xlabel = "hours", ylabel = "𝒮, storages")
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

ax2 = Axis(fig[1, 2]; title = "flux partition at ARM SGP (vegetated tile)", xlabel = "hours", ylabel = "W m⁻²")
lines!(ax2, hours, [at_sgp(LE_ts, n)  for n in 1:Nt]; color = :navy,      label = "blended LE")
lines!(ax2, hours, [at_sgp(H_ts, n)   for n in 1:Nt]; color = :orangered, label = "blended H")
lines!(ax2, hours, [at_sgp(LEᶜ_ts, n) for n in 1:Nt]; color = :seagreen,  linestyle = :dash, label = "canopy LEᵛ")
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

# Surface energy balance closure at SGP: εσ and α from the run's own surfaces.
σˢᵇ = 5.670374419e-8
ε = 0.96
αˢᵍᵖ = albedo[iˢᵍᵖ, jˢᵍᵖ]
Rn = [(1 - αˢᵍᵖ) * at_sgp(SW_ts, n) + ε * (at_sgp(LW_ts, n) - σˢᵇ * at_sgp(LST_ts, n)^4) for n in 1:Nt]
ax4 = Axis(fig[2, 2]; title = "surface energy balance at ARM SGP", xlabel = "hours", ylabel = "W m⁻²")
lines!(ax4, hours, Rn; color = :black, label = "net radiation (from LST)")
lines!(ax4, hours, [at_sgp(H_ts, n) + at_sgp(LE_ts, n) + at_sgp(Gᶜ_ts, n) for n in 1:Nt];
       color = :crimson, linestyle = :dash, label = "H + LE + G")
axislegend(ax4; position = :lt)

Label(fig[0, 1:2], "CONUS $(tag) — the canopy air space at the ARM SGP pixel", fontsize = 18)
save("$(tag)_sgp_column.png", fig)
@info "Saved $(tag)_sgp_column.png"

# ## The mechanism map: soil moisture organizes the flux partition (case day, 1300 CST)

n1300 = argmin(abs.(times .- (times[1] + (3 * 24 + 19) * 3600)))   # 19 UTC on 20 May
𝒮map  = mask_water(interior(𝒮_ts[n1300], :, :, 1))
Hmap  = mask_water(interior(H_ts[n1300], :, :, 1))
LEmap = mask_water(interior(LE_ts[n1300], :, :, 1))
bowen = Hmap ./ max.(LEmap, 10)

fig = Figure(size = (1800, 900), fontsize = 15)
for (k, (title, data, colormap, colorrange)) in enumerate(
        (("saturation 𝒮", 𝒮map, :dense, (0, 1)),
         ("Bowen ratio H/LE", bowen, :balance, (0, 4)),
         ("latent heat (W m⁻²)", LEmap, :solar, (0, 500)),
         ("sensible heat (W m⁻²)", Hmap, :lajolla, (0, 400))))
    row, column = fldmod1(k, 2)
    ax = Axis(fig[row, 2column - 1]; title, aspect = DataAspect())
    hidedecorations!(ax)
    hm = heatmap!(ax, λ, φ, data; colormap, colorrange, nan_color = :lightsteelblue1)
    Colorbar(fig[row, 2column], hm)
end
Label(fig[0, 1:4], "Soil moisture organizes the surface fluxes — 20 May 2011, 1300 CST", fontsize = 18)
save("$(tag)_mechanism.png", fig)
@info "Saved $(tag)_mechanism.png"

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
@info "Saved $(tag)_water_budget.png"

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
qᵛ_ts = series(surface_file, "qᵛ"); qʳ_ts = series(aloft_file, "qʳ")
w_ts  = series(aloft_file, "w")

fig = Figure(size = (1800, 950), fontsize = 14)
m = Observable(first(case))
θᵥn = @lift interior(θᵥ_ts[$m], :, :, 1)
Un  = @lift interior(U_ts[$m], :, :, 1)
qʳn = @lift interior(qʳ_ts[$m], :, :, 1)
wn  = @lift interior(w_ts[$m], :, :, 1)
for (k, (title, obs, colormap, colorrange)) in enumerate(
        (("surface θᵥ (K)", θᵥn, :thermal, (285, 320)),
         ("surface wind speed (m s⁻¹)", Un, :speed, (0, 25)),
         ("rain water qʳ at 2 km (g kg⁻¹)", qʳn, :dense, (0, 2e-3)),
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

close(static)
@info "All figures rendered for $(tag)."
