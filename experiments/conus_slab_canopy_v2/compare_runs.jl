# Two runs side by side on the fields every configuration writes (land temperature, saturation,
# latent and sensible heat, friction velocity, rain): maps of the case-day 1300 CST differences,
# land-mean time series, and a table of land-mean numbers.
#   julia --project=<docs> compare_runs.jl TAG_A TAG_B [LABEL_A] [LABEL_B]
using Oceananigans
using NumericalEarth
using CairoMakie
using JLD2
using Printf
using Statistics: mean, quantile

tag_a, tag_b = ARGS[1], ARGS[2]
label_a = length(ARGS) ≥ 3 ? ARGS[3] : tag_a
label_b = length(ARGS) ≥ 4 ? ARGS[4] : tag_b

series(tag, name) = FieldTimeSeries("$(tag)_land.jld2", name; backend = OnDisk())
names = ("Tˡᵃ", "𝒮", "LE", "H", "u★", "Jʳⁿ")
A = Dict(name => series(tag_a, name) for name in names)
B = Dict(name => series(tag_b, name) for name in names)

times = A["LE"].times
Nt = min(length(times), length(B["LE"].times))
grid = A["LE"].grid
λ, φ, _ = nodes(grid, Center(), Center(), Center())
water = jldopen(f -> f["water"], "$(tag_a)_static.jld2")
land = .!water
mask(a) = ifelse.(water, NaN, a)
field(fts, n) = interior(fts[n], :, :, 1)
lmean(fts, n) = mean(field(fts, n)[land])

case_hours = 3 * 24 .+ (13, 19, 1)   # 0700, 1300, 1900 CST on 20 May
n13 = argmin(abs.(times .- (3 * 86400 + 19 * 3600)))

@printf("%s vs %s — land means\n", label_a, label_b)
@printf("%-28s %12s %12s %12s\n", "quantity", label_a[1:min(end, 12)], label_b[1:min(end, 12)], "A − B")
for (name, unit, scale) in (("LE", "W/m²", 1), ("H", "W/m²", 1), ("Tˡᵃ", "K", 1), ("𝒮", "", 1), ("u★", "m/s", 1))
    a = mean(field(A[name], n13)[land]) * scale; b = mean(field(B[name], n13)[land]) * scale
    @printf("%-28s %12.2f %12.2f %12.2f\n", "$name at 1300 CST ($unit)", a, b, a - b)
end
Δts = diff(times[1:Nt])
rain(X) = sum(lmean(X["Jʳⁿ"], n) * Δts[n - 1] for n in 2:Nt)
@printf("%-28s %12.2f %12.2f %12.2f\n", "rain over the run (kg/m²)", rain(A), rain(B), rain(A) - rain(B))
LEa = [lmean(A["LE"], n) for n in 1:Nt]; LEb = [lmean(B["LE"], n) for n in 1:Nt]
Ha  = [lmean(A["H"], n) for n in 1:Nt];  Hb  = [lmean(B["H"], n) for n in 1:Nt]
@printf("%-28s %12.2f %12.2f %12.2f\n", "mean LE over the run (W/m²)", mean(LEa), mean(LEb), mean(LEa) - mean(LEb))
@printf("%-28s %12.2f %12.2f %12.2f\n", "mean H over the run (W/m²)", mean(Ha), mean(Hb), mean(Ha) - mean(Hb))
Ta = [lmean(A["Tˡᵃ"], n) for n in 1:Nt]; Tb = [lmean(B["Tˡᵃ"], n) for n in 1:Nt]
@printf("%-28s %12.2f %12.2f %12.2f\n", "land T range day 4 (K)", maximum(Ta[Nt÷4*3+1:Nt]) - minimum(Ta[Nt÷4*3+1:Nt]),
        maximum(Tb[Nt÷4*3+1:Nt]) - minimum(Tb[Nt÷4*3+1:Nt]), 0)

fig = Figure(size = (1800, 1500), fontsize = 15)
panels = (("LE — $label_a (W m⁻²)", field(A["LE"], n13), :solar, (0, 500)),
          ("LE — $label_b (W m⁻²)", field(B["LE"], n13), :solar, (0, 500)),
          ("Δ LE (A − B)", field(A["LE"], n13) .- field(B["LE"], n13), :balance, (-200, 200)),
          ("H — $label_a (W m⁻²)", field(A["H"], n13), :lajolla, (0, 400)),
          ("H — $label_b (W m⁻²)", field(B["H"], n13), :lajolla, (0, 400)),
          ("Δ H (A − B)", field(A["H"], n13) .- field(B["H"], n13), :balance, (-200, 200)),
          ("Tˡᵃ — $label_a (K)", field(A["Tˡᵃ"], n13), :thermal, (280, 320)),
          ("Tˡᵃ — $label_b (K)", field(B["Tˡᵃ"], n13), :thermal, (280, 320)),
          ("Δ Tˡᵃ (A − B, K)", field(A["Tˡᵃ"], n13) .- field(B["Tˡᵃ"], n13), :balance, (-5, 5)))
for (k, (title, data, colormap, colorrange)) in enumerate(panels)
    row, column = fldmod1(k, 3)
    ax = Axis(fig[row, 2column - 1]; title, aspect = DataAspect())
    hidedecorations!(ax)
    hm = heatmap!(ax, λ, φ, mask(data); colormap, colorrange, nan_color = :lightsteelblue1)
    Colorbar(fig[row, 2column], hm)
end
hours = times[1:Nt] ./ 3600
ax = Axis(fig[4, 1:6]; title = "land means: $label_a (solid) vs $label_b (dashed)", xlabel = "hours since 17 May 00 UTC", ylabel = "W m⁻²")
lines!(ax, hours, LEa; color = :navy, label = "LE"); lines!(ax, hours, LEb; color = :navy, linestyle = :dash)
lines!(ax, hours, Ha; color = :orangered, label = "H"); lines!(ax, hours, Hb; color = :orangered, linestyle = :dash)
lines!(ax, hours, [3600 * 100 * lmean(A["Jʳⁿ"], n) for n in 1:Nt]; color = :steelblue, label = "rain × 100 (mm hr⁻¹)")
lines!(ax, hours, [3600 * 100 * lmean(B["Jʳⁿ"], n) for n in 1:Nt]; color = :steelblue, linestyle = :dash)
axislegend(ax; position = :lt)
axT = Axis(fig[5, 1:6]; title = "land-mean slab temperature", xlabel = "hours", ylabel = "K")
lines!(axT, hours, Ta; color = :firebrick, label = label_a); lines!(axT, hours, Tb; color = :firebrick, linestyle = :dash, label = label_b)
axislegend(axT; position = :lt)
Label(fig[0, 1:6], "$label_a vs $label_b — 20 May 2011 1300 CST and land-mean series", fontsize = 18)
save("$(tag_a)_vs_$(tag_b).png", fig)
@info "Saved $(tag_a)_vs_$(tag_b).png"
