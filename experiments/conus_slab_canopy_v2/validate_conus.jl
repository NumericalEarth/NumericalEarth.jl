# Quantitative checks on a finished run: spin-up repeatability, case-day flux partition,
# urban signature, water-budget closure, energy closure at ARM SGP, and the extremes.
#   julia --project=<docs> validate_conus.jl [TAG]
using Oceananigans
using NumericalEarth
using JLD2
using Printf
using Statistics: mean, median, quantile

tag = isempty(ARGS) ? "conus12km_v2" : ARGS[1]
series(name) = FieldTimeSeries("$(tag)_land.jld2", name; backend = OnDisk())

LST = series("LST"); 𝒮 = series("𝒮"); T = series("Tˡᵃ")
Tᵛ = series("Tᵛ"); Tᵍ = series("Tᵍ"); Tᵃᶜ = series("Tᵃᶜ")
LE = series("LE"); H = series("H"); Gᶜ = series("Gᶜ"); u★ = series("u★")
W = series("W"); Wᶜ = series("Wᶜ"); Wᵖ = series("Wᵖ")
P = series("P"); E = series("E"); R = series("R"); D = series("D")
Jʳⁿ = series("Jʳⁿ"); Eʷ = series("Eʷ"); α = series("αᵉᶠᶠ")
SW = series("ℐꜜˢʷ"); LW = series("ℐꜜˡʷ")

times = LST.times
Nt = length(times)
grid = LST.grid
λ, φ, _ = nodes(grid, Center(), Center(), Center())

static = jldopen("$(tag)_static.jld2")
water = static["water"]
lai = static["leaf_area_index"]
fveg = static["vegetation_fraction"]
urban = static["urban_cover"]
ε = static["emissivity"]
land_cells = .!water
println("run $tag, land cover $(static["landcover_source"]); land cells: ", count(land_cells), " / ", length(land_cells))

field(fts, n) = interior(fts[n], :, :, 1)
lmean(fts, n) = mean(field(fts, n)[land_cells])

println("\n== spin-up: land-mean LST daily min/max (K)")
for d in 0:3
    sel = [n for n in 1:Nt if d * 86400 <= times[n] < (d + 1) * 86400]
    vals = [lmean(LST, n) for n in sel]
    @printf("  day %d (%s): min %.2f  max %.2f  amplitude %.2f\n", d + 1,
            ("17 May", "18 May", "19 May", "20 May")[d + 1], minimum(vals), maximum(vals), maximum(vals) - minimum(vals))
end

n13 = argmin(abs.(times .- (3 * 86400 + 19 * 3600)))
LE13 = field(LE, n13); H13 = field(H, n13); 𝒮13 = field(𝒮, n13); LST13 = field(LST, n13); u13 = field(u★, n13)
println("\n== case-day 1300 CST (frame $n13):")
@printf("  land LE: mean %.1f  10%% %.1f  90%% %.1f W/m²\n", mean(LE13[land_cells]), quantile(vec(LE13[land_cells]), (0.1, 0.9))...)
@printf("  land H : mean %.1f  10%% %.1f  90%% %.1f W/m²\n", mean(H13[land_cells]), quantile(vec(H13[land_cells]), (0.1, 0.9))...)
for (lo, hi) in ((0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))
    band = land_cells .& (lo .<= 𝒮13 .< hi)
    @printf("  𝒮 ∈ [%.1f, %.1f): %6d cells  Bowen %.2f  LE %.0f  H %.0f W/m²\n", lo, min(hi, 1), count(band),
            mean(H13[band]) / mean(LE13[band]), mean(LE13[band]), mean(H13[band]))
end

iˢᵍᵖ = argmin(abs.(λ .+ 97.485)); jˢᵍᵖ = argmin(abs.(φ .- 36.605))
@printf("  SGP pixel: LE %.1f  H %.1f  𝒮 %.2f  LAI %.2f  fveg %.2f\n", LE13[iˢᵍᵖ, jˢᵍᵖ], H13[iˢᵍᵖ, jˢᵍᵖ], 𝒮13[iˢᵍᵖ, jˢᵍᵖ], lai[iˢᵍᵖ, jˢᵍᵖ], fveg[iˢᵍᵖ, jˢᵍᵖ])

println("\n== urban signature at 1300 CST (built-up land fraction > 0.3 vs rural vegetated cells):")
urban_cells = land_cells .& (urban .> 0.3)
rural_cells = land_cells .& (urban .< 0.02) .& (fveg .> 0.3)
for (name, cells) in (("urban", urban_cells), ("rural", rural_cells))
    @printf("  %-5s %6d cells: LST %.1f K  H %.0f  LE %.0f W/m²  u★ %.2f m/s  ℓᵐ(bare) %.3f m\n", name, count(cells),
            mean(LST13[cells]), mean(H13[cells]), mean(LE13[cells]), mean(u13[cells]), exp(mean(log.(static["momentum_roughness_bare"][cells]))))
end

println("\n== extremes:")
Tᵛ13 = field(Tᵛ, n13); imax = argmax(Tᵛ13)
@printf("  max Tᵛ at 1300 CST %.1f K at (%.2f, %.2f), LAI %.2f, fveg %.2f, water %s\n", Tᵛ13[imax], λ[imax[1]], φ[imax[2]], lai[imax], fveg[imax], water[imax])
LEmax, nmax = findmax(n -> maximum(field(LE, n)), 1:Nt)
LEn = field(LE, nmax); imaxLE = argmax(LEn)
@printf("  max LE over the run %.0f W/m² at frame %d (t = %.1f h), (%.2f, %.2f), water %s, 𝒮 %.2f, fveg %.2f, u★ %.2f\n",
        LEmax, nmax, times[nmax] / 3600, λ[imaxLE[1]], φ[imaxLE[2]], water[imaxLE], field(𝒮, nmax)[imaxLE], fveg[imaxLE], field(u★, nmax)[imaxLE])
@printf("  land cells with LE > 800 W/m² at that frame: %d (of which water-class: %d)\n", count(LEn .> 800), count((LEn .> 800) .& water))
excess = [quantile(vec((field(Tᵛ, n) .- field(Tᵃᶜ, n))[land_cells]), 0.999) for n in n13-8:n13+8]
@printf("  leaf − canopy-air excess, q99.9 over case-day midday frames: %.1f K\n", maximum(excess))

Δts = diff(times)
acc(fts; sgn = 1) = sum(sgn * lmean(fts, n) * Δts[n - 1] for n in 2:Nt)
rain = acc(Jʳⁿ); evap = acc(E); wet_canopy = acc(Eʷ); runoff = acc(R); drainage = acc(D; sgn = -1)
storage = (lmean(W, Nt) - lmean(W, 1)) + (lmean(Wᶜ, Nt) - lmean(Wᶜ, 1)) + (lmean(Wᵖ, Nt) - lmean(Wᵖ, 1))
println("\n== land water budget over 96 h (land-mean, kg/m²):")
@printf("  rain in %.2f | evap %.2f + wet-canopy %.2f + runoff %.2f + drainage %.2f | Δstorage %.2f\n", rain, evap, wet_canopy, runoff, drainage, storage)
residual = rain - (evap + wet_canopy + runoff + drainage) - storage
@printf("  residual (rain − losses − Δstorage): %.3f kg/m² (%.1f%% of rain)\n", residual, 100 * abs(residual) / max(rain, 1e-9))

σˢᵇ = 5.670374419e-8
at(fts, n) = interior(fts[n], iˢᵍᵖ, jˢᵍᵖ, 1)[]
case = [n for n in 1:Nt if times[n] >= 3 * 86400]
resid = [(1 - at(α, n)) * at(SW, n) + ε[iˢᵍᵖ, jˢᵍᵖ] * (at(LW, n) - σˢᵇ * at(LST, n)^4) - (at(H, n) + at(LE, n) + at(Gᶜ, n)) for n in case]
@printf("\n== SGP surface energy residual (Rn − H − LE − G) over the case day: median %.1f  max|.| %.1f W/m²\n", median(abs.(resid)), maximum(abs.(resid)))
@printf("== SGP midday temperatures: Tᵛ %.1f  Tᵍ %.1f  Tᵃᶜ %.1f  LST %.1f  slab %.1f K\n",
        at(Tᵛ, n13), at(Tᵍ, n13), at(Tᵃᶜ, n13), at(LST, n13), at(T, n13))
close(static)
