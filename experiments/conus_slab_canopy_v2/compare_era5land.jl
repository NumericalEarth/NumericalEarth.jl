# Compare the 12 km finals against ERA5 skin temperature and ERA5-Land layer-1 soil water:
# hourly maps, land-mean and ARM SGP series, per-day statistics, and two videos.
#
# The model's soil water is a 0–50 cm slab and ERA5-Land's layer 1 is 0–7 cm, so soil
# water is compared through its change since the initial time, which both share.
#
# usage: julia compare_era5land.jl   (env: CANOPY_TAG, BUCKET_TAG, OUT)

using NumericalEarth
using CopernicusClimateDataStore
using Oceananigans
using NCDatasets
using JLD2
using CairoMakie
using Printf
using Statistics: mean
using Dates
import Downloads

canopy_tag = get(ENV, "CANOPY_TAG", "conus12km_final_canopy")
bucket_tag = get(ENV, "BUCKET_TAG", "conus12km_final_bucket")
out = get(ENV, "OUT", "era5_check_12km")

start_date = DateTime(2011, 5, 17, 0)
stop_date  = DateTime(2011, 5, 23, 0)
dates = collect(start_date:Hour(1):stop_date)
era5_dir = "era5"
slab_depth = 0.5

# ## ERA5 hourly skin temperature (0.25°): download the hours the runs did not need

region = BoundingBox(longitude = (-134.5, -60.5), latitude = (20.5, 53.5))
Downloads.download(MetadataSet(:skin_temperature; dataset = ERA5HourlySingleLevel(),
                               dates, region, dir = era5_dir); threads = 4)

skt_path(date) = joinpath(era5_dir,
    "skin_temperature_ERA5HourlySingleLevel_$(Dates.format(date, "yyyy-mm-ddTHH"))_-134.5_-60.5_20.5_53.5.nc")

# ## Model fields, sampled hourly

static = jldopen("$(canopy_tag)_static.jld2")
λ = static["longitude"]; φ = static["latitude"]
water = static["water"]
land = .!water
Nx, Ny = length(λ), length(φ)

series(file, name) = FieldTimeSeries(file, name; backend = OnDisk())
slab(ts, n) = Float32.(interior(ts[n], :, :, 1))

LST_ts = series("$(canopy_tag)_land.jld2", "LST")
W_A_ts = series("$(canopy_tag)_land.jld2", "W")
T_C_ts = series("$(bucket_tag)_land.jld2", "Tˡᵃ")
𝒮_C_ts = series("$(bucket_tag)_land.jld2", "𝒮")

Nt = length(dates)
hourly = [findfirst(t -> abs(t - 3600 * (n - 1)) < 1, LST_ts.times) for n in 1:Nt]
@assert all(!isnothing, hourly)

T_A = Array{Float32}(undef, Nx, Ny, Nt); θ_A = similar(T_A)
T_C = similar(T_A);                      θ_C = similar(T_A)
for n in 1:Nt
    T_A[:, :, n] = slab(LST_ts, hourly[n])
    θ_A[:, :, n] = slab(W_A_ts, hourly[n]) ./ (1000slab_depth)
    T_C[:, :, n] = slab(T_C_ts, hourly[n])
    θ_C[:, :, n] = slab(𝒮_C_ts, hourly[n]) .* 0.45   # the bucket's initialization scale
end

# ## ERA5 products, interpolated bilinearly to the model grid

function bilinear_stack(λs, φs, F, λt, φt)
    if φs[1] > φs[end]
        φs = reverse(φs); F = reverse(F; dims = 2)
    end
    G = Array{Float32}(undef, length(λt), length(φt), size(F, 3))
    for (j, y) in enumerate(φt), (i, x) in enumerate(λt)
        i₀ = clamp(searchsortedlast(λs, x), 1, length(λs) - 1)
        j₀ = clamp(searchsortedlast(φs, y), 1, length(φs) - 1)
        wx = (x - λs[i₀]) / (λs[i₀+1] - λs[i₀])
        wy = (y - φs[j₀]) / (φs[j₀+1] - φs[j₀])
        for n in axes(F, 3)
            G[i, j, n] = (1-wx) * (1-wy) * F[i₀, j₀,   n] + wx * (1-wy) * F[i₀+1, j₀,   n] +
                         (1-wx) *    wy  * F[i₀, j₀+1, n] + wx *    wy  * F[i₀+1, j₀+1, n]
        end
    end
    return G
end

nan(x) = Float32.(replace(x, missing => NaN32))

T_E = let
    first_file = Dataset(skt_path(dates[1]))
    λs = nan(first_file["longitude"][:]); φs = nan(first_file["latitude"][:])
    close(first_file)
    F = Array{Float32}(undef, length(λs), length(φs), Nt)
    for (n, date) in enumerate(dates)
        Dataset(skt_path(date)) do ds
            F[:, :, n] = nan(ds["skt"][:, :, 1])
        end
    end
    bilinear_stack(λs, φs, F, λ, φ)
end

swvl_path = joinpath(era5_dir, "volumetric_soil_water_layer_1_ERA5HourlyLand_2011_-134.0_-61.0_21.0_53.0.nc")
θ_E = Dataset(swvl_path) do ds
    i₀ = findfirst(==(start_date), nomissing(ds["valid_time"][:]))
    bilinear_stack(nan(ds["longitude"][:]), nan(ds["latitude"][:]),
                   nan(ds["swvl1"][:, :, i₀:i₀+Nt-1]), λ, φ)
end

Δθ(θ) = θ .- θ[:, :, 1:1]
Δθ_A = Δθ(θ_A); Δθ_C = Δθ(θ_C); Δθ_E = Δθ(θ_E)

# ## Statistics on the common valid land cells, area-weighted

valid = land .& all(isfinite, T_E; dims = 3)[:, :, 1] .& isfinite.(θ_E[:, :, 1])
weights = repeat(cosd.(φ)', Nx, 1) .* valid
land_mean(F) = [sum(F[:, :, n] .* weights) / sum(weights) for n in axes(F, 3)]

function pattern_stats(A, B, n)
    a = A[:, :, n][valid]; b = B[:, :, n][valid]; w = weights[valid]
    bias = sum((a .- b) .* w) / sum(w)
    rmse = sqrt(sum((a .- b) .^ 2 .* w) / sum(w))
    ā = sum(a .* w) / sum(w); b̄ = sum(b .* w) / sum(w)
    corr = sum((a .- ā) .* (b .- b̄) .* w) /
           sqrt(sum((a .- ā) .^ 2 .* w) * sum((b .- b̄) .^ 2 .* w))
    return bias, rmse, corr
end

sgp = (argmin(abs.(λ .+ 97.485)), argmin(abs.(φ .- 36.605)))

open("$(out)_stats.txt", "w") do io
    for (label, M) in (("canopy LST", T_A), ("bucket skin", T_C))
        println(io, "$label vs ERA5 skin temperature (19Z ≈ 1300 CST):")
        for day in 17:22
            n = findfirst(==(DateTime(2011, 5, day, 19)), dates)
            b, r, c = pattern_stats(M, T_E, n)
            @printf(io, "  May %2d  bias %+6.2f K  rmse %5.2f K  r %5.3f\n", day, b, r, c)
        end
    end
    println(io, "soil-water change over 144 h vs ERA5-Land layer 1:")
    for (label, M) in (("canopy Δθ(0–50 cm)", Δθ_A), ("bucket Δθ", Δθ_C))
        b, r, c = pattern_stats(M, Δθ_E, Nt)
        @printf(io, "  %-20s bias %+7.4f  rmse %6.4f  r %5.3f\n", label, b, r, c)
    end
end
println(read("$(out)_stats.txt", String))

# ## Maps

mask(F) = ifelse.(valid, F, NaN32)
timestamp(n) = Dates.format(dates[n], "u d HH:00") * "Z"

function temperature_maps(n, filename)
    fig = Figure(size = (1750, 780))
    for (row, model, label) in ((1, T_A, "canopy LST"), (2, T_C, "bucket skin"))
        pairs = ((model[:, :, n], "$label, $(timestamp(n))", :thermal, (270, 315)),
                 (T_E[:, :, n], "ERA5 skin temperature", :thermal, (270, 315)),
                 (model[:, :, n] .- T_E[:, :, n], "model − ERA5", :balance, (-10, 10)))
        for (col, (F, title, cmap, crange)) in enumerate(pairs)
            ax = Axis(fig[row, 2col-1]; title, xlabel = "λ (°)", ylabel = "φ (°)")
            hm = heatmap!(ax, λ, φ, mask(F); colormap = cmap, colorrange = crange)
            Colorbar(fig[row, 2col], hm)
        end
    end
    save(filename, fig)
end

temperature_maps(findfirst(==(DateTime(2011, 5, 20, 19)), dates), "$(out)_skin_day.png")
temperature_maps(findfirst(==(DateTime(2011, 5, 20, 9)),  dates), "$(out)_skin_night.png")

let fig = Figure(size = (1750, 780))
    pairs = ((1, 1, θ_A[:, :, Nt], "canopy θ (0–50 cm), $(timestamp(Nt))", :viridis, (0, 0.5)),
             (1, 2, θ_E[:, :, Nt], "ERA5-Land θ (0–7 cm)", :viridis, (0, 0.5)),
             (1, 3, θ_C[:, :, Nt], "bucket θ equivalent", :viridis, (0, 0.5)),
             (2, 1, Δθ_A[:, :, Nt], "canopy Δθ over 144 h", :balance, (-0.15, 0.15)),
             (2, 2, Δθ_E[:, :, Nt], "ERA5-Land Δθ over 144 h", :balance, (-0.15, 0.15)),
             (2, 3, Δθ_C[:, :, Nt], "bucket Δθ over 144 h", :balance, (-0.15, 0.15)))
    for (row, col, F, title, cmap, crange) in pairs
        ax = Axis(fig[row, 2col-1]; title, xlabel = "λ (°)", ylabel = "φ (°)")
        hm = heatmap!(ax, λ, φ, mask(F); colormap = cmap, colorrange = crange)
        Colorbar(fig[row, 2col], hm)
    end
    save("$(out)_soil_maps.png", fig)
end

# ## Series: land means and the ARM SGP pixel

hours = 0:Nt-1
let fig = Figure(size = (1500, 900))
    axT = Axis(fig[1, 1]; title = "land-mean skin temperature", xlabel = "hours since 17 May 00Z", ylabel = "K")
    lines!(axT, hours, land_mean(T_A); label = "canopy LST")
    lines!(axT, hours, land_mean(T_C); label = "bucket skin")
    lines!(axT, hours, land_mean(T_E); color = :black, linestyle = :dash, label = "ERA5")
    axislegend(axT, position = :lt)

    axP = Axis(fig[1, 2]; title = "ARM SGP skin temperature", xlabel = "hours since 17 May 00Z", ylabel = "K")
    lines!(axP, hours, T_A[sgp..., :]; label = "canopy LST")
    lines!(axP, hours, T_C[sgp..., :]; label = "bucket skin")
    lines!(axP, hours, T_E[sgp..., :]; color = :black, linestyle = :dash, label = "ERA5")
    axislegend(axP, position = :lt)

    axθ = Axis(fig[2, 1]; title = "land-mean soil-water change", xlabel = "hours since 17 May 00Z", ylabel = "Δθ (m³ m⁻³)")
    lines!(axθ, hours, land_mean(Δθ_A); label = "canopy (0–50 cm)")
    lines!(axθ, hours, land_mean(Δθ_C); label = "bucket")
    lines!(axθ, hours, land_mean(Δθ_E); color = :black, linestyle = :dash, label = "ERA5-Land (0–7 cm)")
    axislegend(axθ, position = :lt)

    axs = Axis(fig[2, 2]; title = "ARM SGP soil-water change", xlabel = "hours since 17 May 00Z", ylabel = "Δθ (m³ m⁻³)")
    lines!(axs, hours, Δθ_A[sgp..., :]; label = "canopy (0–50 cm)")
    lines!(axs, hours, Δθ_C[sgp..., :]; label = "bucket")
    lines!(axs, hours, Δθ_E[sgp..., :]; color = :black, linestyle = :dash, label = "ERA5-Land (0–7 cm)")
    axislegend(axs, position = :lt)

    save("$(out)_series.png", fig)
end

# ## Videos: skin temperature and soil-water change against ERA5, hourly

let n = Observable(1)
    fig = Figure(size = (1750, 420))
    panels = ((@lift(mask(T_A[:, :, $n])), @lift("canopy LST, " * timestamp($n)), :thermal, (270, 315)),
              (@lift(mask(T_E[:, :, $n])), "ERA5 skin temperature", :thermal, (270, 315)),
              (@lift(mask(T_A[:, :, $n] .- T_E[:, :, $n])), "canopy − ERA5", :balance, (-10, 10)))
    for (col, (F, title, cmap, crange)) in enumerate(panels)
        ax = Axis(fig[1, 2col-1]; title, xlabel = "λ (°)", ylabel = "φ (°)")
        hm = heatmap!(ax, λ, φ, F; colormap = cmap, colorrange = crange)
        Colorbar(fig[1, 2col], hm)
    end
    CairoMakie.record(nn -> n[] = nn, fig, "$(out)_skin.mp4", 1:Nt; framerate = 6)
end

let n = Observable(1)
    fig = Figure(size = (1750, 420))
    panels = ((@lift(mask(Δθ_A[:, :, $n])), @lift("canopy Δθ (0–50 cm), " * timestamp($n))),
              (@lift(mask(Δθ_E[:, :, $n])), "ERA5-Land Δθ (0–7 cm)"),
              (@lift(mask(Δθ_C[:, :, $n])), "bucket Δθ"))
    for (col, (F, title)) in enumerate(panels)
        ax = Axis(fig[1, 2col-1]; title, xlabel = "λ (°)", ylabel = "φ (°)")
        hm = heatmap!(ax, λ, φ, F; colormap = :balance, colorrange = (-0.15, 0.15))
        Colorbar(fig[1, 2col], hm)
    end
    CairoMakie.record(nn -> n[] = nn, fig, "$(out)_soil.mp4", 1:Nt; framerate = 6)
end

@info "ERA5 comparison written: $(out)_{stats.txt, skin_day.png, skin_night.png, soil_maps.png, series.png, skin.mp4, soil.mp4}"
