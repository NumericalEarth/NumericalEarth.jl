# Which structure damps the slab's excursions without slowing its response? At the observation
# layer's own depth h = 0.28 m, with the calibrated drainage K₀, 20-day eager runs of
# structural alternatives scored per window against ERA5-Land:
#
#   - free drainage (the configuration of the conductivity-only calibration);
#   - Darcy exchange with a deep reservoir held at the head of ERA5-Land's 28–100 cm layer
#     (capillary resupply during dry-downs, pass-through during storms), for several
#     exchange lengths;
#   - a per-cell infiltration capacity, reduced in steps, with saturation-excess runoff.
#
#   REFINEMENT=1 NSTEPS=2880 END_DATE=2020-04-22 julia --project=docs exchange_experiments.jl

include(joinpath(@__DIR__, "map_setup.jl"))
using Oceananigans.OutputReaders: FieldTimeSeries

q = jldopen(f -> f["q"], "map_logK_r$(refinement)_gpu.jld2")
θ_obs = hourly_observations()
Nh = size(θ_obs, 1)
windows = ("days 0–6.25" => 1:151, "days 6.25–12.25" => 151:295, "days 12.25–20" => 295:Nh)

# The deep reservoir's pressure head: ERA5-Land's 28–100 cm water content through each
# cell's own van Genuchten curve, Π = −(1/α)(𝒮^(−1/m) − 1)^(1/n).
function deep_head(θ, α, n, ν, θʳ)
    𝒮 = clamp((θ - θʳ) / (ν - θʳ), 1e-6, 1)
    m = 1 - 1 / n
    return 𝒮 ≥ 1 ? 0.0 : -(𝒮^(-1 / m) - 1)^(1 / n) / α
end
Πᵈ = FieldTimeSeries{Center, Center, Nothing}(cpu_grid, era5_land.times)
for k in eachindex(era5_land.times)
    interior(Πᵈ[k], :, :, 1) .= deep_head.(era5_land.layer_3[k, :, :], static.inverse_air_entry_head,
                                               static.pore_size_uniformity, static.porosity, static.residual_liquid_fraction)
end
@info @sprintf("deep head from ERA5-Land layer 3: median %.2f m, range [%.2f, %.2f] m", median(interior(Πᵈ)), extrema(interior(Πᵈ))...)

capacity(mm_per_hour) = surface_property(cpu_grid, fill(FT(mm_per_hour / 3600), Nx, Ny))
runs = [
    "free drainage" => (;),
    ["Darcy exchange, ℓ = $(ℓ) m" => (; deep_liquid_flux = DarcyDeepLiquidFlux(FT; exchange_length = ℓ), deep_pressure_head = Πᵈ)
     for ℓ in (0.18, 0.36, 0.72, 1.5)]...,
    ["infiltration ≤ $(c) mm h⁻¹" => (; infiltration_capacity = capacity(c)) for c in (5, 2, 1)]...,
]

results = Dict{String, Any}()
for (name, hydrology) in runs
    result = forward_map(fill(h₀, Nx, Ny); record = true, modify! = with_conductivity(q), hydrology...)
    results[name] = result.snapshots.θ
    line = join([@sprintf("  %s: RMS %.4f r %.2f σ %.2f", w, s.rms, s.r, s.σ) for (w, s) in
                 ((w, window_scores(result.snapshots.θ, θ_obs, hours)) for (w, hours) in windows)], " |")
    @info @sprintf("%-28s", name) * line
end

jldsave("exchange_experiments_r$(refinement).jld2"; θ_obs, runs = Dict(k => v for (k, v) in results), windows = Dict(windows))

# ## Figure: domain-median trajectories for each family

med(x) = [median(x[k, :, :][land]) for k in 1:Nh]
days = (0:Nh-1) ./ 24
fig = Figure(size = (1600, 900), fontsize = 15)
Label(fig[0, 1], "Structural alternatives at h = 0.28 m with calibrated K₀, 20 days against ERA5-Land"; fontsize = 18)
for (row, family, palette) in ((1, "Darcy exchange", Makie.categorical_colors(:viridis, 5)),
                               (2, "infiltration", Makie.categorical_colors(:plasma, 4)))
    ax = Axis(fig[row, 1]; title = family, xlabel = "day", ylabel = "θ (m³ m⁻³)")
    vspan!(ax, 0, 6.25; color = (:gray, 0.08))
    lines!(ax, days, med(θ_obs); color = :black, linewidth = 2.5, label = "ERA5-Land")
    lines!(ax, days, med(results["free drainage"]); color = :firebrick, linewidth = 2, label = "free drainage")
    for (i, (name, _)) in enumerate(filter(r -> startswith(first(r), family), runs))
        lines!(ax, days, med(results[name]); color = palette[i], linewidth = 2, label = name)
    end
    axislegend(ax; position = :rb, nbanks = 2)
end
save("exchange_experiments_r$(refinement).png", fig)
@info "saved exchange_experiments_r$(refinement).png"
