# Out-of-sample validation of the joint (h, K₀) calibration: one continuous run from the
# calibration initial state (1 Apr 2020) over the full ingested window — the first
# 6.25 days are the fitting window, everything after is held out — scored against
# ERA5-Land per window, beside the uncalibrated model. Requires caches ingested with an
# extended window, e.g. END_DATE=2020-04-22.
#
#   REFINEMENT=1 NSTEPS=2880 END_DATE=2020-04-22 CALIBRATION=map_joint_r1_gpu CALIBRATION_STEPS=900 julia --project=docs map_validation.jl

include(joinpath(@__DIR__, "map_setup.jl"))
using Statistics: std

calibration = get(ENV, "CALIBRATION", "map_joint_r$(refinement)_gpu")
calibration_steps = parse(Int, get(ENV, "CALIBRATION_STEPS", "900"))
tag = "validation_of_$(calibration)"
steps_per_hour = round(Int, 3600 / Δt)
Nh = Nsteps ÷ steps_per_hour + 1
length(era5_land.times) ≥ Nh || error("era5_land cache holds $(length(era5_land.times)) hours, need $Nh; re-ingest with a later END_DATE")

it = jldopen(f -> Dict(k => f[k] for k in keys(f)), "$(calibration).jld2")
calibrated_depths = haskey(it, "depths") ? it["depths"] : fill(FT(it["h₀"]), Nx, Ny)
q = haskey(it, "q") ? it["q"] : log10.(max.(static.matching_point_conductivity, 1e-9))
q_conductivity_only = jldopen(f -> f["q"], get(ENV, "CONDUCTIVITY_ONLY", "map_logK_r$(refinement)_gpu$(tag_suffix).jld2"))

conductivity_field(model) = model.land.hydrology.soil.soil.hydraulic_conductivity.matching_point_conductivity
cpu_scratch = surface_field(land_grid(CPU(), FT))
function set_cells!(field, values, fill_value)
    set!(cpu_scratch, values)
    parent(cpu_scratch) .= ifelse.(parent(cpu_scratch) .== 0, fill_value, parent(cpu_scratch))
    parent(field) .= parent(cpu_scratch)
    return field
end
q_fill = median(q[land])
with_conductivity(qv) = model -> (set_cells!(conductivity_field(model), exp10.(qv), exp10(q_fill)); nothing)

calibrated   = forward_map(calibrated_depths; record = true, modify! = with_calibration(it))
conductivity_only = forward_map(fill(h₀, Nx, Ny); record = true, modify! = with_conductivity(q_conductivity_only))
uncalibrated = forward_map(fill(h₀, Nx, Ny); record = true)

# ## Window-split scores on the hourly series

θ_obs = zeros(Nh, Nx, Ny)
for m in 1:Nh
    θ_obs[m, :, :] .= era5_land_soil_water(era5_land, m)
end
calibration_hours = 1:(calibration_steps ÷ steps_per_hour + 1)
validation_hours  = last(calibration_hours):Nh

function window_scores(result, hours)
    r = Float64[]; σratio = Float64[]; mse = Float64[]
    for c in findall(land)
        m, o = result.snapshots.θ[hours, c], θ_obs[hours, c]
        push!(mse, sum(abs2, m .- o) / length(hours))
        push!(σratio, std(m) / std(o))
        std(m) > 0 && push!(r, cor(m, o))
    end
    return (; rms = sqrt(mean(mse)), median_r = median(r), median_σratio = median(σratio),
              cell_rms = [sqrt(sum(abs2, result.snapshots.θ[hours, i, j] .- θ_obs[hours, i, j]) / length(hours))
                          for i in 1:Nx, j in 1:Ny])
end

for (name, result) in (("calibrated (h, K₀)", calibrated), ("K₀ only, h = h₀", conductivity_only), ("uncalibrated", uncalibrated))
    fit, held = window_scores(result, calibration_hours), window_scores(result, validation_hours)
    @info @sprintf("%-18s fitting window: RMS %.4f, median r %.3f, σ ratio %.2f;  held out (%.1f days): RMS %.4f, median r %.3f, σ ratio %.2f",
                   name, fit.rms, fit.median_r, fit.median_σratio, (Nh - last(calibration_hours)) / 24,
                   held.rms, held.median_r, held.median_σratio)
end

cal_fit, cal_held = window_scores(calibrated, calibration_hours), window_scores(calibrated, validation_hours)
unc_held = window_scores(uncalibrated, validation_hours)

jldsave("$(tag).jld2"; θ_obs, calibration_steps, Nsteps,
        snapshots = Dict(pairs(calibrated.snapshots)), snapshots_uncalibrated = Dict(pairs(uncalibrated.snapshots)),
        cell_rms_fit = cal_fit.cell_rms, cell_rms_held = cal_held.cell_rms, cell_rms_held_uncalibrated = unc_held.cell_rms,
        weight, longitude = static.longitude, latitude = static.latitude)

# ## Figure

λ, φ = static.longitude, static.latitude
mask(a) = ifelse.(land, a, NaN)
med(x) = [median(x[m, :, :][land]) for m in 1:Nh]
t_split = (last(calibration_hours) - 1) / 24

fig = Figure(size = (1800, 900), fontsize = 16)
Label(fig[0, 1:6], @sprintf("Out-of-sample validation, Central Borneo: calibrated on days 0–%.2f, held out through day %.2f — held-out RMS %.4f (calibrated) vs %.4f (uncalibrated) m³ m⁻³",
                            t_split, (Nh - 1) / 24, cal_held.rms, unc_held.rms); fontsize = 19)

ax = Axis(fig[1, 1:6]; title = "domain-median θ(t)", xlabel = "day", ylabel = "θ (m³ m⁻³)")
days = (0:Nh-1) ./ 24
vspan!(ax, t_split, (Nh - 1) / 24; color = (:seagreen, 0.08))
lines!(ax, days, med(θ_obs); color = :black, linewidth = 2, label = "ERA5-Land")
lines!(ax, days, med(uncalibrated.snapshots.θ); color = :steelblue, linewidth = 2, label = "uncalibrated")
lines!(ax, days, med(conductivity_only.snapshots.θ); color = :darkorange, linewidth = 2, label = "K₀ only, h = h₀")
lines!(ax, days, med(calibrated.snapshots.θ); color = :firebrick, linewidth = 2, label = "calibrated (h, K₀)")
vlines!(ax, [t_split]; color = :gray30, linestyle = :dash)
text!(ax, t_split + 0.15, 0.0; text = "held out →", space = :relative, align = (:left, :bottom), color = :gray30)
axislegend(ax; position = :rb)

function panel!(pos, data, title, label; colormap = :amp, colorrange)
    axp = Axis(fig[pos...]; title, aspect = DataAspect(), xlabel = "longitude", ylabel = "latitude")
    hm = heatmap!(axp, λ, φ, data; colormap, colorrange)
    Colorbar(fig[pos[1], pos[2] + 1], hm; label)
    return axp
end

rlim = (0, maximum(filter(isfinite, mask(unc_held.cell_rms))))
panel!((2, 1), mask(cal_held.cell_rms), "held-out RMS, calibrated", "m³ m⁻³"; colorrange = rlim)
panel!((2, 3), mask(unc_held.cell_rms), "held-out RMS, uncalibrated", "m³ m⁻³"; colorrange = rlim)
ax2 = Axis(fig[2, 5:6]; title = "per-cell RMS: fitting vs held out (calibrated)",
           xlabel = "fitting-window RMS", ylabel = "held-out RMS")
lims = (0, maximum(filter(isfinite, [mask(cal_fit.cell_rms); mask(cal_held.cell_rms)])) * 1.05)
lines!(ax2, [0, lims[2]], [0, lims[2]]; color = :gray, linestyle = :dash)
scatter!(ax2, cal_fit.cell_rms[land], cal_held.cell_rms[land]; color = :firebrick, markersize = 6, alpha = 0.6)
xlims!(ax2, lims); ylims!(ax2, lims)

save("$(tag).png", fig)
@info "saved $(tag).png"
