# Does the depth calibration track ERA5-Land θ in time, or only its mean? And which
# hydraulic parameter unlocks tracking? Per-cell correlation and variance-ratio
# diagnostics at the calibrated depths, then K₀ and n sweeps at h = h₀ (the observation
# layer's own depth, so model and target θ share their support).

include(joinpath(@__DIR__, "map_setup.jl"))
using Statistics: std

it = jldopen(f -> Dict(k => f[k] for k in keys(f)), "map_iterations_r1_gpu_two_sided.jld2")
calibrated_depths = it["depths"]

Nh = Nsteps ÷ round(Int, 3600 / Δt) + 1
θ_obs = zeros(Nh, Nx, Ny)
for k in 1:Nh
    θ_obs[k, :, :] .= era5_land_soil_water(era5_land, k)
end

function tracking(result)
    θm = result.snapshots.θ
    r  = fill(NaN, Nx, Ny); σratio = fill(NaN, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        land[i, j] || continue
        m, o = θm[:, i, j], θ_obs[:, i, j]
        σm, σo = std(m), std(o)
        σratio[i, j] = σm / σo
        (σm > 0 && σo > 0) && (r[i, j] = cor(m, o))
    end
    ok = [r[c] for c in CartesianIndices(r) if land[c] && !isnan(r[c])]
    return (; rms = rms(result.losses), median_r = median(ok), q1_r = quantile(ok, 0.25),
              median_σratio = median([σratio[c] for c in CartesianIndices(σratio) if land[c]]))
end

report(name, t) = @info @sprintf("%-34s run-RMS %.4f, median r %.3f (q1 %.3f), median σ_model/σ_obs %.3f",
                                 name, t.rms, t.median_r, t.q1_r, t.median_σratio)

hydrology_of(model) = model.land.hydrology.soil.soil

scale_K!(scale) = model -> (parent(hydrology_of(model).hydraulic_conductivity.matching_point_conductivity) .*= scale; nothing)
function shift_n!(δ)
    return function (model)
        h = hydrology_of(model)
        parent(h.retention_curve.pore_size_uniformity) .+= δ
        parent(h.hydraulic_conductivity.pore_size_uniformity) .+= δ
        return nothing
    end
end
both!(scale, δ) = model -> (scale_K!(scale)(model); shift_n!(δ)(model); nothing)

report("calibrated depths (two-sided)", tracking(forward_map(calibrated_depths; record = true)))
report("uniform h = h₀ = 0.28 m", tracking(forward_map(fill(h₀, Nx, Ny); record = true)))

for scale in (1e1, 1e2, 1e3, 1e4)
    report(@sprintf("h₀, K₀ × 10^%g", log10(scale)), tracking(forward_map(fill(h₀, Nx, Ny); record = true, modify! = scale_K!(scale))))
end
for δ in (0.1, 0.2, 0.4, 0.8)
    report(@sprintf("h₀, n + %.1f", δ), tracking(forward_map(fill(h₀, Nx, Ny); record = true, modify! = shift_n!(δ))))
end
for scale in (1e1, 1e2), δ in (0.2, 0.4)
    report(@sprintf("h₀, K₀ × 10^%g, n + %.1f", log10(scale), δ), tracking(forward_map(fill(h₀, Nx, Ny); record = true, modify! = both!(scale, δ))))
end
