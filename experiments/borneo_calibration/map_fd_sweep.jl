# Per-cell finite-difference convergence of the map's slab-depth sensitivity against the
# adjoint map saved by `map_calibration.jl`: as the step shrinks below the model's switch
# roughness the central difference should settle on the adjoint cell by cell.

include(joinpath(@__DIR__, "map_setup.jl"))
using Statistics: cor, median

tag = "map_calibration_r$(refinement)_$(get(ENV, "ARCH", "gpu"))"
saved = jldopen(f -> Dict(k => f[k] for k in ("adjoint_map", "h₀", "losses")), "$(tag).jld2")
adjoint_map = saved["adjoint_map"]

results = []
for relative in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
    δ = relative * h₀
    fd = (forward_map(fill(h₀ + δ, Nx, Ny)).losses .- forward_map(fill(h₀ - δ, Nx, Ny)).losses) ./ 2δ
    ratio = adjoint_map[land] ./ fd[land]
    @info @sprintf("δ/h₀ = %7.1e: correlation %.4f, max |adjoint − FD| = %.3e (%.1f%% of max |FD|), median adjoint/FD %.4f, cells within 1%%: %d of %d",
                   relative, cor(vec(adjoint_map[land]), vec(fd[land])), maximum(abs.(adjoint_map .- fd)[land]),
                   100 * maximum(abs.(adjoint_map .- fd)[land]) / maximum(abs.(fd[land])), median(ratio),
                   count(abs.(ratio .- 1) .< 0.01), count(land))
    push!(results, (relative, fd))
end

jldsave("$(tag)_fd_sweep.jld2"; relatives = first.(results), fd_maps = last.(results), adjoint_map)

λ, φ = static.longitude, static.latitude
mask(a) = ifelse.(land, a, NaN)
fig = Figure(size = (2000, 900), fontsize = 14)
Label(fig[0, 1:(2 * (length(results) + 1))], "Slab-depth sensitivity of the run-mean soil-water mismatch: adjoint vs central finite difference as the step shrinks"; fontsize = 18)
slim = maximum(abs, adjoint_map[land])
ax = Axis(fig[1, 1]; title = "adjoint ∂L/∂h", aspect = DataAspect())
hm = heatmap!(ax, λ, φ, mask(adjoint_map); colormap = :balance, colorrange = (-slim, slim))
Colorbar(fig[1, 2], hm; label = "m⁻¹")
for (k, (relative, fd)) in enumerate(results)
    ax = Axis(fig[1, 2k + 1]; title = @sprintf("FD, δ/h₀ = %.0e", relative), aspect = DataAspect())
    heatmap!(ax, λ, φ, mask(fd); colormap = :balance, colorrange = (-slim, slim))
    hideydecorations!(ax)
end
ax = Axis(fig[2, 1:(2 * (length(results) + 1))]; title = "cell-by-cell", xlabel = "finite difference (m⁻¹)", ylabel = "adjoint (m⁻¹)")
for (relative, fd) in results
    scatter!(ax, vec(fd[land]), vec(adjoint_map[land]); markersize = 6, label = @sprintf("δ/h₀ = %.0e", relative))
end
lines!(ax, [-slim, slim], [-slim, slim]; color = :black, linestyle = :dash)
axislegend(ax; position = :lt)
save("$(tag)_fd_sweep.png", fig)
@info "saved $(tag)_fd_sweep.png"
