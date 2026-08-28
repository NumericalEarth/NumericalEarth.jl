# One Central Borneo forest column: the vegetated slab land forced by ERA5, compared with
# ERA5-Land soil water, then differentiated. The soil-water mismatch over the run,
#
#     L(h) = (1/N) Σₙ (θ(tₙ; h) − θᴱᴿᴬ⁵ᴸ(tₙ))²,        θ = Mˡᵃ / (ρˡ h),
#
# is differentiated with respect to the slab depth `h` by Enzyme reverse mode through the
# Reactant-compiled coupled time step (the sum is accumulated inside the traced loop),
# checked against a finite difference, and one gradient step with a line search along
# −dL/dh is taken and re-run.
#
#   REFINEMENT=1 CELL=9,9 julia --project=docs column_calibration.jl

include(joinpath(@__DIR__, "column_setup.jl"))

@info "forward run at h = $h₀ m"
forward = forward_column(h₀)
loss(series) = mean((series.θ .- θ_target).^2)
@info @sprintf("run-mean (θ − θᴱᴿᴬ⁵ᴸ)² = %.4e (RMS %.4f);  θ(t_end) = %.4f vs %.4f", loss(forward), sqrt(loss(forward)), forward.θ[end], θ_target[end])

# ## The compiled reverse pass

backend = get(ENV, "ARCH", "cpu")
Reactant.set_default_backend(backend)

function soil_water_loss(model, h, θ₀, T₀, q₀, θ_target, Δt, nsteps)
    initialize_column!(model, h, θ₀, T₀, q₀)
    L = sum(zero(parent(h)))
    @trace mincut=true checkpointing=true track_numbers=false for n in 1:nsteps
        time_step!(model, Δt)
        L += sum((soil_water(model, h) .- θ_target[n, :, :, :]).^2)
    end
    return L / nsteps
end

function grad_soil_water_loss(model, dmodel, h, dh, θ₀, T₀, q₀, θ_target, Δt, nsteps)
    parent(dh) .= 0
    _, L = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                           soil_water_loss, Enzyme.Active,
                           Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(h, dh),
                           Enzyme.Const(θ₀), Enzyme.Const(T₀), Enzyme.Const(q₀), Enzyme.Const(θ_target),
                           Enzyme.Const(Δt), Enzyme.Const(nsteps))
    return dh, L
end

grid_ad = RectilinearGrid(ReactantState(), FT; size = (), topology = (Flat, Flat, Flat))
h_ad = surface_field(grid_ad); parent(h_ad) .= h₀
dh_ad = Enzyme.make_zero(h_ad)
θ_target_ad = Reactant.to_rarray(reshape(θ_target, Nsteps, 1, 1, 1))
model_ad = borneo_coupled_model(grid_ad, FT, column, parameters; slab_depth = surface_field(grid_ad),
                                surface_layer_height, boundary_layer_height, inner_iterations, similarity_iterations)
Oceananigans.initialize!(model_ad)
dmodel = Enzyme.make_zero(model_ad)

@info "compiling the reverse pass over $Nsteps steps..."
compile_seconds = @elapsed compiled = Reactant.@compile raise=true raise_first=true sync=true grad_soil_water_loss(
    model_ad, dmodel, h_ad, dh_ad, θ₀, T₀, q₀, θ_target_ad, Δt, Nsteps)
run_seconds = @elapsed dh_out, L_ad = compiled(model_ad, dmodel, h_ad, dh_ad, θ₀, T₀, q₀, θ_target_ad, Δt, Nsteps)
dL_dh = first(Array(parent(dh_out)))
L_compiled = Reactant.to_number(L_ad)
@info @sprintf("adjoint: L = %.6e (eager %.6e), dL/dh = %.6e m⁻¹  [compile %.0f s, run %.1f s]",
               L_compiled, loss(forward), dL_dh, compile_seconds, run_seconds)

# ## Finite-difference check and one gradient step with a line search

δ = 1e-3 * h₀
fd = (loss(forward_column(h₀ + δ)) - loss(forward_column(h₀ - δ))) / 2δ
@info @sprintf("finite difference dL/dh = %.6e m⁻¹ (adjoint / FD = %.4f)", fd, dL_dh / fd)

# Descend along −dL/dh: the step length is picked by a backtracking line search over
# eager forward runs (a run costs a few seconds), the depth kept within [0.02, 5] m.
direction = -sign(dL_dh)
trial_depths = [clamp(h₀ + direction * s, 0.02, 5.0) for s in h₀ .* (2.0 .^ (2:-1:-4))]
trials = [(depth = d, series = forward_column(d)) for d in trial_depths]
best = argmin([loss(t.series) for t in trials])
h₁, calibrated = trials[best].depth, trials[best].series
@info "line search: " * join([@sprintf("h = %.3f → L = %.3e", t.depth, loss(t.series)) for t in trials], ";  ")
@info @sprintf("gradient step: h %.3f → %.3f m;  L %.4e → %.4e (RMS %.4f → %.4f);  θ(t_end) %.4f → %.4f (target %.4f)",
               h₀, h₁, loss(forward), loss(calibrated), sqrt(loss(forward)), sqrt(loss(calibrated)),
               forward.θ[end], calibrated.θ[end], θ_target[end])

jldsave("column_calibration_r$(refinement)_i$(i)_j$(j).jld2"; cell, λ, φ, h₀, h₁, θ_target, θ_obs, θ_obs_layer_1,
        trial_depths, trial_losses = [loss(t.series) for t in trials],
        forward = Dict(pairs(forward)), calibrated = Dict(pairs(calibrated)), dL_dh, fd, L_compiled,
        compile_seconds, run_seconds, parameters = Dict(pairs(parameters)))

# ## Figure

t = forward.t
t_obs = era5_land.times ./ 3600
fig = Figure(size = (1800, 1100), fontsize = 15)
Label(fig[0, 1:3], @sprintf("Borneo forest column %.2f°E %.2f°N — %s, LAI %.1f, canopy %.0f m, f_veg %.2f;  dL/dh adjoint %.3e vs FD %.3e m⁻¹",
                            λ, φ, replace(static.canopy_class[i, j], "_" => " "), static.leaf_area_index[i, j],
                            static.canopy_height[i, j], static.vegetation_fraction[i, j], dL_dh, fd); fontsize = 18)

ax = Axis(fig[1, 1]; title = "ERA5 forcing", xlabel = "t (h)", ylabel = "rain (mm h⁻¹)")
barplot!(ax, t, forward.rain .* 3600; color = (:steelblue, 0.6), gap = 0)
ax2 = Axis(fig[1, 1]; yaxisposition = :right, ylabel = "shortwave (W m⁻²)")
hidexdecorations!(ax2); hidespines!(ax2)
lines!(ax2, t, forward.sw; color = :orange)

ax = Axis(fig[1, 2]; title = "Soil water: slab vs ERA5-Land", xlabel = "t (h)", ylabel = "θ (m³ m⁻³)")
lines!(ax, t_obs, θ_obs; color = :black, linewidth = 2, label = "ERA5-Land 0–28 cm")
lines!(ax, t_obs, θ_obs_layer_1; color = :gray, linestyle = :dash, label = "ERA5-Land 0–7 cm")
lines!(ax, t, forward.θ; color = :firebrick, linewidth = 2, label = @sprintf("slab, h = %.2f m (RMS %.3f)", h₀, sqrt(loss(forward))))
lines!(ax, t, calibrated.θ; color = :seagreen, linewidth = 2, label = @sprintf("slab, h = %.2f m after one step (RMS %.3f)", h₁, sqrt(loss(calibrated))))
axislegend(ax; position = :lt)

ax = Axis(fig[1, 3]; title = "Temperatures", xlabel = "t (h)", ylabel = "T (K)")
lines!(ax, t, forward.Tᵃ; color = :purple, label = "air (ERA5, lifted)")
lines!(ax, t, forward.T; color = :firebrick, label = "slab Tˡᵃ")
lines!(ax, t, forward.Tˡᵉᵃᶠ; color = :seagreen, label = "leaf")
lines!(ax, t, forward.Tᵍ; color = :sienna, label = "soil skin")
lines!(ax, t, forward.Tᵃᶜ; color = :black, linestyle = :dash, label = "canopy air")
axislegend(ax; position = :lt, nbanks = 2)

ax = Axis(fig[2, 1]; title = "Turbulent heat fluxes (positive upward)", xlabel = "t (h)", ylabel = "W m⁻²")
lines!(ax, t, forward.LE; color = :navy, linewidth = 2, label = "LE")
lines!(ax, t, forward.LEᶜ; color = :seagreen, label = "LE canopy")
lines!(ax, t, forward.LEʷ; color = :teal, linestyle = :dash, label = "LE wet canopy")
lines!(ax, t, forward.LEᵍ; color = :sienna, label = "LE soil")
lines!(ax, t, forward.H; color = :orange, linewidth = 2, label = "H")
axislegend(ax; position = :lt, nbanks = 2)

ax = Axis(fig[2, 2]; title = "Water stores", xlabel = "t (h)", ylabel = "kg m⁻²")
lines!(ax, t, forward.Wᶜ; color = :seagreen, label = "canopy Wᶜ")
lines!(ax, t, forward.Wᵖ; color = :steelblue, label = "pond Wᵖ")
ax2 = Axis(fig[2, 2]; yaxisposition = :right, ylabel = "𝒮")
hidexdecorations!(ax2); hidespines!(ax2)
lines!(ax2, t, forward.𝒮; color = :black, linestyle = :dash)
axislegend(ax; position = :lt)

ax = Axis(fig[2, 3]; title = "Soil water fluxes (mm h⁻¹)", xlabel = "t (h)", ylabel = "mm h⁻¹")
lines!(ax, t, forward.P .* 3600; color = :steelblue, label = "throughfall + pond re-offer")
lines!(ax, t, forward.E .* 3600; color = :navy, label = "evaporation Jᵛ")
lines!(ax, t, forward.R .* 3600; color = :gray, label = "runoff")
lines!(ax, t, .-forward.D .* 3600; color = :sienna, label = "drainage (down)")
axislegend(ax; position = :lt, nbanks = 2)

save("column_calibration_r$(refinement)_i$(i)_j$(j).png", fig)
@info "saved column_calibration_r$(refinement)_i$(i)_j$(j).png"
