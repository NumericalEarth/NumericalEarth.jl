# Iterative calibration of the per-cell slab depth on the Central Borneo box: gradient
# descent on the run-mean soil-water mismatch against ERA5-Land,
#
#     L(h) = (1/N) Σₙ Σᵢⱼ wᵢⱼ (θᵢⱼ(tₙ; h) − θᴱᴿᴬ⁵ᴸᵢⱼ(tₙ))²,        θ = Mˡᵃ / (ρˡ h),
#
# with one compiled reverse pass per iteration (the objective resets the clocks and the
# state, so the same compiled function serves every iterate) and a per-cell line search
# along −∂L/∂h over eager forward runs. Depths stay within [0.02, 5] m.
#
#   REFINEMENT=1 ARCH=gpu NSTEPS=900 NITER=8 julia --project=docs map_calibration_iterations.jl

include(joinpath(@__DIR__, "map_setup.jl"))
using Oceananigans.Architectures: ReactantState
using Reactant
using Enzyme
using Reactant: @trace
using CairoMakie
using Printf

Niter = parse(Int, get(ENV, "NITER", "8"))
two_sided = get(ENV, "LINE_SEARCH", "adjoint_sign") == "two_sided"
tag = "map_iterations_r$(refinement)_$(backend)" * (two_sided ? "_two_sided" : "")

# ## The compiled objective: clock and state reset inside, so it is re-callable

reset_clock!(clock) = (clock.time = zero(clock.time); clock.iteration = zero(clock.iteration); nothing)

function soil_water_loss(model, h, θ₀, T₀, q₀, w, θᵗ, Δt, nsteps)
    reset_clock!(model.clock)
    reset_clock!(model.land.clock)
    initialize_map!(model, h, θ₀, T₀, q₀)
    L = sum(zero(parent(h)))
    @trace mincut=true checkpointing=true track_numbers=false for n in 1:nsteps
        time_step!(model, Δt)
        L += sum(cell_loss(model, h, w, θᵗ[n, :, :, :]))
    end
    return L / nsteps
end

function grad_soil_water_loss(model, dmodel, h, dh, θ₀, T₀, q₀, w, θᵗ, Δt, nsteps)
    parent(dh) .= 0
    _, L = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                           soil_water_loss, Enzyme.Active,
                           Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(h, dh),
                           Enzyme.Const(θ₀), Enzyme.Const(T₀), Enzyme.Const(q₀), Enzyme.Const(w), Enzyme.Const(θᵗ),
                           Enzyme.Const(Δt), Enzyme.Const(nsteps))
    return dh, L
end

Reactant.set_default_backend(backend)
grid_ad = land_grid(ReactantState(), FT)
fields_ad = map_fields(grid_ad, fill(h₀, Nx, Ny))
dh_ad = Enzyme.make_zero(fields_ad.h)
θ_target_ad = Reactant.to_rarray(θ_target)
s_ad = surface_parameters(static, grid_ad, FT)
model_ad = borneo_coupled_model(grid_ad, FT, forcing, s_ad; slab_depth = surface_field(grid_ad),
                                exchanger_correction = correction, surface_layer_height, boundary_layer_height,
                                inner_iterations, similarity_iterations)
Oceananigans.initialize!(model_ad)

dmodel₀ = Enzyme.make_zero(model_ad)
@info "compiling the reverse pass over $Nsteps steps on the $backend backend..."
compile_seconds = @elapsed compiled = Reactant.@compile raise=true raise_first=true sync=true grad_soil_water_loss(
    model_ad, dmodel₀, fields_ad.h, dh_ad, fields_ad.θ₀, fields_ad.T₀, fields_ad.q₀,
    fields_ad.w, θ_target_ad, Δt, Nsteps)
@info @sprintf("compiled in %.0f s", compile_seconds)

# ## Descent: adjoint direction, per-cell backtracking line search on eager runs

cpu_field = surface_field(land_grid(CPU(), FT))
function set_depths!(field, depths)
    set!(cpu_field, depths)
    parent(cpu_field) .= ifelse.(parent(cpu_field) .== 0, h₀, parent(cpu_field))
    parent(field) .= parent(cpu_field)
    return field
end

trial_scales = [2.0 .^ (2:-1:-4); 0.0]
depths = fill(FT(h₀), Nx, Ny)
history = []
gradient_seconds = Float64[]

for iteration in 1:Niter
    set_depths!(fields_ad.h, depths)
    dmodel = Enzyme.make_zero(model_ad)   # fresh shadow: a stale one seeds the adjoint
    t = @elapsed dh_out, L_ad = compiled(model_ad, dmodel, fields_ad.h, dh_ad, fields_ad.θ₀, fields_ad.T₀,
                                          fields_ad.q₀, fields_ad.w, θ_target_ad, Δt, Nsteps)
    push!(gradient_seconds, t)
    g = Array(interior(dh_out, :, :, 1))
    L = Reactant.to_number(L_ad)

    # The adjoint gives the local direction; a two-sided search also tests the opposite branch,
    # which rescues cells whose per-cell loss is bimodal in h (thin slabs track drying dips,
    # deep slabs hold the initial value — the local gradient can point into the worse basin).
    direction = -sign.(g)
    factors = two_sided ? vcat([(1 .+ direction .* s) for s in trial_scales], [(1 .- direction .* s) for s in trial_scales[1:end-1]]) :
                          [(1 .+ direction .* s) for s in trial_scales]
    trial_losses = [forward_map(clamp.(depths .* f, 0.02, 5.0)).losses for f in factors]
    best = [argmin([trial_losses[k][i, j] for k in eachindex(factors)]) for i in 1:Nx, j in 1:Ny]
    new_depths = [land[i, j] ? clamp(depths[i, j] * factors[best[i, j]][i, j], 0.02, 5.0) : depths[i, j]
                  for i in 1:Nx, j in 1:Ny]
    moved = count((new_depths .!= depths)[land])
    push!(history, (; iteration, L, depths = copy(depths), gradient = g))
    @info @sprintf("iteration %d: L = %.5e (RMS %.4f), |∂L/∂h| ∈ [%.1e, %.1e], %d of %d cells moved, gradient in %.0f s",
                   iteration, L, sqrt(L / count(land)), extrema(abs.(g[land]))..., moved, count(land), t)
    depths .= new_depths
    moved == 0 && break
end

final = forward_map(depths; record = true)
initial = forward_map(fill(h₀, Nx, Ny))
@info @sprintf("calibration: L %.5e → %.5e;  run-RMS mismatch %.4f → %.4f m³ m⁻³;  h ∈ [%.2f, %.2f] m, median %.2f",
               sum(initial.losses), sum(final.losses), rms(initial.losses), rms(final.losses),
               extrema(depths[land])..., median(depths[land]))

jldsave("$(tag).jld2"; h₀, depths, θ_target_end, weight,
        history = [(h.iteration, h.L, h.depths, h.gradient) for h in history],
        gradient_seconds, compile_seconds,
        initial_losses = initial.losses, final_losses = final.losses,
        θ_end_initial = initial.θ_end, θ_end_final = final.θ_end,
        snapshots = Dict(pairs(final.snapshots)),
        longitude = static.longitude, latitude = static.latitude)

# ## Figure

λ, φ = static.longitude, static.latitude
mask(a) = ifelse.(land, a, NaN)
fig = Figure(size = (1900, 1250), fontsize = 15)
Label(fig[0, 1:6], @sprintf("Iterative slab-depth calibration against ERA5-Land, Central Borneo at ≈ %d km (%s backend): %d adjoint iterations, RMS %.4f → %.4f m³ m⁻³",
                            resolution_km, backend, length(history), rms(initial.losses), rms(final.losses)); fontsize = 18)

ax = Axis(fig[1, 1:2]; title = "loss over the iterations", xlabel = "iteration", ylabel = "L", yscale = log10)
scatterlines!(ax, [h.iteration for h in history], [h.L for h in history]; color = :firebrick)

function panel!(pos, data, title, label; colormap = :viridis, colorrange = nothing)
    ax = Axis(fig[pos...]; title, aspect = DataAspect(), xlabel = "longitude", ylabel = "latitude")
    hm = isnothing(colorrange) ? heatmap!(ax, λ, φ, data; colormap) : heatmap!(ax, λ, φ, data; colormap, colorrange)
    Colorbar(fig[pos[1], pos[2] + 1], hm; label)
    return ax
end

panel!((1, 3), mask(depths), "calibrated slab depth", "m"; colormap = :viridis)
slim = maximum(abs, filter(isfinite, mask(history[1].gradient)))
panel!((1, 5), mask(history[1].gradient), "∂L/∂h at h₀ (first adjoint)", "m⁻¹"; colormap = :balance, colorrange = (-slim, slim))

θlim = extrema(filter(isfinite, [mask(θ_target_end); mask(initial.θ_end); mask(final.θ_end)]))
panel!((2, 1), mask(θ_target_end), "ERA5-Land θ (0–28 cm) at t_end", "m³ m⁻³"; colormap = :tempo, colorrange = θlim)
panel!((2, 3), mask(initial.θ_end), @sprintf("slab θ at t_end, h = %.2f m", h₀), "m³ m⁻³"; colormap = :tempo, colorrange = θlim)
panel!((2, 5), mask(final.θ_end), "slab θ at t_end, calibrated", "m³ m⁻³"; colormap = :tempo, colorrange = θlim)

rlim = (0, maximum(sqrt.(initial.losses[land])))
panel!((3, 1), mask(sqrt.(initial.losses)), "run-RMS mismatch, h₀", "m³ m⁻³"; colormap = :amp, colorrange = rlim)
panel!((3, 3), mask(sqrt.(final.losses)), "run-RMS mismatch, calibrated", "m³ m⁻³"; colormap = :amp, colorrange = rlim)
improvement = sqrt.(initial.losses) .- sqrt.(final.losses)
ilim = maximum(abs, filter(isfinite, mask(improvement)))
panel!((3, 5), mask(improvement), "RMS improvement", "m³ m⁻³"; colormap = :balance, colorrange = (-ilim, ilim))

save("$(tag).png", fig)
@info "saved $(tag).png"
