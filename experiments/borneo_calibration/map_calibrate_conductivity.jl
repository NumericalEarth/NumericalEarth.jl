# Per-cell calibration of the saturated (matching-point) hydraulic conductivity at fixed
# slab depth h = h₀ = 0.28 m — the depth of the ERA5-Land 0–28 cm blend, so model and
# target θ share their support. The active parameter is q = log₁₀K₀; drainage is ρK(𝒮)
# with K ∝ K₀, so q controls how fast the slab dries between showers,
#
#     L(q) = (1/N) Σₙ Σᵢⱼ wᵢⱼ (θᵢⱼ(tₙ; 10^q) − θᴱᴿᴬ⁵ᴸᵢⱼ(tₙ))²,      θ = Mˡᵃ / (ρˡ h₀).
#
# Same machinery as the slab-depth calibration: one compiled reverse pass re-called every
# iteration, two-sided per-cell line search over eager runs, q within [−1, +4] decades
# of the pedotransfer value.
#
#   REFINEMENT=1 ARCH=gpu NSTEPS=900 NITER=10 julia --project=docs map_calibrate_conductivity.jl

include(joinpath(@__DIR__, "map_setup.jl"))
using Oceananigans.Architectures: ReactantState
using Statistics: std
using Reactant
using Enzyme
using Reactant: @trace
using CairoMakie
using Printf

Niter = parse(Int, get(ENV, "NITER", "10"))
tag = "map_logK_r$(refinement)_$(backend)$(tag_suffix)"

q_pedotransfer = log10.(max.(static.matching_point_conductivity, 1e-9))
q_low, q_high = q_pedotransfer .- 1, q_pedotransfer .+ 4
q_fill = median(q_pedotransfer[weight .> 0])

conductivity_field(model) = model.land.hydrology.soil.soil.hydraulic_conductivity.matching_point_conductivity

# ## The compiled objective, re-callable: clocks, state and K₀ set inside the trace

reset_clock!(clock) = (clock.time = zero(clock.time); clock.iteration = zero(clock.iteration); nothing)

function conductivity_loss(model, k, h, θ₀, T₀, q₀, w, θᵗ, Δt, nsteps)
    reset_clock!(model.clock)
    reset_clock!(model.land.clock)
    # exp(ln10 · q), not exp10: Reactant has no traced exp10
    parent(conductivity_field(model)) .= exp.(log(10) .* parent(k))
    initialize_map!(model, h, θ₀, T₀, q₀)
    L = sum(zero(parent(h)))
    @trace mincut=true checkpointing=true track_numbers=false for n in 1:nsteps
        time_step!(model, Δt)
        L += sum(cell_loss(model, h, w, θᵗ[n, :, :, :]))
    end
    return L / nsteps
end

function grad_conductivity_loss(model, dmodel, k, dk, h, θ₀, T₀, q₀, w, θᵗ, Δt, nsteps)
    parent(dk) .= 0
    _, L = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                           conductivity_loss, Enzyme.Active,
                           Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(k, dk),
                           Enzyme.Const(h), Enzyme.Const(θ₀), Enzyme.Const(T₀), Enzyme.Const(q₀),
                           Enzyme.Const(w), Enzyme.Const(θᵗ), Enzyme.Const(Δt), Enzyme.Const(nsteps))
    return dk, L
end

Reactant.set_default_backend(backend)
grid_ad = land_grid(ReactantState(), FT)
fields_ad = map_fields(grid_ad, fill(h₀, Nx, Ny))
k_ad = surface_property(grid_ad, q_pedotransfer)
dk_ad = Enzyme.make_zero(k_ad)
θ_target_ad = Reactant.to_rarray(θ_target)
s_ad = surface_parameters(static, grid_ad, FT)
model_ad = borneo_coupled_model(grid_ad, FT, forcing, s_ad; slab_depth = surface_field(grid_ad),
                                exchanger_correction = correction, surface_layer_height, boundary_layer_height,
                                inner_iterations, similarity_iterations, hydrology_options(grid_ad)...)
Oceananigans.initialize!(model_ad)

dmodel₀ = Enzyme.make_zero(model_ad)
@info "compiling the reverse pass over $Nsteps steps on the $backend backend..."
compile_seconds = @elapsed compiled = Reactant.@compile raise=true raise_first=true sync=true grad_conductivity_loss(
    model_ad, dmodel₀, k_ad, dk_ad, fields_ad.h, fields_ad.θ₀, fields_ad.T₀, fields_ad.q₀,
    fields_ad.w, θ_target_ad, Δt, Nsteps)
@info @sprintf("compiled in %.0f s", compile_seconds)

# ## Descent: adjoint direction, two-sided per-cell line search in decades of K₀

cpu_scratch = surface_field(land_grid(CPU(), FT))
function set_cells!(field, values, fill_value)
    set!(cpu_scratch, values)
    parent(cpu_scratch) .= ifelse.(parent(cpu_scratch) .== 0, fill_value, parent(cpu_scratch))
    parent(field) .= parent(cpu_scratch)
    return field
end
with_conductivity(qv) = model -> (set_cells!(conductivity_field(model), exp10.(qv), exp10(q_fill)); nothing)

trial_steps = [1.0, 0.5, 0.25, 0.125, 0.0625]
q = copy(q_pedotransfer)
history = []
gradient_seconds = Float64[]

for iteration in 1:Niter
    set_cells!(k_ad, q, q_fill)
    dmodel = Enzyme.make_zero(model_ad)   # fresh shadow: a stale one seeds the adjoint
    t = @elapsed dk_out, L_ad = compiled(model_ad, dmodel, k_ad, dk_ad, fields_ad.h, fields_ad.θ₀,
                                          fields_ad.T₀, fields_ad.q₀, fields_ad.w, θ_target_ad, Δt, Nsteps)
    push!(gradient_seconds, t)
    g = Array(interior(dk_out, :, :, 1))
    L = Reactant.to_number(L_ad)

    direction = -sign.(g)
    offsets = vcat([direction .* s for s in trial_steps], [-direction .* s for s in trial_steps], [zero(direction)])
    trial_losses = [forward_map(fill(h₀, Nx, Ny); modify! = with_conductivity(clamp.(q .+ o, q_low, q_high))).losses
                    for o in offsets]
    best = [argmin([trial_losses[m][i, j] for m in eachindex(offsets)]) for i in 1:Nx, j in 1:Ny]
    new_q = [land[i, j] ? clamp(q[i, j] + offsets[best[i, j]][i, j], q_low[i, j], q_high[i, j]) : q[i, j]
             for i in 1:Nx, j in 1:Ny]
    moved = count((new_q .!= q)[land])
    push!(history, (; iteration, L, q = copy(q), gradient = g))
    @info @sprintf("iteration %d: L = %.5e (RMS %.4f), |∂L/∂q| ∈ [%.1e, %.1e], %d of %d cells moved, gradient in %.0f s",
                   iteration, L, sqrt(L / count(land)), extrema(abs.(g[land]))..., moved, count(land), t)
    q .= new_q
    moved == 0 && break
end

final = forward_map(fill(h₀, Nx, Ny); record = true, modify! = with_conductivity(q))
initial = forward_map(fill(h₀, Nx, Ny); record = true)

# ## Tracking metrics: does the calibrated slab follow the ERA5-Land series in time?

Nh = Nsteps ÷ round(Int, 3600 / Δt) + 1
θ_obs = zeros(Nh, Nx, Ny)
for m in 1:Nh
    θ_obs[m, :, :] .= era5_land_soil_water(era5_land, m)
end
function tracking(result)
    r = Float64[]; σratio = Float64[]
    for c in findall(land)
        mo, ob = result.snapshots.θ[:, c], θ_obs[:, c]
        push!(σratio, std(mo) / std(ob))
        std(mo) > 0 && push!(r, cor(mo, ob))
    end
    return (; median_r = median(r), median_σratio = median(σratio))
end
t0, t1 = tracking(initial), tracking(final)
@info @sprintf("calibration: L %.5e → %.5e;  run-RMS %.4f → %.4f m³ m⁻³;  median r %.3f → %.3f;  median σ ratio %.2f → %.2f;  Δq ∈ [%.2f, %.2f] decades, median %.2f",
               sum(initial.losses), sum(final.losses), rms(initial.losses), rms(final.losses),
               t0.median_r, t1.median_r, t0.median_σratio, t1.median_σratio,
               extrema((q .- q_pedotransfer)[land])..., median((q .- q_pedotransfer)[land]))

jldsave("$(tag).jld2"; h₀, q, q_pedotransfer, θ_target_end, weight,
        history = [(h.iteration, h.L, h.q, h.gradient) for h in history],
        gradient_seconds, compile_seconds,
        initial_losses = initial.losses, final_losses = final.losses,
        θ_end_initial = initial.θ_end, θ_end_final = final.θ_end,
        snapshots = Dict(pairs(final.snapshots)), snapshots_initial = Dict(pairs(initial.snapshots)),
        θ_obs, longitude = static.longitude, latitude = static.latitude)

# ## Figure

λ, φ = static.longitude, static.latitude
mask(a) = ifelse.(land, a, NaN)
fig = Figure(size = (1900, 1250), fontsize = 15)
Label(fig[0, 1:6], @sprintf("Conductivity calibration at h = %.2f m against ERA5-Land, Central Borneo (%s backend): %d adjoint iterations, RMS %.4f → %.4f m³ m⁻³, median r %.2f → %.2f",
                            h₀, backend, length(history), rms(initial.losses), rms(final.losses), t0.median_r, t1.median_r); fontsize = 18)

ax = Axis(fig[1, 1:2]; title = "loss over the iterations", xlabel = "iteration", ylabel = "L", yscale = log10)
scatterlines!(ax, [h.iteration for h in history], [h.L for h in history]; color = :firebrick)

function panel!(pos, data, title, label; colormap = :viridis, colorrange = nothing)
    ax = Axis(fig[pos...]; title, aspect = DataAspect(), xlabel = "longitude", ylabel = "latitude")
    hm = isnothing(colorrange) ? heatmap!(ax, λ, φ, data; colormap) : heatmap!(ax, λ, φ, data; colormap, colorrange)
    Colorbar(fig[pos[1], pos[2] + 1], hm; label)
    return ax
end

panel!((1, 3), mask(q .- q_pedotransfer), "calibrated Δlog₁₀K₀", "decades"; colormap = :viridis)
θmed(result) = [median(result.snapshots.θ[m, :, :][land]) for m in 1:Nh]
ax2 = Axis(fig[1, 5:6]; title = "domain-median θ(t)", xlabel = "hour", ylabel = "θ (m³ m⁻³)")
lines!(ax2, 0:Nh-1, [median(θ_obs[m, :, :][land]) for m in 1:Nh]; color = :black, label = "ERA5-Land")
lines!(ax2, 0:Nh-1, θmed(initial); color = :steelblue, label = "pedotransfer K₀")
lines!(ax2, 0:Nh-1, θmed(final); color = :firebrick, label = "calibrated K₀")
axislegend(ax2; position = :rb)

θlim = extrema(filter(isfinite, [mask(θ_target_end); mask(initial.θ_end); mask(final.θ_end)]))
panel!((2, 1), mask(θ_target_end), "ERA5-Land θ (0–28 cm) at t_end", "m³ m⁻³"; colormap = :tempo, colorrange = θlim)
panel!((2, 3), mask(initial.θ_end), "slab θ at t_end, pedotransfer K₀", "m³ m⁻³"; colormap = :tempo, colorrange = θlim)
panel!((2, 5), mask(final.θ_end), "slab θ at t_end, calibrated K₀", "m³ m⁻³"; colormap = :tempo, colorrange = θlim)

rlim = (0, maximum(sqrt.(initial.losses[land])))
panel!((3, 1), mask(sqrt.(initial.losses)), "run-RMS mismatch, pedotransfer", "m³ m⁻³"; colormap = :amp, colorrange = rlim)
panel!((3, 3), mask(sqrt.(final.losses)), "run-RMS mismatch, calibrated", "m³ m⁻³"; colormap = :amp, colorrange = rlim)
improvement = sqrt.(initial.losses) .- sqrt.(final.losses)
ilim = maximum(abs, filter(isfinite, mask(improvement)))
panel!((3, 5), mask(improvement), "RMS improvement", "m³ m⁻³"; colormap = :balance, colorrange = (-ilim, ilim))

save("$(tag).png", fig)
@info "saved $(tag).png"
