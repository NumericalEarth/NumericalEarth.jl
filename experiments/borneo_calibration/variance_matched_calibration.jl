# Joint (h, log₁₀K₀) calibration with a loss that pays for amplitude: the run-mean squared
# mismatch plus a per-cell penalty on the temporal standard deviation of θ against the
# observed one,
#
#     L = (1/N) Σₙ Σᵢⱼ wᵢⱼ (θᵢⱼ(tₙ) − θᵢⱼᴱᴿᴬ⁵ᴸ(tₙ))²  +  λ Σᵢⱼ wᵢⱼ (σᵢⱼ[θ] − σᵢⱼ[θᴱᴿᴬ⁵ᴸ])².
#
# The squared-error term alone is minimized by damped predictions (σ = r·σᵒ); the second
# term removes that incentive, which is what a depth-only amplitude knob otherwise exploits.
# Same compiled-adjoint machinery as the joint calibration: the moments Σθ and Σθ² are
# accumulated inside the traced loop and turned into σ after it.
#
#   REFINEMENT=1 ARCH=gpu NSTEPS=1764 END_DATE=2020-04-22 VARIANCE_WEIGHT=5 NITER=8 julia --project=docs variance_matched_calibration.jl

include(joinpath(@__DIR__, "map_setup.jl"))
using Oceananigans.Architectures: ReactantState
using Reactant
using Enzyme
using Reactant: @trace

Niter = parse(Int, get(ENV, "NITER", "8"))
λ = parse(FT, get(ENV, "VARIANCE_WEIGHT", "5"))
tag = "map_variance_matched_r$(refinement)_$(backend)"

q_pedotransfer = log10.(max.(static.matching_point_conductivity, 1e-9))
q_low, q_high = q_pedotransfer .- 1, q_pedotransfer .+ 4
q_fill = median(q_pedotransfer[land])
q_start = isfile("map_logK_r$(refinement)_gpu.jld2") ? jldopen(f -> f["q"], "map_logK_r$(refinement)_gpu.jld2") : copy(q_pedotransfer)

# The observed temporal standard deviation per cell over the fitting window, in the parent layout.
σ_obs = dropdims(std(θ_target; dims = 1, corrected = false); dims = 1)

# ## The compiled objective

reset_clock!(clock) = (clock.time = zero(clock.time); clock.iteration = zero(clock.iteration); nothing)

function variance_matched_loss(model, h, k, θ₀, T₀, q₀, w, θᵗ, σᵒ, λ, Δt, nsteps)
    reset_clock!(model.clock)
    reset_clock!(model.land.clock)
    parent(conductivity_field(model)) .= exp.(log(10) .* parent(k))
    initialize_map!(model, h, θ₀, T₀, q₀)
    L = sum(zero(parent(h)))
    Σθ = zero(parent(h))
    Σθ² = zero(parent(h))
    @trace mincut=true checkpointing=true track_numbers=false for n in 1:nsteps
        time_step!(model, Δt)
        θ = soil_water(model, h)
        L += sum(parent(w) .* (θ .- θᵗ[n, :, :, :]) .^ 2)
        Σθ = Σθ .+ θ
        Σθ² = Σθ² .+ θ .^ 2
    end
    μ = Σθ ./ nsteps
    σ = sqrt.(max.(Σθ² ./ nsteps .- μ .^ 2, 0) .+ 1e-16)
    return L / nsteps + λ * sum(parent(w) .* (σ .- σᵒ) .^ 2)
end

function grad_variance_matched_loss(model, dmodel, h, dh, k, dk, θ₀, T₀, q₀, w, θᵗ, σᵒ, λ, Δt, nsteps)
    parent(dh) .= 0
    parent(dk) .= 0
    _, L = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                           variance_matched_loss, Enzyme.Active,
                           Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(h, dh), Enzyme.Duplicated(k, dk),
                           Enzyme.Const(θ₀), Enzyme.Const(T₀), Enzyme.Const(q₀), Enzyme.Const(w), Enzyme.Const(θᵗ),
                           Enzyme.Const(σᵒ), Enzyme.Const(λ), Enzyme.Const(Δt), Enzyme.Const(nsteps))
    return dh, dk, L
end

Reactant.set_default_backend(backend)
grid_ad = land_grid(ReactantState(), FT)
fields_ad = map_fields(grid_ad, fill(h₀, Nx, Ny))
dh_ad = Enzyme.make_zero(fields_ad.h)
k_ad = surface_property(grid_ad, q_start)
dk_ad = Enzyme.make_zero(k_ad)
θ_target_ad = Reactant.to_rarray(θ_target)
σ_obs_ad = Reactant.to_rarray(σ_obs)
s_ad = surface_parameters(static, grid_ad, FT)
model_ad = borneo_coupled_model(grid_ad, FT, forcing, s_ad; slab_depth = surface_field(grid_ad),
                                exchanger_correction = correction, surface_layer_height, boundary_layer_height,
                                inner_iterations, similarity_iterations)
Oceananigans.initialize!(model_ad)

@info "compiling the reverse pass over $Nsteps steps on the $backend backend..."
compile_seconds = @elapsed compiled = Reactant.@compile raise=true raise_first=true sync=true grad_variance_matched_loss(
    model_ad, Enzyme.make_zero(model_ad), fields_ad.h, dh_ad, k_ad, dk_ad, fields_ad.θ₀, fields_ad.T₀, fields_ad.q₀,
    fields_ad.w, θ_target_ad, σ_obs_ad, λ, Δt, Nsteps)
@info @sprintf("compiled in %.0f s", compile_seconds)

# ## Descent: one adjoint per iteration, per-cell search over an (h factor) × (Δq) grid on the same objective

σ_obs_interior = σ_obs[1 + Hx:Nx + Hx, 1 + Hy:Ny + Hy, 1]
objective(result) = result.losses .+ λ .* (sqrt.(max.(result.θ_variance, 0)) .- σ_obs_interior) .^ 2

h_factors = [4.0, 2.0, sqrt(2), 1.0, 1/sqrt(2), 0.5, 0.25]
q_offsets = [0.5, 0.25, 0.0, -0.25, -0.5]
depths = fill(FT(h₀), Nx, Ny)
q = copy(q_start)
history = []

for iteration in 1:Niter
    set_cells!(fields_ad.h, depths, h₀)
    set_cells!(k_ad, q, q_fill)
    dmodel = Enzyme.make_zero(model_ad)
    t = @elapsed dh_out, dk_out, L_ad = compiled(model_ad, dmodel, fields_ad.h, dh_ad, k_ad, dk_ad,
                                                  fields_ad.θ₀, fields_ad.T₀, fields_ad.q₀,
                                                  fields_ad.w, θ_target_ad, σ_obs_ad, λ, Δt, Nsteps)
    gh, gq = Array(interior(dh_out, :, :, 1)), Array(interior(dk_out, :, :, 1))
    L = Reactant.to_number(L_ad)

    trials = [(f, o) for f in h_factors, o in q_offsets]
    trial_objectives = [objective(forward_map(clamp.(depths .* f, 0.02, 5.0); modify! = with_conductivity(clamp.(q .+ o, q_low, q_high))))
                        for (f, o) in trials]
    best = [argmin([trial_objectives[m][i, j] for m in eachindex(trials)]) for i in 1:Nx, j in 1:Ny]
    new_depths = [land[i, j] ? clamp(depths[i, j] * trials[best[i, j]][1], 0.02, 5.0) : depths[i, j] for i in 1:Nx, j in 1:Ny]
    new_q = [land[i, j] ? clamp(q[i, j] + trials[best[i, j]][2], q_low[i, j], q_high[i, j]) : q[i, j] for i in 1:Nx, j in 1:Ny]
    moved = count(((new_depths .!= depths) .| (new_q .!= q))[land])
    push!(history, (; iteration, L, depths = copy(depths), q = copy(q), gh, gq))
    @info @sprintf("iteration %d: L = %.5e, |∂L/∂h| ≤ %.1e, |∂L/∂q| ≤ %.1e, %d of %d cells moved, gradient in %.0f s",
                   iteration, L, maximum(abs.(gh[land])), maximum(abs.(gq[land])), moved, count(land), t)
    depths .= new_depths
    q .= new_q
    moved == 0 && break
end

final = forward_map(depths; record = true, modify! = with_conductivity(q))
initial = forward_map(fill(h₀, Nx, Ny); record = true)
θ_obs = hourly_observations()
fit = 1:(Nsteps ÷ round(Int, 3600 / Δt) + 1)
s0, s1 = window_scores(initial.snapshots.θ, θ_obs, fit), window_scores(final.snapshots.θ, θ_obs, fit)
@info @sprintf("calibration (λ = %g): run-RMS %.4f → %.4f;  median r %.3f → %.3f;  median σ ratio %.2f → %.2f;  h ∈ [%.2f, %.2f] m (median %.2f), Δq median %.2f decades",
               λ, s0.rms, s1.rms, s0.r, s1.r, s0.σ, s1.σ, extrema(depths[land])..., median(depths[land]), median((q .- q_pedotransfer)[land]))

jldsave("$(tag).jld2"; h₀, λ, depths, q, q_pedotransfer, weight,
        history = [(h.iteration, h.L, h.depths, h.q, h.gh, h.gq) for h in history], compile_seconds,
        initial_losses = initial.losses, final_losses = final.losses,
        snapshots = Dict(pairs(final.snapshots)), snapshots_initial = Dict(pairs(initial.snapshots)),
        θ_obs, longitude = static.longitude, latitude = static.latitude)
@info "saved $(tag).jld2"
