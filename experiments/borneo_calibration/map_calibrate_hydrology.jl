# Joint per-cell calibration of the slab hydrology under Darcy exchange, with the depth pinned
# at the observation layer's h₀. Three fields, selected by FIELDS (default "K0,exchange"):
#
#   K0         q = log₁₀K₀, the saturated conductivity (drainage and exchange ∝ K)
#   exchange   λ = log ℓ, the Darcy exchange length to the deep reservoir
#   retention  ν = log(n − 1), the van Genuchten exponent, with α slaved so the retention curve
#              keeps its pedotransfer saturation at ψ★ = 1 m
#   deephead   δ = log ψᵈ, the (time-constant) suction of the deep reservoir, Πᵈ = −ψᵈ, started
#              from the DEEP_HEAD=constant value, or the reanalysis "mean" or "initial" head
#   thickness  ζ = log hᵈ, the thickness of the prognostic deep store (DEEP_STORE=1)
#   deepK0     κ = log₁₀K₀ᵈ, the store's own saturated conductivity (its drainage ∝ Kᵈ)
#   watertable ω = log ℓ₂, the distance from the store to a saturated head below it (DEEP_STORE_DRAINAGE=watertable)
#
#     L = (1/N) Σₙ Σᵢⱼ wᵢⱼ (θᵢⱼ(tₙ; q, λ, ν) − θᵢⱼᴱᴿᴬ⁵ᴸ(tₙ))²,      θ = Mˡᵃ / (ρˡ h₀),
#
# plus, with DEEP_LOSS=1, the same mismatch of the deep store's θᵈ = Mᵈ / (ρˡ hᵈ) to the 28–100 cm layer.
#
# One compiled reverse pass returns all three adjoints. Descent is per cell along the adjoint
# direction, each field scaled by its own step (a diagonal preconditioner) and capped, with a
# backtracking line search over eager runs; inactive fields keep their start values. Bounds
# keep n away from the retention pole.
#
#   REFINEMENT=1 ARCH=gpu NSTEPS=1764 END_DATE=2020-04-22 DEEP_FLUX=darcy EXCHANGE_FIELD=1 \
#   FIELDS=K0,exchange WARM_START=map_logK_r1_gpu_darcy_12d.jld2 TAG_SUFFIX=_darcy_12d \
#   julia --project=docs map_calibrate_hydrology.jl

include(joinpath(@__DIR__, "map_setup.jl"))
using Oceananigans.Architectures: ReactantState
using Reactant
using Enzyme
using Reactant: @trace

exchange_field || error("set EXCHANGE_FIELD=1 so the model carries the exchange length as a Field")
Niter = parse(Int, get(ENV, "NITER", "8"))
active = Symbol.(split(get(ENV, "FIELDS", "K0,exchange"), ","))
constant_deep_head = deep_head_mode in ("constant", "mean", "initial") && !deep_store
:deephead in active && !constant_deep_head && error("calibrating the deep head needs DEEP_HEAD=constant, mean or initial")
(:thickness in active || :deepK0 in active) && !deep_store && error("calibrating the deep store needs DEEP_STORE=1")
deep_loss && !deep_store && error("the deep-layer loss needs DEEP_STORE=1")
:watertable in active && deep_store_drainage != "watertable" && error("calibrating the water-table depth needs DEEP_STORE_DRAINAGE=watertable")
tag = "map_hydrology_" * join(string.(active), "_") * "_r$(refinement)_$(backend)$(tag_suffix)"

# ## Start values, bounds and characteristic steps

q_pedotransfer = log10.(max.(static.matching_point_conductivity, 1e-9))
warm_start = get(ENV, "WARM_START", "")   # a K₀-only or hydrology file; every field it carries starts from it
warm = isfile(warm_start) ? jldopen(f -> Dict(k => f[k] for k in keys(f)), warm_start) : Dict{String, Any}()
q = get(warm, "q", copy(q_pedotransfer))
λ = get(warm, "log_exchange_length", fill(log(exchange_length), Nx, Ny))
ν_pedotransfer = log.(static.pore_size_uniformity .- 1)
ν = get(warm, "log_n_minus_1", copy(ν_pedotransfer))
δ = get(warm, "log_deep_suction", constant_deep_head ? log.(-Array(interior(deep_pressure_head_on(cpu_grid), :, :, 1))) : fill(log(1.0), Nx, Ny))
ζ = get(warm, "log_thickness", fill(log(deep_store_thickness), Nx, Ny))
κ = get(warm, "q_deep", copy(q))
ω = get(warm, "log_water_table", fill(log(water_table_length), Nx, Ny))

bounds = (; K0 = (q_pedotransfer .- 1, q_pedotransfer .+ 4),
            exchange = (fill(log(0.1), Nx, Ny), fill(log(3.0), Nx, Ny)),
            retention = (ν_pedotransfer .- 0.5, ν_pedotransfer .+ 0.7),
            deephead = (fill(log(0.3), Nx, Ny), fill(log(30.0), Nx, Ny)),
            thickness = (fill(log(0.1), Nx, Ny), fill(log(3.0), Nx, Ny)),
            deepK0 = (q_pedotransfer .- 2, q_pedotransfer .+ 4),
            watertable = (fill(log(0.3), Nx, Ny), fill(log(30.0), Nx, Ny)))
steps = (; K0 = 0.25, exchange = 0.25, retention = 0.1, deephead = 0.25, thickness = 0.25, deepK0 = 0.25, watertable = 0.25)
fills = (; K0 = median(q[land]), exchange = log(exchange_length), retention = median(ν[land]), deephead = median(δ[land]),
           thickness = log(deep_store_thickness), deepK0 = median(κ[land]), watertable = log(water_table_length))
water_table = deep_store && deep_store_drainage == "watertable"
𝒮★ = saturation_at_matching_head

# ## The compiled objective: every field set inside the trace

reset_clock!(clock) = (clock.time = zero(clock.time); clock.iteration = zero(clock.iteration); nothing)

function set_retention!(model, ν, 𝒮★)
    n = 1 .+ exp.(parent(ν))
    for f in pore_size_uniformity_fields(model)
        parent(f) .= n
    end
    m = 1 .- 1 ./ n
    parent(air_entry_field(model)) .= (𝒮★ .^ (-1 ./ m) .- 1) .^ (1 ./ n) ./ matching_head
    return nothing
end

function hydrology_loss(model, k, l, ν, d, ζ, κ, ω, 𝒮★, h, θ₀, T₀, q₀, θᵈ₀, w, θᵗ, θᵈᵗ, Δt, nsteps)
    reset_clock!(model.clock)
    reset_clock!(model.land.clock)
    parent(conductivity_field(model)) .= exp.(log(10) .* parent(k))
    parent(exchange_length_field(model)) .= exp.(parent(l))
    constant_deep_head && (parent(deep_pressure_head_field(model)) .= -exp.(parent(d)))
    deep_store && (parent(thickness_field(model)) .= exp.(parent(ζ)))
    deep_store && (parent(deep_conductivity_field(model)) .= exp.(log(10) .* parent(κ)))
    water_table && (parent(water_table_field(model)) .= exp.(parent(ω)))
    set_retention!(model, ν, 𝒮★)
    initialize_map!(model, h, θ₀, T₀, q₀, θᵈ₀)
    L = sum(zero(parent(h)))
    @trace mincut=true checkpointing=true track_numbers=false for n in 1:nsteps
        time_step!(model, Δt)
        L += sum(cell_loss(model, h, w, θᵗ[n, :, :, :]))
        deep_loss && (L += deep_loss_weight * sum(deep_cell_loss(model, w, θᵈᵗ[n, :, :, :])))
    end
    return L / nsteps
end

function grad_hydrology_loss(model, dmodel, k, dk, l, dl, ν, dν, d, dd, ζ, dζ, κ, dκ, ω, dω, 𝒮★, h, θ₀, T₀, q₀, θᵈ₀, w, θᵗ, θᵈᵗ, Δt, nsteps)
    for g in (dk, dl, dν, dd, dζ, dκ, dω)
        parent(g) .= 0
    end
    _, L = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
                           hydrology_loss, Enzyme.Active,
                           Enzyme.Duplicated(model, dmodel), Enzyme.Duplicated(k, dk), Enzyme.Duplicated(l, dl),
                           Enzyme.Duplicated(ν, dν), Enzyme.Duplicated(d, dd), Enzyme.Duplicated(ζ, dζ), Enzyme.Duplicated(κ, dκ),
                           Enzyme.Duplicated(ω, dω), Enzyme.Const(𝒮★), Enzyme.Const(h), Enzyme.Const(θ₀), Enzyme.Const(T₀), Enzyme.Const(q₀), Enzyme.Const(θᵈ₀),
                           Enzyme.Const(w), Enzyme.Const(θᵗ), Enzyme.Const(θᵈᵗ), Enzyme.Const(Δt), Enzyme.Const(nsteps))
    return dk, dl, dν, dd, dζ, dκ, dω, L
end

Reactant.set_default_backend(backend)
grid_ad = land_grid(ReactantState(), FT)
fields_ad = map_fields(grid_ad, fill(h₀, Nx, Ny))
k_ad, l_ad, ν_ad, d_ad, ζ_ad, κ_ad, ω_ad = (surface_property(grid_ad, p) for p in (q, λ, ν, δ, ζ, κ, ω))
dk_ad, dl_ad, dν_ad, dd_ad, dζ_ad, dκ_ad, dω_ad = (Enzyme.make_zero(f) for f in (k_ad, l_ad, ν_ad, d_ad, ζ_ad, κ_ad, ω_ad))
𝒮★_ad = Reactant.to_rarray(parent(set_cells!(surface_field(cpu_grid), 𝒮★, median(𝒮★[land]))))
θ_target_ad = Reactant.to_rarray(θ_target)
θᵈ_target_ad = Reactant.to_rarray(θᵈ_target)
s_ad = surface_parameters(static, grid_ad, FT)
model_ad = borneo_coupled_model(grid_ad, FT, forcing, s_ad; slab_depth = surface_field(grid_ad),
                                exchanger_correction = correction, surface_layer_height, boundary_layer_height,
                                inner_iterations, similarity_iterations, hydrology_options(grid_ad)...)
Oceananigans.initialize!(model_ad)

@info "compiling the reverse pass over $Nsteps steps on the $backend backend..."
compile_seconds = @elapsed compiled = Reactant.@compile raise=true raise_first=true sync=true grad_hydrology_loss(
    model_ad, Enzyme.make_zero(model_ad), k_ad, dk_ad, l_ad, dl_ad, ν_ad, dν_ad, d_ad, dd_ad, ζ_ad, dζ_ad, κ_ad, dκ_ad, ω_ad, dω_ad, 𝒮★_ad, fields_ad.h,
    fields_ad.θ₀, fields_ad.T₀, fields_ad.q₀, fields_ad.θᵈ₀, fields_ad.w, θ_target_ad, θᵈ_target_ad, Δt, Nsteps)
@info @sprintf("compiled in %.0f s", compile_seconds)

# ## Descent: preconditioned adjoint direction, per-cell backtracking over eager runs

parameters = Dict(:K0 => q, :exchange => λ, :retention => ν, :deephead => δ, :thickness => ζ, :deepK0 => κ, :watertable => ω)
calibration(v) = merge(Dict("q" => v[:K0], "log_exchange_length" => v[:exchange], "log_n_minus_1" => v[:retention]),
                       constant_deep_head ? Dict("log_deep_suction" => v[:deephead]) : Dict{String, Any}(),
                       deep_store ? Dict("log_thickness" => v[:thickness], "q_deep" => v[:deepK0]) : Dict{String, Any}(),
                       water_table ? Dict("log_water_table" => v[:watertable]) : Dict{String, Any}())
factors = [1.0, 0.5, 0.25, 0.125, -0.25, 0.0]
history = []

for iteration in 1:Niter
    set_cells!(k_ad, parameters[:K0], fills.K0)
    set_cells!(l_ad, parameters[:exchange], fills.exchange)
    set_cells!(ν_ad, parameters[:retention], fills.retention)
    set_cells!(d_ad, parameters[:deephead], fills.deephead)
    set_cells!(ζ_ad, parameters[:thickness], fills.thickness)
    set_cells!(κ_ad, parameters[:deepK0], fills.deepK0)
    set_cells!(ω_ad, parameters[:watertable], fills.watertable)
    dmodel = Enzyme.make_zero(model_ad)
    t = @elapsed dk_out, dl_out, dν_out, dd_out, dζ_out, dκ_out, dω_out, L_ad = compiled(model_ad, dmodel, k_ad, dk_ad, l_ad, dl_ad, ν_ad, dν_ad, d_ad, dd_ad,
                                                                                          ζ_ad, dζ_ad, κ_ad, dκ_ad, ω_ad, dω_ad, 𝒮★_ad, fields_ad.h, fields_ad.θ₀, fields_ad.T₀, fields_ad.q₀,
                                                                                          fields_ad.θᵈ₀, fields_ad.w, θ_target_ad, θᵈ_target_ad, Δt, Nsteps)
    gradients = Dict(:K0 => Array(interior(dk_out, :, :, 1)), :exchange => Array(interior(dl_out, :, :, 1)),
                     :retention => Array(interior(dν_out, :, :, 1)), :deephead => Array(interior(dd_out, :, :, 1)),
                     :thickness => Array(interior(dζ_out, :, :, 1)), :deepK0 => Array(interior(dκ_out, :, :, 1)),
                     :watertable => Array(interior(dω_out, :, :, 1)))
    L = Reactant.to_number(L_ad)

    # Each active field moves along its adjoint, scaled to its characteristic step and capped at 4 steps.
    directions = Dict(f => clamp.(-gradients[f] ./ median(abs.(gradients[f][land])), -4, 4) .* steps[f] for f in active)
    trial(factor) = Dict(f => (f in active ? clamp.(parameters[f] .+ factor .* directions[f], bounds[f]...) : parameters[f])
                         for f in keys(parameters))
    trials = [trial(factor) for factor in factors]
    trial_losses = [forward_map(fill(h₀, Nx, Ny); modify! = with_calibration(calibration(v))).losses for v in trials]
    best = [argmin([trial_losses[m][i, j] for m in eachindex(factors)]) for i in 1:Nx, j in 1:Ny]
    new_parameters = Dict(f => [land[i, j] ? trials[best[i, j]][f][i, j] : parameters[f][i, j] for i in 1:Nx, j in 1:Ny] for f in keys(parameters))
    moved = count(reduce((a, b) -> a .| b, [new_parameters[f] .!= parameters[f] for f in active])[land])
    push!(history, (; iteration, L, parameters = Dict(f => copy(parameters[f]) for f in keys(parameters)), gradients))
    @info @sprintf("iteration %d: L = %.5e (RMS %.4f), %s, %d of %d cells moved, gradient in %.0f s",
                   iteration, L, sqrt(L / count(land)),
                   join([@sprintf("|∂L/∂%s| ≤ %.1e", f, maximum(abs.(gradients[f][land]))) for f in active], ", "),
                   moved, count(land), t)
    for f in keys(parameters)
        parameters[f] .= new_parameters[f]
    end
    moved == 0 && break
end

final = forward_map(fill(h₀, Nx, Ny); record = true, modify! = with_calibration(calibration(parameters)))
initial = forward_map(fill(h₀, Nx, Ny); record = true)
θ_obs = hourly_observations()
fit = 1:(Nsteps ÷ round(Int, 3600 / Δt) + 1)
s0, s1 = window_scores(initial.snapshots.θ, θ_obs, fit), window_scores(final.snapshots.θ, θ_obs, fit)
@info @sprintf("calibration (%s): run-RMS %.4f → %.4f;  median r %.3f → %.3f;  median σ ratio %.2f → %.2f;  Δq median %.2f;  ℓ ∈ [%.2f, %.2f] m (median %.2f);  n ∈ [%.3f, %.3f] (median %.3f);  deep suction ∈ [%.2f, %.2f] m (median %.2f);  store thickness ∈ [%.2f, %.2f] m (median %.2f);  Δq_deep median %.2f;  water table ∈ [%.2f, %.2f] m below the store (median %.2f)",
               join(string.(active), ", "), s0.rms, s1.rms, s0.r, s1.r, s0.σ, s1.σ, median((parameters[:K0] .- q_pedotransfer)[land]),
               extrema(exp.(parameters[:exchange][land]))..., median(exp.(parameters[:exchange][land])),
               extrema(1 .+ exp.(parameters[:retention][land]))..., median(1 .+ exp.(parameters[:retention][land])),
               extrema(exp.(parameters[:deephead][land]))..., median(exp.(parameters[:deephead][land])),
               extrema(exp.(parameters[:thickness][land]))..., median(exp.(parameters[:thickness][land])),
               median((parameters[:deepK0] .- q_pedotransfer)[land]),
               extrema(exp.(parameters[:watertable][land]))..., median(exp.(parameters[:watertable][land])))

jldsave("$(tag).jld2"; h₀, q = parameters[:K0], log_exchange_length = parameters[:exchange], log_n_minus_1 = parameters[:retention],
        (constant_deep_head ? (; log_deep_suction = parameters[:deephead]) : (;))...,
        (deep_store ? (; log_thickness = parameters[:thickness], q_deep = parameters[:deepK0]) : (;))...,
        (water_table ? (; log_water_table = parameters[:watertable]) : (;))...,
        active = string.(active), q_pedotransfer, ν_pedotransfer, weight, deep_head_mode,
        history = [(h.iteration, h.L, h.parameters, h.gradients) for h in history], compile_seconds,
        initial_losses = initial.losses, final_losses = final.losses,
        snapshots = Dict(pairs(final.snapshots)), snapshots_initial = Dict(pairs(initial.snapshots)),
        θ_obs, longitude = static.longitude, latitude = static.latitude)

# ## Figure

λ°, φ° = static.longitude, static.latitude
mask(a) = ifelse.(land, a, NaN)
fig = Figure(size = (1900, 1250), fontsize = 15)
Label(fig[0, 1:6], @sprintf("Hydrology calibration under Darcy exchange (%s), Central Borneo: %d adjoint iterations, RMS %.4f → %.4f m³ m⁻³, median r %.2f → %.2f, σ ratio %.1f → %.1f",
                            join(string.(active), " + "), length(history), s0.rms, s1.rms, s0.r, s1.r, s0.σ, s1.σ); fontsize = 18)
ax = Axis(fig[1, 1:2]; title = "loss over the iterations", xlabel = "iteration", ylabel = "L", yscale = log10)
scatterlines!(ax, [h.iteration for h in history], [h.L for h in history]; color = :firebrick)
function panel!(pos, data, title, label; colormap = :viridis, colorrange = nothing)
    axp = Axis(fig[pos...]; title, aspect = DataAspect(), xlabel = "longitude", ylabel = "latitude")
    hm = isnothing(colorrange) ? heatmap!(axp, λ°, φ°, data; colormap) : heatmap!(axp, λ°, φ°, data; colormap, colorrange)
    Colorbar(fig[pos[1], pos[2] + 1], hm; label)
    return axp
end
panel!((1, 3), mask(parameters[:K0] .- q_pedotransfer), "calibrated Δlog₁₀K₀", "decades")
Nh = length(fit)
med(x) = [median(x[m, :, :][land]) for m in 1:Nh]
ax2 = Axis(fig[1, 5:6]; title = "domain-median θ(t)", xlabel = "hour", ylabel = "θ (m³ m⁻³)")
lines!(ax2, 0:Nh-1, med(θ_obs); color = :black, label = "ERA5-Land")
lines!(ax2, 0:Nh-1, med(initial.snapshots.θ); color = :steelblue, label = "start")
lines!(ax2, 0:Nh-1, med(final.snapshots.θ); color = :firebrick, label = "calibrated")
axislegend(ax2; position = :rb)
panel!((2, 1), mask(exp.(parameters[:exchange])), "calibrated exchange length ℓ", "m")
panel!((2, 3), mask(1 .+ exp.(parameters[:retention])), "calibrated retention exponent n", "–")
panel!((2, 5), mask(exp.(parameters[:deephead])), "deep suction ψᵈ (constant head)", "m")
rlim = (0, maximum(sqrt.(initial.losses[land])))
panel!((3, 1), mask(sqrt.(initial.losses)), "run-RMS mismatch, start", "m³ m⁻³"; colormap = :amp, colorrange = rlim)
panel!((3, 3), mask(sqrt.(final.losses)), "run-RMS mismatch, calibrated", "m³ m⁻³"; colormap = :amp, colorrange = rlim)
improvement = sqrt.(initial.losses) .- sqrt.(final.losses)
ilim = maximum(abs, filter(isfinite, mask(improvement)))
panel!((3, 5), mask(improvement), "RMS improvement", "m³ m⁻³"; colormap = :balance, colorrange = (-ilim, ilim))
save("$(tag).png", fig)
@info "saved $(tag).png"
