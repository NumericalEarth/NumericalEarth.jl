# The Central Borneo box: the vegetated slab land forced by ERA5 on every cell, then one
# reverse pass through the compiled coupled step for the whole map,
#
#     L(h) = (1/N) Σₙ Σᵢⱼ wᵢⱼ (θᵢⱼ(tₙ; h) − θᴱᴿᴬ⁵ᴸᵢⱼ(tₙ))²,        θ = Mˡᵃ / (ρˡ h),
#
# giving ∂L/∂h(λ, φ) — the pointwise slab-depth sensitivity, since columns are independent
# (the sum over steps is accumulated inside the traced loop). The map is checked against a
# per-cell finite difference (uniform ±δh perturbation) and one gradient step is taken per
# cell along −∂L/∂h, with the step length chosen per cell by a line search over eager
# forward runs. `ARCH=gpu` runs the reverse pass on the Reactant GPU backend; the eager
# forward runs stay on the CPU.
#
#   REFINEMENT=1 julia --project=docs map_calibration.jl

include(joinpath(@__DIR__, "borneo_config.jl"))
include(joinpath(@__DIR__, "borneo_model.jl"))
using Oceananigans.Architectures: ReactantState
using Reactant
using Enzyme
using Reactant: @trace
using CairoMakie
using Statistics: mean, median, cor
using Printf

FT = Float64
Δt = 10minutes
Nsteps = parse(Int, get(ENV, "NSTEPS", "900"))       # 150 h = 6.25 days; a perfect square for checkpointing
run_hours = Nsteps * Δt / 3600
h₀ = 0.28
lapse_rate = 6.5e-3
inner_iterations = parse(Int, get(ENV, "INNER_ITERATIONS", "6"))         # canopy Newton iterations per step
similarity_iterations = parse(Int, get(ENV, "SIMILARITY_ITERATIONS", "4"))  # Monin–Obukhov iterates per step
backend = get(ENV, "ARCH", "cpu")
tag = "map_calibration_r$(refinement)_$(backend)"

static    = load_static()
forcing   = load_cache("forcing")
era5_land = load_cache("era5_land")
cpu_grid  = land_grid()

# The hourly ERA5-Land target interpolated to every model step, in the parent layout the
# loss reads (halos zero).
Hx, Hy = cpu_grid.Hx, cpu_grid.Hy
θ_target = zeros(FT, Nsteps, Nx + 2Hx, Ny + 2Hy, 1)
for n in 1:Nsteps
    t = n * Δt / 3600
    k = clamp(floor(Int, t) + 1, 1, length(era5_land.times) - 1)
    a = t - (k - 1)
    θ_target[n, 1 + Hx:Nx + Hx, 1 + Hy:Ny + Hy, 1] .= (1 - a) .* era5_land_soil_water(era5_land, k) .+ a .* era5_land_soil_water(era5_land, k + 1)
end
θ_target_end = θ_target[end, 1 + Hx:Nx + Hx, 1 + Hy:Ny + Hy, 1]
weight = FT.(.!static.water)
θ₀ = FT.(static.initial_soil_water)
T₀ = FT.(forcing.skin_temperature)
correction = AltitudeCorrection(forcing.land_elevation, forcing.era5_elevation; lapse_rate)

# ## State initialization shared by every run (parent-level, so it traces on any backend)

function initialize_map!(model, h, θ₀, T₀, q₀)
    hydrology = model.land.hydrology.soil.soil
    parent(hydrology.slab_depth) .= parent(h)
    parent(model.land.water_storage) .= 1000 .* parent(θ₀) .* parent(h)
    parent(model.land.temperature) .= parent(T₀)
    ν, θʳ = parent(hydrology.porosity), parent(hydrology.residual_liquid_fraction)
    parent(model.land.saturation) .= clamp.((parent(θ₀) .- θʳ) ./ (ν .- θʳ), 0, 1)
    parent(model.land.prognostic.canopy_water_storage) .= 0
    parent(model.land.prognostic.surface_water_storage) .= 0
    for tile in (model.interfaces.atmosphere_land_interface.vegetated, model.interfaces.atmosphere_land_interface.bare)
        parent(tile.temperature.state.temperature) .= parent(T₀)
        parent(tile.temperature.state.specific_humidity) .= parent(q₀)
    end
    update_state!(model)   # fluxes consistent with the reset state
    return nothing
end

soil_water(model, h) = parent(model.land.water_storage) ./ (1000 .* parent(h))
cell_loss(model, h, w, θᵗ) = parent(w) .* (soil_water(model, h) .- θᵗ).^2

# Fields on `grid` for the state and target arrays; `h` carries h₀ into the halos too, so
# the parent-level θ = M / (ρ h) stays finite where the loss weight is zero.
function map_fields(grid, depth)
    h = surface_property(grid, fill(FT(1), Nx, Ny)); parent(h) .= 0; set!(h, depth); parent(h) .= ifelse.(parent(h) .== 0, h₀, parent(h))
    return (; h, θ₀ = surface_property(grid, θ₀), T₀ = surface_property(grid, T₀),
              q₀ = surface_property(grid, forcing.q[1]), w = surface_property(grid, weight))
end

# ## Eager CPU forward, recording hourly snapshots

snapshot_names = (:θ, :T, :LST, :𝒮, :Wᶜ, :LE, :LEᶜ, :LEᵍ, :H, :rain, :E)

function forward_map(depth; record = false)
    fields = map_fields(cpu_grid, depth)
    s = surface_parameters(static, cpu_grid, FT)
    model = borneo_coupled_model(cpu_grid, FT, forcing, s; slab_depth = surface_field(cpu_grid),
                                 exchanger_correction = correction, surface_layer_height, boundary_layer_height,
                                 inner_iterations, similarity_iterations)
    initialize_map!(model, fields.h, fields.θ₀, fields.T₀, fields.q₀)
    interface = model.interfaces.atmosphere_land_interface
    land = model.land
    steps_per_hour = round(Int, 3600 / Δt)
    snapshots = record ? NamedTuple{snapshot_names}(ntuple(_ -> zeros(Float32, Nsteps ÷ steps_per_hour + 1, Nx, Ny), length(snapshot_names))) : nothing
    take!(k) = (snapshots.θ[k, :, :]   .= interior(land.water_storage, :, :, 1) ./ (1000 .* interior(fields.h, :, :, 1));
                snapshots.T[k, :, :]   .= interior(land.temperature, :, :, 1);
                snapshots.LST[k, :, :] .= interior(interface.temperature.effective, :, :, 1);
                snapshots.𝒮[k, :, :]   .= interior(land.saturation, :, :, 1);
                snapshots.Wᶜ[k, :, :]  .= interior(land.prognostic.canopy_water_storage, :, :, 1);
                snapshots.LE[k, :, :]  .= interior(interface.fluxes.latent_heat, :, :, 1);
                snapshots.LEᶜ[k, :, :] .= interior(interface.temperature.canopy_latent_heat, :, :, 1);
                snapshots.LEᵍ[k, :, :] .= interior(interface.temperature.soil_latent_heat, :, :, 1);
                snapshots.H[k, :, :]   .= interior(interface.fluxes.sensible_heat, :, :, 1);
                snapshots.rain[k, :, :] .= interior(model.interfaces.exchanger.atmosphere.state.Jʳⁿ, :, :, 1);
                snapshots.E[k, :, :]   .= interior(land.fluxes.vapor_flux, :, :, 1))
    record && take!(1)
    losses = zeros(FT, size(parent(fields.h)))
    wall = time_ns()
    for n in 1:Nsteps
        time_step!(model, Δt)
        losses .+= cell_loss(model, fields.h, fields.w, view(θ_target, n, :, :, :))
        record && n % steps_per_hour == 0 && take!(n ÷ steps_per_hour + 1)
    end
    @info @sprintf("eager forward (%d × %d, %d steps) in %.1f s", Nx, Ny, Nsteps, 1e-9 * (time_ns() - wall))
    losses = losses[1 + Hx:Nx + Hx, 1 + Hy:Ny + Hy, 1] ./ Nsteps
    θ_end = interior(land.water_storage, :, :, 1) ./ (1000 .* interior(fields.h, :, :, 1))
    return (; losses, θ_end = Array(θ_end), snapshots)
end

@info "forward run at h = $h₀ m"
forward = forward_map(fill(h₀, Nx, Ny); record = true)
land = weight .> 0
rms(losses) = sqrt(sum(losses) / count(land))
@info @sprintf("L = run-mean Σ wᵢⱼ (θ − θᴱᴿᴬ⁵ᴸ)² = %.4e over %d land cells; RMS mismatch %.4f m³ m⁻³",
               sum(forward.losses), count(land), rms(forward.losses))

# ## The compiled reverse pass over the whole map

Reactant.set_default_backend(backend)

function soil_water_loss(model, h, θ₀, T₀, q₀, w, θᵗ, Δt, nsteps)
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

grid_ad = land_grid(ReactantState(), FT)
fields_ad = map_fields(grid_ad, fill(h₀, Nx, Ny))
dh_ad = Enzyme.make_zero(fields_ad.h)
θ_target_ad = Reactant.to_rarray(θ_target)
s_ad = surface_parameters(static, grid_ad, FT)
model_ad = borneo_coupled_model(grid_ad, FT, forcing, s_ad; slab_depth = surface_field(grid_ad),
                                exchanger_correction = correction, surface_layer_height, boundary_layer_height,
                                inner_iterations, similarity_iterations)
Oceananigans.initialize!(model_ad)
dmodel = Enzyme.make_zero(model_ad)

@info "compiling the reverse pass over $Nsteps steps on the $backend backend..."
compile_seconds = @elapsed compiled = Reactant.@compile raise=true raise_first=true sync=true grad_soil_water_loss(
    model_ad, dmodel, fields_ad.h, dh_ad, fields_ad.θ₀, fields_ad.T₀, fields_ad.q₀, fields_ad.w, θ_target_ad, Δt, Nsteps)
run_seconds = @elapsed dh_out, L_ad = compiled(model_ad, dmodel, fields_ad.h, dh_ad, fields_ad.θ₀, fields_ad.T₀,
                                                fields_ad.q₀, fields_ad.w, θ_target_ad, Δt, Nsteps)
adjoint_map = Array(interior(dh_out, :, :, 1))
L_compiled = Reactant.to_number(L_ad)
@info @sprintf("adjoint: L = %.6e (eager %.6e); ⟨∂L/∂h⟩ = %+.4e, range [%+.3e, %+.3e] m⁻¹  [compile %.0f s, run %.1f s]",
               L_compiled, sum(forward.losses), mean(adjoint_map), extrema(adjoint_map)..., compile_seconds, run_seconds)

# ## Per-cell finite difference and one gradient step per cell

δ = 1e-3 * h₀
fd_map = (forward_map(fill(h₀ + δ, Nx, Ny)).losses .- forward_map(fill(h₀ - δ, Nx, Ny)).losses) ./ 2δ
@info @sprintf("finite difference: ⟨∂L/∂h⟩ = %+.4e; max |adjoint − FD| = %.3e (%.2f%% of max |FD|); correlation %.4f",
               mean(fd_map), maximum(abs.(adjoint_map .- fd_map)[land]),
               100 * maximum(abs.(adjoint_map .- fd_map)[land]) / maximum(abs.(fd_map)),
               cor(vec(adjoint_map[land]), vec(fd_map[land])))

# Each column descends along −∂L/∂h by a step length of its own, chosen among a geometric
# set of trial lengths from the eager forward runs (columns are independent, so one run
# evaluates every cell's loss at a given trial length). Depths stay within [0.02, 5] m.
direction = -sign.(adjoint_map)
trial_scales = h₀ .* (2.0 .^ (4:-1:-4))
trial_losses = [forward_map(clamp.(h₀ .+ direction .* s, 0.02, 5.0)).losses for s in trial_scales]
best = [argmin([trial_losses[k][i, j] for k in eachindex(trial_scales)]) for i in 1:Nx, j in 1:Ny]
h₁ = [land[i, j] ? clamp(h₀ + direction[i, j] * trial_scales[best[i, j]], 0.02, 5.0) : h₀ for i in 1:Nx, j in 1:Ny]
calibrated = forward_map(h₁)
improved = count((calibrated.losses .< forward.losses)[land])
@info @sprintf("gradient step with line search: L %.4e → %.4e (RMS mismatch %.4f → %.4f); %d of %d land cells improved; h₁ ∈ [%.3f, %.3f] m, median %.3f m",
               sum(forward.losses), sum(calibrated.losses), rms(forward.losses), rms(calibrated.losses),
               improved, count(land), extrema(h₁[land])..., median(h₁[land]))

jldsave("$(tag).jld2"; h₀, h₁, θ_target_end, weight, adjoint_map, fd_map, losses = forward.losses, trial_scales,
        calibrated_losses = calibrated.losses, θ_end = forward.θ_end, θ_end_calibrated = calibrated.θ_end,
        snapshots = Dict(pairs(forward.snapshots)), L_compiled, compile_seconds, run_seconds, Nsteps, Δt,
        longitude = static.longitude, latitude = static.latitude)

# ## Figure

λ, φ = static.longitude, static.latitude
mask(a) = ifelse.(land, a, NaN)
fig = Figure(size = (1900, 1250), fontsize = 15)
Label(fig[0, 1:6], @sprintf("Slab-depth sensitivity of the %.2f-day run-mean soil-water mismatch, Central Borneo at ≈ %d km (%s backend): adjoint vs finite difference, and one gradient step (RMS %.3f → %.3f)",
                            run_hours / 24, resolution_km, backend, rms(forward.losses), rms(calibrated.losses)); fontsize = 18)

function panel!(pos, data, title, label; colormap = :viridis, colorrange = nothing)
    ax = Axis(fig[pos...]; title, aspect = DataAspect(), xlabel = "longitude", ylabel = "latitude")
    hm = isnothing(colorrange) ? heatmap!(ax, λ, φ, data; colormap) : heatmap!(ax, λ, φ, data; colormap, colorrange)
    Colorbar(fig[pos[1], pos[2] + 1], hm; label)
    return ax
end

θlim = extrema(filter(isfinite, [mask(θ_target_end); mask(forward.θ_end); mask(calibrated.θ_end)]))
panel!((1, 1), mask(θ_target_end), "ERA5-Land θ (0–28 cm) at t_end", "m³ m⁻³"; colormap = :tempo, colorrange = θlim)
panel!((1, 3), mask(forward.θ_end), @sprintf("slab θ at t_end, h = %.2f m", h₀), "m³ m⁻³"; colormap = :tempo, colorrange = θlim)
mlim = maximum(abs, filter(isfinite, mask(forward.θ_end .- θ_target_end)))
panel!((1, 5), mask(forward.θ_end .- θ_target_end), "mismatch θ − θᴱᴿᴬ⁵ᴸ at t_end", "m³ m⁻³"; colormap = :balance, colorrange = (-mlim, mlim))

slim = maximum(abs, filter(isfinite, [mask(adjoint_map); mask(fd_map)]))
panel!((2, 1), mask(adjoint_map), "adjoint ∂L/∂h — one reverse pass", "m⁻¹"; colormap = :balance, colorrange = (-slim, slim))
panel!((2, 3), mask(fd_map), "finite-difference ∂L/∂h", "m⁻¹"; colormap = :balance, colorrange = (-slim, slim))
dlim = max(maximum(abs, filter(isfinite, mask(adjoint_map .- fd_map))), eps())
panel!((2, 5), mask(adjoint_map .- fd_map), "adjoint − FD", "m⁻¹"; colormap = :balance, colorrange = (-dlim, dlim))

panel!((3, 1), mask(h₁), "slab depth after one gradient step", "m"; colormap = :viridis)
panel!((3, 3), mask(sqrt.(forward.losses)), "run-RMS mismatch, h₀", "m³ m⁻³"; colormap = :amp, colorrange = (0, maximum(sqrt.(forward.losses[land]))))
panel!((3, 5), mask(sqrt.(calibrated.losses)), "run-RMS mismatch after the step", "m³ m⁻³"; colormap = :amp, colorrange = (0, maximum(sqrt.(forward.losses[land]))))

save("$(tag).png", fig)
@info "saved $(tag).png"
