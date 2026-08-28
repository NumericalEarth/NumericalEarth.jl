# One Central Borneo forest column: the vegetated slab land forced by ERA5, compared with
# ERA5-Land soil water, then differentiated. The soil-water mismatch at the end of the run,
#
#     L(h) = (θ(t_end; h) − θᴱᴿᴬ⁵ᴸ(t_end))²,        θ = Mˡᵃ / (ρˡ h),
#
# is differentiated with respect to the slab depth `h` by Enzyme reverse mode through the
# Reactant-compiled coupled time step, checked against a finite difference, and one
# Gauss–Newton step `h ← h − 2L / (dL/dh)` is taken and re-run.
#
#   REFINEMENT=1 CELL=9,9 julia --project=docs column_calibration.jl

include(joinpath(@__DIR__, "borneo_config.jl"))
include(joinpath(@__DIR__, "borneo_model.jl"))
using Oceananigans.Architectures: ReactantState
using Reactant
using Enzyme
using Reactant: @trace
using CairoMakie
using Statistics: mean
using Printf

FT = Float64
Δt = 10minutes
Nsteps = 30^2                          # 150 h = 6.25 days; a perfect square for checkpointing
run_hours = Nsteps * Δt / 3600
h₀ = 0.28                              # the ERA5-Land 0–28 cm column the target is built from
lapse_rate = 6.5e-3
inner_iterations = parse(Int, get(ENV, "INNER_ITERATIONS", "6"))         # canopy Newton iterations per step
similarity_iterations = parse(Int, get(ENV, "SIMILARITY_ITERATIONS", "4"))  # Monin–Obukhov iterates per step

static    = load_static()
forcing   = load_forcing = load_cache("forcing")
era5_land = load_cache("era5_land")
cpu_grid  = land_grid()

# ## The column: a densely vegetated land cell near the domain center

function choose_cell(static)
    candidates = findall(.!static.water .& (static.vegetation_fraction .> 0.7) .& (static.leaf_area_index .> 2))
    center = (Nx + 1) / 2, (Ny + 1) / 2
    return Tuple(candidates[argmin([hypot(c[1] - center[1], c[2] - center[2]) for c in candidates])])
end
cell = haskey(ENV, "CELL") ? Tuple(parse.(Int, split(ENV["CELL"], ","))) : choose_cell(static)
i, j = cell
λ, φ = static.longitude[i], static.latitude[j]
@info @sprintf("column (%d, %d) at %.2f°E %.2f°N: %s, LAI %.2f, canopy %.1f m, f_veg %.2f, ℓᵐ %.2f m, ν %.3f, θʳ %.3f, θ₀ %.3f, z %.0f m (ERA5 %.0f m)",
               i, j, λ, φ, static.canopy_class[i, j], static.leaf_area_index[i, j], static.canopy_height[i, j],
               static.vegetation_fraction[i, j], static.vegetated_roughness_length[i, j],
               static.porosity[i, j], static.residual_liquid_fraction[i, j], static.initial_soil_water[i, j],
               forcing.land_elevation[i, j], forcing.era5_elevation[i, j])

parameters = surface_parameters(static, nothing, FT, cell)
column = column_forcing(forcing, cpu_grid, cell)

# Lift the ERA5 near-surface state from ERA5's terrain to the ETOPO surface: a lapse-rate
# temperature shift and the matching hydrostatic pressure change.
Δz = forcing.land_elevation[i, j] - forcing.era5_elevation[i, j]
column = merge(column, (; T = column.T .- lapse_rate * Δz,
                          p = column.p .* exp.(-9.81 * Δz ./ (287 .* column.T))))

θ_obs = [era5_land_soil_water(era5_land, n)[i, j] for n in eachindex(era5_land.times)]
θ_obs_layer_1 = era5_land.layer_1[:, i, j]
θ₀ = FT(static.initial_soil_water[i, j])
T₀ = FT(forcing.skin_temperature[i, j])
q₀ = FT(column.q[1])
n_end = round(Int, run_hours) + 1
θ_target = FT(θ_obs[n_end])

# ## State initialization shared by every run

function initialize_column!(model, h, θ₀, T₀, q₀)
    hydrology = model.land.hydrology.soil.soil
    parent(hydrology.slab_depth) .= parent(h)
    parent(model.land.water_storage) .= 1000 .* θ₀ .* parent(h)
    parent(model.land.temperature) .= T₀
    ν, θʳ = hydrology.porosity, hydrology.residual_liquid_fraction
    parent(model.land.saturation) .= clamp((θ₀ - θʳ) / (ν - θʳ), 0, 1)
    parent(model.land.prognostic.canopy_water_storage) .= 0
    parent(model.land.prognostic.surface_water_storage) .= 0
    for tile in (model.interfaces.atmosphere_land_interface.vegetated, model.interfaces.atmosphere_land_interface.bare)
        parent(tile.temperature.state.temperature) .= T₀
        parent(tile.temperature.state.specific_humidity) .= q₀
    end
    update_state!(model)   # fluxes consistent with the reset state
    return nothing
end

soil_water(model, h) = interior(model.land.water_storage) ./ (1000 .* interior(h))
scalar(f) = first(interior(f))

# ## Eager forward run, recording the column's evolution

series_names = (:t, :T, :θ, :𝒮, :Wᶜ, :Wᵖ, :LE, :LEᶜ, :LEᵍ, :LEʷ, :H, :Tᵃᶜ, :Tˡᵉᵃᶠ, :Tᵍ, :E, :P, :R, :D, :u★, :Tᵃ, :rain, :sw)

function forward_column(depth)
    grid = RectilinearGrid(CPU(), FT; size = (), topology = (Flat, Flat, Flat))
    h = surface_field(grid); parent(h) .= depth
    model = borneo_coupled_model(grid, FT, column, parameters; slab_depth = surface_field(grid),
                                 surface_layer_height, boundary_layer_height, inner_iterations, similarity_iterations)
    initialize_column!(model, h, θ₀, T₀, q₀)
    interface = model.interfaces.atmosphere_land_interface
    land = model.land
    atmosphere_state = model.interfaces.exchanger.atmosphere.state
    series = NamedTuple{series_names}(ntuple(_ -> zeros(Nsteps), length(series_names)))
    for n in 1:Nsteps
        time_step!(model, Δt)
        series.t[n]  = model.clock.time / 3600
        series.T[n]  = scalar(land.temperature)
        series.θ[n]  = first(soil_water(model, h))
        series.𝒮[n]  = scalar(land.saturation)
        series.Wᶜ[n] = scalar(land.prognostic.canopy_water_storage)
        series.Wᵖ[n] = scalar(land.prognostic.surface_water_storage)
        series.LE[n]  = scalar(interface.fluxes.latent_heat)
        series.LEᶜ[n] = scalar(interface.temperature.canopy_latent_heat)
        series.LEᵍ[n] = scalar(interface.temperature.soil_latent_heat)
        series.LEʷ[n] = scalar(interface.temperature.canopy_wet_latent_heat)
        series.H[n]   = scalar(interface.fluxes.sensible_heat)
        series.Tᵃᶜ[n]   = scalar(interface.temperature.interface)
        series.Tˡᵉᵃᶠ[n] = scalar(interface.temperature.canopy)
        series.Tᵍ[n]    = scalar(interface.temperature.soil_skin)
        series.E[n]  = scalar(land.fluxes.vapor_flux)
        series.P[n]  = scalar(land.fluxes.liquid_precipitation_flux)
        series.R[n]  = scalar(land.diagnostics.surface_water_runoff)
        series.D[n]  = scalar(land.diagnostics.deep_liquid_flux)
        series.u★[n] = scalar(interface.fluxes.friction_velocity)
        series.Tᵃ[n]   = scalar(atmosphere_state.T)
        series.rain[n] = scalar(atmosphere_state.Jʳⁿ)
        series.sw[n]   = scalar(model.interfaces.exchanger.radiation.state.ℐꜜˢʷ)
    end
    return series
end

@info "forward run at h = $h₀ m"
forward = forward_column(h₀)
loss(series) = (series.θ[end] - θ_target)^2
@info @sprintf("θ(t_end) = %.4f vs ERA5-Land %.4f  →  L = %.3e", forward.θ[end], θ_target, loss(forward))

# ## The compiled reverse pass

backend = get(ENV, "ARCH", "cpu")
Reactant.set_default_backend(backend)

function soil_water_loss(model, h, θ₀, T₀, q₀, θ_target, Δt, nsteps)
    initialize_column!(model, h, θ₀, T₀, q₀)
    @trace mincut=true checkpointing=true track_numbers=false for _ in 1:nsteps
        time_step!(model, Δt)
    end
    return sum((soil_water(model, h) .- θ_target).^2)
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
model_ad = borneo_coupled_model(grid_ad, FT, column, parameters; slab_depth = surface_field(grid_ad),
                                surface_layer_height, boundary_layer_height, inner_iterations, similarity_iterations)
Oceananigans.initialize!(model_ad)
dmodel = Enzyme.make_zero(model_ad)

@info "compiling the reverse pass over $Nsteps steps..."
compile_seconds = @elapsed compiled = Reactant.@compile raise=true raise_first=true sync=true grad_soil_water_loss(
    model_ad, dmodel, h_ad, dh_ad, θ₀, T₀, q₀, θ_target, Δt, Nsteps)
run_seconds = @elapsed dh_out, L_ad = compiled(model_ad, dmodel, h_ad, dh_ad, θ₀, T₀, q₀, θ_target, Δt, Nsteps)
dL_dh = first(Array(parent(dh_out)))
L_compiled = Reactant.to_number(L_ad)
@info @sprintf("adjoint: L = %.6e (eager %.6e), dL/dh = %.6e m⁻¹  [compile %.0f s, run %.1f s]",
               L_compiled, loss(forward), dL_dh, compile_seconds, run_seconds)

# ## Finite-difference check and one Gauss–Newton step

δ = 1e-3 * h₀
fd = (loss(forward_column(h₀ + δ)) - loss(forward_column(h₀ - δ))) / 2δ
@info @sprintf("finite difference dL/dh = %.6e m⁻¹ (adjoint / FD = %.4f)", fd, dL_dh / fd)

Δh = clamp(-2 * L_compiled / dL_dh, -0.5h₀, 0.5h₀)
h₁ = h₀ + Δh
calibrated = forward_column(h₁)
@info @sprintf("Gauss–Newton step: h %.3f → %.3f m;  θ(t_end) %.4f → %.4f (target %.4f);  L %.3e → %.3e",
               h₀, h₁, forward.θ[end], calibrated.θ[end], θ_target, loss(forward), loss(calibrated))

jldsave("column_calibration_r$(refinement)_i$(i)_j$(j).jld2"; cell, λ, φ, h₀, h₁, θ_target, θ_obs, θ_obs_layer_1,
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
lines!(ax, t, forward.θ; color = :firebrick, linewidth = 2, label = @sprintf("slab, h = %.2f m", h₀))
lines!(ax, t, calibrated.θ; color = :seagreen, linewidth = 2, label = @sprintf("slab, h = %.2f m (one step)", h₁))
vlines!(ax, [run_hours]; color = :black, linestyle = :dot)
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
