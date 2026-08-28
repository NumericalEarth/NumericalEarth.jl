# What the reduced solver budget costs physically: the real-data Borneo column run eagerly
# with the full budget (16 canopy Newton iterations, 8 Monin–Obukhov iterates) and with the
# budget the compiled reverse pass uses, compared over the whole run.

include(joinpath(@__DIR__, "borneo_config.jl"))
include(joinpath(@__DIR__, "borneo_model.jl"))
using CairoMakie
using Statistics: mean
using Printf

FT = Float64
Δt = 10minutes
Nsteps = 900
h₀ = 0.28
lapse_rate = 6.5e-3

static    = load_static()
forcing   = load_cache("forcing")
era5_land = load_cache("era5_land")
cpu_grid  = land_grid()

cell = haskey(ENV, "CELL") ? Tuple(parse.(Int, split(ENV["CELL"], ","))) : (9, 9)
i, j = cell
parameters = surface_parameters(static, nothing, FT, cell)
column = column_forcing(forcing, cpu_grid, cell)
Δz = forcing.land_elevation[i, j] - forcing.era5_elevation[i, j]
column = merge(column, (; T = column.T .- lapse_rate * Δz, p = column.p .* exp.(-9.81 * Δz ./ (287 .* column.T))))
θ₀ = FT(static.initial_soil_water[i, j]); T₀ = FT(forcing.skin_temperature[i, j]); q₀ = FT(column.q[1])

function run_column(inner_iterations, similarity_iterations)
    grid = RectilinearGrid(CPU(), FT; size = (), topology = (Flat, Flat, Flat))
    h = surface_field(grid); parent(h) .= h₀
    model = borneo_coupled_model(grid, FT, column, parameters; slab_depth = surface_field(grid),
                                 surface_layer_height, boundary_layer_height, inner_iterations, similarity_iterations)
    hydrology = model.land.hydrology.soil.soil
    parent(hydrology.slab_depth) .= h₀
    parent(model.land.water_storage) .= 1000 * θ₀ * h₀
    parent(model.land.temperature) .= T₀
    parent(model.land.saturation) .= clamp((θ₀ - hydrology.residual_liquid_fraction) / (hydrology.porosity - hydrology.residual_liquid_fraction), 0, 1)
    parent(model.land.prognostic.canopy_water_storage) .= 0
    parent(model.land.prognostic.surface_water_storage) .= 0
    for tile in (model.interfaces.atmosphere_land_interface.vegetated, model.interfaces.atmosphere_land_interface.bare)
        parent(tile.temperature.state.temperature) .= T₀
        parent(tile.temperature.state.specific_humidity) .= q₀
    end
    update_state!(model)
    interface = model.interfaces.atmosphere_land_interface
    scalar(f) = first(interior(f))
    series = (t = zeros(Nsteps), θ = zeros(Nsteps), T = zeros(Nsteps), LE = zeros(Nsteps), H = zeros(Nsteps),
              Tˡᵉᵃᶠ = zeros(Nsteps), Tᵍ = zeros(Nsteps), u★ = zeros(Nsteps))
    wall = @elapsed for n in 1:Nsteps
        time_step!(model, Δt)
        series.t[n] = model.clock.time / 3600
        series.θ[n] = scalar(model.land.water_storage) / (1000h₀)
        series.T[n] = scalar(model.land.temperature)
        series.LE[n] = scalar(interface.fluxes.latent_heat)
        series.H[n] = scalar(interface.fluxes.sensible_heat)
        series.Tˡᵉᵃᶠ[n] = scalar(interface.temperature.canopy)
        series.Tᵍ[n] = scalar(interface.temperature.soil_skin)
        series.u★[n] = scalar(interface.fluxes.friction_velocity)
    end
    @info @sprintf("budget (%d Newton, %d MO): %d steps in %.1f s", inner_iterations, similarity_iterations, Nsteps, wall)
    return series
end

budgets = ((16, 8), (6, 4), (4, 4))
runs = [run_column(b...) for b in budgets]
reference = runs[1]
for (b, r) in zip(budgets[2:end], runs[2:end])
    @info @sprintf("(%d, %d) vs (16, 8): max |ΔLE| = %.2f W m⁻² (RMS %.2f), max |ΔH| = %.2f (RMS %.2f), max |ΔT| = %.3f K, max |Δθ| = %.2e, θ(t_end) %.5f vs %.5f",
                   b..., maximum(abs.(r.LE .- reference.LE)), sqrt(mean((r.LE .- reference.LE).^2)),
                   maximum(abs.(r.H .- reference.H)), sqrt(mean((r.H .- reference.H).^2)),
                   maximum(abs.(r.T .- reference.T)), maximum(abs.(r.θ .- reference.θ)), r.θ[end], reference.θ[end])
end

fig = Figure(size = (1600, 900), fontsize = 15)
Label(fig[0, 1:2], @sprintf("Solver budget sensitivity, Borneo column (%d, %d): full budget vs the budgets the compiled adjoint can afford", i, j); fontsize = 18)
labels = ["$(b[1]) Newton, $(b[2]) MO" for b in budgets]
colors = [:black, :firebrick, :steelblue]
for (k, (name, ylabel)) in enumerate((("LE", "LE (W m⁻²)"), ("H", "H (W m⁻²)"), ("θ", "θ (m³ m⁻³)"), ("Tˡᵉᵃᶠ", "leaf T (K)")))
    ax = Axis(fig[fldmod1(k, 2)...]; title = name, xlabel = "t (h)", ylabel)
    for (r, label, color) in zip(runs, labels, colors)
        lines!(ax, r.t, getproperty(r, Symbol(name)); label, color, linewidth = k == 1 ? 1.5 : 1.2)
    end
    k == 1 && axislegend(ax; position = :lt)
end
save("budget_comparison_i$(i)_j$(j).png", fig)
@info "saved budget_comparison_i$(i)_j$(j).png"
