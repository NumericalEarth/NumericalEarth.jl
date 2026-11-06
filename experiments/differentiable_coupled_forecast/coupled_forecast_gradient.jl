# # Sensitivity of a coupled land–ocean–atmosphere forecast to its initial conditions
#
# A 2D Breeze atmosphere over an Oceananigans ocean with a `SlabLand` island in the middle, built on
# `ReactantState`, compiled, and differentiated with Enzyme. The setup mirrors
# `examples/breeze_over_slab_land.jl` without radiation, with ocean in place of the slab's dry edges:
#
#   * the ocean is a hydrostatic Oceananigans model whose bathymetry rises to the surface over the
#     central half of the domain; those dry columns are the land, and the model's `ocean_fraction`
#     (0 there, 1 elsewhere) has the atmosphere take its surface fluxes from the land there and from
#     the ocean elsewhere;
#   * the atmosphere is compressible with a fixed number of acoustic substeps, so the step has a
#     fixed trip count and traces;
#   * both similarity-theory flux solvers run a fixed number of iterations for the same reason.
#
# A steady wind blows from the ocean onto the land. The forecast quantity is the mean land skin
# temperature after `stop_time`, and one reverse pass gives its gradient with respect to the initial
# atmospheric potential temperature density `ρθ`, ocean temperature, and land skin temperature.
#
#   julia --project=experiments/differentiable_coupled_forecast experiments/differentiable_coupled_forecast/coupled_forecast_gradient.jl
#
# EXCLUDED_PATTERNS=greedy_while_loop_batch_fission,while_deadresult leaves Enzyme-JAX patterns out of the
# compile (`patches.jl`).

include("coupled_model.jl")

using CairoMakie

model = coupled_model(arch)
δ = initial_perturbations(model)

# ## The gradient

function forecast_gradient(model, dmodel, δ, ∂δ, land_columns, Δt, Nt)
    _, J = Enzyme.autodiff(
        Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal),
        forecast_land_temperature, Enzyme.Active,
        Enzyme.Duplicated(model, dmodel),
        Enzyme.Duplicated(δ, ∂δ),
        Enzyme.Const(land_columns),
        Enzyme.Const(Δt),
        Enzyme.Const(Nt))
    return J
end

dmodel = Enzyme.make_zero(model)
∂δ = Enzyme.make_zero(δ)

@info "Compiling the gradient of a $(prettytime(stop_time)) coupled forecast ($Nt steps)..."
compile_time = @elapsed compiled_gradient = Reactant.@compile raise=true raise_first=true sync=true forecast_gradient(
    model, dmodel, δ, ∂δ, land_columns, Δt, Nt)

@info "Compiled in $(prettytime(compile_time)); running..."
run_time = @elapsed J = compiled_gradient(model, dmodel, δ, ∂δ, land_columns, Δt, Nt)

@info @sprintf("Ran in %s: J = %.4f K, |∂J/∂ρθ| = %.3e, |∂J/∂Tᵒᶜ| = %.3e, |∂J/∂Tˡᵃ| = %.3e",
               prettytime(run_time), Reactant.to_number(J),
               map(f -> sqrt(sum(abs2, Array(interior(f)))), values(∂δ))...)

# ## Finite-difference check
#
# Warm the whole initial ocean by `±ε` and compare the change in the forecast with the gradient's
# projection on that direction. Ocean warming reaches the land only through the atmosphere, so this
# checks the derivative through every coupling. The compiled gradient is reused for its primal on two
# fresh models.

ε = 0.01 # K

function perturbed_forecast(ε)
    model = coupled_model(arch)
    δ = Enzyme.make_zero(∂δ)
    set!(δ.Tᵒᶜ, ε)
    return Reactant.to_number(compiled_gradient(model, Enzyme.make_zero(model), δ, Enzyme.make_zero(δ), land_columns, Δt, Nt))
end

finite_difference = (perturbed_forecast(ε) - perturbed_forecast(-ε)) / 2ε
adjoint = sum(Array(interior(∂δ.Tᵒᶜ)))

@info @sprintf("dJ/dTᵒᶜ along a uniform warming: adjoint %+.6e, finite difference %+.6e K/K", adjoint, finite_difference)

# ## Sensitivity maps

fig = Figure(size = (1200, 400))

ax_ρθ = Axis(fig[1, 1], title = "∂J/∂(ρθ)₀", xlabel = "x (m)", ylabel = "z (m)", limits = (nothing, (0, 3e3)))
ax_T  = Axis(fig[1, 2], title = "∂J/∂T₀ᵒᶜ", xlabel = "x (m)", ylabel = "z (m)")
ax_Tˡᵃ = Axis(fig[1, 3], title = "∂J/∂T₀ˡᵃ", xlabel = "x (m)")

hm_ρθ = heatmap!(ax_ρθ, on_architecture(CPU(), ∂δ.ρθ); colormap = :balance)
hm_T  = heatmap!(ax_T,  on_architecture(CPU(), ∂δ.Tᵒᶜ); colormap = :balance)
lines!(ax_Tˡᵃ, on_architecture(CPU(), ∂δ.Tˡᵃ); color = :black, linewidth = 2)

Colorbar(fig[2, 1], hm_ρθ, vertical = false)
Colorbar(fig[2, 2], hm_T,  vertical = false)

save("coupled_forecast_sensitivity.png", fig)
fig
