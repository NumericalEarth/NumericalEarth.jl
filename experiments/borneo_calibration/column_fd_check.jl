# Convergence of the finite-difference slab-depth sensitivity of the column's run-mean
# soil-water loss with the step size, against the adjoint saved by `column_calibration.jl`.

include(joinpath(@__DIR__, "column_setup.jl"))

saved = jldopen(f -> Dict(k => f[k] for k in ("dL_dh", "fd", "h₀")), "column_calibration_r$(refinement)_i$(i)_j$(j).jld2")
loss(series) = mean((series.θ .- θ_target).^2)

L₀ = loss(forward_column(h₀))
@info @sprintf("L(h₀ = %.3f) = %.6e;  adjoint dL/dh = %.6e", h₀, L₀, saved["dL_dh"])
for relative in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
    δ = relative * h₀
    central = (loss(forward_column(h₀ + δ)) - loss(forward_column(h₀ - δ))) / 2δ
    forward = (loss(forward_column(h₀ + δ)) - L₀) / δ
    @info @sprintf("δ/h₀ = %7.1e:  central %.6e (adjoint/central %.4f),  one-sided %.6e", relative, central, saved["dL_dh"] / central, forward)
end
