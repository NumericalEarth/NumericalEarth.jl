# Are the floor-pinned cells of the calibration in the wrong basin? Evaluate their per-cell
# loss with the calibrated depths, and with those seven cells alone moved to the deep bound.

include(joinpath(@__DIR__, "map_setup.jl"))
using Printf

it = jldopen(f -> Dict(k => f[k] for k in keys(f)), "map_iterations_r1_gpu.jld2")
depths = it["depths"]
floor_cells = findall((depths .<= 0.021) .& land)

deep = copy(depths); deep[floor_cells] .= 5.0
mid  = copy(depths); mid[floor_cells] .= 0.28
losses_floor = forward_map(depths).losses
losses_deep  = forward_map(deep).losses
losses_mid   = forward_map(mid).losses

@printf("%-7s %-7s %10s %10s %10s   best\n", "lon", "lat", "RMS@0.02", "RMS@0.28", "RMS@5.0")
for c in floor_cells
    i, j = Tuple(c)
    r = (sqrt(losses_floor[i, j]), sqrt(losses_mid[i, j]), sqrt(losses_deep[i, j]))
    @printf("%-7.2f %-7.2f %10.4f %10.4f %10.4f   %s\n", static.longitude[i], static.latitude[j], r...,
            ("0.02 m", "0.28 m", "5 m")[argmin(collect(r))])
end
@printf("\ntotal L: calibrated %.5e;  floor cells at 5 m %.5e;  at 0.28 m %.5e\n",
        sum(losses_floor), sum(losses_deep), sum(losses_mid))
