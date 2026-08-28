# How the checkpointed reverse pass of the bare slab scales with the number of traced steps:
# HLO size and compile time for perfect-square step counts. A graph that keeps growing with
# the step count means the `@trace` loop is unrolled rather than kept as a loop.

include(joinpath(@__DIR__, "compile_scaling_setup.jl"))

step_counts = parse.(Int, split(get(ENV, "STEPS", "9,16,36,64,144"), ","))

grid = RectilinearGrid(ReactantState(), FT; size = (), topology = (Flat, Flat, Flat))
model = bare_slab_model(grid, parameters)
Oceananigans.initialize!(model)
h = surface_field(grid); parent(h) .= 0.3
dh = Enzyme.make_zero(h)
dmodel = Enzyme.make_zero(model)

for nsteps in step_counts
    t_hlo = @elapsed lines = length(split(string(Reactant.@code_hlo raise=true raise_first=true grad(model, dmodel, h, dh, Δt, nsteps)), '\n'))
    t = @elapsed compiled = Reactant.@compile raise=true raise_first=true sync=true grad(model, dmodel, h, dh, Δt, nsteps)
    tr = @elapsed out = compiled(model, dmodel, h, dh, Δt, nsteps)
    @info @sprintf("[bare slab] reverse over %3d steps: %6d HLO lines (traced in %.0f s), compiled in %.1f s, ran in %.2f s, dL/dh = %.4e, L = %.4e",
                   nsteps, lines, t_hlo, t, tr, first(Array(parent(out[1]))), Reactant.to_number(out[2]))
end
