# How the Reactant compile of the coupled step scales with the land configuration: HLO size
# and compile time of one forward step, then of the reverse pass over 1 and 4 steps, for
# the bare slab of `examples/era5_forced_slab_land.jl` and for the two-tile canopy land at
# two iteration budgets. Smallest cases first so a partial log is still informative.

include(joinpath(@__DIR__, "compile_scaling_setup.jl"))

function hlo_lines(f, args...)
    t = @elapsed lines = try
        length(split(string(Reactant.@code_hlo raise=true raise_first=true f(args...)), '\n'))
    catch err
        @warn "code_hlo failed" exception = (err, catch_backtrace()); -1
    end
    @info @sprintf("    traced + raised to %d HLO lines in %.1f s", lines, t)
    return lines
end

function measure(name, build; steps = (1, 4))
    grid = RectilinearGrid(ReactantState(), FT; size = (), topology = (Flat, Flat, Flat))
    model = build(grid)
    Oceananigans.initialize!(model)
    h = surface_field(grid); parent(h) .= 0.3
    dh = Enzyme.make_zero(h)
    dmodel = Enzyme.make_zero(model)

    lines = hlo_lines(forward_step!, model, Δt)
    t = @elapsed compiled = Reactant.@compile raise=true raise_first=true sync=true forward_step!(model, Δt)
    @info @sprintf("[%s] forward step: %d HLO lines, compiled in %.1f s", name, lines, t)
    compiled(model, Δt)

    for nsteps in steps
        lines = hlo_lines(grad, model, dmodel, h, dh, Δt, nsteps)
        t = @elapsed compiled = Reactant.@compile raise=true raise_first=true sync=true grad(model, dmodel, h, dh, Δt, nsteps)
        tr = @elapsed out = compiled(model, dmodel, h, dh, Δt, nsteps)
        @info @sprintf("[%s] reverse over %d steps: %d HLO lines, compiled in %.1f s, ran in %.2f s, dL/dh = %.4e, L = %.4e",
                       name, nsteps, lines, t, tr, first(Array(parent(out[1]))), Reactant.to_number(out[2]))
    end
    return nothing
end

measure("bare slab, MO 8", grid -> bare_slab_model(grid, parameters))
measure("canopy tiles, inner 4, MO 4", grid -> borneo_coupled_model(grid, FT, idealized_forcing, parameters; slab_depth = surface_field(grid), inner_iterations = 4, similarity_iterations = 4))
measure("canopy tiles, inner 16, MO 8", grid -> borneo_coupled_model(grid, FT, idealized_forcing, parameters; slab_depth = surface_field(grid)))
