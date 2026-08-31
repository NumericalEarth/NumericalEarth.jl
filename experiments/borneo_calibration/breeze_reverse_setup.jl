# Model, loss and measurement helpers for the Breeze reverse-pass compile experiments.

using NumericalEarth
using Oceananigans
using Oceananigans.Units
using Breeze
using Oceananigans.Architectures: ReactantState
using Reactant
using Enzyme
using Printf

Reactant.set_default_backend(get(ENV, "ARCH", "cpu"))
maxrss_gb() = Sys.maxrss() / 2^30

function breeze_model(FT, Nx, Nz; kw...)
    grid = RectilinearGrid(ReactantState(), FT; size = (Nx, Nz), halo = (5, 5),
                           x = (-1kilometer, 1kilometer), z = (0, 1kilometer),
                           topology = (Periodic, Flat, Bounded))
    simulation = atmosphere_simulation(grid; potential_temperature = 295.0, kw...)
    set!(simulation.model, θ = simulation.model.dynamics.reference_state.surface_potential_temperature, u = 2)
    return simulation.model
end

function loss(model, Δt)
    time_step!(model, Δt)
    return sum(parent(model.dynamics.total_density))
end

function reverse!(model, dmodel, Δt)
    _, L = Enzyme.autodiff(Enzyme.set_strong_zero(Enzyme.ReverseWithPrimal), loss, Enzyme.Active,
                           Enzyme.Duplicated(model, dmodel), Enzyme.Const(Δt))
    return L
end

function measure(name; Nx, Nz, FT = Float64, compile = false, kw...)
    try
        model = breeze_model(FT, Nx, Nz; kw...)
        dmodel = Enzyme.make_zero(model)
        Δt = FT(2)
        t_hlo = @elapsed lines = length(split(string(Reactant.@code_hlo raise=true raise_first=true reverse!(model, dmodel, Δt)), '\n'))
        @info @sprintf("[%s] reverse HLO: %d lines in %.0f s (peak rss %.1f GB)", name, lines, t_hlo, maxrss_gb())
        if compile
            t_c = @elapsed compiled = Reactant.@compile raise=true raise_first=true sync=true reverse!(model, dmodel, Δt)
            t_r = @elapsed L = compiled(model, dmodel, Δt)
            @info @sprintf("[%s] compiled in %.0f s, ran in %.1f s, L = %.6e (peak rss %.1f GB)", name, t_c, t_r, Reactant.to_number(L), maxrss_gb())
        end
    catch err
        io = IOBuffer(); showerror(io, err); message = first(split(String(take!(io)), '\n'))
        @error "[$name] FAILED: $message" exception = (err, catch_backtrace())
    end
    GC.gc()
    return nothing
end

