# Where the Breeze reverse-pass compile explodes: for a doubly periodic x–z Breeze model,
# separate the Reactant pipeline (`@code_hlo`: trace, Enzyme MLIR differentiation, raise,
# optimization) from XLA's backend compile (`@compile`), across grid size, float type and
# microphysics. The earlier 8 × 8 Float64 case sat at 81 GB for 3 h 20 min in `@compile`
# without finishing.
#
#   ARCH=cpu julia --project=docs breeze_reverse_scaling.jl

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

measure("6×8 Float64";                     Nx = 6, Nz = 8)
measure("6×8 Float64, no microphysics";    Nx = 6, Nz = 8, microphysics = nothing)
measure("6×8 Float32";                     Nx = 6, Nz = 8, FT = Float32)
measure("8×8 Float64 (the 81 GB case)";    Nx = 8, Nz = 8)
measure("16×8 Float64";                    Nx = 16, Nz = 8)
measure("6×8 Float32 compile";             Nx = 6, Nz = 8, FT = Float32, compile = true)
measure("6×8 Float64 compile";             Nx = 6, Nz = 8, compile = true)
