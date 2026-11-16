# Forward compiles of increasing size, for locating what a gradient compile spends its time and memory on.
# Each rung prints its compile time and the peak resident memory of the session so far; a killed session
# stopped at the last rung announced. The gradient itself is `coupled_forecast_gradient.jl`.
#
#   julia --project=. compile_ladder.jl

include("coupled_model.jl")

model = coupled_model(arch)
δ = initial_perturbations(model)

function climb(f, name, args...)
    @info "Compiling $name..."
    compile_time = @elapsed Reactant.@compile raise=true raise_first=true sync=true f(args...)
    @info @sprintf("%s compiled in %s, peak memory %s", name, prettytime(compile_time), Base.format_bytes(Sys.maxrss()))
    return nothing
end

climb(time_step!, "one land step", model.land, Δt)
climb(time_step!, "one ocean step", model.ocean.model, Δt)
climb(time_step!, "one atmosphere step", model.atmosphere.model, Δt)
climb(time_step!, "one coupled step", model, Δt)
climb(forecast_land_temperature, "the $Nt-step forecast", model, δ, land_columns, Δt, Nt)
