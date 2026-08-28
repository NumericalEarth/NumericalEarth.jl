# Eager CPU probe: step the idealized 0D bare slab and canopy column and report the first
# NaN in any land, interface or exchanger field, step by step.

include(joinpath(@__DIR__, "compile_scaling_setup.jl"))

function nan_fields(model)
    found = String[]
    walk(prefix, x::AbstractArray) = any(isnan, Array(x)) && push!(found, prefix)
    walk(prefix, x::Field) = walk(prefix, parent(x))
    walk(prefix, ::Union{Oceananigans.Fields.ZeroField, Oceananigans.Fields.ConstantField}) = nothing
    walk(prefix, x::NamedTuple) = foreach(k -> walk("$prefix.$k", x[k]), keys(x))
    walk(prefix, x::Tuple) = foreach(i -> walk("$prefix[$i]", x[i]), eachindex(x))
    function walk(prefix, x)
        isstructtype(typeof(x)) || return nothing
        for name in fieldnames(typeof(x))
            name in (:grid, :clock, :atmosphere, :radiation) && continue
            walk("$prefix.$name", getfield(x, name))
        end
    end
    walk("land", model.land)
    walk("interface", model.interfaces.atmosphere_land_interface)
    walk("exchanger", model.interfaces.exchanger)
    return found
end

function probe(name, build; nsteps = 6)
    grid = RectilinearGrid(CPU(), FT; size = (), topology = (Flat, Flat, Flat))
    model = build(grid)
    h = surface_field(grid); parent(h) .= 0.3
    reset!(model, h)
    interface = model.interfaces.atmosphere_land_interface
    bad = nan_fields(model)
    @info "[$name] before stepping: $(isempty(bad) ? "clean" : join(bad, ", "))"
    for n in 1:nsteps
        time_step!(model, Δt)
        bad = nan_fields(model)
        @info @sprintf("[%s] step %d: T = %.3f  M = %.3f  𝒮 = %.3f  LE = %.2f  H = %.2f  u★ = %.4f  E = %.3e  %s",
                       name, n, first(interior(model.land.temperature)), first(interior(model.land.water_storage)),
                       first(interior(model.land.saturation)), first(interior(interface.fluxes.latent_heat)),
                       first(interior(interface.fluxes.sensible_heat)), first(interior(interface.fluxes.friction_velocity)),
                       first(interior(model.land.fluxes.vapor_flux)),
                       isempty(bad) ? "clean" : "NaN in " * join(bad, ", "))
        isempty(bad) || break
    end
    return nothing
end

probe("bare slab", grid -> bare_slab_model(grid, parameters))
probe("canopy tiles", grid -> borneo_coupled_model(grid, FT, idealized_forcing, parameters; slab_depth = surface_field(grid)))
