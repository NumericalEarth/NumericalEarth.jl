# Where does the compiled primal go NaN? Compile the state reset plus 0, 1 and 2 unrolled
# steps, then the `@trace` loop, for the idealized 0D bare slab and canopy column, and
# report the land state and fluxes each time.

include(joinpath(@__DIR__, "compile_scaling_setup.jl"))

function state_probe(model, h, Δt, nsteps)
    reset!(model, h)
    for _ in 1:nsteps
        time_step!(model, Δt)
    end
    interface = model.interfaces.atmosphere_land_interface
    return (T  = sum(parent(model.land.temperature)),
            M  = sum(parent(model.land.water_storage)),
            𝒮  = sum(parent(model.land.saturation)),
            LE = sum(parent(interface.fluxes.latent_heat)),
            H  = sum(parent(interface.fluxes.sensible_heat)),
            u★ = sum(parent(interface.fluxes.friction_velocity)),
            E  = sum(parent(model.land.fluxes.vapor_flux)),
            P  = sum(parent(model.land.fluxes.liquid_precipitation_flux)),
            Tₐ = sum(parent(model.interfaces.exchanger.atmosphere.state.T)),
            qₐ = sum(parent(model.interfaces.exchanger.atmosphere.state.q)),
            uₐ = sum(parent(model.interfaces.exchanger.atmosphere.state.u)),
            SW = sum(parent(model.interfaces.exchanger.radiation.state.ℐꜜˢʷ)),
            t  = model.clock.time,
            L  = sum((soil_water(model, h) .- θ_target).^2))
end

function traced_probe(model, h, Δt, nsteps)
    L = loss(model, h, Δt, nsteps)
    return (L = L, T = sum(parent(model.land.temperature)), M = sum(parent(model.land.water_storage)), t = model.clock.time)
end

show_values(nt) = join(["$k = $(round(Reactant.to_number(v); sigdigits = 6))" for (k, v) in pairs(nt)], "  ")

for (name, build) in (("bare slab", grid -> bare_slab_model(grid, parameters)),
                      ("canopy tiles", grid -> borneo_coupled_model(grid, FT, idealized_forcing, parameters; slab_depth = surface_field(grid), inner_iterations = 4, similarity_iterations = 4)))
    grid = RectilinearGrid(ReactantState(), FT; size = (), topology = (Flat, Flat, Flat))
    model = build(grid)
    Oceananigans.initialize!(model)
    h = surface_field(grid); parent(h) .= 0.3
    for nsteps in (0, 1, 2)
        out = Reactant.@jit raise=true raise_first=true sync=true state_probe(model, h, Δt, nsteps)
        @info "[$name] unrolled $nsteps steps: " * show_values(out)
    end
    for nsteps in (1, 4)
        out = Reactant.@jit raise=true raise_first=true sync=true traced_probe(model, h, Δt, nsteps)
        @info "[$name] @trace $nsteps steps: " * show_values(out)
    end
end
