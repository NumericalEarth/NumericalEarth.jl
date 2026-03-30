mutable struct PrescribedLand{G, T, F, TI}
    grid :: G
    clock :: Clock{T}
    freshwater_flux :: F   # NamedTuple, e.g. (rivers=FTS, icebergs=FTS)
    times :: TI
end

function Base.summary(pl::PrescribedLand)
    Nx, Ny, Nz = size(pl.grid)
    Nt = length(pl.times)
    sz_str = string(Nx, "×", Ny, "×", Nz, "×", Nt)
    return string(sz_str, " PrescribedLand")
end

function Base.show(io::IO, pl::PrescribedLand)
    print(io, summary(pl), " on ", grid_name(pl.grid), ":", '\n')
    print(io, "├── times: ", prettysummary(pl.times), '\n')
    flux_names = keys(pl.freshwater_flux)
    print(io, "└── freshwater_flux: ", flux_names)
end

"""
    PrescribedLand(freshwater_flux; clock=Clock{Float64}(time=0))

Return a `PrescribedLand` component from a `NamedTuple` of `FieldTimeSeries`
representing freshwater fluxes (e.g. rivers, icebergs).
"""
function PrescribedLand(freshwater_flux; clock=Clock{Float64}(time=0))
    first_flux = first(freshwater_flux)
    grid = first_flux.grid
    times = first_flux.times

    land = PrescribedLand(grid, clock, freshwater_flux, times)
    update_state!(land)
    return land
end

@inline function update_state!(land::PrescribedLand)
    time = Time(land.clock.time)
    ftses = extract_field_time_series(land)

    for fts in ftses
        update_field_time_series!(fts, time)
    end
    return nothing
end

@inline function time_step!(land::PrescribedLand, Δt)
    tick!(land.clock, Δt)
    update_state!(land)
    return nothing
end

# No net fluxes to update for prescribed land
update_net_fluxes!(coupled_model, ::PrescribedLand) = nothing

#####
##### Checkpointing
#####

import Oceananigans: prognostic_state, restore_prognostic_state!

function prognostic_state(land::PrescribedLand)
    return (; clock = prognostic_state(land.clock))
end

function restore_prognostic_state!(land::PrescribedLand, state)
    restore_prognostic_state!(land.clock, state.clock)
    update_state!(land)
    return land
end

restore_prognostic_state!(land::PrescribedLand, ::Nothing) = land
