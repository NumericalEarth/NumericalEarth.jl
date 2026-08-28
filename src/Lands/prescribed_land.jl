# Currently `freshwater_flux` is the only state held here because river runoff
# and iceberg calving are the only land→ocean couplings we use. Atmosphere--land
# coupling would require additional fields (e.g. albedo, skin temperature); see
# https://github.com/NumericalEarth/NumericalEarth.jl/issues/30 for the related
# discussion of moving surface albedo to a radiation component.
mutable struct PrescribedLand{G, T, F, TI, R} <: AbstractPrescribedComponent
    grid :: G
    clock :: Clock{T}
    freshwater_flux :: F     # NamedTuple, e.g. (rivers=FTS, icebergs=FTS)
    times :: TI
    river_routing :: R       # `nothing`, or a `RiverRouting` mapping native river
                             # mouths to coastal ocean cells (see river_routing.jl)
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
    PrescribedLand(freshwater_flux; clock=nothing, river_routing=nothing)

Return a `PrescribedLand` component from a `NamedTuple` of `FieldTimeSeries`
representing freshwater fluxes (e.g. rivers, icebergs).

If `clock` is not provided, defaults to a `Clock` whose time type matches the
element type of `freshwater_flux`.

When `river_routing` is a `NamedTuple` of [`RiverRouting`](@ref) (one per freshwater
component), each component is scattered onto coastal ocean cells conserving volume
(see [`GloFASPrescribedLand`](@ref), [`JRA55PrescribedLand`](@ref)). When it is
`nothing` (the default), the flux is interpolated pointwise onto the ocean grid.
"""
function PrescribedLand(freshwater_flux; clock=nothing, river_routing=nothing)
    first_flux = first(freshwater_flux)
    grid = first_flux.grid
    times = first_flux.times

    if isnothing(clock)
        clock = Clock{eltype(first_flux)}(time=0)
    end

    land = PrescribedLand(grid, clock, freshwater_flux, times, river_routing)
    update_state!(land)
    return land
end

@inline function Oceananigans.TimeSteppers.update_state!(land::PrescribedLand)
    time = Time(land.clock.time)
    ftses = extract_field_time_series(land)

    for fts in ftses
        update_field_time_series!(fts, time)
    end
    return nothing
end

@inline function Oceananigans.TimeSteppers.time_step!(land::PrescribedLand, Δt)
    tick!(land.clock, Δt)
    update_state!(land)
    return nothing
end

# No net fluxes to update for prescribed land; and there's no SlabLand-style
# closure to build an atmosphere-land flux interface from, so the dispatch
# returns nothing for this land type.
EarthSystemModels.update_net_fluxes!(coupled_model, ::PrescribedLand) = nothing
EarthSystemModels.InterfaceComputations.atmosphere_land_interface(grid, atmosphere, land::PrescribedLand; kw...) = nothing

EarthSystemModels.adopt_clock(land::PrescribedLand, clock) = EarthSystemModels.reclock(land, clock)

#####
##### Checkpointing
#####

function Oceananigans.prognostic_state(land::PrescribedLand)
    return (; clock = prognostic_state(land.clock))
end

function Oceananigans.restore_prognostic_state!(land::PrescribedLand, state)
    restore_prognostic_state!(land.clock, state.clock)
    update_state!(land)
    return land
end

Oceananigans.restore_prognostic_state!(land::PrescribedLand, ::Nothing) = land
