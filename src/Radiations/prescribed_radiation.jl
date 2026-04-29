"""
    PrescribedRadiation{G, T, FT, SW, LW, S, IF, TI}

Top-level radiation component holding prescribed downwelling shortwave and
longwave radiation as `FieldTimeSeries`, plus per-surface radiative properties
(albedo, emissivity) and the Stefan–Boltzmann constant. Diagnostic radiative
fluxes (one `InterfaceRadiationFlux` per surface) are populated by the
`apply_air_sea_*_radiative_fluxes!` kernels at every step; `interface_fluxes`
is `nothing` until the radiation is paired with an `EarthSystemModel` (which
allocates the per-surface buffers on the exchange grid).
"""
mutable struct PrescribedRadiation{G, T, FT, SW, LW, S, TI}
    grid :: G
    clock :: Clock{T}
    downwelling_shortwave :: SW
    downwelling_longwave :: LW
    surface_properties :: S
    stefan_boltzmann_constant :: FT
    # NamedTuple of `InterfaceRadiationFlux`, allocated at `EarthSystemModel`
    # construction time once the exchange grid is known. Untyped so the field
    # can be reassigned from `nothing` to a populated NamedTuple.
    interface_fluxes
    times :: TI
end

function Base.summary(r::PrescribedRadiation)
    Nx, Ny, Nz = size(r.grid)
    Nt = length(r.times)
    sz_str = string(Nx, "×", Ny, "×", Nz, "×", Nt)
    return string(sz_str, " PrescribedRadiation")
end

function Base.show(io::IO, r::PrescribedRadiation)
    print(io, summary(r), " on ", grid_name(r.grid), ":", '\n')
    print(io, "├── times: ", prettysummary(r.times), '\n')
    print(io, "├── stefan_boltzmann_constant: ", prettysummary(r.stefan_boltzmann_constant), '\n')
    print(io, "└── surface_properties: ", keys(r.surface_properties))
end

# Filter out surfaces whose property kwarg was passed as `nothing`.
@inline function _filter_surface_properties(; ocean=nothing, sea_ice=nothing, snow=nothing, land=nothing)
    pairs = ()
    isnothing(ocean)   || (pairs = (pairs..., :ocean   => ocean))
    isnothing(sea_ice) || (pairs = (pairs..., :sea_ice => sea_ice))
    isnothing(snow)    || (pairs = (pairs..., :snow    => snow))
    isnothing(land)    || (pairs = (pairs..., :land    => land))
    return NamedTuple(pairs)
end

"""
    PrescribedRadiation(downwelling_shortwave, downwelling_longwave;
                        clock = nothing,
                        ocean_surface = SurfaceRadiationProperties(0.05, 0.97),
                        sea_ice_surface = SurfaceRadiationProperties(0.7, 1.0),
                        snow_surface = nothing,
                        land_surface = nothing,
                        stefan_boltzmann_constant = 5.67e-8)

Construct a `PrescribedRadiation` component from `FieldTimeSeries` of
downwelling shortwave and longwave radiation. Grid + times are inferred from
the shortwave FTS.

Pass `*_surface = nothing` to omit that surface from `surface_properties`.
"""
function PrescribedRadiation(downwelling_shortwave::FieldTimeSeries,
                             downwelling_longwave::FieldTimeSeries;
                             clock = nothing,
                             ocean_surface = SurfaceRadiationProperties(0.05, 0.97),
                             sea_ice_surface = SurfaceRadiationProperties(0.7, 1.0),
                             snow_surface = nothing,
                             land_surface = nothing,
                             stefan_boltzmann_constant = 5.67e-8)

    grid  = downwelling_shortwave.grid
    times = downwelling_shortwave.times
    FT    = eltype(downwelling_shortwave)

    if isnothing(clock)
        clock = Clock{FT}(time = 0)
    end

    surface_properties = _filter_surface_properties(ocean = ocean_surface,
                                                    sea_ice = sea_ice_surface,
                                                    snow = snow_surface,
                                                    land = land_surface)

    radiation = PrescribedRadiation(grid,
                                    clock,
                                    downwelling_shortwave,
                                    downwelling_longwave,
                                    surface_properties,
                                    convert(FT, stefan_boltzmann_constant),
                                    nothing, # interface_fluxes — populated at ESM construction
                                    times)
    update_state!(radiation)
    return radiation
end

"""
    PrescribedRadiation(grid, times = [zero(grid)]; kwargs...)

Construct a `PrescribedRadiation` with zero downwelling shortwave and longwave
fields on `grid`. Useful for emission-only mode (the surface radiates ϵσT⁴ but
no incoming radiation is absorbed). All other keyword arguments are forwarded
to the FTS-form constructor.
"""
function PrescribedRadiation(grid, times = [zero(eltype(grid))]; kwargs...)
    sw = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    lw = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    return PrescribedRadiation(sw, lw; kwargs...)
end

@inline function update_state!(radiation::PrescribedRadiation)
    time = Time(radiation.clock.time)
    ftses = extract_field_time_series(radiation)

    for fts in ftses
        update_field_time_series!(fts, time)
    end
    return nothing
end

@inline function time_step!(radiation::PrescribedRadiation, Δt)
    tick!(radiation.clock, Δt)
    update_state!(radiation)
    return nothing
end

# Prescribed radiation has no internal state to update from net fluxes.
update_net_fluxes!(coupled_model, ::PrescribedRadiation) = nothing

"""
    allocate_interface_fluxes!(radiation, exchange_grid, surfaces)

Populate `radiation.interface_fluxes` with one `InterfaceRadiationFlux`
per surface present in the model.
"""
function allocate_interface_fluxes!(radiation::PrescribedRadiation, exchange_grid, surfaces)
    pairs = (surface => InterfaceRadiationFlux(exchange_grid) for surface in surfaces)
    radiation.interface_fluxes = NamedTuple(pairs)
    return nothing
end

#####
##### Checkpointing
#####

import Oceananigans: prognostic_state, restore_prognostic_state!

function prognostic_state(radiation::PrescribedRadiation)
    return (; clock = prognostic_state(radiation.clock))
end

function restore_prognostic_state!(radiation::PrescribedRadiation, state)
    restore_prognostic_state!(radiation.clock, state.clock)
    update_state!(radiation)
    return radiation
end

restore_prognostic_state!(radiation::PrescribedRadiation, ::Nothing) = radiation
