#####
##### Prescribed atmosphere (as opposed to dynamically evolving / prognostic)
#####

mutable struct PrescribedAtmosphere{FT, G, T, U, P, C, F, TP, TI}
    grid :: G
    clock :: Clock{T}
    velocities :: U
    pressure :: P
    tracers :: C
    freshwater_flux :: F
    thermodynamics_parameters :: TP
    times :: TI
    surface_layer_height :: FT
    boundary_layer_height :: FT
end

function Base.summary(pa::PrescribedAtmosphere{FT}) where FT
    Nx, Ny, Nz = size(pa.grid)
    Nt = length(pa.times)
    sz_str = string(Nx, "×", Ny, "×", Nz, "×", Nt)
    return string(sz_str, " PrescribedAtmosphere{$FT}")
end

function Base.show(io::IO, pa::PrescribedAtmosphere)
    print(io, summary(pa), " on ", grid_name(pa.grid), ":", '\n')
    print(io, "├── times: ", prettysummary(pa.times), '\n')
    print(io, "├── surface_layer_height: ", prettysummary(pa.surface_layer_height), '\n')
    print(io, "└── boundary_layer_height: ", prettysummary(pa.boundary_layer_height))
end

function default_atmosphere_velocities(grid, times)
    ua = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    va = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    return (u=ua, v=va)
end

function default_atmosphere_tracers(grid, times)
    Ta = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    qa = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    parent(Ta) .= 273.15 + 20
    return (T=Ta, q=qa)
end

"""
    PrescribedPrecipitationFlux(; rain=nothing, snow=nothing)
    PrescribedPrecipitationFlux(rain, snow)

Container for prescribed precipitation fluxes. Either component may be `nothing`
to indicate that the corresponding precipitation type is not represented by the
atmosphere (e.g. rain-only datasets). Used as the `freshwater_flux` of a
`PrescribedAtmosphere`; downstream callers query the snow component via
[`surface_snowfall_flux`](@ref) so that prognostic atmospheres with or without
snow can dispatch on this type as well.
"""
struct PrescribedPrecipitationFlux{R, S}
    rain :: R
    snow :: S
end

PrescribedPrecipitationFlux(; rain=nothing, snow=nothing) =
    PrescribedPrecipitationFlux(rain, snow)

Adapt.adapt_structure(to, ff::PrescribedPrecipitationFlux) =
    PrescribedPrecipitationFlux(adapt(to, ff.rain), adapt(to, ff.snow))

function default_freshwater_flux(grid, times)
    rain = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    snow = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    return PrescribedPrecipitationFlux(rain, snow)
end

@inline field_data(::Nothing) = nothing
@inline field_data(field) = field.data

@inline surface_snowfall_flux(::Nothing) = nothing
@inline surface_snowfall_flux(atmos::PrescribedAtmosphere) = surface_snowfall_flux(atmos.freshwater_flux)
@inline surface_snowfall_flux(ff::PrescribedPrecipitationFlux) = field_data(ff.snow)

@inline surface_rainfall_flux(::Nothing) = nothing
@inline surface_rainfall_flux(atmos::PrescribedAtmosphere) = surface_rainfall_flux(atmos.freshwater_flux)
@inline surface_rainfall_flux(ff::PrescribedPrecipitationFlux) = field_data(ff.rain)

""" The standard unit of atmospheric pressure; 1 standard atmosphere (atm) = 101,325 Pascals (Pa)
in SI units. This is approximately equal to the mean sea-level atmospheric pressure on Earth. """
function default_atmosphere_pressure(grid, times)
    pa = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    parent(pa) .= 101325
    return pa
end

@inline function update_state!(atmos::PrescribedAtmosphere)
    time = Time(atmos.clock.time)
    ftses = extract_field_time_series(atmos)

    for fts in ftses
        update_field_time_series!(fts, time)
    end
    return nothing
end

@inline function time_step!(atmos::PrescribedAtmosphere, Δt)
    tick!(atmos.clock, Δt)

    update_state!(atmos)

    return nothing
end

@inline thermodynamics_parameters(atmos::Nothing) = nothing
@inline thermodynamics_parameters(atmos::PrescribedAtmosphere) = atmos.thermodynamics_parameters
@inline surface_layer_height(atmos::PrescribedAtmosphere) = atmos.surface_layer_height
@inline boundary_layer_height(atmos::PrescribedAtmosphere) = atmos.boundary_layer_height

# No need to compute anything here...
update_net_fluxes!(coupled_model, ::PrescribedAtmosphere) = nothing

"""
    PrescribedAtmosphere(grid, times=[zero(grid)];
                         clock = Clock{Float64}(time = 0),
                         surface_layer_height = 10, # meters
                         boundary_layer_height = 512, # meters
                         thermodynamics_parameters = AtmosphereThermodynamicsParameters(eltype(grid)),
                         velocities      = default_atmosphere_velocities(grid, times),
                         tracers         = default_atmosphere_tracers(grid, times),
                         pressure        = default_atmosphere_pressure(grid, times),
                         freshwater_flux = default_freshwater_flux(grid, times))

Return a representation of a prescribed time-evolving atmospheric
state with data given at `times`.

Note: downwelling shortwave / longwave radiation is now part of the
top-level `radiation` component (see `PrescribedRadiation`,
`JRA55PrescribedRadiation`).
"""
function PrescribedAtmosphere(grid, times=[zero(grid)];
                              clock = Clock{Float64}(time = 0),
                              surface_layer_height = 10,
                              boundary_layer_height = 512,
                              thermodynamics_parameters = AtmosphereThermodynamicsParameters(eltype(grid)),
                              velocities      = default_atmosphere_velocities(grid, times),
                              tracers         = default_atmosphere_tracers(grid, times),
                              pressure        = default_atmosphere_pressure(grid, times),
                              freshwater_flux = default_freshwater_flux(grid, times))

    FT = eltype(grid)
    if isnothing(thermodynamics_parameters)
        thermodynamics_parameters = AtmosphereThermodynamicsParameters(FT)
    end

    atmosphere = PrescribedAtmosphere(grid,
                                      clock,
                                      velocities,
                                      pressure,
                                      tracers,
                                      freshwater_flux,
                                      thermodynamics_parameters,
                                      times,
                                      convert(FT, surface_layer_height),
                                      convert(FT, boundary_layer_height))
    update_state!(atmosphere)

    return atmosphere
end

#####
##### Chekpointing
#####

import Oceananigans: prognostic_state, restore_prognostic_state!

function prognostic_state(atmos::PrescribedAtmosphere)
    return (; clock = prognostic_state(atmos.clock))
end

function restore_prognostic_state!(atmos::PrescribedAtmosphere, state)
    restore_prognostic_state!(atmos.clock, state.clock)
    update_state!(atmos)
    return atmos
end

restore_prognostic_state!(atmos::PrescribedAtmosphere, ::Nothing) = atmos
