#####
##### Prescribed atmosphere (as opposed to dynamically evolving / prognostic)
#####

mutable struct PrescribedAtmosphere{S, FT, G, T, U, Θ, Q, M, P, C, F, TP, TI} <: AbstractPrescribedComponent
    source :: S
    grid :: G
    clock :: Clock{T}
    velocities :: U
    temperature :: Θ
    specific_humidity :: Q
    microphysical_variables :: M
    pressure :: P
    tracers :: C
    precipitation_flux :: F
    thermodynamics_parameters :: TP
    times :: TI
    surface_layer_height :: FT
    boundary_layer_height :: FT
end

function Base.summary(atmos::PrescribedAtmosphere{<:Any, FT}) where FT
    Nx, Ny, Nz = size(atmos.grid)
    Nt = length(atmos.times)
    sz_str = string(Nx, "×", Ny, "×", Nz, "×", Nt)
    return string(sz_str, " PrescribedAtmosphere{$FT}")
end

function Base.show(io::IO, atmos::PrescribedAtmosphere)
    print(io, summary(atmos), " on ", grid_name(atmos.grid), ":", '\n')
    print(io, "├── times: ", prettysummary(atmos.times), '\n')
    print(io, "├── surface_layer_height: ", prettysummary(atmos.surface_layer_height), '\n')
    print(io, "└── boundary_layer_height: ", prettysummary(atmos.boundary_layer_height))
end

velocity_boundary_conditions(grid, loc) = FieldBoundaryConditions(grid, loc)

function velocity_boundary_conditions(grid::OrthogonalSphericalShellGrids.TripolarGrid, loc)
    north_boundary_condition = OrthogonalSphericalShellGrids.north_fold_boundary_condition(grid)(-1)
    return FieldBoundaryConditions(grid, loc; north = north_boundary_condition)
end

# Surface (2D, z-`Nothing`) vs 3D (z-`Center`) defaults are inferred from the grid's vertical size:
# `Nz == 1` ⇒ surface forcing (ocean / sea-ice coupling); `Nz > 1` ⇒ a 3D atmosphere (e.g. a
# `NestedSimulation` parent). A single-level grid — even one with a `Bounded` z, like an ocean
# coupling grid — is treated as surface. Dataset builders pass their own fields explicitly and so are
# unaffected; only the `default_*` paths consult this. Override any field via kwarg.
@inline is_three_dimensional(grid) = size(grid, 3) > 1

function default_atmosphere_velocities(grid, times)
    # The horizontal velocity boundary conditions carry the tripolar north-fold sign flip, a property
    # of the *horizontal* grid — so `u`/`v` need them whether the atmosphere is 2D or 3D. `w` keeps the
    # grid's default (scalar fold). On non-tripolar grids these are the plain default BCs.
    if is_three_dimensional(grid)
        boundary_conditions = velocity_boundary_conditions(grid, (Center(), Center(), Center()))
        u = FieldTimeSeries{Center, Center, Center}(grid, times; boundary_conditions)
        v = FieldTimeSeries{Center, Center, Center}(grid, times; boundary_conditions)
        w = FieldTimeSeries{Center, Center, Center}(grid, times)
        return (; u, v, w)
    else
        boundary_conditions = velocity_boundary_conditions(grid, (Center(), Center(), nothing))
        u = FieldTimeSeries{Center, Center, Nothing}(grid, times; boundary_conditions)
        v = FieldTimeSeries{Center, Center, Nothing}(grid, times; boundary_conditions)
        return (; u, v)
    end
end

function default_atmosphere_temperature(grid, times)
    LZ = is_three_dimensional(grid) ? Center : Nothing
    T = FieldTimeSeries{Center, Center, LZ}(grid, times)
    parent(T) .= 273.15 + 20   # a sane uniform placeholder (real atmospheres set this explicitly)
    return T
end

function default_atmosphere_specific_humidity(grid, times)
    LZ = is_three_dimensional(grid) ? Center : Nothing
    return FieldTimeSeries{Center, Center, LZ}(grid, times)
end

# Cloud / precipitation species: none by default. A 3D parent (e.g. from ERA5 pressure-level data)
# populates this with a `NamedTuple` such as `(; qᶜˡ, qʳ, qᶜⁱ, qˢ)`.
default_atmosphere_microphysical_variables(grid, times) = NamedTuple()

"""
    PrescribedPrecipitationFlux(; rain=nothing, snow=nothing)
    PrescribedPrecipitationFlux(rain, snow)

Container for prescribed precipitation fluxes. Either component may be `nothing`
to indicate that the corresponding precipitation type is not represented by the
atmosphere (e.g. rain-only datasets). Used as the `precipitation_flux` of a
`PrescribedAtmosphere`; downstream callers query the snow component via
`surface_snowfall_flux` so that prognostic atmospheres with or without
snow can dispatch on this type as well.
"""
struct PrescribedPrecipitationFlux{R, S}
    rain :: R
    snow :: S
end

PrescribedPrecipitationFlux(; rain=nothing, snow=nothing) =
    PrescribedPrecipitationFlux(rain, snow)

Adapt.adapt_structure(to, precipitation_flux::PrescribedPrecipitationFlux) =
    PrescribedPrecipitationFlux(adapt(to, precipitation_flux.rain), adapt(to, precipitation_flux.snow))

# A surface (Flat-z) atmosphere defaults to a 2D precipitation flux for ocean / sea-ice / land
# coupling; a 3D atmosphere carries precipitation in `microphysical_variables` instead,
# so it has none by default (pass `precipitation_flux` explicitly to add one).
function default_precipitation_flux(grid, times)
    is_three_dimensional(grid) && return nothing
    rain = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    snow = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    return PrescribedPrecipitationFlux(rain, snow)
end

@inline field_data(::Nothing) = nothing
@inline field_data(field) = field.data

@inline surface_snowfall_flux(::Nothing) = nothing
@inline surface_snowfall_flux(atmos::PrescribedAtmosphere) = surface_snowfall_flux(atmos.precipitation_flux)
@inline surface_snowfall_flux(precipitation_flux::PrescribedPrecipitationFlux) = field_data(precipitation_flux.snow)

@inline surface_rainfall_flux(::Nothing) = nothing
@inline surface_rainfall_flux(atmos::PrescribedAtmosphere) = surface_rainfall_flux(atmos.precipitation_flux)
@inline surface_rainfall_flux(precipitation_flux::PrescribedPrecipitationFlux) = field_data(precipitation_flux.rain)

""" The standard unit of atmospheric pressure; 1 standard atmosphere (atm) = 101,325 Pascals (Pa)
in SI units. This is approximately equal to the mean sea-level atmospheric pressure on Earth. """
function default_atmosphere_pressure(grid, times)
    LZ = is_three_dimensional(grid) ? Center : Nothing
    p = FieldTimeSeries{Center, Center, LZ}(grid, times)
    parent(p) .= 101325   # a sane uniform placeholder (real atmospheres set this explicitly)
    return p
end

@inline function Oceananigans.TimeSteppers.update_state!(atmos::PrescribedAtmosphere)
    time = Time(atmos.clock.time)
    ftses = extract_field_time_series(atmos)

    for fts in ftses
        update_field_time_series!(fts, time)
    end
    return nothing
end

@inline function Oceananigans.TimeSteppers.time_step!(atmos::PrescribedAtmosphere, Δt)
    tick!(atmos.clock, Δt)

    update_state!(atmos)

    return nothing
end

@inline EarthSystemModels.thermodynamics_parameters(atmos::Nothing) = nothing
@inline EarthSystemModels.thermodynamics_parameters(atmos::PrescribedAtmosphere) = atmos.thermodynamics_parameters
@inline EarthSystemModels.surface_layer_height(atmos::PrescribedAtmosphere) = atmos.surface_layer_height
@inline EarthSystemModels.boundary_layer_height(atmos::PrescribedAtmosphere) = atmos.boundary_layer_height

# No need to compute anything here...
EarthSystemModels.update_net_fluxes!(coupled_model, ::PrescribedAtmosphere) = nothing

EarthSystemModels.adopt_clock(atmos::PrescribedAtmosphere, clock) = EarthSystemModels.reclock(atmos, clock)

"""
    PrescribedAtmosphere(grid, times=[zero(grid)];
                         source = nothing,
                         clock = Clock{Float64}(time = 0),
                         surface_layer_height = 10, # meters
                         boundary_layer_height = 512, # meters
                         thermodynamics_parameters = AtmosphereThermodynamicsParameters(eltype(grid)),
                         velocities              = default_atmosphere_velocities(grid, times),
                         temperature             = default_atmosphere_temperature(grid, times),
                         specific_humidity       = default_atmosphere_specific_humidity(grid, times),
                         microphysical_variables = default_atmosphere_microphysical_variables(grid, times),
                         pressure                = default_atmosphere_pressure(grid, times),
                         tracers                 = NamedTuple(),
                         precipitation_flux      = default_precipitation_flux(grid, times))

Return a prescribed, time-evolving atmospheric state with data on `grid` at `times`.

The state holds `velocities`, `temperature`, `specific_humidity`, an optional
`microphysical_variables` `NamedTuple` (cloud / precipitation species such as `(; qᶜˡ, qʳ, qᶜⁱ, qˢ)`,
empty by default), `pressure`, gas-species `tracers` (e.g. CO₂; empty by default), and — for a
surface atmosphere — a `precipitation_flux` (a 3D atmosphere defaults to none, carrying precipitation
in `microphysical_variables` instead).

`source` records what the atmosphere was built from (a dataset object, e.g. `ERA5HourlyPressureLevels()`;
`nothing` for a hand-built atmosphere). Dataset constructors set it so that aliases like
`ERA5PrescribedAtmosphere = PrescribedAtmosphere{<:ERA5Dataset}` dispatch on provenance.

Surface (2D, `(Center, Center, Nothing)`) vs 3D (`(Center, Center, Center)`, adding `w`)
default fields are inferred from the grid's vertical size: `Nz == 1` builds a surface atmosphere
(ocean / sea-ice coupling), `Nz > 1` a 3D atmosphere (e.g. a [`NestedSimulation`](@ref) parent).
Pass any field explicitly to override; pass `precipitation_flux = nothing` to omit it.

!!! compat "Radiation component"
    The downwelling shortwave / longwave radiation part of the top-level `radiation`
    component (see [`Radiations.PrescribedRadiation`](@ref NumericalEarth.Radiations.PrescribedRadiation),
    [`DataWrangling.JRA55.JRA55PrescribedRadiation`](@ref NumericalEarth.DataWrangling.JRA55.JRA55PrescribedRadiation)).
"""
function PrescribedAtmosphere(grid, times=[zero(grid)];
                              source = nothing,
                              clock = Clock{Float64}(time = 0),
                              surface_layer_height = 10,
                              boundary_layer_height = 512,
                              thermodynamics_parameters = AtmosphereThermodynamicsParameters(eltype(grid)),
                              velocities              = default_atmosphere_velocities(grid, times),
                              temperature             = default_atmosphere_temperature(grid, times),
                              specific_humidity       = default_atmosphere_specific_humidity(grid, times),
                              microphysical_variables = default_atmosphere_microphysical_variables(grid, times),
                              pressure                = default_atmosphere_pressure(grid, times),
                              tracers                 = NamedTuple(),
                              precipitation_flux      = default_precipitation_flux(grid, times))

    FT = eltype(grid)
    if isnothing(thermodynamics_parameters)
        thermodynamics_parameters = AtmosphereThermodynamicsParameters(FT)
    end

    atmos = PrescribedAtmosphere(source,
                                 grid,
                                 clock,
                                 velocities,
                                 temperature,
                                 specific_humidity,
                                 microphysical_variables,
                                 pressure,
                                 tracers,
                                 precipitation_flux,
                                 thermodynamics_parameters,
                                 times,
                                 convert(FT, surface_layer_height),
                                 convert(FT, boundary_layer_height))
    update_state!(atmos)

    return atmos
end

#####
##### Chekpointing
#####

function Oceananigans.prognostic_state(atmos::PrescribedAtmosphere)
    return (; clock = prognostic_state(atmos.clock))
end

function Oceananigans.restore_prognostic_state!(atmos::PrescribedAtmosphere, state)
    restore_prognostic_state!(atmos.clock, state.clock)
    update_state!(atmos)
    return atmos
end

Oceananigans.restore_prognostic_state!(atmos::PrescribedAtmosphere, ::Nothing) = atmos
