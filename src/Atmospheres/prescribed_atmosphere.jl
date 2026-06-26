#####
##### Prescribed atmosphere (as opposed to dynamically evolving / prognostic)
#####

mutable struct PrescribedAtmosphere{FT, G, T, U, Θ, Q, M, P, C, F, TP, TI} <: AbstractPrescribedComponent
    grid :: G
    clock :: Clock{T}
    velocities :: U
    temperature :: Θ
    specific_humidity :: Q
    microphysical_variables :: M
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

velocity_boundary_conditions(grid, loc) = FieldBoundaryConditions(grid, loc)

function velocity_boundary_conditions(grid::OrthogonalSphericalShellGrids.TripolarGrid, loc)
    north_bc = OrthogonalSphericalShellGrids.north_fold_boundary_condition(grid)(-1)
    return FieldBoundaryConditions(grid, loc; north = north_bc)
end

# Surface (2D, z-`Nothing`) vs volumetric (3D, z-`Center`) defaults are inferred from the grid:
# a `Flat` vertical ⇒ surface forcing (ocean / sea-ice coupling); a resolved vertical ⇒ a 3D
# atmosphere (e.g. a `NestedSimulation` parent). Dataset builders pass their own fields explicitly
# and so are unaffected; only the `default_*` paths consult this. Override any field via kwarg.
@inline volumetric_atmosphere(grid) = topology(grid, 3) !== Flat

function default_atmosphere_velocities(grid, times)
    if volumetric_atmosphere(grid)
        ua = FieldTimeSeries{Center, Center, Center}(grid, times)
        va = FieldTimeSeries{Center, Center, Center}(grid, times)
        wa = FieldTimeSeries{Center, Center, Center}(grid, times)
        return (u=ua, v=va, w=wa)
    else
        velocity_bcs = velocity_boundary_conditions(grid, (Center(), Center(), nothing))
        ua = FieldTimeSeries{Center, Center, Nothing}(grid, times; boundary_conditions = velocity_bcs)
        va = FieldTimeSeries{Center, Center, Nothing}(grid, times; boundary_conditions = velocity_bcs)
        return (u=ua, v=va)
    end
end

function default_atmosphere_temperature(grid, times)
    if volumetric_atmosphere(grid)
        return FieldTimeSeries{Center, Center, Center}(grid, times)
    else
        Ta = FieldTimeSeries{Center, Center, Nothing}(grid, times)
        parent(Ta) .= 273.15 + 20
        return Ta
    end
end

function default_atmosphere_specific_humidity(grid, times)
    LZ = volumetric_atmosphere(grid) ? Center : Nothing
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
atmosphere (e.g. rain-only datasets). Used as the `freshwater_flux` of a
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

Adapt.adapt_structure(to, ff::PrescribedPrecipitationFlux) =
    PrescribedPrecipitationFlux(adapt(to, ff.rain), adapt(to, ff.snow))

# Freshwater flux is 2D regardless of the atmosphere's dimensionality
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
    if volumetric_atmosphere(grid)
        return FieldTimeSeries{Center, Center, Center}(grid, times)
    else
        pa = FieldTimeSeries{Center, Center, Nothing}(grid, times)
        parent(pa) .= 101325
        return pa
    end
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

EarthSystemModels.adopt_clock(atmosphere::PrescribedAtmosphere, clock) = EarthSystemModels.reclock(atmosphere, clock)

"""
    PrescribedAtmosphere(grid, times=[zero(grid)];
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
                         freshwater_flux         = default_freshwater_flux(grid, times))

Return a prescribed, time-evolving atmospheric state with data on `grid` at `times`.

The state holds `velocities`, `temperature`, `specific_humidity`, an optional
`microphysical_variables` `NamedTuple` (cloud / precipitation species such as `(; qᶜˡ, qʳ, qᶜⁱ, qˢ)`,
empty by default), `pressure`, gas-species `tracers` (e.g. CO₂; empty by default), and a surface
`freshwater_flux`.

Surface (2D, `(Center, Center, Nothing)`) vs volumetric (3D, `(Center, Center, Center)`, adding `w`)
default fields are inferred from the grid: a `Flat` vertical builds a surface atmosphere (ocean /
sea-ice coupling), a resolved vertical a 3D atmosphere (e.g. a [`NestedSimulation`](@ref) parent).
Pass any field explicitly to override; pass `freshwater_flux = nothing` to omit it.

!!! compat "Radiation component"
    The downwelling shortwave / longwave radiation part of the top-level `radiation`
    component (see [`Radiations.PrescribedRadiation`](@ref NumericalEarth.Radiations.PrescribedRadiation),
    [`DataWrangling.JRA55.JRA55PrescribedRadiation`](@ref NumericalEarth.DataWrangling.JRA55.JRA55PrescribedRadiation)).
"""
function PrescribedAtmosphere(grid, times=[zero(grid)];
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
                              freshwater_flux         = default_freshwater_flux(grid, times))

    FT = eltype(grid)
    if isnothing(thermodynamics_parameters)
        thermodynamics_parameters = AtmosphereThermodynamicsParameters(FT)
    end

    atmosphere = PrescribedAtmosphere(grid,
                                      clock,
                                      velocities,
                                      temperature,
                                      specific_humidity,
                                      microphysical_variables,
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

function Oceananigans.prognostic_state(atmos::PrescribedAtmosphere)
    return (; clock = prognostic_state(atmos.clock))
end

function Oceananigans.restore_prognostic_state!(atmos::PrescribedAtmosphere, state)
    restore_prognostic_state!(atmos.clock, state.clock)
    update_state!(atmos)
    return atmos
end

Oceananigans.restore_prognostic_state!(atmos::PrescribedAtmosphere, ::Nothing) = atmos
