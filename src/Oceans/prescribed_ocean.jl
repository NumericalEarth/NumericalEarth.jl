using Oceananigans.TimeSteppers: Clock, tick!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: ConstantField, ZeroField
using Oceananigans.OutputReaders: extract_field_time_series, update_field_time_series!
using Oceananigans.Utils: prettytime, prettysummary

"""
    PrescribedOcean(grid, timeseries;
                    density = 1025.6,
                    heat_capacity = 3995.6,
                    clock = Clock{eltype(grid)}(time=0))

An ocean component for `EarthSystemModel` whose state is prescribed by
`FieldTimeSeries`. At each time step the ocean velocities, temperature,
and salinity are copied from `timeseries` at the current model time,
rather than being computed prognostically.

This is useful for computing surface flux climatologies: pair a
`PrescribedOcean` (with, e.g., ECCO or ERA5 SST) with a
`PrescribedAtmosphere` to diagnose turbulent air-sea fluxes over an
arbitrary period without running a dynamical ocean model.

Arguments
=========

- `grid`: An Oceananigans grid for the ocean domain.

- `timeseries`: A `NamedTuple` of `FieldTimeSeries` providing the
  prescribed ocean state.  Recognised keys are `:u`, `:v`, `:T`, and
  `:S`.  Missing keys default to zero (velocities) or constant
  (salinity = 35) fields.

Keyword Arguments
=================

- `density`: Reference seawater density in kg/m³.  Default: 1025.6.
- `heat_capacity`: Seawater specific heat in J/(kg·K).  Default: 3995.6.
- `clock`: `Clock` for tracking ocean time.
"""
struct PrescribedOcean{FT, G, Clk, U, TR, TS, ρ, C}
    grid :: G
    clock :: Clk
    velocities :: U
    tracers :: TR
    timeseries :: TS
    density :: ρ
    heat_capacity :: C
end

function PrescribedOcean(grid, timeseries;
                         FT = eltype(grid),
                         density = 1025.6,
                         heat_capacity = 3995.6,
                         clock = Clock{FT}(time = 0))

    u = CenterField(grid)
    v = CenterField(grid)
    T = CenterField(grid)
    S = CenterField(grid)

    velocities = (; u, v, w = ZeroField())
    tracers    = (; T, S)

    return PrescribedOcean{FT, typeof(grid), typeof(clock),
                           typeof(velocities), typeof(tracers),
                           typeof(timeseries),
                           typeof(density), typeof(heat_capacity)}(
                               grid, clock, velocities, tracers,
                               timeseries, density, heat_capacity)
end

#####
##### Display
#####

function Base.summary(ocean::PrescribedOcean{FT}) where FT
    A = nameof(typeof(architecture(ocean.grid)))
    G = nameof(typeof(ocean.grid))
    return string("PrescribedOcean{$FT, $A, $G}",
                  "(time = ", prettytime(ocean.clock.time),
                  ", iteration = ", ocean.clock.iteration, ")")
end

function Base.show(io::IO, ocean::PrescribedOcean)
    print(io, summary(ocean), "\n",
          "├── grid: ", summary(ocean.grid), "\n",
          "├── density: ", prettysummary(ocean.density), "\n",
          "├── heat_capacity: ", prettysummary(ocean.heat_capacity), "\n",
          "├── timeseries keys: ", keys(ocean.timeseries), "\n",
          "└── tracers: ", keys(ocean.tracers))
end

Base.eltype(::PrescribedOcean{FT}) where FT = FT

#####
##### EarthSystemModels interface
#####

reference_density(ocean::PrescribedOcean) = ocean.density
heat_capacity(ocean::PrescribedOcean) = ocean.heat_capacity
exchange_grid(atmosphere, ocean::PrescribedOcean, sea_ice) = ocean.grid
temperature_units(::PrescribedOcean) = DegreesCelsius()

ocean_temperature(ocean::PrescribedOcean)         = ocean.tracers.T
ocean_salinity(ocean::PrescribedOcean)             = ocean.tracers.S

ocean_surface_temperature(ocean::PrescribedOcean) = ocean.tracers.T
ocean_surface_salinity(ocean::PrescribedOcean)    = ocean.tracers.S
ocean_surface_velocities(ocean::PrescribedOcean)  = ocean.velocities.u, ocean.velocities.v

#####
##### InterfaceComputations interface
#####

function ComponentExchanger(ocean::PrescribedOcean, exchange_grid)
    u = ocean.velocities.u
    v = ocean.velocities.v
    T = ocean.tracers.T
    S = ocean.tracers.S
    return ComponentExchanger((; u, v, T, S), nothing)
end

net_fluxes(ocean::PrescribedOcean) = nothing

interpolate_state!(exchanger, grid, ::PrescribedOcean, coupled_model) = nothing

update_net_fluxes!(coupled_model, ocean::PrescribedOcean) = nothing

#####
##### Time stepping — copy prescribed data into model fields
#####

function Oceananigans.TimeSteppers.time_step!(ocean::PrescribedOcean, Δt;
                                              callbacks = [], euler = true)
    tick!(ocean.clock, Δt)
    time = Time(ocean.clock.time)

    # Update and copy from any FieldTimeSeries in the timeseries NamedTuple
    ts = ocean.timeseries

    if length(ts) > 0
        for fts in extract_field_time_series(ts)
            update_field_time_series!(fts, time)
        end

        haskey(ts, :u) && parent(ocean.velocities.u) .= parent(ts.u[time])
        haskey(ts, :v) && parent(ocean.velocities.v) .= parent(ts.v[time])
        haskey(ts, :T) && parent(ocean.tracers.T)    .= parent(ts.T[time])
        haskey(ts, :S) && parent(ocean.tracers.S)    .= parent(ts.S[time])
    end

    return nothing
end

Oceananigans.TimeSteppers.update_state!(::PrescribedOcean) = nothing
Oceananigans.Simulations.timestepper(::PrescribedOcean) = nothing

# Guard: OceanOnlyModel adds FreezingLimitedOceanTemperature which
# assumes ocean.model — use AtmosphereOceanModel instead.
import NumericalEarth.EarthSystemModels: OceanOnlyModel
function OceanOnlyModel(ocean::PrescribedOcean; kw...)
    throw(ArgumentError(
        "OceanOnlyModel cannot be used with PrescribedOcean. " *
        "Use `AtmosphereOceanModel(atmosphere, ocean; ...)` instead."))
end
