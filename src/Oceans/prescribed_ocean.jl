using Oceananigans.OutputReaders: update_field_time_series!, FieldTimeSeries

"""
    PrescribedOcean(grid, times=[zero(grid)];
                    density = 1025,
                    heat_capacity = 4000,
                    clock = Clock{FT}(time=0),
                    sea_surface_temperature = default_prescribed_sst(grid, times),
                    sea_surface_salinity = default_prescribed_sss(grid, times),
                    velocities = default_prescribed_velocities(grid, times))

A prescribed ocean component for `EarthSystemModel` with sea surface
temperature, salinity, and velocities prescribed as `FieldTimeSeries`.

The ocean state does not evolve in response to surface fluxes — it follows
the prescribed data. Surface fluxes are still computed so the atmosphere
feels the ocean.

Arguments
=========

- `grid`: An Oceananigans grid for the ocean domain.
- `times`: Time instances for the prescribed data. Default: `[zero(grid)]` (constant).

Keyword Arguments
=================

- `density`: Seawater density in kg/m³. Default: 1025.
- `heat_capacity`: Seawater specific heat capacity in J/(kg·K). Default: 4000.
- `clock`: Clock for tracking ocean time. Default: `Clock{FT}(time=0)`.
- `sea_surface_temperature`: `FieldTimeSeries` for SST.
- `sea_surface_salinity`: `FieldTimeSeries` for SSS.
- `velocities`: `NamedTuple` of `FieldTimeSeries` for `(u, v)`.
"""
struct PrescribedOcean{FT, G, Clk, SST, SSS, U, TI, F, ρ, C}
    grid :: G
    clock :: Clk
    sea_surface_temperature :: SST
    sea_surface_salinity :: SSS
    velocities :: U
    times :: TI
    temperature_flux :: F
    density :: ρ
    heat_capacity :: C
end

function default_prescribed_sst(grid, times)
    return FieldTimeSeries{Center, Center, Nothing}(grid, times)
end

function default_prescribed_sss(grid, times)
    sss = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    parent(sss) .= 35
    return sss
end

function default_prescribed_velocities(grid, times)
    u = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    v = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    return (; u, v)
end

function PrescribedOcean(grid, times=[zero(grid)];
                         FT = eltype(grid),
                         density = 1025,
                         heat_capacity = 4000,
                         clock = Clock{FT}(time=0),
                         sea_surface_temperature = default_prescribed_sst(grid, times),
                         sea_surface_salinity = default_prescribed_sss(grid, times),
                         velocities = default_prescribed_velocities(grid, times))

    temperature_flux = CenterField(grid)

    return PrescribedOcean{FT}(grid, clock,
                               sea_surface_temperature,
                               sea_surface_salinity,
                               velocities, times,
                               temperature_flux,
                               convert(FT, density),
                               convert(FT, heat_capacity))
end

PrescribedOcean{FT}(grid, clock, sst, sss, vel, times, tf, ρ, c) where FT =
    PrescribedOcean{FT, typeof(grid), typeof(clock), typeof(sst), typeof(sss),
                    typeof(vel), typeof(times), typeof(tf), typeof(ρ), typeof(c)}(
                    grid, clock, sst, sss, vel, times, tf, ρ, c)

function Oceananigans.set!(ocean::PrescribedOcean; T=nothing, S=nothing, u=nothing, v=nothing)
    !isnothing(T) && (parent(ocean.sea_surface_temperature) .= T)
    !isnothing(S) && (parent(ocean.sea_surface_salinity) .= S)
    !isnothing(u) && (parent(ocean.velocities.u) .= u)
    !isnothing(v) && (parent(ocean.velocities.v) .= v)
    return nothing
end

function Base.summary(ocean::PrescribedOcean{FT}) where FT
    A = nameof(typeof(architecture(ocean.grid)))
    G = nameof(typeof(ocean.grid))
    Nt = length(ocean.times)
    return string("PrescribedOcean{$FT, $A, $G}",
                  "(Nt = ", Nt, ")")
end

function Base.show(io::IO, ocean::PrescribedOcean)
    print(io, summary(ocean), "\n",
          "├── grid: ", summary(ocean.grid), "\n",
          "├── times: ", prettysummary(ocean.times), "\n",
          "├── density: ", prettysummary(ocean.density), "\n",
          "└── heat_capacity: ", prettysummary(ocean.heat_capacity))
end

Base.eltype(::PrescribedOcean{FT}) where FT = FT

#####
##### EarthSystemModels interface
#####

reference_density(ocean::PrescribedOcean) = ocean.density
heat_capacity(ocean::PrescribedOcean) = ocean.heat_capacity
exchange_grid(ocean::PrescribedOcean) = ocean.grid
temperature_units(::PrescribedOcean) = DegreesKelvin()

ocean_temperature(ocean::PrescribedOcean) = ocean.sea_surface_temperature
ocean_salinity(ocean::PrescribedOcean) = ocean.sea_surface_salinity
ocean_surface_temperature(ocean::PrescribedOcean) = ocean.sea_surface_temperature
ocean_surface_salinity(ocean::PrescribedOcean) = ocean.sea_surface_salinity
ocean_surface_velocities(ocean::PrescribedOcean) = ocean.velocities.u, ocean.velocities.v

#####
##### InterfaceComputations interface
#####

function ComponentExchanger(ocean::PrescribedOcean, exchange_grid)
    grid = ocean.grid
    T = CenterField(grid)
    S = CenterField(grid)
    u = CenterField(grid)
    v = CenterField(grid)

    # Initialize from the first time snapshot
    interior(T) .= interior(ocean.sea_surface_temperature)[:, :, :, 1]
    interior(S) .= interior(ocean.sea_surface_salinity)[:, :, :, 1]
    interior(u) .= interior(ocean.velocities.u)[:, :, :, 1]
    interior(v) .= interior(ocean.velocities.v)[:, :, :, 1]

    return ComponentExchanger((; u, v, T, S), nothing)
end

function net_fluxes(ocean::PrescribedOcean)
    grid = ocean.grid
    Jˢ = CenterField(grid)
    τx = CenterField(grid)
    τy = CenterField(grid)
    return (T=ocean.temperature_flux, S=Jˢ, u=τx, v=τy)
end

function interpolate_state!(exchanger, grid, ocean::PrescribedOcean, coupled_model)
    # Copy from FieldTimeSeries to exchanger snapshot fields.
    # For single-time data (constant), time index 1 is always correct.
    # TODO: proper temporal interpolation for multi-time prescribed data.
    n = 1
    interior(exchanger.state.T) .= interior(ocean.sea_surface_temperature)[:, :, :, n]
    interior(exchanger.state.S) .= interior(ocean.sea_surface_salinity)[:, :, :, n]
    interior(exchanger.state.u) .= interior(ocean.velocities.u)[:, :, :, n]
    interior(exchanger.state.v) .= interior(ocean.velocities.v)[:, :, :, n]
    return nothing
end

# Prescribed ocean does not evolve, so net flux assembly is not needed.
# The atmosphere still receives its fluxes from compute_atmosphere_ocean_fluxes!.
update_net_fluxes!(coupled_model, ocean::PrescribedOcean) = nothing

#####
##### Time stepping — update FieldTimeSeries backends, tick the clock
#####

function Oceananigans.TimeSteppers.time_step!(ocean::PrescribedOcean, Δt)
    tick!(ocean.clock, Δt)
    time = Time(ocean.clock.time)

    update_field_time_series!(ocean.sea_surface_temperature, time)
    update_field_time_series!(ocean.sea_surface_salinity, time)
    update_field_time_series!(ocean.velocities.u, time)
    update_field_time_series!(ocean.velocities.v, time)

    return nothing
end
