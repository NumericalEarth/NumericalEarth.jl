using Oceananigans.Architectures: architecture
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.OutputReaders: update_field_time_series!, FieldTimeSeries, cpu_interpolating_time_indices
using Oceananigans.TimeSteppers: Clock, tick!
using Oceananigans.Units: Time
using Oceananigans.Utils: launch!, prettysummary, prettytime
using KernelAbstractions: @kernel, @index
using NumericalEarth.EarthSystemModels: AbstractPrescribedComponent

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
mutable struct PrescribedOcean{FT, G, C, T, S, U, TI, R, HC} <: AbstractPrescribedComponent
    grid :: G
    clock :: C
    sea_surface_temperature :: T
    sea_surface_salinity :: S
    velocities :: U
    times :: TI
    density :: R
    heat_capacity :: HC

    function PrescribedOcean{FT}(grid::G, clock::C, sst::T, sss::S, u::U, times::TI, ρ::R, cp::HC) where {FT, G, C, T, S, U, TI, R, HC}
        return new{FT, G, C, T, S, U, TI, R, HC}(grid, clock, sst, sss, u, times, ρ, cp)
    end
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

    return PrescribedOcean{FT}(grid, 
                               clock,
                               sea_surface_temperature,
                               sea_surface_salinity,
                               velocities, 
                               times,
                               convert(FT, density),
                               convert(FT, heat_capacity))
end

Grids.grid(ocean::PrescribedOcean) = ocean.grid

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

EarthSystemModels.is_sea_ice_component(::PrescribedOcean) = false

function EarthSystemModels.adopt_clock(ocean::PrescribedOcean{FT}, clock) where FT
    new_clock = EarthSystemModels.matching_clock(ocean.clock, clock)
    isnothing(new_clock) && return ocean
    EarthSystemModels.warn_clock_coercion(ocean, new_clock)
    return PrescribedOcean{FT}(ocean.grid, 
                               new_clock,
                               ocean.sea_surface_temperature,
                               ocean.sea_surface_salinity,
                               ocean.velocities, 
                               ocean.times,
                               ocean.density, 
                               ocean.heat_capacity)
end

EarthSystemModels.reference_density(ocean::PrescribedOcean) = ocean.density
EarthSystemModels.heat_capacity(ocean::PrescribedOcean) = ocean.heat_capacity
EarthSystemModels.temperature_units(::PrescribedOcean) = DegreesKelvin()

EarthSystemModels.ocean_temperature(ocean::PrescribedOcean) = ocean.sea_surface_temperature
EarthSystemModels.ocean_salinity(ocean::PrescribedOcean) = ocean.sea_surface_salinity
EarthSystemModels.ocean_surface_temperature(ocean::PrescribedOcean) = ocean.sea_surface_temperature
EarthSystemModels.ocean_surface_salinity(ocean::PrescribedOcean) = ocean.sea_surface_salinity
EarthSystemModels.ocean_surface_velocities(ocean::PrescribedOcean) = ocean.velocities.u, ocean.velocities.v

#####
##### InterfaceComputations interface
#####

function EarthSystemModels.InterfaceComputations.ComponentExchanger(ocean::PrescribedOcean, exchange_grid)
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

EarthSystemModels.InterfaceComputations.net_fluxes(ocean::PrescribedOcean) = nothing

# Read each series at the ocean clock, linearly interpolating between the two snapshots that
# bracket it. `time_interpolated_getindex` returns the first snapshot whenever the bracketing
# indices coincide, so a single-time (constant) series needs no special case. The interpolation
# weights are computed once on the host — `cpu_interpolating_time_indices` — and reach the kernel
# as a `TimeInterpolator`, rather than being recomputed from `times` in every thread.
function EarthSystemModels.interpolate_state!(exchanger, grid, ocean::PrescribedOcean, coupled_model)
    Tᵒ = ocean.sea_surface_temperature
    Sᵒ = ocean.sea_surface_salinity
    uᵒ, vᵒ = ocean.velocities

    ocean_grid = ocean.grid
    arch = architecture(ocean_grid)
    t = ocean.clock.time
    interpolator(fts) = cpu_interpolating_time_indices(arch, fts.times, fts.time_indexing, t)

    launch!(arch, ocean_grid, :xy, _interpolate_prescribed_ocean_state!,
            exchanger.state.T, exchanger.state.S, exchanger.state.u, exchanger.state.v,
            Tᵒ, Sᵒ, uᵒ, vᵒ,
            interpolator(Tᵒ), interpolator(Sᵒ), interpolator(uᵒ), interpolator(vᵒ))

    # The flux kernel also evaluates halo columns; a 0 K halo temperature NaNs the solve.
    fill_halo_regions!(exchanger.state.T)
    fill_halo_regions!(exchanger.state.S)
    fill_halo_regions!(exchanger.state.u)
    fill_halo_regions!(exchanger.state.v)

    return nothing
end

@kernel function _interpolate_prescribed_ocean_state!(T, S, u, v, Tᵒ, Sᵒ, uᵒ, vᵒ, τT, τS, τu, τv)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T[i, j, 1] = Tᵒ[i, j, 1, τT]
        S[i, j, 1] = Sᵒ[i, j, 1, τS]
        u[i, j, 1] = uᵒ[i, j, 1, τu]
        v[i, j, 1] = vᵒ[i, j, 1, τv]
    end
end

# A prescribed ocean follows its data rather than responding to fluxes, so net flux assembly
# is not needed. The atmosphere still receives its fluxes from compute_atmosphere_ocean_fluxes!.
EarthSystemModels.update_net_fluxes!(coupled_model, ocean::PrescribedOcean) = nothing

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
