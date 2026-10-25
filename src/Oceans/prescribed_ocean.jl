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
                    temperature = default_prescribed_temperature(grid, times),
                    salinity = default_prescribed_salinity(grid, times),
                    velocities = default_prescribed_velocities(grid, times),
                    free_surface = nothing)

A prescribed ocean component with temperature, salinity, velocities and (optionally) free surface
prescribed as `FieldTimeSeries`.

The ocean state does not evolve in response to surface fluxes — it follows the prescribed data.
As an `EarthSystemModel` component, surface fluxes are still computed so the atmosphere feels the
ocean. The series may be surface fields (e.g. sea surface temperature, the default) or volumetric
fields, in which case the surface is their top level; a volumetric `PrescribedOcean` (see
`PrescribedOcean(region, dates, dataset)`) can also drive a regional ocean as the parent of
[`nested_ocean_model`](@ref).

Arguments
=========

- `grid`: An Oceananigans grid for the ocean domain.
- `times`: Time instances for the prescribed data. Default: `[zero(grid)]` (constant).

Keyword Arguments
=================

- `density`: Seawater density in kg/m³. Default: 1025.
- `heat_capacity`: Seawater specific heat capacity in J/(kg·K). Default: 4000.
- `clock`: Clock for tracking ocean time. Default: `Clock{FT}(time=0)`.
- `temperature`: `FieldTimeSeries` for the (sea surface or volumetric) temperature.
- `salinity`: `FieldTimeSeries` for the (sea surface or volumetric) salinity.
- `velocities`: `NamedTuple` of `FieldTimeSeries` for `(u, v)`.
- `free_surface`: `FieldTimeSeries` for the sea surface height, or `nothing`. Default: `nothing`.
"""
mutable struct PrescribedOcean{FT, G, C, T, S, U, E, TI, R, HC} <: AbstractPrescribedComponent
    grid :: G
    clock :: C
    temperature :: T
    salinity :: S
    velocities :: U
    free_surface :: E
    times :: TI
    density :: R
    heat_capacity :: HC

    function PrescribedOcean{FT}(grid::G, clock::C, T::TT, S::SS, u::U, η::E, times::TI, ρ::R, cp::HC) where {FT, G, C, TT, SS, U, E, TI, R, HC}
        return new{FT, G, C, TT, SS, U, E, TI, R, HC}(grid, clock, T, S, u, η, times, ρ, cp)
    end
end

function default_prescribed_temperature(grid, times)
    return FieldTimeSeries{Center, Center, Nothing}(grid, times)
end

function default_prescribed_salinity(grid, times)
    S = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    parent(S) .= 35
    return S
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
                         temperature = default_prescribed_temperature(grid, times),
                         salinity = default_prescribed_salinity(grid, times),
                         velocities = default_prescribed_velocities(grid, times),
                         free_surface = nothing)

    ocean = PrescribedOcean{FT}(grid,
                                clock,
                                temperature,
                                salinity,
                                velocities,
                                free_surface,
                                times,
                                convert(FT, density),
                                convert(FT, heat_capacity))

    fill_prescribed_halo_regions!(ocean)

    return ocean
end

Grids.grid(ocean::PrescribedOcean) = ocean.grid

function Oceananigans.set!(ocean::PrescribedOcean; T=nothing, S=nothing, u=nothing, v=nothing)
    !isnothing(T) && set!(ocean.temperature, T)
    !isnothing(S) && set!(ocean.salinity, S)
    !isnothing(u) && set!(ocean.velocities.u, u)
    !isnothing(v) && set!(ocean.velocities.v, v)
    fill_prescribed_halo_regions!(ocean)
    return nothing
end

# The exchanger reads the halo columns straight out of the time series (see
# `interpolate_prescribed_state!`), so their halos must be valid. Fill them whenever the data is
# set; a partly-in-memory series fills its own halos each time it loads a new window.
function fill_prescribed_halo_regions!(ocean::PrescribedOcean)
    fill_halo_regions!(ocean.temperature)
    fill_halo_regions!(ocean.salinity)
    fill_halo_regions!(ocean.velocities.u)
    fill_halo_regions!(ocean.velocities.v)
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
                               ocean.temperature,
                               ocean.salinity,
                               ocean.velocities,
                               ocean.free_surface,
                               ocean.times,
                               ocean.density,
                               ocean.heat_capacity)
end

EarthSystemModels.reference_density(ocean::PrescribedOcean) = ocean.density
EarthSystemModels.heat_capacity(ocean::PrescribedOcean) = ocean.heat_capacity
EarthSystemModels.temperature_units(::PrescribedOcean) = DegreesKelvin()

EarthSystemModels.ocean_temperature(ocean::PrescribedOcean) = ocean.temperature
EarthSystemModels.ocean_salinity(ocean::PrescribedOcean) = ocean.salinity
EarthSystemModels.ocean_surface_temperature(ocean::PrescribedOcean) = ocean.temperature
EarthSystemModels.ocean_surface_salinity(ocean::PrescribedOcean) = ocean.salinity
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

    state = (; u, v, T, S)
    interpolate_prescribed_state!(state, ocean)

    return ComponentExchanger(state, nothing)
end

EarthSystemModels.InterfaceComputations.net_fluxes(ocean::PrescribedOcean) = nothing
EarthSystemModels.InterfaceComputations.biogeochemical_interface(exchanger, ::PrescribedOcean; kwargs...) = NamedTuple()

EarthSystemModels.interpolate_state!(exchanger, grid, ocean::PrescribedOcean, coupled_model) =
    interpolate_prescribed_state!(exchanger.state, ocean)

# Read each series at the ocean clock, linearly interpolating between the two snapshots that bracket it.
# The interface flux kernels run over the halo columns too (`interface_kernel_parameters` spans 0:N+1), and
# the atmosphere's face-centered momentum fluxes average them into the first interior face, so the
# interpolation covers those columns rather than leaving them at zero (#754).
function interpolate_prescribed_state!(state, ocean::PrescribedOcean)
    Tᵒ = ocean.temperature
    Sᵒ = ocean.salinity
    uᵒ, vᵒ = ocean.velocities

    grid = ocean.grid
    arch = architecture(grid)
    t = ocean.clock.time
    interpolator(fts) = cpu_interpolating_time_indices(arch, fts.times, fts.time_indexing, t)

    # The surface is the top level: the only level of a surface series, the last of a volumetric one
    kᴺ = size(Tᵒ, 3)

    launch!(arch, grid, interface_kernel_parameters(grid), _interpolate_prescribed_ocean_state!,
            state.T, state.S, state.u, state.v,
            Tᵒ, Sᵒ, uᵒ, vᵒ, kᴺ,
            interpolator(Tᵒ), interpolator(Sᵒ), interpolator(uᵒ), interpolator(vᵒ))

    return nothing
end

@kernel function _interpolate_prescribed_ocean_state!(T, S, u, v, Tᵒ, Sᵒ, uᵒ, vᵒ, kᴺ, τT, τS, τu, τv)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T[i, j, 1] = Tᵒ[i, j, kᴺ, τT]
        S[i, j, 1] = Sᵒ[i, j, kᴺ, τS]
        u[i, j, 1] = uᵒ[i, j, kᴺ, τu]
        v[i, j, 1] = vᵒ[i, j, kᴺ, τv]
    end
end

# A prescribed ocean follows its data rather than responding to fluxes, so net flux assembly is not needed.
EarthSystemModels.update_net_fluxes!(coupled_model, ocean::PrescribedOcean) = nothing

#####
##### Time stepping — update FieldTimeSeries backends, tick the clock
#####

function Oceananigans.TimeSteppers.time_step!(ocean::PrescribedOcean, Δt)
    tick!(ocean.clock, Δt)
    time = Time(ocean.clock.time)

    update_field_time_series!(ocean.temperature, time)
    update_field_time_series!(ocean.salinity, time)
    update_field_time_series!(ocean.velocities.u, time)
    update_field_time_series!(ocean.velocities.v, time)
    update_prescribed_free_surface!(ocean.free_surface, time)

    return nothing
end

update_prescribed_free_surface!(η, time) = update_field_time_series!(η, time)
update_prescribed_free_surface!(::Nothing, time) = nothing
